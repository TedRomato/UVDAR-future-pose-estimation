# Architecture — current_pose_estimation

## Directory layout

```
current_pose_estimation/
├── configs/
│   ├── default.yaml          # Single-run defaults (features + hyperparams)
│   └── sweep.yaml            # Per-variant sweep definitions
│
├── data/                     # Data loading & feature engineering
│   ├── __init__.py
│   ├── loaders.py            # load_xyz(), load_blinkers()
│   ├── alignment.py          # interpolate_on_index(), forward_fill_with_age()
│   └── features.py           # build_features(cfg, run_dir) → X, Y, t, meta
│
├── models/                   # Neural-network definitions
│   ├── __init__.py
│   └── mlp.py                # build_model(), build_optimizer()
│
├── evaluation/               # Post-training analysis
│   ├── __init__.py           # Style constants + plotting helpers
│   ├── reconstruct.py        # load_run() — rebuild model & predict
│   ├── visualize.py          # Single-run visualizer
│   └── compare.py            # Multi-run overlay comparison
│
├── training.py               # train_pipeline() — split → norm → train → predict
├── utils.py                  # load_config(), load_sweep_configs(), save_results()
├── train.py                  # Single-run CLI
├── sweep.py                  # Parallel sweep CLI (Aim tracking)
└── ARCHITECTURE.md           # This file
```

## Config schema

```yaml
# ── Features ────────────────────────────────
features:
  blinkers:
    enabled: false
    max_leds: 4
    min_leds: 2
  uvdar:
    enabled: true
    components:          # extensible list
      - position         # x, y, z  (3 features)
      # - variance       # var_x, var_y, var_z  (3 features)
      # - orientation    # qx, qy, qz, qw  (4 features)
  derived:               # features computed from multiple modalities
    []                   # e.g. [age]  (only valid when blinkers + uvdar)

# ── Hyperparameters ─────────────────────────
learning_rate: 0.001
epochs: 100
layers: [64, 64]
batch_size: 0            # 0 = full-batch
val_split: 0.2
val_padding: 0
split_mode: sequential    # sequential | random
activation: relu
optimizer: adamw
weight_decay: 0.0
seed: 42
residual_learning: false  # true → learn UVDAR error (requires uvdar.position)
```

## Data flow

```
CSV files  →  loaders.py  →  alignment.py  →  features.py  →  build_features()
                                                                    │
                                                          ┌─────────┴──────────┐
                                                          │  X, Y, t_sec, meta │
                                                          └─────────┬──────────┘
                                                                    │
                                                              training.py
                                                            train_pipeline()
                                                                    │
                                                          ┌─────────┴──────────┐
                                                          │     artifacts      │
                                                          └─────────┬──────────┘
                                                                    │
                                                               utils.py
                                                            save_results()
                                                                    │
                                                          ┌─────────┴──────────┐
                                                          │   results/<run>/   │
                                                          │  config.yaml       │
                                                          │  model.pt          │
                                                          │  normalization.json│                                                          │  metrics.json      │                                                          │  learning_curves.png│
                                                          └────────────────────┘
```

## Result files

Each training run produces a directory under `results/` with the following
structure:

```
results/<name>_val<loss>/
├── config.yaml            # Full config used (features + hyperparams + _meta)
├── model.pt               # Model state dict (torch.save)
├── normalization.json     # Z-score stats: X_mean, X_std, Y_mean, Y_std
├── metrics.json           # RMSE metrics on validation data (NN vs UVDAR)
└── learning_curves.png    # Train/val MSE loss per epoch
```

### config.yaml

Contains the complete config that produced this run, plus a `_meta` block:

```yaml
features: { ... }           # Exact features block used
layers: [64, 64]
activation: relu
# ... all hyperparameters ...
_meta:
  dataset_path: /absolute/path/to/data/LARGE_DATASET
  in_dim: 17
  feature_names: [u1, v1, m1, ..., pred_x, pred_y, pred_z, age]
```

### normalization.json

```json
{
  "X_mean": [[...]],
  "X_std":  [[...]],
  "Y_mean": [[...]],
  "Y_std":  [[...]]
}
```

Each value is a 2-D list (shape `(1, D)`) matching the corresponding
feature/target dimensionality.  Used to z-score normalise inputs and
denormalise predictions at inference time.

### model.pt

PyTorch `state_dict` saved with `torch.save()`.  To reconstruct the model,
use `build_model(cfg, in_dim, out_dim)` from `models/mlp.py` and load the
weights — or simply call `evaluation.reconstruct.load_run()` which handles
everything automatically.

### metrics.json

Validation-set RMSE metrics computed after training, comparing the NN output
against both ground truth and the raw UVDAR system output:

```json
{
  "nn_rmse_val":        0.1234,
  "nn_rmse_val_x":      0.0456,
  "nn_rmse_val_y":      0.0789,
  "nn_rmse_val_z":      0.0901,
  "uvdar_rmse_val":     0.2345,
  "uvdar_rmse_val_x":   0.0987,
  "uvdar_rmse_val_y":   0.1234,
  "uvdar_rmse_val_z":   0.1456,
  "improvement_pct":    47.35,
  "improvement_pct_x":  53.80,
  "improvement_pct_y":  36.06,
  "improvement_pct_z":  38.12
}
```

- `nn_rmse_val` — 3-D Euclidean RMSE of the NN predictions on validation data
- `uvdar_rmse_val` — 3-D Euclidean RMSE of the raw UVDAR predictions on
  the same validation data (present only when `predicted_relative_pose.csv`
  exists in the dataset)
- `improvement_pct` — percentage reduction in RMSE: positive means the NN
  is better than raw UVDAR.  Computed as `(1 - nn_rmse / uvdar_rmse) * 100`
- Per-axis variants (`_x`, `_y`, `_z`) are also included

## Key design decisions

1. **Config-driven features**: The `features` block in the config fully
   determines input dimensionality.  No `--model-type` flag; the config
   *is* the model type.

2. **Single `build_features()` entry point**: Training, sweep, and
   evaluation all call the same function — no data-loading duplication.

3. **Forward-fill for fusion**: When blinkers + UVDAR are both enabled,
   blinker timestamps define the reference timeline. UVDAR values are
   carried forward (last-observation-carried-forward) with an `age`
   derived feature measuring staleness.

4. **Extensible components**: Adding a new UVDAR component (e.g. variance)
   requires:
   - Register it in `data/features.py::_UVDAR_COMPONENTS`
   - Add it to the config: `features.uvdar.components: [position, variance]`
   - Nothing else changes.

5. **Aim tracking**: Always on during sweeps. Each sweep worker logs
   per-epoch losses and final metrics.

6. **Residual learning** (optional): When `residual_learning: true`,
   the network learns the *UVDAR error* `(Y - UVDAR_baseline)` instead
   of the full mapping `X → Y`.  At inference the predicted residual is
   added back to the UVDAR baseline.  This requires `features.uvdar`
   with a `position` component.  Default is `false` (direct mapping).

7. **Split mode**: `split_mode: sequential` (default) keeps chronological
   order — early data trains, late data validates, with an optional
   `val_padding` gap between them.  `split_mode: random` shuffles rows
   (seeded by `seed`) before the split; `val_padding` is ignored.

8. **Named sweep groups**: In `sweep.yaml`, `sweeps` is a dict of
   named groups (not a flat list).  Group names encode the experiment
   configuration (e.g. `fusion-age-seq`, `blinkers-4led-seq`).  Each
   config receives a `_sweep_name` key of the form `<group>-<index>`
   that is used for result folder and Aim experiment naming.  Legacy
   flat-list format is still supported for backwards compatibility.

## CLI examples

```bash
# Single training run
python train.py ../../data/LARGE_DATASET baseline --config configs/default.yaml

# Hyperparameter sweep
python sweep.py --variant fusion --run-dir ../../data/LARGE_DATASET --processes 8

# Visualize a result
python -m evaluation.visualize results/baseline_val0.012345

# Compare multiple results
python -m evaluation.compare results/run1_val* results/run2_val*
```
