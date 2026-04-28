# Architecture — clean_directory/nn

A small, config-driven MLP that estimates the relative pose of a target
UAV from UVDAR-camera blinker detections and/or the existing UVDAR pose
estimate.  Ported from `neural_networks/current_pose_estimation/` and
simplified to match the clean `clean_directory/data/<run>/` CSV layout.

## Directory layout

```
clean_directory/nn/
├── configs/
│   ├── default.yaml              # Single-run defaults
│   ├── sweep.yaml                # Combined sweep across all modalities
│   └── sweeps/
│       ├── blinkers_tuning.yaml
│       ├── fusion_tuning.yaml
│       └── uvdar_tuning.yaml
│
├── data/
│   ├── __init__.py
│   ├── loaders.py                # load_xyz(), load_blinkers()
│   └── features.py               # build_features(cfg, run_dir) → X, Y, t_ns, meta
│
├── models/
│   ├── __init__.py
│   └── mlp.py                    # build_model(), build_optimizer()
│
├── evaluation/
│   ├── __init__.py               # Style constants + plotting helpers
│   ├── reconstruct.py            # load_run() — rebuild model & predict
│   ├── visualize.py              # Single-run visualizer
│   └── compare.py                # Multi-run overlay comparison
│
├── training.py                   # train_pipeline() — split → norm → train → predict
├── utils.py                      # load_config(), load_sweep_configs(), save_results()
├── train.py                      # Single-run CLI
├── sweep.py                      # Parallel sweep CLI
└── ARCHITECTURE.md               # This file
```

## Expected input data

The training scripts read CSVs from a `run_dir` produced by
`clean_directory/parse.py`:

```
clean_directory/data/<run>/
├── true_relative_pose.csv          # t, x, y, z [, qx, qy, qz, qw]
├── predicted_relative_pose.csv     # t, x, y, z [, qx, qy, qz, qw]   (UVDAR baseline)
└── blinkers_right.csv              # t, points, image_height, image_width
```

All CSVs share a common nanosecond timestamp `t`.  Odometries are already
interpolated to blinker timestamps and the UVDAR rows only exist where
blinkers were available, so feature construction is a pure exact-match
join — no interpolation, no forward fill, no aging.

## Config schema

```yaml
# ── Features ────────────────────────────────
features:
  blinkers:
    enabled: true
    max_leds: 4
    min_leds: 2
  uvdar:
    enabled: true
    components:          # extensible list
      - position         # x, y, z  (3 features)
  derived: []            # placeholder; no derived features in this build

# ── Hyperparameters ─────────────────────────
learning_rate: 0.001
epochs: 100
layers: [32, 32]
batch_size: 0            # 0 = full-batch
val_split: 0.2
val_padding: 0
split_mode: sequential   # sequential | random
activation: relu
optimizer: adam          # adam | adamw | sgd
weight_decay: 0.0001
seed: 42
residual_learning: false # true → learn UVDAR error (requires features.uvdar)
```

## Data flow

```
CSV files  →  loaders.py  →  features.py (exact-join)  →  build_features()
                                                                │
                                                      ┌─────────┴──────────┐
                                                      │  X, Y, t_ns, meta  │
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
                                                      ┌─────────┴────────────┐
                                                      │   results/<run>/     │
                                                      │   config.yaml        │
                                                      │   model.pt           │
                                                      │   normalization.json │
                                                      │   metrics.json       │
                                                      │   learning_curves.png│
                                                      └──────────────────────┘
```

## Feature construction (`build_features`)

1. Load `true_relative_pose.csv` (target).
2. If `blinkers.enabled`: load `blinkers_right.csv`, parse the
   `points` cell, sort detections lexicographically by `(u, v)`,
   normalise pixel coordinates to `[-1, 1]`, pad/truncate to
   `max_leds`, and append a binary mask + `n_visible` count.  The
   blinker timestamps become the **reference timeline**.
3. If `uvdar.enabled`: exact-join `predicted_relative_pose.csv` on the
   reference timeline.  Rows where any UVDAR component is missing are
   **dropped** from training.
4. Always exact-join the target on the surviving timeline; drop any
   row without a target.
5. Always try to load the UVDAR `position` baseline for metrics
   reporting (NaN-safe: rows missing the baseline are skipped when
   computing RMSE / improvement %).

## Residual learning

When `residual_learning: true` and `features.uvdar` is enabled, the
network is trained to predict `Y - uvdar_baseline` instead of `Y`.
At inference time the baseline is added back.  Because UVDAR-missing
rows are dropped up-front, the baseline is always finite for every
training row.

## Train / val split

* `sequential` (default): the first `1 − val_split` rows are train,
  the remaining `val_split` are val.  `val_padding` inserts a gap.
* `random`: rows are shuffled with `cfg.seed` before splitting.
  `val_padding` is ignored.

## CLI

Single run:

```bash
cd clean_directory/nn
python train.py ../data/<run> baseline --config configs/default.yaml
```

Sweep:

```bash
cd clean_directory/nn
python sweep.py --sweep-config configs/sweep.yaml --run-dir ../data/<run> --processes 8
python sweep.py --sweep-config configs/sweeps/fusion_tuning.yaml --group fusion-core-residual
```

Visualise / compare a saved run:

```bash
cd clean_directory/nn
python -m evaluation.visualize results/baseline_rmse0.123456
python -m evaluation.compare  results/run_a results/run_b
```

Override the dataset path stored in `config.yaml`:

```bash
python -m evaluation.visualize results/baseline_rmse0.123 --run-dir ../data/other_run
```

## Result directory contents

```
results/<name>_rmse<value>/
├── config.yaml          # full config + _meta.dataset_path + feature_names
├── model.pt             # PyTorch state_dict
├── normalization.json   # X_mean, X_std, Y_mean, Y_std
├── metrics.json         # nn_rmse_val, uvdar_rmse_val, improvement_pct, per-axis
└── learning_curves.png  # train/val loss vs epoch
```
