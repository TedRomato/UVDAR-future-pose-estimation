"""
utils.py — Config loading, results saving, and misc helpers.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# ------------------------------------------------------------------ #
#  Config loading                                                     #
# ------------------------------------------------------------------ #

def load_config(path: str | None = None) -> dict:
    """
    Load a YAML config and validate required keys.

    Parameters
    ----------
    path : str or None
        Path to YAML file.  If *None*, loads ``configs/default.yaml``
        relative to this package directory.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    required = [
        "learning_rate", "epochs", "layers", "batch_size",
        "val_split", "activation", "optimizer", "weight_decay",
    ]
    for k in required:
        if k not in raw:
            raise KeyError(f"Missing required config key: '{k}'")

    # Ensure correct types for hyperparameters
    raw["learning_rate"] = float(raw["learning_rate"])
    raw["epochs"]        = int(raw["epochs"])
    raw["layers"]        = list(raw["layers"])
    raw["batch_size"]    = int(raw["batch_size"])
    raw["val_split"]     = float(raw["val_split"])
    raw["activation"]    = str(raw["activation"])
    raw["optimizer"]     = str(raw["optimizer"])
    raw["weight_decay"]  = float(raw["weight_decay"])

    return raw


# ------------------------------------------------------------------ #
#  Sweep config loading                                               #
# ------------------------------------------------------------------ #

def load_sweep_configs(
    path: str | None = None,
    group: str | None = None,
) -> list[dict]:
    """
    Load sweep configurations from a sweep YAML file.

    The YAML must have two top-level keys:

    - ``bases``: a dict of named base-preset configs.
    - ``sweeps``: a dict of ``{group_name: [entry, ...]}``.

    Each entry is a dict with at least:

    - ``name`` — human-readable run identifier.
    - ``base`` — which preset from ``bases`` to inherit from.
    - any additional override keys (deep-merged on top of the base).

    Parameters
    ----------
    path : str
        Path to the sweep YAML (e.g. ``configs/sweeps/uvdar_tuning.yaml``).
    group : str or None
        If given, only return configs whose group name matches (supports
        comma-separated list, e.g. ``"fusion-age-ablation,uvdar-core-residual"``).
    """
    if path is None:
        raise ValueError(
            "A sweep config path is required. "
            "Available configs are in configs/sweeps/."
        )

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    bases = raw.get("bases", {})
    sweeps = raw.get("sweeps", {})

    if not bases:
        raise ValueError(f"No 'bases' defined in {path}")
    if not sweeps:
        raise ValueError(f"No 'sweeps' defined in {path}")

    # Optional group filter
    allowed_groups: set[str] | None = None
    if group:
        allowed_groups = {g.strip() for g in group.split(",")}
        unknown = allowed_groups - set(sweeps.keys())
        if unknown:
            raise KeyError(
                f"Unknown group(s): {unknown}. "
                f"Available: {list(sweeps.keys())}"
            )

    configs: list[dict] = []

    for group_name, entries in sweeps.items():
        if allowed_groups and group_name not in allowed_groups:
            continue

        if not isinstance(entries, list):
            raise TypeError(
                f"Sweep group '{group_name}' must be a list, "
                f"got {type(entries).__name__}"
            )

        for entry in entries:
            entry_name = entry.get("name")
            base_name = entry.get("base")

            if not entry_name:
                raise ValueError(
                    f"Entry in group '{group_name}' is missing a 'name' key: {entry}"
                )
            if not base_name:
                raise ValueError(
                    f"Entry '{entry_name}' in group '{group_name}' "
                    f"is missing a 'base' key"
                )
            if base_name not in bases:
                raise KeyError(
                    f"Entry '{entry_name}' references unknown base '{base_name}'. "
                    f"Available bases: {list(bases.keys())}"
                )

            # Build overrides: everything except the meta keys
            override = {k: v for k, v in entry.items() if k not in ("name", "base")}

            cfg = _deep_merge(bases[base_name], override)
            cfg["_sweep_name"] = entry_name
            cfg["_sweep_group"] = group_name
            configs.append(cfg)

    return configs


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ------------------------------------------------------------------ #
#  Results directory                                                  #
# ------------------------------------------------------------------ #

def ensure_fresh_results_dir(
    name: str, subdir: str | None = None,
) -> str:
    """Create ``./results/[subdir/]<name>``; raise if it already exists."""
    parts = [".", "results"]
    if subdir:
        parts.append(subdir)
    parts.append(name)
    out = os.path.join(*parts)

    if os.path.exists(out):
        raise FileExistsError(f"Results folder already exists: {out}")
    os.makedirs(out, exist_ok=False)
    return out


# ------------------------------------------------------------------ #
#  Save results                                                       #
# ------------------------------------------------------------------ #

def save_results(
    name: str,
    cfg: dict,
    artifacts: dict,
    run_dir: str,
    *,
    results_subdir: str | None = None,
    meta: dict | None = None,
) -> str:
    """
    Save model weights, normalization stats, config, and learning curves.

    Parameters
    ----------
    name : str
        Base name (RMSE will be appended).
    cfg : dict
        Full config used for training.
    artifacts : dict
        Output of ``train_pipeline()``.
    run_dir : str
        Path to the data directory used.
    meta : dict, optional
        Extra metadata (``in_dim``, ``feature_names``, …).

    Returns
    -------
    str — path to the created results directory.
    """
    # Compute metrics early so we can use RMSE in the directory name
    metrics = _compute_metrics(artifacts)

    rmse_str = f"{metrics['nn_rmse_val']:.6f}"
    results_dir = ensure_fresh_results_dir(
        f"{name}_rmse{rmse_str}", subdir=results_subdir,
    )
    print(f"Creating results directory: {results_dir}")

    # ── Config (with traceability metadata) ───────────────────────────
    cfg_out = dict(cfg)
    cfg_out["_meta"] = {
        "dataset_path": os.path.abspath(run_dir),
    }
    if meta:
        cfg_out["_meta"].update(meta)
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)

    # ── Model weights ─────────────────────────────────────────────────
    torch.save(
        artifacts["model"].state_dict(),
        os.path.join(results_dir, "model.pt"),
    )

    # ── Normalization stats ───────────────────────────────────────────
    norm_json = {k: v.tolist() for k, v in artifacts["norm_stats"].items()}
    with open(os.path.join(results_dir, "normalization.json"), "w") as f:
        json.dump(norm_json, f, indent=2)

    # ── Learning curves ───────────────────────────────────────────────
    train_losses = artifacts["train_losses"]
    val_losses   = artifacts["val_losses"]
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train loss (MSE)")
    ax.plot(epochs, val_losses,   label="val loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised MSE)")
    ax.set_title("Learning curves")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "learning_curves.png"), dpi=300)
    plt.close(fig)

    # ── Metrics (RMSE on validation data) ─────────────────────────────
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _print_metrics(metrics)

    print(f"Results saved to: {results_dir}")
    return results_dir


# ------------------------------------------------------------------ #
#  RMSE metrics                                                       #
# ------------------------------------------------------------------ #

def _rmse(errors: np.ndarray) -> float:
    """Root-mean-square error (Euclidean norm for 3-D, scalar for 1-D)."""
    return float(np.sqrt(np.mean(errors ** 2)))


def _improvement_pct(old_rmse: float, new_rmse: float) -> float:
    """Percentage improvement: positive = NN is better."""
    if abs(old_rmse) <= 1e-12:
        return float("nan")
    return (1.0 - new_rmse / old_rmse) * 100.0


def _compute_metrics(artifacts: dict) -> dict:
    """
    Compute RMSE metrics on validation data.

    Returns a JSON-serialisable dict with:
    - ``nn_rmse_val``       — NN RMSE on val set (3-D Euclidean)
    - ``nn_rmse_val_x/y/z`` — per-axis NN RMSE
    - If UVDAR baseline is available:
        - ``uvdar_rmse_val``        — raw UVDAR RMSE on val set
        - ``uvdar_rmse_val_x/y/z``  — per-axis raw UVDAR RMSE
        - ``improvement_pct``       — overall % improvement
        - ``improvement_pct_x/y/z`` — per-axis % improvement
    """
    Y_all    = artifacts["Y_all"]
    pred_all = artifacts["pred_all"]
    val_mask = artifacts["val_mask"]

    Y_val    = Y_all[val_mask]
    pred_val = pred_all[val_mask]

    # NN metrics
    nn_err_3d = np.linalg.norm(pred_val - Y_val, axis=1)
    nn_rmse   = _rmse(nn_err_3d)

    axis_names = ["x", "y", "z"]
    nn_rmse_axes = {}
    for i, ax in enumerate(axis_names):
        nn_rmse_axes[ax] = _rmse(pred_val[:, i] - Y_val[:, i])

    metrics: dict = {
        "nn_rmse_val":   round(nn_rmse, 6),
        "nn_rmse_val_x": round(nn_rmse_axes["x"], 6),
        "nn_rmse_val_y": round(nn_rmse_axes["y"], 6),
        "nn_rmse_val_z": round(nn_rmse_axes["z"], 6),
    }

    # UVDAR baseline metrics (if available)
    uvdar_baseline = artifacts.get("uvdar_baseline")
    if uvdar_baseline is not None:
        uvdar_val = uvdar_baseline[val_mask]
        uvdar_err_3d = np.linalg.norm(uvdar_val - Y_val, axis=1)
        uvdar_rmse   = _rmse(uvdar_err_3d)

        uvdar_rmse_axes = {}
        for i, ax in enumerate(axis_names):
            uvdar_rmse_axes[ax] = _rmse(uvdar_val[:, i] - Y_val[:, i])

        metrics["uvdar_rmse_val"]   = round(uvdar_rmse, 6)
        metrics["uvdar_rmse_val_x"] = round(uvdar_rmse_axes["x"], 6)
        metrics["uvdar_rmse_val_y"] = round(uvdar_rmse_axes["y"], 6)
        metrics["uvdar_rmse_val_z"] = round(uvdar_rmse_axes["z"], 6)

        metrics["improvement_pct"]   = round(_improvement_pct(uvdar_rmse, nn_rmse), 2)
        metrics["improvement_pct_x"] = round(_improvement_pct(uvdar_rmse_axes["x"], nn_rmse_axes["x"]), 2)
        metrics["improvement_pct_y"] = round(_improvement_pct(uvdar_rmse_axes["y"], nn_rmse_axes["y"]), 2)
        metrics["improvement_pct_z"] = round(_improvement_pct(uvdar_rmse_axes["z"], nn_rmse_axes["z"]), 2)

    return metrics


def _print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict."""
    print("── Validation RMSE ──")
    print(f"  NN   (3D): {metrics['nn_rmse_val']:.4f} m  "
          f"(x={metrics['nn_rmse_val_x']:.4f}, y={metrics['nn_rmse_val_y']:.4f}, z={metrics['nn_rmse_val_z']:.4f})")
    if "uvdar_rmse_val" in metrics:
        print(f"  UVDAR(3D): {metrics['uvdar_rmse_val']:.4f} m  "
              f"(x={metrics['uvdar_rmse_val_x']:.4f}, y={metrics['uvdar_rmse_val_y']:.4f}, z={metrics['uvdar_rmse_val_z']:.4f})")
        print(f"  Improvement: {metrics['improvement_pct']:.2f}%  "
              f"(x={metrics['improvement_pct_x']:.2f}%, y={metrics['improvement_pct_y']:.2f}%, z={metrics['improvement_pct_z']:.2f}%)")
