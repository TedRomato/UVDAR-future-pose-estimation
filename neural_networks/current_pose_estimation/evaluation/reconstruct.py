"""
evaluation/reconstruct.py — Load a saved results directory, reconstruct the
model, and produce predictions for visualization / comparison.

Public API:
    load_run(results_dir, run_dir_override=None) → dict
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from data.features import build_features
from models.mlp import build_model


# ------------------------------------------------------------------ #
#  Run-directory resolution                                           #
# ------------------------------------------------------------------ #

def _is_valid_run_dir(path: Path, cfg: dict) -> bool:
    """Check that *path* contains the CSVs needed by the config."""
    if not path.is_dir():
        return False
    # Always need true_relative_pose.csv
    if not (path / "true_relative_pose.csv").exists():
        return False
    feat = cfg.get("features", {})
    if feat.get("uvdar", {}).get("enabled", False):
        if not (path / "predicted_relative_pose.csv").exists():
            return False
    if feat.get("blinkers", {}).get("enabled", False):
        if not (path / "blinkers_seen_right.csv").exists():
            return False
    return True


def _find_data_root(start_dir: Path) -> Path | None:
    for parent in (start_dir, *start_dir.parents):
        cand = parent / "data"
        if cand.is_dir():
            return cand
    return None


def resolve_run_dir(user_value: str, cfg: dict) -> Path:
    """
    Resolve a run-directory string to an existing directory that contains
    the CSV files required by *cfg*.

    Tries the literal path first, then various relative-to-data-root heuristics.
    """
    raw = Path(os.path.expanduser(user_value.strip()))
    as_is = raw if raw.is_absolute() else (Path.cwd() / raw)
    as_is = as_is.resolve()
    if _is_valid_run_dir(as_is, cfg):
        return as_is

    data_root = _find_data_root(Path.cwd())
    if data_root is None:
        data_root = _find_data_root(Path(__file__).resolve().parent)

    candidates: list[Path] = []
    if data_root is not None:
        repo_root = data_root.parent
        parts = raw.parts
        if parts[:1] == ("data",):
            candidates.append((repo_root / raw).resolve())
        candidates.append((data_root / raw).resolve())
        candidates.append((data_root / raw / "csv_data").resolve())
        if len(parts) >= 1:
            flight = parts[0]
            rest = Path(*parts[1:]) if len(parts) > 1 else Path()
            candidates.append((data_root / flight / "csv_data" / rest).resolve())
            if len(parts) == 1:
                candidates.append((data_root / flight / "csv_data").resolve())

    for cand in candidates:
        if _is_valid_run_dir(cand, cfg):
            return cand

    raise FileNotFoundError(
        f"Run directory not found or missing required CSVs: {user_value}\n"
        f"Tried: {as_is}\n"
        + "\n".join(f"Tried: {c}" for c in candidates)
    )


# ------------------------------------------------------------------ #
#  Load a results directory                                           #
# ------------------------------------------------------------------ #

def load_run(
    results_dir: str,
    run_dir_override: str | None = None,
) -> dict:
    """
    Reconstruct a model from a saved results directory and produce
    predictions on the original dataset.

    Returns
    -------
    dict with keys:
        cfg, model, Y_all, pred_res_all, t_all, val_split, run_dir,
        pred_rel_xyz_all (old UVDAR baseline, if available), feature_names
    """
    results_dir = os.path.abspath(results_dir)

    # ── Config ────────────────────────────────────────────────────────
    cfg_path = os.path.join(results_dir, "config.yaml")
    # Backwards compatibility: old name was config_used.yaml
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(results_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found in {results_dir}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Normalization ─────────────────────────────────────────────────
    norm_path = os.path.join(results_dir, "normalization.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")
    with open(norm_path, "r") as f:
        norm_raw = json.load(f)
    norm_stats = {k: np.array(v) for k, v in norm_raw.items()}
    X_mean, X_std = norm_stats["X_mean"], norm_stats["X_std"]
    Y_mean, Y_std = norm_stats["Y_mean"], norm_stats["Y_std"]

    # ── Dataset path ──────────────────────────────────────────────────
    if run_dir_override:
        run_dir = resolve_run_dir(run_dir_override, cfg)
    else:
        # Try _meta.dataset_path from config, fall back to dataset_path.txt
        meta = cfg.get("_meta", {})
        ds_path = meta.get("dataset_path")
        if not ds_path:
            ds_path_file = os.path.join(results_dir, "dataset_path.txt")
            if os.path.exists(ds_path_file):
                with open(ds_path_file) as f:
                    ds_path = f.readline().strip()
        if not ds_path:
            raise FileNotFoundError(
                "Cannot determine dataset path. Pass --run-dir explicitly."
            )
        run_dir = resolve_run_dir(ds_path, cfg)

    # ── Build features (same pipeline as training) ────────────────────
    X_all, Y_all, t_all, feat_meta = build_features(cfg, str(run_dir))

    # Extract UVDAR baseline before filtering
    uvdar_baseline = feat_meta.pop("uvdar_baseline", None)

    # Filter NaN/inf
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, t_all = X_all[ok], Y_all[ok], t_all[ok]
    if uvdar_baseline is not None:
        uvdar_baseline = uvdar_baseline[ok]

    in_dim  = feat_meta["in_dim"]
    out_dim = Y_all.shape[1]

    # ── Model reconstruction ──────────────────────────────────────────
    # Try new weight file name, fall back to old
    weights_path = os.path.join(results_dir, "model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(results_dir, "model_state_dict.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found in {results_dir}")

    model = build_model(cfg, in_dim=in_dim, out_dim=out_dim)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Predict (denormalised) ────────────────────────────────────────
    X_all_n = (X_all - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_target = pred_norm * Y_std + Y_mean

    # Residual learning: add UVDAR baseline back for final predictions
    if cfg.get("residual_learning", False):
        if uvdar_baseline is None:
            raise ValueError(
                "Saved config has residual_learning=true but no UVDAR baseline "
                "is available in the dataset."
            )
        pred_res_all = pred_target + uvdar_baseline
    else:
        pred_res_all = pred_target

    # ── UVDAR baseline for comparison (if available) ──────────────────
    pred_rel_xyz_all = uvdar_baseline

    val_split = cfg.get("val_split", 0.2)

    result = {
        "cfg": cfg,
        "model": model,
        "Y_all": Y_all,
        "pred_res_all": pred_res_all,
        "t_all": t_all,
        "val_split": val_split,
        "run_dir": run_dir,
        "feature_names": feat_meta["feature_names"],
    }
    if pred_rel_xyz_all is not None:
        result["pred_rel_xyz_all"] = pred_rel_xyz_all
    return result


# ── Label helper ──────────────────────────────────────────────────────

_VAL_SUFFIX_RE = re.compile(r"_val[\d.]+$")


def friendly_name(results_dir: str) -> str:
    """Derive a short label from a results folder name, stripping _val{number}."""
    name = Path(results_dir).resolve().name
    return _VAL_SUFFIX_RE.sub("", name)
