"""
evaluation/reconstruct.py — Load a saved results directory, reconstruct
the model, and produce predictions for visualization / comparison.

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
#  Load a results directory                                           #
# ------------------------------------------------------------------ #

def load_run(
    results_dir: str,
    run_dir_override: str | None = None,
) -> dict:
    """
    Reconstruct a model from a saved results directory and produce
    predictions on the original dataset.

    Parameters
    ----------
    results_dir : str
        Path to a folder produced by :func:`utils.save_results`.
    run_dir_override : str, optional
        If given, use this dataset path instead of the one stored in
        ``config.yaml`` under ``_meta.dataset_path``.

    Returns
    -------
    dict with keys:
        cfg, model, Y_all, pred_res_all, t_all, val_split, run_dir,
        pred_rel_xyz_all (UVDAR baseline, if available), feature_names
    """
    results_dir = os.path.abspath(results_dir)

    # ── Config ────────────────────────────────────────────────────────
    cfg_path = os.path.join(results_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
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
        run_dir = os.path.abspath(os.path.expanduser(run_dir_override))
    else:
        meta = cfg.get("_meta", {})
        ds_path = meta.get("dataset_path")
        if not ds_path:
            raise FileNotFoundError(
                "Cannot determine dataset path from config. "
                "Pass --run-dir explicitly."
            )
        run_dir = ds_path
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Dataset directory not found: {run_dir}")

    # ── Build features (same pipeline as training) ────────────────────
    X_all, Y_all, t_all, feat_meta = build_features(cfg, str(run_dir))
    uvdar_baseline = feat_meta.pop("uvdar_baseline", None)

    # Filter NaN/inf (mirror training.py)
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, t_all = X_all[ok], Y_all[ok], t_all[ok]
    if uvdar_baseline is not None:
        uvdar_baseline = uvdar_baseline[ok]

    in_dim  = feat_meta["in_dim"]
    out_dim = Y_all.shape[1]

    # ── Model reconstruction ──────────────────────────────────────────
    weights_path = os.path.join(results_dir, "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

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
                "Saved config has residual_learning=true but no UVDAR "
                "baseline is available in the dataset."
            )
        pred_res_all = pred_target + uvdar_baseline
    else:
        pred_res_all = pred_target

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
    if uvdar_baseline is not None:
        result["pred_rel_xyz_all"] = uvdar_baseline
    return result


# ── Label helper ──────────────────────────────────────────────────────

# New runs use ``_rmse{value}``; legacy runs used ``_val{value}``.
_VAL_SUFFIX_RE = re.compile(r"_(val|rmse)[\d.]+$")


def friendly_name(results_dir: str) -> str:
    """Derive a short label from a results folder name."""
    name = Path(results_dir).resolve().name
    return _VAL_SUFFIX_RE.sub("", name)
