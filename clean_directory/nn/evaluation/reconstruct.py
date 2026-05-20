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
from data.loaders import DATASET_LAYOUT, load_xyz
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
    feat_meta.pop("uvdar_baseline", None)  # we re-load it ourselves below

    # Filter NaN/inf (mirror training.py)
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, t_all = X_all[ok], Y_all[ok], t_all[ok]

    # ── Raw GT (full timeline, before feature-validity masking) ───────
    # Loaded directly from the dataset CSV so we can show the true GT
    # trajectory even at timestamps where features (and therefore NN
    # predictions) are unavailable.
    gt_csv = os.path.join(run_dir, DATASET_LAYOUT["flier_odom_in_camera_frame"])
    if os.path.exists(gt_csv):
        gt_df = load_xyz(gt_csv)
        gt_raw_t_ns = gt_df["t_ns"].to_numpy(dtype=np.int64)
        gt_raw_xyz  = gt_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        print(f"[load_run] Raw GT: {len(gt_raw_t_ns)} samples from {gt_csv}")
    else:
        gt_raw_t_ns = np.empty(0, dtype=np.int64)
        gt_raw_xyz  = np.empty((0, 3), dtype=np.float32)
        print(f"[load_run] Raw GT: {gt_csv} not found")

    # ── UVDAR baseline (always loaded, exact-match on t_all) ──────────
    # No interpolation — just look up the row whose timestamp matches.
    # Misses (no UVDAR sample at this blinker time) stay as NaN, so
    # plotting / RMSE code can ignore them naturally.
    uvdar_csv = os.path.join(run_dir, DATASET_LAYOUT["uvdar_estimate_in_camera_frame"])
    if os.path.exists(uvdar_csv):
        uvdar_df = (load_xyz(uvdar_csv)
                    .drop_duplicates(subset="t_ns", keep="first")
                    .set_index("t_ns"))
        uvdar_baseline = (uvdar_df.reindex(t_all)[["x", "y", "z"]]
                          .to_numpy(dtype=np.float32))
        n_match = int(np.isfinite(uvdar_baseline).all(axis=1).sum())
        print(f"[load_run] UVDAR baseline: {n_match}/{len(t_all)} exact matches")
    else:
        uvdar_baseline = np.full((len(t_all), 3), np.nan, dtype=np.float32)
        print(f"[load_run] UVDAR baseline: {uvdar_csv} not found, using NaN")

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
        if not np.isfinite(uvdar_baseline).all():
            raise ValueError(
                "Saved config has residual_learning=true but the UVDAR "
                "baseline has NaN rows — cannot add residual."
            )
        pred_res_all = pred_target + uvdar_baseline
    else:
        pred_res_all = pred_target

    val_split = cfg.get("val_split", 0.2)

    return {
        "cfg": cfg,
        "model": model,
        "Y_all": Y_all,
        "pred_res_all": pred_res_all,
        "pred_rel_xyz_all": uvdar_baseline,   # always present, NaN where missing
        "t_all": t_all,
        "val_split": val_split,
        "run_dir": run_dir,
        "feature_names": feat_meta["feature_names"],
        "gt_raw_t_ns": gt_raw_t_ns,
        "gt_raw_xyz":  gt_raw_xyz,
    }


# ── Label helper ──────────────────────────────────────────────────────

# New runs use ``_rmse{value}``; legacy runs used ``_val{value}``.
_VAL_SUFFIX_RE = re.compile(r"_(val|rmse)[\d.]+$")


def friendly_name(results_dir: str) -> str:
    """Derive a short label from a results folder name."""
    name = Path(results_dir).resolve().name
    return _VAL_SUFFIX_RE.sub("", name)
