#!/usr/bin/env python3
"""
Shared constants, helpers, and loaders used by visualize_results.py
and compare_results.py.
"""

import os
import re
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import helpers  # load_xyz, align_two_streams
from current_pose_estimation_blinkers import (
    load_blinkers, preprocess_blinkers, _align_blinkers_to_target,
    INPUT_DIM as BLINKERS_INPUT_DIM,
)
from current_pose_estimation_blinkers_and_3d import (
    _align_all as _align_blinkers_and_3d,
    INPUT_DIM as BLINKERS_AND_3D_INPUT_DIM,
)


# ── visualization style ──────────────────────────────────────────────

# Ground truth & old-system colours (shared by all tools)
GT_COLOR = "#000000"       # black
OLD_COLOR = "#D55E00"      # vermilion

GT_LINESTYLE = "-"
OLD_LINESTYLE = "--"

GT_LINEWIDTH = 2.0
SYS_LINEWIDTH = 1.6

# Background shading for train / validation regions
TRAIN_BG_COLOR = (0.12, 0.47, 0.71, 0.10)
VAL_BG_COLOR   = (1.00, 0.50, 0.05, 0.10)

# RMSE annotation defaults
RMSE_TEXT_LOC = (0.01, 0.98)
RMSE_TEXT_KW = dict(
    transform=None,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
)

# Distinguishable palette for up to 5 NN prediction lines
NN_COLORS = [
    "#0072B2",   # blue
    "#009E73",   # teal / bluish green
    "#CC79A7",   # reddish purple
    "#56B4E9",   # sky blue
    "#E69F00",   # amber
]

AXIS_LABELS = ["Relative X", "Relative Y", "Relative Z"]


# ── model reconstruction ─────────────────────────────────────────────

def build_model_from_config(cfg, in_dim=3, out_dim=3):
    hidden_sizes = cfg["layers"]
    act_name = cfg["activation"].lower()

    layers = []
    cur_dim = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(cur_dim, h))
        if act_name == "relu":
            layers.append(nn.ReLU())
        elif act_name == "leaky_relu":
            layers.append(nn.LeakyReLU(0.01))
        elif act_name == "tanh":
            layers.append(nn.Tanh())
        elif act_name == "gelu":
            layers.append(nn.GELU())
        elif act_name == "elu":
            layers.append(nn.ELU())
        else:
            raise ValueError(f"Unsupported activation: {cfg['activation']}")
        cur_dim = h
    layers.append(nn.Linear(cur_dim, out_dim))
    return nn.Sequential(*layers)


# ── plotting helpers ──────────────────────────────────────────────────

def shade_train_val(ax, t_all, val_split):
    """Shade background: blue for training region, orange for validation."""
    N = len(t_all)
    n_train = int(round(N * (1.0 - val_split)))
    n_train = max(1, min(N - 1, n_train))
    t_split = t_all[n_train] if n_train < N else t_all[-1]
    ax.axvspan(t_all[0], t_split, color=TRAIN_BG_COLOR, zorder=0)
    ax.axvspan(t_split, t_all[-1], color=VAL_BG_COLOR, zorder=0)


def split_masks(num_samples, val_split):
    n_train = int(round(num_samples * (1.0 - val_split)))
    n_train = max(1, min(num_samples - 1, n_train))
    train_mask = np.zeros(num_samples, dtype=bool)
    train_mask[:n_train] = True
    val_mask = ~train_mask
    return train_mask, val_mask


def rmse(values):
    return float(np.sqrt(np.mean(values ** 2)))


def improvement_pct(old_rmse, new_rmse):
    if abs(old_rmse) <= 1e-12:
        return float("nan")
    return (1.0 - (new_rmse / old_rmse)) * 100.0


# ── run-dir resolution ────────────────────────────────────────────────

# Files required per model type
_REQUIRED_FILES_3D = ("predicted_relative_pose.csv", "true_relative_pose.csv")
_REQUIRED_FILES_BLINKERS = ("blinkers_seen_right.csv", "true_relative_pose.csv")
_REQUIRED_FILES_BLINKERS_AND_3D = (
    "blinkers_seen_right.csv", "predicted_relative_pose.csv", "true_relative_pose.csv",
)

_REQUIRED_FILES = {
    "3d":              _REQUIRED_FILES_3D,
    "blinkers":        _REQUIRED_FILES_BLINKERS,
    "blinkers_and_3d": _REQUIRED_FILES_BLINKERS_AND_3D,
}


def _is_valid_run_dir(path: Path, model_type: str = "3d") -> bool:
    req = _REQUIRED_FILES.get(model_type, _REQUIRED_FILES_3D)
    return path.is_dir() and all((path / f).exists() for f in req)


def _find_data_root(start_dir: Path) -> Path | None:
    start_dir = start_dir.resolve()
    for parent in (start_dir, *start_dir.parents):
        cand = parent / "data"
        if cand.is_dir():
            return cand
    return None


def _repo_data_root() -> Path | None:
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        data_root = _find_data_root(start)
        if data_root is not None:
            return data_root
    return None


def resolve_run_dir(user_value: str, model_type: str = "3d") -> Path:
    raw = Path(os.path.expanduser(user_value.strip()))
    as_is = raw if raw.is_absolute() else (Path.cwd() / raw)
    as_is = as_is.resolve()
    if _is_valid_run_dir(as_is, model_type):
        return as_is

    data_root = _repo_data_root()
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
        if _is_valid_run_dir(cand, model_type):
            return cand

    req = _REQUIRED_FILES.get(model_type, _REQUIRED_FILES_3D)
    msg_lines = [
        f"Run directory not found or missing required CSVs: {user_value}",
        f"Required files: {', '.join(req)}",
        f"Tried: {as_is}",
    ]
    for cand in candidates:
        msg_lines.append(f"Tried: {cand}")
    for cand in candidates:
        if cand.is_dir() and not _is_valid_run_dir(cand, model_type):
            try:
                valid_subs = [p.name for p in cand.iterdir()
                              if _is_valid_run_dir(p, model_type)]
            except OSError:
                valid_subs = []
            if valid_subs:
                msg_lines.append(
                    f"Hint: '{user_value}' looks like a parent directory. "
                    f"Available runs: {', '.join(sorted(valid_subs))}"
                )
                break
    raise FileNotFoundError("\n".join(msg_lines))


# ── load a results directory (config + model + normalization) ─────────

def load_results_dir(results_dir: str, run_dir_override: str | None = None):
    """
    Load config, normalization stats, model weights, and produce predictions
    for the dataset from a saved results directory.

    Supports both ``model_type="3d"`` and ``model_type="blinkers"``.
    The model type is read from ``model_info.json`` (falls back to ``"3d"``
    for results saved before that file existed).

    Returns
    -------
    dict with keys:
        cfg, model, model_type, Y_all, pred_res_all, t_all, val_split, run_dir
        For model_type="3d" only: pred_rel_xyz_all  (old-system baseline)
    """
    results_dir = os.path.abspath(results_dir)

    # ── Config ────────────────────────────────────────────────────────
    cfg_path = os.path.join(results_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Model info (type + dimensions) ────────────────────────────────
    model_info_path = os.path.join(results_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
    else:
        # Legacy results saved before model_info.json existed → assume 3d
        model_info = {"model_type": "3d", "in_dim": 3, "out_dim": 3}

    model_type = model_info.get("model_type", "3d")
    in_dim     = int(model_info.get("in_dim", 3))
    out_dim    = int(model_info.get("out_dim", 3))

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
        run_dir = resolve_run_dir(run_dir_override, model_type=model_type)
    else:
        dataset_path_file = os.path.join(results_dir, "dataset_path.txt")
        if not os.path.exists(dataset_path_file):
            raise FileNotFoundError(f"Dataset path file not found: {dataset_path_file}")
        with open(dataset_path_file, "r") as f:
            run_dir = resolve_run_dir(f.readline().strip(), model_type=model_type)

    # ── Model reconstruction ──────────────────────────────────────────
    model = build_model_from_config(cfg, in_dim=in_dim, out_dim=out_dim)
    weights_path = os.path.join(results_dir, "model_state_dict.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # ── Load & align data (depends on model_type) ─────────────────────
    true_rel = helpers.load_xyz(str(run_dir / "true_relative_pose.csv"))

    if model_type == "blinkers_and_3d":
        blinker_df = load_blinkers(str(run_dir / "blinkers_seen_right.csv"))
        blinker_features, blinker_times = preprocess_blinkers(blinker_df)
        pred_rel = helpers.load_xyz(str(run_dir / "predicted_relative_pose.csv"))
        X_all, Y_all, t_all = _align_blinkers_and_3d(
            blinker_features, blinker_times, pred_rel, true_rel,
        )
    elif model_type == "blinkers":
        blinker_df = load_blinkers(str(run_dir / "blinkers_seen_right.csv"))
        blinker_features, blinker_times = preprocess_blinkers(blinker_df)
        X_all, Y_all, t_all = _align_blinkers_to_target(
            blinker_features, blinker_times, true_rel,
        )
    else:
        # model_type == "3d"
        pred_rel = helpers.load_xyz(str(run_dir / "predicted_relative_pose.csv"))
        pred_rel_xyz_all, true_rel_xyz_all, t_all = helpers.align_two_streams(
            pred_rel, true_rel,
        )
        Y_all = true_rel_xyz_all
        X_all = pred_rel_xyz_all

    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all = X_all[ok], Y_all[ok]
    t_all = t_all[ok]

    # Always try to load the old UVDAR prediction for comparison plotting,
    # even when the NN input is something else (e.g. blinkers).
    pred_rel_xyz_all = None
    pred_rel_csv = run_dir / "predicted_relative_pose.csv"
    if pred_rel_csv.exists():
        try:
            pred_rel = helpers.load_xyz(str(pred_rel_csv))
            # Interpolate onto the same timeline used by the NN
            t_ms = (t_all * 1000).round().astype(np.int64)
            pred_aligned = helpers.interpolate_on_index(t_ms, pred_rel)
            pred_rel_xyz_all = pred_aligned.values.astype(np.float32)
        except Exception as e:
            print(f"Warning: could not load predicted_relative_pose.csv for comparison: {e}")
            pred_rel_xyz_all = None

    # ── Predict ───────────────────────────────────────────────────────
    X_all_n = (X_all - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_res_all = pred_norm * Y_std + Y_mean

    val_split = cfg.get("val_split", 0.2)

    result = {
        "cfg": cfg,
        "model": model,
        "model_type": model_type,
        "Y_all": Y_all,
        "pred_res_all": pred_res_all,
        "t_all": t_all,
        "val_split": val_split,
        "run_dir": run_dir,
    }
    if pred_rel_xyz_all is not None:
        result["pred_rel_xyz_all"] = pred_rel_xyz_all
    return result


# ── label helper ──────────────────────────────────────────────────────

_VAL_SUFFIX_RE = re.compile(r"_val[\d.]+$")


def friendly_name(results_dir: str) -> str:
    """Derive a short label from a results folder name, stripping _val{number}."""
    name = Path(results_dir).resolve().name
    return _VAL_SUFFIX_RE.sub("", name)
