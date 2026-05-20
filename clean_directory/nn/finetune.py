#!/usr/bin/env python3
"""
finetune.py — Warm-start a sim-trained model and continue training on
real-world data.

Strategy:
    * Train  = first 80% of REAL  +  sim run's val slice (replay against
               catastrophic forgetting).
    * Val    = last  20% of REAL.
    * Norm   = reuse the sim run's X/Y mean/std (never recompute).
    * Init   = load the sim run's `model.pt` weights.
    * LR     = `source_lr / 10` by default (override via `--config`).
    * Save   = best checkpoint by real-val RMSE.

Outputs live in `clean_directory/nn/results/<name>_rmse<val>/` and have
the same layout as `train.py`, so `evaluation.visualize` /
`evaluation.compare` work without changes. Adds `finetune_meta.json`.

Usage:
    python finetune.py <source_results_dir> <real_run_dir> <name>
    python finetune.py results/sim_rmse1.42 ../data/real_world fine1 \\
                       --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml

from data.features import build_features
from models.mlp import build_model, build_optimizer
from training import (
    apply_normalization,
    set_seeds,
    train_val_split,
)
from utils import save_results


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _rmse_3d(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.linalg.norm(pred - true, axis=1) ** 2)))


def _forward(
    model: nn.Module,
    X: np.ndarray,
    X_mean: np.ndarray, X_std: np.ndarray,
    Y_mean: np.ndarray, Y_std: np.ndarray,
) -> np.ndarray:
    """Forward-pass *X* and return denormalised predictions."""
    Xn = apply_normalization(X, X_mean, X_std)
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xn).float()).numpy()
    return pred_n * Y_std + Y_mean


def _load_source(source_dir: str) -> tuple[dict, dict, dict]:
    """Load (cfg, norm_arrays, state_dict) from a sim results directory."""
    cfg_path  = os.path.join(source_dir, "config.yaml")
    norm_path = os.path.join(source_dir, "normalization.json")
    weights   = os.path.join(source_dir, "model.pt")
    for p in (cfg_path, norm_path, weights):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(norm_path) as f:
        norm = {k: np.asarray(v) for k, v in json.load(f).items()}
    state = torch.load(weights, map_location="cpu", weights_only=True)
    return cfg, norm, state


def _build_real_artifacts(
    model: nn.Module,
    cfg: dict,
    norm: dict,
    X_real: np.ndarray, Y_real: np.ndarray, t_real: np.ndarray,
    train_mask: np.ndarray, val_mask: np.ndarray,
    train_losses: list[float], val_losses: list[float],
) -> dict:
    """Build the artifacts dict that ``save_results`` expects."""
    pred_all = _forward(
        model, X_real,
        norm["X_mean"], norm["X_std"], norm["Y_mean"], norm["Y_std"],
    )
    return {
        "model":            model,
        "X_all":            X_real,
        "Y_all":            Y_real,
        "t_all":            t_real,
        "idx_tr":           np.where(train_mask)[0],
        "idx_val":          np.where(val_mask)[0],
        "train_mask":       train_mask,
        "val_mask":         val_mask,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss":   float(val_losses[-1]),
        "pred_all":         pred_all,
        "norm_stats":       norm,  # unchanged source norm stats
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune a sim-trained model on real-world data.")
    ap.add_argument("source_results_dir",
                    help="Sim results folder (config.yaml + model.pt + "
                         "normalization.json).")
    ap.add_argument("real_run_dir",
                    help="Parsed real-world dataset directory.")
    ap.add_argument("name",
                    help="Base name for the new results folder.")
    ap.add_argument("--config", default=None,
                    help="Optional YAML overriding learning_rate, epochs, "
                         "batch_size. Anything else is ignored.")
    args = ap.parse_args()

    # ── Load source run ───────────────────────────────────────────────
    source_dir = os.path.abspath(args.source_results_dir)
    real_dir   = os.path.abspath(args.real_run_dir)
    print(f"[finetune] source : {source_dir}")
    print(f"[finetune] real   : {real_dir}")

    src_cfg, norm, state = _load_source(source_dir)
    sim_dir = src_cfg["_meta"]["dataset_path"]
    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(
            f"Sim dataset path from source config not found: {sim_dir}")
    print(f"[finetune] sim    : {sim_dir}")

    # ── Apply CLI overrides ───────────────────────────────────────────
    cfg = copy.deepcopy(src_cfg)
    cfg["learning_rate"] = float(src_cfg["learning_rate"]) / 10.0  # default: ÷10
    cfg["epochs"]        = 30
    if args.config is not None:
        with open(args.config) as f:
            user_overrides = yaml.safe_load(f) or {}
        for k in ("learning_rate", "epochs", "batch_size"):
            if k in user_overrides:
                cfg[k] = user_overrides[k]
                print(f"[finetune] override {k} = {cfg[k]}")
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["epochs"]        = int(cfg["epochs"])
    cfg["batch_size"]    = int(cfg["batch_size"])
    cfg["val_split"]     = 0.2  # for the real-world split, sequential

    # Track lineage in the saved config
    cfg["_meta"] = {**src_cfg.get("_meta", {}),
                    "parent_run": source_dir,
                    "finetune": True}

    # ── Build features ────────────────────────────────────────────────
    print("[finetune] building SIM features (for replay)...")
    X_sim, Y_sim, t_sim, _ = build_features(src_cfg, sim_dir)
    ok = np.isfinite(X_sim).all(axis=1) & np.isfinite(Y_sim).all(axis=1)
    X_sim, Y_sim, t_sim = X_sim[ok], Y_sim[ok], t_sim[ok]
    _, idx_val_sim = train_val_split(len(X_sim), src_cfg)
    X_sim_val, Y_sim_val = X_sim[idx_val_sim], Y_sim[idx_val_sim]
    print(f"[finetune]   sim val (replay): {len(X_sim_val)} rows")

    print("[finetune] building REAL features...")
    X_real, Y_real, t_real, _ = build_features(src_cfg, real_dir)
    ok = np.isfinite(X_real).all(axis=1) & np.isfinite(Y_real).all(axis=1)
    X_real, Y_real, t_real = X_real[ok], Y_real[ok], t_real[ok]

    # Sequential 80/20 split on real
    n_real = len(X_real)
    real_split_cfg = {**src_cfg, "split_mode": "sequential", "val_split": 0.2,
                      "val_padding": 0}
    idx_tr_real, idx_val_real = train_val_split(n_real, real_split_cfg)
    X_real_tr, Y_real_tr = X_real[idx_tr_real], Y_real[idx_tr_real]
    X_real_val, Y_real_val = X_real[idx_val_real], Y_real[idx_val_real]
    print(f"[finetune]   real train: {len(X_real_tr)} rows  "
          f"real val: {len(X_real_val)} rows")

    if len(X_sim_val) > 4 * len(X_real_tr):
        print(f"[finetune]   WARNING: replay set ({len(X_sim_val)}) is more "
              f"than 4× real-train ({len(X_real_tr)}); replay may swamp "
              f"the real signal. Consider trimming.")

    # ── Combine train: real + sim replay ──────────────────────────────
    X_train = np.concatenate([X_real_tr, X_sim_val], axis=0)
    Y_train = np.concatenate([Y_real_tr, Y_sim_val], axis=0)
    print(f"[finetune]   combined train: {len(X_train)} rows "
          f"(real {len(X_real_tr)} + sim replay {len(X_sim_val)})")

    # ── Normalise everything with SOURCE stats ────────────────────────
    Xm, Xs = norm["X_mean"], norm["X_std"]
    Ym, Ys = norm["Y_mean"], norm["Y_std"]

    Xtr_n  = apply_normalization(X_train,    Xm, Xs)
    Xrv_n  = apply_normalization(X_real_val, Xm, Xs)
    Xsv_n  = apply_normalization(X_sim_val,  Xm, Xs)

    Ytr_n  = apply_normalization(Y_train,    Ym, Ys)
    Yrv_n  = apply_normalization(Y_real_val, Ym, Ys)
    Ysv_n  = apply_normalization(Y_sim_val,  Ym, Ys)

    Xtr_t = torch.from_numpy(Xtr_n).float()
    Ytr_t = torch.from_numpy(Ytr_n).float()
    Xrv_t = torch.from_numpy(Xrv_n).float()
    Yrv_t = torch.from_numpy(Yrv_n).float()
    Xsv_t = torch.from_numpy(Xsv_n).float()
    Ysv_t = torch.from_numpy(Ysv_n).float()

    # ── Build model and warm-start ────────────────────────────────────
    g = set_seeds(int(cfg.get("seed", 42)))
    in_dim, out_dim = X_train.shape[1], Y_train.shape[1]
    model = build_model(cfg, in_dim, out_dim)
    model.load_state_dict(state)
    print(f"[finetune] warm-started from {os.path.basename(source_dir)}/model.pt")

    opt = build_optimizer(cfg, model)

    # ── Zero-shot baselines (in METRES, denormalised) ─────────────────
    pred_real_val_before = _forward(model, X_real_val, Xm, Xs, Ym, Ys)
    pred_sim_val_before  = _forward(model, X_sim_val,  Xm, Xs, Ym, Ys)
    rmse_real_before = _rmse_3d(pred_real_val_before, Y_real_val)
    rmse_sim_before  = _rmse_3d(pred_sim_val_before,  Y_sim_val)
    print(f"[finetune] zero-shot real-val RMSE: {rmse_real_before:.4f} m")
    print(f"[finetune] zero-shot sim-val  RMSE: {rmse_sim_before:.4f} m")

    # ── Training loop with best-by-real-val checkpointing ─────────────
    loss_fn = nn.MSELoss()
    use_minibatch = cfg["batch_size"] > 0
    if use_minibatch:
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_real_val = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    best_epoch    = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_minibatch:
            for xb, yb in dl_tr:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            pred = model(Xtr_t)
            loss = loss_fn(pred, Ytr_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss  = loss_fn(model(Xtr_t), Ytr_t).item()
            rv_loss  = loss_fn(model(Xrv_t), Yrv_t).item()
            sv_loss  = loss_fn(model(Xsv_t), Ysv_t).item()
        train_losses.append(tr_loss)
        val_losses.append(rv_loss)

        if rv_loss < best_real_val:
            best_real_val = rv_loss
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg["epochs"]:
            print(f"  epoch {epoch:3d}  train {tr_loss:.4f}  "
                  f"real_val {rv_loss:.4f}  sim_val {sv_loss:.4f}")

    print(f"[finetune] best real-val (normalised MSE) at epoch {best_epoch}: "
          f"{best_real_val:.4f}")
    model.load_state_dict(best_state)

    # ── After-training metrics in METRES ──────────────────────────────
    pred_real_val_after = _forward(model, X_real_val, Xm, Xs, Ym, Ys)
    pred_sim_val_after  = _forward(model, X_sim_val,  Xm, Xs, Ym, Ys)
    rmse_real_after = _rmse_3d(pred_real_val_after, Y_real_val)
    rmse_sim_after  = _rmse_3d(pred_sim_val_after,  Y_sim_val)
    print(f"[finetune] real-val RMSE: {rmse_real_before:.4f}  →  "
          f"{rmse_real_after:.4f} m")
    print(f"[finetune] sim-val  RMSE: {rmse_sim_before:.4f}  →  "
          f"{rmse_sim_after:.4f} m  (forgetting check)")
    if rmse_sim_after > 1.5 * rmse_sim_before:
        print("[finetune]   WARNING: sim-val RMSE grew >50%. "
              "Lower LR or shrink replay ratio.")

    # ── Build train/val masks against real timeline only ──────────────
    train_mask = np.zeros(n_real, dtype=bool)
    val_mask   = np.zeros(n_real, dtype=bool)
    train_mask[idx_tr_real]  = True
    val_mask[idx_val_real]   = True

    artifacts = _build_real_artifacts(
        model, cfg, norm,
        X_real, Y_real, t_real,
        train_mask, val_mask,
        train_losses, val_losses,
    )

    # ── Save (mirrors train.py's layout) ──────────────────────────────
    feat_meta = {
        "in_dim":        in_dim,
        "feature_names": src_cfg.get("_meta", {}).get("feature_names", []),
    }
    results_dir = save_results(
        args.name, cfg, artifacts, real_dir,
        results_subdir=None, meta=feat_meta,
    )

    # Lineage / forgetting log
    finetune_meta = {
        "source_results_dir":      source_dir,
        "real_run_dir":            real_dir,
        "sim_run_dir":             sim_dir,
        "n_real_train":            int(len(X_real_tr)),
        "n_real_val":              int(len(X_real_val)),
        "n_sim_replay":            int(len(X_sim_val)),
        "lr_used":                 cfg["learning_rate"],
        "epochs_used":             cfg["epochs"],
        "best_epoch":              best_epoch,
        "rmse_real_val_before":    rmse_real_before,
        "rmse_real_val_after":     rmse_real_after,
        "rmse_sim_val_before":     rmse_sim_before,
        "rmse_sim_val_after":      rmse_sim_after,
    }
    with open(os.path.join(results_dir, "finetune_meta.json"), "w") as f:
        json.dump(finetune_meta, f, indent=2)
    print(f"[finetune] wrote {os.path.join(results_dir, 'finetune_meta.json')}")


if __name__ == "__main__":
    main()
