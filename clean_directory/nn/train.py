#!/usr/bin/env python3
"""
train.py — Single-run training CLI.

Usage:
    python train.py <run_dir> <name> [--config path/to/config.yaml] [--target-dir subdir]

Example:
    python train.py ../data/sim baseline-fusion --config configs/default.yaml
"""

import argparse

from data.features import build_features
from training import train_pipeline
from utils import load_config, save_results


def main():
    ap = argparse.ArgumentParser(
        description="Train a pose-estimation network (config-driven).",
    )
    ap.add_argument("run_dir", help="Folder with the required CSV files")
    ap.add_argument("name", help="Base name for the results folder")
    ap.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    ap.add_argument(
        "--target-dir", default=None,
        help="Sub-directory under ./results/ for outputs",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:", cfg)

    # Build features from config
    X, Y, t_ns, meta = build_features(cfg, args.run_dir)

    # Extract UVDAR baseline for residual learning / metrics
    uvdar_baseline = meta.pop("uvdar_baseline", None)

    # Train
    artifacts = train_pipeline(
        X, Y, cfg,
        in_dim=meta["in_dim"],
        t_all=t_ns,
        uvdar_baseline=uvdar_baseline,
    )

    # Save
    results_dir = save_results(
        args.name, cfg, artifacts, args.run_dir,
        results_subdir=args.target_dir,
        meta=meta,
    )

    print(f"Final val loss: {artifacts['final_val_loss']:.6f}")
    print(f"Results: {results_dir}")

    return {
        "results_dir": results_dir,
        "final_train_loss": artifacts["final_train_loss"],
        "final_val_loss": artifacts["final_val_loss"],
        "train_losses": artifacts["train_losses"],
        "val_losses": artifacts["val_losses"],
    }


if __name__ == "__main__":
    main()
