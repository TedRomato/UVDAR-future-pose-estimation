#!/usr/bin/env python3
"""
sweep.py — Parallel hyperparameter sweep.

Usage:
    python sweep.py --sweep-config configs/sweep.yaml --run-dir ../data/sim
    python sweep.py --sweep-config configs/sweep.yaml --processes 8 --prefix my-sweep
"""

import os
import argparse
import multiprocessing as mp

# Limit threads to avoid freezing with many parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from data.features import build_features
from training import train_pipeline
from utils import load_sweep_configs, save_results


# ------------------------------------------------------------------ #
#  Worker                                                             #
# ------------------------------------------------------------------ #

def _run_single(args):
    """Train a single configuration (designed for multiprocessing.Pool)."""
    idx, cfg, run_dir, prefix, target_dir = args
    sweep_name = cfg.pop("_sweep_name", str(idx))
    sweep_group = cfg.pop("_sweep_group", "ungrouped")
    base_name = f"{prefix}-{sweep_group}-{sweep_name}"

    # Build result sub-directory: <target_dir>/<group>
    if target_dir:
        results_subdir = os.path.join(target_dir, sweep_group)
    else:
        results_subdir = sweep_group

    print(f"[{base_name}] Starting with cfg: { {k: v for k, v in cfg.items() if k != 'features'} }")

    try:
        # Build features from config
        X, Y, t_ns, meta = build_features(cfg, run_dir)

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
            f"{prefix}-{sweep_name}", cfg, artifacts, run_dir,
            results_subdir=results_subdir,
            meta=meta,
        )

        # Read back metrics for the summary print
        import json as _json
        metrics_path = os.path.join(results_dir, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as _f:
                metrics = _json.load(_f)

        print(f"[{base_name}] Done. val_loss={artifacts['final_val_loss']:.6f}")

        return {
            "base_name": base_name,
            "cfg": cfg,
            "results_dir": results_dir,
            "final_train_loss": artifacts["final_train_loss"],
            "final_val_loss": artifacts["final_val_loss"],
            "metrics": metrics,
            "error": None,
        }

    except Exception as e:
        print(f"[{base_name}] ERROR: {e}")
        return {
            "base_name": base_name,
            "cfg": cfg,
            "results_dir": None,
            "final_train_loss": None,
            "final_val_loss": None,
            "error": str(e),
        }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep.")

    parser.add_argument(
        "--group", default=None,
        help="Comma-separated sweep group name(s) to run (default: all groups)",
    )
    parser.add_argument(
        "--run-dir",
        default="../data/sim",
        help="Path to dataset folder with CSV files",
    )
    parser.add_argument(
        "--sweep-config", required=True,
        help="Path to sweep YAML (e.g. configs/sweep.yaml)",
    )
    parser.add_argument(
        "--prefix", default="sweep",
        help="Prefix for result folder names (default: sweep)",
    )
    parser.add_argument(
        "--processes", type=int, default=12,
        help="Number of parallel processes (default: 12)",
    )
    parser.add_argument(
        "--target-dir", default=None,
        help="Sub-directory under ./results/ for outputs",
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    configs = load_sweep_configs(path=args.sweep_config, group=args.group)

    print("===================================================")
    print(f" Dataset:          {run_dir}")
    print(f" Group filter:     {args.group or '(all)'}")
    print(f" Sweep configs:    {len(configs)} entries")
    print(f" Name prefix:      {args.prefix}")
    print(f" Parallel workers: {args.processes}")
    print(f" Target dir:       ./results/{args.target_dir or ''}")
    print("===================================================")

    os.makedirs("results", exist_ok=True)

    jobs = [
        (idx, cfg, run_dir, args.prefix, args.target_dir)
        for idx, cfg in enumerate(configs)
    ]

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=args.processes) as pool:
        results = pool.map(_run_single, jobs)

    # Summary
    print("\n================ SWEEP SUMMARY ================")
    for r in results:
        if r["error"] is None:
            print(
                f"{r['base_name']}: "
                f"val_loss={r['final_val_loss']:.6f}, "
                f"train_loss={r['final_train_loss']:.6f}, "
                f"dir={r['results_dir']}"
            )
        else:
            print(f"{r['base_name']}: ERROR -> {r['error']}")

    ok = [r for r in results if r["error"] is None]
    if ok:
        print("\nSorted by NN RMSE (validation):")
        for r in sorted(ok, key=lambda x: x.get("metrics", {}).get("nn_rmse_val", x["final_val_loss"])):
            m = r.get("metrics", {})
            nn_rmse = m.get("nn_rmse_val", "N/A")
            uvdar_rmse = m.get("uvdar_rmse_val", "N/A")
            impr = m.get("improvement_pct", "N/A")
            nn_str = f"{nn_rmse:.4f}" if isinstance(nn_rmse, (int, float)) else nn_rmse
            uvdar_str = f"{uvdar_rmse:.4f}" if isinstance(uvdar_rmse, (int, float)) else uvdar_rmse
            impr_str = f"{impr:.2f}%" if isinstance(impr, (int, float)) else impr
            print(f"  {r['base_name']} -> NN={nn_str}m, UVDAR={uvdar_str}m, impr={impr_str}, dir={r['results_dir']}")


if __name__ == "__main__":
    main()
