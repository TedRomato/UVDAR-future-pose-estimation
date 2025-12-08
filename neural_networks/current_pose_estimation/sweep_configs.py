#!/usr/bin/env python3
import os
import argparse
import multiprocessing as mp

# --------------------------------------------------------------------
# LIMIT THREADS (helps prevent freezing with many processes)
# --------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from main import train_model


# --------------------------------------------------------------------
# CONFIG SWEEP DEFINITIONS (unchanged)
# --------------------------------------------------------------------
BASE_CFG = {
    "epochs": 200,
    "val_split": 0.2,
    "weight_decay": 1e-4,
    "activation": "relu",
    "optimizer": "adamw",
}

CONFIGS = [
    {**BASE_CFG, "learning_rate": 1e-3, "layers": [256, 128], "batch_size": 0},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [256, 128, 64], "batch_size": 256},
    {**BASE_CFG, "learning_rate": 1e-4, "layers": [256], "batch_size": 512},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [256, 256], "batch_size": 0},

    {**BASE_CFG, "learning_rate": 1e-3, "layers": [128, 128], "batch_size": 256},
    {**BASE_CFG, "learning_rate": 5e-4, "layers": [512, 256], "batch_size": 512},
    {**BASE_CFG, "learning_rate": 1e-4, "layers": [128, 64], "batch_size": 0},

    {**BASE_CFG, "learning_rate": 3e-4, "layers": [256, 256, 128], "batch_size": 256, "activation": "gelu"},
    {**BASE_CFG, "learning_rate": 1e-3, "layers": [256, 256, 256], "batch_size": 512, "activation": "gelu"},

    {**BASE_CFG, "learning_rate": 5e-4, "layers": [256, 128, 64], "batch_size": 128, "optimizer": "adam"},

    {**BASE_CFG, "learning_rate": 3e-4, "layers": [512, 256, 128], "batch_size": 256,
     "optimizer": "adam", "weight_decay": 3e-4},

    {**BASE_CFG, "learning_rate": 1e-4, "layers": [256, 256], "batch_size": 1024,
     "optimizer": "sgd", "activation": "tanh", "weight_decay": 1e-3},
]


# --------------------------------------------------------------------
# WORKER
# --------------------------------------------------------------------

def _run_single(args):
    """Wrapper so it works nicely with multiprocessing."""
    idx, cfg, run_dir, prefix = args
    base_name = f"{prefix}-{idx}"

    print(f"[{base_name}] Starting with cfg={cfg}")
    try:
        res = train_model(cfg, run_dir, base_name)
        print(f"[{base_name}] Done. val_loss={res['final_val_loss']:.6f}")
        return {
            "base_name": base_name,
            "cfg": cfg,
            "results_dir": res["results_dir"],
            "final_train_loss": res["final_train_loss"],
            "final_val_loss": res["final_val_loss"],
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


# --------------------------------------------------------------------
# MAIN WITH ARGUMENTS
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a configuration sweep of neural models.")

    parser.add_argument(
        "--run-dir",
        default="../../data/random_flight/csv_data/test2",
        help="Path to dataset folder containing estimations.csv, odom1.csv, odom2.csv",
    )

    parser.add_argument(
        "--prefix",
        default="baseline",
        help="Prefix used for naming the result folders (default: baseline)",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Number of parallel processes (default: 2)",
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    prefix = args.prefix
    nproc = args.processes

    print("===================================================")
    print(f" Dataset:          {run_dir}")
    print(f" Name prefix:      {prefix}")
    print(f" Parallel workers: {nproc}")
    print("===================================================")

    os.makedirs("results", exist_ok=True)

    # Prepare jobs
    jobs = [(idx, cfg, run_dir, prefix) for idx, cfg in enumerate(CONFIGS)]

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=nproc) as pool:
        results = pool.map(_run_single, jobs)

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

    ok_results = [r for r in results if r["error"] is None]
    if ok_results:
        print("\nSorted by validation loss:")
        for r in sorted(ok_results, key=lambda x: x["final_val_loss"]):
            print(
                f"{r['base_name']} -> val_loss={r['final_val_loss']:.6f}, dir={r['results_dir']}"
            )


if __name__ == "__main__":
    main()
