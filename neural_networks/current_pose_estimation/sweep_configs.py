#!/usr/bin/env python3
import os
import argparse
import multiprocessing as mp
from datetime import datetime

# --------------------------------------------------------------------
# LIMIT THREADS (helps prevent freezing with many processes)
# --------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from aim import Run
from main import train_model


# --------------------------------------------------------------------
# CONFIG SWEEP DEFINITIONS (unchanged)
# --------------------------------------------------------------------
BASE_CFG = {
    "epochs": 25,
    "val_split": 0.7,
    "val_padding": 20,
    "weight_decay": 1e-4,
    "activation": "relu",
    "optimizer": "adamw",
    "seed": 42,
}

CONFIGS = []    

CONFIGS = [
    # --- A) Capacity sweep (AdamW, relu, lr=3e-4, bs=256) ---
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [8],      "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [8,8],      "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [16],      "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [16, 16],  "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32],      "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32],  "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [64, 64],  "batch_size": 256},
]

CONFIGS += [
    # --- B) LR sweep (layers=[32,32], AdamW, relu, bs=256) ---
    {**BASE_CFG, "learning_rate": 1e-3, "layers": [32, 32], "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 256},
    {**BASE_CFG, "learning_rate": 1e-4, "layers": [32, 32], "batch_size": 256},
]

CONFIGS += [
    # --- C) Batch size sweep (layers=[32,32], AdamW, relu, lr=3e-4) ---
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 128},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 256},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 512},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 0},   # full-batch baseline
]

CONFIGS += [
    # --- D) Activation check (same everything else) ---
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 256, "activation": "tanh"},
    {**BASE_CFG, "learning_rate": 3e-4, "layers": [32, 32], "batch_size": 256, "activation": "gelu"},
]




# --------------------------------------------------------------------
# WORKER
# --------------------------------------------------------------------

def _run_single(args):
    """Wrapper so it works nicely with multiprocessing."""
    idx, cfg, run_dir, prefix, aim_repo, target_dir, model_type = args
    base_name = f"{prefix}-{idx}"

    print(f"[{base_name}] Starting with cfg={cfg}")
    
    # Initialize Aim run for tracking
    aim_run = Run(repo=aim_repo, experiment=prefix)
    aim_run.name = base_name
    
    # Log metadata
    aim_run["date"] = datetime.now().isoformat()
    aim_run["dataset_path"] = run_dir
    
    # Log all hyperparameters individually for better filtering
    aim_run["hparams"] = {
        "model_type": model_type,
        "learning_rate": cfg.get("learning_rate"),
        "epochs": cfg.get("epochs"),
        "layers": cfg.get("layers"),
        "batch_size": cfg.get("batch_size"),
        "val_split": cfg.get("val_split"),
        "val_padding": cfg.get("val_padding"),
        "weight_decay": cfg.get("weight_decay"),
        "activation": cfg.get("activation"),
        "optimizer": cfg.get("optimizer"),
        "seed": cfg.get("seed"),
    }
    
    try:
        res = train_model(cfg, run_dir, base_name, results_subdir=target_dir,
                          model_type=model_type)
        
        # Log learning curves (per-epoch losses)
        train_losses = res["train_losses"]
        val_losses = res["val_losses"]
        for epoch, (tr_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            aim_run.track(tr_loss, name="loss", step=epoch, context={"subset": "train"})
            aim_run.track(val_loss, name="loss", step=epoch, context={"subset": "val"})
        
        # Log final metrics
        aim_run["final_train_loss"] = res["final_train_loss"]
        aim_run["final_val_loss"] = res["final_val_loss"]
        aim_run["results_dir"] = res["results_dir"]
        
        print(f"[{base_name}] Done. val_loss={res['final_val_loss']:.6f}")
        aim_run.close()
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
        aim_run["error"] = str(e)
        aim_run.close()
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
        help="Path to dataset folder containing neccesary csv files",
    )

    parser.add_argument(
        "--prefix",
        default="baseline",
        help="Prefix used for naming the result folders (default: baseline)",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=12,
        help="Number of parallel processes (default: 12)",
    )

    parser.add_argument(
        "--aim-repo",
        default=".aim",
        help="Path to Aim repository for tracking (default: .aim)",
    )

    parser.add_argument(
        "--target-dir",
        default=None,
        help="Sub-directory under ./results/ for saving outputs (e.g. 'my_experiment')",
    )

    parser.add_argument(
        "--model-type",
        choices=["3d", "blinkers", "blinkers_and_3d"],
        default="3d",
        help="Which model to train (default: 3d)",
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    prefix = args.prefix
    nproc = args.processes
    aim_repo = os.path.abspath(args.aim_repo)
    target_dir = args.target_dir
    model_type = args.model_type

    print("===================================================")
    print(f" Dataset:          {run_dir}")
    print(f" Model type:       {model_type}")
    print(f" Name prefix:      {prefix}")
    print(f" Parallel workers: {nproc}")
    print(f" Aim repo:         {aim_repo}")
    print(f" Target dir:       ./results/{target_dir or ''}")
    print("===================================================")

    os.makedirs("results", exist_ok=True)

    # Prepare jobs
    jobs = [(idx, cfg, run_dir, prefix, aim_repo, target_dir, model_type)
            for idx, cfg in enumerate(CONFIGS)]

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
