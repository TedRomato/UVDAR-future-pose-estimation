import argparse

# train script

from current_pose_estimation_nn import train_model_core
from result_manager import save_results
from helpers import load_config

def train_model(cfg, run_dir: str, name: str):
    """High-level wrapper: train core + save results, keep old API."""
    artifacts = train_model_core(cfg, run_dir)
    results_dir = save_results(name, cfg, artifacts, run_dir=run_dir)
    return {
        "results_dir": results_dir,
        "final_train_loss": artifacts["final_train_loss"],
        "final_val_loss": artifacts["final_val_loss"],
    }


# ---------- CLI wrapper ----------

def main():
    cfg  = load_config()
    print("Loaded config:", cfg)

    ap = argparse.ArgumentParser(
        description="UVDAR [x,y,z] -> residual (odom2-odom1) with strict time alignment."
    )
    ap.add_argument("run_dir", help="Folder containing estimations.csv, odom1.csv, odom2.csv")
    ap.add_argument("name", help="Base name of the results run (without val loss)")
    args = ap.parse_args()

    result = train_model(cfg, args.run_dir, args.name)


if __name__ == "__main__":
    main()