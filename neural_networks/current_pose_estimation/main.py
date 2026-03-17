import argparse

# train script

from current_pose_estimation_3d import train_model_core as train_3d_core
from current_pose_estimation_blinkers import train_model_core as train_blinkers_core
from current_pose_estimation_blinkers_and_3d import train_model_core as train_blinkers_and_3d_core
from result_manager import save_results
from helpers import load_config

# Map of model type → core training function
_TRAIN_CORES = {
    "3d":               train_3d_core,
    "blinkers":         train_blinkers_core,
    "blinkers_and_3d":  train_blinkers_and_3d_core,
}


def train_model(cfg, run_dir: str, name: str, results_subdir: str = None,
                model_type: str = "3d"):
    """High-level wrapper: train core + save results, keep old API."""
    core_fn = _TRAIN_CORES[model_type]
    artifacts = core_fn(cfg, run_dir)
    results_dir = save_results(name, cfg, artifacts, run_dir=run_dir,
                               results_subdir=results_subdir,
                               model_type=model_type)
    return {
        "results_dir": results_dir,
        "final_train_loss": artifacts["final_train_loss"],
        "final_val_loss": artifacts["final_val_loss"],
        "train_losses": artifacts["train_losses"],
        "val_losses": artifacts["val_losses"],
    }


# ---------- CLI wrapper ----------

def main():
    cfg  = load_config()
    print("Loaded config:", cfg)

    ap = argparse.ArgumentParser(
        description="Train a pose-estimation network (3d or blinkers)."
    )
    ap.add_argument("run_dir", help="Folder containing the required CSV files")
    ap.add_argument("name", help="Base name of the results run (without val loss)")
    ap.add_argument("--model-type", choices=list(_TRAIN_CORES), default="3d",
                    help="Which model to train: '3d' (predicted_relative_pose → "
                         "true_relative_pose) or 'blinkers' (LED detections → "
                         "true_relative_pose). Default: 3d")
    ap.add_argument("--target-dir", default=None,
                    help="Sub-directory under ./results/ for saving outputs")
    args = ap.parse_args()

    result = train_model(cfg, args.run_dir, args.name,
                         results_subdir=args.target_dir,
                         model_type=args.model_type)


if __name__ == "__main__":
    main()