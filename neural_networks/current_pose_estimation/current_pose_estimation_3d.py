import os
import numpy as np

import helpers
import network_core


# ---------- core ML logic (no plotting, no saving) ----------

def train_model_core(cfg, run_dir: str):
    """
    Core training: load data, train model, compute predictions.
    NO filesystem / plotting here.

    Returns:
        dict with model, losses, arrays, normalization, etc.
    """
    print("Using config:", cfg)

    # Load data — predicted relative pose (input) and true relative pose (target)
    pred_rel = helpers.load_xyz(os.path.join(run_dir, "predicted_relative_pose.csv"))
    true_rel = helpers.load_xyz(os.path.join(run_dir, "true_relative_pose.csv"))

    # Align on predicted_relative_pose timeline
    pred_rel_xyz_all, true_rel_xyz_all, t_all = helpers.align_two_streams(pred_rel, true_rel)
    print(f"Aligned predicted relative pose entries: {len(pred_rel_xyz_all)}")

    # Targets & inputs on EXACT same timestamps
    Y_all = true_rel_xyz_all            # true relative pose (ground truth)
    X_all = pred_rel_xyz_all            # predicted relative pose (old system)

    # Delegate splitting, normalizing, building, training, and predicting
    # to the shared pipeline (in_dim=3, out_dim=3).
    artifacts = network_core.train_pipeline(
        X_all, Y_all, cfg,
        in_dim=3, out_dim=3,
        t_all=t_all,
        extra_arrays={"pred_rel_xyz_all": pred_rel_xyz_all},
    )

    # Keep backward-compatible key expected by result_manager / visualize
    artifacts["pred_res_all"] = artifacts.pop("pred_all")

    return artifacts
