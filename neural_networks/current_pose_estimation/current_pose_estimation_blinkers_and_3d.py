"""
current_pose_estimation_blinkers_and_3d.py
==========================================
Estimate the 3-D relative position of a UAV by combining:

  * LED (blinker) pixel detections  →  13-D feature vector
  * Old-system predicted relative pose  →  3-D  [x, y, z]

into a single **16-D** input vector, then training an MLP whose output
is the true relative pose [x, y, z].

Input CSVs
----------
* ``blinkers_seen_right.csv``        – columns: time, points, image_height, image_width
* ``predicted_relative_pose.csv``    – columns: time, x, y, z

Target CSV
----------
* ``true_relative_pose.csv``         – columns: time, x, y, z

Feature vector layout (16-D)
-----------------------------
  [u1, v1, m1,  u2, v2, m2,  u3, v3, m3,  u4, v4, m4,  n_visible,  pred_x, pred_y, pred_z]
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^
             blinker features (13)                       count          3-D pose (3)

The blinker part is preprocessed identically to
``current_pose_estimation_blinkers`` (same MAX_LEDS, MIN_LEDS, sorting,
normalization).  The 3-D pose values are concatenated as-is (the shared
``network_core.train_pipeline`` z-score-normalizes the whole vector).
"""

import os

import numpy as np

import helpers
import network_core
from current_pose_estimation_blinkers import (
    load_blinkers,
    preprocess_blinkers,
    INPUT_DIM as BLINKER_INPUT_DIM,   # 13
)

# Total input dimensionality: blinker features + [x, y, z]
INPUT_DIM = BLINKER_INPUT_DIM + 3    # 16


# ------------------------------------------------------------------ #
#  Alignment: merge blinkers + predicted_relative_pose + true_rel     #
# ------------------------------------------------------------------ #

def _align_all(
    blinker_features: np.ndarray,
    blinker_times: np.ndarray,
    pred_rel_df,
    true_rel_df,
):
    """
    Build the combined feature matrix and target on a common timeline.

    The blinker timestamps define the reference timeline; both
    ``predicted_relative_pose`` and ``true_relative_pose`` are
    interpolated onto it.

    Returns
    -------
    X : np.ndarray, shape (N, 16)
        Blinker features (13) + predicted relative pose (3).
    Y : np.ndarray, shape (N, 3)
        True relative pose.
    t : np.ndarray, shape (N,)
        Timestamps (seconds).
    """
    # Millisecond index from blinker timestamps
    blinker_ms = (blinker_times * 1000).round().astype(np.int64)

    # Deduplicate (keep first occurrence)
    _, unique_idx = np.unique(blinker_ms, return_index=True)
    unique_idx.sort()
    blinker_ms       = blinker_ms[unique_idx]
    blinker_features = blinker_features[unique_idx]

    # Interpolate both pose streams onto the blinker timeline
    pred_aligned = helpers.interpolate_on_index(blinker_ms, pred_rel_df)
    true_aligned = helpers.interpolate_on_index(blinker_ms, true_rel_df)

    pred_xyz = pred_aligned.values.astype(np.float32)   # (N, 3)
    Y        = true_aligned.values.astype(np.float32)   # (N, 3)

    # Concatenate: [blinker_13 | pred_xyz_3]
    X = np.concatenate([blinker_features, pred_xyz], axis=1)   # (N, 16)

    t_sec = blinker_ms.astype(np.float64) / 1000.0
    return X, Y, t_sec


# ------------------------------------------------------------------ #
#  Core training entry-point                                          #
# ------------------------------------------------------------------ #

def train_model_core(cfg, run_dir: str):
    """
    Core training: load blinker + predicted-relative-pose data,
    preprocess, concatenate, train model, predict.
    NO filesystem / plotting here.

    Parameters
    ----------
    cfg : dict
        Training configuration (layers, activation, optimizer, lr, …).
    run_dir : str
        Directory containing ``blinkers_seen_right.csv``,
        ``predicted_relative_pose.csv``, and ``true_relative_pose.csv``.

    Returns
    -------
    dict – same schema as ``current_pose_estimation_3d.train_model_core``
           (model, losses, arrays, normalization, etc.)
    """
    print("Using config:", cfg)

    # --- Load blinker detections ---
    blinker_csv = os.path.join(run_dir, "blinkers_seen_right.csv")
    blinker_df  = load_blinkers(blinker_csv)
    print(f"Loaded blinker CSV: {len(blinker_df)} rows")

    blinker_features, blinker_times = preprocess_blinkers(blinker_df)
    print(f"Valid blinker rows (≥MIN_LEDS detection): {len(blinker_features)}")

    # --- Load predicted relative pose (old system) ---
    pred_rel = helpers.load_xyz(os.path.join(run_dir, "predicted_relative_pose.csv"))
    print(f"Loaded predicted_relative_pose: {len(pred_rel)} rows")

    # --- Load target (true relative pose) ---
    true_rel = helpers.load_xyz(os.path.join(run_dir, "true_relative_pose.csv"))

    # --- Align on blinker timeline ---
    X_all, Y_all, t_all = _align_all(
        blinker_features, blinker_times, pred_rel, true_rel,
    )
    print(f"Aligned entries (blinkers+3d ↔ true_rel): {len(X_all)}")

    # --- Delegate to shared pipeline (in_dim=16, out_dim=3) ---
    artifacts = network_core.train_pipeline(
        X_all, Y_all, cfg,
        in_dim=INPUT_DIM,
        out_dim=3,
        t_all=t_all,
    )

    # Rename generic key for backward compat with result_manager / visualize
    artifacts["pred_res_all"] = artifacts.pop("pred_all")

    return artifacts
