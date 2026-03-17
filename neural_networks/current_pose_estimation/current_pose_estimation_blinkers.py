"""
current_pose_estimation_blinkers.py
===================================
Estimate the 3-D relative position of a UAV from LED (blinker) pixel
detections, using the shared ``network_core`` training pipeline.

Input CSV  : ``blinkers_seen_right.csv``
             columns: time, points, image_height, image_width
             ``points`` is a stringified list of [u, v, id] detections.

Target CSV : ``true_relative_pose.csv``
             columns: time, x, y, z  (+ optional orientation columns)

Preprocessing
-------------
1. Parse the ``points`` column from its string representation.
2. Drop the LED-ID from each detection → keep only (u, v).
3. Skip rows with zero detections.
4. Sort detections by u ascending, then v ascending.
5. Normalize pixel coordinates to roughly [-1, 1]:
       u_norm = (u - width/2) / (width/2)
       v_norm = (v - height/2) / (height/2)
6. Build a fixed-size 13-D feature vector per row (max 4 LEDs):
       [u1, v1, m1, u2, v2, m2, u3, v3, m3, u4, v4, m4, n_visible]
   where mi=1 if that LED slot is occupied, 0 otherwise.

Example
-------
Row:  2926.632, "[[567.0,168.0,-2.0],[577.0,168.0,1.0],[565.0,169.0,-2.0]]", 480, 752

After stripping ID → [(567,168), (577,168), (565,169)]
Sorted by (u,v)   → [(565,169), (567,168), (577,168)]
Normalized (w=752, h=480):
   u_norm = (u - 376) / 376,  v_norm = (v - 240) / 240
   → (0.5027, -0.2958), (0.5080, -0.3000), (0.5346, -0.3000)
Feature vector (3 LEDs, 1 padded slot):
   [0.5027, -0.2958, 1,  0.5080, -0.3000, 1,  0.5346, -0.3000, 1,  0, 0, 0,  3]
"""

import ast
import os

import numpy as np
import pandas as pd

import helpers
import network_core

# Maximum number of LED slots in the fixed-size feature vector.
MAX_LEDS = 4

# Minimum number of LEDs required to keep a row (rows with fewer are dropped).
MIN_LEDS = 2

# Dimensionality of the feature vector: 3 values per slot + n_visible.
INPUT_DIM = MAX_LEDS * 3 + 1  # 13


# ------------------------------------------------------------------ #
#  Blinker CSV loading & preprocessing                                #
# ------------------------------------------------------------------ #

def _parse_detections(raw: str) -> list[list[float]]:
    """Safely parse the stringified list of detections.

    Each detection is [u, v, id].  Returns a Python list of lists.
    Handles both well-formed strings and edge cases (empty, malformed).
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def load_blinkers(csv_path: str) -> pd.DataFrame:
    """
    Read ``blinkers_seen_right.csv`` and return a DataFrame with columns:
        time    – float timestamp (seconds)
        points  – raw string (kept for traceability)
        image_height, image_width – ints
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Ensure expected columns exist
    for col in ("time", "points", "image_height", "image_width"):
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in {csv_path}")
    return df


def preprocess_blinkers(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a blinkers DataFrame into a (features, timestamps) pair.

    Returns
    -------
    features : np.ndarray, shape (M, 13)
        One 13-D feature vector per *valid* row (rows with ≥1 detection).
    timestamps : np.ndarray, shape (M,)
        Corresponding timestamps (float seconds).
    """
    features_list: list[np.ndarray] = []
    timestamps_list: list[float] = []

    total_rows = len(df)
    log_every = max(1, total_rows // 10) if total_rows > 0 else 1
    skipped_empty = 0

    print(f"[preprocess_blinkers] Start: processing {total_rows} rows")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        detections = _parse_detections(str(row["points"]))

        # Drop LED ID from each detection → keep only (u, v)
        uv_pairs = [(d[0], d[1]) for d in detections if len(d) >= 2]

        # Skip rows with fewer than MIN_LEDS visible LEDs
        if len(uv_pairs) < MIN_LEDS:
            skipped_empty += 1
            if idx % log_every == 0 or idx == total_rows:
                print(
                    f"[preprocess_blinkers] Progress {idx}/{total_rows} | "
                    f"valid={len(features_list)} skipped_empty={skipped_empty}"
                )
            continue

        # Sort by u ascending, then v ascending
        uv_pairs.sort(key=lambda p: (p[0], p[1]))

        # Image resolution for this row
        width  = float(row["image_width"])
        height = float(row["image_height"])

        half_w = width  / 2.0
        half_h = height / 2.0

        # Normalize coordinates to approximately [-1, 1]
        uv_norm = [
            ((u - half_w) / half_w, (v - half_h) / half_h)
            for u, v in uv_pairs
        ]

        # Keep at most MAX_LEDS detections
        n_visible = min(len(uv_norm), MAX_LEDS)
        uv_used = uv_norm[:MAX_LEDS]

        # Build the fixed-size vector: [u1,v1,m1, u2,v2,m2, ..., n_visible]
        vec = np.zeros(INPUT_DIM, dtype=np.float32)
        for i, (un, vn) in enumerate(uv_used):
            base = i * 3
            vec[base]     = un       # normalized u
            vec[base + 1] = vn       # normalized v
            vec[base + 2] = 1.0      # mask: slot occupied

        vec[-1] = float(n_visible)

        features_list.append(vec)
        timestamps_list.append(float(row["time"]))

        if idx % log_every == 0 or idx == total_rows:
            print(
                f"[preprocess_blinkers] Progress {idx}/{total_rows} | "
                f"valid={len(features_list)} skipped_empty={skipped_empty}"
            )

    features   = np.stack(features_list, axis=0)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    print(
        f"[preprocess_blinkers] Done: valid={len(features_list)} "
        f"skipped_empty={skipped_empty}"
    )
    return features, timestamps


# ------------------------------------------------------------------ #
#  Alignment helper                                                   #
# ------------------------------------------------------------------ #

def _align_blinkers_to_target(
    blinker_features: np.ndarray,
    blinker_times: np.ndarray,
    target_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align blinker features with the true-relative-pose target on the
    blinker timeline using interpolation.

    Returns
    -------
    X : np.ndarray, shape (N, 13)
    Y : np.ndarray, shape (N, 3)
    t : np.ndarray, shape (N,)
    """
    # Build a ms index from blinker timestamps
    blinker_ms = (blinker_times * 1000).round().astype(np.int64)

    # Deduplicate (keep first occurrence)
    _, unique_idx = np.unique(blinker_ms, return_index=True)
    unique_idx.sort()
    blinker_ms       = blinker_ms[unique_idx]
    blinker_features = blinker_features[unique_idx]

    # Interpolate the target onto the blinker timeline
    target_aligned = helpers.interpolate_on_index(blinker_ms, target_df)
    Y = target_aligned.values.astype(np.float32)

    t_sec = blinker_ms.astype(np.float64) / 1000.0
    return blinker_features, Y, t_sec


# ------------------------------------------------------------------ #
#  Core training entry-point                                          #
# ------------------------------------------------------------------ #

def train_model_core(cfg, run_dir: str):
    """
    Core training: load blinker data, preprocess, train model, predict.
    NO filesystem / plotting here.

    Parameters
    ----------
    cfg : dict
        Training configuration (layers, activation, optimizer, lr, …).
    run_dir : str
        Directory containing ``blinkers_seen_right.csv`` and
        ``true_relative_pose.csv``.

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
    print(f"Valid blinker rows (≥1 detection): {len(blinker_features)}")

    # --- Load target (true relative pose) ---
    true_rel = helpers.load_xyz(os.path.join(run_dir, "true_relative_pose.csv"))

    # --- Align on blinker timeline ---
    X_all, Y_all, t_all = _align_blinkers_to_target(
        blinker_features, blinker_times, true_rel,
    )
    print(f"Aligned entries (blinker ↔ true_rel): {len(X_all)}")

    # --- Delegate to shared pipeline (in_dim=13, out_dim=3) ---
    artifacts = network_core.train_pipeline(
        X_all, Y_all, cfg,
        in_dim=INPUT_DIM,
        out_dim=3,
        t_all=t_all,
    )

    # Rename generic key for backward compat with result_manager / visualize
    artifacts["pred_res_all"] = artifacts.pop("pred_all")

    return artifacts
