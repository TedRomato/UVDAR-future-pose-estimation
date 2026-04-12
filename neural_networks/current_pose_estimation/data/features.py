"""
data/features.py — Configurable feature construction for pose estimation.

Central function:
    build_features(cfg, run_dir) → (X, Y, t_sec, meta)

Reads ``cfg["features"]`` to decide which modalities, components, and
derived features to load and assemble.  Returns a single feature matrix
*X* whose columns are tracked by name in *meta["feature_names"]*.
"""

import os

import numpy as np
import pandas as pd

from data.loaders import load_xyz, load_blinkers, _parse_detections
from data.alignment import interpolate_on_index, forward_fill_with_age


# ------------------------------------------------------------------ #
#  Blinker preprocessing                                              #
# ------------------------------------------------------------------ #

def _blinker_feature_names(max_leds: int) -> list[str]:
    """Return canonical feature names for the blinker vector."""
    names = []
    for i in range(1, max_leds + 1):
        names.extend([f"u{i}", f"v{i}", f"m{i}"])
    names.append("n_visible")
    return names


def _preprocess_blinkers(
    df: pd.DataFrame,
    max_leds: int = 4,
    min_leds: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a blinkers DataFrame into ``(features, timestamps)``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_blinkers`.
    max_leds : int
        Fixed number of LED slots in the feature vector.
    min_leds : int
        Rows with fewer visible LEDs are dropped.

    Returns
    -------
    features : np.ndarray, shape (M, max_leds*3 + 1)
    timestamps : np.ndarray, shape (M,)
    """
    input_dim = max_leds * 3 + 1
    features_list: list[np.ndarray] = []
    timestamps_list: list[float] = []

    for _, row in df.iterrows():
        detections = _parse_detections(str(row["points"]))
        uv_pairs = [(d[0], d[1]) for d in detections if len(d) >= 2]

        if len(uv_pairs) < min_leds:
            continue

        uv_pairs.sort(key=lambda p: (p[0], p[1]))

        width = float(row["image_width"])
        height = float(row["image_height"])
        half_w, half_h = width / 2.0, height / 2.0

        uv_norm = [
            ((u - half_w) / half_w, (v - half_h) / half_h)
            for u, v in uv_pairs
        ]

        n_visible = min(len(uv_norm), max_leds)
        uv_used = uv_norm[:max_leds]

        vec = np.zeros(input_dim, dtype=np.float32)
        for i, (un, vn) in enumerate(uv_used):
            base = i * 3
            vec[base] = un
            vec[base + 1] = vn
            vec[base + 2] = 1.0
        vec[-1] = float(n_visible)

        features_list.append(vec)
        timestamps_list.append(float(row["time"]))

    features = np.stack(features_list, axis=0)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    return features, timestamps


def _deduplicate_ms(
    times_sec: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert seconds → ms, deduplicate, return (ms_array, filtered_features)."""
    ms = (times_sec * 1000).round().astype(np.int64)
    _, unique_idx = np.unique(ms, return_index=True)
    unique_idx.sort()
    return ms[unique_idx], features[unique_idx]


# ------------------------------------------------------------------ #
#  UVDAR component registry                                           #
# ------------------------------------------------------------------ #

# Each component maps to: (columns_to_read, feature_names)
_UVDAR_COMPONENTS = {
    "position": {
        "columns": ["x", "y", "z"],
        "names":   ["pred_x", "pred_y", "pred_z"],
    },
    # Future components:
    # "variance": {
    #     "columns": ["var_x", "var_y", "var_z"],
    #     "names":   ["var_x", "var_y", "var_z"],
    # },
    # "orientation": {
    #     "columns": ["qx", "qy", "qz", "qw"],
    #     "names":   ["qx", "qy", "qz", "qw"],
    # },
}


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def build_features(
    cfg: dict,
    run_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load and align input features based on ``cfg["features"]``.

    Parameters
    ----------
    cfg : dict
        Full config with a ``features`` block.
    run_dir : str
        Directory containing the required CSV files.

    Returns
    -------
    X : np.ndarray, shape (N, in_dim)
        Concatenated input features.
    Y : np.ndarray, shape (N, 3)
        True relative pose (target).
    t_sec : np.ndarray, shape (N,)
        Timestamps in seconds.
    meta : dict
        ``in_dim`` (int), ``feature_names`` (list[str]).
    """
    feat_cfg = cfg["features"]

    blinkers_cfg = feat_cfg.get("blinkers", {})
    uvdar_cfg = feat_cfg.get("uvdar", {})
    derived = feat_cfg.get("derived", []) or []

    use_blinkers = blinkers_cfg.get("enabled", False)
    use_uvdar = uvdar_cfg.get("enabled", False)

    if not use_blinkers and not use_uvdar:
        raise ValueError("At least one of features.blinkers or features.uvdar must be enabled.")

    # --- Always load target ---
    true_rel = load_xyz(os.path.join(run_dir, "true_relative_pose.csv"))

    parts: list[tuple[np.ndarray, list[str]]] = []  # (array, names)
    ref_ms: np.ndarray | None = None

    # ── Blinkers ──────────────────────────────────────────────────────
    if use_blinkers:
        max_leds = int(blinkers_cfg.get("max_leds", 4))
        min_leds = int(blinkers_cfg.get("min_leds", 2))

        blinker_df = load_blinkers(os.path.join(run_dir, "blinkers_seen_right.csv"))
        blinker_feats, blinker_times = _preprocess_blinkers(blinker_df, max_leds, min_leds)

        ref_ms, blinker_feats = _deduplicate_ms(blinker_times, blinker_feats)
        parts.append((blinker_feats, _blinker_feature_names(max_leds)))

        print(f"[build_features] blinkers: {len(blinker_feats)} valid rows, "
              f"{max_leds * 3 + 1}-D features")

    # ── UVDAR ─────────────────────────────────────────────────────────
    if use_uvdar:
        pred_rel = load_xyz(os.path.join(run_dir, "predicted_relative_pose.csv"))
        components = uvdar_cfg.get("components", ["position"])

        if ref_ms is not None:
            # Blinkers are the reference → forward-fill UVDAR
            for comp_name in components:
                comp = _UVDAR_COMPONENTS[comp_name]
                filled_vals, age_sec, valid_mask = forward_fill_with_age(
                    ref_ms, pred_rel, columns=comp["columns"],
                )
                # We must apply valid_mask after processing ALL components,
                # so store it now and filter later.

            # Apply validity mask (entries before first UVDAR reading)
            # Re-run forward fill to collect all parts consistently.
            # We need the valid_mask from the first component (all share same timeline).
            _, _, valid_mask = forward_fill_with_age(ref_ms, pred_rel, columns=["x", "y", "z"])

            # Filter reference timeline + existing parts
            ref_ms = ref_ms[valid_mask]
            parts = [(arr[valid_mask], names) for arr, names in parts]

            # Now add UVDAR components (re-fill on filtered timeline)
            for comp_name in components:
                comp = _UVDAR_COMPONENTS[comp_name]
                filled_vals, age_sec, _ = forward_fill_with_age(
                    ref_ms, pred_rel, columns=comp["columns"],
                )
                parts.append((filled_vals, comp["names"]))
                print(f"[build_features] uvdar.{comp_name}: {len(filled_vals)} rows, "
                      f"{len(comp['names'])}-D (forward-filled)")

            # Derived: age (only meaningful when blinkers + uvdar)
            if "age" in derived:
                parts.append((age_sec.reshape(-1, 1), ["age"]))
                print(f"[build_features] derived.age: appended")

        else:
            # UVDAR-only → UVDAR timestamps are the reference
            ref_ms = np.unique(pred_rel["t_ms"].values)
            pred_on_ref = pred_rel.set_index("t_ms").loc[ref_ms]

            for comp_name in components:
                comp = _UVDAR_COMPONENTS[comp_name]
                vals = pred_on_ref[comp["columns"]].values.astype(np.float32)
                parts.append((vals, comp["names"]))
                print(f"[build_features] uvdar.{comp_name}: {len(vals)} rows, "
                      f"{len(comp['names'])}-D")

            # Derived features that require both modalities: warn & skip
            if "age" in derived:
                print("[build_features] WARNING: 'age' derived feature requires "
                      "both blinkers and uvdar — skipping.")

    # ── Align target ──────────────────────────────────────────────────
    true_aligned = interpolate_on_index(ref_ms, true_rel)
    Y = true_aligned.values.astype(np.float32)

    # ── UVDAR baseline for RMSE comparison ────────────────────────────
    # Always try to load the raw UVDAR position prediction so that
    # downstream code can compute RMSE(UVDAR) vs RMSE(NN).
    uvdar_baseline = None
    pred_rel_csv = os.path.join(run_dir, "predicted_relative_pose.csv")
    if os.path.exists(pred_rel_csv):
        try:
            _pred_rel = load_xyz(pred_rel_csv)
            _aligned = interpolate_on_index(ref_ms, _pred_rel)
            uvdar_baseline = _aligned.values.astype(np.float32)
            print(f"[build_features] uvdar_baseline: {uvdar_baseline.shape} loaded for comparison")
        except Exception as e:
            print(f"[build_features] WARNING: could not load UVDAR baseline: {e}")

    # ── Concatenate all parts ─────────────────────────────────────────
    X = np.concatenate([p[0] for p in parts], axis=1).astype(np.float32)
    feature_names = [n for p in parts for n in p[1]]
    t_sec = ref_ms.astype(np.float64) / 1000.0

    meta = {
        "in_dim": X.shape[1],
        "feature_names": feature_names,
    }
    if uvdar_baseline is not None:
        meta["uvdar_baseline"] = uvdar_baseline

    print(f"[build_features] Final: X={X.shape}, Y={Y.shape}, "
          f"features={feature_names}")

    return X, Y, t_sec, meta
