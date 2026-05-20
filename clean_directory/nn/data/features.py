"""
data/features.py — Feature construction for the clean_directory NN.

Central function:
    build_features(cfg, run_dir) → (X, Y, t_ns, meta)

Reads ``cfg["features"]`` to decide which modalities and components to
load and assemble.  All inputs share a common nanosecond timestamp `t`,
so alignment is a simple exact-timestamp lookup — no interpolation, no
forward fill, no aging.  Returns a single feature matrix *X* whose
columns are tracked by name in *meta["feature_names"]*.
"""

import os

import numpy as np
import pandas as pd

from data.loaders import DATASET_LAYOUT, load_xyz, load_blinkers, _parse_detections


# ------------------------------------------------------------------ #
#  Blinker preprocessing                                              #
# ------------------------------------------------------------------ #

def _blinker_feature_names(max_leds: int, encoding: str = "absolute") -> list[str]:
    """Return canonical feature names for the blinker vector.

    encoding == "absolute":
        [u1, v1, m1, ..., uK, vK, mK, n_visible]
    encoding == "centroid":
        [c_u, c_v, du1, dv1, m1, ..., duK, dvK, mK, n_visible]
    """
    names: list[str] = []
    if encoding == "centroid":
        names.extend(["c_u", "c_v"])
        for i in range(1, max_leds + 1):
            names.extend([f"du{i}", f"dv{i}", f"m{i}"])
    elif encoding == "absolute":
        for i in range(1, max_leds + 1):
            names.extend([f"u{i}", f"v{i}", f"m{i}"])
    else:
        raise ValueError(f"Unknown blinker encoding: {encoding!r}")
    names.append("n_visible")
    return names


def _preprocess_blinkers(
    df: pd.DataFrame,
    max_leds: int = 4,
    min_leds: int = 2,
    encoding: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a blinkers DataFrame into ``(features, t_ns)``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_blinkers` (columns ``t_ns, points,
        image_height, image_width``).
    max_leds : int
        Fixed number of LED slots in the feature vector.
    min_leds : int
        Rows with fewer visible LEDs are dropped.
    encoding : {"absolute", "centroid"}
        Per-row layout. See :func:`_blinker_feature_names`.

    Returns
    -------
    features : np.ndarray, shape (M, in_dim)
        ``in_dim = max_leds*3 + 1`` for ``absolute``,
        ``in_dim = 2 + max_leds*3 + 1`` for ``centroid``.
    t_ns : np.ndarray of int64, shape (M,)
    """
    if encoding not in ("absolute", "centroid"):
        raise ValueError(f"Unknown blinker encoding: {encoding!r}")

    in_dim = max_leds * 3 + 1
    if encoding == "centroid":
        in_dim += 2  # c_u, c_v

    features_list: list[np.ndarray] = []
    t_list: list[int] = []

    for _, row in df.iterrows():
        detections = _parse_detections(str(row["points"]))
        uv_pairs = [(d[0], d[1]) for d in detections if len(d) >= 2]

        if len(uv_pairs) < min_leds:
            continue

        width = float(row["image_width"])
        height = float(row["image_height"])
        half_w, half_h = width / 2.0, height / 2.0

        uv_norm = np.asarray(
            [((u - half_w) / half_w, (v - half_h) / half_h)
             for u, v in uv_pairs],
            dtype=np.float32,
        )

        # Cap visible count at max_leds (extra detections are ignored;
        # which ones are dropped is encoding-specific).
        if encoding == "absolute":
            # Sort by (u, v) and keep the first max_leds (legacy behaviour).
            order = np.lexsort((uv_norm[:, 1], uv_norm[:, 0]))
            uv_norm = uv_norm[order][:max_leds]
            n_visible = uv_norm.shape[0]

            vec = np.zeros(in_dim, dtype=np.float32)
            for i in range(n_visible):
                base = i * 3
                vec[base]     = uv_norm[i, 0]
                vec[base + 1] = uv_norm[i, 1]
                vec[base + 2] = 1.0
            vec[-1] = float(n_visible)

        else:  # "centroid"
            # Centroid over ALL visible LEDs (not just the kept ones).
            c_u, c_v = float(uv_norm[:, 0].mean()), float(uv_norm[:, 1].mean())
            offsets = uv_norm - np.array([c_u, c_v], dtype=np.float32)

            # Sort visible LEDs CCW by angle from +u axis, starting at 0.
            angles = np.arctan2(offsets[:, 1], offsets[:, 0])
            angles = np.where(angles < 0.0, angles + 2.0 * np.pi, angles)
            order = np.argsort(angles, kind="stable")
            offsets = offsets[order][:max_leds]
            n_visible = offsets.shape[0]

            vec = np.zeros(in_dim, dtype=np.float32)
            vec[0] = c_u
            vec[1] = c_v
            for i in range(n_visible):
                base = 2 + i * 3
                vec[base]     = offsets[i, 0]
                vec[base + 1] = offsets[i, 1]
                vec[base + 2] = 1.0
            vec[-1] = float(n_visible)

        features_list.append(vec)
        t_list.append(int(row["t_ns"]))

    features = np.stack(features_list, axis=0)
    t_ns = np.asarray(t_list, dtype=np.int64)
    return features, t_ns


def _deduplicate_t(
    t_ns: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate on ``t_ns``, keep first occurrence (sorted)."""
    _, unique_idx = np.unique(t_ns, return_index=True)
    unique_idx.sort()
    return t_ns[unique_idx], features[unique_idx]


# ------------------------------------------------------------------ #
#  UVDAR component registry                                           #
# ------------------------------------------------------------------ #

# Each component maps to: (columns_to_read, feature_names)
_UVDAR_COMPONENTS = {
    "position": {
        "columns": ["x", "y", "z"],
        "names":   ["pred_x", "pred_y", "pred_z"],
    },
}


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _exact_lookup(
    ref_t: np.ndarray,
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact timestamp join: for each ``ref_t[i]``, return the matching row
    of ``df[columns]`` if present, else NaN.

    Parameters
    ----------
    ref_t : np.ndarray of int64
        Reference timestamps (already deduplicated).
    df : pd.DataFrame
        Must contain a ``t_ns`` column and all of ``columns``.
    columns : list[str]
        Columns to extract.

    Returns
    -------
    values : np.ndarray, shape (len(ref_t), len(columns)), float32
        NaN-filled for missing timestamps.
    valid : np.ndarray of bool, shape (len(ref_t),)
        True where an exact match was found.
    """
    src = df.drop_duplicates(subset="t_ns", keep="first").set_index("t_ns")
    aligned = src.reindex(ref_t)
    values = aligned[columns].to_numpy(dtype=np.float32, copy=False)
    valid = ~np.isnan(values).any(axis=1)
    return values, valid


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def build_features(
    cfg: dict,
    run_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load and align input features based on ``cfg["features"]``.

    All sources share a common nanosecond timestamp `t`, so this is a
    pure exact-match join — no interpolation.

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
    t_ns : np.ndarray of int64, shape (N,)
        Timestamps in nanoseconds.
    meta : dict
        ``in_dim`` (int), ``feature_names`` (list[str]), optional
        ``uvdar_baseline`` (np.ndarray, NaN where UVDAR missing).
    """
    feat_cfg = cfg["features"]

    blinkers_cfg = feat_cfg.get("blinkers", {})
    uvdar_cfg = feat_cfg.get("uvdar", {})

    use_blinkers = blinkers_cfg.get("enabled", False)
    use_uvdar = uvdar_cfg.get("enabled", False)

    if not use_blinkers and not use_uvdar:
        raise ValueError(
            "At least one of features.blinkers or features.uvdar must be enabled.")

    # --- Always load target ---
    true_rel = load_xyz(os.path.join(run_dir, DATASET_LAYOUT["flier_odom_in_camera_frame"]))

    parts: list[tuple[np.ndarray, list[str]]] = []  # (array, names)
    ref_t: np.ndarray | None = None

    # ── Blinkers ──────────────────────────────────────────────────────
    if use_blinkers:
        max_leds = int(blinkers_cfg.get("max_leds", 4))
        min_leds = int(blinkers_cfg.get("min_leds", 2))
        encoding = str(blinkers_cfg.get("encoding", "absolute"))

        blinker_df = load_blinkers(os.path.join(run_dir, DATASET_LAYOUT["blinkers_right"]))
        blinker_feats, blinker_t = _preprocess_blinkers(
            blinker_df, max_leds, min_leds, encoding,
        )

        ref_t, blinker_feats = _deduplicate_t(blinker_t, blinker_feats)
        parts.append((blinker_feats, _blinker_feature_names(max_leds, encoding)))

        print(f"[build_features] blinkers ({encoding}): {len(blinker_feats)} valid rows, "
              f"{blinker_feats.shape[1]}-D features")

    # ── UVDAR (as input) ──────────────────────────────────────────────
    pred_rel_csv = os.path.join(run_dir, DATASET_LAYOUT["uvdar_estimate_in_camera_frame"])
    pred_rel = load_xyz(pred_rel_csv) if os.path.exists(pred_rel_csv) else None

    if use_uvdar:
        if pred_rel is None:
            raise FileNotFoundError(
                f"uvdar.enabled=true but {pred_rel_csv} not found")
        components = uvdar_cfg.get("components", ["position"])

        if ref_t is None:
            # UVDAR-only → UVDAR timestamps are the reference
            ref_t = np.unique(pred_rel["t_ns"].to_numpy(dtype=np.int64))

        # Exact-join all components on ref_t.
        # Drop ref_t rows where ANY UVDAR component is missing.
        keep_mask = np.ones(len(ref_t), dtype=bool)
        comp_arrays: list[tuple[np.ndarray, list[str]]] = []
        for comp_name in components:
            comp = _UVDAR_COMPONENTS[comp_name]
            vals, valid = _exact_lookup(ref_t, pred_rel, comp["columns"])
            keep_mask &= valid
            comp_arrays.append((vals, comp["names"]))

        # Filter reference + existing parts + UVDAR parts
        ref_t = ref_t[keep_mask]
        parts = [(arr[keep_mask], names) for arr, names in parts]
        for vals, names in comp_arrays:
            parts.append((vals[keep_mask], names))
            print(f"[build_features] uvdar.{names}: {keep_mask.sum()} rows")

    # ── Align target (exact join, drop rows missing target) ───────────
    Y_full, Y_valid = _exact_lookup(ref_t, true_rel, ["x", "y", "z"])
    if not Y_valid.all():
        n_drop = (~Y_valid).sum()
        print(f"[build_features] dropping {n_drop} rows without true pose")
        ref_t = ref_t[Y_valid]
        parts = [(arr[Y_valid], names) for arr, names in parts]
        Y_full = Y_full[Y_valid]
    Y = Y_full

    # ── UVDAR baseline for RMSE comparison ────────────────────────────
    # Always try to load the raw UVDAR position prediction so that
    # downstream code can compute RMSE(UVDAR) vs RMSE(NN).  Rows
    # without an exact UVDAR match get NaN (metrics are NaN-safe).
    uvdar_baseline = None
    if pred_rel is not None:
        baseline_vals, _ = _exact_lookup(ref_t, pred_rel, ["x", "y", "z"])
        uvdar_baseline = baseline_vals
        n_valid = int(np.isfinite(uvdar_baseline).all(axis=1).sum())
        print(f"[build_features] uvdar_baseline: {uvdar_baseline.shape} "
              f"({n_valid} valid rows for comparison)")

    # ── Concatenate all parts ─────────────────────────────────────────
    X = np.concatenate([p[0] for p in parts], axis=1).astype(np.float32)
    feature_names = [n for p in parts for n in p[1]]

    meta = {
        "in_dim": X.shape[1],
        "feature_names": feature_names,
    }
    if uvdar_baseline is not None:
        meta["uvdar_baseline"] = uvdar_baseline

    print(f"[build_features] Final: X={X.shape}, Y={Y.shape}, "
          f"features={feature_names}")

    return X, Y, ref_t, meta
