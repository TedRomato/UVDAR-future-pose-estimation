"""
data/alignment.py — Time-alignment utilities for multi-rate sensor data.

Provides:
    interpolate_on_index   – reindex a t_ms-indexed DataFrame onto a target timeline
    forward_fill_with_age  – carry last UVDAR observation forward + compute age
"""

import numpy as np
import pandas as pd


def interpolate_on_index(target_idx: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex *df* (with ``t_ms, x, y, z`` columns) onto *target_idx*
    (integer milliseconds) using linear interpolation, then fill edges.

    Parameters
    ----------
    target_idx : np.ndarray of int64
        Target millisecond timestamps.
    df : pd.DataFrame
        Must have columns ``t_ms, x, y, z``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``t_ms`` with columns ``x, y, z``.
    """
    d = df[["t_ms", "x", "y", "z"]].set_index("t_ms").sort_index()
    out = (
        d.reindex(target_idx)
        .interpolate(method="index")
        .ffill()
        .bfill()
    )
    out.index.name = "t_ms"
    return out[["x", "y", "z"]]


def forward_fill_with_age(
    ref_ms: np.ndarray,
    source_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every timestamp in *ref_ms*, find the **last available** row in
    *source_df* (no future look-ahead) and compute the age in seconds.

    Parameters
    ----------
    ref_ms : np.ndarray of int64
        Reference timeline in milliseconds (sorted, unique).
    source_df : pd.DataFrame
        Must have a ``t_ms`` column.  Remaining columns or *columns*
        are the values to carry forward.
    columns : list[str], optional
        Columns to extract.  Defaults to ``["x", "y", "z"]``.

    Returns
    -------
    filled_values : np.ndarray, shape (M, len(columns)), float32
        Forward-filled values for each *valid* reference timestamp.
    age_sec : np.ndarray, shape (M,), float32
        Seconds elapsed since the source measurement used.
    valid_mask : np.ndarray of bool, shape (N,)
        True for reference entries that have at least one prior source value.
    """
    if columns is None:
        columns = ["x", "y", "z"]

    src = source_df.sort_values("t_ms")
    src_ms = src["t_ms"].values                            # sorted int64
    src_vals = src[columns].values                         # (K, C)

    # searchsorted(side="right") gives the index *after* the last
    # src_ms <= ref_ms, so idx-1 is the last one ≤.
    idx = np.searchsorted(src_ms, ref_ms, side="right") - 1

    # Entries before the first source reading have no valid prior value.
    valid_mask = idx >= 0

    safe_idx = np.clip(idx, 0, len(src_ms) - 1)
    filled_values = src_vals[safe_idx].astype(np.float32)  # (N, C)
    filled_ms = src_ms[safe_idx]                           # (N,)

    age_ms = ref_ms - filled_ms                            # int64
    age_sec = (age_ms.astype(np.float64) / 1000.0).astype(np.float32)

    return filled_values, age_sec, valid_mask
