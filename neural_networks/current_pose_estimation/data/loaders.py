"""
data/loaders.py — CSV loading functions for pose-estimation data.

Provides:
    load_xyz       – load a time,x,y,z CSV with millisecond indexing
    load_blinkers  – load blinkers_seen_right.csv
"""

import ast

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  XYZ (pose) loader                                                  #
# ------------------------------------------------------------------ #

def load_xyz(path: str) -> pd.DataFrame:
    """
    Load a CSV with ``time,x,y,z`` columns.

    Adds an integer-millisecond column ``t_ms`` for exact time matching
    and deduplicates on it (keeps first occurrence).

    Returns
    -------
    pd.DataFrame
        Sorted by time, with columns ``time, x, y, z, t_ms``.
    """
    df = pd.read_csv(path)[["time", "x", "y", "z"]].dropna().sort_values("time")
    df["t_ms"] = (df["time"] * 1000).round().astype(np.int64)
    df = df.drop_duplicates(subset="t_ms")
    return df


# ------------------------------------------------------------------ #
#  Blinker (LED detection) loader                                     #
# ------------------------------------------------------------------ #

def _parse_detections(raw: str) -> list[list[float]]:
    """Safely parse the stringified list of [u, v, id] detections."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def load_blinkers(csv_path: str) -> pd.DataFrame:
    """
    Read ``blinkers_seen_right.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``time, points, image_height, image_width``.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in ("time", "points", "image_height", "image_width"):
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in {csv_path}")
    return df
