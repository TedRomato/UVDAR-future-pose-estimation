"""
data/loaders.py — CSV loading for the clean_directory NN.

The clean_directory CSVs already share a common nanosecond timestamp `t`
column, so loaders just expose it as ``t_ns`` and dedup on it.

Provides:
    load_xyz       — load a t,x,y,z(,...) CSV
    load_blinkers  — load blinkers_right.csv
"""

import ast

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  XYZ (pose) loader                                                  #
# ------------------------------------------------------------------ #

def load_xyz(path: str) -> pd.DataFrame:
    """
    Load a CSV with at least ``t,x,y,z`` columns (extra columns ignored).

    Returns
    -------
    pd.DataFrame
        Sorted by time, with columns ``t_ns, x, y, z``.
    """
    df = pd.read_csv(path)
    for col in ("t", "x", "y", "z"):
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in {path}")
    df = df[["t", "x", "y", "z"]].dropna()
    df["t_ns"] = df["t"].astype(np.int64)
    df = (df[["t_ns", "x", "y", "z"]]
          .sort_values("t_ns")
          .drop_duplicates(subset="t_ns", keep="first")
          .reset_index(drop=True))
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
    Read ``blinkers_right.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``t_ns, points, image_height, image_width``.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in ("t", "points", "image_height", "image_width"):
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in {csv_path}")
    df["t_ns"] = df["t"].astype(np.int64)
    return df[["t_ns", "points", "image_height", "image_width"]]
