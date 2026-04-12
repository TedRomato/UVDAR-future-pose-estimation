"""
evaluation — Visualization, comparison, and result reconstruction tools.
"""

import numpy as np


# ── Shared style constants ────────────────────────────────────────────

GT_COLOR    = "#000000"      # black
OLD_COLOR   = "#D55E00"      # vermilion

GT_LINESTYLE  = "-"
OLD_LINESTYLE = "--"

GT_LINEWIDTH  = 2.0
SYS_LINEWIDTH = 1.6

# Background shading for train / validation regions
TRAIN_BG_COLOR = (0.12, 0.47, 0.71, 0.10)
VAL_BG_COLOR   = (1.00, 0.50, 0.05, 0.10)

# RMSE annotation defaults
RMSE_TEXT_LOC = (0.01, 0.98)
RMSE_TEXT_KW = dict(
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
)

# Distinguishable palette for up to 5 NN overlays
NN_COLORS = [
    "#0072B2",   # blue
    "#009E73",   # teal / bluish green
    "#CC79A7",   # reddish purple
    "#56B4E9",   # sky blue
    "#E69F00",   # amber
]

AXIS_LABELS = ["Relative X", "Relative Y", "Relative Z"]


# ── Plotting helpers ──────────────────────────────────────────────────

def shade_train_val(ax, t_all, val_split):
    """Shade background: blue for training, orange for validation."""
    N = len(t_all)
    n_train = int(round(N * (1.0 - val_split)))
    n_train = max(1, min(N - 1, n_train))
    t_split = t_all[n_train] if n_train < N else t_all[-1]
    ax.axvspan(t_all[0], t_split, color=TRAIN_BG_COLOR, zorder=0)
    ax.axvspan(t_split, t_all[-1], color=VAL_BG_COLOR, zorder=0)


def split_masks(num_samples: int, val_split: float):
    n_train = int(round(num_samples * (1.0 - val_split)))
    n_train = max(1, min(num_samples - 1, n_train))
    train_mask = np.zeros(num_samples, dtype=bool)
    train_mask[:n_train] = True
    return train_mask, ~train_mask


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def improvement_pct(old_rmse: float, new_rmse: float) -> float:
    if abs(old_rmse) <= 1e-12:
        return float("nan")
    return (1.0 - (new_rmse / old_rmse)) * 100.0


def insert_gap_nans(t: np.ndarray, *arrays: np.ndarray,
                    gap_factor: float = 3.0):
    """Return copies of *t* and each array with NaN rows inserted wherever
    the time step exceeds ``gap_factor * median_dt``, so matplotlib breaks
    the line instead of drawing a long diagonal across the gap.

    Parameters
    ----------
    t : 1-D array of timestamps (must be sorted).
    *arrays : arbitrary number of arrays whose first axis matches *t*.
    gap_factor : a gap is declared when ``dt > gap_factor * median_dt``.

    Returns
    -------
    (t_out, *arrays_out) – same shapes but with extra NaN rows at gaps.
    """
    if len(t) < 2:
        return (t,) + arrays

    dt = np.diff(t)
    median_dt = np.median(dt)
    if median_dt <= 0:
        return (t,) + arrays

    gap_indices = np.where(dt > gap_factor * median_dt)[0]
    if len(gap_indices) == 0:
        return (t,) + arrays

    # Build new arrays with NaN rows inserted after each gap index.
    # np.insert treats indices as positions in the *original* array.
    insert_positions = gap_indices + 1
    t_out = np.insert(t.astype(float), insert_positions, np.nan)

    out_arrays = []
    for arr in arrays:
        a = arr.astype(float)
        if a.ndim == 1:
            out_arrays.append(np.insert(a, insert_positions, np.nan))
        else:
            nan_rows = np.full((1, a.shape[1]), np.nan)
            out_arrays.append(
                np.insert(a, insert_positions, nan_rows, axis=0))

    return (t_out,) + tuple(out_arrays)
