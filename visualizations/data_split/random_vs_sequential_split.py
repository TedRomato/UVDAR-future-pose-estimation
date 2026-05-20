"""Two-pane top-down (X-Y) visualization of `flier_odom` from `sim_all`.

Demonstrates why a sequential train/test split is preferable to a random
split for sequential odometry data: random selection picks "test" points
that are numerically very close to neighbouring "train" points, which
leaks information and makes a constant prediction look artificially good.

Left  pane: random 20 % of points coloured orange (the "random split").
Right pane: last  20 % of points coloured orange (the "sequential split").

A single constant prediction is plotted as a green '+' on both panes.
The MSE of the orange-point distance to the green cross is printed to
stdout for each pane. Each pane has its own Save button that exports
just that pane to .pgf, .pdf and .svg.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- constants ------------------------------------------------------
NS_PER_S = 10**9

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "clean_directory" / "data" / "sim_all" / "flier_odom.csv"
)

# Time window (seconds since the first odom sample).
T_START = 142.0
T_END   = 150.0
# 115 - 117
TARGET_HZ = 60.0

# Constant "prediction" overlaid as a green cross on both panes.
# Set to None to use the mean of all sampled (blue + orange) XY points
# as the automatic constant predictor. Set to a (x, y) tuple to override.
GREEN_CROSS_XY = None  # e.g. (-14.25, -10.4)

ORANGE_FRAC = 0.20
RANDOM_SEED = 0

OUT_DIR = Path(__file__).resolve().parent
BASENAME = "random_vs_sequential_split"


# ---- data loading ---------------------------------------------------
def load_xy_csv(path):
    """Load t (ns int), x, y from a clean-parser pose CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append(SimpleNamespace(
                    t=int(r["t"]),
                    x=float(r["x"]),
                    y=float(r["y"])))
            except (KeyError, ValueError):
                continue
    return out


def downsample_to_hz(rows, hz):
    """Keep the first sample of every 1/hz second bucket."""
    if not rows or hz <= 0:
        return list(rows)
    bucket_ns = int(NS_PER_S / hz)
    out = []
    last_bucket = None
    for p in rows:
        b = p.t // bucket_ns
        if b != last_bucket:
            out.append(p)
            last_bucket = b
    return out


# ---- pipeline -------------------------------------------------------
rows = load_xy_csv(DATA_PATH)
if not rows:
    raise SystemExit(f"No rows loaded from {DATA_PATH}")

t0 = rows[0].t
t_lo = t0 + int(T_START * NS_PER_S)
t_hi = t0 + int(T_END   * NS_PER_S)
window = [p for p in rows if t_lo <= p.t < t_hi]
if not window:
    raise SystemExit(
        f"No samples in window [{T_START}, {T_END}) s "
        f"(data spans {(rows[-1].t - t0) / NS_PER_S:.1f} s)"
    )

sampled = downsample_to_hz(window, TARGET_HZ)
N = len(sampled)
xy = np.array([(p.x, p.y) for p in sampled], dtype=float)

k = max(1, int(round(N * ORANGE_FRAC)))

rng = np.random.default_rng(RANDOM_SEED)
random_orange_idx = np.sort(rng.choice(N, size=k, replace=False))
sequential_orange_idx = np.arange(N - k, N)

def _cross_for(orange_idx):
    """Return (gx, gy): override constant if set, else mean of blue points."""
    if GREEN_CROSS_XY is not None:
        return GREEN_CROSS_XY
    mask = np.zeros(N, dtype=bool)
    mask[orange_idx] = True
    blue_pts = xy[~mask]
    return float(blue_pts[:, 0].mean()), float(blue_pts[:, 1].mean())


cross_random     = _cross_for(random_orange_idx)
cross_sequential = _cross_for(sequential_orange_idx)


def mse_to_cross(idx, cross):
    pts = xy[idx]
    d2 = (pts[:, 0] - cross[0]) ** 2 + (pts[:, 1] - cross[1]) ** 2
    return float(np.mean(d2))


mse_random     = mse_to_cross(random_orange_idx,     cross_random)
mse_sequential = mse_to_cross(sequential_orange_idx, cross_sequential)

print(f"window:           [{T_START:.2f}, {T_END:.2f}) s")
print(f"sampled @ {TARGET_HZ:.0f} Hz: N = {N} points, orange k = {k}")
print(f"green cross (random split):     ({cross_random[0]:.3f}, {cross_random[1]:.3f})")
print(f"green cross (sequential split): ({cross_sequential[0]:.3f}, {cross_sequential[1]:.3f})")
print(f"MSE (random  20 % orange -> green cross): {mse_random:.6f}")
print(f"MSE (last    20 % orange -> green cross): {mse_sequential:.6f}")


# ---- plotting -------------------------------------------------------
BLUE   = "tab:blue"
ORANGE = "tab:orange"
GREEN  = "tab:green"


def _scatter_split(ax, orange_idx, cross):
    mask = np.zeros(N, dtype=bool)
    mask[orange_idx] = True
    blue_pts = xy[~mask]
    orange_pts = xy[mask]
    ax.scatter(blue_pts[:, 0], blue_pts[:, 1],
               s=8, c=BLUE, edgecolors="none", zorder=2, label="Training")
    ax.scatter(orange_pts[:, 0], orange_pts[:, 1],
               s=8, c=ORANGE, edgecolors="none", zorder=3, label="Validation")
    ax.plot(cross[0], cross[1], marker="+", color=GREEN,
            markersize=14, markeredgewidth=2.5, linestyle="none", zorder=4,
            label="Prediction")
    ax.legend(loc="best", markerscale=1.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")


fig, (ax_random, ax_seq) = plt.subplots(1, 2, figsize=(10, 5))
_scatter_split(ax_random, random_orange_idx,     cross_random)
_scatter_split(ax_seq,    sequential_orange_idx, cross_sequential)

# Equalize axis limits across both panes for fair visual comparison.
xlim = (min(ax_random.get_xlim()[0], ax_seq.get_xlim()[0]),
        max(ax_random.get_xlim()[1], ax_seq.get_xlim()[1]))
ylim = (min(ax_random.get_ylim()[0], ax_seq.get_ylim()[0]),
        max(ax_random.get_ylim()[1], ax_seq.get_ylim()[1]))
for ax in (ax_random, ax_seq):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.tight_layout(rect=(0, 0.08, 1, 1))


# ---- per-pane save buttons -----------------------------------------
_save_buttons = []
_save_axes = []


def _save_pane(ax, suffix):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sa in _save_axes:
        sa.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    base = OUT_DIR / f"{BASENAME}_{suffix}"
    for ext in ("pgf", "pdf", "svg"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches=bbox, pad_inches=0.02)
        print(f"saved {out}")
    for sa in _save_axes:
        sa.set_visible(True)
    fig.canvas.draw_idle()


def _add_pane_save_button(ax, suffix, x_pos):
    save_ax = fig.add_axes([x_pos, 0.02, 0.10, 0.05])
    btn = Button(save_ax, f"Save {suffix}")
    btn.on_clicked(lambda _evt: _save_pane(ax, suffix))
    _save_buttons.append(btn)
    _save_axes.append(save_ax)


_add_pane_save_button(ax_random, "random",     0.18)
_add_pane_save_button(ax_seq,    "sequential", 0.72)

plt.show()
