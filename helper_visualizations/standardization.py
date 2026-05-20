"""Visualize z-score standardization on a real feature distribution.

Source: ``z`` column of an odometry CSV from a real flight. Both the
original and the standardized histograms are drawn on the same axis;
the standardized histogram uses finer bins so its narrow shape stays
readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- data ----------------------------------------------------------
CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "clean_directory" / "data" / "sim_all" / "flier_odom.csv"
)
FEATURE_LABEL = r"$z$ [m]"

df = pd.read_csv(CSV_PATH, usecols=["z"])
z = df["z"].to_numpy()
z = z[z >= 4.0]

mu = float(np.mean(z))
sigma = float(np.std(z))
z_std = (z - mu) / sigma


# ---- colors --------------------------------------------------------
ORIG_COLOR = "tab:blue"
STD_COLOR  = "tab:orange"
MEAN_COLOR = "tab:red"

N_BINS_ORIG = 60
# Use a bin width matched to the standardized spread, so its bins are
# proportionally finer on the shared axis.
N_BINS_STD = 60


# ---- figure --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.6))

orig_lo, orig_hi = float(np.min(z)), float(np.max(z))
pad_o = 0.05 * (orig_hi - orig_lo)
bins_orig = np.linspace(orig_lo - pad_o, orig_hi + pad_o, N_BINS_ORIG)

std_lo, std_hi = float(np.min(z_std)), float(np.max(z_std))
pad_s = 0.15 * (std_hi - std_lo)
bins_std = np.linspace(std_lo - pad_s, std_hi + pad_s, N_BINS_STD)

ax.hist(
    z, bins=bins_orig, color=ORIG_COLOR, alpha=0.75,
    edgecolor="white", linewidth=0.4, label=f"original distribution",
)
ax.hist(
    z_std, bins=bins_std, color=STD_COLOR, alpha=0.85,
    edgecolor="white", linewidth=0.4,
    label=r"standardized distribution",
)

# mean and +- 1 sigma reference lines for both distributions
ax.axvline(mu, color=MEAN_COLOR, linestyle="-", linewidth=1.0,
           label=fr"original mean")
ax.axvline(mu - sigma, color="0.4", linestyle="-", linewidth=1.0,
           label=fr"original standard deviation from mean")
ax.axvline(mu + sigma, color="0.4", linestyle="-", linewidth=1.0)

ax.axvline(0.0, color=MEAN_COLOR, linestyle="--", linewidth=1.0,
           label=r"standardized mean")
ax.axvline(-1.0, color="0.4", linestyle="--", linewidth=1.0,
           label=r"standard deviation from mean after standardization")
ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1.0)

# # arrow showing shift + scale from original cluster to standardized cluster
# y_arrow = ax.get_ylim()[1] * 0.85
# ax.add_patch(FancyArrowPatch(
#     (mu, y_arrow), (0.0, y_arrow),
#     arrowstyle="->", mutation_scale=14, color=MEAN_COLOR, linewidth=1.4,
# ))
# ax.text(
#     (mu + 0.0) / 2.0, y_arrow * 1.04,
#     fr"shift by $-\mu$, scale by $1/\sigma$ ($\sigma={sigma:.2f}$)",
#     color=MEAN_COLOR, ha="center", va="bottom",
# )

ax.set_xlabel("value")
ax.set_ylabel("count")
ax.legend(loc="upper right")

fig.tight_layout()

# ---- save button ---------------------------------------------------
SAVE_BASENAME = str(Path(__file__).resolve().parent.parent / "standardization")

save_ax = fig.add_axes([0.88, 0.02, 0.10, 0.05])
_button = Button(save_ax, "Save")

def _save(event):
    save_ax.set_visible(False)
    fig.savefig(f"{SAVE_BASENAME}.pgf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{SAVE_BASENAME}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{SAVE_BASENAME}.svg", bbox_inches="tight", pad_inches=0.02)
    save_ax.set_visible(True)
    fig.canvas.draw_idle()
    print(f"Saved {SAVE_BASENAME}.{{pgf,pdf,svg}}")

_button.on_clicked(_save)
fig._save_button = _button
fig._save_ax = save_ax

if __name__ == "__main__":
    plt.show()
