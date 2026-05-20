"""Simple ReLU activation visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- data ----------------------------------------------------------
x = np.linspace(-3.0, 3.0, 601)
y = np.maximum(0.0, x)


# ---- figure --------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))

ax.axhline(0.0, color="0.5", linewidth=0.6)
ax.axvline(0.0, color="0.5", linewidth=0.6)

ax.plot(x, y, color="tab:blue", label=r"$\mathrm{ReLU}(x)=\max(0, x)$")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathrm{ReLU}(x)$")
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-0.5, 3.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

fig.tight_layout()


# ---- save button ---------------------------------------------------
SAVE_BASENAME = str(Path(__file__).resolve().parent.parent / "relu")

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
