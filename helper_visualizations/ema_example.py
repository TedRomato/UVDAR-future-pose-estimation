"""Tiny illustrative example of an Exponential Moving Average (EMA)
applied to a noisy continuous signal.

The dark-blue trace is the raw signal (smooth underlying sine wave
plus high-frequency Gaussian noise); the light-blue trace is the
same signal filtered with the time-aware EMA used in
``clean_directory/nn/evaluation/visualize.py``:

    alpha_k = 1 - exp(-dt_k / tau)
    y[k]    = alpha_k * x[k] + (1 - alpha_k) * y[k - 1]

with ``y[0] = x[0]``. With a uniform sample interval this reduces to
a fixed-alpha EMA, but the form is sample-rate independent.

A "Save" button in the bottom-right exports the figure to
``ema_example.{pgf,pdf,svg}`` (the button itself is hidden during
export).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- parameters ------------------------------------------------------
N_SAMPLES   = 600
DURATION_S  = 6.0
NOISE_STD   = 0.35
EMA_TAU_S   = 0.1   # time constant tau [s]; smaller = more responsive
SEED        = 13

COLOR_RAW   = "#0b3d91"   # dark blue
COLOR_EMA   = "#4ea3ff"   # light blue

OUTPUT_BASENAME = "ema_example"


# ---- signal helpers --------------------------------------------------
def make_signal(n: int, duration_s: float, noise_std: float, seed: int):
    """Underlying smooth signal + high-frequency Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration_s, n)
    clean = (
        0.6* np.sin(2.0 * np.pi * 0.5 * t)
        + 0.4 * np.sin(2.0 * np.pi * 1.3 * t + 0.7)
    )
    noisy = clean + rng.normal(0.0, noise_std, size=n)
    return t, noisy


def ema_filter_time_aware(
    x: np.ndarray, t: np.ndarray, tau: float,
) -> np.ndarray:
    """Causal time-aware EMA: alpha_k = 1 - exp(-dt_k / tau).

    Matches ``_ema_filter_time_aware`` in
    ``clean_directory/nn/evaluation/visualize.py`` (minus NaN / reset_gap
    handling, which isn't needed for this synthetic example).
    """
    if tau <= 0.0:
        raise ValueError(f"EMA tau must be > 0, got {tau}")
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for k in range(1, len(x)):
        dt = t[k] - t[k - 1]
        if dt <= 0.0:
            y[k] = y[k - 1]
            continue
        alpha = 1.0 - np.exp(-dt / tau)
        y[k] = alpha * x[k] + (1.0 - alpha) * y[k - 1]
    return y


# ---- plotting --------------------------------------------------------
def make_figure():
    t, raw = make_signal(N_SAMPLES, DURATION_S, NOISE_STD, SEED)
    smoothed = ema_filter_time_aware(raw, t, EMA_TAU_S)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, raw,      color=COLOR_RAW, linewidth=1.0, alpha=0.9,
            label="Raw signal")
    ax.plot(t, smoothed, color=COLOR_EMA, linewidth=2.0,
            label=fr"EMA filtered")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Value")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def attach_save_button(fig):
    save_ax = fig.add_axes([0.82, 0.02, 0.12, 0.05])
    button = Button(save_ax, "Save")

    def _save(_event):
        save_ax.set_visible(False)
        for ext in ("pgf", "pdf", "svg"):
            fig.savefig(f"{OUTPUT_BASENAME}.{ext}",
                        bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {OUTPUT_BASENAME}.{{pgf,pdf,svg}}")

    button.on_clicked(_save)
    return button  # keep reference alive


def main():
    fig = make_figure()
    _button = attach_save_button(fig)  # noqa: F841 (kept alive by ref)
    plt.show()


if __name__ == "__main__":
    main()
