"""Visualize linear interpolation between two asynchronous data streams.

Two streams share a common time axis:
- Odometry (z): dense samples at ``--odom-hz`` with smooth flight-like values.
- Blinkers: sparse pulses at ``--blink-hz`` with a per-pulse ``--dropout``
  probability.  Drawn as dots on a constant horizontal "blinker level" line
  because the blinker payload itself has no meaningful y value here.

A representative blinker is chosen near the middle of two odometry samples;
a dashed vertical line connects it to the linearly interpolated odometry
value at its timestamp, where a coloured dot ringed in black marks the
interpolation point.

Run::

    python3 helper_visualizations/interpolation_example.py
    python3 helper_visualizations/interpolation_example.py \\
        --odom-hz 10 --blink-hz 4 --dropout 0.3
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Button


# ============================================================================
# Config
# ============================================================================

DURATION_S         = 6.0
ODOM_FREQ_HZ       = 8.0
BLINK_FREQ_HZ      = 3.0
BLINK_DROPOUT_PROB = 0.3
SEED               = 7

# Use matplotlib default color cycle.
ODOM_COLOR  = "C0"
BLINK_COLOR = "C1"
INTERP_COLOR = "C3"
INTERP_RING  = "black"


# ============================================================================
# Data generation
# ============================================================================

def _generate_odom(duration_s: float, freq_hz: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Smooth flight-like z(t): sum of low-frequency sines + slow drift."""
    dt = 1.0 / freq_hz
    t = np.arange(0.0, duration_s + dt * 0.5, dt)
    n_components = 4
    freqs  = rng.uniform(0.10, 0.55, n_components)
    phases = rng.uniform(0.0, 2 * np.pi, n_components)
    amps   = rng.uniform(0.40, 1.00, n_components)
    z = sum(a * np.sin(2 * np.pi * f * t + p)
            for a, f, p in zip(amps, freqs, phases))
    z = z + 0.35 * t  # gentle ascent
    return t, z


def _generate_blinkers(duration_s: float, freq_hz: float,
                       dropout: float, rng: np.random.Generator) -> np.ndarray:
    dt = 1.0 / freq_hz
    t = np.arange(0.0, duration_s + dt * 0.5, dt)
    keep = rng.uniform(0.0, 1.0, t.size) >= dropout
    return t[keep]


def _pick_target_blinker(odom_t: np.ndarray, blink_t: np.ndarray
                          ) -> tuple[float, int]:
    """Pick the blinker whose timestamp lies closest to the midpoint of an
    odometry segment.  Returns (blinker_time, odom_segment_start_index)."""
    if blink_t.size == 0:
        raise ValueError(
            "no blinkers survived dropout; increase --blink-hz or lower --dropout")
    best: tuple[float, int] | None = None
    best_score = np.inf
    for tb in blink_t:
        idx = int(np.searchsorted(odom_t, tb)) - 1
        if idx < 0 or idx >= odom_t.size - 1:
            continue
        t0, t1 = odom_t[idx], odom_t[idx + 1]
        score = abs(tb - 0.5 * (t0 + t1))
        if score < best_score:
            best_score = score
            best = (float(tb), idx)
    if best is None:
        raise ValueError("no blinker fell inside any odom segment")
    return best


def _linear_interp(odom_t: np.ndarray, odom_z: np.ndarray,
                   idx: int, tb: float) -> float:
    t0, t1 = odom_t[idx], odom_t[idx + 1]
    z0, z1 = odom_z[idx], odom_z[idx + 1]
    alpha = (tb - t0) / (t1 - t0)
    return float(z0 + alpha * (z1 - z0))


# ============================================================================
# Rendering
# ============================================================================

def render(odom_hz: float, blink_hz: float, dropout: float,
           duration_s: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    odom_t, odom_z = _generate_odom(duration_s, odom_hz, rng)
    blink_t = _generate_blinkers(duration_s, blink_hz, dropout, rng)
    tb, idx = _pick_target_blinker(odom_t, blink_t)
    z_interp = _linear_interp(odom_t, odom_z, idx, tb)

    fig, ax = plt.subplots(figsize=(9, 5))

    # --- Odometry (plotted first to establish y limits) -----------------
    odom_line, = ax.plot(odom_t, odom_z, color=ODOM_COLOR, linewidth=2, zorder=2)
    ax.plot(odom_t, odom_z, "o", color=ODOM_COLOR, markersize=6, zorder=3)

    # Lock y scale to odom data so the bottom of the axis is well-defined.
    ax.autoscale(axis="y")
    y_bottom = ax.get_ylim()[0]

    # --- Blinkers (ticks on the bottom axis spine, no data y value) ------
    # Use a blended transform: x in data coords, y in axes fraction.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(blink_t, 0, 0.06, transform=trans,
              color=BLINK_COLOR, linewidth=2, zorder=3, clip_on=False)

    # --- Interpolation visual --------------------------------------------
    ax.plot([tb, tb], [y_bottom, z_interp], "--",
            color=INTERP_COLOR, linewidth=2, zorder=4)
    ax.plot(tb, z_interp, "o",
            color=INTERP_COLOR, markersize=8,
            markeredgecolor=INTERP_RING, markeredgewidth=1,
            zorder=5)

    # --- Cosmetics --------------------------------------------------------
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Odometry z [m]")
    ax.grid(True, linewidth=0.6, alpha=0.7)

    # Build legend manually so the blinker handle shows as a vertical tick.
    blink_proxy = Line2D([0], [0], color=BLINK_COLOR, linewidth=2,
                         marker="|", markersize=12, linestyle="none")
    interp_dot = ax.plot([], [], "o", color=INTERP_COLOR, markersize=8,
                         markeredgecolor=INTERP_RING, markeredgewidth=1)[0]
    ax.legend(handles=[odom_line, blink_proxy, interp_dot],
              labels=["Odometry", "Blinker timestamps", "Interpolated odometry value"],
              loc="upper left")

    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.14)

    # --- Save button (same pattern as sibling helpers) -------------------
    save_stem = "interpolation_example"

    def _save(event):
        save_ax.set_visible(False)
        for ext in ("pgf", "pdf", "svg"):
            fig.savefig(f"{save_stem}.{ext}",
                        bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {save_stem}.{{pgf,pdf,svg}}")

    save_ax = fig.add_axes([0.85, 0.02, 0.12, 0.05])
    save_button = Button(save_ax, "Save")  # noqa: F841  (keep reference alive)
    save_button.on_clicked(_save)

    plt.show()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--odom-hz", type=float, default=ODOM_FREQ_HZ,
                    help=f"Odometry sampling frequency in Hz (default: {ODOM_FREQ_HZ}).")
    ap.add_argument("--blink-hz", type=float, default=BLINK_FREQ_HZ,
                    help=f"Blinker pulse frequency in Hz (default: {BLINK_FREQ_HZ}).")
    ap.add_argument("--dropout", type=float, default=BLINK_DROPOUT_PROB,
                    help=f"Probability of dropping each blinker pulse, in [0, 1] "
                         f"(default: {BLINK_DROPOUT_PROB}).")
    ap.add_argument("--duration", type=float, default=DURATION_S,
                    help=f"Time window in seconds (default: {DURATION_S}).")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"Random seed for reproducibility (default: {SEED}).")
    args = ap.parse_args()

    if not (0.0 <= args.dropout <= 1.0):
        raise SystemExit("--dropout must be in [0, 1]")
    if args.odom_hz <= 0 or args.blink_hz <= 0 or args.duration <= 0:
        raise SystemExit("--odom-hz, --blink-hz, --duration must be positive")

    render(args.odom_hz, args.blink_hz, args.dropout, args.duration, args.seed)


if __name__ == "__main__":
    main()
