#!/usr/bin/env python3
"""Dropout-metric explanation plot.

For every 0.5 s bin in a selected time window we colour the background:

  - GREEN ("valid"):   >=1 UVDAR pose estimate in the bin
  - RED   ("dropout"): 0 UVDAR pose estimates in the bin

On top of the coloured background we draw the UVDAR x-coordinate
estimate over time.

Usage:
    python3 -m clean_directory.data.visualize_dropout_metric <csv_dir>
    python3 -m clean_directory.data.visualize_dropout_metric <csv_dir> 30
    python3 -m clean_directory.data.visualize_dropout_metric <csv_dir> 30 60
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Make sibling `plot_style.py` (in repo root) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import apply_style  # noqa: E402

from clean_directory.dataset_layout import (  # noqa: E402
    UVDAR_ESTIMATE_IN_CAMERA_FRAME,
)

apply_style()


# --- config ----------------------------------------------------------------
NS_PER_S    = 10**9
BIN_S       = 0.5
OUTPUT_PDF  = "dropout_metric.pdf"

COLOR_VALID        = "tab:green"
COLOR_DROPOUT      = "tab:red"
ALPHA_BG           = 0.35


# --- CSV loaders -----------------------------------------------------------
def load_estimate_csv(path):
    """Return list[SimpleNamespace(t_ns:int, x:float)]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append(SimpleNamespace(t=int(r["t"]), x=float(r["x"])))
            except (KeyError, ValueError):
                continue
    return out


def load_blinkers_csv(path):
    """Unused; retained as a no-op for backward compatibility."""
    raise NotImplementedError("load_blinkers_csv is no longer used")



# --- helpers ---------------------------------------------------------------
def filter_window(rows, t_lo_ns, t_hi_ns):
    return [p for p in rows if t_lo_ns <= p.t < t_hi_ns]


def classify_bins(estimates, t_lo_ns, t_hi_ns, bin_s=BIN_S):
    """Return list of (bin_start_s, bin_end_s, label) with label in
    {'valid','dropout'}. Times are seconds relative to ``t_lo_ns``."""
    bin_ns = int(bin_s * NS_PER_S)
    n_bins = max(1, (t_hi_ns - t_lo_ns + bin_ns - 1) // bin_ns)

    has_est = [False] * n_bins
    for p in estimates:
        i = (p.t - t_lo_ns) // bin_ns
        if 0 <= i < n_bins:
            has_est[i] = True

    out = []
    for i in range(n_bins):
        b0 = i * bin_s
        b1 = b0 + bin_s
        label = "valid" if has_est[i] else "dropout"
        out.append((b0, b1, label))
    return out


# --- plot ------------------------------------------------------------------
def make_plot(estimates, bins, t_lo_ns, duration_s):
    fig, ax = plt.subplots(figsize=(12, 4.5))

    label_color = {
        "valid":   COLOR_VALID,
        "dropout": COLOR_DROPOUT,
    }
    for b0, b1, lab in bins:
        ax.axvspan(b0, b1, color=label_color[lab], alpha=ALPHA_BG, lw=0)

    # UVDAR x-estimate (relative seconds).
    ts = [(p.t - t_lo_ns) / NS_PER_S for p in estimates]
    xs = [p.x for p in estimates]
    ax.plot(ts, xs, color="orange", linewidth=0,
            marker="o", markersize=3.5, markeredgecolor="black",
            markeredgewidth=0.3, label="UVDAR x estimate")

    # Bin boundaries as vertical dashed grey lines.
    for b0, _, _ in bins:
        ax.axvline(b0, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
    if bins:
        ax.axvline(bins[-1][1], color="grey", linestyle="--",
                   linewidth=0.6, alpha=0.6)

    # Background legend entries (proxies).
    ax.fill_between([], [], [], color=COLOR_VALID,
                    alpha=ALPHA_BG, label="Valid")
    ax.fill_between([], [], [], color=COLOR_DROPOUT,
                    alpha=ALPHA_BG, label="Dropout")

    ax.set_xlim(0, duration_s)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("UVDAR x estimate [m]")
    ax.set_title(f"")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


# --- save button -----------------------------------------------------------
def attach_save_button(fig, out_path):
    save_ax = fig.add_axes([0.82, 0.02, 0.12, 0.05])
    save_button = Button(save_ax, "Save")

    def _save(event):
        save_ax.set_visible(False)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {out_path}")

    save_button.on_clicked(_save)
    # Keep a reference so it isn't garbage-collected.
    fig._save_button = save_button


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("start", nargs="?", type=float, default=0.0,
                    help="start time in seconds (default 0)")
    ap.add_argument("duration", nargs="?", type=float, default=30.0,
                    help="duration in seconds (default 30)")
    args = ap.parse_args()

    d = os.path.abspath(args.csv_dir)
    estimates = load_estimate_csv(os.path.join(d, UVDAR_ESTIMATE_IN_CAMERA_FRAME))

    if not estimates:
        print("No data found.")
        return

    # CSV `t` starts near 0 already; anchor to the earliest sample so the
    # user's start argument is always relative to the first sample.
    t0_ns = min(p.t for p in estimates)

    t_lo_ns = t0_ns + int(args.start    * NS_PER_S)
    t_hi_ns = t0_ns + int((args.start + args.duration) * NS_PER_S)

    estimates = filter_window(estimates, t_lo_ns, t_hi_ns)

    bins = classify_bins(estimates, t_lo_ns, t_hi_ns)

    # Summary.
    n_total = len(bins)
    n_valid = sum(1 for _, _, l in bins if l == "valid")
    n_drop  = sum(1 for _, _, l in bins if l == "dropout")
    dropout_rate = (n_drop / n_total) if n_total > 0 else float("nan")
    print(f"window: [{args.start:.2f}, {args.start + args.duration:.2f}] s  "
          f"bin = {BIN_S:g} s")
    print(f"  total bins:   {n_total}")
    print(f"  valid bins:   {n_valid}")
    print(f"  dropout bins: {n_drop}")
    print(f"  dropout rate: {dropout_rate:.3%}")

    fig, _ = make_plot(estimates, bins, t_lo_ns, args.duration)
    attach_save_button(fig, OUTPUT_PDF)
    plt.show()


if __name__ == "__main__":
    main()
