#!/usr/bin/env python3
"""Visualize blinker pixel positions from a clean-parser dataset.

Reads `blinkers_right.csv` (columns: t,points,image_height,image_width)
where `points` is a JSON-ish list like `[[u, v, id], ...]`. Plots image
x (u) and y (v) of the blinkers over time. When multiple blinkers are
present at the same timestamp, plots the average of their positions.

Usage:
    python3 visualize_blinkers.py <csv_dir>
    python3 visualize_blinkers.py <csv_dir> --start 0:30 --duration 600
    python3 visualize_blinkers.py <csv_dir> --file blinkers_right.csv
"""

import argparse
import ast
import csv
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt


NS_PER_S = 10**9


def load_blinkers_csv(path):
    """Load blinker CSV, averaging multiple points per row.

    Returns a list of SimpleNamespace(t, x, y, n) where (x, y) is the
    mean pixel position of the `n` blinkers seen at time `t` (ns).
    Rows with no points are skipped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    img_h = img_w = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                pts = ast.literal_eval(r["points"]) if r["points"] else []
            except (ValueError, SyntaxError):
                continue
            if not pts:
                continue
            xs = [float(p[0]) for p in pts if len(p) >= 2]
            ys = [float(p[1]) for p in pts if len(p) >= 2]
            if not xs:
                continue
            try:
                t = int(r["t"])
            except (KeyError, ValueError):
                continue
            out.append(SimpleNamespace(
                t=t, x=sum(xs) / len(xs), y=sum(ys) / len(ys), n=len(xs)))
            if img_h is None:
                try:
                    img_h = int(r["image_height"])
                    img_w = int(r["image_width"])
                except (KeyError, ValueError, TypeError):
                    pass
    return out, img_w, img_h


def downsample(data, max_points=15000):
    if len(data) <= max_points:
        return data
    return data[::len(data) // max_points]


def filter_window(rows, t_lo, t_hi):
    return [p for p in rows if t_lo <= p.t < t_hi]


def plot_with_gaps(ax, rows, field, label, color, max_gap_ns=5 * NS_PER_S):
    if not rows:
        return
    ts, vs = [], []
    for i, p in enumerate(rows):
        if i > 0 and p.t - rows[i - 1].t > max_gap_ns:
            ts.append(float("nan"))
            vs.append(float("nan"))
        ts.append(p.t / NS_PER_S)
        vs.append(getattr(p, field))
    ax.plot(ts, vs, label=label, color=color, linewidth=0.8)


def parse_hhmm(text):
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got '{text}'")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def load_join_times(run_dir):
    p = os.path.join(run_dir, "used_rosbags.txt")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Join times:"):
                payload = line[len("Join times:"):].strip()
                if not payload:
                    return []
                return [int(v) for v in payload.split(",") if v.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--file", default="blinkers_right.csv",
                    help="Blinker CSV filename inside csv_dir.")
    ap.add_argument("--start", default="0:00", metavar="HH:MM")
    ap.add_argument("--duration", type=float, default=3600.0, metavar="SEC")
    args = ap.parse_args()

    d = os.path.abspath(args.csv_dir)
    path = os.path.join(d, args.file)
    print(f"loading: {path}")
    rows, img_w, img_h = load_blinkers_csv(path)
    joins = load_join_times(d)

    t_lo_ns = int(parse_hhmm(args.start) * NS_PER_S)
    t_hi_ns = t_lo_ns + int(args.duration * NS_PER_S)
    rows = filter_window(rows, t_lo_ns, t_hi_ns)
    joins = [j for j in joins if t_lo_ns <= j < t_hi_ns]

    if not rows:
        print(f"No blinker data in window "
              f"[{t_lo_ns / NS_PER_S:.0f}s .. {t_hi_ns / NS_PER_S:.0f}s].")
        return

    rows_ds = downsample(rows)
    multi = sum(1 for p in rows if p.n > 1)
    print(f"rows={len(rows)} (ds {len(rows_ds)})  "
          f"multi-point rows={multi}  image={img_w}x{img_h}  "
          f"joins={len(joins)}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)

    plot_with_gaps(axes[0], rows_ds, "x", "blinker x (mean)", "tab:blue")
    axes[0].set_ylabel("Image x [px]")
    if img_w:
        axes[0].set_ylim(0, img_w)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize="small")

    plot_with_gaps(axes[1], rows_ds, "y", "blinker y (mean)", "tab:red")
    axes[1].set_ylabel("Image y [px]")
    if img_h:
        axes[1].set_ylim(img_h, 0)  # image-y grows downward
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize="small")

    axes[-1].set_xlabel("Time [s]")

    for k, jt in enumerate(joins):
        for ax in axes:
            ax.axvline(jt / NS_PER_S, color="k", linestyle="--",
                       linewidth=1.0, alpha=0.6,
                       label="bag join" if (k == 0 and ax is axes[0]) else None)
    if joins:
        axes[0].legend(loc="best", fontsize="small")

    fig.suptitle(f"{os.path.basename(d)} — {args.file}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
