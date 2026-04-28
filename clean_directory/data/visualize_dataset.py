#!/usr/bin/env python3
"""Visualize a clean-parser dataset directory.

Plots ground-truth flier (`flier_odom_in_camera_frame`) vs UVDAR estimate
(`uvdar_estimate_in_camera_frame`) — both already in the observer camera
frame — as x/y/z vs time. Bag-join boundaries (from `used_rosbags.txt`)
are drawn as vertical dashed lines.

With ``--extras`` adds two extra panels: inter-UAV distance and flier
speed (computed from `flier_odom_in_camera_frame`).

Usage:
    python3 visualize_dataset.py <csv_dir>
    python3 visualize_dataset.py <csv_dir> --extras
    python3 visualize_dataset.py <csv_dir> --start 0:30 --duration 600
"""

import argparse
import csv
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt


# ============================================================================
# CSV loading
# ============================================================================

def load_pose_csv(path):
    """Load a t,x,y,z,(qx,qy,qz,qw) CSV. `t` stays as int nanoseconds."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append(SimpleNamespace(
                    t=int(r["t"]),
                    x=float(r["x"]), y=float(r["y"]), z=float(r["z"])))
            except (KeyError, ValueError):
                continue
    return out


def load_join_times(run_dir):
    """Read `Join times:` line from used_rosbags.txt as int nanoseconds."""
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


# ============================================================================
# Helpers
# ============================================================================

def downsample(data, max_points=15000):
    if len(data) <= max_points:
        return data
    return data[::len(data) // max_points]


def filter_window(poses, t_lo, t_hi):
    return [p for p in poses if t_lo <= p.t < t_hi]


NS_PER_S = 10**9


def plot_with_gaps(ax, poses, field, label, color, max_gap_ns=5 * NS_PER_S):
    """Plot in seconds on x-axis; gaps detected in ns."""
    if not poses:
        return
    ts, vs = [], []
    for i, p in enumerate(poses):
        if i > 0 and p.t - poses[i - 1].t > max_gap_ns:
            ts.append(float("nan"))
            vs.append(float("nan"))
        ts.append(p.t / NS_PER_S)
        vs.append(getattr(p, field))
    ax.plot(ts, vs, label=label, color=color, linewidth=0.8)


def compute_speed(poses, bin_size=20):
    """Returns (t_seconds, speed_m_per_s); `poses.t` is in ns."""
    if len(poses) < 2:
        return [], []
    raw_t, raw_s = [], []
    for i in range(1, len(poses)):
        dt_ns = poses[i].t - poses[i - 1].t
        if dt_ns <= 0:
            continue
        dx = poses[i].x - poses[i - 1].x
        dy = poses[i].y - poses[i - 1].y
        dz = poses[i].z - poses[i - 1].z
        raw_s.append(math.sqrt(dx * dx + dy * dy + dz * dz) / (dt_ns / NS_PER_S))
        raw_t.append(((poses[i].t + poses[i - 1].t) * 0.5) / NS_PER_S)
    ts, ss = [], []
    for s in range(0, len(raw_s) - bin_size + 1, bin_size):
        chunk_s = raw_s[s:s + bin_size]
        chunk_t = raw_t[s:s + bin_size]
        ss.append(sum(chunk_s) / len(chunk_s))
        ts.append(sum(chunk_t) / len(chunk_t))
    return ts, ss


def compute_distance(poses):
    """||(x,y,z)|| at every sample (camera-frame flier IS already the rel pos).

    Returns (t_seconds, distance_m).
    """
    ts = [p.t / NS_PER_S for p in poses]
    ds = [math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) for p in poses]
    return ts, ds


# ============================================================================
# Main plot
# ============================================================================

def parse_hhmm(text):
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got '{text}'")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--start", default="0:00", metavar="HH:MM")
    ap.add_argument("--duration", type=float, default=3600.0, metavar="SEC")
    ap.add_argument("--extras", action="store_true",
                    help="Also show distance + speed panels.")
    ap.add_argument("--relative", action="store_true",
                    help="Show flier pose in observer camera frame instead of "
                         "absolute world poses.")
    args = ap.parse_args()

    d = os.path.abspath(args.csv_dir)
    if args.relative:
        truth_csv = "flier_odom_in_camera_frame.csv"
        est_csv   = "uvdar_estimate_in_camera_frame.csv"
    else:
        truth_csv = "flier_odom.csv"
        est_csv   = "original_uvdar_estimate.csv"
    print(f"truth: {truth_csv}  est: {est_csv}")
    truth = load_pose_csv(os.path.join(d, truth_csv))
    est   = load_pose_csv(os.path.join(d, est_csv))
    joins = load_join_times(d)

    # CLI window is in seconds; convert to ns to match CSV `t`.
    t_lo_ns = int(parse_hhmm(args.start) * NS_PER_S)
    t_hi_ns = t_lo_ns + int(args.duration * NS_PER_S)
    truth = filter_window(truth, t_lo_ns, t_hi_ns)
    est   = filter_window(est,   t_lo_ns, t_hi_ns)
    joins = [j for j in joins if t_lo_ns <= j < t_hi_ns]

    if not (truth or est):
        print(f"No data in window [{t_lo_ns / NS_PER_S:.0f}s .. {t_hi_ns / NS_PER_S:.0f}s].")
        return

    truth_ds = downsample(truth)
    est_ds   = downsample(est)
    print(f"truth={len(truth)} (ds {len(truth_ds)})  "
          f"est={len(est)} (ds {len(est_ds)})  joins={len(joins)}")

    n_panels = 5 if args.extras else 3
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 3.2 * n_panels), sharex=True)

    for i, (field, ylabel) in enumerate(zip("xyz", ["X [m]", "Y [m]", "Z [m]"])):
        plot_with_gaps(axes[i], truth_ds, field, "ground truth", "tab:blue")
        plot_with_gaps(axes[i], est_ds,   field, "UVDAR estimate", "tab:red")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="best", fontsize="small")

    if args.extras:
        d_t, d_v = compute_distance(truth)
        if d_t:
            d_t = downsample(d_t)
            d_v = downsample(d_v)
            axes[3].plot(d_t, d_v, color="tab:purple", linewidth=0.8,
                         label="observer<->flier distance")
        axes[3].set_ylabel("Distance [m]")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="best")

        s_t, s_v = compute_speed(truth)
        if s_t:
            s_t = downsample(s_t)
            s_v = downsample(s_v)
            axes[4].plot(s_t, s_v, color="tab:orange", linewidth=0.8,
                         label="flier speed")
        axes[4].set_ylabel("Speed [m/s]")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(loc="best")

    axes[-1].set_xlabel("Time [s]")

    for k, jt in enumerate(joins):
        for ax in axes:
            ax.axvline(jt / NS_PER_S, color="k", linestyle="--", linewidth=1.0, alpha=0.6,
                       label="bag join" if (k == 0 and ax is axes[0]) else None)
    if joins:
        axes[0].legend(loc="best", fontsize="small")

    fig.suptitle(os.path.basename(d))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
