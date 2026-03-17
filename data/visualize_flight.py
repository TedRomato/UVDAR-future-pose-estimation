#!/usr/bin/env python3
"""
Visualization helpers for UVDAR flight data.

Can be used standalone (``python3 visualize_flight.py <csv_dir>``) or as an
importable module — ``bag_parser_multi`` imports ``plot_all`` from here.
"""
import argparse
import os
import math
import time as _time
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


# == PoseData-based plotting (used by bag_parser_multi) ========================

def _downsample(data, max_points=15000):
    """Uniformly-strided subset with at most max_points entries."""
    if len(data) <= max_points:
        return data
    return data[::len(data) // max_points]


def _plot_axis(ax, poses, field, label, color):
    if not poses:
        return
    ax.plot([p.time for p in poses],
            [getattr(p, field) for p in poses],
            label=label, color=color, linewidth=0.8)


def _plot_with_gaps(ax, poses, field, label, color, max_gap=5.0):
    """Plot data, inserting NaN breaks at time gaps > max_gap seconds."""
    if not poses:
        return
    times, values = [], []
    for i, p in enumerate(poses):
        if i > 0 and p.time - poses[i - 1].time > max_gap:
            times.append(float('nan'))
            values.append(float('nan'))
        times.append(p.time)
        values.append(getattr(p, field))
    ax.plot(times, values, label=label, color=color, linewidth=0.8)


def compute_speed(poses, bin_size: int = 20):
    """Compute speed magnitude (m/s) from a list of PoseData.

    Consecutive samples are first differentiated to get per-step speeds,
    then averaged in non-overlapping bins of ``bin_size`` to produce one
    output sample per bin.  The output timestamp is the mean time of the bin.

    Returns (times, speeds) lists (length ≈ len(poses) / bin_size).
    """
    if len(poses) < 2:
        return [], []

    # Per-step instantaneous speeds
    raw_times, raw_speeds = [], []
    for i in range(1, len(poses)):
        dt = poses[i].time - poses[i - 1].time
        if dt <= 0:
            continue
        dx = poses[i].x - poses[i - 1].x
        dy = poses[i].y - poses[i - 1].y
        dz = poses[i].z - poses[i - 1].z
        raw_speeds.append(math.sqrt(dx * dx + dy * dy + dz * dz) / dt)
        raw_times.append((poses[i].time + poses[i - 1].time) * 0.5)

    # Bin-average
    times, speeds = [], []
    for start in range(0, len(raw_speeds) - bin_size + 1, bin_size):
        chunk_s = raw_speeds[start:start + bin_size]
        chunk_t = raw_times[start:start + bin_size]
        speeds.append(sum(chunk_s) / len(chunk_s))
        times.append(sum(chunk_t) / len(chunk_t))
    return times, speeds


def compute_inter_uav_distance(od1, od2):
    """Compute Euclidean distance between uav1 and uav2 over time.

    Uses nearest-neighbour interpolation: for each odom2 timestamp,
    pick the closest odom1 sample.

    Returns (times, distances) lists.
    """
    if not od1 or not od2:
        return [], []
    times1 = np.array([p.time for p in od1])
    x1 = np.array([p.x for p in od1])
    y1 = np.array([p.y for p in od1])
    z1 = np.array([p.z for p in od1])

    times, dists = [], []
    for p2 in od2:
        idx = np.argmin(np.abs(times1 - p2.time))
        dx = p2.x - x1[idx]
        dy = p2.y - y1[idx]
        dz = p2.z - z1[idx]
        times.append(p2.time)
        dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return times, dists


def _filter_time_window(poses, t_start, t_end):
    """Return only poses with t_start <= time < t_end."""
    return [p for p in poses if t_start <= p.time < t_end]


def plot_all(pred_rel, true_rel, od1, od2, est=None, join_times=None,
             title="Relative Pose & Odom", start_time=0.0, duration=3600.0):
    """Render 5-panel plot: x, y, z, inter-UAV distance, speed.

    All inputs are lists of PoseData objects.
    *start_time* and *duration* (seconds) control the visible time window.
    """
    if not (pred_rel or true_rel or od1 or od2):
        print("No data to plot.")
        return
    est = est or []
    join_times = join_times or []

    # Apply time window
    t_end = start_time + duration
    od1 = _filter_time_window(od1, start_time, t_end)
    od2 = _filter_time_window(od2, start_time, t_end)
    est = _filter_time_window(est, start_time, t_end)
    pred_rel = _filter_time_window(pred_rel, start_time, t_end)
    true_rel = _filter_time_window(true_rel, start_time, t_end)
    join_times = [jt for jt in join_times if start_time <= jt < t_end]

    if not (pred_rel or true_rel or od1 or od2):
        print(f"No data in window [{start_time:.0f}s .. {t_end:.0f}s].")
        return

    print(f"Plotting window [{start_time:.0f}s .. {t_end:.0f}s] "
          f"(od1={len(od1)} od2={len(od2)} est={len(est)} "
          f"pred={len(pred_rel)} true={len(true_rel)})")

    _t0 = _time.monotonic()

    # Downsample for plotting performance
    od1_ds, od2_ds = _downsample(od1), _downsample(od2)
    pred_rel_ds, true_rel_ds = _downsample(pred_rel), _downsample(true_rel)
    est_ds = _downsample(est)
    print(f"  Downsampled in {_time.monotonic() - _t0:.2f}s "
          f"(od1={len(od1_ds)} od2={len(od2_ds)} est={len(est_ds)} "
          f"pred={len(pred_rel_ds)} true={len(true_rel_ds)})")

    _t1 = _time.monotonic()
    speed_t, speed_v = compute_speed(od2)
    print(f"  Computed speed ({len(speed_t)} pts) in {_time.monotonic() - _t1:.2f}s")

    _t1 = _time.monotonic()
    dist_t, dist_v = compute_inter_uav_distance(od1_ds, od2_ds)
    print(f"  Computed inter-UAV distance ({len(dist_t)} pts) in {_time.monotonic() - _t1:.2f}s")

    _t1 = _time.monotonic()
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panels 1-3: x, y, z
    for i, (field, ylabel) in enumerate(zip("xyz", ["X [m]", "Y [m]", "Z [m]"])):
        _plot_axis(axes[i], od1_ds, field, "odom1", "tab:blue")
        _plot_axis(axes[i], od2_ds, field, "odom2", "tab:orange")
        _plot_with_gaps(axes[i], est_ds, field, "raw prediction (local)", "tab:green")
        _plot_axis(axes[i], true_rel_ds, field, "odom2 rel to fcu", "tab:cyan")
        _plot_with_gaps(axes[i], pred_rel_ds, field, "pred rel to fcu", "tab:red")
        axes[i].set_ylabel(ylabel)
        axes[i].legend(loc="best", fontsize="small")
        axes[i].grid(True, alpha=0.3)
    print(f"  Plotted xyz panels in {_time.monotonic() - _t1:.2f}s")

    _t1 = _time.monotonic()
    # Panel 4: inter-UAV distance
    if dist_t:
        dt, dv = _downsample(dist_t), _downsample(dist_v)
        axes[3].plot(dt, dv, color="tab:purple", linewidth=0.8, label="uav1<->uav2 dist")
    axes[3].set_ylabel("Distance [m]")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    # Panel 5: UAV2 speed
    if speed_t:
        st, sv = _downsample(speed_t), _downsample(speed_v)
        axes[4].plot(st, sv, color="tab:orange", linewidth=0.8, label="uav2 speed")
    axes[4].set_ylabel("Speed [m/s]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.3)
    print(f"  Plotted distance & speed panels in {_time.monotonic() - _t1:.2f}s")

    _t1 = _time.monotonic()
    # Bag join markers
    for i, jt in enumerate(join_times):
        for ax in axes:
            ax.axvline(jt, color="k", linestyle="--", linewidth=1.0, alpha=0.7,
                       label="bag join" if i == 0 and ax is axes[0] else None)
    if join_times:
        axes[0].legend(loc="best", fontsize="small")

    fig.suptitle(title)
    fig.tight_layout()
    print(f"  Layout done in {_time.monotonic() - _t1:.2f}s")
    print(f"  Total plot_all: {_time.monotonic() - _t0:.2f}s — showing window...")
    plt.show()


# == CSV → PoseData conversion (standalone mode) ==============================

NEEDED_COLS = ["time", "x", "y", "z",
               "roll_sin", "roll_cos", "pitch_sin", "pitch_cos",
               "yaw_sin", "yaw_cos"]


def _load_pose_list(csv_path):
    """Read a pose CSV into a list of lightweight PoseData-like objects.

    Returns [] if the file doesn't exist or is empty.
    """
    if not os.path.exists(csv_path):
        return []
    import csv
    poses = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                poses.append(SimpleNamespace(
                    time=float(row["time"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    roll_sin=float(row["roll_sin"]),
                    roll_cos=float(row["roll_cos"]),
                    pitch_sin=float(row["pitch_sin"]),
                    pitch_cos=float(row["pitch_cos"]),
                    yaw_sin=float(row["yaw_sin"]),
                    yaw_cos=float(row["yaw_cos"]),
                ))
            except (KeyError, ValueError):
                continue
    return poses


def _load_join_times(run_dir):
    """Read bag-boundary join times from used_rosbags.txt (if present)."""
    txt_path = os.path.join(run_dir, "used_rosbags.txt")
    if not os.path.exists(txt_path):
        return []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Join times:"):
                payload = line[len("Join times:"):].strip()
                if not payload:
                    return []
                return [float(v) for v in payload.split(",")]
    return []


# == Main ======================================================================

def _parse_hhmm(text):
    """Parse 'HH:MM' string to seconds offset."""
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got '{text}'")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def main():
    ap = argparse.ArgumentParser(
        description="Visualise flight CSV data (identical 5-panel plot to bag_parser_multi)."
    )
    ap.add_argument(
        "run_dir",
        help="Folder containing CSV files (odom1.csv, odom2.csv, estimations.csv, "
             "predicted_relative_pose.csv, true_relative_pose.csv).",
    )
    ap.add_argument(
        "--start", default="0:00", metavar="HH:MM",
        help="Start time offset into the data (default: 0:00).",
    )
    ap.add_argument(
        "--duration", type=float, default=3600.0, metavar="SEC",
        help="Duration of the window in seconds (default: 3600 = 1 hour).",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    start_sec = _parse_hhmm(args.start)

    print(f"Loading CSVs from {run_dir} ...")

    print("  Loading odom1.csv ...")
    od1 = _load_pose_list(os.path.join(run_dir, "odom1.csv"))
    print(f"  Loading odom2.csv ...")
    od2 = _load_pose_list(os.path.join(run_dir, "odom2.csv"))
    print(f"  Loading estimations.csv ...")
    est = _load_pose_list(os.path.join(run_dir, "estimations.csv"))
    print(f"  Loading predicted_relative_pose.csv ...")
    pred_rel = _load_pose_list(os.path.join(run_dir, "predicted_relative_pose.csv"))
    print(f"  Loading true_relative_pose.csv ...")
    true_rel = _load_pose_list(os.path.join(run_dir, "true_relative_pose.csv"))

    print(f"Done. odom1={len(od1)}  odom2={len(od2)}  est={len(est)}  "
          f"pred_rel={len(pred_rel)}  true_rel={len(true_rel)}")

    if od2:
        total_dur = od2[-1].time - od2[0].time
        print(f"Total duration: {total_dur:.0f}s ({total_dur/3600:.2f}h)")

    join_times = _load_join_times(run_dir)
    if join_times:
        print(f"Loaded {len(join_times)} bag join times from used_rosbags.txt")

    plot_all(pred_rel, true_rel, od1, od2, est=est,
             join_times=join_times,
             title=os.path.basename(run_dir),
             start_time=float(start_sec), duration=args.duration)


if __name__ == "__main__":
    main()
