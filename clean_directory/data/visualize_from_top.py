#!/usr/bin/env python3
"""Top-down (X-Y) view of a clean-parser dataset.

Plots the flier's world-frame path downsampled to ~3 Hz with line
segments between neighbouring points, and the observer as a single
averaged pose with body X/Y axes (red/green) plus a short camera
direction stub rotated -60 deg from observer body X. The world origin
is also marked with its X/Y axes for reference. Z is ignored.

Usage:
    python3 visualize_from_top.py <csv_dir>
    python3 visualize_from_top.py <csv_dir> --start 0:30 --duration 600
    python3 visualize_from_top.py <csv_dir> --hz 5 --camera-deg -60
"""

import argparse
import csv
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt


NS_PER_S = 10**9
DEFAULT_HZ = 3.0
DEFAULT_CAMERA_DEG = 70.0


def load_pose_csv(path):
    """Load t,x,y,qx,qy,qz,qw rows. Z is dropped."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append(SimpleNamespace(
                    t=int(r["t"]),
                    x=float(r["x"]), y=float(r["y"]),
                    qx=float(r["qx"]), qy=float(r["qy"]),
                    qz=float(r["qz"]), qw=float(r["qw"])))
            except (KeyError, ValueError):
                continue
    return out


def filter_window(rows, t_lo, t_hi):
    return [p for p in rows if t_lo <= p.t < t_hi]


def parse_hhmm(text):
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got '{text}'")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


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


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def average_yaw(yaws):
    """Circular mean of yaws (radians)."""
    sx = sum(math.sin(y) for y in yaws)
    sy = sum(math.cos(y) for y in yaws)
    return math.atan2(sx, sy)


def draw_frame(ax, x, y, yaw, length, label=None):
    """Draw a 2D body frame: red = body X, green = body Y."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Body X axis (red).
    ax.plot([x, x + length * cy], [y, y + length * sy],
            color="red", linewidth=2.0)
    # Body Y axis (green) = body X rotated +90 deg.
    ax.plot([x, x - length * sy], [y, y + length * cy],
            color="green", linewidth=2.0)
    if label:
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize="small")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--start", default="0:00", metavar="HH:MM")
    ap.add_argument("--duration", type=float, default=3600.0, metavar="SEC")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ,
                    help="Downsample rate for the flier path.")
    ap.add_argument("--camera-deg", type=float, default=DEFAULT_CAMERA_DEG,
                    help="Magnitude of camera yaw offset from observer body "
                         "X, in degrees. Left camera uses +deg, right uses "
                         "-deg.")
    ap.add_argument("--camera", choices=("left", "right", "both"),
                    default="both",
                    help="Which camera direction stub(s) to draw.")
    args = ap.parse_args()

    d = os.path.abspath(args.csv_dir)
    flier = load_pose_csv(os.path.join(d, "flier_odom.csv"))
    obs   = load_pose_csv(os.path.join(d, "observer_odom.csv"))

    t_lo_ns = int(parse_hhmm(args.start) * NS_PER_S)
    t_hi_ns = t_lo_ns + int(args.duration * NS_PER_S)
    flier = filter_window(flier, t_lo_ns, t_hi_ns)
    obs   = filter_window(obs,   t_lo_ns, t_hi_ns)

    if not flier or not obs:
        print(f"No data in window [{t_lo_ns / NS_PER_S:.0f}s .. "
              f"{t_hi_ns / NS_PER_S:.0f}s]: "
              f"flier={len(flier)} observer={len(obs)}.")
        return

    flier_ds = downsample_to_hz(flier, args.hz)
    obs_ds   = downsample_to_hz(obs,   args.hz)
    print(f"flier={len(flier)} (ds {len(flier_ds)})  "
          f"observer={len(obs)} (ds {len(obs_ds)})  hz={args.hz}")

    obs_x = sum(p.x for p in obs_ds) / len(obs_ds)
    obs_y = sum(p.y for p in obs_ds) / len(obs_ds)
    obs_yaw = average_yaw([yaw_from_quat(p.qx, p.qy, p.qz, p.qw)
                           for p in obs_ds])
    print(f"observer mean: x={obs_x:.3f} y={obs_y:.3f} "
          f"yaw={math.degrees(obs_yaw):.1f} deg")

    fig, ax = plt.subplots(figsize=(9, 9))

    # Flier path.
    fx = [p.x for p in flier_ds]
    fy = [p.y for p in flier_ds]
    ax.plot(fx, fy, color="tab:blue", linewidth=1.0,
            marker="o", markersize=2.5,
            label=f"flier path ({args.hz:g} Hz)")

    # Pick axis lengths from scene size.
    span = max(max(fx) - min(fx), max(fy) - min(fy),
               abs(obs_x), abs(obs_y), 1.0)
    axis_len = max(0.5, 0.05 * span)
    cam_len  = axis_len * 1.4

    # World origin frame.
    draw_frame(ax, 0.0, 0.0, 0.0, axis_len, label="origin")
    ax.plot(0.0, 0.0, marker="x", color="black", markersize=8)

    # Observer averaged pose.
    draw_frame(ax, obs_x, obs_y, obs_yaw, axis_len, label="observer")
    ax.plot(obs_x, obs_y, marker="o", color="black", markersize=5)

    # Camera direction stubs: +deg = left camera, -deg = right camera.
    cams = []
    if args.camera in ("left", "both"):
        cams.append((+args.camera_deg, "tab:purple", "left camera"))
    if args.camera in ("right", "both"):
        cams.append((-args.camera_deg, "tab:orange", "right camera"))
    for deg, color, name in cams:
        cam_yaw = obs_yaw + math.radians(deg)
        ax.plot([obs_x, obs_x + cam_len * math.cos(cam_yaw)],
                [obs_y, obs_y + cam_len * math.sin(cam_yaw)],
                color=color, linewidth=2.0,
                label=f"{name} ({deg:+.0f} deg from body X)")

    # Single legend entries for axis colors.
    ax.plot([], [], color="red",   linewidth=2.0, label="body / world X")
    ax.plot([], [], color="green", linewidth=2.0, label="body / world Y")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="best", fontsize="small")
    ax.set_title(f"{os.path.basename(d)} — top view")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
