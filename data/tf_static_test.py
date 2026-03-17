#!/usr/bin/env python3
"""
Print the T_fixed_local (fixed_origin → local_origin) static transform
from every .bag file under bags/.

This is the transform used in bag_parser_multi.py. Helps diagnose artifacts
where values change across successive rosbags in the same flight.
"""

import os
import sys
import re
import math

try:
    import rosbag
except Exception:
    sys.exit("ERROR: Could not import 'rosbag'. Source /opt/ros/<distro>/setup.bash first.")

BAGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bags")

FRAME_FIXED = "uav1/fixed_origin"
CHILD_LOCAL = "uav1/local_origin"

BAG_FLIGHT_ID_RE = re.compile(
    r"^(flight_\d{4}_\d{2}_\d{2}__\d{2}_\d{2})(?:_\d+)?\.bag$"
)


def quat_to_rpy(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw


bag_files = sorted(f for f in os.listdir(BAGS_DIR) if f.endswith(".bag"))
if not bag_files:
    sys.exit(f"No .bag files found in {BAGS_DIR}")

prev_flight_id = None

for fname in bag_files:
    path = os.path.join(BAGS_DIR, fname)

    # Detect flight boundaries
    m = BAG_FLIGHT_ID_RE.match(fname)
    flight_id = m.group(1) if m else None
    if flight_id != prev_flight_id:
        print(f"\n{'='*70}")
        print(f"Flight: {flight_id or 'unknown'}")
        print(f"{'='*70}")
        prev_flight_id = flight_id

    try:
        static_tfs = []
        with rosbag.Bag(path) as bag:
            for _topic, msg, _t in bag.read_messages(topics=["/tf_static"]):
                if hasattr(msg, "transforms"):
                    static_tfs.extend(msg.transforms)
    except Exception as e:
        print(f"  {fname}: ERROR - {e}")
        continue

    if not static_tfs:
        print(f"  {fname}: no /tf_static messages")
        continue

    # Find the fixed_origin → local_origin transform (forward or reverse)
    found = False
    for tf_msg in static_tfs:
        fid = tf_msg.header.frame_id.lstrip("/")
        cid = tf_msg.child_frame_id.lstrip("/")

        if (fid == FRAME_FIXED and cid == CHILD_LOCAL) or \
           (fid == CHILD_LOCAL and cid == FRAME_FIXED):
            tr = tf_msg.transform.translation
            rot = tf_msg.transform.rotation
            roll, pitch, yaw = quat_to_rpy(rot.x, rot.y, rot.z, rot.w)
            direction = f"{fid} → {cid}"
            print(f"  {fname}:")
            print(f"    direction: {direction}")
            print(f"    translation: x={tr.x:.6f}  y={tr.y:.6f}  z={tr.z:.6f}")
            print(f"    quaternion:  x={rot.x:.6f}  y={rot.y:.6f}  z={rot.z:.6f}  w={rot.w:.6f}")
            print(f"    RPY (deg):   roll={math.degrees(roll):.4f}  pitch={math.degrees(pitch):.4f}  yaw={math.degrees(yaw):.4f}")
            found = True

    if not found:
        # List all static TF frames for debugging
        print(f"  {fname}: {len(static_tfs)} static TFs, but no {FRAME_FIXED} ↔ {CHILD_LOCAL}")
        for tf_msg in static_tfs:
            fid = tf_msg.header.frame_id.lstrip("/")
            cid = tf_msg.child_frame_id.lstrip("/")
            tr = tf_msg.transform.translation
            print(f"    {fid} → {cid}  t=({tr.x:.4f}, {tr.y:.4f}, {tr.z:.4f})")
