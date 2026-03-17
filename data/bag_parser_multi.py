#!/usr/bin/env python3
"""
Multi-bag ROS Bag Parser — Plot Only
=====================================

Parses one or more ROS bag files and plots odom1, odom2 and UVDAR
estimation graphs (x, y, z vs time).

Usage:
    python3 bag_parser_multi.py --from 2026-02-06 --to 2026-02-06
    python3 bag_parser_multi.py /path/to/bags/*.bag
"""

import sys
import os
import math
import re
import glob
import json
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import rosbag
except Exception:
    sys.stderr.write(
        "ERROR: Could not import 'rosbag'. "
        "Did you source /opt/ros/<distro>/setup.bash?\n"
    )
    sys.exit(2)


# == Configuration =============================================================

TOPIC_OD1 = "/uav1/estimation_manager/odom_main"
TOPIC_OD2 = "/uav2/estimation_manager/odom_main"
UVDAR_CANDIDATES = ["/uav1/uvdar/filteredPoses", "/uav1/uvdar/measuredPoses"]
TOPIC_POINTS_SEEN = "/uav1/uvdar/points_seen_right"
TOPIC_BLINKERS_SEEN = "/uav1/uvdar/blinkers_seen_right"

BAG_FILENAME_RE = re.compile(
    r"^flight_(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})(?:_\d+)?\.bag$"
)
BAG_FLIGHT_ID_RE = re.compile(
    r"^(flight_\d{4}_\d{2}_\d{2}__\d{2}_\d{2})(?:_\d+)?\.bag$"
)

DATE_FORMATS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d_%H:%M",
    "%Y-%m-%d",
]


# == Date helpers ==============================================================

def parse_user_date(text: str, end_of_day: bool = False) -> datetime:
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text.strip(), fmt)
            if end_of_day and "%H" not in fmt:
                dt = dt.replace(hour=23, minute=59)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{text}'. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM'.")


def flight_id_from_bag(filename: str) -> Optional[str]:
    """Extract the flight identity (date+time prefix) from a bag filename.

    E.g. 'flight_2026_02_19__22_20_0.bag' -> 'flight_2026_02_19__22_20'
    Returns None for non-matching filenames.
    """
    m = BAG_FLIGHT_ID_RE.match(os.path.basename(filename))
    return m.group(1) if m else None


def datetime_from_bag_filename(filename: str) -> Optional[datetime]:
    m = BAG_FILENAME_RE.match(os.path.basename(filename))
    if not m:
        return None
    year, month, day, hour, minute = (int(g) for g in m.groups())
    return datetime(year, month, day, hour, minute)


def find_bags_in_range(bags_dir, date_from, date_to):
    matched = []
    for fname in sorted(os.listdir(bags_dir)):
        if not fname.endswith(".bag"):
            continue
        dt = datetime_from_bag_filename(fname)
        if dt is None:
            continue
        if date_from and dt < date_from:
            continue
        if date_to and dt > date_to:
            continue
        matched.append(os.path.join(bags_dir, fname))
    return matched


# == Data classes ==============================================================

@dataclass
class PoseData:
    time: float
    x: float
    y: float
    z: float
    roll_sin: float
    roll_cos: float
    pitch_sin: float
    pitch_cos: float
    yaw_sin: float
    yaw_cos: float


@dataclass
class PointsSeenData:
    time: float
    points: List[Tuple[float, float, float]]  # List of (x, y, value) tuples
    image_height: int
    image_width: int


# == Helpers ===================================================================

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


def msg_to_pose_data(tsec, position, orientation):
    roll, pitch, yaw = quat_to_rpy(
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    return PoseData(
        time=tsec,
        x=position.x, y=position.y, z=position.z,
        roll_sin=math.sin(roll), roll_cos=math.cos(roll),
        pitch_sin=math.sin(pitch), pitch_cos=math.cos(pitch),
        yaw_sin=math.sin(yaw), yaw_cos=math.cos(yaw),
    )


# == SE(3) transform helpers ===================================================

def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3×3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.dot(q, q)
    if n < 1e-12:
        return np.eye(3)
    q *= math.sqrt(2.0 / n)
    outer = np.outer(q, q)
    return np.array([
        [1.0 - outer[1, 1] - outer[2, 2],       outer[0, 1] - outer[2, 3],       outer[0, 2] + outer[1, 3]],
        [      outer[0, 1] + outer[2, 3], 1.0 - outer[0, 0] - outer[2, 2],       outer[1, 2] - outer[0, 3]],
        [      outer[0, 2] - outer[1, 3],       outer[1, 2] + outer[0, 3], 1.0 - outer[0, 0] - outer[1, 1]],
    ])


def make_se3(tx: float, ty: float, tz: float,
             qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Build a 4×4 homogeneous SE(3) matrix from translation + quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def euler_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) -> quaternion (x, y, z, w)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def rotation_matrix_to_quat(R):
    """3×3 rotation matrix -> quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                0.25 * s, (R[1, 0] - R[0, 1]) / s)


def pose_to_se3(p: PoseData) -> np.ndarray:
    """PoseData -> 4×4 SE(3) matrix."""
    roll = math.atan2(p.roll_sin, p.roll_cos)
    pitch = math.atan2(p.pitch_sin, p.pitch_cos)
    yaw = math.atan2(p.yaw_sin, p.yaw_cos)
    qx, qy, qz, qw = euler_to_quat(roll, pitch, yaw)
    return make_se3(p.x, p.y, p.z, qx, qy, qz, qw)


def se3_to_pose_data(time: float, T: np.ndarray) -> PoseData:
    """4×4 SE(3) matrix -> PoseData at given timestamp."""
    x, y, z = T[:3, 3]
    qx, qy, qz, qw = rotation_matrix_to_quat(T[:3, :3])
    roll, pitch, yaw = quat_to_rpy(qx, qy, qz, qw)
    return PoseData(
        time=time, x=float(x), y=float(y), z=float(z),
        roll_sin=math.sin(roll), roll_cos=math.cos(roll),
        pitch_sin=math.sin(pitch), pitch_cos=math.cos(pitch),
        yaw_sin=math.sin(yaw), yaw_cos=math.cos(yaw),
    )


def _tf_msg_to_se3(tf_msg) -> np.ndarray:
    """Convert a geometry_msgs/TransformStamped message to a 4×4 SE(3) matrix."""
    tr = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    return make_se3(tr.x, tr.y, tr.z, rot.x, rot.y, rot.z, rot.w)


def _find_transform(tf_messages, frame_id: str, child_frame_id: str) -> Optional[np.ndarray]:
    """Find the latest transform matching (frame_id, child_frame_id) in a list of
    geometry_msgs/TransformStamped messages.  Returns 4×4 SE(3) or None.
    Also tries the reverse direction (inverted) if the forward one isn't found."""
    for msg in reversed(tf_messages):
        fid = msg.header.frame_id.lstrip("/")
        cid = msg.child_frame_id.lstrip("/")
        if fid == frame_id and cid == child_frame_id:
            return _tf_msg_to_se3(msg)
    # Try reverse direction
    for msg in reversed(tf_messages):
        fid = msg.header.frame_id.lstrip("/")
        cid = msg.child_frame_id.lstrip("/")
        if fid == child_frame_id and cid == frame_id:
            return np.linalg.inv(_tf_msg_to_se3(msg))
    return None


def get_transform_components(bag_path, uav_id=1, T_fixed_local=None):
    """Extract static + dynamic TF components from a bag.

    Returns (T_fixed_local, dynamic_fcu_fixed).

    If T_fixed_local is provided (e.g. cached from the first bag of a flight),
    it is used directly.  Otherwise it is read from /tf_static in this bag.
    """
    prefix = f"uav{uav_id}"
    FRAME_FCU   = f"{prefix}/fcu"
    FRAME_FIXED = f"{prefix}/fixed_origin"
    CHILD_LOCAL = f"{prefix}/local_origin"

    # Only read /tf_static when we still need T_fixed_local
    read_topics = ["/tf"]
    if T_fixed_local is None:
        read_topics.append("/tf_static")

    tf_dynamic, tf_static = [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=read_topics):
            transforms = msg.transforms if hasattr(msg, "transforms") else []
            (tf_static if topic == "/tf_static" else tf_dynamic).extend(transforms)

    print(f"  Found {len(tf_static)} static TFs and {len(tf_dynamic)} dynamic TFs")

    # T_fixed←local: use provided value or extract from /tf_static
    if T_fixed_local is None:
        T_fixed_local = _find_transform(tf_static, FRAME_FIXED, CHILD_LOCAL)
        if T_fixed_local is None:
            raise RuntimeError(
                f"Transform {FRAME_FIXED} → {CHILD_LOCAL} not found in /tf_static. "
                "Make sure the first bag of the flight contains /tf_static."
            )

    # Dynamic transforms: fcu ← fixed_origin
    dynamic_fcu_fixed = []
    for tf_msg in tf_dynamic:
        fid = tf_msg.header.frame_id.lstrip("/")
        cid = tf_msg.child_frame_id.lstrip("/")
        if fid == FRAME_FCU and cid == FRAME_FIXED:
            stamp = tf_msg.header.stamp.to_sec()
            dynamic_fcu_fixed.append((stamp, _tf_msg_to_se3(tf_msg)))

    if not dynamic_fcu_fixed:
        raise RuntimeError(f"No dynamic TF {FRAME_FCU} → {FRAME_FIXED} in /tf")

    dynamic_fcu_fixed.sort(key=lambda x: x[0])
    return T_fixed_local, dynamic_fcu_fixed


def _advance_cursor(timestamp, dynamic_tfs, cursor):
    """Move cursor to closest-in-time entry (forward-only, O(1) amortised)."""
    last = len(dynamic_tfs) - 1
    while cursor < last and abs(dynamic_tfs[cursor + 1][0] - timestamp) <= abs(dynamic_tfs[cursor][0] - timestamp):
        cursor += 1
    return cursor, dynamic_tfs[cursor][1]


def transform_pose_list_to_fcu(poses, T_fixed_local, dynamic_fcu_fixed):
    """Transform poses from local_origin to uav1/fcu frame.

    T(t) = T_fcu←fixed(t) · T_fixed←local · T_pose
    """
    transformed = []
    cursor = 0
    for p in poses:
        cursor, T_fcu_fixed = _advance_cursor(p.time, dynamic_fcu_fixed, cursor)
        T = T_fcu_fixed @ T_fixed_local @ pose_to_se3(p)
        transformed.append(se3_to_pose_data(p.time, T))
    return transformed


def compute_true_relative_pose(odom1, odom2):
    """Pose of uav2 relative to uav1's FCU frame.

    For each odom2 timestamp, finds nearest odom1 and computes:
        T_rel = inv(T_uav1) · T_uav2
    """
    if not odom1 or not odom2:
        return []
    result = []
    cursor = 0
    last = len(odom1) - 1
    for p2 in odom2:
        while cursor < last and abs(odom1[cursor + 1].time - p2.time) <= abs(odom1[cursor].time - p2.time):
            cursor += 1
        T_rel = np.linalg.inv(pose_to_se3(odom1[cursor])) @ pose_to_se3(p2)
        result.append(se3_to_pose_data(p2.time, T_rel))
    return result


# == Bag parsing ===============================================================

def get_uvdar_topic(bag_path):
    try:
        with rosbag.Bag(bag_path) as bp:
            topics = bp.get_type_and_topic_info()[1]
            for cand in UVDAR_CANDIDATES:
                if cand in topics:
                    return cand
    except Exception:
        pass
    return UVDAR_CANDIDATES[0]


def parse_bag(bag_path, T_fixed_local: Optional[np.ndarray] = None):
    """Parse a single bag file.

    Returns (estimations, predicted_relative_pose, true_relative_pose,
             odom1, odom2, points_seen, blinkers_seen, T_fixed_local_used).
    """
    estimations, odom1, odom2 = [], [], []
    points_seen, blinkers_seen = [], []

    uvdar_topic = get_uvdar_topic(bag_path)
    print(f"  Using UVDAR topic: {uvdar_topic}")
    topic_list = [uvdar_topic, TOPIC_OD1, TOPIC_OD2,
                  TOPIC_POINTS_SEEN, TOPIC_BLINKERS_SEEN]

    print("Opening bag...")
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=topic_list):
            tsec = t.to_sec()

            if topic == TOPIC_OD1 and msg._type == "nav_msgs/Odometry":
                odom1.append(msg_to_pose_data(tsec, msg.pose.pose.position, msg.pose.pose.orientation))

            elif topic == TOPIC_OD2 and msg._type == "nav_msgs/Odometry":
                odom2.append(msg_to_pose_data(tsec, msg.pose.pose.position, msg.pose.pose.orientation))

            elif topic == uvdar_topic:
                if msg._type == "mrs_msgs/PoseWithCovarianceArrayStamped":
                    if hasattr(msg, "poses") and len(msg.poses) > 0:
                        p = msg.poses[0].pose
                        estimations.append(msg_to_pose_data(tsec, p.position, p.orientation))
                elif msg._type == "geometry_msgs/PoseArray":
                    if hasattr(msg, "poses") and len(msg.poses) > 0:
                        p = msg.poses[0]
                        estimations.append(msg_to_pose_data(tsec, p.position, p.orientation))

            elif topic in (TOPIC_POINTS_SEEN, TOPIC_BLINKERS_SEEN):
                pts = []
                if hasattr(msg, 'points'):
                    for pt in msg.points:
                        x = pt.x if hasattr(pt, 'x') else 0.0
                        y = pt.y if hasattr(pt, 'y') else 0.0
                        v = pt.value if hasattr(pt, 'value') else 0.0
                        pts.append((x, y, v))
                h = msg.image_height if hasattr(msg, 'image_height') else 0
                w = msg.image_width if hasattr(msg, 'image_width') else 0
                entry = PointsSeenData(time=tsec, points=pts,
                                       image_height=h, image_width=w)
                if topic == TOPIC_POINTS_SEEN:
                    points_seen.append(entry)
                else:
                    blinkers_seen.append(entry)

    # Transform estimations to FCU frame
    predicted_relative_pose = []
    T_fixed_local_used = None
    if estimations:
        try:
            T_fl, dyn_fcu_fixed = get_transform_components(
                bag_path, uav_id=1, T_fixed_local=T_fixed_local)
            T_fixed_local_used = T_fl
            predicted_relative_pose = transform_pose_list_to_fcu(
                estimations, T_fl, dyn_fcu_fixed)
            print(f"  {len(predicted_relative_pose)} estimations -> FCU frame")
        except Exception as e:
            print(f"  Warning: FCU transform failed: {e}")

    true_relative_pose = compute_true_relative_pose(odom1, odom2)

    return (estimations, predicted_relative_pose, true_relative_pose,
            odom1, odom2, points_seen, blinkers_seen, T_fixed_local_used)


# == Time helpers ==============================================================

def offset_poses(poses, offset):
    """Add a time offset to every entry (PoseData or PointsSeenData)."""
    for p in poses:
        p.time += offset


def get_first_time(*pose_lists):
    """Minimum first timestamp across non-empty pose lists."""
    first = None
    for pl in pose_lists:
        if pl:
            t = pl[0].time
            if first is None or t < first:
                first = t
    return first if first is not None else 0.0


def get_last_time(*pose_lists):
    """Maximum last timestamp across pose lists (0.0 if all empty)."""
    last = 0.0
    for pl in pose_lists:
        if pl:
            last = max(last, pl[-1].time)
    return last


def shift_positions(poses, dx, dy, dz):
    """Translate xyz positions of every PoseData by a fixed offset."""
    for p in poses:
        p.x += dx
        p.y += dy
        p.z += dz


# == CSV helpers ===============================================================

def save_points_seen_csv(data_list, filepath):
    """Save a list of PointsSeenData to CSV with points array as JSON string."""
    with open(filepath, 'w') as f:
        f.write("time,points,image_height,image_width\n")
        for d in data_list:
            # Convert points list to JSON array string
            points_json = json.dumps(d.points)
            f.write(f"{d.time:.6f},\"{points_json}\",{d.image_height},{d.image_width}\n")
    total_points = sum(len(d.points) for d in data_list)
    print(f"  Saved {len(data_list)} rows ({total_points} total points) -> {filepath}")


def save_pose_csv(data_list, filepath):
    """Save a list of PoseData to CSV."""
    with open(filepath, 'w') as f:
        f.write("time,x,y,z,roll_sin,roll_cos,pitch_sin,pitch_cos,yaw_sin,yaw_cos\n")
        for d in data_list:
            f.write(f"{d.time:.6f},{d.x:.6f},{d.y:.6f},{d.z:.6f},"
                   f"{d.roll_sin:.6f},{d.roll_cos:.6f},"
                   f"{d.pitch_sin:.6f},{d.pitch_cos:.6f},"
                   f"{d.yaw_sin:.6f},{d.yaw_cos:.6f}\n")
    print(f"  Saved {len(data_list)} rows -> {filepath}")


def save_used_bags_txt(bag_paths, filepath, total_hours: float, join_times=None):
    """Save used rosbag filenames, join times, and total duration in hours."""
    join_times = join_times or []
    with open(filepath, 'w') as f:
        for bag_path in bag_paths:
            f.write(f"{os.path.basename(bag_path)}\n")
        if join_times:
            f.write(f"\nJoin times: {','.join(f'{jt:.4f}' for jt in join_times)}\n")
        f.write(f"\nTotal hours: {total_hours:.2f}\n")
    print(f"  Saved {len(bag_paths)} bag names + {len(join_times)} join times -> {filepath}")


# == Plotting (imported from visualize_flight) =================================

from visualize_flight import plot_all  # noqa: E402


# == Main ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Parse & plot multi-bag ROS data")
    parser.add_argument("--from", dest="date_from", metavar="DATE",
                        help="Start date (YYYY-MM-DD or 'YYYY-MM-DD HH:MM').")
    parser.add_argument("--to", dest="date_to", metavar="DATE",
                        help="End date (YYYY-MM-DD or 'YYYY-MM-DD HH:MM').")
    parser.add_argument("--bags-dir", "-d", default=None,
                        help="Directory containing .bag files (default: ./bags).")
    parser.add_argument("--csv-dir", default=None,
                        help="Directory to save CSVs.")
    parser.add_argument("bags", nargs="*",
                        help="Explicit bag file path(s).")
    args = parser.parse_args()

    # -- Collect bag files --
    bags_dir = os.path.abspath(args.bags_dir) if args.bags_dir else os.path.join(os.getcwd(), "bags")
    bag_files = []

    if args.date_from or args.date_to:
        date_from = parse_user_date(args.date_from) if args.date_from else None
        date_to = parse_user_date(args.date_to, end_of_day=True) if args.date_to else None
        if not os.path.isdir(bags_dir):
            sys.exit(f"ERROR: Bags directory not found: {bags_dir}")
        bag_files = find_bags_in_range(bags_dir, date_from, date_to)
    elif args.bags:
        for pat in args.bags:
            bag_files.extend(glob.glob(pat) if ("*" in pat or "?" in pat) else [pat])
    else:
        if os.path.isdir(bags_dir):
            bag_files = find_bags_in_range(bags_dir, None, None)
        if not bag_files:
            parser.print_help()
            sys.exit(1)

    bag_files = sorted(f for f in bag_files if os.path.isfile(f) and f.endswith(".bag"))
    if not bag_files:
        sys.exit("ERROR: No valid bag files found.")

    print(f"Found {len(bag_files)} bag(s):")
    for bf in bag_files:
        print(f"  {os.path.basename(bf)}")

    # -- Parse & merge all bags --
    all_est, all_pred_rel, all_true_rel = [], [], []
    all_od1, all_od2 = [], []
    all_pts_seen, all_blk_seen = [], []
    join_times = []
    used_bag_files = []
    cached_T = None
    current_flight_id = None

    for i, bag_path in enumerate(bag_files):
        print(f"\n[{i + 1}/{len(bag_files)}] {os.path.basename(bag_path)}")

        # Reset cached transform when a new flight starts
        fid = flight_id_from_bag(bag_path)
        if fid is not None and fid != current_flight_id:
            print(f"  New flight detected: {fid}")
            cached_T = None
            current_flight_id = fid

        try:
            est, pred_rel, true_rel, od1, od2, pts, blk, T_used = parse_bag(
                bag_path, T_fixed_local=cached_T)
            used_bag_files.append(bag_path)
            if T_used is not None:
                cached_T = T_used
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  est={len(est)} pred_rel={len(pred_rel)} true_rel={len(true_rel)} "
              f"od1={len(od1)} od2={len(od2)} pts={len(pts)} blk={len(blk)}")

        if not est and not od1 and not od2:
            continue

        # Zero-base this bag's timestamps
        t0 = get_first_time(est, od1, od2)
        for data in (est, pred_rel, true_rel, od1, od2, pts, blk):
            offset_poses(data, -t0)

        # Shift to continue after previous data
        prev_end = get_last_time(all_est, all_od1, all_od2)
        if all_est or all_od1 or all_od2:
            join_times.append(prev_end)
            for data in (est, pred_rel, true_rel, od1, od2, pts, blk):
                offset_poses(data, prev_end)

        # Align raw prediction reference across bags
        if all_est and est:
            shift_positions(est,
                            all_est[-1].x - est[0].x,
                            all_est[-1].y - est[0].y,
                            all_est[-1].z - est[0].z)

        all_est.extend(est)
        all_pred_rel.extend(pred_rel)
        all_true_rel.extend(true_rel)
        all_od1.extend(od1)
        all_od2.extend(od2)
        all_pts_seen.extend(pts)
        all_blk_seen.extend(blk)

    print(f"\nTotal: est={len(all_est)} pred_rel={len(all_pred_rel)} "
          f"true_rel={len(all_true_rel)} od1={len(all_od1)} od2={len(all_od2)}")
    if all_od2:
        dur = all_od2[-1].time - all_od2[0].time
        print(f"Duration: {dur:.1f}s ({dur / 3600:.2f}h)")

    # -- Save CSVs --
    if args.csv_dir:
        d = os.path.abspath(args.csv_dir)
        os.makedirs(d, exist_ok=True)
        csv_map = {
            "odom1.csv": all_od1,
            "odom2.csv": all_od2,
            "estimations.csv": all_est,
            "predicted_relative_pose.csv": all_pred_rel,
            "true_relative_pose.csv": all_true_rel,
        }
        for name, data in csv_map.items():
            if data:
                save_pose_csv(data, os.path.join(d, name))
        if all_pts_seen:
            save_points_seen_csv(all_pts_seen, os.path.join(d, "points_seen_right.csv"))
        if all_blk_seen:
            save_points_seen_csv(all_blk_seen, os.path.join(d, "blinkers_seen_right.csv"))
        total_hours = (all_od2[-1].time - all_od2[0].time) / 3600.0 if all_od2 else 0.0
        save_used_bags_txt(used_bag_files, os.path.join(d, "used_rosbags.txt"), total_hours,
                          join_times=join_times)

    # -- Plot (first hour from the start) --
    plot_all(all_pred_rel, all_true_rel, all_od1, all_od2,
             est=all_est, join_times=join_times, start_time=0.0)


if __name__ == "__main__":
    main()