#!/usr/bin/env python3
"""
Real-world Paired-bag ROS Bag Parser
====================================

Parses paired real-world rosbags (one per UAV) listed in
``ok_flights.txt``.  Each pair consists of:

    - flier bag    (default UAV: uav14)  — provides flier odom
    - observer bag (default UAV: uav9)   — provides observer odom,
                                           UVDAR poses + image points,
                                           and the TF tree

Reuses the SE(3) / TF / CSV machinery from :mod:`bag_parser` so the
output schema is identical to the simulated dataset (``odom1.csv`` is
the observer / UVDAR-equipped UAV, ``odom2.csv`` is the flier).

Pairs are concatenated in the order they appear in ``ok_flights.txt``,
with a configurable time buffer (default 5 s) inserted between pairs
so that downstream consumers can detect the gaps.

Example usage
-------------

    # Use defaults (data/real_world_data/{bags,ok_flights.txt,csv_data})
    python3 bag_parser_real_world.py

    # 10-second gap between pairs, custom CSV directory
    python3 bag_parser_real_world.py --buffer 10 \\
        --csv-dir data/real_world_data/csv_data

    # Custom dataset location
    python3 bag_parser_real_world.py \\
        --bags-dir /path/to/real_world_data/bags \\
        --ok-flights /path/to/real_world_data/ok_flights.txt \\
        --csv-dir   /path/to/real_world_data/csv_data

    # Skip the matplotlib plot at the end
    python3 bag_parser_real_world.py --no-plot
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Optional, Tuple

try:
    import rosbag  # noqa: F401  (sanity-check the env early)
except Exception:
    sys.stderr.write(
        "ERROR: Could not import 'rosbag'. "
        "Did you source /opt/ros/<distro>/setup.bash?\n"
    )
    sys.exit(2)

# Reuse everything we can from the simulated-data parser.
from bag_parser import (
    PoseData,
    PointsSeenData,
    msg_to_pose_data,
    get_transform_components,
    transform_pose_list_to_fcu,
    compute_true_relative_pose,
    offset_poses,
    get_first_time,
    get_last_time,
    save_pose_csv,
    save_points_seen_csv,
    save_used_bags_txt,
    _tf_msg_to_se3,
    se3_to_pose_data,
    make_se3,
    euler_to_quat,
)


# == Configuration =============================================================

OBSERVER_UAV = "uav9"
FLIER_UAV = "uav14"

OBS_TOPIC_ODOM   = f"/{OBSERVER_UAV}/estimation_manager/odom_main"
FLIER_TOPIC_ODOM = f"/{FLIER_UAV}/estimation_manager/odom_main"

UVDAR_CANDIDATES = [
    f"/{OBSERVER_UAV}/uvdar/filteredPoses",
    f"/{OBSERVER_UAV}/uvdar/measuredPoses",
]
TOPIC_POINTS_SEEN   = f"/{OBSERVER_UAV}/uvdar/points_seen_right"
TOPIC_BLINKERS_SEEN = f"/{OBSERVER_UAV}/uvdar/blinkers_seen_right"


# == ok_flights.txt parsing ====================================================

def parse_ok_flights(path: str) -> List[Tuple[str, str, str]]:
    """Parse ``ok_flights.txt``.

    Format (header line ignored)::

        Flier, Observer
        <flier>.bag - <observer>.bag - <free-form note>

    Returns a list of ``(flier_basename, observer_basename, note)`` tuples
    in file order (i.e. chronological).
    """
    pairs: List[Tuple[str, str, str]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("flier"):
                continue
            parts = [p.strip() for p in line.split(" - ")]
            if len(parts) < 2:
                print(f"  Skipping malformed line: {line!r}")
                continue
            flier, observer = parts[0], parts[1]
            note = parts[2] if len(parts) >= 3 else ""
            if not flier.endswith(".bag") or not observer.endswith(".bag"):
                print(f"  Skipping (no .bag suffix): {line!r}")
                continue
            pairs.append((flier, observer, note))
    return pairs


# == Per-bag readers ===========================================================

def _pick_uvdar_topic(bag_path: str) -> str:
    """Pick the first UVDAR topic that actually exists in the bag."""
    import rosbag as _rb
    try:
        with _rb.Bag(bag_path) as bp:
            present = bp.get_type_and_topic_info()[1]
            for cand in UVDAR_CANDIDATES:
                if cand in present:
                    return cand
    except Exception:
        pass
    return UVDAR_CANDIDATES[0]


def _stamp_ns(msg) -> Optional[int]:
    """Return msg's stamp as integer nanoseconds, or None if not present."""
    hdr = getattr(msg, "header", None)
    stamp = getattr(hdr, "stamp", None) if hdr is not None else getattr(msg, "stamp", None)
    if stamp is None:
        return None
    return stamp.to_nsec()


def _first_odom_ns(bag_path: str, topic: str) -> Optional[int]:
    """Return the integer-nanosecond ``header.stamp`` of the first odom message."""
    import rosbag as _rb
    with _rb.Bag(bag_path) as bag:
        for _topic, msg, _t in bag.read_messages(topics=[topic]):
            if msg._type == "nav_msgs/Odometry":
                return msg.header.stamp.to_nsec()
    return None


def _last_odom_ns(bag_path: str, topic: str) -> Optional[int]:
    """Return the integer-nanosecond ``header.stamp`` of the last odom message."""
    import rosbag as _rb
    last_ns: Optional[int] = None
    with _rb.Bag(bag_path) as bag:
        for _topic, msg, _t in bag.read_messages(topics=[topic]):
            if msg._type == "nav_msgs/Odometry":
                last_ns = msg.header.stamp.to_nsec()
    return last_ns


def _ns_to_rel_sec(stamp_ns: int, base_ns: int) -> float:
    """Subtract base in integer ns first, then convert — keeps full precision."""
    return (stamp_ns - base_ns) * 1e-9


def _read_fcu_from_anchor_tfs(bag_path: str, uav: str, anchor: str,
                              base_ns: int) -> List[Tuple[float, np.ndarray]]:
    """Collect dynamic TFs and return ``T_fcu<-anchor`` (REP-105 convention).

    Both UAVs publish ``parent=<uav>/fcu, child=<uav>/<anchor>`` dynamically;
    per REP-105 that matrix maps ``p_anchor -> p_fcu``, i.e. it IS
    ``T_fcu<-anchor`` directly. The reverse direction is inverted on the fly.

    Returns a time-sorted list of ``(rel_sec, T_fcu_from_anchor)``; ``rel_sec``
    is rebased to the pair's ``base_ns``.
    """
    import rosbag as _rb
    parent = f"{uav}/fcu"
    child  = f"{uav}/{anchor}"
    tfs: List[Tuple[float, np.ndarray]] = []
    with _rb.Bag(bag_path) as bag:
        for _topic, msg, _t in bag.read_messages(topics=["/tf"]):
            for tf in getattr(msg, "transforms", []):
                fid = tf.header.frame_id.lstrip("/")
                cid = tf.child_frame_id.lstrip("/")
                if fid == parent and cid == child:
                    T_fcu_from_anchor = _tf_msg_to_se3(tf)
                elif fid == child and cid == parent:
                    T_fcu_from_anchor = np.linalg.inv(_tf_msg_to_se3(tf))
                else:
                    continue
                stamp_ns = tf.header.stamp.to_nsec()
                tfs.append((_ns_to_rel_sec(stamp_ns, base_ns),
                            T_fcu_from_anchor))
    tfs.sort(key=lambda x: x[0])
    return tfs


def _nearest_tf(timestamp: float, tfs: List[Tuple[float, np.ndarray]],
                cursor: int) -> Tuple[int, np.ndarray]:
    """Forward-only nearest-neighbour lookup in a time-sorted TF list."""
    last = len(tfs) - 1
    while (cursor < last
           and abs(tfs[cursor + 1][0] - timestamp)
               <= abs(tfs[cursor][0] - timestamp)):
        cursor += 1
    return cursor, tfs[cursor][1]


def _compute_true_relative_pose_via_anchor(
        flier_tfs: List[Tuple[float, np.ndarray]],
        observer_tfs: List[Tuple[float, np.ndarray]],
        yaw_correction_deg: float = 0.0) -> List[PoseData]:
    """Pose of flier's FCU in observer's FCU frame, anchored through a shared
    parent frame (e.g. ``gps_baro_origin``).

    Each TF entry is ``T_fcu<-anchor`` (REP-105: parent=fcu, child=anchor):

        T_obs_fcu<-fl_fcu  =  R_yaw  @  T_obs_fcu<-anchor  @  T_anchor<-fl_fcu
                           =  R_yaw  @  T_obs              @  inv(T_fl)

    The optional ``yaw_correction_deg`` is applied on the LEFT (in observer
    FCU); use it to compensate for a constant heading offset between the
    two UAVs' anchor frames (their headings are initialised independently).

    For each flier TF timestamp, the observer TF closest in time is used.
    """
    if not flier_tfs or not observer_tfs:
        return []
    if abs(yaw_correction_deg) > 1e-9:
        qx, qy, qz, qw = euler_to_quat(
            0.0, 0.0, np.deg2rad(yaw_correction_deg))
        R_yaw = make_se3(0, 0, 0, qx, qy, qz, qw)
    else:
        R_yaw = np.eye(4)
    out: List[PoseData] = []
    cursor = 0
    for t_fl, T_fl_from_anchor in flier_tfs:
        cursor, T_obs_from_anchor = _nearest_tf(t_fl, observer_tfs, cursor)
        T_rel = R_yaw @ T_obs_from_anchor @ np.linalg.inv(T_fl_from_anchor)
        out.append(se3_to_pose_data(t_fl, T_rel))
    return out


def _read_observer_bag(bag_path: str, base_ns: int):
    """Read odom + UVDAR + points/blinkers from the observer bag.

    All timestamps are returned as ``(stamp_ns - base_ns) * 1e-9`` seconds,
    where ``stamp_ns`` is the publisher ``header.stamp`` in integer ns and
    ``base_ns`` is the common pair base. Subtracting in integer ns first
    avoids the ~200 ns float-precision loss of subtracting two ~1.7e9 doubles.
    Messages with ``stamp_ns < base_ns`` are dropped.
    """
    import rosbag as _rb
    odom_obs: List[PoseData] = []
    estimations: List[PoseData] = []
    points_seen: List[PointsSeenData] = []
    blinkers_seen: List[PointsSeenData] = []

    uvdar_topic = _pick_uvdar_topic(bag_path)
    print(f"  Observer UVDAR topic: {uvdar_topic}")

    topic_list = [
        OBS_TOPIC_ODOM, uvdar_topic,
        TOPIC_POINTS_SEEN, TOPIC_BLINKERS_SEEN,
    ]

    uvdar_frame_warned = False

    with _rb.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=topic_list):
            stamp_ns = _stamp_ns(msg)
            if stamp_ns is None or stamp_ns < base_ns:
                continue
            tsec = _ns_to_rel_sec(stamp_ns, base_ns)

            if topic == OBS_TOPIC_ODOM and msg._type == "nav_msgs/Odometry":
                odom_obs.append(msg_to_pose_data(
                    tsec, msg.pose.pose.position, msg.pose.pose.orientation))

            elif topic == uvdar_topic:
                expected_frame = f"{OBSERVER_UAV}/local_origin"
                fid = getattr(getattr(msg, "header", None), "frame_id", "")
                if (not uvdar_frame_warned and fid
                        and fid.lstrip("/") != expected_frame):
                    print(f"  Warning: UVDAR frame_id={fid!r} "
                          f"(expected {expected_frame!r})")
                    uvdar_frame_warned = True
                if msg._type == "mrs_msgs/PoseWithCovarianceArrayStamped":
                    if hasattr(msg, "poses") and len(msg.poses) > 0:
                        p = msg.poses[0].pose
                        estimations.append(msg_to_pose_data(
                            tsec, p.position, p.orientation))
                elif msg._type == "geometry_msgs/PoseArray":
                    if hasattr(msg, "poses") and len(msg.poses) > 0:
                        p = msg.poses[0]
                        estimations.append(msg_to_pose_data(
                            tsec, p.position, p.orientation))

            elif topic in (TOPIC_POINTS_SEEN, TOPIC_BLINKERS_SEEN):
                pts = []
                if hasattr(msg, "points"):
                    for pt in msg.points:
                        x = pt.x if hasattr(pt, "x") else 0.0
                        y = pt.y if hasattr(pt, "y") else 0.0
                        v = pt.value if hasattr(pt, "value") else 0.0
                        pts.append((x, y, v))
                h = msg.image_height if hasattr(msg, "image_height") else 0
                w = msg.image_width  if hasattr(msg, "image_width")  else 0
                entry = PointsSeenData(time=tsec, points=pts,
                                       image_height=h, image_width=w)
                if topic == TOPIC_POINTS_SEEN:
                    points_seen.append(entry)
                else:
                    blinkers_seen.append(entry)

    return odom_obs, estimations, points_seen, blinkers_seen


def _read_flier_bag(bag_path: str, base_ns: int) -> List[PoseData]:
    """Read flier odom; timestamps as (stamp_ns - base_ns) * 1e-9 seconds."""
    import rosbag as _rb
    odom_flier: List[PoseData] = []
    with _rb.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=[FLIER_TOPIC_ODOM]):
            if topic == FLIER_TOPIC_ODOM and msg._type == "nav_msgs/Odometry":
                stamp_ns = msg.header.stamp.to_nsec()
                if stamp_ns < base_ns:
                    continue
                odom_flier.append(msg_to_pose_data(
                    _ns_to_rel_sec(stamp_ns, base_ns),
                    msg.pose.pose.position, msg.pose.pose.orientation))
    return odom_flier


# == Pair parsing ==============================================================

def parse_real_world_pair(flier_bag: str, observer_bag: str):
    """Parse one (flier, observer) bag pair.

    Returns a tuple matching :func:`bag_parser.parse_bag` shape::

        (estimations, predicted_relative_pose, true_relative_pose,
         odom1, odom2, points_seen, blinkers_seen, T_fixed_local_used)

    where ``odom1`` is the observer (UVDAR-equipped, ego) and ``odom2``
    is the flier. Times are seconds relative to the later of the two
    bags' first odom stamp (computed in integer ns for full precision).
    """
    # Pick a common ns base = later of the two odom starts (= overlap start).
    obs_first_ns = _first_odom_ns(observer_bag, OBS_TOPIC_ODOM)
    fl_first_ns  = _first_odom_ns(flier_bag,    FLIER_TOPIC_ODOM)
    if obs_first_ns is None or fl_first_ns is None:
        raise RuntimeError("missing odom in one of the bags")
    base_ns = max(obs_first_ns, fl_first_ns)
    offset_ns = fl_first_ns - obs_first_ns
    print(f"  Bag start offset (flier - observer): "
          f"{offset_ns * 1e-9:+.6f}s ({offset_ns:+d} ns)")

    # Sanity-check the overlap window using last odom stamps in ns.
    obs_last_ns = _last_odom_ns(observer_bag, OBS_TOPIC_ODOM)
    fl_last_ns  = _last_odom_ns(flier_bag,    FLIER_TOPIC_ODOM)
    if obs_last_ns is not None and fl_last_ns is not None:
        if fl_first_ns > obs_last_ns or obs_first_ns > fl_last_ns:
            print(f"  WARNING: observer and flier wall-clock windows do not overlap")

    print(f"  Reading observer: {os.path.basename(observer_bag)}")
    odom1, estimations, points_seen, blinkers_seen = _read_observer_bag(
        observer_bag, base_ns)

    print(f"  Reading flier:    {os.path.basename(flier_bag)}")
    odom2 = _read_flier_bag(flier_bag, base_ns)

    # Transform estimations from observer's local_origin to observer's fcu frame.
    predicted_relative_pose: List[PoseData] = []
    T_fixed_local_used: Optional[np.ndarray] = None
    observer_uav_id = int(OBSERVER_UAV.replace("uav", ""))
    if estimations:
        try:
            T_fl, dyn_fcu_fixed = get_transform_components(
                observer_bag, uav_id=observer_uav_id, T_fixed_local=None)
            T_fixed_local_used = T_fl
            # dyn_fcu_fixed timestamps are absolute seconds; rebase to match.
            base_sec = base_ns * 1e-9
            dyn_rebased = [(stamp - base_sec, T) for stamp, T in dyn_fcu_fixed]
            predicted_relative_pose = transform_pose_list_to_fcu(
                estimations, T_fl, dyn_rebased)
            print(f"  {len(predicted_relative_pose)} estimations -> {OBSERVER_UAV}/fcu frame")
        except Exception as e:
            print(f"  Warning: FCU transform failed: {e}")

    # True relative pose: anchor BOTH UAVs in a shared parent frame, then
    # express the flier in observer's FCU. The frame chosen below was
    # determined by debug_relative_frames.py to give the best alignment
    # with the UVDAR prediction; per-UAV `gps_baro_origin` works here
    # because for this dataset both UAVs share an effectively common GPS+
    # barometric anchor (a small constant yaw offset is corrected below).
    SHARED_ANCHOR = "gps_baro_origin"
    YAW_CORRECTION_DEG = -120.0  # tweak after running debug script
    obs_tfs = _read_fcu_from_anchor_tfs(observer_bag, OBSERVER_UAV,
                                        SHARED_ANCHOR, base_ns)
    fl_tfs  = _read_fcu_from_anchor_tfs(flier_bag,    FLIER_UAV,
                                        SHARED_ANCHOR, base_ns)
    print(f"  fcu->{SHARED_ANCHOR} TFs: observer={len(obs_tfs)} "
          f"flier={len(fl_tfs)}  yaw_correction={YAW_CORRECTION_DEG:+.1f} deg")
    if not obs_tfs or not fl_tfs:
        print(f"  Warning: missing fcu->{SHARED_ANCHOR} TF on one side; "
              "falling back to odom_main inv(T1)*T2 (frames may not match)")
        true_relative_pose = compute_true_relative_pose(odom1, odom2)
    else:
        true_relative_pose = _compute_true_relative_pose_via_anchor(
            fl_tfs, obs_tfs, yaw_correction_deg=YAW_CORRECTION_DEG)
        print(f"  {len(true_relative_pose)} true relative poses via {SHARED_ANCHOR}")

    return (estimations, predicted_relative_pose, true_relative_pose,
            odom1, odom2, points_seen, blinkers_seen, T_fixed_local_used)


# == Main ======================================================================

def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "real_world_data")
    return {
        "bags_dir":   os.path.join(base, "bags"),
        "ok_flights": os.path.join(base, "ok_flights.txt"),
        "csv_dir":    os.path.join(base, "csv_data"),
    }


def main():
    defaults = _default_paths()

    parser = argparse.ArgumentParser(
        description="Parse paired real-world rosbags into the dataset CSV schema.")
    parser.add_argument("--bags-dir", default=defaults["bags_dir"],
                        help="Directory containing 'flier/' and 'observer/' subdirs "
                             f"(default: {defaults['bags_dir']}).")
    parser.add_argument("--ok-flights", default=defaults["ok_flights"],
                        help=f"Path to ok_flights.txt (default: {defaults['ok_flights']}).")
    parser.add_argument("--csv-dir", default=defaults["csv_dir"],
                        help=f"Output CSV directory (default: {defaults['csv_dir']}).")
    parser.add_argument("--buffer", type=float, default=5.0,
                        help="Seconds of empty time inserted between pairs (default: 5.0).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the matplotlib summary plot at the end.")
    args = parser.parse_args()

    bags_dir   = os.path.abspath(args.bags_dir)
    flier_dir  = os.path.join(bags_dir, "flier")
    obs_dir    = os.path.join(bags_dir, "observer")
    ok_path    = os.path.abspath(args.ok_flights)
    csv_dir    = os.path.abspath(args.csv_dir)
    buffer_s   = float(args.buffer)

    for d, label in [(flier_dir, "flier"), (obs_dir, "observer")]:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: missing bag subdirectory: {d} ({label})")
    if not os.path.isfile(ok_path):
        sys.exit(f"ERROR: ok_flights file not found: {ok_path}")

    pairs = parse_ok_flights(ok_path)
    if not pairs:
        sys.exit(f"ERROR: no valid pairs in {ok_path}")

    print(f"Found {len(pairs)} pair(s) in {ok_path}:")
    for fl, ob, note in pairs:
        print(f"  flier={fl}  observer={ob}  {('# ' + note) if note else ''}")

    # Aggregated containers
    all_est, all_pred_rel, all_true_rel = [], [], []
    all_od1, all_od2 = [], []
    all_pts_seen, all_blk_seen = [], []
    join_times: List[float] = []
    used_bag_files: List[str] = []

    for i, (flier_name, observer_name, note) in enumerate(pairs):
        flier_path = os.path.join(flier_dir, flier_name)
        obs_path   = os.path.join(obs_dir,   observer_name)
        print(f"\n[{i + 1}/{len(pairs)}] {flier_name} <-> {observer_name}"
              f"{('  # ' + note) if note else ''}")

        if not os.path.isfile(flier_path):
            print(f"  ERROR: missing flier bag: {flier_path}")
            continue
        if not os.path.isfile(obs_path):
            print(f"  ERROR: missing observer bag: {obs_path}")
            continue

        try:
            est, pred_rel, true_rel, od1, od2, pts, blk, _T = \
                parse_real_world_pair(flier_path, obs_path)
        except Exception as e:
            print(f"  ERROR while parsing pair: {e}")
            continue

        used_bag_files.append(flier_path)
        used_bag_files.append(obs_path)

        print(f"  est={len(est)} pred_rel={len(pred_rel)} true_rel={len(true_rel)} "
              f"od1={len(od1)} od2={len(od2)} pts={len(pts)} blk={len(blk)}")

        if not est and not od1 and not od2:
            continue

        # Times are already pair-relative seconds (base_ns subtracted at read).
        # Clip to overlap end (later odom end belongs to the longer-running bag).
        if od1 and od2:
            t_end_pair = min(od1[-1].time, od2[-1].time)

            def _clip(lst, lo, hi):
                lst[:] = [p for p in lst if lo <= p.time <= hi]

            for data in (est, pred_rel, true_rel, od1, od2, pts, blk):
                _clip(data, 0.0, t_end_pair)
        else:
            print("  Skipping: missing odom on one side")
            continue

        # ---- Trim leading stationary flier ----
        MOTION_THRESH = 0.3  # metres
        x0, y0, z0 = od2[0].x, od2[0].y, od2[0].z
        t_motion_start = None
        for p in od2:
            if ((p.x - x0) ** 2 + (p.y - y0) ** 2
                    + (p.z - z0) ** 2) ** 0.5 > MOTION_THRESH:
                t_motion_start = p.time
                break

        # ---- Trim trailing stationary flier (iterate from the end) ----
        xN, yN, zN = od2[-1].x, od2[-1].y, od2[-1].z
        t_motion_end = None
        for p in reversed(od2):
            if ((p.x - xN) ** 2 + (p.y - yN) ** 2
                    + (p.z - zN) ** 2) ** 0.5 > MOTION_THRESH:
                t_motion_end = p.time
                break

        if t_motion_start is None or t_motion_end is None:
            print(f"  Warning: flier never moved > {MOTION_THRESH}m; "
                  f"keeping pair untrimmed")
        else:
            lead = t_motion_start - od2[0].time
            trail = od2[-1].time - t_motion_end
            if lead > 0 or trail > 0:
                print(f"  Trimming stationary: {lead:.2f}s leading, "
                      f"{trail:.2f}s trailing")
                for data in (est, pred_rel, true_rel, od1, od2, pts, blk):
                    _clip(data, t_motion_start, t_motion_end)
                    offset_poses(data, -t_motion_start)

        if not est and not od1 and not od2:
            continue

        # Append after the previous pair, with a gap = buffer
        prev_end = get_last_time(all_est, all_od1, all_od2)
        if all_est or all_od1 or all_od2:
            gap_start = prev_end + buffer_s
            join_times.append(prev_end)
            for data in (est, pred_rel, true_rel, od1, od2, pts, blk):
                offset_poses(data, gap_start)

        all_est.extend(est)
        all_pred_rel.extend(pred_rel)
        all_true_rel.extend(true_rel)
        all_od1.extend(od1)
        all_od2.extend(od2)
        all_pts_seen.extend(pts)
        all_blk_seen.extend(blk)

    print(f"\nTotal: est={len(all_est)} pred_rel={len(all_pred_rel)} "
          f"true_rel={len(all_true_rel)} od1={len(all_od1)} od2={len(all_od2)}")
    if all_od1:
        dur = all_od1[-1].time - all_od1[0].time
        print(f"Duration (observer odom): {dur:.1f}s ({dur / 3600:.2f}h)")

    # -- Save CSVs --
    os.makedirs(csv_dir, exist_ok=True)
    csv_map = {
        "odom1.csv":                   all_od1,       # observer (uav9)
        "odom2.csv":                   all_od2,       # flier    (uav14)
        "estimations.csv":             all_est,
        "predicted_relative_pose.csv": all_pred_rel,
        "true_relative_pose.csv":      all_true_rel,
    }
    for name, data in csv_map.items():
        if data:
            save_pose_csv(data, os.path.join(csv_dir, name))
    if all_pts_seen:
        save_points_seen_csv(all_pts_seen, os.path.join(csv_dir, "points_seen_right.csv"))
    if all_blk_seen:
        save_points_seen_csv(all_blk_seen, os.path.join(csv_dir, "blinkers_seen_right.csv"))

    total_hours = (all_od1[-1].time - all_od1[0].time) / 3600.0 if all_od1 else 0.0
    save_used_bags_txt(used_bag_files, os.path.join(csv_dir, "used_rosbags.txt"),
                       total_hours, join_times=join_times)

    # -- Plot --
    if not args.no_plot:
        try:
            from visualize_flight import plot_all
            plot_all(all_pred_rel, all_true_rel, all_od1, all_od2,
                     est=all_est, join_times=join_times, start_time=0.0)
        except Exception as e:
            print(f"  Warning: plotting skipped ({e})")


if __name__ == "__main__":
    main()
