#!/usr/bin/env python3
"""
Debug: try every reasonable shared-anchor frame + yaw rotations, compare to
the UVDAR predicted relative pose.

For each of these candidate dynamic TFs in the observer / flier bag::

    <uav>/fcu  <->  <uav>/utm_origin
    <uav>/fcu  <->  <uav>/world_origin
    <uav>/fcu  <->  <uav>/gps_baro_origin
    <uav>/fcu  <->  <uav>/fixed_origin
    <uav>/fcu  <->  <uav>/stable_origin

we compute T_obs_fcu<-fl_fcu(t) by anchoring both UAVs through the candidate
frame, then plot x,y,z of the result alongside the UVDAR prediction.

Additionally, each candidate is replicated with extra yaw rotations of
0, +/-60, +/-120, +/-180 deg applied IN THE OBSERVER FCU FRAME on the left
(i.e. the resulting position is rotated about the z axis), so any constant
heading offset between the shared frames becomes obvious.

Usage::

    python3 data/debug_relative_frames.py                    # first pair in ok_flights.txt
    python3 data/debug_relative_frames.py --pair-index 2     # 3rd pair
    python3 data/debug_relative_frames.py --flier <bag> --observer <bag>
"""
import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import rosbag  # noqa: F401
except Exception:
    sys.stderr.write("ERROR: rosbag missing. Source /opt/ros/<distro>/setup.bash\n")
    sys.exit(2)

from bag_parser import (
    PoseData, msg_to_pose_data,
    get_transform_components, transform_pose_list_to_fcu,
    se3_to_pose_data, _tf_msg_to_se3, make_se3, euler_to_quat,
)
from bag_parser_real_world import (
    OBSERVER_UAV, FLIER_UAV,
    OBS_TOPIC_ODOM, FLIER_TOPIC_ODOM,
    UVDAR_CANDIDATES,
    _pick_uvdar_topic, _first_odom_ns, _ns_to_rel_sec,
    parse_ok_flights, _default_paths,
)


CANDIDATE_FRAMES = [
    "gps_baro_origin",  # confirmed by visual inspection as best anchor
]
# Fine sweep around -120 deg (light-green trace from the prior broad sweep).
YAW_OFFSETS_DEG = [-150, -135, -130, -125, -120, -115, -110, -105, -90, 0]


# ---- TF readers --------------------------------------------------------------

def _read_fcu_from_anchor(bag_path, uav, anchor, base_ns
                          ) -> List[Tuple[float, np.ndarray]]:
    """Return ``T_fcu<-anchor`` time series (REP-105: parent=fcu, child=anchor)."""
    parent = f"{uav}/fcu"
    child  = f"{uav}/{anchor}"
    out: List[Tuple[float, np.ndarray]] = []
    with rosbag.Bag(bag_path) as bag:
        for _topic, msg, _t in bag.read_messages(topics=["/tf", "/tf_static"]):
            for tf in getattr(msg, "transforms", []):
                fid = tf.header.frame_id.lstrip("/")
                cid = tf.child_frame_id.lstrip("/")
                if fid == parent and cid == child:
                    T = _tf_msg_to_se3(tf)
                elif fid == child and cid == parent:
                    T = np.linalg.inv(_tf_msg_to_se3(tf))
                else:
                    continue
                stamp_ns = tf.header.stamp.to_nsec()
                out.append((_ns_to_rel_sec(stamp_ns, base_ns), T))
    out.sort(key=lambda x: x[0])
    return out


def _read_uvdar_pred_in_obs_fcu(observer_bag, base_ns) -> List[PoseData]:
    """Run the same observer->fcu transform chain used by the main parser."""
    estimations: List[PoseData] = []
    uvdar_topic = _pick_uvdar_topic(observer_bag)
    with rosbag.Bag(observer_bag) as bag:
        for topic, msg, _t in bag.read_messages(topics=[uvdar_topic]):
            stamp_ns = msg.header.stamp.to_nsec()
            if stamp_ns < base_ns:
                continue
            tsec = _ns_to_rel_sec(stamp_ns, base_ns)
            if msg._type == "mrs_msgs/PoseWithCovarianceArrayStamped":
                if hasattr(msg, "poses") and msg.poses:
                    p = msg.poses[0].pose
                    estimations.append(msg_to_pose_data(
                        tsec, p.position, p.orientation))
            elif msg._type == "geometry_msgs/PoseArray":
                if hasattr(msg, "poses") and msg.poses:
                    p = msg.poses[0]
                    estimations.append(msg_to_pose_data(
                        tsec, p.position, p.orientation))
    if not estimations:
        return []
    obs_id = int(OBSERVER_UAV.replace("uav", ""))
    T_fl, dyn = get_transform_components(observer_bag, uav_id=obs_id, T_fixed_local=None)
    base_sec = base_ns * 1e-9
    dyn_rebased = [(s - base_sec, T) for s, T in dyn]
    return transform_pose_list_to_fcu(estimations, T_fl, dyn_rebased)


# ---- Composition -------------------------------------------------------------

def _nearest(timestamp, tfs, cursor):
    last = len(tfs) - 1
    while (cursor < last
           and abs(tfs[cursor + 1][0] - timestamp)
               <= abs(tfs[cursor][0] - timestamp)):
        cursor += 1
    return cursor, tfs[cursor][1]


def compose_relative(flier_tfs, observer_tfs, yaw_deg=0.0) -> List[PoseData]:
    """T_obs_fcu<-fl_fcu = R_yaw @ T_obs_fcu<-anchor @ inv(T_fl_fcu<-anchor).

    A pre-rotation of yaw_deg about z is applied on the LEFT (in observer
    fcu).  This is the place to check for any constant heading offset
    between the two UAVs' anchor frames.
    """
    if not flier_tfs or not observer_tfs:
        return []
    if abs(yaw_deg) > 1e-9:
        qx, qy, qz, qw = euler_to_quat(0.0, 0.0, np.deg2rad(yaw_deg))
        R_yaw = make_se3(0, 0, 0, qx, qy, qz, qw)
    else:
        R_yaw = np.eye(4)
    out: List[PoseData] = []
    cursor = 0
    for t_fl, T_fl in flier_tfs:
        cursor, T_obs = _nearest(t_fl, observer_tfs, cursor)
        T_rel = R_yaw @ T_obs @ np.linalg.inv(T_fl)
        out.append(se3_to_pose_data(t_fl, T_rel))
    return out


# ---- Plot --------------------------------------------------------------------

def _plot_series(ax, poses, label, color, alpha=0.9, lw=0.8):
    if not poses:
        return
    ax.plot([p.time for p in poses],
            [getattr(p, ax._field) for p in poses],
            label=label, color=color, alpha=alpha, linewidth=lw)


def plot_candidates(predicted, candidates: dict, title_suffix=""):
    """candidates: {label: List[PoseData]}.  One figure per axis (x,y,z)."""
    fields = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    cmap = plt.get_cmap("tab20")
    for ax, field in zip(axes, fields):
        ax._field = field
        _plot_series(ax, predicted, "predicted (UVDAR)", "black", alpha=1.0, lw=1.4)
        for i, (label, poses) in enumerate(candidates.items()):
            _plot_series(ax, poses, label, cmap(i % 20),
                         alpha=0.9, lw=0.7)
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"True-relative-pose candidate frames {title_suffix}")
    axes[-1].set_xlabel("time [s]")
    # One shared legend below the plots (it'll be long)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ---- Main --------------------------------------------------------------------

def main():
    defaults = _default_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags-dir", default=defaults["bags_dir"])
    ap.add_argument("--ok-flights", default=defaults["ok_flights"])
    ap.add_argument("--pair-index", type=int, default=0,
                    help="0-based index into ok_flights.txt (default: 0).")
    ap.add_argument("--flier", help="Override flier bag path.")
    ap.add_argument("--observer", help="Override observer bag path.")
    ap.add_argument("--no-yaw-sweep", action="store_true",
                    help="Plot only yaw=0 for each frame (one figure total).")
    args = ap.parse_args()

    if args.flier and args.observer:
        flier_path = os.path.abspath(args.flier)
        obs_path   = os.path.abspath(args.observer)
    else:
        bags_dir = os.path.abspath(args.bags_dir)
        pairs = parse_ok_flights(os.path.abspath(args.ok_flights))
        if not pairs:
            sys.exit("No pairs found in ok_flights.txt")
        if args.pair_index >= len(pairs):
            sys.exit(f"--pair-index {args.pair_index} out of range (have {len(pairs)})")
        flier_name, observer_name, note = pairs[args.pair_index]
        flier_path = os.path.join(bags_dir, "flier", flier_name)
        obs_path   = os.path.join(bags_dir, "observer", observer_name)
        print(f"Pair {args.pair_index}: {flier_name} <-> {observer_name}  {note}")

    for p in (flier_path, obs_path):
        if not os.path.isfile(p):
            sys.exit(f"Missing bag: {p}")

    # Common ns base = later odom start
    obs_first_ns = _first_odom_ns(obs_path,   OBS_TOPIC_ODOM)
    fl_first_ns  = _first_odom_ns(flier_path, FLIER_TOPIC_ODOM)
    if obs_first_ns is None or fl_first_ns is None:
        sys.exit("missing odom in one bag")
    base_ns = max(obs_first_ns, fl_first_ns)
    print(f"  Bag start offset (flier - observer): "
          f"{(fl_first_ns - obs_first_ns) * 1e-9:+.6f}s")

    # UVDAR prediction in observer's FCU
    print("Reading UVDAR predicted poses ...")
    predicted = _read_uvdar_pred_in_obs_fcu(obs_path, base_ns)
    print(f"  predicted poses: {len(predicted)}")

    # Build candidate-frame dict
    print("Reading candidate anchor TFs ...")
    frame_tfs = {}
    for anchor in CANDIDATE_FRAMES:
        obs_tfs = _read_fcu_from_anchor(obs_path,   OBSERVER_UAV, anchor, base_ns)
        fl_tfs  = _read_fcu_from_anchor(flier_path, FLIER_UAV,    anchor, base_ns)
        print(f"  {anchor:20s}  observer={len(obs_tfs):6d}  flier={len(fl_tfs):6d}")
        if obs_tfs and fl_tfs:
            frame_tfs[anchor] = (obs_tfs, fl_tfs)

    if not frame_tfs:
        sys.exit("No candidate anchor frames available in both bags.")

    # One figure per anchor, all yaw offsets overlaid.
    yaws = [0.0] if args.no_yaw_sweep else YAW_OFFSETS_DEG
    for anchor, (obs_tfs, fl_tfs) in frame_tfs.items():
        cands = {}
        for yaw in yaws:
            label = f"{anchor} yaw={yaw:+.0f}"
            cands[label] = compose_relative(fl_tfs, obs_tfs, yaw_deg=yaw)
        plot_candidates(predicted, cands,
                        title_suffix=f"[{anchor}, "
                                     f"{os.path.basename(flier_path)}]")

    print("Showing plots ... close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
