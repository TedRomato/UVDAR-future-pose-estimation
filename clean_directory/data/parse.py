#!/usr/bin/env python3
"""Clean multi-UAV bag parser.

Auto-detects REAL (paired bags via flight_pairs.txt) vs SIMULATION
(single bag per flight via flights.txt) datasets, parses raw topics,
aligns timestamps, interpolates odometry to UVDAR/blinker timestamps,
and writes CSVs in the OBSERVER CAMERA FRAME.

Usage:
    source /opt/ros/noetic/setup.bash
    python3 parse.py --input  data/real_world_data \
                     --output data/real_world_data/clean_csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import rosbag
except ImportError:
    sys.stderr.write("ERROR: source /opt/ros/<distro>/setup.bash first\n")
    sys.exit(2)

from se3 import make_T, tf_msg_to_T, T_to_xyzquat


# ============================================================================
# Config
# ============================================================================

def load_uav_config(input_dir):
    """Parse uav_config.txt, return (observer, flier) UAV name strings."""
    cfg = {}
    with open(os.path.join(input_dir, "uav_config.txt")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip().lower()] = v.strip().strip('"').strip("'")
    return cfg["observer"], cfg["flier"]


def detect_dataset_type(input_dir):
    """Return 'real' if flight_pairs.txt present, 'sim' if flights.txt."""
    if os.path.exists(os.path.join(input_dir, "flight_pairs.txt")):
        return "real"
    if os.path.exists(os.path.join(input_dir, "flights.txt")):
        return "sim"
    raise FileNotFoundError(
        f"Need flight_pairs.txt or flights.txt in {input_dir}")


def bags_dir(input_dir):
    sub = os.path.join(input_dir, "bags")
    return sub if os.path.isdir(sub) else input_dir


# ============================================================================
# Flight list parsing
# ============================================================================

def load_real_pairs(input_dir):
    """Return list of (observer_bag_path, flier_bag_path)."""
    pairs = []
    bdir = bags_dir(input_dir)
    with open(os.path.join(input_dir, "flight_pairs.txt")) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            parts = [p for p in parts if p.endswith(".bag")]
            print("parts", parts)
            if len(parts) < 2:
                continue
            obs, fl = parts[0], parts[1]
            pairs.append((os.path.join(bdir, obs), os.path.join(bdir, fl)))
    return pairs


def load_sim_flights(input_dir):
    """Return list of bag paths."""
    flights = []
    bdir = bags_dir(input_dir)
    with open(os.path.join(input_dir, "flights.txt")) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name = line.split()[0]
            if not name.endswith(".bag"):
                name += ".bag"
            flights.append(os.path.join(bdir, name))
    return flights


# ============================================================================
# Raw topic loading
# ============================================================================

def _stamp_ns(msg):
    """Read msg.header.stamp as integer nanoseconds (precise; no float64 loss)."""
    return msg.header.stamp.to_nsec()

def _stamp_ns_no_header(msg):
    """Read msg.stamp as integer nanoseconds (precise; no float64 loss)."""
    return msg.stamp.to_nsec()


def _odom_entry(msg):
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    return {"t": _stamp_ns(msg),
            "x": p.x, "y": p.y, "z": p.z,
            "qx": o.x, "qy": o.y, "qz": o.z, "qw": o.w}


def _uvdar_entry(msg):
    """First pose from msg.poses[].pose."""
    if not msg.poses:
        return None
    pose = msg.poses[0].pose
    pos, ori = pose.position, pose.orientation
    return {"t": _stamp_ns(msg),
            "x": pos.x, "y": pos.y, "z": pos.z,
            "qx": ori.x, "qy": ori.y, "qz": ori.z, "qw": ori.w}


def _blinker_entry(msg):
    pts = [(float(p.x), float(p.y), float(p.value)) for p in msg.points]
    return {"t": _stamp_ns_no_header(msg),
            "points": pts,
            "image_height": int(msg.image_height),
            "image_width": int(msg.image_width)}


def _tf_entries(msg):
    out = []
    for tf in msg.transforms:
        out.append({"t": tf.header.stamp.to_nsec(),
                    "frame_id": tf.header.frame_id.lstrip("/"),
                    "child_frame_id": tf.child_frame_id.lstrip("/"),
                    "T": tf_msg_to_T(tf)})
    return out


def _progress_tick(t_ns, t0_ns, t1_ns, next_pct):
    """If `t_ns` crossed the next 5% mark, print and return new threshold."""
    if t1_ns <= t0_ns:
        return next_pct
    pct = 100.0 * (t_ns - t0_ns) / (t1_ns - t0_ns)
    while pct >= next_pct and next_pct <= 100:
        print(f"{next_pct:3d}%", flush=True, end="\r")
        next_pct += 5
    return next_pct


def load_observer_bag(bag_path, observer):
    """Read observer odom, UVDAR estimates, blinkers, /tf, /tf_static."""
    odom_topic = f"/{observer}/estimation_manager/odom_main"
    blink_topic = f"/{observer}/uvdar/blinkers_seen_right"
    uvdar_topic = f"/{observer}/uvdar/measuredPoses"
    out = {"odom": [], "uvdar": [], "blinkers": [], "tf": [], "tf_static": []}
    print("Opening the bag...")
    with rosbag.Bag(bag_path) as bag:
        print("Starting the read...", flush=True)
        t0_ns = int(bag.get_start_time() * 1e9)
        t1_ns = int(bag.get_end_time() * 1e9)
        next_pct = 5
        topics = [odom_topic, blink_topic, uvdar_topic, "/tf", "/tf_static"]
        for topic, msg, t in bag.read_messages(topics=topics):
            if topic == odom_topic:
                out["odom"].append(_odom_entry(msg))
                next_pct = _progress_tick(out["odom"][-1]["t"], t0_ns, t1_ns, next_pct)
            elif topic == uvdar_topic:
                e = _uvdar_entry(msg)
                if e is not None:
                    out["uvdar"].append(e)
            elif topic == blink_topic:
                out["blinkers"].append(_blinker_entry(msg))
            elif topic == "/tf":
                out["tf"].extend(_tf_entries(msg))
            elif topic == "/tf_static":
                out["tf_static"].extend(_tf_entries(msg))
        print("Read finished.", flush=True)
    for k in ("odom", "uvdar", "blinkers", "tf"):
        out[k].sort(key=lambda e: e["t"])
    return out


def load_flier_bag(bag_path, flier):
    """Read flier odom, /tf, /tf_static."""
    odom_topic = f"/{flier}/estimation_manager/odom_main"
    out = {"odom": [], "tf": [], "tf_static": []}
    with rosbag.Bag(bag_path) as bag:
        print("Starting the read...", flush=True)
        t0_ns = int(bag.get_start_time() * 1e9)
        t1_ns = int(bag.get_end_time() * 1e9)
        next_pct = 5
        for topic, msg, t in bag.read_messages(
                topics=[odom_topic, "/tf", "/tf_static"]):
            if topic == odom_topic:
                out["odom"].append(_odom_entry(msg))
                next_pct = _progress_tick(out["odom"][-1]["t"], t0_ns, t1_ns, next_pct)
            elif topic == "/tf":
                out["tf"].extend(_tf_entries(msg))
            elif topic == "/tf_static":
                out["tf_static"].extend(_tf_entries(msg))
        print("Read finished.", flush=True)
    for k in ("odom", "tf"):
        out[k].sort(key=lambda e: e["t"])
    return out


# ============================================================================
# Time alignment
# ============================================================================

MOTION_THRESH = 0.3  # m, for stationary trim


def crop_interval(observer, flier):
    """Crop both raw arrays to the time intersection of the two odoms.

    During load+crop, `t` is integer nanoseconds, so the intersection is
    exact (float64 seconds would lose ~200 ns at epoch values).
    Mutates observer/flier dicts in place.
    """
    t_lo = max(observer["odom"][0]["t"], flier["odom"][0]["t"])
    t_hi = min(observer["odom"][-1]["t"], flier["odom"][-1]["t"])
    if t_hi <= t_lo:
        raise RuntimeError("observer and flier odoms do not overlap in time")
    for d in (observer, flier):
        for k, v in list(d.items()):
            if k.startswith("tf_static"):
                continue
            d[k] = [e for e in v if t_lo <= e["t"] <= t_hi]
    return t_lo, t_hi


def trim_stationary(flier, observer):
    """Drop leading/trailing samples where flier hasn't moved."""
    od = flier["odom"]
    if len(od) < 2:
        return
    p0 = np.array([od[0]["x"], od[0]["y"], od[0]["z"]])
    i_start = 0
    for i, e in enumerate(od):
        if np.linalg.norm([e["x"] - p0[0], e["y"] - p0[1], e["z"] - p0[2]]) > MOTION_THRESH:
            i_start = i
            break
    pE = np.array([od[-1]["x"], od[-1]["y"], od[-1]["z"]])
    i_end = len(od) - 1
    for i in range(len(od) - 1, -1, -1):
        e = od[i]
        if np.linalg.norm([e["x"] - pE[0], e["y"] - pE[1], e["z"] - pE[2]]) > MOTION_THRESH:
            i_end = i
            break
    t_lo, t_hi = od[i_start]["t"], od[i_end]["t"]
    for d in (observer, flier):
        for k, v in list(d.items()):
            if k.startswith("tf_static"):
                continue
            d[k] = [e for e in v if t_lo <= e["t"] <= t_hi]
    return 

# ============================================================================
# Interpolation
# ============================================================================

def interp_odom(odom, target_times):
    """Linearly interpolate odom xyz onto target_times; for the orientation,
    take the quaternion from whichever bracketing odom sample is closest in
    time. Both `odom` and `target_times` must be sorted ascending.
    """
    n = len(odom)
    out = []
    i = 0  # invariant: odom[i].t <= t < odom[i+1].t (when in range)
    for t in target_times:
        while i + 1 < n and odom[i + 1]["t"] <= t:
            i += 1
        if t <= odom[0]["t"]:
            e = odom[0]
            out.append({"t": t, "x": e["x"], "y": e["y"], "z": e["z"],
                        "qx": e["qx"], "qy": e["qy"], "qz": e["qz"], "qw": e["qw"]})
        elif i + 1 >= n:
            e = odom[-1]
            out.append({"t": t, "x": e["x"], "y": e["y"], "z": e["z"],
                        "qx": e["qx"], "qy": e["qy"], "qz": e["qz"], "qw": e["qw"]})
        else:
            a, b = odom[i], odom[i + 1]
            dt = b["t"] - a["t"]
            u = 0.0 if dt <= 0 else (t - a["t"]) / dt
            q = a if (t - a["t"]) <= (b["t"] - t) else b
            out.append({"t": t,
                        "x": a["x"] + u * (b["x"] - a["x"]),
                        "y": a["y"] + u * (b["y"] - a["y"]),
                        "z": a["z"] + u * (b["z"] - a["z"]),
                        "qx": q["qx"], "qy": q["qy"],
                        "qz": q["qz"], "qw": q["qw"]})
    return out


def target_timestamps(observer):
    """Sorted union of blinker + UVDAR estimate timestamps (deduplicated via set)."""
    return sorted({e["t"] for e in observer["blinkers"]}
                  | {e["t"] for e in observer["uvdar"]})


# ============================================================================
# TF lookup
# ============================================================================

def static_T(tf_static_list, frame_id, child_frame_id):
    """Find one static transform T_frame_from_child."""
    for e in tf_static_list:
        if e["frame_id"] == frame_id and e["child_frame_id"] == child_frame_id:
            return e["T"]
    for e in tf_static_list:
        if e["frame_id"] == child_frame_id and e["child_frame_id"] == frame_id:
            return np.linalg.inv(e["T"])
    raise RuntimeError(f"static TF {frame_id} -> {child_frame_id} not found")


def dynamic_stream(tf_list, frame_id, child_frame_id):
    """Time-sorted list of (t, T_frame_from_child) for a dynamic edge."""
    out = []
    for e in tf_list:
        if e["frame_id"] == frame_id and e["child_frame_id"] == child_frame_id:
            out.append((e["t"], e["T"]))
        elif e["frame_id"] == child_frame_id and e["child_frame_id"] == frame_id:
            out.append((e["t"], np.linalg.inv(e["T"])))
    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError(
            f"dynamic TF {frame_id} <-> {child_frame_id} not found")
    return out


class TFCursor:
    """Forward-only nearest-in-time lookup over a dynamic stream."""

    def __init__(self, stream):
        self.s = stream
        self.i = 0

    def at(self, t):
        s = self.s
        last = len(s) - 1
        while (self.i < last
               and abs(s[self.i + 1][0] - t) <= abs(s[self.i][0] - t)):
            self.i += 1
        return s[self.i][1]


# ============================================================================
# Frame conversion
# ============================================================================

def odom_T(e):
    return make_T(e["x"], e["y"], e["z"], e["qx"], e["qy"], e["qz"], e["qw"])


def to_camera_frame_flier(observer_odom, flier_odom, T_fcu_cam):
    """T_camera_flier = inv(T_fcu_cam) @ inv(T_obs_odom) @ T_flier_odom."""
    T_cam_fcu = np.linalg.inv(T_fcu_cam)
    out = []
    for o, f in zip(observer_odom, flier_odom):
        T_rel = T_cam_fcu @ np.linalg.inv(odom_T(o)) @ odom_T(f)
        x, y, z, qx, qy, qz, qw = T_to_xyzquat(T_rel)
        out.append({"t": o["t"], "x": x, "y": y, "z": z,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    return out


def to_camera_frame_uvdar(uvdar, T_fcu_cam, T_fixed_local, dyn_fcu_fixed):
    """T_camera_uvdar = inv(T_fcu_cam) @ T_fcu_fixed(t) @ T_fixed_local @ T_uvdar."""
    T_cam_fcu = np.linalg.inv(T_fcu_cam)
    cur = TFCursor(dyn_fcu_fixed)
    out = []
    for e in uvdar:
        T_uv = make_T(e["x"], e["y"], e["z"],
                      e["qx"], e["qy"], e["qz"], e["qw"])
        T = T_cam_fcu @ cur.at(e["t"]) @ T_fixed_local @ T_uv
        x, y, z, qx, qy, qz, qw = T_to_xyzquat(T)
        out.append({"t": e["t"], "x": x, "y": y, "z": z,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    return out


# ============================================================================
# CSV append + metadata
# ============================================================================

POSE_COLS = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]


def _append_csv(path, cols, rows, t_offset):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(cols)
        for r in rows:
            r = dict(r)
            r["t"] = r["t"] + t_offset
            w.writerow([r[c] if c != "points" else json.dumps(r[c])
                        for c in cols])


def append_pose(path, poses, t_offset):
    _append_csv(path, POSE_COLS, poses, t_offset)


def append_blinkers(path, blinkers, t_offset):
    _append_csv(path,
                ["t", "points", "image_height", "image_width"],
                blinkers, t_offset)


def write_metadata(out_dir, bag_paths, join_times_ns, total_ns):
    path = os.path.join(out_dir, "used_rosbags.txt")
    with open(path, "w") as f:
        for p in bag_paths:
            f.write(f"{os.path.abspath(p)}\n")
        f.write("\n")
        f.write("Join times: " + ",".join(str(j) for j in join_times_ns) + "\n")
        f.write("\n")
        f.write(f"Total hours: {total_ns / 3600e9:.2f}\n")


# ============================================================================
# Per-flight processing
# ============================================================================

BUFFER_NS = 10 * 10**9  # 10 s gap between flights when stitching their timelines

# Static T_fcu_uvcam_left from the MRS UVDAR mount, hard-coded because the
# bags being repaired do not contain this edge in /tf_static. Used only when
# --fix-relative is set; matches uav1/uvcam_left.
T_FCU_UVCAM_LEFT_XYZQ = (
    0.03, 0.1, 0.06,
    -0.6963642403182444, 0.12278780396906709,
    -0.12278780396966833, 0.6963642403216542,
)

# Static TFs are mostly identical across flights; cache them so we can keep
# going even when a bag was recorded without /tf_static.
_STATIC_CACHE = {
    "observer_tf_static": None,  # list of TF entry dicts
    "flier_tf_static": None,
}


def _cache_or_warn_static(side, tf_static):
    """Stash the new static TFs if present; otherwise reuse the cached ones."""
    key = f"{side}_tf_static"
    if tf_static:
        _STATIC_CACHE[key] = tf_static
        return tf_static
    cached = _STATIC_CACHE[key]
    if cached is None:
        raise RuntimeError(
            f"No /tf_static for {side} in this bag and no cached one available.")
    print(f"WARN: no /tf_static for {side} in this bag; using cached "
          f"({len(cached)} entries).")
    return cached


def process_flight(observer_raw, flier_raw, observer, flier, fix_relative=False):
    """Run alignment + interpolation + frame conversion for one flight.

    Returns a dict of all final arrays, with timestamps starting at the
    cropped t_lo (i.e. NOT yet shifted onto the merged timeline).
    """
    print("crop + trim ...", flush=True)
    t_lo, t_hi = crop_interval(observer_raw, flier_raw)
    print(f"Cropped to intersection: {t_lo / 1e9:.2f}s - {t_hi / 1e9:.2f}s")
    trim_stationary(flier_raw, observer_raw)
    print(f"Cropped to flight in interval: {t_lo / 1e9:.2f}s - {t_hi / 1e9:.2f}s")

    # Cache / fallback for static TFs.
    observer_raw["tf_static"] = _cache_or_warn_static(
        "observer", observer_raw["tf_static"])
    flier_raw["tf_static"] = _cache_or_warn_static(
        "flier", flier_raw["tf_static"])

    print("rebasing timestamps...", flush=True)
    base_ns = min(observer_raw["odom"][0]["t"], flier_raw["odom"][0]["t"])
    for d in (observer_raw, flier_raw):
        for k in ("odom", "uvdar", "blinkers", "tf"):
            if not k in d:
                continue
            for e in d[k]:
                e["t"] = e["t"] - base_ns

    print("interpolate observer + flier odom to UVDAR/blinker times ...", flush=True)
    targets = target_timestamps(observer_raw)
    obs_odom = interp_odom(observer_raw["odom"], targets)
    fl_odom  = interp_odom(flier_raw["odom"],  targets)

    print("look up TFs ...", flush=True)
    T_fcu_cam = static_T(observer_raw["tf_static"],
                         f"{observer}/fcu",
                         f"{observer}/uvcam_right")
    T_fixed_local = static_T(observer_raw["tf_static"],
                             f"{observer}/fixed_origin",
                             f"{observer}/local_origin")
    dyn = dynamic_stream(observer_raw["tf"],
                         f"{observer}/fcu",
                         f"{observer}/fixed_origin")

    print("convert to observer camera frame ...", flush=True)
    if fix_relative:
        T_fcu_left = make_T(*T_FCU_UVCAM_LEFT_XYZQ)
        print("  --fix-relative: flier in left-camera frame, "
              "UVDAR still in right-camera frame", flush=True)
        flier_in_cam = to_camera_frame_flier(obs_odom, fl_odom, T_fcu_left)
    else:
        flier_in_cam = to_camera_frame_flier(obs_odom, fl_odom, T_fcu_cam)
    uvdar_in_cam = to_camera_frame_uvdar(observer_raw["uvdar"],
                                         T_fcu_cam, T_fixed_local, dyn)

    duration = max(observer_raw["odom"][-1]["t"], flier_raw["odom"][-1]["t"])
    print(f"flight duration = {duration / 1e9:.2f}s", flush=True)

    return {
        "observer_odom_original": observer_raw["odom"],
        "observer_odom": obs_odom,
        "flier_odom_original": flier_raw["odom"],
        "flier_odom": fl_odom,
        "blinkers_right": observer_raw["blinkers"],
        "original_uvdar_estimate": observer_raw["uvdar"],
        "uvdar_estimate_in_camera_frame": uvdar_in_cam,
        "flier_odom_in_camera_frame": flier_in_cam,
        "duration": duration,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input directory")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--buffer", type=int, default=BUFFER_NS,
                    help=f"Gap between flights when appending (ns, default {BUFFER_NS})")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the visualization at the end.")
    ap.add_argument("--extras", action="store_true",
                    help="Visualization: also show distance + speed panels.")
    ap.add_argument("--relative", action="store_true",
                    help="Visualization: show camera-frame relative pose "
                         "instead of absolute world pose.")
    ap.add_argument("--fix-relative", action="store_true",
                    help="Build flier_odom_in_camera_frame.csv in the LEFT "
                         "camera frame (using a hard-coded T_fcu_uvcam_left) "
                         "while keeping UVDAR in the right camera frame. "
                         "Use when ground truth was recorded on the left "
                         "camera but stored under the right-camera topic.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    print("Directories: ", in_dir, "->", out_dir)

    dataset_type = detect_dataset_type(in_dir)
    observer, flier = load_uav_config(in_dir)
    print(f"Dataset: {dataset_type}  observer={observer}  flier={flier}")

    # Build flight inputs
    if dataset_type == "real":
        flight_inputs = load_real_pairs(in_dir)        # [(obs_bag, fl_bag), ...]
    else:
        flight_inputs = [(b, b) for b in load_sim_flights(in_dir)]

    print(f"Flight pairs: \n  " + "\n  ".join(f"{os.path.basename(o)}  {os.path.basename(f)}" for o, f in flight_inputs))

    # CSV paths
    csv_paths = {
        "observer_odom_original":         os.path.join(out_dir, "observer_odom_original.csv"),
        "observer_odom":                  os.path.join(out_dir, "observer_odom.csv"),
        "flier_odom_original":            os.path.join(out_dir, "flier_odom_original.csv"),
        "flier_odom":                     os.path.join(out_dir, "flier_odom.csv"),
        "blinkers_right":                 os.path.join(out_dir, "blinkers_right.csv"),
        "original_uvdar_estimate":        os.path.join(out_dir, "original_uvdar_estimate.csv"),
        "uvdar_estimate_in_camera_frame": os.path.join(out_dir, "uvdar_estimate_in_camera_frame.csv"),
        "flier_odom_in_camera_frame":     os.path.join(out_dir, "flier_odom_in_camera_frame.csv"),
    }

    timeline_end = 0  # ns
    join_times = []   # list of int ns
    bag_paths_used = []

    n = len(flight_inputs)
    for idx, (obs_bag, fl_bag) in enumerate(flight_inputs):
        print(f"\n[{idx + 1}/{n}] obs={os.path.basename(obs_bag)}  "
              f"fl={os.path.basename(fl_bag)}")

        print("loading bags ...", flush=True)
        if dataset_type == "real":
            print(f"observer bag: {os.path.basename(obs_bag)}", flush=True)
            obs_raw = load_observer_bag(obs_bag, observer)
            print(f"flier bag:    {os.path.basename(fl_bag)}", flush=True)
            fl_raw  = load_flier_bag(fl_bag, flier)
            bag_paths_used.extend([obs_bag, fl_bag])
        else:
            print(f"bag (observer pass): {os.path.basename(obs_bag)}", flush=True)
            obs_raw = load_observer_bag(obs_bag, observer)
            print(f"bag (flier pass):    {os.path.basename(obs_bag)}", flush=True)
            fl_raw  = load_flier_bag(obs_bag, flier)
            bag_paths_used.append(obs_bag)
        print(f"observer: odom={len(obs_raw['odom'])} "
              f"uvdar={len(obs_raw['uvdar'])} "
              f"blinkers={len(obs_raw['blinkers'])} "
              f"tf={len(obs_raw['tf'])} tf_static={len(obs_raw['tf_static'])}")
        print(f"flier:    odom={len(fl_raw['odom'])} "
              f"tf={len(fl_raw['tf'])} tf_static={len(fl_raw['tf_static'])}")

        print("processing ...", flush=True)
        result = process_flight(obs_raw, fl_raw, observer, flier,
                                fix_relative=args.fix_relative)

        # Shift this flight to start at timeline_end + buffer (or 0 if first).
        if timeline_end > 0:
            t_offset = timeline_end + args.buffer
            join_times.append(t_offset)
        else:
            t_offset = 0

        print(f"appending CSVs (t_offset={t_offset / 1e9:.2f}s) ...", flush=True)
        append_pose(csv_paths["observer_odom_original"],
                    result["observer_odom_original"], t_offset)
        append_pose(csv_paths["observer_odom"],
                    result["observer_odom"], t_offset)
        append_pose(csv_paths["flier_odom_original"],
                    result["flier_odom_original"], t_offset)
        append_pose(csv_paths["flier_odom"],
                    result["flier_odom"], t_offset)
        append_pose(csv_paths["original_uvdar_estimate"],
                    result["original_uvdar_estimate"], t_offset)
        append_pose(csv_paths["uvdar_estimate_in_camera_frame"],
                    result["uvdar_estimate_in_camera_frame"], t_offset)
        append_pose(csv_paths["flier_odom_in_camera_frame"],
                    result["flier_odom_in_camera_frame"], t_offset)
        append_blinkers(csv_paths["blinkers_right"],
                        result["blinkers_right"], t_offset)

        timeline_end = t_offset + result["duration"]
        print(f"done.  duration={result['duration'] / 1e9:.2f}s  "
              f"timeline_end={timeline_end / 1e9:.2f}s  ",
              flush=True)

    write_metadata(out_dir, bag_paths_used, join_times, timeline_end)
    print(f"\nDone. Total: {timeline_end / 3600e9:.2f} h. Wrote {out_dir}")

    if not args.no_plot:
        print("\nLaunching visualization ...")
        import visualize_dataset
        # visualizer takes --duration in seconds.
        sys.argv = ["visualize_dataset", out_dir,
                    "--duration", str(max(timeline_end / 1e9 + 1.0, 1.0))]
        if args.extras:
            sys.argv.append("--extras")
        if args.relative:
            sys.argv.append("--relative")
        visualize_dataset.main()


if __name__ == "__main__":
    main()
