#!/usr/bin/env python3
# Usage: python3 simple_rosbag_export.py <path_to_bag>

import sys, os, csv, math

try:
    import rosbag
except Exception:
    sys.stderr.write("ERROR: Could not import 'rosbag'. Did you source /opt/ros/<distro>/setup.bash?\n")
    sys.exit(2)

# ---------- helpers ----------
def quat_to_rpy(qx, qy, qz, qw):
    # ZYX convention (yaw about Z)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    sinp = 2.0 * (qw*qy - qz*qx)
    pitch = math.copysign(math.pi/2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw

def rpy_to_sin_cos(roll, pitch, yaw):
    return (
        math.sin(roll),  math.cos(roll),
        math.sin(pitch), math.cos(pitch),
        math.sin(yaw),   math.cos(yaw),
    )

def make_outdir_from_bag(bag_path):
    """
    helix/bags/1/run.bag -> data/helix/csv_data/1/
    Fallback: data/<parent_of_bag_dir>/csv_data/<bag_dir_name>/
    """
    ap = os.path.abspath(bag_path)
    parts = ap.split(os.sep)
    if "bags" in parts:
        i = parts.index("bags")
        base = os.path.join("data", os.sep.join(parts[:i]))      # e.g. data/helix
        sub  = os.path.join(*parts[i+1:-1]) if len(parts[i+1:-1]) else ""
        out  = os.path.join(base, "csv_data", sub)               # e.g. data/helix/csv_data/1
    else:
        bag_dir = os.path.dirname(ap)
        parent  = os.path.dirname(bag_dir)
        out     = os.path.join("data", parent, "csv_data", os.path.basename(bag_dir))
    os.makedirs(out, exist_ok=True)
    return out

def write_header(w):
    w.writerow(["time","x","y","z",
                "roll_sin","roll_cos","pitch_sin","pitch_cos","yaw_sin","yaw_cos"])

def write_row(w, tsec, p, q):
    roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
    r_s, r_c, p_s, p_c, y_s, y_c = rpy_to_sin_cos(roll, pitch, yaw)
    w.writerow([
        f"{tsec:.6f}",
        f"{p.x:.6f}", f"{p.y:.6f}", f"{p.z:.6f}",
        f"{r_s:.6f}", f"{r_c:.6f}",
        f"{p_s:.6f}", f"{p_c:.6f}",
        f"{y_s:.6f}", f"{y_c:.6f}",
    ])

# ---------- main ----------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: python3 simple_rosbag_export.py <path_to_bag>\n")
        sys.exit(1)

    bag_path = sys.argv[1]
    if not os.path.isfile(bag_path):
        sys.stderr.write(f"ERROR: Bag file not found: {bag_path}\n")
        sys.exit(3)

    outdir = make_outdir_from_bag(bag_path)

    # Fixed topics (odom)
    TOPIC_OD1 = "/uav1/estimation_manager/odom_main"
    TOPIC_OD2 = "/uav2/estimation_manager/odom_main"

    # UV-DAR topic: prefer filtered, else measured
    UVDAR_CAND = ["/uav1/uvdar/filteredPoses", "/uav1/uvdar/measuredPoses"]
    uvdar_topic = None
    try:
        with rosbag.Bag(bag_path) as bp:
            topics = bp.get_type_and_topic_info()[1]
            for cand in UVDAR_CAND:
                if cand in topics:
                    uvdar_topic = cand
                    break
    except Exception:
        pass
    if uvdar_topic is None:
        # If not found, still try filtered (most common) to avoid silent nothing
        uvdar_topic = UVDAR_CAND[0]

    # Create CSVs
    try:
        f_est = open(os.path.join(outdir, "estimations.csv"), "w", newline="")
        f_o1  = open(os.path.join(outdir, "odom1.csv"), "w", newline="")
        f_o2  = open(os.path.join(outdir, "odom2.csv"), "w", newline="")
    except PermissionError:
        sys.stderr.write(f"ERROR: Permission denied creating CSVs in: {outdir}\n")
        sys.exit(4)

    w_est, w_o1, w_o2 = csv.writer(f_est), csv.writer(f_o1), csv.writer(f_o2)
    write_header(w_est); write_header(w_o1); write_header(w_o2)

    topic_list = [uvdar_topic, TOPIC_OD1, TOPIC_OD2]

    try:
        with rosbag.Bag(bag_path) as bag:
            for topic, msg, t in bag.read_messages(topics=topic_list):
                tsec = t.to_sec()

                if topic == TOPIC_OD1 and msg._type == "nav_msgs/Odometry":
                    write_row(w_o1, tsec, msg.pose.pose.position, msg.pose.pose.orientation)

                elif topic == TOPIC_OD2 and msg._type == "nav_msgs/Odometry":
                    write_row(w_o2, tsec, msg.pose.pose.position, msg.pose.pose.orientation)

                elif topic == uvdar_topic:
                    # Expect mrs_msgs/PoseWithCovarianceArrayStamped
                    if msg._type == "mrs_msgs/PoseWithCovarianceArrayStamped":
                        if hasattr(msg, "poses") and len(msg.poses) > 0 and hasattr(msg.poses[0], "pose"):
                            pose = msg.poses[0].pose
                            write_row(w_est, tsec, pose.position, pose.orientation)
                        # else: empty poses array – skip silently
                    # If some bags still contain PoseArray, keep a tiny fallback:
                    elif msg._type == "geometry_msgs/PoseArray":
                        if hasattr(msg, "poses") and len(msg.poses) > 0:
                            pose = msg.poses[0]
                            write_row(w_est, tsec, pose.position, pose.orientation)
                    # else: ignore other types on this topic

    except KeyboardInterrupt:
        pass
    finally:
        f_est.close(); f_o1.close(); f_o2.close()

if __name__ == "__main__":
    main()
