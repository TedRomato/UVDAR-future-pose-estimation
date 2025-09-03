#!/usr/bin/env python3
# rosbag_export_xyzh_multi.py
import argparse, csv, math, os, re
import rosbag

# Try TF for robust quaternion->yaw conversion
try:
    from tf.transformations import euler_from_quaternion
except Exception:
    def euler_from_quaternion(q):
        x, y, z, w = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (0.0, 0.0, yaw)

def quat_to_yaw(qx, qy, qz, qw):
    return euler_from_quaternion([qx, qy, qz, qw])[2]

def norm_angle(a, mode):
    if mode == "pm_pi":
        while a > math.pi:  a -= 2*math.pi
        while a <= -math.pi: a += 2*math.pi
        return a
    # default 0..2pi
    a = a % (2*math.pi)
    return a if a >= 0 else a + 2*math.pi

def sanitize(name: str) -> str:
    # make a safe filename from a ROS topic
    s = re.sub(r'^\/+', '', name)      # drop leading slashes
    s = s.replace('/', '_')
    s = re.sub(r'[^A-Za-z0-9_\-\.]', '_', s)
    return s

def main():
    ap = argparse.ArgumentParser(description="Export x,y,z,heading from rosbag topics to CSV.")
    ap.add_argument("--bag", required=True, help="Path to .bag")
    ap.add_argument("--topics", required=True, nargs="+",
                    help="Topics to export (space-separated).")
    ap.add_argument("--outdir", default="bag_export", help="Output directory")
    ap.add_argument("--heading-source", choices=["auto","orientation","motion"],
                    default="auto", help="Where to take heading from")
    ap.add_argument("--heading-range", choices=["0_2pi","pm_pi"],
                    default="0_2pi", help="Angle normalization")
    ap.add_argument("--min-speed", type=float, default=1e-6,
                    help="Min XY speed (m/s) to compute motion heading")
    ap.add_argument("--no-header", action="store_true",
                    help="Do not write headers (applies to PoseArray too)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bag = rosbag.Bag(args.bag)
    # We’ll open a writer per topic on first encounter
    writers = {}
    files = {}
    # For motion heading we track previous x,y,t per (topic) or (topic,pose_index)
    prev_state = {}

    def get_writer(topic, is_array=False):
        if topic in writers:
            return writers[topic]
        fname = os.path.join(args.outdir, sanitize(topic) + ".csv")
        f = open(fname, "w", newline="")
        files[topic] = f
        w = csv.writer(f)
        if not args.no_header:
            if is_array:
                w.writerow(["time", "x", "y", "z", "heading", "pose_index"])
            else:
                w.writerow(["time", "x", "y", "z", "heading"])
        writers[topic] = w
        return w

    def write_row(topic, tsec, x, y, z, heading, idx=None):
        heading = norm_angle(heading, args.heading_range)
        w = get_writer(topic, is_array=(idx is not None))
        if idx is None:
            w.writerow([f"{tsec:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{heading:.6f}"])
        else:
            w.writerow([f"{tsec:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{heading:.6f}", idx])

    def motion_heading(key, x, y, tsec):
        px, py, pt = prev_state.get(key, (None, None, None))
        if px is None or pt is None:
            prev_state[key] = (x, y, tsec)
            return None
        dt = tsec - pt
        prev_state[key] = (x, y, tsec)
        if dt <= 0:
            return None
        vx, vy = (x - px)/dt, (y - py)/dt
        if math.hypot(vx, vy) < args.min_speed:
            return None
        return math.atan2(vy, vx)

    print("Exporting…")
    count = {t:0 for t in args.topics}

    for topic, msg, t in bag.read_messages(topics=args.topics):
        tsec = t.to_sec()

        # --- Odometry ---
        if msg._type == "nav_msgs/Odometry":
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            q = msg.pose.pose.orientation
            heading = None

            if args.heading_source in ("orientation","auto"):
                heading = quat_to_yaw(q.x, q.y, q.z, q.w)

            if heading is None and args.heading_source in ("motion","auto"):
                heading = motion_heading((topic, None), x, y, tsec)
            if heading is None:
                heading = 0.0

            write_row(topic, tsec, x, y, z, heading)
            count[topic] += 1
            continue

        # --- PoseStamped ---
        if msg._type == "geometry_msgs/PoseStamped":
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            q = msg.pose.orientation
            heading = None

            if args.heading_source in ("orientation","auto"):
                heading = quat_to_yaw(q.x, q.y, q.z, q.w)

            if heading is None and args.heading_source in ("motion","auto"):
                heading = motion_heading((topic, None), x, y, tsec)
            if heading is None:
                heading = 0.0

            write_row(topic, tsec, x, y, z, heading)
            count[topic] += 1
            continue

        # --- PoseWithCovarianceStamped ---
        if msg._type == "geometry_msgs/PoseWithCovarianceStamped":
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            q = msg.pose.pose.orientation
            heading = None

            if args.heading_source in ("orientation","auto"):
                heading = quat_to_yaw(q.x, q.y, q.z, q.w)

            if heading is None and args.heading_source in ("motion","auto"):
                heading = motion_heading((topic, None), x, y, tsec)
            if heading is None:
                heading = 0.0

            write_row(topic, tsec, x, y, z, heading)
            count[topic] += 1
            continue

        # --- PoseArray (e.g., /uav1/uvdar/filteredPoses) ---
        if msg._type == "geometry_msgs/PoseArray":
            # Many poses, same timestamp
            for idx, p in enumerate(msg.poses):
                x = p.position.x
                y = p.position.y
                z = p.position.z
                q = p.orientation
                heading = None
                if args.heading_source in ("orientation","auto"):
                    # If quaternion is identity (0,0,0,0/1), this still works
                    heading = quat_to_yaw(q.x, q.y, q.z, q.w)
                if heading is None and args.heading_source in ("motion","auto"):
                    heading = motion_heading((topic, idx), x, y, tsec)
                if heading is None:
                    heading = 0.0
                write_row(topic, tsec, x, y, z, heading, idx=idx)
                count[topic] += 1
            continue

        # Otherwise: ignore other types on these topics
        # (Add more handlers if needed.)

    bag.close()
    for f in files.values():
        f.close()

    for tpc in args.topics:
        print(f"{tpc}: {count.get(tpc,0)} rows -> {os.path.join(args.outdir, sanitize(tpc)+'.csv')}")

if __name__ == "__main__":
    main()
