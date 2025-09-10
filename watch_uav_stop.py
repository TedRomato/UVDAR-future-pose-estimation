#!/usr/bin/env python3
import rospy, math, time
from nav_msgs.msg import Odometry

class StopWatcher:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/uav2/estimation_manager/odom_main")
        self.speed_thresh = float(rospy.get_param("~speed_thresh", 0.05))   # m/s
        self.hold_time = float(rospy.get_param("~hold_time", 5.0))          # seconds below thresh
        self.max_wait = float(rospy.get_param("~max_wait", 0.0))            # 0 = no limit
        self.use_twist_only = bool(rospy.get_param("~use_twist_only", False))
        self.last_pos = None  # (x,y,z,t)
        self.last_below_since = None
        self.start_wall = time.time()
        rospy.Subscriber(self.topic, Odometry, self.cb)

    def speed_from_pose(self, p, t):
        if self.last_pos is None:
            self.last_pos = (p.x, p.y, p.z, t)
            return None
        x0, y0, z0, t0 = self.last_pos
        dt = t - t0
        self.last_pos = (p.x, p.y, p.z, t)
        if dt <= 0.0:
            return None
        dx, dy, dz = p.x - x0, p.y - y0, p.z - z0
        return math.sqrt(dx*dx + dy*dy + dz*dz) / dt

    def cb(self, msg: Odometry):
        t = msg.header.stamp.to_sec()
        v = msg.twist.twist.linear
        # Prefer twist if available and not all zeros, unless user forced pose diff
        if not self.use_twist_only and (abs(v.x)+abs(v.y)+abs(v.z) < 1e-6):
            spd = self.speed_from_pose(msg.pose.pose.position, t)
            if spd is None:
                return
        else:
            spd = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

        now = time.time()
        if spd < self.speed_thresh:
            if self.last_below_since is None:
                self.last_below_since = now
            # stayed below long enough?
            if now - self.last_below_since >= self.hold_time:
                rospy.signal_shutdown("Stopped long enough")
        else:
            self.last_below_since = None

        if self.max_wait > 0.0 and (now - self.start_wall) >= self.max_wait:
            rospy.signal_shutdown("Max wait reached")

if __name__ == "__main__":
    rospy.init_node("watch_uav_stop")
    StopWatcher()
    rospy.spin()
