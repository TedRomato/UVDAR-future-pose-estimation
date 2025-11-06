#!/usr/bin/env python3
import os, math, yaml, subprocess
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from cameraFOV import PyramidFOV, SamplingConfig

class RandomFOVGoto:
    def __init__(self):
        # --- fixed defaults ---
        self.odom_topic      = "/uav2/estimation_manager/odom_main"
        self.uav1_goto_srv   = "/uav1/control_manager/goto"  # service
        self.uav2_goto_srv   = "/uav2/control_manager/goto"  # service
        self.initial_delay   = 10.0
        self.target_radius   = 0.1     # meters
        self.cfg_path        = os.path.join(os.path.dirname(__file__), "fov_flight.yaml")

        # sampling defaults (can be overridden in YAML)
        dist_min, dist_max, min_z = 5.0, 30.0, 1.0

        # --- load YAML ---
        if not os.path.isfile(self.cfg_path):
            rospy.logfatal(f"[random_fov_goto] Missing config file: {self.cfg_path}")
            raise SystemExit(1)
        with open(self.cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}

        required = ["C", "P_tl", "P_br"]
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            rospy.logfatal(f"[random_fov_goto] Missing keys in cfg: {', '.join(missing)}")
            raise SystemExit(1)

        C        = np.array(data["C"], float)
        P_tl_rel = np.array(data["P_tl"], float)   # relative to C
        P_br_rel = np.array(data["P_br"], float)   # relative to C
        P_tl = C + P_tl_rel
        P_br = C + P_br_rel

        dist_min      = float(data.get("distance_min", dist_min))
        dist_max      = float(data.get("distance_max", dist_max))
        min_z         = float(data.get("min_z",       min_z))
        self.initial_delay = float(data.get("initial_delay", self.initial_delay))
        self.target_radius = float(data.get("target_radius", self.target_radius))

        # --- FOV + sampling ---
        self.fov  = PyramidFOV.from_2_edge_points(C, P_tl, P_br, up_hint=np.array([0.0, 0.0, 1.0]))
        self.samp = SamplingConfig(distance_min=dist_min, distance_max=dist_max, min_z=min_z)

        # --- state ---
        self.current_target = None

        rospy.loginfo(f"[random_fov_goto] Ready | cfg={self.cfg_path} | target_radius={self.target_radius} m")

        # --- move observer (UAV1) ---
        self._goto_service(self.uav1_goto_srv, C[0], C[1], C[2], 0.0)
        rospy.loginfo(f"uav1 → C: [{C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}]")

        rospy.sleep(self.initial_delay)

        # --- send first random target ---
        self._send_random_to_uav2()

        # subscribe for continuous triggering
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom)

    # --- helper: call /control_manager/goto service ---
    def _goto_service(self, srv_name: str, x: float, y: float, z: float, pitch: float):
        try:
            rospy.wait_for_service(srv_name, timeout=5.0)
            cmd = ["rosservice", "call", srv_name, f"[{x:.6f}, {y:.6f}, {z:.6f}, {pitch:.6f}]"]
            subprocess.check_call(cmd)
        except Exception as e:
            rospy.logwarn(f"[random_fov_goto] Service call failed: {e}")

    def _send_random_to_uav2(self):
        P = self.fov.sample_point(self.samp)
        if P is None or len(P) == 0:
            rospy.logwarn("[random_fov_goto] Failed to sample target.")
            return
        self.current_target = np.array(P, float)
        self._goto_service(self.uav2_goto_srv, P[0], P[1], P[2], 0.0)
        rospy.loginfo(f"uav2 → random: [{P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f}]")

    # --- odometry callback: trigger next target when close enough ---
    def cb_odom(self, msg: Odometry):
        if self.current_target is None:
            return

        pos = msg.pose.pose.position
        dx = pos.x - self.current_target[0]
        dy = pos.y - self.current_target[1]
        dz = pos.z - self.current_target[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist <= self.target_radius:
            self._send_random_to_uav2()


if __name__ == "__main__":
    rospy.init_node("random_fov_goto")
    RandomFOVGoto()
    rospy.spin()
