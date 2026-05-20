#!/usr/bin/env python3
"""
Real-drone node: samples random points inside the camera FOV frustum
and commands the FLIER drone to visit them one by one.

Start this node AFTER the observer is already in position.

UAV name resolution (highest priority first):
  1. UAV_NAME environment variable  (set by mrs_docker stack.env)
  2. flier_uav_name in fov_flight.yaml
  3. fallback: "uav2"

Usage (on the flier's onboard computer):
    python3 random_flier_node.py
"""
import os, math, yaml, subprocess
import time
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from cameraFOV import PyramidFOV, SamplingConfig


class RandomFlier:
    def __init__(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "fov_flight.yaml")
        if not os.path.isfile(cfg_path):
            rospy.logfatal(f"[flier] Missing config: {cfg_path}")
            raise SystemExit(1)

        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # --- validate required keys ---
        required = ["C", "P_tl", "P_br"]
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            rospy.logfatal(f"[flier] Missing keys in config: {', '.join(missing)}")
            raise SystemExit(1)

        C    = np.array(data["C"], float)
        P_tl = np.array(data["P_tl"], float)
        P_br = np.array(data["P_br"], float)

        dist_min = float(data.get("distance_min", 5.0))
        dist_max = float(data.get("distance_max", 30.0))
        min_z    = float(data.get("min_z", 1.0))

        self.target_radius  = float(data.get("target_radius", 0.25))
        self.stop_speed_thr = 0.05  # m/s
        self.last_target_time = rospy.Time.now()
        time.sleep(1) 
        self.min_dt_between_targets = rospy.Duration(1)

        # UAV name: env var > YAML > default
        uav = os.environ.get("UAV_NAME", str(data.get("flier_uav_name", "uav14")))
        self.goto_srv  = f"/{uav}/control_manager/goto"
        self.odom_topic = f"/{uav}/estimation_manager/odom_main"

        # --- FOV geometry + sampling config ---
        self.fov  = PyramidFOV.from_2_edge_points(C, P_tl, P_br, up_hint=np.array([0.0, 0.0, 1.0]))
        self.samp = SamplingConfig(distance_min=dist_min, distance_max=dist_max, min_z=min_z)

        # --- state ---
        self.current_target = None
        self.uav_name = uav

        rospy.loginfo(f"[flier] Ready | uav={uav} | target_radius={self.target_radius} m")

        # --- send first random target immediately ---
        self._send_random_target()

        # --- subscribe for continuous triggering ---
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom)

    # ---- goto service helper ----
    @staticmethod
    def _goto_service(srv_name: str, x: float, y: float, z: float, heading: float):
        try:
            rospy.wait_for_service(srv_name, timeout=5.0)
            cmd = ["rosservice", "call", srv_name, f"[{x:.6f}, {y:.6f}, {z:.6f}, {heading:.6f}]"]
            subprocess.check_call(cmd)
        except Exception as e:
            rospy.logwarn(f"[flier] Service call failed: {e}")

    # ---- sample & send ----
    def _send_random_target(self):
        if (rospy.Time.now() - self.last_target_time) < self.min_dt_between_targets:
            return

        P = self.fov.sample_point(self.samp)
        if P is None or len(P) == 0:
            rospy.logwarn("[flier] Failed to sample target.")
            return

        self.current_target = np.array(P, float)
        self._goto_service(self.goto_srv, P[0], P[1], P[2], 0.0)
        rospy.loginfo(f"{self.uav_name} → random: [{P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f}]")
        self.last_target_time = rospy.Time.now()

    # ---- odometry callback: trigger next target when close / stopped ----
    def cb_odom(self, msg: Odometry):
        if self.current_target is None:
            return

        pos = msg.pose.pose.position
        dx = pos.x - self.current_target[0]
        dy = pos.y - self.current_target[1]
        dz = pos.z - self.current_target[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        vel = msg.twist.twist.linear
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)

        if dist <= self.target_radius or speed <= self.stop_speed_thr:
            self._send_random_target()


if __name__ == "__main__":
    rospy.init_node("fov_random_flier")
    RandomFlier()
    rospy.spin()
