#!/usr/bin/env python3
"""
Real-drone node: flies the OBSERVER to the camera pose (C, C_heading)
defined in fov_flight.yaml and holds position.

UAV name resolution (highest priority first):
  1. UAV_NAME environment variable  (set by mrs_docker stack.env)
  2. observer_uav_name in fov_flight.yaml
  3. fallback: "uav1"

Usage (on the observer's onboard computer):
    python3 observer_node.py
"""
import os, yaml, subprocess
import numpy as np
import rospy


class ObserverGoto:
    def __init__(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "fov_flight.yaml")
        if not os.path.isfile(cfg_path):
            rospy.logfatal(f"[observer] Missing config: {cfg_path}")
            raise SystemExit(1)

        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # --- required ---
        if "C" not in data or data["C"] is None:
            rospy.logfatal("[observer] 'C' not found in config")
            raise SystemExit(1)

        C       = np.array(data["C"], float)
        heading = float(data.get("C_heading", 0.0))

        # UAV name: env var > YAML > default
        uav = os.environ.get("UAV_NAME", str(data.get("observer_uav_name", "uav1")))

        goto_srv = f"/{uav}/control_manager/goto"

        rospy.loginfo(f"[observer] Sending {uav} → C: [{C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}], hdg={heading:.2f} rad")
        self._goto_service(goto_srv, C[0], C[1], C[2], heading)
        rospy.loginfo(f"[observer] {uav} goto command sent. Holding position.")

    @staticmethod
    def _goto_service(srv_name: str, x: float, y: float, z: float, heading: float):
        try:
            rospy.wait_for_service(srv_name, timeout=10.0)
            cmd = ["rosservice", "call", srv_name, f"[{x:.6f}, {y:.6f}, {z:.6f}, {heading:.6f}]"]
            subprocess.check_call(cmd)
        except Exception as e:
            rospy.logerr(f"[observer] Service call failed: {e}")


if __name__ == "__main__":
    rospy.init_node("fov_observer")
    ObserverGoto()
    rospy.spin()
