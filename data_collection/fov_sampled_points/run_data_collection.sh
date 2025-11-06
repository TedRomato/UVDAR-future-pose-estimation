#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="$SCRIPT_DIR/random_flight_node.py"
CFG="$SCRIPT_DIR/fov_flight.yaml"

if [ ! -f "$NODE" ]; then
  echo "Error: random_flight_node.py not found in $SCRIPT_DIR"
  exit 1
fi

echo "[1/2] Disable collision avoidance (uav2, then uav1)"
rosservice call /uav2/control_manager/mpc_tracker/collision_avoidance "data: false"
rosservice call /uav1/control_manager/mpc_tracker/collision_avoidance "data: false"


echo "[2/2] Starting random_flight_node node with $CFG"
python3 "$NODE" _fov_cfg:="$CFG" _initial_delay:=15.0
