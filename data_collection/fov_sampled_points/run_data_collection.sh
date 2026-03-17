#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <duration_seconds>"
  exit 1
fi

DURATION="$1"

# ------------------ AUTO-GENERATED FLIGHT NAME ------------------
TIMESTAMP="$(date '+%Y_%m_%d__%H_%M')"
FLIGHT="flight_${TIMESTAMP}"
# ---------------------------------------------------------------

# ------------------ CONFIGURABLE BAG SPLIT SIZE ------------------
# Max size of each bag file in GiB
MAX_BAG_SIZE_GB=1
# Convert to MiB for rosbag --size (1 GiB = 1024 MiB)
MAX_BAG_SIZE_MB=$((MAX_BAG_SIZE_GB * 1024))
# -----------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="$SCRIPT_DIR/random_flight_node.py"
CFG="$SCRIPT_DIR/fov_flight.yaml"

BAG_DIR="$HOME/data/bags"
mkdir -p "$BAG_DIR"

# Load altitude from C: [x, y, z] in fov_flight.yaml
ALTITUDE=$(
  grep -E '^C:' "$CFG" \
  | sed -E 's/.*\[([^]]+)\].*/\1/' \
  | awk -F',' '{gsub(/ /,"",$3); print $3}'
)

if [ -z "$ALTITUDE" ]; then
  echo "Error: could not read altitude from $CFG (C: [x, y, z])"
  exit 1
fi

echo "[1/6] /uav1: goto altitude ${ALTITUDE} m"
rosservice call /uav1/control_manager/goto_altitude "$ALTITUDE"
sleep 3

echo "[2/6] Disable safety (uav2, then uav1)"
rosservice call /uav1/control_manager/mpc_tracker/collision_avoidance "data: false"
rosservice call /uav2/control_manager/mpc_tracker/collision_avoidance "data: false"
rosservice call /uav2/control_manager/use_safety_area false

echo "[3/6] Starting random_flight_node with $CFG"
python3 "$NODE" _fov_cfg:="$CFG" _initial_delay:=15.0 &
RAND_PID=$!

echo "[4/6] Start rosbag record -> $BAG_DIR/${FLIGHT}.bag (split every ${MAX_BAG_SIZE_GB} GiB)"
rosbag record \
  --split \
  --size="${MAX_BAG_SIZE_MB}" \
  -O "$BAG_DIR/$FLIGHT" \
  /uav2/estimation_manager/odom_main \
  /uav1/estimation_manager/odom_main \
  /uav1/uvdar/points_seen_right \
  /uav1/uvdar/blinkers_seen_right \
  /uav1/uvdar/measuredPoses \
  /tf_static \
  /tf \
  /gazebo/uav1/uvdar_bluefox_right/image_raw&
BAG_PID=$!

echo "[5/6] Running random flight for $DURATION seconds..."
sleep "$DURATION"

echo "[6/6] Time limit reached. Stopping processes..."
kill -INT "$BAG_PID" 2>/dev/null || true
kill "$RAND_PID" 2>/dev/null || true

echo "Done."
echo "Flight name : $FLIGHT"
echo "Bags saved  : $BAG_DIR/${FLIGHT}.bag (each up to ~${MAX_BAG_SIZE_GB} GiB)"
