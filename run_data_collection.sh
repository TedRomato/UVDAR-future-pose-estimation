#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset> <flight_number>"
  exit 1
fi

DATASET="$1"
FLIGHT="$2"

BAG_DIR="$HOME/data/$DATASET/bags/$FLIGHT"
mkdir -p "$BAG_DIR"


# Ensure watcher script is available (adjust path if needed)
WATCHER="$(dirname "$0")/watch_uav_stop.py"
if [ ! -x "$WATCHER" ]; then
  echo "Watcher not found or not executable: $WATCHER"
  exit 2
fi

echo "[1/8] /uav1: goto altitude 5 m"
rosservice call /uav1/control_manager/goto_altitude 5

echo "[2/8] Launching trajectory loader for uav2"
roslaunch mrs_uav_trajectory_loader single_uav.launch uav_name:="uav2" &
LAUNCH_PID=$!

cleanup() {
  if ps -p $LAUNCH_PID >/dev/null 2>&1; then
    echo "Stopping trajectory loader (pid $LAUNCH_PID)"
    kill $LAUNCH_PID || true
  fi
}
trap cleanup EXIT

echo "[3/8] Waiting for trajectory loader..."
sleep 10


echo "[4/8] /uav2: goto trajectory start"
rosservice call /uav2/control_manager/goto_trajectory_start

echo "       Sleeping 10 s to settle"
sleep 10

echo "[5/8] Disable collision avoidance (uav2, then uav1)"
rosservice call /uav2/control_manager/mpc_tracker/collision_avoidance "data: false"
rosservice call /uav1/control_manager/mpc_tracker/collision_avoidance "data: false"

echo "[6/8] /uav2: start trajectory tracking"
rosservice call /uav2/control_manager/start_trajectory_tracking

echo "[7/8] Start rosbag record -> $BAG_DIR/flight.bag"
rosbag record -O "$BAG_DIR/flight.bag" \
  /uav2/estimation_manager/odom_main \
  /uav1/estimation_manager/odom_main \
  /uav1/uvdar/measuredPoses &
BAG_PID=$!

# Give rosbag a moment to spin up
sleep 1

echo "[8/8] Watching /uav2 odometry for stop condition..."
python3 "$WATCHER" \
  _topic:=/uav2/estimation_manager/odom_main \
  _speed_thresh:=0.05 \
  _hold_time:=5.0 \
  _max_wait:=0.0 || true

echo "UAV stop detected (or max wait reached). Stopping rosbag…"
if ps -p $BAG_PID >/dev/null 2>&1; then
  kill -INT $BAG_PID
  wait $BAG_PID || true
fi

echo "Flight complete. Bag saved at: $BAG_DIR/flight.bag"
