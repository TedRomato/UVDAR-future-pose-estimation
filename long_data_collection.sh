#!/usr/bin/env bash


RUNTIME="${1:-7200}"
BUFFER="${2:-240}"
TOTAL_TIME=$((RUNTIME + BUFFER))

echo "DONT FORGET TO UPDATE session.yml with the same RUNTIME and BUFFER values!"

while true; do
  START_TIME=$(date '+%F %T')
  EXPECTED_END=$(date -d "+${TOTAL_TIME} seconds" '+%F %T')

  echo "[$START_TIME] Loop started"
  echo "[$START_TIME] Expected to end at $EXPECTED_END (runtime=${RUNTIME}s + buffer=${BUFFER}s)"

  ./start_simulation.sh &
  PID=$!

  sleep "$TOTAL_TIME"

  END_TIME=$(date '+%F %T')
  echo "[$END_TIME] Loop ended, killing process (PID=$PID)"

  kill "$PID" 2>/dev/null
done
