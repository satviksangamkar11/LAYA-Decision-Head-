#!/bin/bash
# launch one detached v2 phase run as a transient systemd service (survives the caller); usage: launch.sh <unit> <iid>
OUT="/mnt/d/local model/results/raw/sandbox-pilot-v2-20261014T210515Z"
systemd-run --unit="$1" --collect --working-directory="/mnt/d/local model" -p StandardOutput="append:$OUT/run-$1.out" -p StandardError="append:$OUT/run-$1.out" \
  python3 "/mnt/d/local model/tools/sandbox_pilot_v2.py" run "$OUT" "$2"
