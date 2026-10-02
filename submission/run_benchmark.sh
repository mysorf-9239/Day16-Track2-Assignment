#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/ml-benchmark"
source /opt/lab16-venv/bin/activate
export LAB_PROJECT_ID=project-b254970d-4ec2-4509-920
export LAB_ZONE=us-central1-a
export LAB_MACHINE_TYPE=e2-medium
export OMP_NUM_THREADS=2

{
  date -u
  hostname
  uname -m
  free -h
  ip -s link
} > resource_before.txt

python3 -u benchmark.py > benchmark_output.txt 2>&1 &
benchmark_pid=$!
top -b -d 1 -p "$benchmark_pid" > resource_during.txt &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
if wait "$benchmark_pid"; then
  benchmark_status=0
else
  benchmark_status=$?
fi
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT

{
  date -u
  free -h
  ip -s link
} > resource_after.txt
cat benchmark_output.txt
exit "$benchmark_status"
