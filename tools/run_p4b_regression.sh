#!/usr/bin/env bash
# [P4B] One pass of the CPU regression suites after wiring contracts into the controller. Usage: bash tools/run_p4b_regression.sh <new out dir>
set -u
R="$1"; mkdir -p "$R"
for t in controller/tests/run_tests.py controller/tests/phaseA_tests.py controller/tests/phaseB_tests.py controller/tests/phaseD1_tests.py controller/tests/phaseF1_tests.py \
         controller/tests/phaseF1b_tests.py controller/tests/phaseF2_tests.py controller/tests/phaseF2b_tests.py controller/tests/phaseF2c_tests.py \
         orchestrator/tests/run_tests.py orchestrator/tests/phaseC_tests.py orchestrator/tests/phaseF3_tests.py orchestrator/tests/phaseF3b_tests.py \
         orchestrator/tests/phaseF3c_tests.py controller/tests/phaseMXC_tests.py; do
  n=$(echo "$t" | tr '/' '_')
  PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 timeout 1500 .venv/Scripts/python.exe "evidence_ladder/$t" > "$R/$n.log" 2>&1
  echo "$t exit=$?" | tee -a "$R/SUMMARY.txt"
done
