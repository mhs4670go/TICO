#!/usr/bin/env bash
set -u
cd /workspace/TICO
source .venv/bin/activate
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$RESULT_ROOT/run_command.py" --name format_all_retry --timeout 3600 -- ./ccex format --no-apply-patches
