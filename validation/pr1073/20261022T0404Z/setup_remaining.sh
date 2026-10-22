#!/usr/bin/env bash
set -u
cd /workspace/TICO
source .venv/bin/activate
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$RESULT_ROOT/run_command.py" --name configure_format --timeout 900 -- ./ccex configure format
python "$RESULT_ROOT/run_command.py" --name environment -- python "$RESULT_ROOT/environment_info.py"
python "$RESULT_ROOT/run_command.py" --name cli_help -- bash -c 'python tico/quantization/recipes/debug/verify_llama_rope.py --help; python tico/quantization/recipes/debug/export_regression.py --help; python tico/quantization/recipes/debug/export_regression.py run --help; python tico/quantization/recipes/debug/export_regression.py isolation --help; python -m tico.quantization.examples.evaluate --help; ./ccex test --help; ./ccex format --help'
