#!/usr/bin/env bash
cd /workspace/TICO
source .venv/bin/activate
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$RESULT_ROOT/run_command.py" --name llama_rope --timeout 900 -- python tico/quantization/recipes/debug/verify_llama_rope.py --repo . --report "$RESULT_ROOT/llama-rope-checks.json"
