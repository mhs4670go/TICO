#!/usr/bin/env bash
set -euo pipefail
EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_ROOT="$(dirname -- "$EVAL_ROOT")"
TICO_REPO="${TICO_REPO:-${CANDIDATE_REPO}}"
export PATH="${VALIDATION_VENV}/bin:$PATH"
export PYTHONPATH="$TICO_REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference
cd "$TICO_REPO"
python "$VALIDATION_ROOT/run.py" --name evaluation-reproduce-prepare --timeout 180 -- python "$EVAL_ROOT/prepare.py"
for target in ppl lm_eval; do
  for mode in fp quant; do
    target_alias="$target"
    if [[ "$target" == lm_eval ]]; then target_alias=mc; fi
    label="reproduce-${mode}-${target_alias}"
    args=(--config "$EVAL_ROOT/config.yaml" --device cpu --tasks "$target" --output-dir "$EVAL_ROOT/$label/protocol" --no-progress)
    if [[ "$mode" == quant ]]; then args+=(--checkpoint "$EVAL_ROOT/quantized_model.pt"); fi
    python "$VALIDATION_ROOT/run.py" --name "$label" --timeout 180 -- python "$EVAL_ROOT/invoke_cli.py" --label "$label" -- "${args[@]}"
  done
done
python "$VALIDATION_ROOT/run.py" --name reproduce-evaluation-verify --timeout 120 -- python "$EVAL_ROOT/verify_results.py" --prefix reproduce
