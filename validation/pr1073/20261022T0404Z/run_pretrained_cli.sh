#!/usr/bin/env bash
set -u
cd /workspace/TICO
source .venv/bin/activate
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$RESULT_ROOT/run_command.py" --name pretrained_fp_cli_ppl --timeout 600 -- python -m tico.quantization.examples.evaluate --config "$RESULT_ROOT/pretrained_eval.yaml" --tasks ppl --set 'calibration.seq_len=32' --set 'export.max_seq_len=16' --set 'evaluation.max_seq_len=128' --set 'evaluation.perplexity.max_seq_len=64' --set 'evaluation.perplexity.stride=32' --set 'evaluation.perplexity.split=test[:12]' --device cpu --no-progress --output-dir "$RESULT_ROOT/pretrained_fp_cli"
fp_status=$?
python "$RESULT_ROOT/run_command.py" --name pretrained_quant_cli_ppl --timeout 600 -- python -m tico.quantization.examples.evaluate --config "$RESULT_ROOT/pretrained_eval.yaml" --checkpoint "$RESULT_ROOT/pretrained_tinyllama_quantized.pt" --tasks ppl --set 'calibration.seq_len=32' --set 'export.max_seq_len=16' --set 'evaluation.max_seq_len=128' --set 'evaluation.perplexity.max_seq_len=64' --set 'evaluation.perplexity.stride=32' --set 'evaluation.perplexity.split=test[:12]' --device cpu --no-progress --output-dir "$RESULT_ROOT/pretrained_quant_cli"
quant_status=$?
printf 'FP_CLI_EXIT=%s QUANT_CLI_EXIT=%s\n' "$fp_status" "$quant_status"
if test "$fp_status" -ne 0 || test "$quant_status" -ne 0; then exit 1; fi
