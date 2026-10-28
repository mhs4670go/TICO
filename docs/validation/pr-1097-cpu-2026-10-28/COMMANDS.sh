#!/usr/bin/env bash
# Exact recorded commands for review; output paths already exist. Use REPRODUCE.md for a fresh run.

# assistant-capacity16-circle-base | FAIL | exit=1 | 5.027s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/gemma-export-review/assistant_capacity16.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-base --report-json ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-base.json)

# assistant-capacity16-circle-r2-base | FAIL | exit=1 | 15.668s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/gemma-export-review/assistant_capacity16.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-r2-base --report-json ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-r2-base.json)

# assistant-capacity16-circle-r2 | FAIL | exit=1 | 16.883s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/gemma-export-review/assistant_capacity16.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-r2 --report-json ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16-r2.json)

# assistant-capacity16-circle | FAIL | exit=1 | 5.078s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/gemma-export-review/assistant_capacity16.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16 --report-json ${VALIDATION_ROOT}/gemma-export-review/assistant-capacity16.json)

# baseline-cpu-fixtures | PASS | exit=0 | 5.528s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.cpu_validation --stage _fixtures --output ${VALIDATION_ROOT}/baseline-cpu-fixtures --seed 1073)

# baseline-cpu-llama-16 | FAIL | exit=1 | 59.398s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.export_regression isolation --candidate-repo ${BASELINE_REPO} --config ${VALIDATION_ROOT}/baseline-cpu-fixtures/llama.json --checkpoint ${VALIDATION_ROOT}/baseline-cpu-fixtures/llama.pt --seq-len 16 --eager-seq-len 48 --decode-steps 4 --seed 1073 --threads 1 --output ${VALIDATION_ROOT}/baseline-cpu-comparison/llama-16 --require-torch-parity --artifacts circle_bundle,circle_full)

# baseline-cpu-qwen3_vl-16-connected | FAIL | exit=1 | 5.026s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.cpu_validation --stage _connected --checkpoint ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.pt --artifacts ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-16/candidate/before/artifacts --output ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-16-connected --capacity 16)

# baseline-cpu-qwen3_vl-16 | FAIL | exit=1 | 56.972s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.export_regression isolation --candidate-repo ${BASELINE_REPO} --config ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.json --checkpoint ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.pt --seq-len 16 --eager-seq-len 48 --decode-steps 4 --seed 1073 --threads 1 --output ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-16 --require-torch-parity)

# baseline-cpu-qwen3_vl-32-connected | FAIL | exit=1 | 5.276s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.cpu_validation --stage _connected --checkpoint ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.pt --artifacts ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-32/candidate/before/artifacts --output ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-32-connected --capacity 32)

# baseline-cpu-qwen3_vl-32 | FAIL | exit=1 | 56.09s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/checked_module.py ${BASELINE_REPO} tico.quantization.recipes.debug.export_regression isolation --candidate-repo ${BASELINE_REPO} --config ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.json --checkpoint ${VALIDATION_ROOT}/baseline-cpu-fixtures/qwen3_vl.pt --seq-len 32 --eager-seq-len 48 --decode-steps 4 --seed 1073 --threads 1 --output ${VALIDATION_ROOT}/baseline-cpu-comparison/qwen3_vl-32 --require-torch-parity)

# baseline-selective | PASS | exit=0 | 8.132s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/baseline_selective.py)

# circle-optimizer | PASS | exit=0 | 1.468s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m test.performance.benchmark_circle_optimizer ${VALIDATION_ROOT}/wrapper-circle/llama_attention_decode/llama_attention_decode.q.circle --repeat 1 --warmup 0 --json)

# conversion-help | PASS | exit=0 | 0.064s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m test.performance.benchmark_conversion_memory --help)

# conversion-module | PASS | exit=0 | 8.588s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m test.performance.benchmark_conversion_memory --sizes-mib 1 --tables 1 --width 16 --threads 1 --repeat 1 --optimization O1 --input-kind module --workdir ${VALIDATION_ROOT}/benchmark-module --keep-outputs --json)

# conversion-pt2 | PASS | exit=0 | 10.64s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m test.performance.benchmark_conversion_memory --sizes-mib 1 --tables 1 --width 16 --threads 1 --repeat 1 --optimization O1 --input-kind pt2 --workdir ${VALIDATION_ROOT}/benchmark-pt2 --keep-outputs --json)

# cpu-validation-help | PASS | exit=0 | 2.199s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m tico.quantization.recipes.debug.cpu_validation --help)

# cpu-validation | FAIL | exit=1 | 227.768s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m tico.quantization.recipes.debug.cpu_validation --stage all --output ${VALIDATION_ROOT}/cpu-validation)

# discovery-inventory | PASS | exit=0 | 6.184s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/discovery_inventory.py)

# environment | PASS | exit=0 | 4.423s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -c 'import os,sys,json,platform,importlib.metadata as m,tico,torch;print(json.dumps({"python":sys.version,"platform":platform.platform(),"cpu_count":os.cpu_count(),"cpu_affinity":list(os.sched_getaffinity(0)),"tico_file":tico.__file__,"cuda_available":torch.cuda.is_available(),"versions":{p:m.version(p) if m.packages_distributions() and p in [d.metadata["Name"] for d in m.distributions()] else None for p in ["torch","torchvision","transformers","datasets","lm_eval","lm-eval","circle-schema","flatbuffers","tico","lintrunner"]}},indent=2))')

# evaluation-cli-fp-mc-final | PASS | exit=0 | 13.257s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-mc-final -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks lm_eval --output-dir ${VALIDATION_ROOT}/evaluation/fp-mc-final/protocol --no-progress)

# evaluation-cli-fp-mc-r2 | PASS | exit=0 | 17.715s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-mc-r2 -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks lm_eval --output-dir ${VALIDATION_ROOT}/evaluation/fp-mc-r2/protocol --no-progress)

# evaluation-cli-fp-mc | FAIL | exit=1 | 5.529s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-mc -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks lm_eval --output-dir ${VALIDATION_ROOT}/evaluation/fp-mc/protocol --no-progress)

# evaluation-cli-fp-ppl-final | PASS | exit=0 | 6.201s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-ppl-final -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks ppl --output-dir ${VALIDATION_ROOT}/evaluation/fp-ppl-final/protocol --no-progress)

# evaluation-cli-fp-ppl-r2 | PASS | exit=0 | 7.434s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-ppl-r2 -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks ppl --output-dir ${VALIDATION_ROOT}/evaluation/fp-ppl-r2/protocol --no-progress)

# evaluation-cli-fp-ppl | FAIL | exit=1 | 5.482s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label fp-ppl -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --device cpu --tasks ppl --output-dir ${VALIDATION_ROOT}/evaluation/fp-ppl/protocol --no-progress)

# evaluation-cli-help | PASS | exit=0 | 5.435s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m tico.quantization.examples.evaluate --help)

# evaluation-cli-quant-mc-final | PASS | exit=0 | 21.621s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label quant-mc-final -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --checkpoint ${VALIDATION_ROOT}/evaluation/quantized_model.pt --device cpu --tasks lm_eval --output-dir ${VALIDATION_ROOT}/evaluation/quant-mc-final/protocol --no-progress)

# evaluation-cli-quant-mc | PASS | exit=0 | 12.467s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label quant-mc -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --checkpoint ${VALIDATION_ROOT}/evaluation/quantized_model.pt --device cpu --tasks lm_eval --output-dir ${VALIDATION_ROOT}/evaluation/quant-mc/protocol --no-progress)

# evaluation-cli-quant-ppl-final | PASS | exit=0 | 7.332s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label quant-ppl-final -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --checkpoint ${VALIDATION_ROOT}/evaluation/quantized_model.pt --device cpu --tasks ppl --output-dir ${VALIDATION_ROOT}/evaluation/quant-ppl-final/protocol --no-progress)

# evaluation-cli-quant-ppl | PASS | exit=0 | 7.839s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/invoke_cli.py --label quant-ppl -- --config ${VALIDATION_ROOT}/evaluation/config.yaml --checkpoint ${VALIDATION_ROOT}/evaluation/quantized_model.pt --device cpu --tasks ppl --output-dir ${VALIDATION_ROOT}/evaluation/quant-ppl/protocol --no-progress)

# evaluation-final-pip-check | PASS | exit=0 | 0.466s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip check)

# evaluation-prepare | PASS | exit=0 | 8.201s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/prepare.py)

# evaluation-verify-results | PASS | exit=0 | 4.683s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/evaluation/verify_results.py)

# external-benchmark-access | FAIL | exit=56 | 0.032s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} curl -I --max-time 15 https://huggingface.co/datasets/Rowan/hellaswag)

# final-source-check | PASS | exit=0 | 1.422s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/final_source_check.py)

# focused-frozen | PASS | exit=0 | 68.438s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} ./ccex test -k frozen)

# format | PASS | exit=0 | 210.185s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} ./ccex format --no-apply-patches)

# full-suite | PASS | exit=0 | 995.207s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} ./ccex test)

# gemma-generic-export-diagnostic | PASS | exit=0 | 5.43s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/gemma-export-review/diagnose.py)

# list-cases | PASS | exit=0 | 5.976s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m tico.quantization.examples.inspector --mode wrapper-smoke --list-cases)

# optimizer-help | PASS | exit=0 | 1.774s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m test.performance.benchmark_circle_optimizer --help)

# optional-lm-eval-hf-install | PASS | exit=0 | 2.219s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip install 'lm-eval[hf]==0.4.13' -c ${VALIDATION_ROOT}/evaluation/constraints-existing.txt)

# optional-lm-eval-install | PASS | exit=0 | 29.193s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip install lm-eval==0.4.13 -c ${VALIDATION_ROOT}/evaluation/constraints-existing.txt)

# packages-final | PASS | exit=0 | 0.265s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip list --format=freeze)

# pip-check-final | PASS | exit=0 | 0.415s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip check)

# pip-check | PASS | exit=0 | 0.365s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m pip check)

# resource-snapshot | PASS | exit=0 | 1.628s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} bash -c 'date -u; nproc; lscpu; free -b; df -B1 /workspace; cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; git status --porcelain; git rev-parse HEAD HEAD^{tree}; python -c "import tico; print(tico.__file__)"')

# wrapper-baseline-encoded-replay | FAIL | exit=1 | 6.98s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/encoded_replay.py --side wrapper-baseline)

# wrapper-baseline-equivalent-encoded-replay | FAIL | exit=1 | 4.882s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/encoded_replay.py --side wrapper-baseline-equivalent)

# wrapper-baseline-equivalent-qwen3_vl_for_conditional_generation | FAIL | exit=1 | 6.182s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/baseline_equivalent.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_for_conditional_generation --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_for_conditional_generation --report-json ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_for_conditional_generation.json)

# wrapper-baseline-equivalent-qwen3_vl_model | FAIL | exit=1 | 6.132s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/baseline_equivalent.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_model --report-json ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_model.json)

# wrapper-baseline-equivalent-qwen3_vl_vision_attention | FAIL | exit=1 | 9.043s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/baseline_equivalent.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_attention --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_vision_attention --report-json ${VALIDATION_ROOT}/wrapper-baseline-equivalent/qwen3_vl_vision_attention.json)

# wrapper-baseline-gemma4_assistant_core | FAIL | exit=1 | 5.029s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_assistant_core --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_assistant_core.json)

# wrapper-baseline-gemma4_for_causal_lm | FAIL | exit=1 | 5.43s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_for_causal_lm --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_for_causal_lm --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_for_causal_lm.json)

# wrapper-baseline-gemma4_for_conditional_generation | FAIL | exit=1 | 5.63s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_for_conditional_generation --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_for_conditional_generation --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_for_conditional_generation.json)

# wrapper-baseline-gemma4_model | FAIL | exit=1 | 11.121s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_model --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_model.json)

# wrapper-baseline-gemma4_text_attention_k_eq_v_prefill | FAIL | exit=1 | 8.639s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_k_eq_v_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_k_eq_v_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_k_eq_v_prefill.json)

# wrapper-baseline-gemma4_text_attention_prefill | FAIL | exit=1 | 8.888s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_prefill.json)

# wrapper-baseline-gemma4_text_attention_shared_kv_decode | PASS | exit=0 | 7.586s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_shared_kv_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_shared_kv_decode --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_text_attention_shared_kv_decode.json)

# wrapper-baseline-gemma4_vision_attention | FAIL | exit=1 | 9.594s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_attention --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_attention --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_attention.json)

# wrapper-baseline-gemma4_vision_encoder | FAIL | exit=1 | 11.447s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_encoder --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_encoder --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_encoder.json)

# wrapper-baseline-gemma4_vision_encoder_layer | FAIL | exit=1 | 11.151s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_encoder_layer --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_encoder_layer --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_encoder_layer.json)

# wrapper-baseline-gemma4_vision_model | FAIL | exit=1 | 13.503s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_model --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_model.json)

# wrapper-baseline-gemma4_vision_patch_embedder | FAIL | exit=1 | 6.13s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_patch_embedder --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_patch_embedder --report-json ${VALIDATION_ROOT}/wrapper-baseline/gemma4_vision_patch_embedder.json)

# wrapper-baseline-llama_attention_decode | PASS | exit=0 | 9.452s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_attention_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/llama_attention_decode --report-json ${VALIDATION_ROOT}/wrapper-baseline/llama_attention_decode.json)

# wrapper-baseline-llama_attention_prefill | FAIL | exit=1 | 7.686s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/llama_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/llama_attention_prefill.json)

# wrapper-baseline-llama_decoder_layer_decode | FAIL | exit=1 | 10.345s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_decoder_layer_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/llama_decoder_layer_decode --report-json ${VALIDATION_ROOT}/wrapper-baseline/llama_decoder_layer_decode.json)

# wrapper-baseline-llama_decoder_layer_prefill | FAIL | exit=1 | 9.088s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_decoder_layer_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/llama_decoder_layer_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/llama_decoder_layer_prefill.json)

# wrapper-baseline-nn_linear | PASS | exit=0 | 6.537s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_linear --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/nn_linear --report-json ${VALIDATION_ROOT}/wrapper-baseline/nn_linear.json)

# wrapper-baseline-nn_tied_embedding | FAIL | exit=1 | 5.729s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_tied_embedding --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/nn_tied_embedding --report-json ${VALIDATION_ROOT}/wrapper-baseline/nn_tied_embedding.json)

# wrapper-baseline-qwen3_vl_for_conditional_generation | FAIL | exit=1 | 5.081s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_for_conditional_generation --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_for_conditional_generation --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_for_conditional_generation.json)

# wrapper-baseline-qwen3_vl_model | FAIL | exit=1 | 5.181s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_model --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_model.json)

# wrapper-baseline-qwen3_vl_text_attention_decode | FAIL | exit=1 | 8.943s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_attention_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_attention_decode --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_attention_decode.json)

# wrapper-baseline-qwen3_vl_text_attention_prefill | FAIL | exit=1 | 8.688s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_attention_prefill.json)

# wrapper-baseline-qwen3_vl_text_decoder_layer_prefill | FAIL | exit=1 | 9.841s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_decoder_layer_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_decoder_layer_prefill --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_decoder_layer_prefill.json)

# wrapper-baseline-qwen3_vl_text_model | FAIL | exit=1 | 5.327s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_model --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_text_model.json)

# wrapper-baseline-qwen3_vl_vision_attention | FAIL | exit=1 | 4.828s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_attention --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_vision_attention --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_vision_attention.json)

# wrapper-baseline-qwen3_vl_vision_block | FAIL | exit=1 | 10.545s
(cd ${BASELINE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${BASELINE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_block --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_vision_block --report-json ${VALIDATION_ROOT}/wrapper-baseline/qwen3_vl_vision_block.json)

# wrapper-capture-missing-contracts | PASS | exit=0 | 6.433s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/capture_missing_contracts.py)

# wrapper-circle-encoded-replay | FAIL | exit=1 | 9.238s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/encoded_replay.py --side wrapper-circle)

# wrapper-circle-gemma4_assistant_core | FAIL | exit=1 | 6.189s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_assistant_core --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_assistant_core.json)

# wrapper-circle-gemma4_for_causal_lm | FAIL | exit=1 | 6.385s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_for_causal_lm --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_for_causal_lm --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_for_causal_lm.json)

# wrapper-circle-gemma4_for_conditional_generation | FAIL | exit=1 | 5.877s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_for_conditional_generation --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_for_conditional_generation --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_for_conditional_generation.json)

# wrapper-circle-gemma4_model | FAIL | exit=1 | 12.103s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_model --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_model.json)

# wrapper-circle-gemma4_multimodal_embedder | PASS | exit=0 | 6.631s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_multimodal_embedder --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_multimodal_embedder --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_multimodal_embedder.json)

# wrapper-circle-gemma4_text_attention_decode | PASS | exit=0 | 10.703s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_decode.json)

# wrapper-circle-gemma4_text_attention_k_eq_v_prefill | FAIL | exit=1 | 10.092s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_k_eq_v_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_k_eq_v_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_k_eq_v_prefill.json)

# wrapper-circle-gemma4_text_attention_prefill | FAIL | exit=1 | 11.194s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_prefill.json)

# wrapper-circle-gemma4_text_attention_shared_kv_decode | PASS | exit=0 | 9.01s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_shared_kv_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_shared_kv_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_shared_kv_decode.json)

# wrapper-circle-gemma4_text_attention_shared_kv_prefill | PASS | exit=0 | 11.796s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_shared_kv_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_shared_kv_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_shared_kv_prefill.json)

# wrapper-circle-gemma4_text_attention_sliding_decode | PASS | exit=0 | 10.691s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_sliding_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_sliding_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_sliding_decode.json)

# wrapper-circle-gemma4_text_attention_sliding_prefill | PASS | exit=0 | 10.411s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_attention_sliding_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_sliding_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_attention_sliding_prefill.json)

# wrapper-circle-gemma4_text_decoder_layer_decode | PASS | exit=0 | 12.959s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_decoder_layer_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_decode.json)

# wrapper-circle-gemma4_text_decoder_layer_prefill | PASS | exit=0 | 12.001s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_decoder_layer_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_prefill.json)

# wrapper-circle-gemma4_text_decoder_layer_shared_kv | PASS | exit=0 | 10.543s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_decoder_layer_shared_kv --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_shared_kv --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_shared_kv.json)

# wrapper-circle-gemma4_text_decoder_layer_sliding_prefill | PASS | exit=0 | 11.854s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_decoder_layer_sliding_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_sliding_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_decoder_layer_sliding_prefill.json)

# wrapper-circle-gemma4_text_mlp | PASS | exit=0 | 7.387s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_mlp --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_mlp --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_mlp.json)

# wrapper-circle-gemma4_text_scaled_word_embedding | PASS | exit=0 | 6.234s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_text_scaled_word_embedding --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_scaled_word_embedding --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_text_scaled_word_embedding.json)

# wrapper-circle-gemma4_vision_attention | FAIL | exit=1 | 10.645s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_attention --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_attention --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_attention.json)

# wrapper-circle-gemma4_vision_encoder | FAIL | exit=1 | 12.907s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_encoder --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_encoder --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_encoder.json)

# wrapper-circle-gemma4_vision_encoder_layer | FAIL | exit=1 | 12.242s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_encoder_layer --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_encoder_layer --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_encoder_layer.json)

# wrapper-circle-gemma4_vision_model | FAIL | exit=1 | 17.184s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_model --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_model.json)

# wrapper-circle-gemma4_vision_patch_embedder | FAIL | exit=1 | 6.379s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_patch_embedder --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_patch_embedder --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_patch_embedder.json)

# wrapper-circle-gemma4_vision_pooler | PASS | exit=0 | 6.433s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case gemma4_vision_pooler --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_pooler --report-json ${VALIDATION_ROOT}/wrapper-circle/gemma4_vision_pooler.json)

# wrapper-circle-llama_attention_decode | PASS | exit=0 | 9.495s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_attention_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/llama_attention_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/llama_attention_decode.json)

# wrapper-circle-llama_attention_prefill | FAIL | exit=1 | 8.432s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/llama_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/llama_attention_prefill.json)

# wrapper-circle-llama_decoder_layer_decode | FAIL | exit=1 | 11.048s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_decoder_layer_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/llama_decoder_layer_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/llama_decoder_layer_decode.json)

# wrapper-circle-llama_decoder_layer_prefill | FAIL | exit=1 | 9.787s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_decoder_layer_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/llama_decoder_layer_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/llama_decoder_layer_prefill.json)

# wrapper-circle-llama_mlp | PASS | exit=0 | 6.598s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case llama_mlp --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/llama_mlp --report-json ${VALIDATION_ROOT}/wrapper-circle/llama_mlp.json)

# wrapper-circle-nn_conv3d | PASS | exit=0 | 6.283s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_conv3d --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/nn_conv3d --report-json ${VALIDATION_ROOT}/wrapper-circle/nn_conv3d.json)

# wrapper-circle-nn_conv3d_special_case | PASS | exit=0 | 6.283s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_conv3d_special_case --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/nn_conv3d_special_case --report-json ${VALIDATION_ROOT}/wrapper-circle/nn_conv3d_special_case.json)

# wrapper-circle-nn_layernorm | PASS | exit=0 | 8.706s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_layernorm --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/nn_layernorm --report-json ${VALIDATION_ROOT}/wrapper-circle/nn_layernorm.json)

# wrapper-circle-nn_linear | PASS | exit=0 | 7.037s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_linear --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/nn_linear --report-json ${VALIDATION_ROOT}/wrapper-circle/nn_linear.json)

# wrapper-circle-nn_tied_embedding | FAIL | exit=1 | 6.129s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case nn_tied_embedding --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/nn_tied_embedding --report-json ${VALIDATION_ROOT}/wrapper-circle/nn_tied_embedding.json)

# wrapper-circle-qwen3_vl_for_conditional_generation | FAIL | exit=1 | 7.036s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_for_conditional_generation --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_for_conditional_generation --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_for_conditional_generation.json)

# wrapper-circle-qwen3_vl_model | FAIL | exit=1 | 6.837s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_model --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_model.json)

# wrapper-circle-qwen3_vl_text_attention_decode | FAIL | exit=1 | 9.593s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_attention_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_attention_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_attention_decode.json)

# wrapper-circle-qwen3_vl_text_attention_prefill | FAIL | exit=1 | 9.344s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_attention_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_attention_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_attention_prefill.json)

# wrapper-circle-qwen3_vl_text_decoder_layer_decode | PASS | exit=0 | 11.201s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_decoder_layer_decode --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_decoder_layer_decode --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_decoder_layer_decode.json)

# wrapper-circle-qwen3_vl_text_decoder_layer_prefill | FAIL | exit=1 | 12.761s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_decoder_layer_prefill --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_decoder_layer_prefill --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_decoder_layer_prefill.json)

# wrapper-circle-qwen3_vl_text_mlp | PASS | exit=0 | 7.793s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_mlp --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_mlp --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_mlp.json)

# wrapper-circle-qwen3_vl_text_model | FAIL | exit=1 | 6.088s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_text_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_model --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_text_model.json)

# wrapper-circle-qwen3_vl_vision_attention | FAIL | exit=1 | 11.161s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_attention --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_attention --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_attention.json)

# wrapper-circle-qwen3_vl_vision_block | FAIL | exit=1 | 12.864s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_block --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_block --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_block.json)

# wrapper-circle-qwen3_vl_vision_mlp | PASS | exit=0 | 8.754s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_mlp --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_mlp --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_mlp.json)

# wrapper-circle-qwen3_vl_vision_model | PASS | exit=0 | 17.33s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_model --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_model --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_model.json)

# wrapper-circle-qwen3_vl_vision_patch_embed | PASS | exit=0 | 6.384s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_patch_embed --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_patch_embed --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_patch_embed.json)

# wrapper-circle-qwen3_vl_vision_patch_merger | PASS | exit=0 | 7.543s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python ${VALIDATION_ROOT}/wrapper-harness/export_case.py --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case qwen3_vl_vision_patch_merger --device cpu --export circle --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_patch_merger --report-json ${VALIDATION_ROOT}/wrapper-circle/qwen3_vl_vision_patch_merger.json)

# wrapper-numerical-all | PASS | exit=0 | 10.674s
(cd ${CANDIDATE_REPO} && env PATH=${VALIDATION_VENV}/bin:"$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH=${CANDIDATE_REPO} python -m tico.quantization.examples.inspector --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case all --device cpu --export none --strict --no-plot --output-dir ${VALIDATION_ROOT}/wrapper-smoke)
