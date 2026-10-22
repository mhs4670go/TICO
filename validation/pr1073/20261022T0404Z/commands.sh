#!/usr/bin/env bash
# Source this environment before replaying validation commands.
# Bootstrap actually executed before the command logger existed:
# git ls-remote https://github.com/mhs4670go/TICO.git refs/heads/ea
# git ls-remote https://github.com/Samsung/TICO.git refs/pull/1073/head
# git clone --branch ea https://github.com/mhs4670go/TICO.git /workspace/TICO
# cd /workspace/TICO && python -m venv .venv
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# After venv setup:
source /workspace/TICO/.venv/bin/activate

# repository
(cd /workspace/TICO && bash -c 'git status --short; git rev-parse HEAD; git branch --show-current; git log --oneline -12; git diff --stat bb09ca852b0ca1b1a6271cfe1e10c162104c997e HEAD; git diff --name-only bb09ca852b0ca1b1a6271cfe1e10c162104c997e HEAD; git worktree list; git ls-remote https://github.com/mhs4670go/TICO.git refs/heads/ea; git ls-remote https://github.com/Samsung/TICO.git refs/pull/1073/head')

# initial_environment
(cd /workspace/TICO && bash -c 'python --version; command -v python; python -m pip list --format=json; lscpu; free -h; df -h /workspace /tmp; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max; if command -v nvidia-smi; then nvidia-smi; else echo GPU_NOT_AVAILABLE; fi; cat /etc/codex/network-policy.json')

# baseline_worktree
(cd /workspace/TICO && git worktree add --detach /workspace/TICO-export-baseline bb09ca852b0ca1b1a6271cfe1e10c162104c997e)

# install
(cd /workspace/TICO && ./ccex install --cpu_only)

# assets
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/inspect_assets.py)

# configure_test
(cd /workspace/TICO && ./ccex configure test --cpu_only)

# configure_format
(cd /workspace/TICO && ./ccex configure format)

# comparator_extra_negatives
(cd /workspace/TICO && env PYTHONPATH=/workspace/TICO python /workspace/TICO/out/pr1073_validation/20261022T0404Z/comparator_extra_negatives.py)

# environment
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/environment_info.py)

# cli_help
(cd /workspace/TICO && bash -c 'python tico/quantization/recipes/debug/verify_llama_rope.py --help; python tico/quantization/recipes/debug/export_regression.py --help; python tico/quantization/recipes/debug/export_regression.py run --help; python tico/quantization/recipes/debug/export_regression.py isolation --help; python -m tico.quantization.examples.evaluate --help; ./ccex test --help; ./ccex format --help')

# install_accelerate
(cd /workspace/TICO && python -m pip install -c /workspace/TICO/out/pr1073_validation/20261022T0404Z/dependency_constraints.txt accelerate==1.12.0)

# install_accelerate_retry
(cd /workspace/TICO && python -m pip install -c /workspace/TICO/out/pr1073_validation/20261022T0404Z/dependency_constraints_noneditable.txt accelerate==1.12.0)

# gemma_internal_multimodal_units
(cd /workspace/TICO && env PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_INTERNAL_TESTS=1 .venv/bin/python -m unittest test.quantization.wrapq.wrappers.gemma4.test_quant_model -v)

# changed_unit_tests
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/run_changed_tests.py)

# format_all
(cd /workspace/TICO && ./ccex format --no-apply-patches)

# dependency_check
(cd /workspace/TICO && python -m pip check)

# environment_final
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/environment_info.py)

# help_verify_llama
(cd /workspace/TICO && python tico/quantization/recipes/debug/verify_llama_rope.py --help)

# help_export_regression_run
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --help)

# help_export_regression_isolation
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py isolation --help)

# help_evaluate
(cd /workspace/TICO && python -m tico.quantization.examples.evaluate --help)

# qwen_tiny_multimodal_units
(cd /workspace/TICO && env PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest test.quantization.wrapq.wrappers.qwen_vl.test_quant_model -v)

# llama_rope
(cd /workspace/TICO && python tico/quantization/recipes/debug/verify_llama_rope.py --repo . --report /workspace/TICO/out/pr1073_validation/20261022T0404Z/llama-rope-checks.json)

# changed_unit_discovery
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/run_changed_discovery.py)

# supplemental_eager_state
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/supplemental_eager_state.py --output out/pr1073_validation/20261022T0404Z/supplemental_eager_state)

# baseline_test_import
(cd /workspace/TICO-export-baseline && python -c 'import tico, torch, transformers; print(tico.__file__); print(torch.__version__, transformers.__version__)')

# create_baseline_llama_fixture
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family llama --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama.pt)

# pretrained_asset_download
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_llama_smoke.py download)

# baseline_llama_export_units
(cd /workspace/TICO-export-baseline && python -m unittest discover -s test -v -k quantization.recipes.test_export_llama)

# generated_image_eager
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/generated_image_eager.py --output out/pr1073_validation/20261022T0404Z/generated_image_eager)

# tiny_llama_old_new
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama.pt --migrate-llama-rope --seq-len 8 --eager-seq-len 12 --decode-steps 4 --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-old-new)

# pretrained_ptq_prepare
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_llama_smoke.py calibrate)

# pretrained_fp_cli_ppl
(cd /workspace/TICO && python -m tico.quantization.examples.evaluate --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_eval.yaml --tasks ppl --set calibration.seq_len=32 --set export.max_seq_len=16 --set evaluation.max_seq_len=128 --set evaluation.perplexity.max_seq_len=64 --set evaluation.perplexity.stride=32 --set 'evaluation.perplexity.split=test[:12]' --device cpu --no-progress --output-dir /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_fp_cli)

# create_baseline_llama_fixture_v2
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family llama --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.pt)

# baseline_llama_causal_units
(cd /workspace/TICO-export-baseline && python -m unittest test.quantization.wrapq.wrappers.llama.test_quant_model_for_casual_lm -v)

# full_unit_suite
(cd /workspace/TICO && ./ccex test)

# tiny_llama_old_new_v2
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.pt --migrate-llama-rope --seq-len 8 --eager-seq-len 12 --decode-steps 4 --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-old-new-v2)

# pretrained_quant_cli_ppl
(cd /workspace/TICO && python -m tico.quantization.examples.evaluate --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_eval.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_tinyllama_quantized.pt --tasks ppl --set calibration.seq_len=32 --set export.max_seq_len=16 --set evaluation.max_seq_len=128 --set evaluation.perplexity.max_seq_len=64 --set evaluation.perplexity.stride=32 --set 'evaluation.perplexity.split=test[:12]' --device cpu --no-progress --output-dir /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_quant_cli)

# pretrained_supplemental
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_llama_smoke.py supplemental)

# baseline_pretrained_generation
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_pretrained_generation.py)

# format_ufmt_all
(cd /workspace/TICO && lintrunner --all-files --take UFMT --config /workspace/TICO/.lintrunner.toml)

# quantization_suite
(cd /workspace/TICO && ./ccex test -k quantization)

# pretrained_supplemental_v2
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_llama_smoke.py supplemental)

# trace_baseline_parity
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/trace_baseline_parity.py)

# create_baseline_qwen_fixture
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family qwen3_vl --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen.pt)

# image_generation_probe
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/image_generation_probe.py --output out/pr1073_validation/20261022T0404Z/image_generation_probe)

# pretrained_protocol_comparison
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/compare_pretrained_protocol.py)

# pretrained_circle_isolation_16
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/resource_monitor.py --report /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_circle_isolation_16_resources.json -- python tico/quantization/recipes/debug/export_regression.py isolation --candidate-repo . --config tico/quantization/examples/configs/llama_export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_tinyllama_quantized.pt --seq-len 16 --eager-seq-len 64 --decode-steps 4 --threads 1 --atol 1e-5 --rtol 1e-4 --artifacts circle_bundle --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_tinyllama_isolation_16)

# format_changed
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/run_changed_static.py)

# trace_baseline_parity_v2
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/trace_baseline_parity.py)

# create_baseline_qwen_fixture_v2
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family qwen3_vl --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_v2.pt)

# qwen_image_generation_baseline
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/image_generation_baseline.py --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/image_generation_probe/qwen3_vl/frozen_full_module.pt --inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/image_generation_probe/qwen3_vl/generation_inputs.pt --report /workspace/TICO/out/pr1073_validation/20261022T0404Z/qwen_image_generation_baseline.json)

# create_baseline_gemma_fixture
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family gemma4 --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma.pt)

# probe_rounding_and_past
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/probe_rounding_and_past.py)

# qwen_image_generation_fresh_baseline
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/qwen_generation_fresh_baseline.py --source-checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/image_generation_probe/qwen3_vl/frozen_full_module.pt --inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/image_generation_probe/qwen3_vl/generation_inputs.pt --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/qwen_image_generation_fresh_baseline)

# create_baseline_qwen_fixture_cached
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family qwen3_vl --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.pt)

# tiny_qwen_old_new_cached
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-old-new-cached)

# tiny_llama_evidence_inventory
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-old-new-v2)

# baseline_qparam_timeout_probe
(cd /workspace/TICO-export-baseline && python -m unittest test.quantization.examples.test_gptq_qparam_handoff.TestLegacyGPTQQParamHandoff.test_full_model_example_rejects_bit_mismatch -v)

# tiny_llama_rope_diagnosis
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/diagnose_llama_circle_rope.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-old-new-v2 --provenance /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.provenance.json)

# create_baseline_gemma_fixture_cached
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family gemma4 --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_cached.pt)

# format_baseline_ufmt_counterparts
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/run_baseline_ufmt.py)

# candidate_qparam_timeout_probe
(cd /workspace/TICO && python -m unittest test.quantization.examples.test_gptq_qparam_handoff.TestLegacyGPTQQParamHandoff.test_full_model_example_rejects_bit_mismatch -v)

# create_candidate_llama_native4100
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO --family llama --capacity 4100 --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native4100.pt)

# pretrained_cached_fixture_prepare
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_calibration_cached.py)

# copy_long_llama_creator
(cd /workspace/TICO && cp -n out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py out/pr1073_validation/20261022T0404Z/create_long_llama_fixture.py)

# create_candidate_llama_native16384
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/create_long_llama_fixture.py --repo /workspace/TICO --family llama --capacity 16384 --output out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native16384.pt)

# pretrained_observer_coverage
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_pretrained_observers.py)

# pretrained_cached_circle_isolation_16
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/resource_monitor.py --report /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_cached_circle_isolation_16_resources.json -- python tico/quantization/recipes/debug/export_regression.py isolation --candidate-repo . --config tico/quantization/examples/configs/llama_export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_tinyllama_quantized_cached.pt --seq-len 16 --eager-seq-len 64 --decode-steps 4 --threads 1 --atol 1e-5 --rtol 1e-4 --artifacts circle_bundle --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_tinyllama_cached_isolation_16)

# tiny_llama_full_original
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.pt --migrate-llama-rope --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts circle_full --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-full-original)

# long_llama_resource_gate
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/long_llama_resource_gate.py)

# llama_long_eager_full_isolation
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python out/pr1073_validation/20261022T0404Z/resource_monitor.py --report out/pr1073_validation/20261022T0404Z/llama_long_eager_full_resources.json -- .venv/bin/python tico/quantization/recipes/debug/export_regression.py isolation --candidate-repo . --config out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native16384.export.yaml --checkpoint out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native16384.pt --artifacts circle_full --seq-len 8 --eager-seq-len 4096 --decode-steps 4 --require-torch-parity --output out/pr1073_validation/20261022T0404Z/tiny-llama-native16384-long-eager-full)

# pretrained_separate_results
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/split_pretrained_results.py)

# audit_long_llama_full
(cd /workspace/TICO && env PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/audit_long_llama_full.py --run out/pr1073_validation/20261022T0404Z/tiny-llama-native16384-long-eager-full)

# create_baseline_gemma_fixture_image70
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/create_tiny_frozen_fixture.py --repo /workspace/TICO-export-baseline --family gemma4 --gemma-max-soft-tokens 70 --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt)

# baseline_llama_attention_units
(cd /workspace/TICO-export-baseline && python -m unittest test.quantization.wrapq.wrappers.llama.test_quant_attention -v)

# prepare_png_circle_inputs
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/prepare_png_circle_inputs.py)

# pretrained_cached_stage_differences_initial
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/analyze_pretrained_circle_records.py)

# combine_gemma_png_inputs
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/combine_gemma_png_inputs.py)

# tiny_gemma_old_new_image70
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --require-torch-parity --set export.vision.granularity=monolithic --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-gemma-old-new-image70)

# tiny_qwen_evidence_inventory
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-old-new-cached)

# summarize_tiny_llama_components
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/summarize_circle_components.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-old-new-v2)

# summarize_tiny_qwen_components
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/summarize_circle_components.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-old-new-cached)

# pretrained_cached_stage_differences_midrun
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/analyze_pretrained_circle_records.py)

# inspect_qwen_parity
(cd /workspace/TICO-export-baseline && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_qwen_parity.py)

# tiny_gemma_candidate_vision_isolation
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py isolation --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --require-torch-parity --set export.vision.granularity=monolithic --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-gemma-candidate-vision-isolation)

# format_all_retry
(cd /workspace/TICO && ./ccex format --no-apply-patches)

# tiny_qwen_full_text_png_original
(cd /workspace/TICO && python tico/quantization/recipes/debug/export_regression.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts '' --include-full-text --eager-inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/png_circle_inputs/qwen_actual_png_kwargs.pt --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-full-text-png-original)

# copy_gemma_text_recorder
(cd /workspace/TICO && cp -n tico/quantization/recipes/debug/export_regression.py out/pr1073_validation/20261022T0404Z/gemma_text_export_recording.py)

# gemma_text_baseline
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/gemma_text_stage_worker.py --repo /workspace/TICO-export-baseline --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/gemma-text-independent-baseline)

# tiny_qwen_full_text_png_dirs_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/export_regression_full_text_dirs.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_qwen_cached.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts '' --include-full-text --eager-inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/png_circle_inputs/qwen_actual_png_kwargs.pt --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-full-text-png-dirs-retry)

# calibration_path_audit
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_calibration_paths.py)

# pretrained_summary_interim
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/write_pretrained_summary.py)

# gemma_text_candidate
(cd /workspace/TICO && env RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/gemma_text_stage_worker.py --repo /workspace/TICO --checkpoint out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --output out/pr1073_validation/20261022T0404Z/gemma-text-independent-candidate --stress)

# pretrained_cached_circle_live_resource_snapshot
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/snapshot_pretrained_running_resources.py)

# preserve_gemma_vision_failure
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/preserve_gemma_vision_failure.py)

# pretrained_selected_stage
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/resource_monitor_v2.py --report /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_selected_stage_resources.json -- python /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_selected_stage_isolation.py --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/pretrained_selected_stage)

# pretrained_cached_stage_differences_final
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/analyze_pretrained_circle_records.py)

# pretrained_summary_after_timeout
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/write_pretrained_summary.py)

# pretrained_cached_stage_differences_timeout
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/analyze_pretrained_circle_records.py)

# tiny_gemma_full_text_png_dirs
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/export_regression_full_text_dirs.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts '' --include-full-text --eager-inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/png_circle_inputs/gemma_actual_png_kwargs.pt --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-dirs)

# tiny_qwen_full_text_png_audit
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-full-text-png-dirs-retry)

# record_png_fullmodel_forwards
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/record_png_fullmodel_forwards.py)

# tiny_qwen_full_text_png_components
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/summarize_circle_components.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-qwen-full-text-png-dirs-retry)

# gemma_text_strict_comparison
(cd /workspace/TICO && env PYTHONPATH=/workspace/TICO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python out/pr1073_validation/20261022T0404Z/compare_gemma_text_stages.py --baseline out/pr1073_validation/20261022T0404Z/gemma-text-independent-baseline --candidate out/pr1073_validation/20261022T0404Z/gemma-text-independent-candidate --output out/pr1073_validation/20261022T0404Z/gemma_text_strict_comparison.json)

# gemma_text_parity_trace
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_gemma_text_parity.py)

# tiny_llama_full_shape_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/export_regression_full_shape_fixture.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo /workspace/TICO --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_llama_v2.pt --migrate-llama-rope --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts circle_full --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-full-shape-retry)

# gemma_text_parity_trace_v2
(cd /workspace/TICO-export-baseline && env PYTHONPATH=/workspace/TICO-export-baseline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_gemma_text_parity.py)

# tiny_gemma_full_text_png_masks_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/export_regression_gemma_full_text_masks.py run --baseline-repo /workspace/TICO-export-baseline --candidate-repo . --config /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.export.yaml --checkpoint /workspace/TICO/out/pr1073_validation/20261022T0404Z/fixtures/baseline_gemma_image70.pt --seq-len 8 --eager-seq-len 12 --decode-steps 4 --artifacts '' --include-full-text --eager-inputs /workspace/TICO/out/pr1073_validation/20261022T0404Z/png_circle_inputs/gemma_actual_png_kwargs.pt --require-torch-parity --output /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-masks-retry)

# audit_tiny_llama_full_shape_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-full-shape-retry)

# format_baseline_mypy_counterparts
(cd /workspace/TICO-export-baseline && env MYPY_CACHE_DIR=/workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_mypy_cache lintrunner --take MYPY --config /workspace/TICO-export-baseline/.lintrunner.toml test/quantization/recipes/test_export_llama_bundle.py tico/quantization/recipes/export/llama.py)

# components_tiny_llama_full_shape_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/summarize_circle_components.py --run /workspace/TICO/out/pr1073_validation/20261022T0404Z/tiny-llama-full-shape-retry)

# gemma_full_text_mask_provenance
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/record_gemma_full_text_masks.py)

# baseline_lintrunner_path_probe
(cd /workspace/TICO-export-baseline && env MYPY_CACHE_DIR=/workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_mypy_cache lintrunner -vv --take MYPY --config /workspace/TICO-export-baseline/.lintrunner.toml test/quantization/recipes/test_export_llama_bundle.py tico/quantization/recipes/export/llama.py)

# format_baseline_ufmt_counterparts_verified
(cd /workspace/TICO-export-baseline && lintrunner -vv --take UFMT --config /workspace/TICO-export-baseline/.lintrunner.toml --paths-from /workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_ufmt_paths.txt)

# format_baseline_mypy_counterparts_verified
(cd /workspace/TICO-export-baseline && env MYPY_CACHE_DIR=/workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_mypy_cache lintrunner -vv --take MYPY --config /workspace/TICO-export-baseline/.lintrunner.toml --paths-from /workspace/TICO/out/pr1073_validation/20261022T0404Z/baseline_mypy_paths.txt)

# summarize_gemma_text_parity
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/summarize_gemma_text_parity.py)

# changed_static_selection_probe
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/probe_changed_static_selection.py)

# pretrained_selected_stage_differences
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/analyze_pretrained_selected_records.py)

# pretrained_summary_final
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/write_pretrained_summary.py)

# summarize_gemma_text_parity_v2
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/summarize_gemma_text_parity.py)

# audit_gemma_full_text_png_masks
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-masks-retry)

# summarize_gemma_full_text_png_masks
(cd /workspace/TICO && python out/pr1073_validation/20261022T0404Z/summarize_circle_components.py out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-masks-retry)

# audit_gemma_full_text_png_masks_corrected_cli
(cd /workspace/TICO && .venv/bin/python out/pr1073_validation/20261022T0404Z/audit_circle_evidence.py --run out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-masks-retry)

# summarize_gemma_full_text_png_masks_corrected_cli
(cd /workspace/TICO && .venv/bin/python out/pr1073_validation/20261022T0404Z/summarize_circle_components.py --run out/pr1073_validation/20261022T0404Z/tiny-gemma-full-text-png-masks-retry)

# build_circle_validation_summary
(cd /workspace/TICO && .venv/bin/python out/pr1073_validation/20261022T0404Z/build_circle_validation_summary.py)

# inspect_gemma_full_text
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/inspect_gemma_full_text.py)

# static_report_generation
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/finalize_static_report.py)

# build_circle_validation_summary_final
(cd /workspace/TICO && .venv/bin/python out/pr1073_validation/20261022T0404Z/build_circle_validation_summary.py)

# artifact_package_preliminary_plan
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/package_validation_artifacts.py --plan)

# artifact_package_preliminary_plan_retry
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/package_validation_artifacts.py --plan)

# final_repository_environment
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/record_final_environment.py)

# artifact_package_canonical_plan
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/package_validation_artifacts.py --plan)

# finalize_validation_report
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/finalize_validation.py)

# finalize_validation_summary
(cd /workspace/TICO && python /workspace/TICO/out/pr1073_validation/20261022T0404Z/build_summary.py --complete)

# Final artifact packaging; capture output outside RESULT_ROOT before sealing.
(cd /workspace/TICO && /workspace/TICO/.venv/bin/python /workspace/TICO/out/pr1073_validation/20261022T0404Z/package_validation_artifacts.py --result-root /workspace/TICO/out/pr1073_validation/20261022T0404Z --archive /workspace/shared/downloads/TICO_PR1073_validation_20261022T0404Z.zip --finalize > /workspace/shared/downloads/TICO_PR1073_validation_20261022T0404Z.zip.packaging.log 2>&1)
