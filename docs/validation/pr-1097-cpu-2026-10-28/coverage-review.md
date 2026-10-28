# PR #1097 source review / coverage map

검토 revision: head `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`, base `261bda28ec55fdac42f59346c950326e4065f8fa`. 아래는 소스에서 확인한 coverage 연결이며 실제 PASS 여부와 수는 본 실행 로그에서 판정한다. 과거 작성자 결과는 실행 증거로 사용하지 않았다.

## 변경과 실행 연결

최근 30일(2026-09-28~2026-10-28)의 base history에 109개 commit이 있다. 전체 목록은 `base-last30d.log`, PR 전체 diff는 `pr.diff`, 파일 목록은 `pr-diff-files.txt`에 보존했다. PR은 149개 파일 변경(8496 insertions, 1071 deletions)으로 공통 observer/lifecycle과 모델별 conditional path, checkpoint/export consumer 및 optimization을 함께 바꾼다.

| 경계 | 최근 변경 및 위험 | 이번 실행과 연결할 기존 테스트/도구 | 검증 계약 |
|---|---|---|---|
| observer/frozen/qparams | #1096 locked Percentile, #1038 EMA alias; PR schema/lifecycle 신설 | `test.quantization.wrapq.observers.test_frozen_state`, `test_execution_compatibility`, `test_lifecycle` (전체 suite) | missing/empty/zero/nonfinite 구분, lock precedence, shape/axis/dtype/range/policy/provenance, readonly owned snapshot, atomic failure/retry, strict=False 손상 거부 |
| GPTQ→PTQ/folding | #1011 handoff validation, #1015 reuse option, #1019 Llama scale, #1074 auto_percdamp | `test.quantization.recipes.test_qparams`, `test_folded_linear_qparam_handoff`, `test_llama_attention_qparam_handoff`, `test_ptq_qparam_handoff`, `test_stage_identity`; GPTQ/FPI tests (전체 suite) | 호환 grid 재사용 및 lock, bit/axis/범위 mismatch 선검증, late invalid에 partial inject 없음, positive uniform folding scale, nonuniform 거부, reuse off/pure PTQ |
| optimization | PR AdaRound joint/global/discrete mutable proxy와 frozen snapshot, block reconstruction snapshot 변경 | `test.quantization.algorithm.adaround.test_joint_adaround`, `test_global_refinement`, `test_discrete_refinement`; `block_reconstruction.test_multistart`, `test_validation` | 실제 gradient/optimizer, accepted finalize와 rollback, weight/grid/provenance/lock 복원, mutable training proxy를 deployment snapshot으로 오인하지 않음 |
| Llama/KV | #1083 storage/compute branching, #1094 integer mask, #1026 inactive fused-scale | `wrapq.wrappers.llama.test_quant_attention`, `test_quant_model`, `test_rope_consistency`; wrapper smoke prefill/decode; CPU driver | tuple/Dynamic/Static cache ABI, delta/present, cache 미수집 거부, mask 의미, prefill/decode numerical/export parity |
| Qwen3-VL/KV/vision | #1087 two-domain cache, #1016 generation positions, #1084 static smoke inputs | `wrapq.wrappers.qwen_vl.test_quant_text_attention`, `test_quant_vision_model`, `recipes.test_qwen_cache_coverage_cpu`; wrapper smoke text/vision; CPU driver | 실제 tiny HF wrapper, imported cache grids, no obsolete past grids, cache 종류/layout, frozen round trip, vision input profile, saved Circle connected replay |
| Gemma4/assistant | #1078 decode calibration, #1079 past-KV, #1086 two-domain, #1076 PLE arithmetic | `wrapq.wrappers.gemma4.test_past_kv_observers`, `test_conditional_capabilities`, `test_pooler_profile_circle`, `test_pooler_export_ownership`; `gemma4_assistant.test_quant_for_causal_lm`; wrapper smoke | shared cache source 역순 freeze plan, sliding rollover, optional vision/token paths, profile constants가 source observer를 변형하지 않음, tied weights와 checkpoint |
| recipe/config/CLI/routing | #1068 evaluation selection, #1092 case matching, #1031 preflight | `recipes.test_stage_identity`, `test_runner_preflight`, `test_evaluation_selection`, `test_evaluation_target_adapters`, config/CLI tests; actual CLI/harness 별도 기록 | invalid target/stage가 model load 전에 실패, top-level tasks와 nested task 목록 구분, config 비변형 |
| FP/Quant eager scoring | #1073 eager protocol, #1085 HellaSwag metric keys | `recipes.test_cpu_evaluation`, `test_eager_evaluation`, `evaluation.test_hellaswag_eval_utils`; CPU driver 및 소량 evaluator 실행 | real HF/frozen forward와 independent NLL, padding/overlap 제외, zero valid token 거부, common context/cache, frozen 보존, `acc,none`/`acc_norm,none` 정확 집계 |
| Circle/export/reference | #1064 file output/memory, #1095 multi-axis repeat, #1090 saturation, #1089 integer div; PR frozen consumers | `wrapq.test_frozen_state_export`, `recipes.test_frozen_snapshot_consumers`, `test_export_execution_preflight`, Circle/serialize/runtime 전체 suite; wrapper Circle, CPU driver, 축소 artifact benchmark | missing calibration artifact 거부, save/reload grid/constant 동일, structural verify/runtime/parity 분리, preflight가 forward/convert 전에 실패, real saved bytes CPU 실행 |

## frozen 핵심 계약의 기존 증거 위치

- Calibration readiness: `test_frozen_state.test_missing_empty_and_zero_are_distinct`, `test_nonfinite_is_invalid_not_missing`, lifecycle `test_uncalibrated_linear_rejects_before_quant`.
- Atomic lifecycle: lifecycle `test_real_percentile_failure_preserves_all_state_and_retry_handle`, `test_install_failure_rolls_back_small_state`, `test_parent_semantic_failure_does_not_commit_children`, `test_partial_optional_subtree_has_no_partial_frozen_grids`.
- Frozen qparams/imported grids: observer `test_every_affine_subtype_prefers_locked_grid`, `test_invalid_imports_leave_existing_snapshot_unchanged`, `test_grid_shape_axis_and_symmetric_policy_validation`, `test_clipping_provenance_moves_as_owned_buffers`.
- Serialization: `test_exact_calibrated_and_imported_state_roundtrip`, `test_whole_module_roundtrip_is_exact`, `test_nonstrict_cannot_bypass_policy_version_or_grid_validation`, `test_legacy_ranges_are_not_frozen_and_percentile_needs_samples`, lifecycle nonstrict wrapper-mode corruption tests.
- Export: `test_frozen_state_export`는 real Circle conversion/reload/reference 출력 및 Llama/Qwen/Gemma chained decode를 수행한다. `test_frozen_snapshot_consumers`는 sparse head snapshot aliasing 및 stateless manifest를 검증한다.
- Conditional: `test_conditional_entry_guard`, `test_conditional_export_entries`, Gemma conditional tests 및 Llama cache capability tests; 미수집 경로를 요청했을 때 예외를 검증하는 negative test는 정상 PASS다.
- CPU dtype: `test_execution_compatibility`는 BF16 scale의 frozen storage validity와 CPU per-channel 실행 제한을 구분한다. BF16 input+FP32 scale은 허용한다. 임의 cast/migration으로 우회하면 안 된다.

## discovery와 의도적 제외

`infra/scripts/test_run.py`의 기본 명령은 `python3 -m unittest discover -s <checkout>/test -v`이다. `test/pt2_to_circle_test/__init__.py:load_tests`는 생성 E2E 중 `test_net.py`, `test_op.py`만 포함한다. 모델 loader는 `./ccex test -m <pattern>`에서 `CCEX_TEST_MODEL`과 `test.pt2_to_circle_test.test_model`로 별도 실행한다. 기본 suite가 PASS해도 full-size model coverage를 뜻하지 않는다.

Gemma4 등 일부 테스트는 `RUN_INTERNAL_TESTS=1`에서만 동작하고, 모델 class가 import되지 않으면 family skip이 가능하다. 실행 로그에서 family별 실제 ok/skip 수를 확인해야 한다. tiny synthetic tests와 smoke registry의 실제 numerical/export가 그 보완 경로다.

fairseq는 smoke registry에 없다. `test/quantization/wrapq/wrappers/fairseq/`는 외부 fairseq import 없이 작은 fairseq-like FP block과 실제 wrappers를 사용한다. MHA incremental self/cross attention/static KV, `test_uncalibrated_optional_masks_reject_before_attention`; decoder layer `test_external_past_requires_its_own_calibrated_capability`, encoder/decoder lifecycle, incremental decoding이 기본 suite에 포함된다. 단 `test_decoder_export_single_step.test_export_decoder_single_step_invokes_convert`는 tico.convert를 mock한 routing test이므로 실제 fairseq Circle numerical export로 세면 안 된다.

`./ccex test -p`는 `test.performance.benchmark_perf`로 Llama-3.2-1B/3B를 로드하고 60/180초 및 1.01 size ratio threshold를 적용한다. CPU tiny 검증에는 실행하지 않고 축소 가능한 conversion-memory/artifact optimizer benchmark로 대체한다. 이는 원래 threshold 통과 증거가 아니다.

CI는 Python 3.12 Ubuntu 24.04에서 default Torch 2.13 full suite, oldest supported 2.11 smoke blocking, candidate 2.14 smoke nonblocking을 정책 모듈에서 resolve한다. UFMT와 Pylint/mypy는 별도 jobs다. requirements의 Transformers 5.18.0/datasets 5.0.1 및 CPU Torch/TorchVision 짝을 유지해야 한다.

## 주요 잔여 범위

Unit routing test의 mock 성공은 실제 evaluator/model/export 성공과 구분한다. CPU driver는 Llama/Qwen 중심의 candidate 내부 확인이며 `--baseline-repo` 없이 baseline regression 비교가 아니다. Gemma4/assistant는 별도 smoke 및 synthetic integration 결과를 사용해야 한다. Public evaluate CLI는 config pipeline을 실행하지 않으므로 Quant로 집계하려면 converted checkpoint가 필요하다. production helper+실제 evaluator에 local task만 연결한 harness는 metric pipeline은 검증하지만 remote dataset와 원래 CLI sample-limit routing을 검증하지 않는다.

## 조사 재현 명령

```bash
cd ${CANDIDATE_REPO}
git diff --stat 261bda28ec55fdac42f59346c950326e4065f8fa HEAD
git diff 261bda28ec55fdac42f59346c950326e4065f8fa HEAD
git log --since=2026-09-28 --until=2026-10-29 --oneline 261bda28ec55fdac42f59346c950326e4065f8fa
cat AGENTS.md tico/quantization/AGENTS.md test/AGENTS.md tico/circle/AGENTS.md
cat docs/development.md docs/torch_version_policy.md
cat tico/quantization/README.md tico/quantization/wrapq/README.md tico/quantization/recipes/README.md
cat tico/quantization/recipes/evaluation/README.md tico/quantization/recipes/export/README.md
cat .github/workflows/check-pr.yaml .github/workflows/check-pytorch-compatibility.yaml
cat infra/scripts/test_run.py test/pt2_to_circle_test/__init__.py test/performance/benchmark_perf.py
rg -n '^ *def test_' test/quantization/wrapq/test_lifecycle.py test/quantization/wrapq/observers/test_frozen_state.py test/quantization/wrapq/test_frozen_state_export.py
```

## 별도 실제 실행 보완

Head discovery만 수행한 `discovery_inventory.py` 결과: 4,760개 수집, 157개 decorator skip, loader error 0. wrapper family별 수집/decorator skip은 Llama 96/3, Qwen 209/12, Gemma4 422/68, assistant 52/0, fairseq 50/0이다. import 누락으로 family 전체가 빠지지는 않았다. 실제 runtime skip/실행 성공 수는 full-suite 로그를 기준으로 한다. 전체 test ID와 skip 사유는 `discovery-inventory.json`에 보존했다.

최신 base에서 `baseline_selective.py`로 public interface, folded GPTQ→PTQ handoff, CPU evaluation 세 module을 실제 실행했다. import 경로 `${BASELINE_REPO}/tico/__init__.py`를 assertion한 뒤 13개 수집/실행/성공, failure/error/skip 0 (unittest 1.045초, import 포함 wall time 8.132초). 각 revision에서 모델을 새로 생성하고 calibration/convert하며 head checkpoint는 base에 전달하지 않았다. PPL은 두 로컬 문장/13개 valid scoring token으로 Llama·Qwen·Gemma FP/noquant/Quant scoring 및 frozen 보존을 검증한다. base PPL 관찰값은 Llama FP 131.49905395507812 / Quant 130.86241149902344, Qwen FP 129.3563232421875 / Quant 128.7648468017578, Gemma FP 123.41830444335938 / Quant 120.72010040283203. Random tiny model 수치이며 모델 품질 판단 근거가 아니다.

```bash
python ${VALIDATION_ROOT}/run.py --name baseline-selective --cwd ${BASELINE_REPO} --timeout 600 -- python ${VALIDATION_ROOT}/baseline_selective.py
python ${VALIDATION_ROOT}/run.py --name discovery-inventory --cwd ${CANDIDATE_REPO} --timeout 300 -- python ${VALIDATION_ROOT}/discovery_inventory.py
```

전체 suite 최종 audit: 4,760개 수집/실행, 4,602 PASS, 158 skip, failure/error 0. unittest 987.679초, process 995.207초, exit 0. 모든 discovery ID의 verbose 결과를 연결하여 누락 0개와 `audit_ok=true`를 확인했다. Family PASS/skip은 Llama 93/3, Qwen 197/12, Gemma4 354/68, assistant 52/0, fairseq 50/0. decorator 이외의 동적 skip은 optional `plotext` 미설치 테스트 1개뿐이다. 157개 의도적 decorator skip의 상세 이유는 `full-suite-summary.json`에 있다. head/base checkout 모두 audit 시 tracked/untracked 변경이 없었다. nested unittest 실행이 로그 중간에 자체 `Ran 0 / OK(skipped=1)`를 출력하므로 집계 parser는 마지막 top-level Ran/result와 전체 ID 결과를 교차 확인한다.

`baseline-ppl-comparison.json`에는 각 revision에서 독립 생성한 tiny Llama/Qwen/Gemma FP/noquant/W4A16 PPL이 동일한 설정에서 정확히 일치함을 기록했다. 이는 축소 scoring loop의 재현 증거이며 full dataset 모델 품질 비교가 아니다.
