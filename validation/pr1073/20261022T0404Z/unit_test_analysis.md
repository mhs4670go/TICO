# PR #1073 unit/regression 실행 분석

검증 대상: `6b2f6a0511458d6a750e2a260d706080a78149dd`.
기준: `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`.

## 원본 focused 실행과 검증 인프라 수정

- `changed_unit_tests.log`: 변경된 test 파일 26개와 `runtime_independence`를 직접 module 경로로 실행했다. 314개 실행, error 11개, 종료 코드 1. 이 원본 결과는 그대로 보존했다.
- 8개 loader error는 재현 스크립트의 호출 방식 문제였다. 저장소는 `unittest discover -s test`를 표준으로 사용하며, 기존 unit test의 `quantization.recipes.optional_dependency_stubs` import가 이 discovery root를 전제로 한다. 직접 `test.quantization...` 모듈 호출에는 `test/`가 import root에 없었다.
- 제품 코드·기존 테스트는 수정하지 않았다. 보조 실행기를 `run_changed_discovery.py`로 분리하여 표준 discovery에 반복 `-k` 필터를 전달했다. import stub을 추가하거나 integration 경로를 우회하지 않았다.
- 기존 unit test의 `optional_dependency_stubs` 및 mock은 unit 범위의 기존 코드다. 이 결과를 pretrained/HF/Circle integration 완료 증거로 간주하지 않는다.

## 원본 실행에서 확인된 별도 오류

1. `test/quantization/recipes/test_export_llama.py:105,147`: `FakeLayerWrapper` fixture는 옛 mask/RoPE helper만 구현하고 config/self_attn 및 canonical rotary snapshot을 제공하지 않는다. 새 exporter의 `tico/quantization/recipes/export/llama.py:338` → `wrappers/llama/static_inputs.py:56` 호출에서 `AttributeError: 'FakeLayerWrapper' object has no attribute 'self_attn'`가 발생했다. mock fixture와 변경된 static-input 계약의 불일치다. 기존 테스트와 제품 코드 모두 수정하지 않았다.
2. `test/quantization/wrapq/wrappers/qwen_vl/test_eager_context.py:225`: full-text `torch.export.export(wrapped,...)`를 `prepared_static_export` 없이 호출한다. `tico/quantization/wrapq/wrappers/qwen_vl/quant_text_model.py:564`의 새 명시적 준비 요구에 의해 `ValueError`가 발생했다. 이 테스트는 baseline에 없다. 제품 export 경로 검증과 별도로 남긴다.

## 최소 수정 제안 (미적용)

- Llama export unit fixture를 새 static-input owner protocol(config, canonical rotary source)과 맞추고, artifact routing mock의 범위를 명시한다.
- Qwen full-text export unit test는 실제 `prepared_static_export(wrapped, max_seq_len=...)` 안에서 tracing하고, context 안에서 expected/actual을 동일한 static profile로 비교한다. 준비 없이 tracing하면 거부되는 별도 negative test를 유지한다.

Canonical discovery 재실행 및 전체 suite 결과는 후속 기록에 추가한다.

## 표준 discovery 재실행

`changed_unit_discovery.log`는 `unittest discover -s test -v`에 27개 변경/관련 module filter를 반복 `-k`로 전달한 결과다. 이름이 겹치는 인접 export/causal-LM 테스트도 포함되어 461개가 실행됐고, error 6개, skip 0개, 종료 코드 1이다. 앞선 8개 import-path 오류는 표준 discovery에서 없어졌다.

추가로 드러난 Llama 기존 테스트 3개는 `test/quantization/wrapq/wrappers/llama/test_quant_model_for_casual_lm.py:160,180,215`에서 prepared context 없이 full-model static adapter를 호출한다. 각각 `tico/quantization/wrapq/wrappers/llama/export_adapters.py:482,532`의 명시적 준비 검증으로 거부된다. 이 역시 테스트 계약 갱신이 필요한 실패이며 자동 PASS나 skip으로 바꾸지 않았다.

`baseline_llama_export_units.log`: baseline의 기존 Llama export unit/bundle 테스트 23개가 같은 interpreter/의존성에서 PASS(종료 코드 0)했다. `baseline_test_import.log`에서 baseline의 실제 `tico.__file__`과 동일 Torch/Transformers 버전을 확인했다. 이는 candidate의 두 routing fixture 실패가 baseline에도 있던 실패가 아님을 보여준다. 실모델 export 성공을 의미하지는 않는다.

`baseline_llama_causal_units.log`: baseline의 기존 Llama causal-LM 테스트 6개 모두 PASS(종료 코드 0). Candidate에서 prepared context 부재로 실패한 3개 테스트가 이 baseline 실행에는 통과했다. 따라서 5개의 기존 Llama 테스트 실패를 baseline 공통 실패로 숨길 수 없다. 의도된 API 계약 변경과 테스트 갱신 누락으로 분류하며, 제품 코드 결함 여부는 별도의 실제 checkpoint/Circle 검증 결과와 구분한다.

전체 표준 suite는 `full_unit_suite.log`와 `full_unit_suite.command.json`에 기록한다. 실제 CPU quota는 4core이며 visible logical CPU는 5개다. unit/regression 실행에 모델/데이터 다운로드는 허용하지 않도록 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 적용했다.

전체 suite의 skip은 개별 사유와 함께 원본 로그/JSON에 별도 기록하며 PASS 수에 포함하지 않는다. Timeout/중단 시 `Ran ... tests` 최종 요약이 없으면 `INCOMPLETE`로 남기고, 그 전까지의 개별 status line 수는 부분 진행 기록으로만 제공한다.

## Owning quantization suite 완료

`./ccex test -k quantization`은 2670개를 실행했고, error 9개, skip 141개, 종료 코드 1이었다. 테스트 시간 1020.806초, 명령 전체 시간 1099.719초다. Skip은 PASS가 아니다. 모든 원본 traceback은 `quantization_suite.log`에 있다.

오류 분류:

- 기존 static-contract 불일치 8개: 앞선 6개와 Llama attention export adapter 테스트 2개.
- `test/quantization/wrapq/wrappers/llama/test_quant_attention.py:480,528`은 explicit `attention_mask` 없이 static adapter를 호출한다. `export_adapters.py:255` → `utils/static_export.py:38`이 `ValueError: Static decoder export requires an explicit attention_mask.`로 거부한다. 사용자가 요구한 누락 host-input 거부 계약에 맞는 동작이므로 제품 검사를 완화하지 않았다. 최소 수정은 테스트에 실제 explicit mask/RoPE를 전달하고 별도 누락-input negative test를 유지하는 것이다. 원본 테스트는 미수정이다.
- `test/quantization/examples/test_gptq_qparam_handoff.py:181 (45초 설정:171)`의 full-model example child는 기존 45초 제한에 걸려 `subprocess.TimeoutExpired`가 발생했다(원본 로그 4704행 이후). 같은 한 테스트를 그대로 분리 재실행한 baseline과 candidate는 각각 33.394초, 35.758초에 PASS(종료 코드 0)했다. Timeout이나 assertion을 수정하지 않았다. 원본 owning-suite ERROR를 지우지 않고 자원 경합에 민감한 subprocess timeout으로 분류한다.

이번 owning-suite에는 실제 tiny model의 저장된 Llama Circle bundle 실행 및 teacher-forced logits/generation 테스트도 포함되어 통과했다. 이들은 실제 `export_llama_bundle`/reference runtime을 사용하나 pretrained 또는 NPU 검증을 의미하지 않는다. `test_fake_quantized_teacher_forced_logits_and_generation`, `test_quantized_artifacts_require_explicit_fake_quantize_mode` 등의 개별 결과를 원본 로그에서 확인할 수 있다.

`baseline_llama_attention_units.log`: baseline의 Llama attention unit 18개가 모두 PASS(종료 코드 0)했다. Candidate에서 explicit mask 부재로 실패한 2개도 baseline에서는 통과했다. 따라서 총 7개 기존 Llama 테스트의 static 계약 갱신 누락과 1개 신규 Qwen 테스트의 준비 누락을 분리하여 기록한다.

## 전체 표준 suite의 최종 상태

`./ccex test`는 1800.011초에서 설정된 1800초 제한에 걸려 종료 코드 124로 중단됐다. **INCOMPLETE**이며 PASS가 아니다. 마지막 실행 중 테스트는 `quantization.wrapq.test_same_padding_concat_export.CNNBoundaryQuantizedExportTest.test_same_padding_convolutions_do_not_emit_pad`다. OOM이나 의존성 import 오류로 중단된 것은 아니다. 가시적 CPU 5개이나 quota 4core였고, 초반 static checker와 CPU를 공유했으며 실제 Circle Conv3D reference 실행에 시간이 들었다.

전체 suite 로그에서 확인 가능한 개별 status line은 `ok` 2020개, `ERROR` 2개, skip 50개다. 경고가 status 출력 사이에 삽입된 테스트는 단순 line counter에 포함되지 않으므로 이 수치를 최종 tests-run 수나 전체 통과 수로 사용하지 않는다. 최종 `Ran ... tests` 요약이 없어서 `tests_run=null`로 기록했다. 2개 ERROR는 위에서 별도 재현한 Llama export fixture 실패다. 전체 suite의 wrapper 나머지 부분과 core `unit_test` 전체 완료를 주장하지 않는다.

Generated op/net E2E 구간은 396개 `ok`, 11개 skip, 0개 FAIL/ERROR status line으로 끝났다. 이 역시 전체 suite PASS 또는 모델/NPU 검증이 아니다.

| 실행 | 결과 | 새 실행 증거 |
|---|---|---|
| 최초 직접-module focused | FAIL, 314개/error11 | `changed_unit_tests.log`; 호출 인프라 문제 8개 포함, 원본 보존 |
| 표준-discovery focused | FAIL, 461개/error6/skip0 | `changed_unit_discovery.log` |
| Owning quantization | FAIL, 2670개/error9/skip141 | `quantization_suite.log`; 위 원인 분류 별도 |
| 전체 표준 | INCOMPLETE, exit124/1800초 | `full_unit_suite.log`, 최종 run-count 없음 |
| Baseline Llama export/bundle | PASS, 23개 | `baseline_llama_export_units.log` |
| Baseline Llama causal-LM | PASS, 6개 | `baseline_llama_causal_units.log` |
| Baseline Llama attention | PASS, 18개 | `baseline_llama_attention_units.log` |
| 45초 timeout 단일 테스트 재실행 | 양쪽 각 1개 PASS | baseline/candidate `*_qparam_timeout_probe.log`; 원본 suite ERROR는 유지 |

모든 테스트 실행의 cwd/실제 argv/환경(비밀 제외)/시작 시각/종료 코드/시간은 대응 `*.command.json`에 있다. `commands.sh`와 실행 스크립트, `unit_results.json`, `unit_failure_classification.json`을 함께 보존했다. 기존 테스트나 제품 파일은 수정하지 않았다.
