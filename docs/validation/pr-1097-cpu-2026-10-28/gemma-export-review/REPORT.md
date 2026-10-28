# Gemma4 generic export 실패 분석

대상 head `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`, base `261bda28ec55fdac42f59346c950326e4065f8fa`. 제품 코드는 변경하지 않았다. 원본 generic case의 export 실패는 보존하며 아래 보완 실행으로 덮어쓰지 않는다.

## 원본 실패와 분류

| Case | head 최초 오류 / 실제 CLI exit | 동일 baseline 결과 | 분류와 영향 |
|---|---|---|---|
| `gemma4_for_causal_lm` | `embedding`: indices는 Long/Int여야 하나 float32 `[1,16,64]`, exit1 | 같은 float32 embedding 오류, exit1 | 기존 smoke export ABI 결함. 실제 token numerical forward는 통과하지만 advertised Circle route는 실패. |
| `gemma4_for_conditional_generation` | `QuantGemma4Model: capability 'vision' is unavailable`, exit1 | `quant_model.py`의 `if multimodal_mask.any()` Dynamo data-dependent branch에서 실패, exit1 | 기존 잘못된 export ABI가 PR의 의도된 readiness guard에 더 일찍 걸림. head/base 모두 같은 호출 계약 위반이며, baseline 오류 문자열 차이만으로 PR 회귀로 분류할 근거가 없다. 실제 필요한 vision calibration이 실패한 증거도 아니다. |
| `gemma4_assistant_core` | `Sliding shared KV exceeds ... valid=10, sliding_kv_length=8`, exit1 | 같은 capacity 오류, exit1 | 기존 fixture 입력 길이/정적 capacity 불일치. 소스는 silent crop를 명시적으로 거부한다. |

원본 명령/traceback은 상위 `runs/wrapper-circle-<case>.json`, `logs/wrapper-circle-<case>.log`와 `wrapper-circle/<case>/artifact-checks.json`이다. base는 `runs/wrapper-baseline-<case>.json`, 해당 log, `wrapper-baseline/<case>/artifact-checks.json` 및 `wrapper-harness/baseline-summary.json`을 참조한다. 실행 import 경로가 head/base checkout을 가리키는지 각각 기록돼 있다.

## 재현으로 확인한 ABI 원인

`diagnose.py`는 실제 tiny HF 모델을 생성해 prepare→calibration3개→convert→정상 numerical forward를 실행하고, **원래 case의 export invocation을 eager로 직접 호출**한다. export/forward를 mock하지 않는다. 3개 case 모두 정상 numerical forward는 PASS이며, 잘못된 export invocation은 Dynamo 이전 eager 단계에서도 원본과 같은 의미의 오류를 발생시켰다. 모든 preexisting state contract는 전후 동일했다. 실행시간5.43초, 원래 process exit0은 이 진단의 기대된 negative 결과를 모두 수집했다는 뜻이며 export 성공을 뜻하지 않는다. 결과는 `diagnostic.json`이다.

두 Gemma4 top-level wrapper에는 `as_export_module`이 없다. `Gemma4ForConditionalGenerationCase.export_module()` (`cases/gemma4.py:2752`)와 `Gemma4ForCausalLMCase.export_module()` (`:2876`)은 그래서 원래 PTQ wrapper를 반환한다. 그러나 각 `export_input()` (`:2762`, `:2886`)은 precomputed tensor-only adapter용 `(inputs_embeds, per_layer_inputs, attention_masks, position_embeddings)`를 만든다. 이 case 메서드는 PR에서 바뀌지 않았다.

- Causal LM의 실제 HF-style forward 첫 인자는 `input_ids`다. float32 embeds가 그 위치에 들어가 embedding indices 오류를 낸다.
- Conditional generation의 실제 첫 네 인자는 `input_ids, pixel_values, pixel_values_videos, input_features`다. tiny 설정의 `per_layer_inputs=None`이므로 두 번째 값은 None이지만 세 번째 **attention mask dict가 pixel_values_videos에 들어간다**. PR은 이 non-None 값에 대해 미수집 vision capability를 올바르게 거부한다. case calibration `_sample()` 자체는 image placeholder ID10을 피한 token IDs0~9의 text-only 입력이며 vision을 관측하지 않았다. 오류를 숨기려고 vision calibration만 억지로 추가하면 잘못된 ABI는 그대로 남는다.

Assistant case는 `cases/gemma4_assistant.py:161`에서 `kv_len=10`, `:163`에서 `sliding_kv_length=8`을 선언한다. `export_input()`의 canonicalizer (`:206`)가 정확히 이 불일치를 거부한다. 기존 실제 Circle unit test `test_export_adapters.TestGemma4AssistantCoreExportAdapter.test_circle_shared_kv_inputs_use_consumer_compute_grid`는 올바른 capacity16/KV8 profile을 써서 기본 전체 suite에서 PASS했다. 이 unit test의 수치 계약은 출력4 grid step/rtol0이며, 보완 strict probe의1e-5/1e-4와 구분해야 한다.

## 수정 제안과 보완 범위

- 두 top-level Gemma case는 실제 adapter와 export input ABI를 한 owner에서 짝지어 구성해야 한다. 원래 HF token forward를 쓸 경우 올바른 keyword input과 static profile preparation을 제공한다. tensor-only route를 쓸 경우 생산 코드의 staged text/decoder adapter를 선택하고 해당 external-embedding/PLE 경로를 **convert 전에 실제 calibration**해야 한다. 임의 cast나 guard 우회로 해결할 문제가 아니다.
- Assistant는 실제 KV 길이를 수용하는 정적 capacity를 선언하거나, 처음부터 capacity 이하인 입력 profile을 선택해야 한다. 원본 오류의 silent crop 거부는 유지해야 한다.
- top-level Gemma generic case 실패가 family 전체 누락을 뜻하지 않는다. 전체 suite와 wrapper의 text attention/prefill/decode/vision/PLE 등 실제 실행 결과를 별도로 읽어야 한다. 반대로 그 성공이 이 세 generic CLI export의 성공을 뜻하지도 않는다.

추가 assistant profile은 `assistant_capacity16.py`에서 **validation runtime registry의 capacity만16**으로 설정한다. original `kv_len=10`, fullcapacity16, 모델, calibration/sample 경로, assertion, tolerance는 유지한다. 원본 제품 파일과 원본 실패는 그대로다. `assistant-complete.json`이 생성되면 해당 실제 export의 결과/로그를 확인한다. `saved/structural/runtime/parity/frozen_unchanged`는 `assistant-capacity16/artifact-checks.json`에 따로 기록한다. 이 추가 profile은 원래 capacity8 case를 PASS로 바꾸는 결과가 아니다.

재현:

```bash
cd ${CANDIDATE_REPO}
python ${VALIDATION_ROOT}/run.py \
  --name gemma-generic-export-diagnostic-rerun --timeout 120 -- \
  python ${VALIDATION_ROOT}/gemma-export-review/diagnose.py
python ${VALIDATION_ROOT}/run.py \
  --name assistant-capacity16-circle-rerun --timeout 240 -- \
  python ${VALIDATION_ROOT}/gemma-export-review/assistant_capacity16.py \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case gemma4_assistant_core --device cpu --export circle \
  --strict --no-plot --output-dir ${VALIDATION_ROOT}/assistant-capacity16-rerun \
  --report-json ${VALIDATION_ROOT}/assistant-capacity16-rerun.json
```

두 번째 재현의 output은 아직 없는 고유 경로를 선택한다. evaluator scoring/전체 모델 품질/NPU integer 검증은 이 분석의 범위가 아니다.

## 보완 harness 초기 오류 이력

`assistant-capacity16-circle` 및 `assistant-capacity16-circle-base` 최초2회는 **모델/export 실행 전** `ModuleNotFoundError: No module named 'observe'`로 실패했다. `export_case.py`가 관측 helper `observe.py`를 sibling import하는데 이를 `runpy.run_path()`로 부르는 validation wrapper가 script directory를 import path에 넣지 않은 harness 자체 오류다. 최초 wrapper 소스는 `assistant_capacity16_initial.py`에, 원본 logs/runs 및 marker는 `assistant-initial-harness-error.json`에 보존했다. 이2회는 제품 export 실패나 baseline 비교 증거로 세지 않는다.

검증 wrapper에서 helper directory만 sys.path에 추가했고, 제품 코드/검증 assertion/입력 값/수치 tolerance는 변경하지 않았다. 새 `assistant-capacity16-circle-r2` 및 `...-r2-base` 이름과 고유 출력 디렉터리로 재실행하며 최종 실제 stage evidence를 사용한다. 수정된 실행은 root baseline CPU export queue 완료 뒤 순차로 연결했다.

## Assistant valid-capacity 실제 export 최종 결과

capacity16 보완 profile을 head와 base에서 각각 독립적으로 prepare/calibrate/convert하여 실제 inspector/Circle export를 실행했다. head16.883초, base15.668초. **두 revision 모두 inspector exit0, 파일 저장 PASS, 재로딩 구조 검사 PASS, built-in reference runtime PASS, frozen 상태 보존 PASS**다. 원래 capacity8 fixture의 export FAIL은 그대로 남는다.

`assistant-capacity16-circle-r2`/`...-r2-base` 명령 자체 exit1은 추가 strict parity probe 실패 때문이다. 동일 `atol=1e-5, rtol=1e-4`에서3개 출력의 최대 오차는 head/base 모두 `[1.9803643226623535e-05, 0.00024640560150146484, 1.1280179023742676e-05]`이며, 앞2개는 FAIL, 세 번째는 PASS다. assertion/tolerance는 완화하지 않았다. 이 수치를 기존 unit test의4-grid 계약으로 재판정해 strict PASS라고 부르지 않는다.

실제 input/출력shape·dtype·각 tensor byte hash와 Torch/Circle 비교 결과가 head/base에서 모두 정확히 같고 모든 weight hash도 같다. 따라서 이 strict 차이는 새 PR numerical 회귀의 증거가 아니다. serialized file SHA는 서로 다르며, 전체 observer inventory도 stateless Identity/사용하지 않는 sparse-head observer topology 변화로 다르므로 전체 grid inventory 동등성까지 주장하지 않는다. `assistant-capacity16-comparison.json`에 양쪽 증거와 비교 assertion을 저장했다.

- Head evidence: `assistant-capacity16-r2/artifact-checks.json`, actual `.circle` file in same directory.
- Base evidence: `assistant-capacity16-r2-base/artifact-checks.json`.
- Original commands/exits/logs: 상위 `runs/assistant-capacity16-circle-r2*.json`, `logs/assistant-capacity16-circle-r2*.log`.
- 완료 marker: `assistant-complete.json` (원본 초기 harness 오류 marker와 분리).
