# 실제 Circle 검증 기록

대상 candidate `6b2f6a0511458d6a750e2a260d706080a78149dd`, baseline `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`. 모든 worker는 동일 가상환경(Python 3.12, Torch 2.13 CPU, Transformers 5.18)에서 지정한 checkout의 실제 TICO를 import했다. 생성 fixture는 CPU float32의 신뢰 가능한 frozen full `PTQWrapper(nn.Module)`이며 pretrained checkpoint가 아니다. Worker에서 calibration, dtype cast, 재양자화를 수행하지 않았다.

## 원본 결과와 구성 요소

| 모델/프로파일 | 실제 완료 범위 | 원본 strict 결과 | 별도 확인한 구성 요소 |
|---|---|---|---|
| Llama tiny random PTQ, native64/export8/eager12+4 | 6 Circle stage × baseline before/candidate before/candidate after = 18; bundle host 포함 | old/new FAIL, isolation FAIL, exit1 | 초기 frozen state 동일, eager 후 weights/observer/qparams/mode 보존. Candidate 전후 모든 Circle 구조·상수/qparams·host·input/output 배열 exact. Strict FAIL의 Torch/Circle 오차는 별도 유지. |
| Qwen3-VL tiny random PTQ, native64/export8/eager12+4, grid[1,2,2] | 9 Circle stage × 3 phase = 27; 실제 vision 두 출력 포함 | old/new FAIL, isolation FAIL, exit1 | Old/new와 candidate 전후 구조·상수/qparams·host·모든 input/output 배열 exact. Frozen state 보존. Vision/embedding/MMembedding/deepstack/head Torch/Circle exact; decoder strict parity 실패. |
| Gemma4 tiny random PTQ, native64/export8, image70 (4 valid visual tokens, 280 patch capacity), monolithic vision + text per-layer requested | Baseline·candidate 모두 실제 vision conversion 후 reference 실행 시 rank2 RMS_NORM 거부 | ERROR, exit2; comparisons_complete=false | 전체 multimodal/per-layer 비교·eager stress 미완료. 텍스트 경로는 별도 harness로 분리 진행. |
| Llama tiny full model, native64/export8/eager12+4 | full `model.q.circle` 1 stage × 3 phase, 각 logits + 4 KV 출력 | old/new FAIL, isolation FAIL, exit1 | Old/new·candidate 전후 graph/host/모든 배열 exact 및 frozen state 보존. 세 phase 모두 logits Torch/Circle 최대 0.0055593401, 모든 KV exact. |
| Llama NEW native16384/original8192, export8/eager4096+4 | 실제 full Circle 1 stage × before/after 2 phase, 각 5 출력 | isolation FAIL, exit1 | checkpoint capacity를 수정하지 않고 처음부터 16384로 생성. >2048 실제 eager/cache decode 및 frozen state 보존. 전후 모든 graph/배열 exact이나 Torch/Circle strict 실패 유지. |
| Qwen full-text + 실제 PNG eager 1회 | `full_text_regression.q.circle` × 3 phase, 각 hidden 출력 | old/new PASS, isolation PASS, exit0 | 모든 Torch/Circle 오차 0, frozen state 보존. 누락 저장 폴더만 고친 runner 복사본이며 원본 ERROR 보존. Full multimodal static graph가 아니다. |
| Gemma 독립 text per-layer, native64/export8/eager12+4 | text 4 stage × 3 phase = 12, 30 probe/90 출력 | old/new FAIL, isolation FAIL, exit1 | 모든 old/new·candidate 전후 Circle binary/contract/qparams/host/숫자 exact, frozen state 보존. Layer0 prefill hidden 0.3541484475; 모든 KV/다른 stage exact. Main vision ERROR는 유지. |
| Gemma full-text + 실제 PNG eager 1회 | `full_text_regression.q.circle` × 3 phase, 각 hidden 출력 | old/new FAIL, isolation FAIL, exit1 | 모든 graph/host/배열 exact, frozen state 보존. 세 phase 모두 Torch/Circle hidden 최대 0.0869361162. 필수 static mask를 제공한 runner 복사본 사용, 원본 ERROR 보존. |

각 완료 phase의 `independent_evidence_inventory.json`은 실제 `.circle` graph I/O 개수와 모든 probe의 input/Circle/Torch 배열 개수, NaN/Inf, Torch parity 항목 수를 독립 확인했다. Llama/Qwen 모두 stage 누락 및 숫자 증거 누락은 없었고 모든 배열은 finite였다. 원본 comparator의 evidence 검증 허점은 `COMPARATOR_GAPS.md`에 별도 FAIL로 남겼다. Torch/Circle parity 실패를 제외한 구조/배열 일치를 전체 strict PASS로 바꾸지 않았다.

## Llama 의도적 RoPE 차이와 다른 실패

`tiny-llama-old-new-v2/llama_rope_diagnosis.json`은 baseline/candidate 실제 Circle에 동일한 외부 input/mask/cos/sin을 넣어 모든 stage/probe 출력이 exact임을 확인했다. Candidate `llama_host.npz`의 각 layer cos/sin은 동일 config로 독립 생성한 정상 HF rotary와 exact였다. Baseline-vs-HF는 cos 최대 `4.4465065e-5`, sin 최대 `0.0068924397`; 위치와 원소 수를 JSON에 저장했다. Mask fill·frozen weights/qparams/mode 및 Circle graph I/O는 변하지 않았다. 이 **host RoPE 및 그 입력에서 유도된 차이만** `EXPECTED_NUMERICAL_CHANGE`의 근거가 있다. 원본 old/new FAIL은 유지한다.

별도 baseline strict parity 실패는 RoPE로 설명하지 않는다. `PARITY_DIAGNOSIS.md`와 `ROUNDING_AND_PAST_DIAGNOSIS.json`에서 실제 softmax quantization grid의 ties-even(Torch) vs ties-away(Circle) 한 step 차이를 재현했다. 원래 fixture의 cached-decode 관측 부족도 확인했다. 초기 Llama delta-K/V는 모든 probe exact였다. 이 fixture의 HF config는 작은 명시적 native capacity64와 원본 llama3 scaling original8192를 사용하여 HF warning이 있으며, 이는 pretrained/native context 모델 검증으로 일반화하지 않는다. 유효한 native16384/original8192의 새 candidate fixture와 full/long-eager 검증은 별도 결과다.

Qwen의 `QWEN_PARITY_DIAGNOSTIC.md/json`은 cache-warmed full-model DynamicCache calibration이 static tuple 전용 past observer를 채우지 않는 것을 확인했다. Hidden 최대 `0.0447399`, delta-K 최대 `0.0416335`, delta-V exact였다. Prefill/delta-K 오차의 모든 원인은 미확정이며 Llama RoPE와 관계가 없다. 두 checkout에서 그래프·배열이 exact이므로 관측된 strict mismatch는 baseline에도 존재하지만, 수치 검증 완료/PASS로 표시할 수 없다.

## Gemma vision 제약과 미적용 제안

Baseline 및 candidate에서 같은 실제 예외를 재현했다.

`UnsupportedCircleOperatorError: RMS_NORM supports rank 3 or 4 inputs, found shape (4, 32)` — `tico/circle/runtime/kernels/nn.py:620`, vision graph operator102, `Gemma4VisionModelPrefillExportAdapter`의 projected pooled features.

제품 수정은 적용하지 않았다. 최소 수정 검토안은 rank2 pooled-feature RMSNorm을 지원 rank3으로 legalize한 뒤 기존 2D I/O로 복원하거나, rank2 reference semantics를 명시적으로 지원하고 별도 정확성 검증을 추가하는 것이다. 단순히 거부 조건을 없애 성공으로 보고하지 않는다.

Opt-in `validate_static_layout=True`의 `image_embeds.shape[1]` token-count 오류는 `FIXTURE_INFRASTRUCTURE.md`에 별도 실패로 기록했다. 기본 False profile은 canonical grid/positions·image placeholder placement·실제 feature shape·checkpoint contract를 별도로 검증한 독립 case다.

## 재현과 입력 출처

실행 명령·cwd·stdout/stderr·exit code는 `commands.sh`, `*.command.json`, `*.log`에 기록했다. 각 frozen checkpoint `.provenance.json`은 source checkout SHA, 생성 코드 SHA256, native capacity, 전체 HF config, dtype, quantization/vision profile, calibration calls, checkpoint SHA256을 기록한다. 원본 실패 후 fixture/검증 인프라 수정 내역은 `FIXTURE_INFRASTRUCTURE.md`와 원본 소스 revision, copied-runner diff/hash에 남겼다.

`png_circle_inputs/`는 실제 생성 PNG를 정상 HF image processor로 변환한 custom kwargs와 processor/config/checkpoint/input hash를 포함한다. 파일 준비는 모델 실행 또는 평가 완료의 증거가 아니다. Custom-image full-text probe는 별도 결과를 확인해야 하며, 한 번의 image forward를 linked-cache multimodal decode로 보고하지 않는다.

## Qwen full-text + 실제 PNG 독립 결과

`tiny-qwen-full-text-png-dirs-retry/summary.json`: **PASS / exit0**, strict `--require-torch-parity`, old/new 및 candidate before/after 모두 완료했다. 실제 `full_text_regression.q.circle` 1 stage × 3 phase이며 각 graph는 text `input_ids`에서 `last_hidden_state`로 가는 full text wrapper다. 세 phase의 Torch/Circle 최대 절대 오차는 모두 **0**이었다. 독립 I/O/finite audit도 PASS이며 graph/상수/qparams/host/input/output arrays가 exact이다.

Candidate eager는 `png_circle_inputs/qwen_4x4.png`를 정상 HF `Qwen2VLImageProcessor`(patch2/temporal2/merge2)로 처리한 `[4,24]` pixels, grid `[1,2,2]`와 원래 고정 token/position kwargs로 **전체 VLM forward 1회**를 실행했다. Frozen state를 보존했다. PNG, processor config 및 입력 hash는 `png_circle_inputs/inputs_manifest.json`에 있다. `--eager-inputs`가 자동 text prefill/decode stress를 대체하며 cache를 호출 간 연결하지 않는다. 이 PASS는 full-text static graph가 한 번의 image eager 실행 전후 격리된다는 증거이고, multimodal full export·vision static isolation·multimodal cached decode 또는 pretrained 품질 PASS가 아니다. 별도 main per-layer strict FAIL은 그대로 유지한다.

첫 원본 runner의 missing-directory ERROR는 `tiny-qwen-full-text-png-original/`에 유지하며, retry의 한 줄 directory-creation 수정은 `export_regression_full_text_dirs.diff`에 명시했다.

Gemma 실제 vision 실패 artifact는 `gemma-vision-failure-artifact/vision_prefill.q.circle`에 저장했다 (165504 bytes, SHA256 `4cb5d8deecfef8cf41338211d4d298170048ef0bf64f1893d4b85b6442b0af5d`). 실제 serialized graph verify와 finite Torch features `[4,32]` 증거는 저장했지만 reference 실행이 rank2 RMS_NORM에서 ERROR였으므로 Circle numerical PASS는 아니다.

## Full-model·긴 eager 및 Gemma text 보완

`tiny-llama-full-shape-retry/README.md`는 원본 full exporter의 calibration shape-input 누락 ERROR와 최소 copied-runner 보완을 구분한다. 같은 frozen checkpoint를 읽고 새로운 calibration 없이 CPU long shape example만 제공했다. Baseline에도 동일하게 존재하는 logits strict 오차를 유지한다. `llama_long_eager_analysis.json`의 native16384 fixture는 원본 runner로 full Circle capacity8을 실제 eager4096+4 전후 실행했으며, 상태와 export component는 동일했다. Static2048 export나 pretrained 대형 모델 검증은 이 결과에 포함되지 않는다.

`GEMMA_TEXT_STAGE_VALIDATION.md`는 vision 실패를 우회해 main run을 PASS로 만드는 대신 독립 text-stage harness의 원본 strict FAIL을 별도 저장했다. `GEMMA_TEXT_PARITY_DIAGNOSIS.md`의 실제 baseline replay는 query rotate-half의 frozen `q_x1/q_x2` grid가 serialized SPLIT/CONCAT에서 사라져 q-cat 2원소 최대 0.4879152775 → 최종 hidden 0.3541484475 차이가 생긴다는 경계를 확인했다. 양 source의 관련 converter pass는 동일하다. 최소 수정 방향은 split/getitem producer domain과 서로 다른 half-grid requantization을 명시적으로 보존하는 것이며, 실제 제품 수정과 수정 후 검증은 수행하지 않았다. 이는 Llama RoPE의 의도적 차이가 아니다.

Gemma full-text 원본 probe는 필수 `attention_mask` dict를 누락하여 정상적인 static 입력 계약 거부로 ERROR였다. 별도 복사본 `export_regression_gemma_full_text_masks.py`는 기록된 frozen checkpoint의 full/sliding template을 trace 전에 `[1,1,8,8]` float32로 잘라 registered buffer에 보관하고 같은 mask를 Torch/Circle 및 양 source에 제공했다. `recorded_host_masks.json/.npz`와 diff/SHA를 결과 폴더에 저장했다. 이 profile의 sliding window는8이므로 seq8의 두 mask 값이 같은 것은 정상이며, sliding cutoff나 더 긴 static mask 검증으로 일반화하지 않는다. Retry는 실제 3 phase/3 stage/3 출력 완료 후 strict FAIL이었다. `GEMMA_FULL_TEXT_DIAGNOSTIC.md/json`은 각 phase의 finite float32 `[1,8,32]` 256원소를 확인했다. 최대 차이 위치 `[0,4,10]`은 Torch `1.6952527761`, Circle `1.7821888924`이며 strict 기준 위반은92원소다. 109 operators/195 tensors/149 qparam metadata/81 constants 및 모든 old/new·candidate 전후 배열은 동일했다. 정확한 full-text parity 원인은 별도 확정하지 않았고 Llama RoPE에 귀속하지 않는다. 독립 감사에서 모든 입력/출력 개수와 finite를 확인했고, 전체 VLM의 실제 PNG eager 1회 후 frozen state도 보존했다. Multimodal linked-cache decode 및 vision static isolation은 미완료다.

최종 기계 판독 요약은 `circle_validation_summary.json`이며 재생성 명령은 `build_circle_validation_summary.command.json`에 있다. 이 B2 tiny-fixture 범위에서 **실제 완료한 Circle stage는 총 68개**다(실패한 vision stage와 별도 pretrained run 제외). 단계 수는 PASS 수가 아니며 위 원본 FAIL/ERROR를 그대로 유지한다.
