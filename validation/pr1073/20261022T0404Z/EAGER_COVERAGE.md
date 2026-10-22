# Eager 검증 요구사항과 기존 테스트의 대응

검토 대상: `6b2f6a0511458d6a750e2a260d706080a78149dd`.
이 문서는 코드/테스트의 커버리지 지도이며 실행 성공 증거는 개별 실행 로그와 JSON이다.

| 요구사항 | 기존 증거 파일/라인 | 검증 수준과 한계 |
|---|---|---|
| calibration/export/eager cap 독립, null native context, 명시 cap, legacy clamped checkpoint 거부 | `test/quantization/recipes/test_eager_evaluation.py:90`, `:123`, `:141` | 작은 dummy config/모델 unit. 실제 checkpoint 아님 |
| config/RoPE 보존, 모델/generation cache 전달, backend 없이 동작 | 같은 파일 `:151`, `:188`, `:235`, `:278`, `:294` | prepare 경로는 실제, HF loader와 evaluator 호출은 mock |
| Llama native S×S 할당 방지, scaled RoPE와 padding/position, 4-step cached decode | `test/quantization/wrapq/wrappers/llama/test_eager_context.py:69`, `:97`, `:127` | 로컬 config로 생성한 실제 tiny HF 모델. 이름의 pretrained는 pretrained 가중치 검증이 아님 |
| Llama calibration보다 긴 입력과 frozen observer, static delta-KV/torch.export | 같은 파일 `:165`, `:195` | real tiny HF/PTQ/Torch Export. Circle 검증 아님 |
| Qwen native template 할당 방지, mRoPE/padding, calibration보다 긴 입력/cache, cache policy | `test/quantization/wrapq/wrappers/qwen_vl/test_eager_context.py:65`, `:92`, `:126`, `:165`, `:180` | 실제 tiny HF text 모델. 임의 mRoPE 위치는 이미지 encoder 실행이 아님 |
| Qwen full text 및 static stage Torch Export | 같은 파일 `:201`, `:229` | 실제 torch.export. Circle 또는 multimodal export 아님 |
| Gemma bounded static template보다 긴 eager, sliding cache offset/shared-KV/PLE, calibration보다 긴 quant 입력 | `test/quantization/wrapq/wrappers/gemma4/test_quant_text_model.py:214`, `:253`, `:304`, `:353` | 실제 tiny HF text/PTQ. image/video 평가 아님 |
| PPL window와 generation cap 분리 | `test_eager_evaluation.py:141`; `test_evaluation_target_adapters.py:94`, `:174`; `test_qwen_adapter.py:518` | 설정 분리는 unit. raw/chat-prefix routing은 mock evaluator. 실제 metric 완료 증거 필요 |
| Video-MME result cache와 KV cache 분리, FP/Quant budget 일치 | `test/quantization/recipes/test_video_mme_evaluation.py:197`, `:213`, `:229` | mock evaluator routing unit. 실제 video 평가 아님 |
| calibration/force_export 평가 거부 | `test_eager_evaluation.py:210`, `:229` | 실제 validation + dummy QuantModule leaf |
| export adapter return_type/필수 host 입력 거부 | `test/quantization/wrapq/utils/test_static_export_inputs.py:177`, `:185`, `:361`, `:405` | 주로 synthetic owner/layer unit. 실제 HF stage는 별도 integration 필요 |
| prepared_static_export 정상/예외 상태 복구 | 같은 파일 `:129`, `:141`, `:148`, `:156`, `:165` | synthetic owner unit. real owner 예외 경로는 보충 스크립트 |

## 보충 검증

`supplemental_eager_state.py`는 제품 소스를 수정하지 않고 기존 fixture의 **실제** Transformers tiny 모델과 affine PTQ를 정상 import한다. 세 family에 calibration=3/export=6/eager-prefill=12/decode=4/native=32를 적용한다. 모든 named parameter, named buffer, observer state/enable flag, QuantModule mode/profile/return policy를 전후 SHA256으로 비교한다. null/native 및 명시 cap, cache 설정, calibration/forced-static 거부, Llama/Qwen actual-owner prepared-static 예외 cleanup을 실행한다. 모델 다운로드, import stub, mock compiler/runtime을 사용하지 않는다. 생성된 파일은 benchmark/pretrained/실제 이미지 평가나 Circle 실행 증거가 아니다.

## 잔여 범위

기존 unit만으로는 FP/Quant pretrained 평가의 tokenizer/processor/revision/sample ID/prompt parity와 실제 완료 metric, real-image/video 평가, pre-PR old/new Circle equivalence, >2048 checkpoint isolation을 입증할 수 없다. 기존 frozen observer 테스트가 weight/profile 전체를 snapshot하지 않는 점은 보충 스크립트로 다룬다.

PPL 계산 자체의 기존 검증은 `test/quantization/wrapq/utils/test_metrics.py:206` (short-stride invariance), `:384` (chat-prefix overlapping windows vs full-logits masked CE reference), `:509` (fresh token 수 및 scoring use_cache=False)에도 있다. 이 테스트들은 작은 DummyLM/Tokenizer를 사용하므로 pretrained Llama/Gemma metric 완료 증거와 구분한다.

실제 작은 이미지 전처리/vision 경로 보충은 `generated_image_eager.py`로 제공한다. deterministic PNG를 저장하고 원격 pretrained processor config 없이 로컬 HF image processor를 생성한다. 목표는 HF/NO_QUANT 이미지 forward parity와 실제 PTQ image-prefill+4-step connected text decode/frozen 상태 검사다. 실행 결과는 generated_image_eager 실행 로그 및 JSON을 따라야 하며 계획된 검사 자체를 PASS로 해석하면 안 된다.

## 새로 실행한 완료 증거

| 실행 | 결과 | 범위 |
|---|---|---|
| `gemma_internal_multimodal_units` | PASS: 12 tests, skip 0, exit 0 | `RUN_INTERNAL_TESTS=1`, 실제 tiny HF vision/text/PTQ + synthetic patches |
| `qwen_tiny_multimodal_units` | PASS: 23 tests, skip 0, exit 0 | 실제 tiny HF image/video patch forward 및 calibration, 실제 영상 benchmark 아님 |
| `supplemental_eager_state` | PASS: Llama/Qwen3-VL/Gemma4, exit 0 | 실제 tiny HF text/PTQ, calibration=3/export=6/eager-prefill=12+decode4/native32 |

Frozen-state exact hash 비교 대상: Llama parameters20/buffers572/observers136/quant_modules57, Qwen parameters26/buffers644/observers160/quant_modules65, Gemma parameters28/buffers706/observers166/quant_modules73. 각 observer의 통계/qparams/enable flag와 모든 QuantModule mode/profile을 포함한다. Llama/Qwen 실제 static owner에 대한 예외 cleanup도 PASS; Gemma prepared-static 예외 cleanup은 이 보충 스크립트 범위에 없다. 원본 로그와 command JSON, `extra_multimodal_unit_results.json`, `supplemental_eager_state/summary.json`을 참조한다.

`generated_image_eager`는 exit 0으로 완료되었다(124.585s). 실제 저장 PNG와 로컬 생성한 HF processor로 전처리한 입력을 사용했다. Qwen3-VL NO_QUANT vs HF 최대 절대오차 2.4437904357910156e-6, Gemma4 1.430511474609375e-6으로 atol=1e-5/rtol=1e-4 검사를 통과했다. Quant vs FP hidden-state 최대 절대차는 각각 0.2602647542953491/0.20679152011871338이며 동일성을 요구하지 않았다. 두 모델 모두 image prefill 다음 4회 연속 KV cache decode를 실제 실행하고 출력 finite 및 frozen 상태 불변을 검증했다. 원격 pretrained 모델, 자연 이미지 benchmark 점수, tokenizer/prompt 평가, Circle export isolation은 이 실행의 범위가 아니다.

## 구현 참조와 legacy 테스트의 경계

- recorded native/default 및 legacy clamped checkpoint 초과 요청 거부: `tico/quantization/recipes/sequence_length.py:41` (특히58행). export capacity는69행, 독립 PPL window는84행이다. 현재 unit의 clamped checkpoint는 실제 옛 pickle을 읽지 않고 `config.max_position_embeddings=2048`로 모사한다(`test_eager_evaluation.py:123`). 실제 pre-PR checkpoint loading은 export 검증 결과를 별도로 따라야 한다.
- 공통 KV-cache default true 및 model/text/generation config 전파: `tico/quantization/recipes/evaluation/eager.py:113`. CALIB/force_export 거부는101/107행이다.
- Video-MME cache 구분: Qwen adapter `qwen3_vl.py:502`의 `evaluation.use_cache`→`kv_cache`,503행의 `evaluation.videomme.use_cache`→결과 cache 경로; Gemma adapter `gemma4.py:481`/482도 같다. 대응 unit은 `test_video_mme_evaluation.py:213`이다.
- Llama PPL 설정 `evaluation.perplexity`, optional stride는 `adapters/llama.py:352`/375. VLM PPL 설정 `evaluation.ppl`, stride는 `adapters/qwen3_vl.py:541`/549. Gemma raw/chat-prefix 보존은 `adapters/gemma4.py:520`/531, PPL-specific cap은543행이다. scoring `use_cache=False`는 `wrapq/utils/metrics.py:408`로 명시된다.
- `test_rope_consistency.py:190`의 migration은 **현재 candidate에서 만든 random tiny 모델**의 `_export_rotary_emb`/`_export_rope_contract_version`를 수동 삭제하고 legacy template을 넣어 모사한다(200행). freeze 이후 `_state_contract`와 parameter identity 보존, per-layer/full RoPE 일치, 재-migration idempotence를 검사한다. baseline 커밋에서 실제로 직렬화한 모듈의 호환성 증거는 아니다.
- `test_eager_context.py:266`은 candidate 모델의 `rotary_emb`를 수동 삭제한 뒤 실제 torch.save/load를 수행한다. serialization 자체는 실제지만 역사적 baseline에서 저장한 체크포인트는 아니다. `test_static_export_inputs.py:233` 역시 synthetic owner 속성 삭제로 missing/legacy provenance를 검사한다. 이 세 검사는 true pre-PR checkpoint export/migration 회귀와 분리한다.

여기서 native capacity32는 이 tiny fixture의 기록된 설정이며, 2048/4096 토큰이나 pretrained native context 검증으로 일반화하지 않는다. 생성 PNG 검증에는 자연 이미지 benchmark, tokenizer/chat-template scoring, Video-MME 실제 영상/metric 실행이 포함되지 않는다. 전체 changed-unit 실패와 old/new comparator의 원래 결과는 이 추가 PASS들에 의해 변경되지 않는다.

## Normal image generate의 별도 실제 실행

후속 `image_generation_probe`에서 Gemma4 FP/Quant 정상 generate2tokens는 PASS이고 raw frozen 상태 변화0이었다. Qwen3-VL은 FP2tokens PASS지만 Quant는 Transformers5.18의 `get_experts_implementation` API 누락으로 FAIL이다. 앞선 manual cached-step/forward PASS를 normal generate PASS로 일반화하지 않는다. 원래 baseline constructor로 만든 독립 작은 PTQ fixture에서도 동일 Qwen Quant 실패를 실제 재현해 preexisting 호환성 오류로 확인했다. 동일 source HF parameter70개 해시와 FP tokens는 일치하지만 독립 baseline calibration이므로 frozen-checkpoint export equivalence가 아니다. 원래 candidate checkpoint를 baseline에 직접 읽힌 별도 시도는 legacy mask 누락으로 먼저 FAIL했고 이 증거도 유지했다. `IMAGE_GENERATION_FINDINGS.md`, `image_generation_analysis.json`과 원본 세 실행 로그/JSON을 참조한다. 제품 accessor delegation은 제안만 했다.

## >2048 실제 eager + full Circle 보충

원래 native16384로 새로 생성한 tiny random Llama frozen checkpoint에서 actual prefill4096+cached decode4와 양쪽 full Circle export/reference 실행을 완료했다. static capacity는8이며 2048-token static export 검증이 아니다. 전후 binary/graph/constants/qparams/I/O/모든 실제 output은 정확히 같고 frozen state는 보존됐다. 그러나 strict Torch/Circle 수치 parity가 실패해 원래 isolation 실행과 독립 audit의 exit1/FAIL을 유지했다. `LLAMA_LONG_EAGER_ISOLATION.md`, `llama_long_eager_analysis.json`, run의 `independent_full_evidence.json` 참조. 기존 checkpoint config를 늘리지 않았으며 제품/runner 수정은 없다.

## Gemma 실제 text-stage Circle 추가 실행

원래 multimodal vision RMS_NORM rank2 ERROR를 보존한 채 독립 명시적 text decoder stage4개를 baseline/candidate-before/candidate-after에서 실제 Torch/Circle 실행했다. 30probes의 hidden/K/V90개 출력이 완전하고 finite이며 decode delta-KV 길이1을 독립 확인했다. candidate eager qtext12+4도 실제 완료되고 전체 frozen 상태 변경0이다. old/new Circle·host·수치는 정확히 같지만 TorchFX는3ops씩 감소했고, 후보 전후에는 FX까지 같다. layer0 sliding-prefill hidden Torch/Circle maxabs0.3541484475 때문에 strict old/new/isolation은 FAIL(exit1); 나머지3stage는 모든probe/출력에서 maxabs0이다. 이는 main vision ERROR의 PASS 대체가 아니다. `GEMMA_TEXT_STAGE_VALIDATION.md`, `gemma_text_strict_comparison.json` 참조.
