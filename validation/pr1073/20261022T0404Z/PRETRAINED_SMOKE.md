# Tiny pretrained Llama 검증

검증 모델은 `Maykeye/TinyLLama-v0`의 4,621,392 parameter 모델이며, revision은 `298338802ab94432b917bcce11382aa151aee50f`이다. Llama-3.2 또는 다중 billion parameter 모델 결과가 아니다. 모델 자산은 선택한 safetensors/tokenizer/config 7개, 총 11,772,381 bytes만 다운로드했다. 원래 CPU/GPU 환경에는 pretrained 모델 및 frozen checkpoint가 없었다.

원본 smoke checkpoint는 새 candidate 코드에서 네 개의 고정 합성 문장, 각 32 tokens로 실제 PTQ를 실행해 만들었다. `npu_export`, linear/embedding/LM-head uint8, activation int16, norm weight int16이며, CPU float32의 full `PTQWrapper` checkpoint이다. Recorded capacity 2048을 변경하지 않았다. SHA256: `5f8b5a9b5f7e8ff182c6fc96d52f232a9958424f68c09039fd9452e101b17345`. 생성 명령과 입력은 `pretrained_ptq_prepare.*`, `pretrained_calibration.json`, `pretrained_checkpoint_identity.json`에 기록했다.

| 검사 | 결과 | 실제 증거 |
|---|---|---|
| FP recipe PPL CLI | PASS | `pretrained_fp_cli_ppl.log`, exit 0; PPL 6019.25 |
| Quant recipe PPL CLI | PASS | `pretrained_quant_cli_ppl.log`, exit 0; PPL 5889.06 |
| 정확한 PPL 재측정 | PASS | FP 6019.2490234375, Quant 5889.0615234375; delta -130.1875 |
| 동일 평가 조건 | PASS | source revision, float32, seed42, 12 sample IDs, 660 token corpus/hash, tokenizer, raw prompt, context128, PPL window64/stride32 일치 |
| 두 고정 prompt likelihood | PASS | 입력 길이 6과73; FP/Quant NLL 각각 39.6580009460/39.4269180298, 359.3289794922/368.0004577637 |
| FP cached greedy generation | PASS | 두 prompt에 실제 `generate(max_new_tokens=8,do_sample=False,use_cache=True)` 실행 |
| Quant cached greedy generation | FAIL | 두 prompt 모두 `get_experts_implementation` AttributeError |
| frozen 상태 보존 | PASS | 평가 및 generation 예외 전후 전체 weights/observers/qparams/modes/config/profiles/return_type 동일 |
| 기존 baseline generation | FAIL, 기존 호환성 문제 | 동일 의존성의 `bb09ca8` source에서 새 baseline PTQ wrapper도 동일 오류 재현 |
| 원본 pretrained Circle isolation | ERROR | exit2, comparisons_complete=false; 첫 decoder INT16 scale[inf]로 reference runtime 거부. Embedding stage만 실제 parity 완료 |
| >2048 eager / 4096+4 stress | BLOCKED | 이 모델의 recorded capacity2048은 4100보다 작음 |

PPL은 기존 `evaluate --tasks ppl --set ...` CLI로 실행했다. FP와 Quant output/protocol 경로를 분리했고, 두 명령은 실제 metric과 종료 로그를 남겼다. calibration32 및 export16보다 긴 64-token PPL window와73-token prompt가 eager에서 처리되었다. 평가 config에는 backend selector가 없다. 같은 PPL 점수나 정답 생성을 합격 조건으로 삼지 않았다.

Generation 오류의 직접 경로는 `tico/quantization/wrapq/wrappers/ptq_wrapper.py:90` → `QuantLlamaForCausalLM` (`quant_model_for_causal_lm.py:45`) → Transformers5.18.0 `generation/utils.py:2480`이다. Transformers는 prefill 이후 decode 최적화 context에서 해당 getter를 무조건 호출한다. 원본 `pretrained_supplemental.log`와 exit1을 보존했다. 검증 스크립트만 예외를 FAIL로 JSON에 기록하고 독립적인 likelihood 및 frozen-state 검사를 계속하도록 보완했으며, 재실행도 exit1이다. 제품 코드나 외부 evaluator를 수정하지 않았다.

Baseline process는 normal source import 위치를 확인했다. Candidate checkpoint를 baseline에서 읽으면 old wrapper의 `causal_mask_template`가 없어 실패했다. 이를 export equivalence로 해석하지 않고, 같은 pretrained FP 자산에서 작은 baseline wrapper를 새로 준비하여 generation API 오류를 독립적으로 재현했다. 이 별도 calibration은 old/new export 비교 증거가 아니다.

최소 수정 제안은 supported Transformers의 generation protocol에 맞는 getter/setter 등 compatibility surface를 wrapper에 제공하고 실제 cached `generate` 회귀 테스트를 추가하는 것이다. Gemma4 wrappers의 관련 delegation을 참고할 수 있다. 임의로 getter만 덧붙이거나 외부 라이브러리를 수정해 이번 결과를 PASS로 바꾸지 않았다.

전체 smoke 결과는 generation FAIL을 포함하므로 FAIL이다. `pretrained_protocol_comparison.json`의 protocol_status PASS는 완료된 PPL의 조건 일치만 의미한다. 원본 checkpoint가 cache-off prefill만으로 calibration되었다는 제한도 기록하며, 이후 별도 cache-calibrated fixture를 사용하면 다른 SHA와 결과로 구분한다.

원본 pretrained Circle run은 `pretrained_tinyllama_isolation_16/summary.json` 및 `candidate/error.json`에 ERROR를 남겼다. `tensors[393]`의 `quantize_per_tensor_307` INT16 `[1,16,16,4]` scale `[inf]`가 직접 원인이다. 원본 fixture가 `use_cache=False` prefill만 calibration하여 cache 관련 observer를 관측하지 못한 제한과 일치한다. tolerance를 변경하지 않았고 같은 checkpoint로2048을 중복 시도하지 않았다. 별도로 cached prefill28+decode4를 calibration한 새 fixture를 준비하여 결과를 구분한다.

Read-only observer 조사(`pretrained_observer_coverage.json`)에서 원본 checkpoint의 각 attention layer `obs_past_key`, `obs_past_value`, `obs_new_k`, `obs_new_v` 총32개 cached scale이 nonfinite임을 확인했다. `compute_qparams`나 calibration을 호출하지 않았다. 별도 cache-calibrated checkpoint는 SHA `c13bbd08a738a29c55c782bd1942f7b4056c20eb4c98b2f1628e1c99c64343ad`이며 이32개 cached scale이 모두 finite이다. 원본 checkpoint와 quality score는 변경하지 않았다. 새 checkpoint의 실제 Circle 결과는 별도 `pretrained_tinyllama_cached_isolation_16` 경로에 남기며, 이 결과가 원본 ERROR를 대체하지 않는다.

## Cache-calibrated fixture와 선택 stage의 최종 결과

새 cache-calibrated full-bundle 요청(`pretrained_tinyllama_cached_isolation_16`)은 1200초 한도에서 exit124/TIMEOUT으로 종료했다. Before phase의 embedding 및 decoder prefill/decode0–3 총9 stage가 실제 실행되었고 after-eager phase는 시작되지 않았다. 따라서 전체 pretrained bundle isolation은 미완료다. 마지막 관측 RSS는1.04GB, 가용 RAM9.87GB였으며, 이 한도는 RAM 부족이 아닌 명시적 실행 시간 한도이다. 모든 완료 stage의 실제 arrays, graph I/O, constants/qparams, 위치별 오차가 남아 있다.

이어 동일한 전체 frozen model 한 인스턴스에서 첫 decoder(layer0)의 prefill/decode만 독립적으로 before → 전체 model eager64 + teacher-forced decode4 → after 순서로 실행했다. `pretrained_selected_stage/summary.json`은 comparisons_complete=true, 최종 FAIL/exit1이다.

| 선택 stage 검사 | 결과 |
|---|---|
| 실제 tico.convert/Torch Export/CircleReferenceRuntime, 두 stage × before/after | 완료 |
| Eager64+4의 다섯 forward와 모든 logits/KV finite | PASS; 최종 KV `[1,16,68,4]` |
| Circle 구조/상수/qparams, 실제 입출력과 새로 생성한 host mask/cos/sin의 before/after 일치 | PASS; 모든 저장된 array의 최대 차이0 |
| 전체 frozen weights/observers/qparams/modes/config/profiles/return_type/export controls 보존 | PASS |
| Strict Torch-vs-Circle prefill parity | FAIL |
| Strict Torch-vs-Circle decode parity | PASS, 원래 atol/rtol 기준 |
| 선택 stage 전체 strict validation | FAIL |

Prefill의 실제 최대 오차는 hidden `4.0411949157714844e-05`, deltaK `8.225440979003906e-05`, deltaV `3.863871097564697e-05`다. Default atol=1e-5, rtol=1e-4를 유지했고 `pretrained_selected_stage_differences.json`에 failing index, 값, count 및 output qparams를 기록했다. 이 수치 차이를 RoPE 수정의 의도적 차이로 분류하거나 원래 FAIL을 PASS로 바꾸지 않았다. Before/after의 같은 Circle 출력만으로 Torch parity를 대신하지 않았다.

선택 stage isolation PASS는 layer0 두 stage에 한정한다. 전체8-layer bundle, pre-PR export equivalence, pretrained Llama-3.2 또는 NPU 검증의 PASS가 아니다. 원본 checkpoint ERROR, 별도 bundle TIMEOUT 및 Quant generation FAIL은 그대로 유효하다. 2048 export는16-token strict numerical failure가 재현되어 반복하지 않았으며, 4096+4 eager는 recorded capacity2048 때문에 BLOCKED다.

선택 stage 실행은459초, 관측 peak process-tree RSS738,463,744bytes이며 실제 Circle 실행 모드는 `fake_quantize` reference다. 별도 NPU latency나 integer backend 동등성을 측정하지 않았다.
