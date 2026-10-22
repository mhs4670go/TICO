# 검증 수준과 결과 해석

이 실행에서 결과는 다음 수준으로 구분한다.

1. **Unit / regression**: 기존 unit의 mock 및 optional-dependency stub을 포함한다. 이 결과는 실제 pretrained·Circle integration 증거로 대체하지 않는다. `changed_unit_tests.log`는 처음 직접 module 경로로 호출해 test discovery root를 제공하지 못한 실행이다. 이후 `changed_unit_discovery.log`에 표준 discovery 재실행 원본을 보존했다.
2. **Tiny real HF**: 로컬 config에서 새로 무작위 초기화한 실제 Transformers 모델, 실제 TICO wrappers/PTQ를 정상 import하여 실행한다. HF/NO_QUANT parity와 frozen state, generated PNG/실제 image processor/cached decode가 포함된다. pretrained 품질 점수는 아니다.
3. **Tiny pretrained**: 고정 revision `Maykeye/TinyLLama-v0`의 실제 가중치를 이용한 FP/Quant PPL·likelihood·generation이다. Llama-3.2 또는 다중-billion 모델 결과가 아니다. CLI protocol JSON과 실제 metric 완료 로그를 함께 확인한다.
4. **Actual Circle**: `tico.convert`와 `CircleReferenceRuntime`으로 저장된 Circle bytes를 실행한다. 성공한 stage의 `.circle`, graph contracts, 모든 input/Torch/Circle output arrays와 delta-KV를 보존한다. 원본 comparator의 증거 누락 결함 때문에 별도 완전성 audit를 수행한다. baseline-created tiny PTQ fixture와 pretrained 체크포인트 결과는 별개다.
5. **NPU**: 범위 밖이며 실행하지 않았다. Circle fake-quantize reference 실행은 실제 NPU·bit-exact 정수 backend 검증이 아니다.

`PASS`는 해당 scope가 실제 완료됐음을 뜻한다. 단독 isolation은 pre-PR equivalence가 아니며, Torch Export는 Circle 실행이 아니다. strict `--require-torch-parity`가 실패하면 Circle pre/post equality가 성립해도 전체 run은 FAIL로 유지한다. 의도적인 RoPE 변경 분석도 원본 old/new FAIL을 지우지 않는다.
