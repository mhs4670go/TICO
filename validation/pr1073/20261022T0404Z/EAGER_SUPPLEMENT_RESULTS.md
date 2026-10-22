# PR1073 Eager 보충 실행 결과

대상 SHA: `6b2f6a0511458d6a750e2a260d706080a78149dd`.
환경: Python3.12, Torch2.13.0+cpu, Transformers5.18.0, 정상 editable TICO import. 아래 작업은 이 실행에서 새로 수행했다. 제품 소스 변경, 외부 evaluator 변경, import stub, mock runtime, tolerance 변경은 없다.

| 모델 | 추가 기존 unit | 실제 tiny-HF text/PTQ eager 상태 | 생성 PNG + 실제 HF processor + vision/text | pretrained benchmark |
|---|---|---|---|---|
| Llama | root의 changed-unit 결과 참조 | PASS | 해당 없음 | 이 보충 실행 범위 아님 |
| Qwen3-VL | PASS 23, skip0, exit0 | PASS | PASS | 미실행 |
| Gemma4 | PASS 12, skip0, exit0; RUN_INTERNAL_TESTS=1 | PASS | PASS | 미실행 |

Text/PTQ 조건은 calibration=3, export=6, eager-prefill=12, decode=4, native=32이다. native/null context, 명시 evaluation cap, config의 positional/RoPE 의미 보존, scoring cache off와 prefill/decode cache on, calibration/force-static 평가 거부를 검사했다. weights, 모든 buffers, observer 통계/scale/zero-point/enable flags, 모든 QuantModule mode/profile/return policy를 전후 exact SHA256으로 비교했다. Llama/Qwen actual owner의 prepared_static_export 예외 cleanup도 PASS이다. Gemma prepared-static 예외 cleanup은 이 보충 스크립트에서 실행하지 않았다.

| 모델 | parameter 수 | buffer 수 | observer 수 | QuantModule 수 |
|---|---:|---:|---:|---:|
| Llama | 20 | 572 | 136 | 57 |
| Qwen3-VL | 26 | 644 | 160 | 65 |
| Gemma4 | 28 | 706 | 166 | 73 |

PNG 입력은 deterministic 64×64 RGB이며 파일 SHA256은 `generated_image_eager/summary.json`에 있다. Qwen은 Qwen2VLImageProcessor(patch16/temporal2/merge2,64×64), Gemma는 Gemma4ImageProcessor(patch4/pooling2,max_soft_tokens70,do_resize=False;16×16 image)를 사용한다. Gemma processor의280 patches에는 padding과4 valid soft tokens가 들어간다. FP/NO_QUANT/Quant가 같은 전처리 tensor를 사용했다.

| 모델 | HF vs NO_QUANT max abs | Quant vs FP hidden max abs delta | cache decode | frozen 상태 |
|---|---:|---:|---|---|
| Qwen3-VL | 2.4437904357910156e-6 | 0.2602647542953491 | image prefill + 연결된4 step 완료 | PASS |
| Gemma4 | 1.430511474609375e-6 | 0.20679152011871338 | image prefill + 연결된4 step 완료 | PASS |

HF vs NO_QUANT만 atol1e-5/rtol1e-4 parity를 요구했다. FP/Quant 동일 점수/출력은 요구하지 않았다. 여기의 delta는 random tiny 모델의 hidden tensor 차이이며 모델 품질 benchmark나 pretrained quantization 품질 추정치가 아니다.

모든 실행은 `commands.sh`와 다음 원본 기록으로 재현한다.

- `gemma_internal_multimodal_units.log` / `.command.json`
- `qwen_tiny_multimodal_units.log` / `.command.json`
- `supplemental_eager_state.log` / `.command.json`, `supplemental_eager_state/summary.json`, family별 before/after state JSON
- `generated_image_eager.log` / `.command.json`, `generated_image_eager/summary.json`, PNG, family별 state JSON 및 실제 input/output tensor `.pt`
- 실행 코드: `supplemental_eager_state.py`, `generated_image_eager.py`
- coverage 지도: `EAGER_COVERAGE.md`

위 PASS는 pretrained 평가, 원래 baseline과의 동등성, Circle 실행 또는 NPU 검증을 뜻하지 않는다. 이 보충 실행에서 제품 버그나 검증 스크립트의 실행 오류는 발견하지 않았다.

후속 normal image `generate()` 추가 검증은 별도 결과이다. Gemma4 FP/Quant2tokens PASS 및 전체 raw frozen 상태 변화0. Qwen3-VL FP PASS, Quant는 experts accessor 누락으로 FAIL이며 원래 baseline constructor의 독립 작은 PTQ fixture에서도 같은 실패를 실제 재현했다. 앞선 text/image-forward PASS는 이 generate 실패를 상쇄하지 않는다. 새로 발견한 호환성 실패와 미적용 최소 수정안, 정확한 baseline 구분은 `IMAGE_GENERATION_FINDINGS.md` 및 `image_generation_analysis.json`을 따른다.
