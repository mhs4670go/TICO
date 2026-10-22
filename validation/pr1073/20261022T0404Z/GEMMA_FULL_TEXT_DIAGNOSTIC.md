# Gemma4 full-text actual Circle strict FAIL 진단

`tiny-gemma-full-text-png-masks-retry/`의 완료된 실제 출력·graph 계약·frozen-state 기록을 읽기 전용으로 조사했다. 추가 모델 실행이나 export를 하지 않았다. `inspect_gemma_full_text` exit 0은 진단 파일 생성 완료이며, 원본 strict 결과는 **FAIL / exit 1**이다.

## 실제 출력

세 phase 모두 실제 `full_text_regression.q.circle` 1개를 FAKE_QUANTIZE 모드로 실행했다. Probe는 `export_example` 하나, graph input 1개, Torch output 1개, Circle output 1개이다.

| Phase | 출력 | finite | Torch–Circle max abs | 최대 위치 / Torch → Circle |
|---|---|---|---:|---|
| baseline before | last_hidden_state float32 [1,8,32] | true | 0.0869361162185669 | [0,4,10] / 1.695252776145935 → 1.782188892364502 |
| candidate before | 동일 | true | 동일 | 동일 |
| candidate after_eager | 동일 | true | 동일 | 동일 |

각 phase에서 threshold `atol=1e-5`, `rtol=1e-4`를 넘는 원소는 256개 중 92개이다. 첫 실패 위치와 실제 양쪽 값, 배열 SHA256은 `GEMMA_FULL_TEXT_DIAGNOSTIC.json`에 있다. 모든 기록 배열은 finite이다.

## Old/new 및 eager 전후 성분 결과

- 세 graph는 operator 109개, tensor 195개, quantization metadata가 있는 tensor 149개, constant tensor 81개이며 완전한 contract hash가 같다.
- Old/new 및 candidate eager 전후의 topology/options, graph I/O, constants, qparams, host 및 모든 input/Torch/Circle output 배열이 exact이다. 별도 hash도 JSON에 저장했다.
- 같은 checkpoint·dependency·초기 quantization state를 사용했으며 frozen weights·observer/qparams·mode가 보존되었다.
- 따라서 이 profile에서 PR에 따른 old/new 차이나 eager가 static artifact를 바꾼 증거는 없다. 그러나 baseline부터 존재하는 동일한 Torch–Circle strict numerical failure 때문에 전체 run이나 strict isolation을 PASS로 바꾸지 않는다.
- 이번 읽기 전용 진단으로 수치 오차의 내부 연산 원인은 확정하지 않았다. 원인 미확정 baseline parity failure로 유지하며 Llama RoPE 수정으로 설명하지 않는다.

## 원본 인프라 ERROR와 복사본 보완

원본 결과를 지우지 않고 다음 순서를 별도 output path로 보존했다.

1. `tiny-qwen-full-text-png-original/`: empty base artifact set에서 full-text artifact directory가 생성되지 않아 `FileNotFoundError`, exit 2. 공통 복사본 `export_regression_full_text_dirs.py`는 worker 내부 output parent만 생성한다.
2. `tiny-gemma-full-text-png-dirs/`: Gemma static/export API가 요구하는 layer-type별 mask dict가 없어서 `NotImplementedError: QuantGemma4TextModel static/export mode requires static masks as a dict keyed by layer type.`, exit 2.
3. `export_regression_gemma_full_text_masks.py`: 위 directory fix를 유지하고 **동일 frozen checkpoint에 이미 기록된** bounded full/sliding mask template `[...,:8,:8]`을 tracing 전에 복사하여 TextProbe buffer로 등록한 후 mask dict로 전달한다. 새 mask 규칙·custom position·qparams·weight·dtype·runtime·tolerance·comparator 변경은 없다.

`export_regression_gemma_full_text_masks.diff`와 `.provenance.json`에 원본/부모/복사본 SHA256 및 정확한 변경 내용이 있다. 원본 product와 baseline source는 변경하지 않았다. 기록 mask의 원본은 `recorded_host_masks.json/npz`이며, full/sliding 모두 `[1,1,8,8]`, float32, min=-120, max=0, SHA256 `98d9f6a5f9d0be1bdc3c98d1a4c332e79d2befbaa999b8f7ce56522a2a0c6bf4`이다. 이 작은 profile의 sliding_window도 8이어서 두 mask가 같으며 sliding cache offset 검증으로 일반화하지 않는다.

## 실제 PNG eager call의 범위

Candidate에서 `png_circle_inputs/gemma_actual_png_kwargs.pt`를 사용하는 실제 PNG→정상 HF processor 입력의 full-model eager call **1회**가 두 export 사이에 실행되었다. `eager.json`은 `kind=user_kwargs`, `calls=1`, frozen state 보존을 기록한다. Input SHA256은 `fde489de9d8bb2ccd08126eab0a96b0b0a9d6d9f21a628194d682af34ecfe1c0`이다.

이 모드는 자동 text prefill/decode stress를 대체하며 use_cache=false이다. CLI에 적힌 eager length/decode count를 실제 실행한 것으로 보고하지 않는다. Cache를 연결한 multimodal decode, full multimodal Circle export, pretrained 품질 평가 또는 NPU 검증도 아니다. 실제 full-model PNG eager logits 완료 증거는 별도 `png_circle_inputs/fullmodel_forward_evidence.json`에 있다.
