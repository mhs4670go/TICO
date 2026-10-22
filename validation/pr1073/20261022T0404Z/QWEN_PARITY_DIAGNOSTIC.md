# Qwen3-VL 실제 Circle strict parity 추가 진단

범위: `tiny-qwen-old-new-cached/`의 실제 완료된 9-stage Circle 출력과 `fixtures/baseline_qwen_cached.pt` tiny random real-HF frozen PTQ checkpoint를 읽기 전용으로 조사했다. 새 export, 재양자화, 제품 수정은 하지 않았다. `inspect_qwen_parity.command.json`의 종료 코드는 0이며 이는 **진단 완료**를 뜻한다. 원래 `--require-torch-parity` 결과는 FAIL이다.

## 완료된 비교와 원본 실패의 구분

- baseline→candidate 및 candidate eager 전→후 모두 9개 stage의 Circle 구조·host·모든 input/Circle/Torch 출력 배열이 exact였다.
- 동일한 Torch–Circle 차이가 baseline, candidate before, candidate after_eager에 반복된다. 이 checkpoint/profile에서 PR의 old/new 출력 회귀나 eager에 의한 static artifact 변화는 관측되지 않았다.
- 하지만 decoder의 Torch–Circle strict parity가 실패하므로 전체 run과 strict isolation 결과를 PASS로 바꾸지 않는다.
- Vision(출력 2개), multimodal embedding, DeepStack fusion, token embedding, lm_head의 모든 실제 probe 출력은 Torch–Circle exact이다.

## Decoder export_example의 실제 오차

아래 수치는 세 phase에서 동일했다. 전체 probe별 최대 위치·양쪽 값·실패 원소 수는 `QWEN_PARITY_DIAGNOSTIC.json`에 저장했다.

| Stage | hidden max abs / 위치 | delta-K max abs / 위치 | delta-V max abs |
|---|---|---|---:|
| prefill layer 0 | 0.0010780766606330872 / [0,5,6] | 0.0064430832862854 / [0,0,0,5] | 0 |
| prefill layer 1 | 0.0015320554375648499 / [0,5,24] | 0.0416334867477417 / [0,0,5,7] | 0 |
| decode layer 0 | 0.044739918783307076 / [0,0,21] | 0.00644307304173708 / [0,0,0,4] | 0 |
| decode layer 1 | 0.03140721470117569 / [0,0,9] | 0.006938904523849487 / [0,0,0,7] | 0 |

`decode_empty_past`는 두 layer의 hidden 출력이 exact지만 delta-K는 여전히 위와 같은 오차를 가진다. 따라서 populated cache 하나만으로 모든 실패를 설명하지 않는다. Qwen delta-K는 Llama v2 진단과 달리 exact가 아니며, Llama RoPE 수정으로 해석하지 않는다.

## 실제 cached calibration과 static-tuple observer coverage의 차이

생성 provenance에는 실제 full-model prefill 4토큰 + cached decode 2회가 calibration 중 실행되었다고 기록되어 있다. 그러나 frozen checkpoint를 읽어 확인한 결과 두 Qwen text layer 모두:

- `obs_past_key` 및 `obs_past_value`: min=+inf, max=-inf, frozen scale=`9.99999993922529e-09`, zero-point=0, `enabled=false`, mode QUANT
- `obs_present_key/value`: 유한한 min/max와 정상 frozen qparams를 보유

원인은 소스 경로로 확인된다. Full text model은 cache가 없으면 HF `DynamicCache`를 만든다 (`quant_text_model.py:532`). Candidate `quant_text_attention.py:392`의 `_normalize_static_past_key_values()`는 tuple/list 입력에서만 `obs_past_key/value`를 사용한다 (`:404`). HF cache object 경로는 `_update_cache_like()`로 들어가 present-cache observer를 적용한다 (`:531`–`:543`). Baseline에서도 동일한 분리 경로가 있다 (`:417`–`:430`, `:556`–`:568`).

따라서 “full-model cached decode를 calibration했다”와 “per-layer static tuple의 past observer를 calibration했다”는 같은 주장이 아니다. 이번 checkpoint에서 후자는 충족되지 않았다. 이 제한은 큰 populated-past 차이에 영향을 줄 수 있지만, prefill·delta-K 오차까지 설명하는 완전한 원인 증명은 아니다.

## 남은 진단과 미적용 최소 제안

- 검증용 새 fixture를 만들 경우 freeze 전에 실제 per-layer tuple cache 경로도 calibration하여 해당 observer의 통계 수집을 확인한다. 기존 frozen file을 수정하거나 같은 체크포인트로 취급하지 않는다.
- Validation infrastructure는 cache-return 성공뿐 아니라 export stage가 사용할 observer 통계의 수집 여부를 검사·기록하는 편이 적절하다.
- Prefill 및 delta-K strict parity 원인은 이번 제한된 읽기 전용 진단으로 확정하지 않았다. **기존 baseline strict numerical failure, 원인 미확정**으로 유지한다. Llama에서 재현한 rounding 차이를 Qwen 전체 실패에 자동 적용하지 않는다.
- 원본 threshold `atol=1e-5`, `rtol=1e-4` 및 원본 FAIL을 유지했다.

이 결과는 pretrained Qwen 평가나 NPU 검증이 아니다.
