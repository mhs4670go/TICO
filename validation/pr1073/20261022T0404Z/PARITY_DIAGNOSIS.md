# Baseline Torch–Circle strict parity 실패 진단

대상은 pretrained가 아닌 `fixtures/baseline_llama_v2.pt` tiny random real-HF PTQ fixture이다. 원본 `tiny-llama-old-new-v2/`의 `--require-torch-parity`, `atol=1e-5`, `rtol=1e-4` 결과를 유지한다. 이 추가 진단은 원본 FAIL을 PASS로 바꾸지 않는다.

`trace_baseline_parity.py`는 baseline `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`의 소스를 정상 import하고 저장된 동일 checkpoint를 읽었다. 실제 adapter eager 실행과 실제 저장된 `.circle`의 `CircleReferenceRuntime(...).run(trace=True)`를 재실행했다. 네 decoder stage 모두 저장된 Torch 출력 및 Circle 출력과 **bit-exact** 재현되었다. Compiler/runtime 대체나 새 export는 사용하지 않았다. Observer 추적은 실제 `fake_quant()` 결과를 수집하고 원본 메서드를 복구했으며, qparams/값은 변경하지 않았다.

| baseline stage / export_example | hidden max abs error | 최대 오차 위치 | delta-K / delta-V max abs error |
|---|---:|---|---|
| prefill layer 0 | 0.00048585236072540283 | [0,5,28] | 0 / 0 |
| prefill layer 1 | 0.0006152968853712082 | [0,5,0] | 0 / 0 |
| decode layer 0 | 0.005344359029550105 | [0,0,15] | 0 / 0 |
| decode layer 1 | 0.03138008713722229 | [0,0,18] | 0 / 0 |

전체 probe/output의 위치·값·실패 원소 수는 `PARITY_DIAGNOSIS.json`에 있다. Embedding/lm_head는 실제 저장된 Torch–Circle 값이 exact였으며, `decode_empty_past`는 두 layer 모두 hidden/delta-K/delta-V exact이다.

## 확인된 prefill rounding 차이

실제 softmax trace의 여섯 key가 보이는 row 5에서 두 head 모두 다음 값을 보였다.

- Torch: `42/255 = 0.16470588743686676`
- Circle: `43/255 = 0.16862745583057404`
- 차이: `1/255 = 0.003921568393707275`

`probe_rounding_and_past.py`가 실제 Circle softmax tensor의 qparams(`scale=0.003921568859368563`, zero-point 0)를 사용하여 별도로 재현했다. Float32 `1/6`을 scale로 나눈 값은 정확히 `42.5`; Torch fake-quant는 ties-to-even으로 42, Circle reference affine quantization은 ties-away-from-zero로 43을 선택했다. 원본 출력에서 관측된 한 quantization step 차이와 일치한다.

- Torch 경로: `tico/quantization/wrapq/observers/affine_base.py:179`
- Circle ties-away 구현: `tico/circle/runtime/quantization.py:53` 및 `:136`
- 해당 runtime와 observer 파일은 pre-PR baseline과 candidate 사이 diff가 없다.

이 차이는 **baseline에 존재하는 quantization rounding 정책 차이**이다. Llama host RoPE 수정으로 설명하거나 old/new intentional-RoPE 예외로 묶지 않는다. 엄격한 parity 조건에서는 계속 FAIL이다. 전체 decoder 오차의 모든 원인을 rounding 하나로 단정하지 않는다. 특히 여러 출력이 동일 qparams를 공유하는 observer↔Circle tensor 대응은 진단용 후보 매칭이며 일반적인 중간 텐서 일대일 증명은 아니다.

## 확인된 fixture cached-decode calibration 부족

v2 creator는 cache 없는 text prefill(8/6/4)만 calibration하고 freeze했다. 두 layer의 `obs_past_key/value` 모두 다음 상태였다 (`ROUNDING_AND_PAST_DIAGNOSIS.json`).

- min=`+inf`, max=`-inf`: cached decode 통계 미수집
- frozen scale=`9.99999993922529e-09`, zero-point=0

Eager는 `quant_attention.py:510`에서 past K/V를 이 observer로 fake-quant한다. 실제 trace에서 이어 붙인 past 구간은 present-cache grid에서 0으로 반올림된 반면 Circle graph의 입력 past qparams는 present-cache 범위를 사용하여 값이 남았다. 예를 들어 layer 0의 112개 past 원소에서 K 최대 차이 0.0731480718, V 최대 차이 0.3221478164였다. 마지막 신규 KV slot은 layer 0에서 exact였고, delta-K/V 출력도 exact이다. 이로 인해 populated-past만 hidden 오차가 크고 empty-past probe는 exact인 패턴이 설명된다.

모든 외부 입력을 Circle에 기록된 quantization grid로 먼저 맞춘 추가 Torch replay에서도 hidden 오차는 줄지 않았다. 따라서 단순히 raw random input이 grid 밖에 있다는 설명만으로 원본 실패를 제거할 수 없다.

이 cached-decode 결과에는 **fixture calibration 부족**이 포함된다. 대표 pretrained checkpoint의 회귀로 일반화하지 않는다. 기존 checkpoint를 수정/재양자화하여 같은 결과로 취급하지 말고, 실제 cached prefill+teacher-forced decode를 calibration 중 실행하는 새 fixture를 새 경로로 생성하여 검증한다. 기존 v2 실패와 provenance는 유지해야 한다.

## 미적용 최소 제안

1. Fixture generator: freeze 전 실제 cache-bearing prefill과 decode를 실행하여 사용되는 past-cache observer의 통계를 수집하고 새 checkpoint를 저장한다. 이는 validation fixture 수정이다.
2. Validation infrastructure: frozen mode와 qparam 존재만으로 calibration 완료를 간주하지 말고, 검증 stage가 사용할 observer의 통계 수집 여부와 유효 scale/zero-point를 기록·검사한다.
3. Torch–Circle equality를 제품 계약으로 요구하려면 WrapQ와 Circle reference의 rounding 정책을 명시적으로 일치시켜야 한다. Circle 정책은 TFLite/ONE 참조 정책으로 문서화되어 있으므로 무조건 ties-even으로 바꾸는 수정은 제안하지 않는다. 현재는 원본 strict FAIL과 개별 rounding 증거를 함께 보고하고 tolerance를 올리지 않는다.

제품·baseline 파일을 변경하지 않았으며 어떠한 수정도 commit/push하지 않았다.
