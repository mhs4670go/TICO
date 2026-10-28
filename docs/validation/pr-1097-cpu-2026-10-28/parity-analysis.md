# 저장 Circle strict parity 진단

head `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`, base `261bda28ec55fdac42f59346c950326e4065f8fa`. 이 문서는 추가 saved-file parity 실패의 원인 분석이며 원래 assertion/tolerance/skip, 제품 코드 또는 공식 결과를 변경하지 않았다. 같은 case의 base 실행 결과/최종 분류는 wrapper baseline 결과를 참조한다. 소스 diff만으로 PR 회귀 여부를 단정하지 않는다.

## 확인한 실패와 의미

`llama_attention_prefill`은 원본 inspector exit0, 저장/구조/runtime 실행/frozen snapshot 보존 PASS지만, Torch 대비 `atol=1e-5, rtol=1e-4` 저장-file parity는 K output에서 FAIL이다. max error `0.013716518878936768`은 K output grid scale `0.013716510497033596` 한 step에 해당하며, 128개 원소 중7개가 다르다. hidden 및 V output은 정확히 같다. 다른 예로 nn_tied_embedding output은 scale `.08418599516153336` 한 step의 차이, decoder hidden은 최대 두 step의 차이가 관찰됐다. 이는 구조 검사나 파일 저장을 수치 parity로 취급할 수 없음을 보여준다.

`CircleDocument.verify()`는 index/dataflow/shape 등 static consistency를 확인한다. quantized value-preserving 연산의 grid 전이와 원래 eager의 fake-quant 의미가 보존됐는지는 별도 수치 검증이 필요하다.

## Llama K/RoPE 경로의 구체적 증거

`parity_trace_diagnostic.py`는 export를 다시 수행하지 않고 동일 seed/config/calibration3회로 실제 tiny eager model을 prepare/calibrate/convert한 뒤, 기존 저장 Circle의 `trace=True` 결과와 observer intermediate를 비교했다. 관측 wrapper는 원래 `fake_quant`를 그대로 호출한다. 재구성 eager의3개 output 배열은 최초 export harness가 캡처한 Torch output과 byte-for-byte 일치한다. 두 배열 집합의 SHA256과 exact equality는 `parity-trace-diagnostic.json`에 저장했다.

`QuantLlamaAttention._rot()`(quant_attention.py:286~297)는 split x1/x2 각각을 서로 다른 observer grid로 fake-quantize한 뒤 NEG/CAT와 또 다른 fake-quant를 적용한다. 저장 graph의 K path는 tensor14 → SPLIT_V(tensor25/26) → NEG(tensor27) → CONCATENATION(tensor28)이다. 명시적 QUANTIZE가 이 경로에 없다.

| eager observer | eager scale, zero-point | 대응 Circle tensor scale, zero-point | max abs | 다른 원소 |
|---|---|---|---:|---:|
| obs_k_x1 | 0.0110453516245, 143 | 0.0133473351598, 152 | 0.00543051958084 | 62 |
| obs_k_x2 | 0.0133473351598, 103 | 0.0141358673573, 112 | 0.208441615105 | 64 |
| obs_k_cat | 0.0133473351598, 152 | 0.0133473351598, 152 | 0.0111548900604 | 62 |
| obs_k_cos | 0.0120482612401, 128 | 0.0120482612401, 128 | 0 | 0 |
| obs_k_sin | 0.0116278557107, 130 | 0.0116278557107, 130 | 0.011627882719 | 7 |
| obs_k_rot | 0.013716510497, 123 | 0.013716510497, 123 | 0.0137165188789 | 7 |

처음 관찰한 K branch 차이는 split 이후다. eager x1/x2의 의도된 grid와 저장된 split tensor의 grid부터 다르고, runtime의 value-preserving SPLIT_V/CONCATENATION은 값을 재양자화하지 않는다. 이는 runtime README에 명시된 동작이다. 이후 k_cos 경로는 정확히 일치하지만 k_cat/k_sin을 거쳐 K output에1step 오차가 남는다. 한편 x2 intermediate 차이 `.2084416`은 output의 단순한 float epsilon 차이보다 크므로 모든 차이를 마지막 rounding sensitivity로 설명할 수 없다.

소스에서 조사할 책임 경계:

- `tico/quantization/passes/fold_quant_ops.py`는 Q/DQ를 metadata로 접는다. split/getitem을 통과한 서로 다른 affine grid의 명시적 전이가 사라지는지 확인해야 한다.
- `propagate_qparam_forward.py:103~109`와 backward의 propagation helper는 같은 dtype이면 기존 qparam을 덮어쓸 수 있다. backward CAT propagation은 입력 grid도 갱신한다. dtype 일치만으로 scale/zero-point 의미까지 같은 것은 아니다.
- `tico/circle/runtime/kernels/__init__.py`는 SPLIT_V/CONCATENATION을 value preserving으로 분류한다. executor.py:349~364는 그 연산에 자동 round-trip을 하지 않는다. 원래 eager FQ 전이를 runtime이 추측해 복구한다고 가정해서는 안 된다.
- 이 pass/runtime 파일들은 이번 PR diff에서 변경되지 않았다. **이 사실만으로 기존 문제 판정은 충분하지 않으며**, 같은 입력/설정의 base artifact 결과로 최종 분류해야 한다.

## 배제한 가설

runtime은 TFLite/luci와 같은 ties-away-from-zero, Torch fake-quant는 ties-even을 사용하므로 먼저 rounding policy 차이를 의심했다. `parity_rounding_diagnostic.py`는 진단 프로세스 안에서만 rounding 함수를 `np.rint`로 바꿔 이미 저장된 Circle6개를 재실행했다. 모든6개에서 원래와 최대 오차/불일치가 그대로였다. **단순 tie rounding 정책만의 문제가 아니다.** 이 대체 실행은 공식 runtime PASS 증거로 집계하지 않는다. 원래 runtime 결과와 tolerance는 그대로 유지했다. 결과/로그는 `parity-rounding-diagnostic.json`/`.log`에 있다.

## 수정 제안과 범위

1. split/view/CAT 등 value-preserving 연산의 producer/consumer grid가 호환되는지를 전체 qparam 계약(scale, zero-point, axis, range, dtype)으로 확인한다. 불일치하면 원래 FQ/Q-DQ를 명시적인 QUANTIZE 경계로 보존하고, metadata overwrite로 전이를 없애지 않는다.
2. fold/propagation/serialization와 runtime의 ownership을 함께 조사한다. runtime-only 문제라고 단정하거나 모든 layout op에서 무조건 requantize하는 수정은 다른 올바른 graph에 이중 rounding을 추가할 수 있다.
3. 기존 KV-domain tests처럼 통제된 grid의 exact parity와 별도로, 독립적으로 관측한 일반 smoke grid의 saved-file parity를 regression으로 추가한다. `--strict --export circle`이 현재는 저장 성공까지만 보는 coverage gap도 보완한다.
4. tolerance를 grid step 수준으로 늘려 이 결과를 PASS로 바꾸지 않는다. source와 serialized graph가 서로 다른 양자화 함수를 계산하는 문제와 허용 가능한 backend float arithmetic 차이를 분리해야 한다.

이 진단으로 Llama K/RoPE branch의 원인을 구체화했지만, nn_tied_embedding·vision 등 나머지 실패가 모두 같은 단일 pass 때문이라고 입증하지는 않았다. 각 baseline 결과 및 모델별 trace가 필요하다. 또한 FAKE_QUANTIZE는 semantic CPU reference이며 NPU integer bit-exact parity를 증명하지 않는다.

## 재실행

```bash
cd ${CANDIDATE_REPO}
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=${CANDIDATE_REPO} ${VALIDATION_VENV}/bin/python \
  ${VALIDATION_ROOT}/parity_trace_diagnostic.py
```

원본 실패의 최소 재현 명령은 `runs/wrapper-circle-llama_attention_prefill.json`의 `export_case.py` invocation이며, 원본 배열은 `wrapper-circle/llama_attention_prefill/parity-arrays.npz`, 저장-file 검사 결과는 같은 디렉터리의 `artifact-checks.json`이다.
