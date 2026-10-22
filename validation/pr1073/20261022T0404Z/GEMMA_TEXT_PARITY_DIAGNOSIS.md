# Gemma4 text stage strict parity 원인 국소화

범위: tiny frozen checkpoint `fixtures/baseline_gemma_image70.pt`(SHA256 `ab9e52cceb5c7442c0d2e03ed6eeddce75c741d78d51603a712d6d3bad19b107`), baseline commit `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`의 기존 `gemma-text-independent-baseline/before` binary/입출력/contract를 조사했다. 새 export, calibration, checkpoint write, 제품 수정은 하지 않았다. NPU/pretrained 검증이 아니다.

**입증된 원인 경계는 query rotate-half의 별도 frozen `q_x1/q_x2` requantization 경계 손실이다.** Serialized SPLIT→CONCAT는 두 half observer의 grid를 유지하지 않고 q-norm/q-cat grid를 사용한다. 동일 입력의 query norm은 exact하지만 그 다음 q-cat 값이 달라지고 attention/MLP를 거쳐 hidden max abs `0.3541484475135803`이 된다. 이것을 sliding mask나 Llama RoPE 수정으로 설명하지 않는다. 원본 strict FAIL과 tolerance를 유지한다.

## 원본 결과와 실제 재실행

| Baseline stage | hidden strict 차이 | delta-K/V 차이 |
|---|---:|---:|
| layer0 sliding prefill | 0.3541484475135803 | 둘 다 0 |
| layer1 full prefill | 0 | 둘 다 0 |
| layer0/1 decode, 모든 원본 probe | 0 | 둘 다 0 |

- layer0의 window=8, prefill 길이=8이므로 이 profile에서 sliding cutoff는 발생하지 않는다. 두 layer의 실제 prefill input mask는 동일한 0/-120 causal matrix다. Replay에서 실제 mask quantized tensor도 Torch와 exact였다.
- 기존 layer0 prefill binary를 실제 `CircleReferenceRuntime(...).run(..., FAKE_QUANTIZE, trace=True)`로 재실행했다. 세 출력 모두 원래 저장된 Circle 출력과 exact였다.
- 같은 frozen baseline decoder adapter를 같은 저장 입력으로 실제 eager 실행했다. 세 출력 모두 원래 Torch 출력과 exact였다.
- 진단 recorder는 `_fq`의 **원본 구현을 그대로 호출하여 반환값을 바꾸지 않고** 입력/출력을 복사한 뒤 복구했다. Compiler/runtime 대체나 import 우회는 없었다. 모델의 frozen state fingerprint는 전후 동일했다.
- 최초 helper는 runtime에 Path를 넘겨 `Expected bytes, received PosixPath`로 실패했다. `gemma_text_parity_trace.log`, `inspect_gemma_text_parity.failed_v1.py`를 보존하고 `.read_bytes()`만 고쳐 별도 `gemma_text_parity_trace_v2.log`로 재실행했다. 진단 exit 0은 **분석 완료**이며 strict parity PASS가 아니다.

## Serialized graph와 frozen observer의 직접 비교

| 경계 | scale | zero-point |
|---|---:|---:|
| frozen `q_x1` | 0.02118774875998497 | 147 |
| frozen `q_x2` | 0.02068050019443035 | 119 |
| serialized q-norm tensor22 / SPLIT25,26 / CAT27 | 0.023234067484736443 | 134 |

두 frozen half grid는 이 binary의 어떤 tensor qparams에도 존재하지 않는다. 실제 trace에서 다음이 모두 exact였다:

1. Circle q-norm tensor22 = Torch q-norm의 fake-quant 출력.
2. Circle SPLIT25/26 = tensor22의 원래 앞/뒤 half. 별도 half requantization이 없다.
3. Circle CAT27 = 이 원래 half들을 뒤/앞 순서로 합친 값.
4. Torch q-cat 입력 = 실제 `q_x2`와 `q_x1` fake-quant 출력을 뒤/앞으로 합친 값.

예를 들어 `[0,4,0,4]`의 q-norm 값은 양쪽 모두 `2.7648539543151855`다. Torch의 frozen `q_x1`은 이를 `2.2882769107818604`로 양자화/clip한다. 이후 q-cat grid에서 `[0,4,0,12]` 값은:

- Torch: `2.2769386768341064`
- Circle: `2.7648539543151855`

q-cat의 총 2개 원소가 다르며 최대 차이는 `0.4879152774810791`이다. 그 뒤 관측된 최대 차이는 q-sin `0.0186268911`, q-rot `0.0233465135`, head0 softmax `0.0823529363`, 최종 MLP residual/hidden `0.3541484475`다. 최종 hidden 차이는 query row4에 국한된다. K/V 경로는 exact라서 동일 출력 RoPE/KV 자체의 전체 불일치라고 볼 근거는 없다.

같은 입력을 같은 serialized grid로 독립 quantize한 61개 매칭 경계에서는 Torch와 Circle quantizer 결과가 모두 exact였다. 따라서 이 실행에서 관측한 최초 차이를 단순 rounding convention 차이로 뭉뚱그리지 않는다. 좁은 half grid의 clip/requantization 자체가 serialized 경로에 없다. Random probe 값이 calibration range를 넘어서 이 차이를 드러냈더라도, 원래 requested parity의 실패를 면제하는 근거는 아니다.

## 소스 소유와 아직 확정하지 않은 부분

Eager에서 half별 observer를 실제 적용하는 코드는 baseline `tico/quantization/wrapq/wrappers/gemma4/quant_text_attention.py:243`–`:248`이다. 이 경계가 binary에 보존되지 않는 것은 실제 trace/contract로 입증했다.

다음은 **정확한 pass 위치를 아직 별도 실행으로 확정하지 않은 소스상 후보**다:

- `tico/quantization/passes/fold_quant_ops.py:43`의 view-chain 목록은 permute/reshape/slice이며 split/getitem을 포함하지 않는다. 기존 producer domain을 찾는 `_inherit_qparam_from_view_chain()`은 `:103`에 있다. Source domain을 인식해야 별도 half quantizer를 requantization으로 남길 수 있다.
- `propagate_qparam_forward.py:75`–`:85`는 dtype이 같으면 destination의 기존 scale/zp를 덮어쓸 수 있고, split/getitem 분기는 `:108`–`:129`에서 producer grid를 전달한다.
- `propagate_qparam_backward.py:55`–`:65`, `:73`–`:81`도 concat output domain을 inputs에 전달한다.
- `tico/utils/convert.py:329`–`:332`는 FoldQuantOps → forward/backward propagation 순서다.

위 세 quantization pass 파일은 baseline과 candidate에서 byte-identical이다. Coverage worker의 별도 `gemma_text_strict_comparison.json`에서도 baseline/candidate-before/candidate-after_eager의 Circle binary·contract·host·실제 입출력이 exact였다. 따라서 이 진단은 **기존 converter의 frozen half-quantization semantics 불일치**를 지목하며 PR이 새로 만든 Gemma 출력 회귀로 분류하지 않는다. 원본 strict old/new/isolation FAIL은 그대로다. `EXPECTED_NUMERICAL_CHANGE`로 승격하지 않는다.

## 미적용 최소 수정 방향과 남은 검증

제품 수정은 하지 않았다. 수정 시에는 split/getitem의 실제 producer quantization domain을 fold 전에 인식하여 **서로 다른 frozen half grid를 explicit requantization으로 보존**하고, propagation이 이를 단순 동일-domain split으로 대체하지 않도록 해야 한다. 단순히 SPLIT output metadata만 다른 grid로 바꾸면 값 변환이 누락될 수 있으므로 explicit 변환을 확인해야 한다. Checkpoint qparams 복사/재calibration이나 tolerance 확대로 실패를 숨기는 방법은 제안하지 않는다.

남은 검증은 pass 전후 snapshot으로 정확한 소실 지점을 확인하고, distinct narrow half grids와 범위 밖 입력을 사용하는 split→reorder→concat regression을 실제 Torch/Circle로 확인하는 것이다. 이번 제한된 진단에서는 재-export/제품 수정 및 수정 후 검증을 실행하지 않았다. 다른 profile·pretrained 모델에서의 빈도나 품질 영향도 미검증이다.

증거: `GEMMA_TEXT_PARITY_TRACE.json/.npz`, `GEMMA_TEXT_PARITY_DIAGNOSIS.json`, 원본 stage records, `gemma_text_parity_trace_v2.command.json`, `summarize_gemma_text_parity_v2.command.json`. 모든 실제 출력과 원본 실패는 보존되어 있다.
