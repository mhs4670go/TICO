# Gemma4 독립 text-stage Circle 검증

**전체 strict 결과는 FAIL(exit1)** 이다. 별도 명시적인 text-stage 검증으로 실제 Circle 실행을 완료했지만, 원래 multimodal export의 vision ERROR는 그대로 유지했다.

| 검증 | 실제 결과 |
|---|---|
| baseline/candidate checkpoint | 같은 frozen full-module, SHA256 `ab9e52cceb5c7442c0d2e03ed6eeddce75c741d78d51603a712d6d3bad19b107` |
| 실제 source import | baseline bb09ca8: `/workspace/TICO-export-baseline/tico/__init__.py`; candidate6b2f6a0: `/workspace/TICO/tico/__init__.py` |
| 실제 Circle text stages | 두 layer 각각 prefill/decode4개, baseline/candidate-before/candidate-after3phase: 총12회 |
| 실제 numerical probes | phase당 prefill2×2 + decode2×3 =10, 총30; 각 probe hidden/K/V3개, 총90개 Circle 출력과90개 Torch golden |
| 독립 completeness 감사 | 모든 예상 stage/graphI/O/출력/finite/qparams 양수·finite·decode delta-KV 길이 검증 완료 |
| candidate eager | text wrapper prefill12 + 연결된 cache decode4, 모두 finite; native64/static8 |
| frozen 상태 | 세 export phase와 eager 후 모두 raw parameter/buffer/observer/mode/profile 변경0 |
| old/new Circle 및 실제 수치 | 4개 Circle binary/complete contract/host arrays/모든 input·Torch·Circle output array 정확히 동일 |
| old/new Torch FX | 4stage 모두 op3개 감소, 아래에 별도 기록; 같다고 보고하지 않음 |
| candidate eager 전후 | Circle/contract/host/실제 arrays/**Torch FX까지** 모두 정확히 동일 |
| strict old/new / strict isolation | **둘 다 FAIL**: layer0 sliding-prefill hidden Torch/Circle parity 실패 |

## 실제 stage별 수치 결과

아래 수치는 baseline, candidate-before, candidate-after 모두 동일하다. atol1e-5/rtol1e-4를 유지했다.

| stage | probe | hidden max abs | K/V max abs | strict Torch/Circle |
|---|---|---:|---:|---|
| layer0 sliding prefill | 기본 + last-key padded | 0.3541484475135803 | 0 / 0 | **FAIL** |
| layer0 decode | 기본 + empty/full past | 0 | 0 / 0 | PASS |
| layer1 full prefill | 기본 + last-key padded | 0 | 0 / 0 | PASS |
| layer1 decode | 기본 + empty/full past | 0 | 0 / 0 | PASS |

Prefill hidden은 `[1,8,32]`, K/V는 `[1,1,8,16]`; decode hidden은 `[1,1,32]`, delta-K/V는 `[1,1,1,16]`이다. 모든 stage는 실제 `tico.convert`, `torch.export`, `CircleReferenceRuntime`의 `fake_quantize` 실행으로 검증했다. NPU 또는 bit-exact 정수 backend 검증은 아니다.

모든 Circle 결과가 서로 같다는 이유로 strict 성공을 부여하지 않았다. 원래 strict comparison/audit JSON과 exit1을 유지한다. 이 text-stage 수치 실패를 Llama RoPE 변경으로 설명하지 않는다. 원인 국소화는 별도 bounded trace 분석 결과를 따른다.

## 범위와 구현

이 checkpoint는 baseline constructor에서 만든 작은 random real-HF/PTQ Gemma4이며 pretrained 모델이 아니다. max_soft_tokens70인 원래 frozen checkpoint/config를 변경하지 않고 양쪽이 같은 파일을 읽었다. text 두 layer는 sliding(window8)/full이고 PLE0/shared-KV0/attention_k_eq_v=False, static8/native64다. 추가 calibration, dtype cast, qparams 변경은 없다.

기존 `export_gemma4_per_layer()`는 vision을 먼저 포함하고 별도 text-only export 함수가 없어, 독립 harness가 원래 소스의 `_attention_contract`(`recipes/export/gemma4.py:644`), 기존 prefill/decode mask helper(603/620행), 기존 `layer.wrapped.as_export_module(..., return_kv=True)`와 `_convert_and_save`를 그대로 호출했다. host cos/sin은 같은 checkpoint의 frozen static template(기록된 `pre_negated_sin` convention)을 사용했다. 임의의 compiler/runtime을 만들거나 대체하지 않았다.

입력 hidden과 past K/V는 기존 exporter처럼 seed1073의 작은 deterministic tensor다. 따라서 이 검증은 명시적인 decoder-stage interface/parity이며 전체 text-model chaining, token embedding/LM head, multimodal fusion, vision, 이미지·영상 평가 또는 generation 품질 검증은 아니다. eager stress도 실제 **qtext wrapper만** 호출해 special image/video token routing과 구분했다.

Recorder는 원래 `export_regression.py`의 **변경 없는 복사본** `gemma_text_export_recording.py`를 정상 Python import한 것이다. 양쪽 worker는 같은 recorder 도구를 쓰고 실제 라이브러리는 각 checkout에서 정상 import했다. 제품이나 baseline 소스 파일은 수정하지 않았다.

worker exit0은 4개 export/reference 실행과 상태 보존 완료를 뜻한다. strict 수치 통과 여부는 별도 `gemma_text_strict_comparison`의 exit1/FAIL로 판정했다.

## Torch FX 차이

Circle binary와 모든 실제 수치는 같지만 baseline 대비 candidate에서 다음 노드가 없어졌다.

- Prefill 두 stage: 492→489 ops. `aten.alias.default`, `aten._assert_tensor_metadata.default`, `aten.to.dtype_layout` 각1개 삭제.
- Decode 두 stage: 505→502 ops. `aten.slice.Tensor`, `aten._assert_tensor_metadata.default`, `aten.to.dtype_layout` 각1개 삭제.

원래 comparator는 FX 목록 차이를 기록하지만 gating하지 않는다. 독립 감사는 이를 숨기지 않고 `old_new all_exact=false`로 기록했다. candidate 전후에는 FX 차이도 없다. 정확한 순서/index는 `gemma_text_fx_differences.json`에 있다.

## 원래 vision 오류 보존

`tiny-gemma-old-new-image70/baseline.log:27`의 `UnsupportedCircleOperatorError: RMS_NORM supports rank3 or4, found shape(4,32)` 및 op102 진단을 그대로 유지했다. 관련 실제 kernel 위치는 `tico/circle/runtime/kernels/nn.py:620`이다. 이 text-stage 검증은 그 vision stage를 PASS로 만들거나 main multimodal old/new ERROR를 대체하지 않는다.

## 재현 및 증거

- `commands.sh`: `gemma_text_baseline`, `gemma_text_candidate`, `gemma_text_strict_comparison`
- 원본 stdout/stderr/종료코드: 위 이름의 `.log`/`.command.json`
- `gemma-text-independent-baseline/` 및 `gemma-text-independent-candidate/`: import/commit/checkpoint 환경, 전체 initial state, before/after snapshot와 full state, manifest, 실제 Circle, 모든 numerical records `.json`/`.npz`, host `.npz`, eager 실제 output hash
- `gemma_text_strict_comparison.json`: strict FAIL, stage/graph/출력 completeness 및 양쪽/전후 비교
- `gemma_text_analysis.json`: raw 결과와 좁은 component 동일성 구분
- `gemma_text_fx_differences.json`: 원래 Torch Export 그래프 변경 원본 목록
- 실행 도구: `gemma_text_stage_worker.py`, `compare_gemma_text_stages.py`, `gemma_text_export_recording.py`

재실행은 같은 frozen checkpoint를 사용하되 worker의 새 output 디렉터리를 지정해야 한다. 모든 원본 결과는 이번 cloud 실행에서 새로 생성했다.
