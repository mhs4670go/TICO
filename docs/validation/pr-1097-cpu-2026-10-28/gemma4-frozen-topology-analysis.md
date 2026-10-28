# Gemma4 vision model frozen hash 변화 진단

대상은 head `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`의 `gemma4_vision_model`이다. 원본 `artifact-checks.json`의 `frozen_unchanged=FAIL`은 그대로 보존한다. 추가 진단 결과, 이 FAIL은 원본 frozen observer/qparam 변형이 아니라 **허용된 독립 pooler profile topology 추가를 whole-model fingerprint가 변화로 취급한 것**이다. 별도의 Circle numerical parity FAIL은 이 결론과 구분해야 한다.

## 원본 관찰

- before fingerprint: `99ac1d5e77e83473d8dceeb1bc7891ebf1eefeea611f7bb0e2f08f91c2ee57a0`
- export 이후 fingerprint: `67e48718ea29f7778e9ae5d8da1e991a330acd127657923f1cbc5720595a29ca`
- reference runtime 이후 fingerprint: 동일한 `67e48718...`
- 최초 Torch export-adapter output SHA256: `020f149866bd30ccb6e39addea7d8aa233c20b27d96e6fa4a967d520968e4ad8`
- 저장된 Circle numerical parity max error는 `.3048877716064453`으로 별도 FAIL이다. 현재 같은 case의 base 결과에서도 정확히 같은 수치 차이가 관찰됐다.

## export 없이 동일한 fingerprint 전이를 재현

`gemma4_topology_diagnostic.py`는 같은 seed123/config/tiny profile/calibration3회를 이용해 실제 model을 prepare/calibrate/convert했다. 이후 **실제 tico.convert/export를 호출하지 않고** `case.export_invocation()`과 adapter forward만 실행했다.

결과는 원본과 before/after fingerprint가 모두 정확히 같았다. adapter forward output SHA256도 최초 캡처와 정확히 일치했다. `validate_frozen_model()`을 profile 생성 전·후와 forward 후에 호출해 모두 통과했다.

profile 생성에 따른 exact contract diff:

| 종류 | 추가 | 삭제 | 기존 항목 변경 |
|---|---:|---:|---:|
| weights | 0 | 0 | 0 |
| observers(버퍼·snapshot·scale/zp·provenance·lock 포함) | 6 | 0 | 0 |
| quantization mode owner | 1 | 0 | 0 |

추가된 observer는 모두 `wrapped.pooler_export.wrapped` 아래의 `obs_act_in`, `obs_pool_in`, `obs_pool_matmul_out`, `obs_pool_out`, `obs_pool_weight`, `obs_root_hidden_size`다. 새 mode owner는 `wrapped.pooler_export.wrapped`다. profile 생성 이후 실제 adapter forward 전후에는 모든 종류에서 추가·삭제·변경이0개다.

따라서 원래 model의 observer/qparam을 재수집하거나 변경했다는 증거는 없다. 오히려 exact contract diff는 모든 기존 상태가 보존됨을 확인했다. 원본 export 이후 hash가 profile 생성 직후 hash와 동일하므로, 실제 export와 reference 실행이 추가 상태 변화를 만들었다는 관찰도 없다.

## 문서·구현 계약

`recipes/export/README.md` 첫 부분은 Gemma4 pooler fixed profile이 별도 quantization owner라고 명시한다. 새 matrix는 자신의 grid를 만들고, source snapshot은 복사된 작은 snapshot으로 유지하며 source observer/reservoir를 공유하지 않는다. 새 matrix observer의 constant-only statistics 수집은 기존 source observer 재calibration과 다르다.

`QuantGemma4VisionModel.as_export_module()`은 원래 patch_embedder/encoder/pooler를 바꾸지 않고 `*_export` 속성으로 profile adapters를 등록한다. `QuantGemma4VisionPooler.as_export_module()`은 새 specialized pooler를 만들고 activation snapshots를 설치하며, 새 pooling matrix의 `obs_pool_weight`만 새 matrix grid를 만든다. 이 새 profile이 source model에 등록되므로 `model.named_modules()` 전체를 hash한 `_state_contract()`에는6개 observer가 추가된다.

기존 `test_pooler_export_ownership.test_actual_vision_export_preparation_preserves_valid_ownership`는 source pooler contract가 그대로인지를 검사한다. `test_refreeze_preserves_source_and_profile_snapshots_once_per_site`, `test_strict_state_dict_roundtrip_with_equivalent_profile_topology`도 source와 profile을 별도 owner로 취급한다. 전체 suite에서 이 테스트들이 실행됐다.

## 보고·harness 개선 제안

원본 global hash FAIL을 숨기거나 삭제하지 않고 다음 두 검사를 분리한다.

1. profile materialization 이전부터 존재하던 source site의 weights/observer frozen snapshot/lock/provenance/mode는 exact equality로 검사한다.
2. 허용된 profile topology 추가는 별도로 기록하고, 각 새 profile의 frozen validity와 생성 이후 export/evaluation 비변경을 검사한다.

이는 tolerance·skip을 완화하는 조치가 아니다. 전체 model tree가 항상 불변이라는 잘못된 harness 전제를 문서의 source-owner 불변 계약으로 정확히 나누는 것이다. 원본 numerical parity FAIL은 전혀 완화하지 않는다. base에서 새 profile이 생성되지 않아 hash가 같았다는 사실을 곧바로 이번 PR의 frozen-state 회귀로 분류해서는 안 된다.

## 증거와 재현

- `gemma4-topology-diagnostic.json`: 전후 전체 contract, exact diff, 출력hash, frozen validation 결과.
- `gemma4-topology-diagnostic.log`: 주요 hash/diff 요약.
- `gemma4_topology_diagnostic.py`: no-export 재현 script.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=${CANDIDATE_REPO} ${VALIDATION_VENV}/bin/python \
  ${VALIDATION_ROOT}/gemma4_topology_diagnostic.py
```

원본 증거는 `wrapper-circle/gemma4_vision_model/artifact-checks.json` 및 `runs/wrapper-circle-gemma4_vision_model.json`에 있다. 제품 코드는 수정하지 않았다.
