# Comparator 보완 검증

검증 대상: `tico/quantization/recipes/debug/export_regression.py` (HEAD `6b2f6a0511458d6a750e2a260d706080a78149dd`). 제품/검증 스크립트 원본은 수정하지 않았다.

`comparator_extra_negatives.py`는 기존 comparator unit fixture를 사용한다. Circle compiler/runtime integration으로 분류하지 않는다. 실행 로그, 원래 exit 1 및 `comparator_extra_negatives.json`을 보존했다.

| 손상시킨 증거 | 기대 comparator 결과 | 실제 결과 | Negative test |
|---|---|---|---|
| 양쪽에서 모든 Torch output 배열 삭제 + `torch_parity=[]` | FAIL | `ok=true` | FAIL |
| 양쪽에서 모든 input 배열 삭제 | FAIL | `ok=true` | FAIL |

원인:

- `export_regression.py:717`의 coverage 조건은 probe당 Circle 배열 하나 이상만 확인한다 (`:720`). Graph의 input/output 개수와 정확한 배열 인덱스 집합을 비교하지 않는다.
- `export_regression.py:732`는 `all()`을 비어 있는 parity 목록에도 적용하므로 `--require-torch-parity`가 누락 증거를 거부하지 않는다.
- 양쪽을 동일하게 훼손하면 `array_keys_equal`도 참으로 남는다 (`:716`).

최소 수정 제안(미적용): 각 stage/probe에서 실제 Circle graph I/O 개수를 기준으로 `pN_input_0..I-1`, `pN_circle_0..O-1`, `pN_torch_0..O-1` 인덱스 집합을 검증하고 parity 항목 수가 O와 일치하며 비어 있지 않도록 검증한다. 실제 기록의 shapes/dtypes와 graph contracts도 확인한다. 해당 실패는 RoPE의 의도적 수치 변경과 무관하다.

이번 검증에서는 원본 comparator 결과를 그대로 보존하고, 별도의 `audit_circle_evidence.py`로 실제 `.circle` graph I/O와 모든 probe 배열 개수·NaN/Inf·Torch parity를 다시 대조한다. 원본 comparator의 PASS만으로 완료를 판단하지 않는다.
