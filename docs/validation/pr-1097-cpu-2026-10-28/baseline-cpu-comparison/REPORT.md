# CPU driver strict parity failure baseline 비교

각 revision의 코드에서 seed1073 FP32 tiny Llama/Qwen을 독립 생성하고 표준 PTQ calibration 2×8 token 및 cache decode 후 freeze했다. head checkpoint를 base에 넘기지 않았으며 schema migration/cast/tolerance 수정은 없다. 모든 실행은 `checked_module.py`에서 실제 `tico.__file__`을 먼저 assertion한다.

Frozen serialization schema가 달라 checkpoint file hash의 동일성을 요구하지 않는다. 공유된 의미의 parameter tensor, mode, observer dtype/qscheme/axis/lock/enabled, scale·zero-point의 값/shape/dtype 및 수집 min/max를 비교했다. Head에 새로 추가된 schema/provenance field가 base checkpoint에도 존재한다고 가정하지 않았다. 각 파일 hash와 config/calibration provenance는 별도로 보존했다.

| case | head | base | 분류 | weights/활성grid/입출력 hash |
|---|---|---|---|---|
| llama-16 | FAIL | FAIL | 기존 수치 parity 실패 (최신 base에서 동일 weight·활성 frozen grid·입력·Torch/Circle 출력과 같은 strict parity 실패 재현; 비활성/schema metadata 차이는 별도 보존) | True/True/True |
| qwen3_vl-16 | FAIL | FAIL | 기존 수치 parity 실패 (최신 base에서 동일 weight·활성 frozen grid·입력·Torch/Circle 출력과 같은 strict parity 실패 재현; 비활성/schema metadata 차이는 별도 보존) | True/True/True |
| qwen3_vl-32 | FAIL | FAIL | 기존 수치 parity 실패 (최신 base에서 동일 weight·활성 frozen grid·입력·Torch/Circle 출력과 같은 strict parity 실패 재현; 비활성/schema metadata 차이는 별도 보존) | True/True/True |

Raw observer equality flag는 모두 false로 그대로 보존했다. Head helper가 lock을 `snapshot.locked`로 기록하고 base helper는 `_qparams_locked`로 기록하는 위치 차이가 있다. 위치를 명시적으로 대응하면 Llama139개 observer의 모든 비교 필드가 같고, Qwen의 활성 frozen observer273개도 모두 같다. Qwen의 나머지 실제 차이는 unused vision scaling observer2개 제거, fused text scale/raw-logit observer4개의 empty grid·enabled 상태, 미사용 DeepStack1의 dummy→empty grid이다. Head에서는 이5개 site의 frozen snapshot이 None이며 fixture는 DeepStack index0만 실행한다. `canonical_observer_differences`에 16개 필드/존재 차이를 생략 없이 보존했으므로 전체 observer state가 같다고 주장하지 않는다. 근거 source: Qwen text attention `_all_observers`의 scale_fusion==none gating, text model `local_observer_groups`의 independent DeepStack capabilities, PR vision attention diff의 unused observer 제거.

Connected saved-Circle replay:

- Qwen capacity 16: head/base FAIL/FAIL, 22 actual calls, 기존 실패, frozen state preserved True/True.
- Qwen capacity 32: head/base FAIL/FAIL, 22 actual calls, 기존 실패, frozen state preserved True/True.

원본 strict Torch parity 실패를 그대로 FAIL로 보존한다. export 전후 state와 배열 동일성은 PASS지만 Torch↔Circle 수치 parity 실패는 별개다. 이는 최신 base에서도 동일한 수치로 재현된 기존 export/reference 경계 문제다. 실제 NPU integer 정확도·성능 또는 full-size quality를 판정하지 않는다.

Commands/logs: root `runs/baseline-cpu-*.json`, `logs/baseline-cpu-*.log`; nested worker logs and per-stage JSON/NPZ/Circle: `baseline-cpu-comparison/<case>/`. 상세 SHA와 모든 배열 hash: `comparison.json`.
