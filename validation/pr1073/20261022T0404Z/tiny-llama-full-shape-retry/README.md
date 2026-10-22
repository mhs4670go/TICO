# Tiny Llama full-model actual Circle 재검증

대상은 같은 baseline-created frozen `fixtures/baseline_llama_v2.pt`이며, export 8 / eager prefill 12 + decode 4 / recorded capacity 64 / CPU float32 조건이다. Pretrained나 2048-token 검증이 아니다.

원본 runner의 첫 시도 `tiny-llama-full-original/`은 baseline adapter의 `Circle export requires at least one calibration input.` 오류로 exit 2였다. 이 로그는 보존했다. Baseline source를 수정하지 않고 별도 runner 복사본에 full-model shape example인 CPU long zeros `[1,8]`만 `ctx.calibration_inputs`로 넣었다. Calibration이나 qparam 업데이트는 실행하지 않았다. Diff 및 두 코드 SHA256은 상위 `export_regression_full_shape_fixture.diff`와 `.provenance.json`에 있다. 원래 frozen-state 검사·비교·threshold는 그대로이다.

| 검증 항목 | 실제 결과 |
|---|---|
| 원본 strict run | FAIL / exit 1, comparisons_complete=true |
| 실제 Circle stage | full `model.q.circle` 1개 × baseline / candidate before / candidate after_eager |
| 출력 증거 | phase별 logits + layer별 K/V 총 5개, 모든 배열 finite 및 graph I/O 개수 확인 |
| baseline→candidate | Circle 구조, host 및 모든 입력/Circle/Torch 출력 배열 exact |
| candidate eager 전→후 | Circle 구조, host 및 모든 배열 exact |
| Frozen weights / observer stats / qparams / mode | 세 phase 및 eager 12+4 후 모두 보존 |
| Torch→Circle logits | 세 phase에서 동일한 strict parity FAIL; max abs `0.0055593401193618774` |
| Torch→Circle 4개 KV 출력 | 모두 exact |

Logits 최대 오차 위치는 `[0,5,10]`으로 Torch `-0.12508507072925568`, Circle `-0.13064441084861755`이다. 이는 baseline에도 동일하게 존재한다. 이번 추가 검증만으로 모든 오차 원인을 확정하지 않으며, full-model strict validation을 PASS로 보고하지 않는다.

독립 검증 `audit_tiny_llama_full_shape_retry`도 strict parity 때문에 exit 1이다. `independent_evidence_inventory.json`에서 입력/출력 증거의 정확한 개수 및 finite 여부를 확인했고, `component_summary.json`에 원본 FAIL과 구조·배열 exact 성분 결과를 분리했다. Threshold `atol=1e-5`, `rtol=1e-4`를 바꾸지 않았다.

원본 실패·새 fixture 재실행·독립 감사 결과는 서로 덮어쓰지 않았다. 제품·baseline 코드, PR metadata를 수정하지 않았다.
