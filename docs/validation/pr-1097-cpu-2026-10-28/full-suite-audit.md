# 전체 기본 suite audit

- 수집 4760, unittest 보고 실행 4760, 결과 `OK (skipped=158)`.
- verbose 결과 4760개 파싱: {'PASS': 4602, 'SKIPPED': 158}.
- unittest 987.679초, 프로세스 wall time 995.207초, 원본 exit code 0.
- discovery import error 0개, 결과 누락 0개.
- decorator skip 157개, 실행 시 동적 skip 1개.
- audit consistency: True.

| family | 수집 | 실제 PASS | SKIPPED | FAIL/ERROR |
|---|---:|---:|---:|---:|
| llama | 96 | 93 | 3 | 0 |
| qwen_vl | 209 | 197 | 12 | 0 |
| gemma4 | 422 | 354 | 68 | 0 |
| gemma4_assistant | 52 | 52 | 0 | 0 |
| fairseq | 50 | 50 | 0 | 0 |

모든 family의 PASS가 0보다 크면 import/전체 skip에 의한 family 누락은 없다. 구조·runtime·수치 개별 assertion은 각 테스트와 원본 로그를 참조한다.

동적 skip:

- `quantization.evaluation.test_evaluation.TestEvaluationUtils.test_plot_two_outputs_installed_plotext`: Optional plotting dependency plotext is not installed

원본 로그: `logs/full-suite.log`. 전체 test ID/status/line와 skip 사유: `full-suite-summary.json`.

Audit 시점 source status: {"head": {"cwd": "${CANDIDATE_REPO}", "revision": "9f631d6651ce3780e3a8e054ab02910b9bd40fdc", "porcelain": "", "tracked_diff_exit": 0}, "base": {"cwd": "${BASELINE_REPO}", "revision": "261bda28ec55fdac42f59346c950326e4065f8fa", "porcelain": "", "tracked_diff_exit": 0}}
