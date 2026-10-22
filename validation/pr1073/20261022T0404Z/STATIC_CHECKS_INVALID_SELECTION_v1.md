# INVALID_SELECTION_v1 — 사용 금지

이 중간 보고서의 최초 baseline PASS 근거는 실제 검사 파일0개였으므로 무효다. 원래 명령과 로그는 그대로 보존하며, 최종 보고서는 명시적 paths-from 재실행 로그만 사용한다. 아래는 수정 이전 중간 보고서의 보존본이다.

# 정적 검사

현재 확정 결과: **UFMT FAIL**. 전체 표준 명령의 첫 실행은900초 timeout으로 미완료이며, 원본 종료 코드124와 빈 buffered 로그를 보존했다. 변경 파일의 정적 검사도600초 timeout으로 미완료였다. 전체 표준 명령은 두 테스트 프로세스 종료 및 메모리 기준을 확인한 뒤04:48 UTC에3600초 cap으로 재실행 중이다. 제품 코드, 원래 설정, formatter patch는 적용하지 않았다.

| 실행 | 범위 | 상태 | 종료 코드 | 증거 |
|---|---|---|---:|---|
| `./ccex format --no-apply-patches` 첫 실행 | 전체 UFMT/Pylint/mypy | TIMEOUT / INCOMPLETE |124| `format_all.log`, `format_all.command.json` |
| 독립 UFMT | 전체 tracked Python 파일 |FAIL,19파일|1| `format_ufmt_all.log`, `ufmt_findings.json` |
| 최초 baseline UFMT counterpart | 파일 선택 0개로 무효 |NO_FILES_CHECKED|0| `baseline_lintrunner_path_probe.log` |
| 변경 파일 정적 검사 |PR변경63개 Python 파일 |TIMEOUT / INCOMPLETE|124| `format_changed.log`, `format_changed.command.json` |
| 전체 표준 명령 재실행 |전체 UFMT/Pylint/mypy |RUNNING|미확정| `format_all_retry.log`, `static_resource_samples.jsonl` |

19개 formatting 실패는 모두 이번 PR diff에 속한다.8개는 새 파일이며, 나머지11개에 대한 최초 baseline 명령은 verbose 감사에서 검사 파일이0개였음이 확인되어 PASS를 철회했다. 명시적 --paths-from을 이용한 baseline 재검증 결과를 별도로 확정한다. 원본 패치는 로그에만 남겼으며 적용하지 않았다.

| candidate 파일 | 첫 reformat 위치 | baseline 구분 |
|---|---:|---|
| `test/quantization/recipes/test_export_llama_bundle.py` |55|기존 원본 재검증 진행|
| `test/quantization/recipes/test_export_regression.py` |46|새 파일|
| `test/quantization/wrapq/utils/test_static_export_inputs.py` |22|새 파일|
| `test/quantization/wrapq/wrappers/llama/test_rope_consistency.py` |21|새 파일|
| `test/quantization/wrapq/wrappers/qwen_vl/test_quant_text_model.py` |26|기존 원본 재검증 진행|
| `tico/quantization/recipes/debug/export_regression.py` |103|새 파일|
| `tico/quantization/recipes/debug/verify_llama_rope.py` |28|새 파일|
| `tico/quantization/recipes/export/circle.py` |22|기존 원본 재검증 진행|
| `tico/quantization/recipes/export/llama.py` |20|기존 원본 재검증 진행|
| `tico/quantization/recipes/export/llama_bundle.py` |22|기존 원본 재검증 진행|
| `tico/quantization/wrapq/examples/llama/quantize_decoder_layer_prefill.py` |137|기존 원본 재검증 진행|
| `tico/quantization/wrapq/examples/quantize_full_qmodel_with_gptq.py` |49|기존 원본 재검증 진행|
| `tico/quantization/wrapq/utils/static_export.py` |16|새 파일|
| `tico/quantization/wrapq/wrappers/gemma4/export_adapters.py` |26|기존 원본 재검증 진행|
| `tico/quantization/wrapq/wrappers/llama/quant_decoder_layer.py` |31|기존 원본 재검증 진행|
| `tico/quantization/wrapq/wrappers/llama/quant_model.py` |37|기존 원본 재검증 진행|
| `tico/quantization/wrapq/wrappers/llama/static_inputs.py` |45|새 파일|
| `tico/quantization/wrapq/wrappers/qwen_vl/quant_text_model.py` |25|기존 원본 재검증 진행|
| `tico/quantization/wrapq/wrappers/qwen_vl/static_inputs.py` |19|새 파일|

최소 수정 제안(미적용): 저장소의 고정 UFMT 도구로 위19개 파일의 formatting patch를 검토·적용한다. 이는 수치 tolerance 또는 product behavior를 바꾸는 수정과 별개다. Pylint/mypy 결과는 별도 완료 후 확정한다.
