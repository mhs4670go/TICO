# TICO PR #1073 검증 자료

검증 결과는 **FAIL**이다. 실행한 범위, 원본 실패, 원인 분석과 미검증 범위를 함께 공개한다.

- 검증 대상: `mhs4670go/TICO`의 `ea`, Samsung/TICO PR #1073
- 검증 HEAD: `6b2f6a0511458d6a750e2a260d706080a78149dd`
- 변경 전 baseline: `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`
- 실행 ID: `20261022T0404Z` — CPU 전용 Codex Cloud 실행

## 읽는 순서

1. [최종 한국어 보고서](PR1073_VALIDATION.md): 모델별 결과표, 실패 원인, 미적용 수정안과 미검증 범위.
2. [검증 수준과 판정 원칙](VALIDATION_LEVELS.md), [모델별 JSON](model_results.json): unit/mock, tiny real-HF, tiny pretrained, actual Circle을 구분한다.
3. [실행 전체 요약](summary.json), [재현 명령](commands.sh), [환경](environment.json), [패키지 버전](packages.json): 명령별 종료 코드와 증거 경로를 확인한다.
4. 아래의 분야별 문서와 원본 JSON/로그를 대조한다. [ChatGPT 검토 요청문](CHATGPT_REVIEW_PROMPT.md)을 사용할 수 있다.

## 핵심 문서와 원본 증거

| 범위 | 설명 | 원본 결과·증거 |
|---|---|---|
| Unit / regression | [테스트 실패 분석](unit_test_analysis.md), [정적 검사](STATIC_CHECKS.md) | [unit 결과](unit_results.json), [분류](unit_failure_classification.json), [canonical 로그](changed_unit_discovery.log), [quantization 로그](quantization_suite.log), [전체 suite 시간 제한](full_unit_suite.command.json), [정적 진단](static_diagnostics.json) |
| Eager / frozen 상태 | [요구사항 대응](EAGER_COVERAGE.md), [실행 결과](EAGER_SUPPLEMENT_RESULTS.md) | [tiny real-HF/PTQ](supplemental_eager_state/summary.json), [실제 PNG processor+연결 decode](generated_image_eager/summary.json) |
| 정상 generate API | [이미지 generation 분석](IMAGE_GENERATION_FINDINGS.md), [generation 문서](GENERATION_PROBE.md) | [FP/Quant 결과](image_generation_probe/summary.json), [원인 분류](image_generation_analysis.json), [독립 fresh-baseline 재현](qwen_image_generation_fresh_baseline/summary.json) |
| 작은 pretrained Llama | [Pretrained smoke](PRETRAINED_SMOKE.md) | [전체 요약](pretrained_summary.json), [FP/Quant 프로토콜](pretrained_protocol_comparison.json), [선택 layer0 Circle](pretrained_selected_stage/summary.json) |
| Actual Circle 전체 | [범위·원본 판정](CIRCLE_VALIDATION.md) | [모델별 실행 요약](circle_validation_summary.json) |
| Llama RoPE | [수치 진단](PARITY_DIAGNOSIS.md) | [32개 검사 결과](llama-rope-checks.json), [old/new 원본 FAIL](tiny-llama-old-new-v2/summary.json), [독립 HF·동일 외부 RoPE 진단](tiny-llama-old-new-v2/llama_rope_diagnosis.json), [완전성 감사](tiny-llama-old-new-v2/independent_evidence_inventory.json) |
| 긴 eager 격리 | [native16384 / eager4096+4](LLAMA_LONG_EAGER_ISOLATION.md) | [분석](llama_long_eager_analysis.json), [원본 strict FAIL](tiny-llama-native16384-long-eager-full/summary.json), [독립 출력 감사](tiny-llama-native16384-long-eager-full/independent_full_evidence.json) |
| Qwen Circle | [진단](QWEN_PARITY_DIAGNOSTIC.md) | [per-layer FAIL](tiny-qwen-old-new-cached/summary.json), [한 번의 PNG eager 전후 full-text PASS](tiny-qwen-full-text-png-dirs-retry/summary.json) |
| Gemma Circle | [독립 text stage](GEMMA_TEXT_STAGE_VALIDATION.md), [half-grid 소실 진단](GEMMA_TEXT_PARITY_DIAGNOSIS.md), [별도 full-text 진단](GEMMA_FULL_TEXT_DIAGNOSTIC.md) | [vision 원본 ERROR](tiny-gemma-old-new-image70/summary.json), [text strict FAIL](gemma_text_strict_comparison.json), [trace 분석](GEMMA_TEXT_PARITY_DIAGNOSIS.json), [full-text strict FAIL](GEMMA_FULL_TEXT_DIAGNOSTIC.json) |
| 검증도구·calibration | [Comparator 결함](COMPARATOR_GAPS.md), [calibration 감사](CALIBRATION_PATH_AUDIT.md), [인프라 보완](FIXTURE_INFRASTRUCTURE.md) | [Comparator negative 결과](comparator_extra_negatives.json) |

## 결과 해석에서 지켜야 할 구분

- **68개**는 완료된 random tiny actual-Circle stage의 수다. PASS 수가 아니며, 실패한 Gemma vision과 별도 pretrained 실행을 제외한다.
- RoPE **32개**는 실제 tiny-HF RoPE 테스트와 synthetic-owner/static-input unit을 함께 포함한다. Llama-3.2 pretrained 검증이 아니다.
- Eager 전후 graph/qparams/출력의 exact equality는 Torch–Circle parity PASS나 pre-PR equivalence PASS를 뜻하지 않는다. 원래 `atol=1e-5`, `rtol=1e-4`, `--require-torch-parity` 실패를 유지했다.
- Llama host cos/sin 차이만 독립 HF 근거에 따라 `EXPECTED_NUMERICAL_CHANGE`로 분석했다. Qwen/Gemma 실패나 다른 Torch–Circle 오차를 그 예외로 처리하지 않았다.
- Qwen full-text PASS는 실제 PNG eager forward **1회** 전후의 text graph 검증이다. 전체 multimodal/video export나 연결 cached decode의 Circle PASS가 아니다.
- 실제 pretrained는 **Maykeye/TinyLLama-v0, 4.62M parameters**의 고정 revision을 쓴 작은 smoke다. PPL/likelihood는 완료했지만 Quant normal generation은 실패했다. Llama/Qwen generation API 오류는 독립 baseline 실행에서도 재현했다. 그 fresh-baseline calibration은 same-checkpoint export 비교가 아니다.
- 선택 pretrained layer0의 prefill hidden/V는 tolerance를 위반하고 delta-K는 rtol 범위 내다. 전체 bundle은 시간 제한으로 미완료다. 전체 unit suite 역시 시간 제한으로 미완료이며 skip/미완료를 PASS로 세지 않았다.
- 대형 pretrained Llama-3.2/Qwen3-VL/Gemma4의 2048-token old/new export, pretrained VLM 평가 및 실제 Video-MME metric은 BLOCKED다. Circle reference는 실제 NPU 검증이 아니다.

## 공개본의 범위와 재현 제한

이 브랜치는 **선택한 텍스트 보고서·스크립트·JSON·로그의 공개 사본**이다. [공개 파일 목록과 제외 내역](PUBLICATION_MANIFEST.json)에서 포함 경로, 원본 SHA256/크기와 제외 이유를 확인한다. [원본 전체 패키지 manifest](artifact_manifest.json)는 sealed 실행 산출물의 인덱스이며, 공개 브랜치의 모든 파일이 그 전체 패키지에 해당한다는 뜻은 아니다.

모델/checkpoint, `.circle`, `.npz`, tensor `.pt`, graph dump/`records/*.json`, 원본 HF tokenizer 파일과 일부 중복된 큰 frozen-state JSON은 이 텍스트 공개본에서 제외된다. 따라서 공개 JSON은 실행의 상태·수치·식별 정보를 읽는 증거이며, 제외한 binary 자체를 이 브랜치에서 다시 실행하거나 byte 단위로 감사했다는 근거는 아니다. 정확한 제외 목록은 공개 manifest를 따른다. 동일 binary의 재실행·전체 출력 재계산에는 원본 Cloud 산출물 또는 별도 전달된 전체 archive가 필요하다.

원본 보고서의 `/workspace/...` 경로, ZIP 포함 설명과 “commit/push 없음”은 **검증 실행 당시의 기록**이다. 이후 허가받은 이 보고서 공개 작업과 구분한다. Sealed 원본은 수정하지 않았다. 공개 사본의 일부 로그·JSON·스크립트에 있는 개인 홈 디렉터리 접두사만 `<USER_HOME>`으로 비식별화했고, 검증 수치와 판정은 유지했다. 공개 manifest에 원본/공개본 SHA256과 비식별화 횟수를 각각 기록한다. `commands.sh`를 다른 환경에서 실행하려면 checkout·결과 root·홈 경로와 누락 binary를 준비하고 export output에 새 경로를 사용해야 한다. `summary.json`의 내부 경로도 공개 manifest의 포함 여부와 함께 읽는다.
