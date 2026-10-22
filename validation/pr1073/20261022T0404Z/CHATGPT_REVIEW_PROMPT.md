# ChatGPT 검토 요청문

아래 내용을 복사하고, 이 문서 또는 같은 디렉터리의 README GitHub URL을 함께 제공하세요.

---

TICO PR #1073의 실제 실행 검증 자료를 검토해 주세요. 먼저 [README](README.md), [최종 보고서](PR1073_VALIDATION.md), [검증 수준](VALIDATION_LEVELS.md), [모델별 JSON](model_results.json)을 읽고, 주장별로 아래 원본 JSON/로그를 대조해 주세요. 상대 링크는 이 문서가 있는 동일 GitHub branch/directory 기준입니다.

검증 HEAD는 `6b2f6a0511458d6a750e2a260d706080a78149dd`, pre-PR baseline은 `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`입니다. `61f3a05`나 최신 main을 baseline으로 대신하지 마세요. 이 자료의 최종 판정은 **FAIL**입니다.

검토할 질문:

1. Calibration/export/evaluation 길이, 원본 positional/RoPE config, mask/position/cache/padding/PPL 옵션과 frozen 상태에 대한 결론이 [요구사항 대응](EAGER_COVERAGE.md), [실제 tiny-HF 실행](supplemental_eager_state/summary.json), [PNG processor+연결 decode](generated_image_eager/summary.json)로 뒷받침되나요? Unit/mock과 실제 integration을 구분해 주세요.
2. [Llama old/new 원본 결과](tiny-llama-old-new-v2/summary.json)와 [독립 HF RoPE 진단](tiny-llama-old-new-v2/llama_rope_diagnosis.json)에서 근거 있는 host RoPE 변경만 의도적 차이로 분류했나요? [긴 eager 실험](llama_long_eager_analysis.json)의 static capacity8/native16384/eager4096+4 범위를 2048-token pretrained static export로 과장하지 않았나요?
3. [Circle 요약](circle_validation_summary.json), [Gemma text 비교](gemma_text_strict_comparison.json), [Gemma 실제 trace](GEMMA_TEXT_PARITY_DIAGNOSIS.json), [별도 full-text 진단](GEMMA_FULL_TEXT_DIAGNOSTIC.json)에서 old/new, candidate eager 전후, Torch–Circle parity 및 출력 완전성을 각각 구분했나요? Exact before/after equality로 strict FAIL을 지우면 안 됩니다.
4. [작은 pretrained smoke](PRETRAINED_SMOKE.md), [원본 수치](pretrained_summary.json), [FP/Quant 조건 비교](pretrained_protocol_comparison.json), [선택 layer0 결과](pretrained_selected_stage/summary.json)는 실제 완료 범위를 정확히 나타내나요? [generation 분석](IMAGE_GENERATION_FINDINGS.md)의 baseline 재현과 same-checkpoint 비교를 혼동하지 않았나요?
5. [테스트 분석](unit_test_analysis.md), [static 검사](STATIC_CHECKS.md), [comparator negative](COMPARATOR_GAPS.md), [인프라 보완](FIXTURE_INFRASTRUCTURE.md)에서 재현되는 제품 문제, 테스트/검증 인프라 문제, baseline 공통 실패와 미확정 원인을 적절히 분류했나요?

판정 원칙:

- **Unit/mock, random tiny real-HF, tiny pretrained, Torch Export, actual Circle, 실제 NPU**를 별도 수준으로 다루세요. RoPE32개는 real-HF와 synthetic/static-input unit의 혼합이며 Circle68stage는 PASS 수가 아닙니다.
- Qwen full-text PASS는 한 번의 PNG eager forward 전후 text graph 범위입니다. Main per-layer 실패, multimodal/video 전체 export, 연결 cached decode의 Circle 검증을 대신하지 않습니다.
- TinyLLama-v0 4.62M smoke를 Llama-3.2 또는 대형 pretrained 검증으로 표현하지 마세요. 소수 샘플 PPL 차이로 전반적 양자화 품질 향상을 결론 내리지 마세요.
- 원래 tolerance `atol=1e-5`, `rtol=1e-4`와 raw FAIL/ERROR/exit code를 유지하세요. `EXPECTED_NUMERICAL_CHANGE`는 독립 근거가 있는 Llama host 차이에만 적용하세요. 시간 제한/skip/누락 자원은 PASS가 아닙니다.
- `evaluation_protocol.json`은 실행 준비 기록입니다. 실제 metric·완료 로그를 함께 확인하세요. 분석 스크립트 exit0도 제품 parity PASS를 뜻하지 않습니다.
- [공개 manifest](PUBLICATION_MANIFEST.json)에 없는 binary/큰 파일은 읽었다고 가정하지 마세요. JSON 수치를 검토하는 것과 원본 checkpoint/Circle/NPZ를 재실행하는 것은 다릅니다. Workspace 절대 경로와 전체 ZIP 설명은 원래 Cloud 실행의 기록입니다.

한국어로 답해 주세요. 결론과 근거 문서/JSON 경로를 연결한 간단한 표, 확정 실패와 미확정 원인, 과장 또는 빠진 증거, 제품에 적용하지 않은 최소 수정안, 추가 검증에 필요한 자원을 구분해 주세요. 접근하지 못한 파일은 명시하고 결과를 추정하지 마세요. 이번 요청은 자료 검토이며 제품 코드·판정 기준·PR 상태의 변경이나 새 대규모 모델 다운로드 요청이 아닙니다.
