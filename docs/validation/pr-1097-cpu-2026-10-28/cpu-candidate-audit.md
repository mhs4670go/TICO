# Candidate CPU driver 실행 층별 audit

이 결과는 head 내부 검증이며 baseline 비교는 `baseline-cpu-comparison/REPORT.md`의 독립 실행으로만 판단한다.

기존 CPU regression/generation/cache coverage/evaluation 테스트 135개가 실제 실행됐다(110/7/12/6). 각 child test summary에 skip/failure/error가 보존되어 있다. Fixture는 표준 adapter + PTQStage로 prepare/calibration/convert하며, seed1073, FP32, 2×8 calibration tokens 및 cache decode를 유지했다.

| 격리 case | phase별 저장·재읽기 구조 검증·reference 실행 성공 stage | phase별 Torch parity stage PASS/FAIL | eager 전후 배열 비교 |
|---|---:|---:|---:|
| Llama16 | before7 / after7 | 3/4 | 129/129, max error0 |
| Qwen16 | before9 / after9 | 6/3 | 131/131, max error0 |
| Qwen32 | before9 / after9 | 5/4 | 131/131, max error0 |

총 50개 phase별 Circle 파일을 저장하고 실제 파일을 `CircleDocument.load_mapped()`로 다시 읽어 `verify(raise_on_error=True)`까지 실행했다. 실제 compiler와 built-in CircleReferenceRuntime으로 86개 numerical probe를 실행했다. Torch output별 strict 비교 156개 PASS / 62개 FAIL, stage별 28개 PASS / 22개 FAIL이며 overall strict status는 세 case 모두 FAIL이다. 검증 assertion이나 tolerance를 바꾸지 않았다(`atol1e-5, rtol1e-4`).

eager 전후 비교 391개 배열이 정확히 같고(max abs0) 구조·execution mode도 같으며 모든 frozen/export/eager state가 보존됐다. 따라서 strict Torch parity FAIL을 lifecycle drift나 observer 재수집으로 보고하지 않는다.

실제 저장된 Qwen stage를 이어서 실행한 connected 경로:

- capacity16: 22개 호출 완료, 호출별 Torch parity 19PASS/3FAIL, output별38PASS/5FAIL. Image-conditioned prefill cache valid12에서 네 번 decode하여16까지 연결, full-cache overflow 거부, frozen state 보존.
- capacity32: 22개 호출 완료, 호출별18PASS/4FAIL, output별36PASS/7FAIL. valid28→32, overflow 거부, frozen state 보존.

세부 artifact 이름, probe별 error, frozen assertion, connected calls 및 원본 summary는 `cpu-candidate-audit.json`과 `cpu-validation/`에 있다. 파일 저장·구조·실행 성공은 수치 parity 성공이나 target NPU integer 실행을 의미하지 않는다.
