# Wrapper CPU 검증

`wrapper_smoke.yaml`은 저장소 설정 그대로이며 CPU/FP32, seed 123, tiny profile, calibration_iters 3이다. Numerical all 실행은 모든 50 case를 실제 prepare/calibrate/convert/forward로 검증했다. CLI의 all mode는 개별 JSON을 쓰지 않으므로 `numerical-summary.json`은 원본 로그의 case/status를 파싱한 집계다.

Circle 검증은 `run_exports.py`가 registry의 supports_circle_export=true인 48 case를 독립 subprocess에서 순차 실행한다. 각 child의 `export_case.py`는 inspector.main()에 원래 CLI 인자를 그대로 전달한다. 실제 `_export_circle`과 `tico.convert`를 호출 전후 관찰하는 spy로 감싸고 원래 구현을 같은 인자로 호출한다. 모델, quantization, export, runtime을 mock하지 않는다. 정확한 converter 입력을 복제해 Torch expected forward를 계산하며, 저장된 .circle 파일을 다시 읽고 CircleDocument.verify와 built-in CircleReferenceRuntime을 실행한다. 모든 출력 개수/shape/dtype 및 atol=1e-5, rtol=1e-4 parity를 검사하고 입력 불변성과 frozen observer/weight/mode 계약 hash를 비교한다. 이 tolerance는 저장소 CPU export regression driver와 동일하며 실패를 피하려고 완화하지 않는다.

각 case의 `wrapper-circle/CASE/artifact-checks.json`은 inspector exit code, artifact 저장, 구조 검증, runtime 실행, parity, frozen 불변성을 별도로 기록한다. `wrapper-circle/CASE.json`은 inspector의 원래 numerical/export 결과다. root `runs/wrapper-circle-CASE.json`은 child wrapper의 exit code와 timeout을 기록한다.

재실행은 새 VALIDATION_ROOT를 만들고 run.py와 wrapper-harness의 경로를 새 root로 바꾼 뒤 run_exports.py를 실행한다. 단일 case는 root runs JSON의 command를 사용한다.
