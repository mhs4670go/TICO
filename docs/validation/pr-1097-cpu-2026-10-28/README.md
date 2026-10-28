# PR #1097 CPU 검증 보고서 묶음

[최종 보고서](VALIDATION_REPORT.md) · [구조화 결과](VALIDATION_RESULTS.json) · [재실행 방법](REPRODUCE.md) · [실제 명령 목록](COMMANDS.sh)

판정은 **INCOMPLETE**다. 기본 suite 4,602 PASS/158 skip/실패0, wrapper numerical50/50 PASS다. 필수 Circle strict parity 실패는 base에서도 같은 입력·출력으로 재현됐다. 새 PR 회귀를 입증한 사례는 없지만 전체 Circle 검증을 PASS로 표시하지 않았다. 세부 결과·제약·최소 재현과 수정 제안은 최종 보고서를 참조한다.

## 보관 범위와 원본 구분

이 브랜치는 검증 당시 base `261bda28ec55fdac42f59346c950326e4065f8fa`에서 만든 보고서 전용 브랜치다. 제품 코드가 포함된 PR head는 `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`이며 실제 검증 tree는 `c81b5f254023b21bb76f1f6d04405b5c7d3f54b2`다. 이 보고서 브랜치 자체를 검증 대상으로 혼동하지 않는다.

기본 suite를 포함한 로그, 실행별 exit code·시간, 결과 JSON, coverage map, 작은 입력·설정, harness snapshot을 모았다. checkpoint/Circle/pt2/NPZ/safetensors/cache 및 중복 graph/tensor dump는 commit하지 않았다. 생성물은 보존한 harness와 seed/config로 다시 만든다. 이미 repository에 있는 정책 문서는 중복 저장하지 않고 원본 revision의 문서 hash만 남겼다.

- [PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json): 원본 파일 SHA256/크기와 이 묶음의 변환본 SHA256/크기, 제외 파일 및 사유. 원본 로그도 경로 변수화·행 끝 공백 정리한 **보고서 사본**이다.
- [ORIGINAL_ARTIFACT_MANIFEST.json](ORIGINAL_ARTIFACT_MANIFEST.json): 최초 실행 때의 원본 생성물 경로·크기·hash. 여기에 나오는 binary 파일이 이 commit에 포함됐다는 뜻은 아니다.
- [VALIDATION_RESULTS.json](VALIDATION_RESULTS.json): 중복된 하위 evidence는 같은 묶음의 상대 경로로 참조한다. full tensor inventory를 생략한 곳은 명시적 packaging note를 붙였다. 테스트 결과·실패 상태·숫자·원래 exit code는 바꾸지 않았다.
- Python harness는 `*.py.txt`로 보관해 repository 기본 test discovery/static-analysis 범위를 늘리지 않는다. 이 사본들은 제품 코드나 상시 테스트가 아니다.

## 경로와 실행 환경 복원

환경 전용 경로는 `${VALIDATION_ROOT}`, `${CANDIDATE_REPO}`, `${BASELINE_REPO}`, `${VALIDATION_VENV}`로 바꿨다. JSON의 변수 문자열은 역사적 명령·환경 기록이며 자동으로 확장되지 않는다. 아래 도구가 실행용 파일에서 실제 경로로 치환한다.

```bash
# 각각 해당 SHA로 고정한 별도 checkout과 문서의 CPU 환경을 먼저 준비한다.
REPORT_DIR=docs/validation/pr-1097-cpu-2026-10-28
python "$REPORT_DIR/restore_harness.py.txt" \
  --output /tmp/pr1097-validation-new \
  --candidate /path/to/pr-head-checkout \
  --baseline /path/to/base-checkout \
  --venv /path/to/configured-venv
export VALIDATION_ROOT=/tmp/pr1097-validation-new
export CANDIDATE_REPO=/path/to/pr-head-checkout
export BASELINE_REPO=/path/to/base-checkout
export VALIDATION_VENV=/path/to/configured-venv
source "$VALIDATION_VENV/bin/activate"
```

복원 도구는 작은 script/config/data만 복사하며 모델 실행·설치·네트워크 요청을 수행하지 않는다. 출력 경로는 새 디렉터리여야 한다. 로그·완료 marker·이전 결과를 복사하지 않아 새로운 실행의 성공으로 오인하지 않는다. 이후 [REPRODUCE.md](REPRODUCE.md)의 명령을 사용한다. 원본 `cpu_validation --output`도 아직 존재하지 않는 경로를 요구한다. `COMMANDS.sh` 전체를 기존 경로에서 무작정 재실행하지 않는다.

현재 PR/base tree가 바뀌면 같은 검증이라고 주장할 수 없다. 재실행마다 SHA와 `tico.__file__`을 다시 확인한다. 대형 pretrained 품질·성능, 실제 NPU integer 실행, 외부 데이터 benchmark가 검증됐다는 뜻은 아니다.
