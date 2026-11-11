# TICO 테스트 정리 5단계: 실제 검증 기록

이 디렉터리는 기록 보관용 브랜치 records/test-stage5-validation-20261111에만
보관한다. 구현 PR에 합치거나 이 브랜치로 PR을 생성하기 위한 자료가 아니다.
기존 실행에서 남은 원자료를 복사한 것이며, 이 기록을 만들면서 성능 시험이나
전체 suite를 다시 실행하지 않았다.

## 코드와 기록의 대응

| 구분 | 코드 기준 / 맥락 |
|---|---|
| 로컬 변경 전 | b7ed542e9a717580456b63bcac000c16cea08ac8의 별도 source archive |
| 로컬 5단계 구현 | 커밋 생성 전 작업 트리. 후속 구현 커밋 053866cfd58245b646be433a7392e5d79c95aaaf |
| CI 타입 수정 | 739d59792fa8b4ad022ddbe3b5306c32669abb05 |
| 최초 PR CI | Samsung/TICO #1126, head 053866c, base ec15b65298a67da42ddf8de507619f1ffdfd2fbc, merge f2859cd68d5ee7cb32b030da4d5ed734aeaa35ec |
| 타입 수정 후 PR CI | head 739d597. merge ref는 ci/pr-info.json, 최종 관찰은 ci/fixed-jobs-final.json 참조 |

측정 당시의 작업 트리 및 스크립트 버전을 매 실행마다 별도 git tree/hash로
저장하지는 않았다. 특히 성능 측정 후 신규 테스트 파일에 UFMT 서식 정리가
적용되었다. 그러므로 사후 생성된 053866c를 모든 측정의 정확한 바이트 단위
소스 snapshot이라고 단정하지 않는다.

## 파일 구성

- metadata/summary.json: 공식 성능 비교 집계와 보존 검사 결과.
- metadata/verification.json: 당시 작성한 로컬 검증 요약. 아래 정적 검사 보완 사항과 함께 읽어야 한다.
- measurements/canonical/: baseline-* 및 after-*의 반복별 JSON, stdout/stderr,
  실제 ccex 전체 wall 기록, 별도 counter/preparation/individual 진단.
- measurements/excluded-preliminary/: before-* 예비·중단 실행. 공식 통계에 포함하지 않는다.
  병렬 검사와 겹친 실행, 잘못된 선택 및 불완전한 기록도 그대로 남겼다.
- validation/local/: 전체 suite, focused regression, dynamic duration 실행의 원래 로그.
- validation/local/initial-static-before-tracking/: 초기 정적 검사 성공 출력.
  신규 파일의 검사 범위를 보장하는 증거는 아니다.
- validation/ci-fix-local/: CI 오류 재현 및 타입 수정 후 로컬 검증 로그.
- ci/: GitHub 관찰 결과와 실제 job 로그의 관련 단계 발췌.
  JSON snapshot의 observed_at은 그 시점의 상태이며 나중의 완료 상태를 추정하지 않는다.
- metadata/ci-local-header-comparison.json: 로컬과 최초 CI의 verbose 테스트 헤더 비교.
- scripts/: 사용했던 계측 스크립트의 최종 보존본. 원문 보존 및 일반 source lint 대상과
  구분하기 위해 .py.txt 확장자를 사용한다.
- provenance.json: 복사 원본과 보관본의 SHA256, 발췌 범위, 처리 방법.
- SHA256SUMS: 보관 파일의 무결성 확인 값. 이 파일 자신은 제외한다.
- verify_records.py: 원자료로 통계·ID·호출 수·로그 요약 및 해시를 재확인한다.

원본 파일명은 유지했다. 원자료의 숫자, 테스트 ID, 오류 내용은 바꾸지 않았다.
ANSI 색상 코드를 제거하고 로컬 절대 경로는 CHECKOUT, BASELINE,
STAGE5_RECORDS, CI_CHECKOUT, TMP 등의 꺾쇠 괄호 label로 치환했다.
로그의 원래 줄 끝 공백은 보존한다. 이 디렉터리의 .gitattributes는 보관 파일의
자동 줄바꿈 변환을 끄고, 로그 파일에 한해서 terminal 출력의 공백 검사를 제외한다.
provenance의 source_sha256은 이 세션의 파일시스템에 남아 있던 원본 파일의
해시이며, GitHub API의 압축 전송 바이트 해시를 의미하지 않는다.
GitHub 원본의 다운로드/인증 설정 단계, 기준 checkout 복제본, .circle/.pt2,
모델·Python cache 및 commit-message.txt는 커밋하지 않았다.

## 로컬 결과

| 명령 | 실제 결과 |
|---|---|
| focused regression, 아래 6개 모듈 | 71개, OK |
| ./ccex test -k test.modules.net.mlp_dyn --durations 3 | 1개, OK |
| ./ccex test | 5,018개 실행, 4,969개 통과, 49개 skip, 실패/오류 0, 종료 코드 0 |
| 타입 수정 후 Circle harness 모듈 | 7개, OK |
| 타입 수정 후 파일 직접 지정 cache-free mypy | 1개 source file, 오류 없음 |
| 타입 수정 후 CI와 동일한 전체 UFMT 및 PYLINT,MYPY 명령 | 각각 종료 코드 0, No lint issues |

~~~bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m unittest -v \
  test.unit_test.utils.test_lazy_generated_tests \
  test.unit_test.utils.test_circle_harness \
  test.unit_test.utils.test_ccex_test \
  test.unit_test.utils.test_result_validation \
  test.unit_test.circle.runtime.test_executor \
  test.unit_test.utils.test_runtime_independence
./ccex test -k test.modules.net.mlp_dyn --durations 3
./ccex test

# 이후 CI 타입 수정 검증
python -m unittest -v test.unit_test.utils.test_circle_harness
python -m mypy --no-incremental --cache-dir=/dev/null \
  --follow-imports=silent --config-file=mypy.ini \
  test/unit_test/utils/test_circle_harness.py
lintrunner --all-files --take UFMT --config .lintrunner.toml
lintrunner --all-files --take PYLINT,MYPY --config .lintrunner.toml
~~~

5,018개 전체 실행은 053866c 구현 시점의 기록이다. 739d597 타입 수정 뒤 로컬에서
전체 suite를 다시 실행한 것으로 읽으면 안 된다. 기록 생성 시에도 재실행하지 않았다.
내부 opt-in, 원격 모델/체크포인트, GPU, 실제 외부 backend 및 실제 Python 3.10/3.11
실행은 수행한 것으로 계산하지 않는다. 구형 Python용 duration result 경로는
당시 Python 3.12에서 capability를 바꾸는 회귀 테스트로 확인했다.

### 초기 정적 검사 기록의 정정

초기 format-full.log 및 format-changed.log의 성공 종료 자체는 실제 관찰이다.
그러나 이 명령들은 신규 파일이 git에 추가되기 전 실행되어, 새 파일 전체의
정적 검증 통과를 보장하지 못했다. 초기 verification.json의 관련 exit 필드를
그보다 강한 검증 범위로 해석해서는 안 된다.

최초 GitHub CI에서 Check format은 통과했지만 Check static analysis가 다음
오류로 실패했다.

~~~text
test/unit_test/utils/test_circle_harness.py:104
MYPY arg-type: Argument after ** must be a mapping, not "object"
~~~

서로 다른 값 타입의 dict를 풀어 전달하던 두 사례를 (shape, dtype, error)
튜플과 명시적 인자로 바꾸었다. 잘못된 shape/dtype의 실제 Circle 검증 및
inference 거부 assertion은 유지했다. 이후 tracked 파일 전체를 대상으로
UFMT/Pylint/MYPY를 실행했고, 직접 파일을 지정한 cache-free mypy도 확인했다.

mypy-before.log는 로컬의 직접 진단이다. 이 파일에는 해당 arg-type 1개 외에
imported mx 모듈의 설정 영향 진단 4개도 있다. GitHub CI 전체 로그와 같은
것이 아니다. mypy-after-target.log의 follow-imports=silent는 가져온 모듈의
진단을 숨기되 대상 파일은 검사하는 옵션이다. 전체 CI-equivalent 검사는
별도의 static.log에 보관했으며 소스에 type-ignore를 추가하지 않았다.

CI 근거:

- 최초 실행: https://github.com/Samsung/TICO/actions/runs/42936224986
- 타입 수정 후 실행: https://github.com/Samsung/TICO/actions/runs/42940769480
- 최초 실패 job: https://github.com/Samsung/TICO/actions/runs/42936224986/job/130246947303
- 수정 후 정적 검사 job: https://github.com/Samsung/TICO/actions/runs/42940769480/job/130261840785

2026-11-11 02:23:05 UTC 관찰에서 수정 후 CI의 9개 job 모두 success였다.
ci/fixed-full.log에는 5,013개 실행, skip 49개, OK 및 checkout 밖 wheel smoke
성공이 기록되어 있다. 이 실행은 1793.892초였지만 다른 환경/checkout의 실행 시간과
성능 비교하지 않는다. ci/fixed-jobs.json은 더 이른 02:09:23 UTC의 진행 중 snapshot이며,
최종 완료 근거는 ci/fixed-jobs-final.json이다.

## 성능 측정 조건과 구간

공식 반복은 각 대상/checkout별 새 프로세스 3회다. 같은 로컬 가상 환경을 사용했고
CPU thread 환경변수는 OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, runtime은 reference였다.
internal opt-in은 켜지 않았고 기존 fixture seed 정책과 dynamic alternative 입력의
지역 generator를 유지했다. 전역 seed 강제나 모델/artifact 공유 cache는 추가하지 않았다.
OS/bytecode cache는 활성화된 warm 상태를 사용했고 매번 지우지 않았다.

대표 버전: Python 3.12.14, Torch 2.13.0+cpu, TorchVision 0.28.0+cpu,
NumPy 2.5.4, circle-schema 0.10.1, Transformers 5.18.0, Datasets 5.0.1.
metadata/summary.json의 tico 0.1.0은 설치 distribution metadata 값이다.
실제 코드 정체성은 commit 기준과 각 JSON의 tico import 경로로 구분했다.
매 sample마다 전체 pip freeze나 CPU/OS fingerprint를 저장한 것은 아니다.
sample별 UTC 시작 시각도 계측하지 않았으므로 파일 수정 시각을 실행 시각으로
대신 해석하지 않는다.

| 필드 | 정확한 범위 |
|---|---|
| collection | measure.py에서 loader.discover('test') 호출 직전부터 반환까지. unittest 등의 선행 stdlib import는 제외하고 발견된 모듈 import/generated 구성은 포함 |
| process_wall | batch.py의 child 시작 준비 전부터 subprocess 종료 및 로그 close 후까지. 프로세스 생성/import/종료와 작은 로그 open/close 비용 포함 |
| ccex_wall | ccexbatch.py에서 subprocess.run('./ccex ...') 직전부터 반환까지. ccex shell/부모 Python/실제 unittest 프로세스 포함 |
| inside_wall | measure.py의 첫 perf_counter부터 결과 JSON을 구성하는 시점까지. interpreter startup, JSON write, process shutdown은 제외 |
| suite | TextTestRunner 생성/run 직전부터 반환까지. class/module fixture와 결과 출력 포함 |
| individual duration | native unittest의 test setup/body/teardown/cleanup. class/module fixture와 discovery 제외 |
| preparation | 별도 counter 진단에서 등록된 generated 모델의 최상위 constructor 및 get_example_inputs 본문 누계 |

새 --durations worker의 Collection 출력은 unittest import 전부터 discovery 종료까지다.
따라서 수동 benchmark의 collection과 측정 경계가 조금 다르다.
전체 wall에서 개별 duration의 합을 뺀 값을 discovery라고 부르지 않았다.

--count 진단은 Torch/helper를 미리 import하고 클래스에 counter/timer를 붙인다.
이 진단의 collection/process 시간은 공식 latency 비교에 사용하지 않았다.
준비 호출 수는 모든 nn.Module child가 아니라 builder에 등록된 최상위 generated
fixture의 호출이다. 최종 measure.py 보존본에는 진단 중 추가된 필드도 들어 있으므로
초기 JSON에 preparation/durations 필드가 없는 것은 원자료 그대로다.

### 대표 명령과 관찰

~~~bash
# A: 1개
./ccex test -k '*test.modules.op.add.SimpleAdd'
# B: addmm 6개를 포함한 17개. 기존 unittest matching 의미
./ccex test -k test.modules.op.add
# C: 1개
./ccex test -k test.modules.net.mlp.
# D: conversion-only 1개
./ccex test -k test.modules.op.linear.FQLinearWithFp32Bias
~~~

아래는 초 단위 중앙값이다. 반복별 값과 최소/최대는 summary.json 및 원자료에 있다.
전수 수집의 wall은 discovery-only child이고 나머지는 실제 ccex 명령이다.

| 대상 | collection 전 → 후 | 전체 wall 전 → 후 | suite 전 → 후 |
|---|---|---|---|
| 전수 수집 | 6.125 → 6.302 | 7.491 → 7.680 | 실행하지 않음 |
| A | 5.977 → 5.974 | 7.701 → 7.860 | 0.650 → 0.665 |
| B | 5.870 → 6.307 | 12.197 → 13.069 | 5.308 → 5.547 |
| C | 5.792 → 6.174 | 8.461 → 8.130 | 0.679 → 0.692 |
| D | 5.951 → 5.755 | 7.512 → 7.970 | 0.724 → 0.707 |

별도 counter 진단의 A 개별 duration 중앙값은 0.773 → 0.877초였다(각 3회).
전체 wall 또는 개별 duration 개선을 입증하지 못했다.
부분 실행을 전체 suite 개선율로 확대하지 않으며, 전체 suite의 전후 wall 비교는 없다.

| 대상 | constructor / 입력 / probe: 전 → 후 |
|---|---|
| discovery만 | 411 / 411 / 0 → 0 / 0 / 0 |
| A·C·D 각각 전체 실행 | 411 / 412 / 1 → 1 / 1 / 1 |
| B 전체 실행 | 411 / 428 / 17 → 17 / 17 / 17 |

계측된 discovery 준비 누계는 0.188초 [0.184, 0.204] → 0이었다.
이는 instrumentation이 있는 진단값이며 전체 명령이 그만큼 빨라졌다는 뜻이 아니다.
선택된 모델 준비는 실행 시점으로 이동했고, 제외된 모델 준비와 중복 입력 생성은
제거되었다. static zero-input probe는 다른 입력 사례/quantization 실행 모드의
검증이므로 제거하지 않았다. 실제 Circle 회귀 결과는 focused 및 harness 로그에 있다.

로컬 전후 discovery는 기존 4,996개 ID가 각 1회, 순서/선언된 skip 48개와 사유가
동일하고 신규 harness 회귀 22개만 추가되었다. 중복·누락·loader error·_FailedTest는 0이다.
실행 중 SkipTest를 포함한 전체 suite skip 49개와 선언된 skip 48개는 다른 집계다.
무매칭 ccex 종료는 당시 Python 3.12에서 이미 nonzero였으며 버그를 고쳤다고
주장하지 않는다. measure.py 자체는 계측용 TestResult로 0건을 성공으로 볼 수 있어
CLI 무매칭 종료 계약의 증거로 사용하지 않는다.

## 로컬과 CI 기록을 합산하지 않는 이유

최초 CI 전체 실행 로그에는 5,013개, skip 49개, OK 및 별도의 checkout 밖 wheel
smoke 성공이 기록되어 있다. 로컬 전체 실행은 5,018개, skip 49개다.
metadata/ci-local-header-comparison.json은 timestamp/ANSI를 정리한 verbose 헤더만
대조한다. nested synthetic Skipped.runTest 출력은 main-suite 집계에서 제외한다.

- full ID 일치: 4,094개.
- 파일명·클래스·메서드는 같고 상위 namespace가 다른 항목: 916개.
- 남은 이름 차이는 로컬 Gemma4 smoke 8개와 CI 3개다. causal/conditional 테스트가
  각각 3→1, vision patch 테스트가 2→1로 나타난다.

헤더 비교는 테스트 본문의 검증 내용이 동등하다는 증명이 아니다.
로컬 전후 ID 보존 주장을 서로 다른 CI checkout에 그대로 적용하지 않는다.
CI는 Python 3.12.16과 PR merge checkout을 사용하고 wheel 설치를 수행했다.
일반 test 단계의 tico.__file__은 기록하지 않았고 설치 metadata와 출력 버전도
다르므로 wheel 코드만 import했다고 단정하지 않는다.
CI의 1870.283초와 로컬의 2288.827초를 비교해 개선율을 계산하지 않는다.
수정 후 CI는 fixed-jobs-final.json의 관찰에서 전체 성공을 확인했다.

## 재검산과 재현

새 test 실행 없이 보관 자료만 확인:

~~~bash
python docs/archive/test-cleanup-stage5/verify_records.py
cd docs/archive/test-cleanup-stage5
sha256sum -c SHA256SUMS
~~~

실제 성능 재현은 별도 출력 디렉터리와 같은 Python 3.12/Torch/도구 환경에서 한다.
[개발 환경 가이드](../../development.md)를 따른다. 다음 명령은 실제 테스트와 변환을
실행하므로 기록 열람만 원하면 실행하지 않는다. 원자료를 덮어쓰지 않는다.

~~~bash
task_records_dir="$PWD/docs/archive/test-cleanup-stage5"
task_replay_dir="$(cd .. && pwd)/tico-stage5-replay"
mkdir -p "$task_replay_dir"
git worktree add --detach "$task_replay_dir/baseline" b7ed542e9a717580456b63bcac000c16cea08ac8
git worktree add --detach "$task_replay_dir/after" 053866cfd58245b646be433a7392e5d79c95aaaf
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python - "$task_records_dir" "$task_replay_dir" <<'PY'
import sys
from pathlib import Path
records, output = map(Path, sys.argv[1:])
for source in (records / "scripts").glob("*.py.txt"):
    text = source.read_text()
    text = text.replace("'<STAGE5_RECORDS>'", repr(str(output)))
    text = text.replace("'<CHECKOUT>'", repr(str(output / "after")))
    (output / source.name.removesuffix(".txt")).write_text(text)
PY
for task_phase in baseline after; do
  (
    cd "$task_replay_dir/$task_phase"
    python "$task_replay_dir/measure.py" --out "$task_replay_dir/warmup-$task_phase.json"
  )
done
for task_phase in baseline after; do
  (
    cd "$task_replay_dir/$task_phase"
    python "$task_replay_dir/batch.py" "$task_phase"
    python "$task_replay_dir/ccexbatch.py" "$task_phase"
  )
done
python "$task_replay_dir/diagnose.py"
python "$task_replay_dir/summarize.py"
~~~

summarize.py 원본은 준비/개별 duration의 별도 진단 집계를 추가하기 전 파일이다.
보관 summary.json의 preparation_diagnostic과 individual_diagnostic은 대응하는
반복별 JSON을 이용해 이후 추가한 값이다. verify_records.py가 이 값도 재검산한다.
기존 scripts는 당시 사용한 일회성 계측 자료이며 production runner/API가 아니다.
