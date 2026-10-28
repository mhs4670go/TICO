# Samsung/TICO PR #1097 CPU 머지 전 검증

## 1. 결론

**필수 Circle 수치 검증이 통과하지 않아 머지 전 판정은 불완전함(INCOMPLETE).** 기본 suite·frozen lifecycle·실제 FP/Quant 평가·축소 artifact benchmark는 통과했지만, 지원 wrapper의 일부 Circle export와 CPU 통합 드라이버의 엄격한 Torch/reference parity는 실패했다. 이 결과를 전체 PASS로 판정하지 않았다.

확인된 실패는 최신 base의 기존 문제 또는 smoke fixture/harness 문제로 분리되었으며, PR의 frozen state 변경에 의한 새로운 회귀를 입증한 사례는 없다. 저장 Circle strict 수치 실패17개와 추가 assistant strict 차이는 base에서도 동일 출력·오차로 재현됐다. CPU staged/connected 실패도 독립 base fixture로 재현했으며 상세 비교는 아래에 기록했다. 제품 코드는 수정하지 않았고 commit·merge·push·PR 댓글을 수행하지 않았다. 과거 작성자 테스트 숫자는 이번 결과에 합산하지 않았다.

핵심 frozen 계약은 실제 negative/round-trip/export 테스트로 통과했다. 반면 **파일 저장, 구조 검증, reference 실행, Torch 수치 parity는 각각 다른 결과**다. reference는 CPU 의미 검증이며 NPU integer 실행을 뜻하지 않는다.

## 2. 검증 대상과 환경

| 항목 | 실제 값 |
|---|---|
| PR | https://github.com/Samsung/TICO/pull/1097 |
| PR head / 실제 검증 commit | `9f631d6651ce3780e3a8e054ab02910b9bd40fdc` |
| 최신 main / merge-base | `261bda28ec55fdac42f59346c950326e4065f8fa` |
| 실제 검증 tree | `c81b5f254023b21bb76f1f6d04405b5c7d3f54b2` |
| 최신 base 통합 | base가 head의 ancestor임을 확인. head 자체를 detached worktree에서 검증; 로컬 merge 없음 |
| GitHub merge ref 확인 | `6f3016abb6f190d01a28b047fe1cef4496d5a94a`; 부모는 위 base/head와 일치. 테스트에는 사용하지 않음 |
| 원본 checkout | `${BASELINE_REPO}`, 시작 HEAD=base, 시작 working tree clean |
| 검증 checkout | `${CANDIDATE_REPO}`, 테스트·base 비교마다 실제 `tico.__file__` 확인 |
| VALIDATION_ROOT | `${VALIDATION_ROOT}` |
| CPU/RAM/disk | CPU 5개 노출, cgroup quota 4 core; RAM 약17GiB 노출/cgroup16GiB; 작업 시작 disk 약30GiB 여유; swap 없음 |
| 실행 정책 | CPU 명시, FP32/tiny, seed123(wrapper)/1073(recipe/eval)/2026(artifact); OMP/MKL/OPENBLAS=1. 무거운 export·driver·benchmark 순차. lint는 저장소 기본 병렬 설정 |
| Python/OS | Python3.12.14, Linux6.18.55 x86_64/glibc2.41 |
| 기존 핵심 의존성 | Torch2.13.0+cpu, TorchVision0.28.0+cpu, Transformers5.18.0, datasets5.0.1, circle-schema0.10.1, flatbuffers25.12.19 |
| 추가 평가 의존성 | lm-eval[hf]0.4.13, accelerate1.15.0, peft0.21.2. 기존 package 버전 constraint를 유지한 선택적 설치; 초기/최종 pip check PASS. 전체 venv 사전 목록은 `packages-venv-before-lm-eval.txt`, 별도 system Python 목록은 `packages-system-initial.txt`로 구분 |
| runtime | 저장 Circle bytes를 읽는 built-in `reference`; quantized wrapper artifact는 `FAKE_QUANTIZE` 명시. ONE/ONERT 미사용 |
| 네트워크 | restricted proxy; Hugging Face CONNECT403. 외부 데이터 요청은 1회 bounded probe 후 중단 |

관련 AGENTS(루트, quantization, test, circle, passes), development/Torch policy, CI·dependency·quantization/recipe/evaluation/export README를 확인했다. 최근30일 base history 109 commits와 PR149개 변경 파일을 조사해 실행 경계에 연결했다. [coverage map](coverage-review.md), [원본 diff](pr.diff), [30일 history](base-last30d.log), [source 문서 snapshot](source-context/sha256.json), [revision 기록](revisions.json), [환경 로그](logs/resource-snapshot.log)를 참조한다.

현재 CI는 default Torch2.13 full suite, oldest2.11 blocking smoke, candidate2.14 nonblocking smoke를 정책 모듈에서 resolve한다. 이번 실행은 준비된2.13 CPU 환경을 재사용했다. 다른 Torch family까지 검증했다고 주장하지 않는다.

## 3. 실행 결과

명령의 `R`은 VALIDATION_ROOT, `C`는 검증 checkout이다. 아래는 영역별 요약이며 **실행한 원문 명령·cwd·환경·timeout·원래 exit code·초 단위 시간은 [COMMANDS.sh](COMMANDS.sh), [commands.json](commands.json), `runs/*.json`에 모두 있다.** 도움말·import·수집 성공은 실행 coverage에 포함하지 않았다. 주요 원본 로그는 `logs/`, 드라이버 child 로그는 `cpu-validation/`에 있다.

| 영역 / 실제 명령 | 모델·입력 / 수집·실행 결과 | 상태·wall time | 확인한 계약 / 증거 |
|---|---|---|---|
| `python -m pip check` (초기/최종) | dependency 충돌0 | PASS | [초기](logs/pip-check.log), [최종](logs/pip-check-final.log), [최종 버전](logs/packages-final.log) |
| `./ccex test -k frozen` | 61개 처리/61 PASS/0 fail/0 skip | PASS,68.438s | 실제 freeze/readiness/round-trip/Circle 포함; [로그](logs/focused-frozen.log) |
| `./ccex test` | 4,760개 수집·처리;4,602 PASS,158 skip,0 fail/error/import error | PASS,995.207s (unittest987.679s) | [전체 로그](logs/full-suite.log), [모든 test ID/집계](full-suite-summary.json) |
| `./ccex format --no-apply-patches` | UFMT·Pylint·mypy 전체; patch 적용 없음 | PASS,210.185s | `No lint issues`; 최초 lintrunner init-cache warning은 exit0과 구분. [로그](logs/format.log) |
| `python -m tico.quantization.examples.inspector --mode wrapper-smoke --list-cases` 후 `--config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke --case all --device cpu --export none --strict --no-plot --output-dir R/wrapper-smoke` | 현재 registry50개 모두 실제 prepare/cal3/convert/forward; 모든 family tiny/FP32 | PASS50/50,10.674s | shape·finite·case별 기존 numerical threshold; [원본](logs/wrapper-numerical-all.log), [집계](wrapper-smoke/summary.json) |
| `python R/wrapper-harness/export_case.py ... --case CASE --export circle --strict --no-plot --output-dir R/wrapper-circle/CASE --report-json R/wrapper-circle/CASE.json` (지원48개) | 원래 inspector.main과 실제 tico.convert 호출;48 attempted, 원본 strict harness25 PASS/23 FAIL; 의도적 미지원2개 별도 제외 | FAIL, loop452.337s | 원본 저장42/구조42 PASS; 원본 runtime41 PASS+1 binding-harness 오류. raw strict parity25 PASS/16 FAIL/나머지7 미완료. [전체 matrix](wrapper-harness/WRAPPER_REPORT.md), [원본 단계별 JSON](wrapper-harness/export-summary.json) |
| 동일 export의 보완 replay | saved42개 모두 실제 reference 실행(딕셔너리 binder 보완 포함); 동일 encoded ABI 입력으로 strict 비교25 PASS/17 FAIL | FAIL 유지 | tolerance1e-5/1e-4 그대로; 원본 실패 보존. [encoded 비교](wrapper-harness/encoded-revision-comparison.json) |
| `python -m tico.quantization.recipes.debug.cpu_validation --stage all --output R/cpu-validation` | 실제 HF Llama90,432/Qwen150,272 params, cal2×8+cache decode; tests110+7+12+6=135 모두 PASS, fixtures PASS | 전체 FAIL/exit1,227.768s | Llama16/Qwen16/Qwen32 및 connected16/32 5개 child FAIL. [stage/exit/시간](cpu-validation/summary.json), [원본](logs/cpu-validation.log) |
| 위 CPU driver Circle 상세 | before/after 각각25 stages, 합50개 저장·구조·runtime 성공. Torch stage parity14/25 PASS,11/25 FAIL(전후 동일). eager48token+decode4 | FAIL(strict parity) | 전후391개 input/Torch/Circle 배열 비교 모두 정확히0 차이; frozen/eager state 보존. [상세 audit](cpu-candidate-audit.json) |
| 위 saved Qwen connected 실행 | capacity16/32 각각22 calls 실제 연결; 12→16,28→32 cache. 총44 calls,86 output 비교 중74 PASS/12 strict FAIL | FAIL(strict parity) | saved stage를 실제 소비; 상태 보존. `qwen3_vl-{16,32}-connected` 로그·JSON; 호출 성공을 numerical PASS로 세지 않음 |
| `python R/evaluation/prepare.py` | 실제 HF Llama90,432params, seed1073; cal2×8+decode2→convert→checkpoint save/load | PASS,8.201s | frozen/weight exact roundtrip, 평가 데이터와 calibration 역할 분리; [생성/해시](evaluation/fixture-evidence.json) |
| `python R/evaluation/invoke_cli.py --label {fp,quant}-ppl-final -- --config R/evaluation/config.yaml [--checkpoint R/evaluation/quantized_model.pt] --device cpu --tasks ppl --output-dir ... --no-progress` | 실제 public evaluate CLI; local JSON4행/28 input tokens; 각각3 forwards/27 scoring tokens; window16/stride8/context64/batch1 | FP PASS6.201s;Quant PASS7.332s | PPL127.899391/128.043839, finite; Quant checkpoint 실제 로드. 같은 입력 hash·seed·설정, 평가 전후 frozen 동일. [보고서](evaluation/REPORT.md) |
| 위 tracing CLI의 `--tasks lm_eval` | 실제 helper/HFLM/evaluator, synthetic HellaSwag 형식4행×4 choices; FP/Quant 각각16 likelihood forwards/32 scoring tokens | FP PASS13.257s;Quant PASS21.621s | acc/acc_norm 각.25; 실제 sample 평균==summary, `acc,none`/`acc_norm,none` 정확히 읽음. 모델 품질 판정에 사용하지 않음 |
| `python -m test.performance.benchmark_conversion_memory --sizes-mib 1 --tables 1 --width 16 --threads 1 --repeat 1 --optimization O1 --input-kind {module,pt2} --workdir R/benchmark-KIND --keep-outputs --json` | 1MiB constant, bytes/file 각2경로×2 input-kind=4 conversions; 각각3 probes=총12 | PASS,8.588s/10.640s | 모두 구조 검증 및 reference parity PASS,max_abs0; buffer/metadata/output hash 동일, output1,049,080bytes. [module](logs/conversion-module.log), [pt2](logs/conversion-pt2.log) |
| `python -m test.performance.benchmark_circle_optimizer R/wrapper-circle/llama_attention_decode/llama_attention_decode.q.circle --repeat 1 --warmup 0 --json` | numerical PASS였던 실제 tiny artifact; until-no-change/restart 각각12 pass executions,0 changes | PASS,1.468s | 구조 검증·두 결과 bytes/hash 동일. 입력은 이미 O1이라 non-empty 최적화 성능 검증은 아님. [로그](logs/circle-optimizer.log) |
| latest base 대표 tests (동일3 modules) | public interface/GPTQ handoff/실제 Llama-Qwen-Gemma PPL,각 revision에서 독립 모델 준비;13/13 PASS/skip0 | PASS,8.132s | import path 확인, checkpoint 공유 없음. [base 요약](baseline-selective-summary.json), [동일 PPL 비교](baseline-ppl-comparison.json) |
| latest base wrapper26개 + 동등 fixture 추가 비교 | 실패 사례 우선 및 positive representative; 상세 상태와 hash는 아래 문제·wrapper 보고서 | 비교 완료 결과 참조 | 다른 checkout의 editable import 혼입 방지. 초기 HF signature fixture 오류는 더 뒤쪽 export 실패의 증거로 사용하지 않음 |
| 독립 base CPU `_fixtures`→`export_regression isolation`→`_connected` | Llama16/Qwen16/Qwen32 및 Qwen connected16/32, 각 revision 자체 checkpoint | strict FAIL5개 동일 재현 | weights·활성 grid·782개 numerical 배열 hash 동일; bytes46/50 동일, 나머지 contracts/수치 동일. [비교](baseline-cpu-comparison/REPORT.md), [명령](baseline-cpu-comparison/commands.json) |
| assistant valid-capacity16 추가 export (`assistant_capacity16.py`) | 실제 같은 kv10, capacity만8→16; head/base 각각 별도 PTQ | inspector/save/structure/runtime/frozen PASS; strict parity FAIL,16.883s/15.668s | head/base input/Torch/Circle 출력 hash 동일, max_error0.000246406. 원본 case failure와 별도. [요약](gemma-export-review/summary.json) |
| `curl -I --max-time 15 https://huggingface.co/datasets/Rowan/hellaswag` | 외부 실제 benchmark dataset | BLOCKED,exit56 | CONNECT403. [원본](logs/external-benchmark-access.log). 나머지 로컬 검증 계속 |

기본 suite family audit는 다음과 같다. family 집계는 wrapper 하위 test ID 기준이며 표 전체 합계와 별도로 중복 합산하지 않는다.

| family | 수집 | PASS | skip | fail/error |
|---|---:|---:|---:|---:|
| Llama |96|93|3|0|
| Qwen-VL |209|197|12|0|
| Gemma4 |422|354|68|0|
| Gemma4 assistant |52|52|0|0|
| fairseq |50|50|0|0|

158 skip의 원문 사유는 [전체 결과 JSON](full-suite-summary.json)에 있다. 내부 opt-in125개, opt-in 실제 checkpoint2개, CUDA1개, >2GiB3개, ONE1개, 기존 미사용 Conv3d11개, 미지원 operator/shape 등과 선택 plotting1개를 포함한다. 모델 family 전체 import 누락이나 전부 skip인 결과는 없다. 기본 discovery는 unit/quantization 및 생성 op/net E2E를 포함하고 full-size model loader는 `-m` opt-in이다. fairseq50개는 실제 작은 wrapper/incremental-cache tests이며 mock 기반 export routing 성공을 실제 fairseq Circle export로 집계하지 않았다.

Frozen readiness, atomic failure/retry, locked Percentile/import grid 보존, qparam dtype/shape/axis/range/provenance, non-finite/empty/zero-range/per-channel mismatch, corrupted/legacy checkpoint 거부, conditional capability, GPTQ reuse on/off/folding, AdaRound·block reconstruction의 gradient/finalization/rollback는 [coverage map의 실제 test ID](coverage-review.md)를 통해 이번 full suite에 연결된다. CPU BF16 per-channel 제한이나 legacy 거부를 cast/migration으로 우회하지 않았다.

## 4. 발견한 문제와 baseline 비교

### A. 저장 Circle의 엄격한 Torch parity 불일치 — 기존 문제 확인 / 일부 원인 분석

원본 wrapper strict probe16건, 올바른 dict binding을 적용한 Gemma model까지 포함한 same-encoded-input probe17건에서 불일치가 있다. source code·assertion·tolerance를 바꾸지 않았고, 원래 float 입력 비교와 integer ABI에 맞춘 입력 비교를 모두 보존했다. baseline에서도 **17개 수치 실패 전부**의 actual input/Torch/Circle hash가 정확히 일치한다. Qwen vision의 기존 HF positional fixture 문제는 keyword로만 동등하게 연결한 별도 baseline 실행과 구분한다.

최소 재현은 [REPRODUCE.md](REPRODUCE.md)의 `export_case.py --case llama_attention_prefill` 명령이다. artifact-checks.json은 inspector exit0/저장·구조·실행 성공과 parity FAIL을 나눠 기록한다. 예를 들어 Llama K max error0.0137165, tied embedding0.084186, Gemma attention0.478296 등이 있다. 개별 원본 traceback/배열/qparam·input·weight hash는 `wrapper-circle/CASE/` 및 per-case logs에 있다.

[RoPE 진단](parity-analysis.md)은 eager 재구성 output hash가 원본과 정확히 일치함을 확인한 뒤, split별 eager grid와 serialized grid가 달라지고 명시적 재양자화 경계가 없는 지점을 찾았다. 단순 ties-even/away rounding 차이만으로는 재현 차이가 사라지지 않았다. 모든 case 원인을 이 하나로 단정하지 않는다. 수정 제안은 folding/propagation/serialization/runtime 경계에서 서로 다른 grid의 explicit FQ 의미를 보존하고, 동일 grid라는 증거가 있을 때만 split/concat을 양자화 관점에서 값 보존으로 취급하는 것이다. tolerance 완화로 처리하지 않는다.

CPU 드라이버도 자체 `--require-torch-parity` 기준에서 Llama16의 decoder4 stages, Qwen16의 embedding/prefill3 stages, Qwen32의4 stages 및 연결 replay에서 실패한다. before/after391배열의 차이는 모두0이고 frozen state가 보존돼 **export/eager 이후 state drift와 구분**된다. candidate isolation 실행 자체는 base 비교가 아니다. 추가로 각 revision에서 독립적으로 PTQ한 fixture를 사용해 base Llama16/Qwen16/Qwen32와 connected16/32를 같은 조건으로 실행했다. 동일 stage에서 같은 strict 실패가 재현됐고, before/after에서 총782개 actual input/Torch/Circle 배열 hash와 Circle contracts가 일치했다. Circle bytes는46/50개가 동일하며, Qwen vision4개는 bytes가 다르지만 contracts와 모든 수치 배열은 동일했다. [독립 base 비교](baseline-cpu-comparison/REPORT.md), [전체 hash 비교](baseline-cpu-comparison/comparison.json), [명령](baseline-cpu-comparison/commands.json)을 참조한다. observer serialization metadata는 신규 frozen schema와 구버전 형식을 구분해 정규화한 상세 비교를 보존했다.

### B. supports_circle_export=True인 generic smoke의 입력/용량 구성 오류

기본 config에서6개 case는 artifact를 만들지 못했다.

- `qwen3_vl_text_model`: fixed full-model export에 필요한 `prepared_static_export(model,max_seq_len)` context 부재.
- `qwen3_vl_model`, `qwen3_vl_for_conditional_generation`: compilation용 explicit position_ids 등의 static 입력 준비 부재. 초기 base에는 HF mm_token_type_ids fixture 오류도 있어 동등 fixture 비교를 추가했다.
- `gemma4_for_causal_lm`, `gemma4_for_conditional_generation`: export adapter가 없어서 HF token forward로 돌아가는데 positional tuple은 precomputed embedding ABI다. float embedding이 input_ids로 들어가고 attention-mask dict가 pixel_values_videos로 들어간다. head의 vision capability guard는 이 잘못된 호출을 더 일찍 거부한 것으로, 정상 vision 경로의 quantization 실패로 판정하지 않았다. 실제 prepare/cal3/convert와 정상 token forward는 통과했다.
- `gemma4_assistant_core`: kv_len10을 sliding capacity8 adapter에 넣는 기본 fixture 문제. 실제 unit Circle 경로는 유효한 capacity16을 사용한다. 작은 capacity16 추가 export는 head/base 모두 inspector exit0, 저장·구조·reference·frozen 보존 PASS였다. 엄격한 global tolerance에서 두 출력의 max error는0.0000198036/0.000246406으로 head/base가 동일해 strict parity FAIL로 남겼다. 정상 output-grid 단위 허용오차를 쓰는 기존 unit 테스트는 full suite에서 통과했지만, 추가 strict probe를 그 기준으로 바꾸지 않았다. [추가 결과](gemma-export-review/assistant-complete.json)는 원본 capacity8 failure를 대체하지 않는다.

재현: 원래 inspector CLI에 각 `--case CASE --export circle --strict --no-plot`을 준다. 같은 seed123/FP32/tiny/cal3을 사용한다. [Gemma/assistant 상세 원인·source 위치](gemma-export-review/REPORT.md), [wrapper별 baseline/원본 명령](wrapper-harness/WRAPPER_REPORT.md)을 참조한다. 수정 제안은 generic case를 정확한 tensor-only/static export adapter와 입력에 연결하고, 올바른 capacity 및 필요한 calibration 경로를 선언하는 것이다. 모호한 vision calibration 추가나 assertion 삭제로 우회하지 않는다.

`gemma4_text_model`, `gemma4_assistant_sparse_head` 두 일반 export 미지원 case는 registry의 명시 사유에 따라 의도적 SKIPPED이며 위6개 실패와 다르다.

### C. 검증 harness의 초기 오류 — 제품 회귀에서 제외, 원본 보존

- FP 모델에도 frozen-only `_state_contract`를 적용한 최초 PPL/MC tracing2건은 평가 전에 실패했다. FP는 tensor state_dict digest, Quant는 원래 frozen contract로 각각 검사한 final4 runs는 실제 계산을 통과했다. [수정 이력/최소 재현](evaluation/HARNESS_CORRECTIONS.md).
- Gemma model의 dict pytree를 `ModelInputSpec.bind`로 직접 넘겨 원래 probe runtime 단계가 실패했다. serialized input 이름과 tensor leaves를 확인한 보완 binder로 실제 reference 실행했으며, 이후 numerical FAIL은 그대로 남는다.
- Gemma vision의 전체 topology hash가 export profile materialization으로 달라졌다. [정밀 diff](gemma4-frozen-topology-analysis.md)는 기존 weight/observer/grid/mode 변경0·삭제0, 독립 profile observer6개/owner1개 추가만 확인했다. profile forward 이후 추가 변화0, validate_frozen_model PASS였다. 기존 frozen grid mutation이라는 잘못된 판정은 제외하고 원본 raw hash FAIL은 보존했다.
- assistant 추가 probe 최초2건은 sibling observe 모듈 import 경로를 빠뜨린 validation wrapper 자체 오류로 모델 실행 전에 실패했다. 초기 script/log를 보존하고 경로를 고친 r2의 실제 head/base 결과를 사용했다.
- lm-eval plain 설치에 HF extra(accelerate)가 없어 선택적 extra를 추가했다. 기존 주요 package version을 바꾸지 않았고 최종 pip check PASS다.

외부 dataset403은 환경 BLOCKED이며 제품 실패나 정확도 회귀가 아니다. timeout/OOM은 이번 실행에서 발생하지 않았다.

## 5. 남은 검증 범위

- **INCOMPLETE/FAIL:** 위 generic export6개와 엄격한 Circle numerical 실패. 전용 staged 파일이 저장·실행됐다는 사실만으로 이 수치 검증을 통과로 바꾸지 않았다.
- **BLOCKED:** 외부 실제 HellaSwag/WikiText 등 다운로드. 실제 evaluator의 local4-example 실행은 완료했지만 외부 데이터·pretrained 품질을 검증하지 않았다.
- **SKIPPED:** full-size model loader/전체 benchmark dataset, `./ccex test -p`의 Llama1B/3B threshold benchmark, >2GiB 실제 serialization, 선택 ONE/ONERT, CUDA/NPU. 기본 suite의 tiny synthetic 모델과 wrapper smoke로 변경 경계를 검증했다.
- Llama lm-eval public CLI에 없는 sample limit을 만들어 쓰지 않았다. MC harness가 실제 evaluator에 TaskManager/local data/limit4/seed/cache-off를 전달한 경계는 [평가 보고서](evaluation/REPORT.md)에 명시했다. PPL은 실제 datasets JSON loader이며 데이터량 자체가4행이다.
- 실제 public VLM benchmark CLI 전체 routing 및 pretrained 모든 family 품질은 미검증. unit routing mocks·프로토콜 JSON 생성·도움말 성공을 실제 scoring으로 세지 않았다.
- CPU reference FAKE_QUANTIZE는 integer NPU bit-exact 동작·GPU/NPU 성능 검증이 아니다. 1회 tiny 측정시간이나 optimizer 출력 speedup 비율로 성능 향상/회귀를 주장하지 않는다. converter 측정1.20~1.39s/peak RSS 약298~307MiB는 관찰값이며 준비·검증시간과 구분된다.

## 6. 재실행 방법과 산출물

[REPRODUCE.md](REPRODUCE.md)에 revision 고정, 환경 재사용/새 설치, 동일 import 경로 확인, 새 출력 디렉터리, 기본 suite·wrapper·driver·소량 evaluation·benchmark 명령을 정리했다. [COMMANDS.sh](COMMANDS.sh)는 원래 실제 명령 목록이며 이미 존재하는 CPU driver output 경로를 그대로 재사용하면 의도대로 실패하므로 새 경로로 실행한다.

- 기계 판독 결과: [VALIDATION_RESULTS.json](VALIDATION_RESULTS.json).
- 실행 기록/원본 stdout+stderr: `runs/`, `logs/`, `cpu-validation/` 및 baseline 하위 디렉터리.
- 작은 입력/config/harness: `configs/wrapper_smoke.yaml`, `wrapper-harness/`, `evaluation/`, `gemma-export-review/`, `run.py`, 각 진단 script.
- 체크포인트는 seed/model/calibration 생성방법과 SHA256을 기록했으며 소스에 commit하지 않았다. 생성 artifact의 경로·크기·hash는 최종 `ORIGINAL_ARTIFACT_MANIFEST.json`에 있다.
- source docs/CI/dependency snapshot, 초기 working tree, revision/tree 및 최종 source 상태를 함께 보존했다.

최종 source check: head/base 모두 시작 SHA와 동일하고 working tree clean, tracked diff0. [최종 환경·source 기록](final-source-environment.json). 최종 보고서 작성일: 2026-10-28 UTC.
