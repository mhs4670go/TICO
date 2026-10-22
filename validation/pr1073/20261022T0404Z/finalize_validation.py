"""Finalize the Korean report and explicit model coverage matrix."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
report = r'''# TICO PR #1073 실행 검증 결과

**최종 판정: FAIL. 실행 가능한 검증을 수행했으며, 미완료 검사는 INCOMPLETE, 자산 부족 검사는 BLOCKED로 남겼다.** Tiny 모델에서 eager 길이 독립성·상태 보존·Llama HF RoPE 일치를 확인했다. 완료된 Circle 비교에서는 candidate의 eager 전후 artifact 변경을 발견하지 못했다. 그러나 unit/static 검사, 실제 Quant generation 및 Torch–Circle strict parity 실패가 남아 전체 정상 동작이나 pre-PR equivalence를 승인할 수 없다.

검증 HEAD: **`6b2f6a0511458d6a750e2a260d706080a78149dd`**. `mhs4670go/TICO`의 `ea` 및 Samsung/TICO PR1073 원격 HEAD와 일치했다. Baseline: **`bb09ca852b0ca1b1a6271cfe1e10c162104c997e`**, detached worktree `/workspace/TICO-export-baseline`. `61f3a05`/최신 main으로 대체하지 않았다.

결과 root는 `/workspace/TICO/out/pr1073_validation/20261022T0404Z`다. 제품·baseline 소스와 기존 테스트를 수정하지 않았다. commit/push/merge/PR metadata 변경 없음. 검증 fixture와 최소 runner 복사본만 결과 root에 추가했다.

## 모델별 결과

원본 strict 판정은 **atol=1e-5, rtol=1e-4, --require-torch-parity**를 유지한다. 전후 exact는 별도 성분 일치이며 strict FAIL을 PASS로 바꾸지 않는다. Unit/mock, tiny real-HF, tiny pretrained, actual Circle을 구분한다.

| 모델·수준 | eager unit/regression | 실제 평가 smoke | RoPE HF parity | per-layer actual Circle | full-model/full-text actual Circle | old/new strict | eager 전후 strict / 별도 관측 | frozen 상태 |
|---|---|---|---|---|---|---|---|---|
| Llama tiny HF/PTQ | 길이/cache/padding 보충 PASS; 기존 static-contract tests FAIL | pretrained 별도 행 | PASS32tests, 실제 HF/Torch Export | bundle6stage×3phase 완료, FAIL | full1×3 FAIL; native16384/eager4096+4/full export8도 FAIL | FAIL; host RoPE만 근거 있는 의도적 차이 | strict FAIL; 완료된 graph/qparams/host/배열 전후 exact | PASS, 실제 전체 snapshot |
| Llama pretrained Maykeye4.62M | unit와 구분 | PPL·likelihood PASS; FP generate PASS / Quant FAIL | Llama-3.2 pretrained BLOCKED | 원본 ERROR; 새 전체bundle INCOMPLETE; 선택 layer0 prefill/decode 완료 FAIL | 미실행 | 기존 pretrained checkpoint 없어 BLOCKED | 선택2stage strict FAIL, 전후 exact; 전체 미완료 | 평가·예외 및 선택stage PASS |
| Qwen3-VL tiny HF/PTQ | 추가 multimodal23tests PASS; 신규 full-text 준비누락 unit FAIL | PNG+연결decode4 PASS; generate FP PASS/Quant FAIL; pretrained BLOCKED | Llama 대상 아님; mRoPE unit | vision/text9×3 완료, decoder FAIL | full-text1×3 PASS, PNG eager1회 전후 | per-layer FAIL / full-text PASS | per-layer FAIL·전후 exact / full-text PASS | 완료 범위 PASS |
| Gemma4 tiny HF/PTQ | 추가 multimodal12tests PASS; sliding-offset/PLE/shared-KV unit | PNG+연결decode4 및 FP/Quant generate PASS; pretrained BLOCKED | Llama 대상 아님 | vision ERROR; 독립 text4×3 완료 FAIL | full-text1×3 완료 FAIL | main ERROR / text FAIL | vision 미완료; text/full-text FAIL·전후 exact | 완료 범위 PASS |

Random tiny fixture의 실제 Circle **68stage**를 완료했다. 실패한 Gemma vision과 별도 pretrained 실행은 이 수에서 제외한다. Stage 수는 PASS 수가 아니다. 별도 pretrained 전체 bundle은 before9stage/after0stage, 선택 2stage는 before/after 모두 완료했다. 모든 integration은 정상 import의 실제 Transformers, torch.export, tico.convert, CircleReferenceRuntime을 사용했다. Import stub/package 초기화 우회/mock compiler/runtime으로 integration PASS를 만들지 않았다. 기존 unit mock/stub은 unit으로만 보고한다. **Circle reference는 실제 NPU 실행·속도·메모리 검증이 아니다.**

## 환경·소스·재현

- Python3.12.14, Torch2.13.0+cpu, torchvision0.28.0+cpu, Transformers5.18.0, circle-schema0.10.1, datasets4.8.4, NumPy2.5.4, accelerate1.12.0. lm-eval/lmms-eval 없음.
- 가시 CPU5개/cgroup quota4CPU, RAM 약 17GiB, swap/GPU 없음, 시작 디스크 여유 약 30GiB. 개별 실행 자원과 최종 상태를 JSON에 기록했다.
- Root/관련 AGENTS.md·CLAUDE.md와 개발·테스트 지침, evaluation README, STATIC_EXPORT_REGRESSION, LLAMA_ROPE_UNIFICATION 및 debug 구현을 읽고 실제 CLI --help를 실행했다.
- 표준 `./ccex install --cpu_only`, `./ccex configure test --cpu_only`, `./ccex configure format`으로 `.venv`를 구성했다. 평가에 필요한 accelerate만 기존 dependency constraint를 보존하여 추가했다. 전역 업그레이드 없음, pip check 기록 보존.
- GitHub/PyPI/PyTorch/HF 접근 확인. 초기 project 설정·알려진 output/model/checkpoint 경로와 승인 HF cache에 기존 자산이 없었다. 개인 디렉터리 광범위 검색/자격증명 출력/multi-GB 다운로드 없음.
- 두 checkout은 같은 Python/dependency를 쓰며 worker의 실제 tico.__file__/SHA를 확인했다. Old/new는 같은 frozen full nn.Module CPU float32 파일을 읽었고 worker 내부 cast/calibration/재양자화 없음. Llama legacy migration은 candidate에만 적용했다. 별도 generation 진단의 fresh baseline calibration은 export equivalence로 부르지 않았다.
- `commands.sh`, `*.command.json`, `*.log`에 cwd·명령·비밀을 제외한 환경·시간·원래 종료 코드·stdout/stderr를 보존했다. Python 종료 코드를 tee로 잃지 않았다. FAIL 이후에도 독립 검증을 계속했다. Coordinator의 새로운 output 하위 디렉터리는 미리 만들지 않았다.

```bash
source /workspace/TICO/.venv/bin/activate
export RESULT_ROOT=/workspace/TICO/out/pr1073_validation/20261022T0404Z
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
./ccex test -k quantization
./ccex test
./ccex format --no-apply-patches
python tico/quantization/recipes/debug/verify_llama_rope.py \
  --repo . --report "$RESULT_ROOT/llama-rope-checks.json"
```

원래 실행의 정확한 fixture/config/CLI는 `commands.sh`와 개별 command JSON을 따른다. Export 재실행에는 새 output 경로를 사용해야 한다. `environment.json`, `packages.json`, `final_repository_environment.json` 및 각 checkpoint provenance에 실제 소스·환경·effective config·SHA256을 남겼다.

## Unit·전체 suite·정적 검사

| 실행 | 원본 결과 | 근거 |
|---|---|---|
| 변경/관련 canonical discovery | FAIL461tests/errors6/skip0 | changed_unit_discovery.log |
| 최초 직접 module 호출 | FAIL314/errors11 | 8개 test-root import 인프라 오류 포함, 원본 보존 |
| ./ccex test -k quantization | FAIL2670/errors9/skip141, exit1 | 완주; skip은 PASS 아님 |
| 전체 ./ccex test | INCOMPLETE1800초/exit124 | 최종 Ran-summary 없음. Generated op/net396ok/11skip만 별도 관측 |
| Qwen/Gemma 추가 multimodal unit | PASS23/12, skip0 | 실제 tiny HF/합성patch, pretrained/video benchmark 아님 |
| 전체 표준 lint/type 재실행 | FAIL835.68초/exit1 | UFMT19파일, Pylint5오류, mypy59오류+3advisory note |
| 실제 baseline counterpart lint | PASS UFMT11파일/mypy2파일 | 명시적 paths와 실제 adapter 실행 확인 |

Quantization 오류는 static-contract8개와 기존 45초 child timeout1개다. Llama routing fixture2개는 canonical rotary owner가 없고, attention adapter2개는 필수 mask를 누락하며, full static adapter3개는 prepared context 없이 호출한다. 신규 Qwen full-text1개도 준비를 누락했다. Baseline 기존 Llama export23/causal6/attention18tests는 같은 환경에서 PASS여서 기존 7개 Llama 실패를 공통 baseline 실패로 숨기지 않는다. 테스트를 새 계약에 맞추고 missing-input negative를 유지하는 최소안을 제안만 한다. 제품 검사를 완화하지 않았다. 파일/라인은 `unit_test_analysis.md` 참조.

Timeout 테스트는 기존 45초 제한 그대로 isolated candidate/baseline에서 35.758/33.394초에 PASS했다. 원본 suite ERROR는 유지한다. 전체 test1800초/최초 static900초/changed600초 중단은 OOM 판정이 아닌 시간 제한이다. 이후 표준 전체 static은 실제 완주했다.

Static86diagnostic은 모두 PR diff 파일이다. UFMT19개 중 새 8/기존 11, Pylint5개는 새 테스트, mypy59오류 중 새 파일 57/기존 표현식 2개다. 기존 2위치는 `test/quantization/recipes/test_export_llama_bundle.py:48`, `tico/quantization/recipes/export/llama.py:338`. 전체 행/메시지/미적용 typing·formatter 수정안은 `STATIC_CHECKS.md`, `static_diagnostics.json`에 있다.

최초 baseline lint는 exit0이나 `Linting files: []`여서 **NO_FILES_CHECKED 무효**로 철회했다. 당시 중간보고도 INVALID_SELECTION_v1로 보존했다. 실제 11/2파일의 `--paths-from` 재실행만 PASS 근거다. Changed600초도 의도 63개 대신 origin/main diff59개를 골라 INCORRECT_SELECTION+TIMEOUT이다. 최종 전체 범위는 완료된 표준 --all-files 검사다.

## A: eager·평가·상태

`EAGER_COVERAGE.md`는 요구사항과 파일/라인을 연결한다. Null/native·명시 cap·legacy-clamped 거부, 원본 positional/RoPE 보존, native S×S 미할당, padding/cache/position, Gemma sliding offset/PLE/shared-KV, PPL window/stride와 generation cap 분리, raw/chat-prefix 및 Llama evaluation.perplexity/VLM evaluation.ppl, backend 생략, cache on/off/config 전달, Video-MME 결과 cache와 KV정책 구분을 포함한다. Dummy config/mock evaluator routing은 unit이며 실제 benchmark 완료 증거가 아니다. Scoring use_cache=False는 의도된 정책으로 처리했다.

보충 real-HF/PTQ는 세 모델 calibration3/export6/eager12+연결 decode4/native32에서 PASS. 모든 weight/buffer/observer 통계·scale/zp·flag·mode/profile/return policy를 전후 exact hash 비교했다. Llama20parameters/572buffers/136observers/57quant modules, Qwen26/644/160/65, Gemma28/706/166/73이다. Calibration/forced-static 평가 거부와 Llama/Qwen 실제 prepared-static 예외 cleanup PASS. Gemma는 해당 owner helper가 없어 no-op context를 예외 복구 PASS로 주장하지 않았다. Missing mask/RoPE negative/synthetic cleanup은 별도 unit이다.

실제 저장 PNG+정상 로컬 HF processor로 Qwen/Gemma image prefill+연결 cached decode4, finite output/frozen 불변을 완료했다. HF/NO_QUANT maxabs Qwen2.4437904358e-6/Gemma1.4305114746e-6은 원래 tolerance PASS. Quant/FP hidden 차이는 0.26026475/0.20679152이며 품질 점수 동일성을 요구하지 않았다. Gemma normal image generate2는 FP/Quant `[251,245]`; Qwen은 FP완료/Quant accessor FAIL이다. Random weights/생성 PNG이며 pretrained 이미지 benchmark가 아니다. PLE/shared-KV unit와 해당 기능 off인 Circle fixture도 구분한다.

### 작은 pretrained Llama

`Maykeye/TinyLLama-v0`, revision `298338802ab94432b917bcce11382aa151aee50f`, 실제 4,621,392parameters. 선택 7파일약 11.8MB와 Wikitext cache약 13MiB만 받았다. Llama-3.2 검증이 아니다. 원본 full PTQ SHA256 `5f8b5a9b5f7e8ff182c6fc96d52f232a9958424f68c09039fd9452e101b17345`, CPU float32/npu_export W8A16/native2048/calibration32/export16.

기존 recipe --tasks/--set 실제 CLI로 FP/Quant PPL을 각각 완료했다. 같은 12Wikitext test row/660token corpus hash/tokenizer/revision/float32/seed42/cap128/window64/stride32/scoring을 적용하고 output/cache를 분리했다. Metric/완료로그도 확인했다. 별도 normal-model 호출로 고정 6/73token likelihood와 FP cached generate8을 완료했다. 73은 calibration/export보다 길다.

| metric | FP | Quant | Quant−FP |
|---|---:|---:|---:|
| PPL 고정 smoke | 6019.2490234375 | 5889.0615234375 | -130.1875 |
| NLL sample0/5targets | 39.6580009460 | 39.4269180298 | -0.2310829162 |
| NLL sample1/72targets | 359.3289794922 | 368.0004577637 | +8.6714782715 |

소수 smoke이며 전체 품질/향상 결론이 아니다. Quant generation FAIL과 예외 전후 frozen PASS를 함께 기록했다. `PRETRAINED_SMOKE.md`, `pretrained_summary.json`, protocol/checkpoint/asset identity와 실제 metric 로그 참조.

## B-1: Llama RoPE

원본 verify_llama_rope.py는 **32tests/0skip/exit0**. 이 총계에는 실제 tiny HF 모델 RoPE 테스트와 synthetic-owner/static-input unit이 함께 포함된다. Default/linear/llama3 및 hf/pre_negated_sin, 독립 HF/full/standalone/child, config 생성/명시 주입, padding prefill+teacher-forced decode/deltaKV, custom model rotary, static snapshot 격리, 저장·복원/legacy migration/frozen qparams, 실제 torch.export parity를 포함한다. 일부 unit legacy는 candidate 속성 삭제 모사다. 별도 Circle fixture는 실제 baseline에서 저장한 같은 full module을 사용했다.

Candidate bundle host cos/sin은 같은 config/capacity/position의 독립 HF와 **maxabs0**. Baseline-vs-HF cos4.4465065e-5/sin0.0068924397이며 위치/원소수를 저장했다. 실제 양 Circle에 같은 외부 mask/cos/sin을 넣은 6stage 모든 probe replay는 exact. Weight/qparams/mode/I/O 보존. **이 host 차이만 EXPECTED_NUMERICAL_CHANGE**이며 원본 old/new FAIL/exit1은 유지한다. Qwen/Gemma/다른 parity 실패에 적용하지 않는다. Tiny native64/original8192 scaling HF 경고도 공개하며 native16384/original8192 별도 실험과 구분한다. Llama-3.2 config/weights는 없어 BLOCKED다.

## B-2: actual Circle·원인·미적용 수정안

완료 phase의 serialized graph I/O·shape/dtype·topology/options·constants/qparams·host와 모든 probe input/Torch/Circle output/deltaKV를 독립 감사했다. 누락/NaN/Inf 없음. 원본 JSON/exit code 보존. 전체 목록은 `CIRCLE_VALIDATION.md`, `circle_validation_summary.json`.

1. **Llama 기존 parity 문제:** 실제 softmax42.5 tie에서 Torch42/255 ties-even vs Circle43/255 ties-away. `observers/affine_base.py:179`, `circle/runtime/quantization.py:53,136` 소스는 baseline 동일. 초기 fixture populated-past 관측 부족도 별도 확인했다. 별도 native64/export8 full-model에서는 logits 최대 오차 0.0055593401이 3phase에서 동일하고 4KV는 exact다. RoPE 탓으로 묶지 않는다. `PARITY_DIAGNOSIS.md`, `ROUNDING_AND_PAST_DIAGNOSIS.json`.
2. **Qwen per-layer FAIL:** hidden0.0447399/deltaK0.0416335/deltaV exact. Old/new·전후 graph/host/모든 배열 exact라 baseline에도 존재한다. DynamicCache가 tuple 전용 past observer를 채우지 않는 것은 확인했으나 past-free prefill/deltaK의 모든원인은 미확정이다. 설명되지 않은 strict 수치 실패로 남긴다. Full-text만 PNG eager1회 전후 strict PASS/maxabs0이며 full multimodal/video PASS가 아니다.
3. **Gemma vision ERROR:** 실제 op102 RMS_NORM rank2[4,32]를 양 checkout의 `tico/circle/runtime/kernels/nn.py:620`에서 거부. 실패 Circle SHA256 `4cb5d8deecfef8cf41338211d4d298170048ef0bf64f1893d4b85b6442b0af5d`와 finite Torch features 보존. 제안은 검증된 rank2지원 또는 rank3 legalize/2DI/O복원; 제품 미수정.
4. **Gemma text semantics:** 4stage×3phase/30probes/90outputs complete·finite. Layer0prefill hidden0.3541484475, K/V와 나머지 3stage exact. 실제 trace에서 qnorm은 exact지만 frozen q_x1/q_x2 서로 다른 grid가 serialized SPLIT→CONCAT에 보존되지 않는다. 값 2.764853954가 Torch halfclip2.288276911→qcat2.276938677, Circle은 2.764853954유지. Softmax/hidden으로전파. Maskexact/window8=seq8/동일입력 quantizer61경계 exact여서 단순 mask/rounding/LlamaRoPE가 아니다. 관련 pass는 baseline 동일, 정확한 소실 pass는 미확정. 최소안은 split/getitem producer domain 인식과 **distinct half-grid explicit requantization 보존**. `GEMMA_TEXT_PARITY_DIAGNOSIS.md`의 실제 trace/소스 라인/제안 참조. 수정 후 검증은 미실행.
5. **Gemma full-text 별도 FAIL:** 기록된 필수 mask를 제공한 copied runner3phase완료. hidden[1,8,32] max0.0869361162 at[0,4,10],92/256초과. Old/new·전후 graph/모든 배열 exact. 내부 원인은 미확정이며 per-layer원인으로 자동 대체하지 않는다. `GEMMA_FULL_TEXT_DIAGNOSTIC.md`.
6. **작은 pretrained Circle:** 원본 cached observer32개 scale inf로 ERROR exit2. 별도 28prefill+4decode calibration **새 파일** SHA256 `c13bbd08a738a29c55c782bd1942f7b4056c20eb4c98b2f1628e1c99c64343ad` 생성. 전체 bundle1200초 INCOMPLETE(before9stage/after0), RAM 부족이 아닌 bounded runtime. 같은 새 full model instance의 선택 layer0prefill/decode2stage는 전후완료. Eager64+decode4/native2048·전후 exact·frozen 보존, 그러나 prefill 최대 오차는 hidden 4.041e-5/deltaK 8.225e-5/deltaV 3.864e-5이며 hidden/V만 tolerance 위반이다(delta-K는 rtol 범위 내). 원본 strict FAIL/exit1을 유지한다. 이를 8layer전체 PASS로 표현하지 않는다.

처음부터 native16384로 만든 새 22,688parameter checkpoint SHA256 `bc5efaaaf81d4e3d4efdd8b9bbb790971e58a171cdbf26f80577739f2ac602bf`에서 **eager4096+연결 decode4**와 full Circle전후 실행 완료. Static capacity8/117ops/140qparam tensors/5outputs 전후 exact/frozen 보존이지만 Torch–Circle0.009165 strict FAIL유지. Peak tree RSS약 1.28GiB. 옛 checkpoint config를 늘리지 않았다. >2048eager이며 2048static/pretrained검증이 아니다.

### 다른 제품·검증도구 실패

- **기존 generation API 호환성:** Transformers5.18 `generation/utils.py:2480`의 get_experts_implementation을 QuantLlama/Qwen이 제공하지 않아 actual generateFAIL. Baseline fresh PTQ에서도 재현한 PREEXISTING_COMPATIBILITY_FAILURE. Llama `quant_model_for_causal_lm.py:45`/Qwen `quant_for_conditional_generation.py:42`에 Gemma와 같은 get/set delegation+actual generate회귀 검사를 추가하는 최소안만 제안한다. Manual decodePASS는 generate PASS가 아니다.
- **Comparator negative2건 FAIL:** 모든 Torch output/parity 또는 모든 input을 양쪽에서삭제해도 원본이 성공처리. `export_regression.py:717` coverage 부족/`:732` all([]). 정확한 graph I/O indexed key set/nonempty parity/output별 finite 검사 제안. 제품 comparator 미수정, 실제 결과는별도 inventory 감사. `COMPARATOR_GAPS.md`.
- **Calibration 위험:** `CALIBRATION_PATH_AUDIT.md` 관련 16함수 baseline 동일. Qwen stock cache-off prompt-only와 tuple past 관측 공백은 source/fixture근거 있는 기존 위험이며 실제 pretrained recipe 실패 증거는 아니다. Llama decode_steps0/Gemma shared-present 관측과 구분한다.
- **Old/new 분류:** Llama host만 EXPECTED_NUMERICAL_CHANGE. 다른완료 case의 실제 old/new Circle 수치는 exact다. Gemma text Torch FX는 3ops감소(alias/assert/to등), Circle는 exact이며 candidate전후 FX도 exact다. Qwen prefill/Gemma full-text 미확정 Torch–Circle오차는 baseline-existing unexplainedparity로 남긴다. 설명되지 않은 old/new차이를 RoPE 예외나 PASS로 덮지 않았다.

## 검증 인프라 보완

원래 오류·로그·exit code/스크립트 revision을 보존하고 새 output/SHA로 분리했다. 비교 기준/tolerance/제품은 그대로다.

| 원래문제 | 허용된보완 |
|---|---|
| 직접unit import경로 | 표준discovery호출 |
| Fixture root/grid/cache관측 | fullPTQWrapper/tuplegrid/별도cachedcheckpoint, 이전파일보존 |
| Gemmaopt-inlayoutvalidator의2Dshape[1] token수오류 | 원래TrueFAIL보존, 별도기본False profile의grid/feature/placeholder를 명시검사; 같은case의flag만내려PASS로바꾸지않음 |
| BaselineLlamafull shape witness누락 | copiedrunner에CPUlongshapeexample만제공, calibration없음 |
| Fulltextemptyartifactset 저장parent없음 | copiedrunner한줄mkdir; coordinatoroutput은사전생성하지않음 |
| Gemmafulltext필수maskdict없음 | 같은frozen파일의기록full/slidingtemplate을buffer로제공 |
| Tracehelper Path-vs-bytes | read_bytes만고친v2replay, 실패v1보존 |
| Lintzero-files/changedselection | 무효철회후matchedpaths/완료전체표준재실행 |

`FIXTURE_INFRASTRUCTURE.md`, copied runner diff/provenance, `pretrained_validation_infrastructure_changes.json`에 정확한 변경/hash가 있다. Qwen/Gemma fulltext의 custom PNG kwargs는 **자동 stress를 대체**한 use_cacheFalse forward1회다. CLI eager length/decode count를 실제 실행으로 세지 않는다. PNG 연결 decode4검증은 별도 fixture다.

## 미검증 범위·필요 자원

- **BLOCKED:** Llama-3.2/Qwen3-VL/Gemma4 실제 pretrained old/new2048export, pretrained VLM 이미지 평가, Video-MME 실제 metric. 필요한 자산: 신뢰할 full nn.Module CPU float32(state_dict 아님), 일치하는 원본 model/tokenizer/processor revision, calibration provenance, recorded capacity, qprofile/vision grid/attention/PLE/shared-KV설정+SHA256, 평가 데이터/sample IDs. 큰 평가에는 lm-eval/lmms-eval도필요하다.
- Random tiny baseline 생성 checkpoint의 same-file old/new는 기존 사용자 pretrained 비교를대신하지 않는다. 큰 FP/Quant도 같은 revision/전처리/chat template/few-shot/seed/cap/generation/scoring으로 별도 output/cache에서 실행해야 한다.
- CPU-only17GiB에서 multi-billion2048export는 checkpoint별 RAM/disk 예산을 먼저 평가하고 충분한 호스트에서 순차 실행해야 한다. 승인 없는 multi-GB download/새 전체 calibration은시작하지 않았다. 이 범위가 실제 OOM했다는뜻은 아니다.
- Pretrained native2048을 4096으로조작하지않았다. >2048pretrained isolation에는 capacity≥4100의신뢰할 새 checkpoint가 필요. 완료 native16384random/static8과구분한다.
- 전체 test1800초/작은 pretrained 전체 bundle1200초 INCOMPLETE는 더 긴 CPU 실행 예산이 필요. Completed subset을전체 PASS로세지않는다.
- Gemma vision runtime 미지원 및 strict수치 FAIL 해결 전 전체 multimodal parity 승인 불가. Full-text는 full multimodal/video export가 아니다. NPU benchmark는 범위 밖이다.

## 산출물

`summary.json`은원본 판정/명령/evidence index/model matrix를연결한다. `commands.sh`, 원본로그/JSON, 실제 Circle/NPZ/PNG, checkpoint/config/provenance를 결과 root에보존한다. `artifact_manifest.json`은배포 파일 SHA256과제외한 transient cache를기록한다. 다운로드 ZIP에는로그/스크립트/결과/실제 모델·checkpoint·출력을포함하고 mypy/HF dataset 재생성 cache만제외한다. 모든 원본은 workspace에 유지한다.

핵심부속문서: CIRCLE_VALIDATION.md, PRETRAINED_SMOKE.md, EAGER_COVERAGE.md, EAGER_SUPPLEMENT_RESULTS.md, LLAMA_LONG_EAGER_ISOLATION.md, GEMMA_TEXT_PARITY_DIAGNOSIS.md, GEMMA_FULL_TEXT_DIAGNOSTIC.md, COMPARATOR_GAPS.md, STATIC_CHECKS.md, unit_test_analysis.md, VALIDATION_LEVELS.md.
'''

report_path = root / 'PR1073_VALIDATION.md'
prior = root / 'PR1073_VALIDATION_IN_PROGRESS.md'
if report_path.exists() and not prior.exists():
    prior.write_bytes(report_path.read_bytes())
report_path.write_text(report, encoding='utf-8')
matrix = {
    'policy': 'Strict raw status retained separately from exact before/after component equality.',
    'Llama_tiny_real_HF': {
        'eager_supplement': 'PASS', 'rope_HF_32_tests': 'PASS',
        'per_layer_circle': 'FAIL', 'full_circle': 'FAIL',
        'old_new': 'FAIL_EXPECTED_HOST_ROPE_CHANGE_PLUS_BASELINE_PARITY_FAILURE',
        'eager_isolation_strict': 'FAIL', 'completed_before_after_components': 'EXACT',
        'long_eager': 'COMPLETED_4096_PLUS_4_NATIVE16384_STATIC8_STRICT_FAIL',
        'frozen_state': 'PASS'},
    'Llama_tiny_pretrained': {
        'model': 'Maykeye/TinyLLama-v0', 'ppl_likelihood': 'PASS',
        'generation_fp': 'PASS', 'generation_quant': 'FAIL_PREEXISTING_API',
        'original_circle': 'ERROR_NONFINITE_CACHE_SCALE',
        'cache_calibrated_full_bundle': 'INCOMPLETE_TIMEOUT_1200_SECONDS',
        'selected_layer0_prefill_decode_strict': 'FAIL',
        'selected_before_after_components': 'EXACT', 'frozen_state': 'PASS',
        'preexisting_pretrained_old_new': 'BLOCKED'},
    'Qwen3_VL_tiny_real_HF': {
        'additional_multimodal_unit': 'PASS_23', 'PNG_connected_decode': 'PASS',
        'normal_generate_fp': 'PASS', 'normal_generate_quant': 'FAIL_PREEXISTING_API',
        'per_layer_circle_old_new_and_isolation': 'FAIL_BASELINE_PARITY',
        'full_text_PNG_one_forward_old_new_and_isolation': 'PASS',
        'completed_before_after_components': 'EXACT', 'frozen_state': 'PASS',
        'pretrained_evaluation': 'BLOCKED'},
    'Gemma4_tiny_real_HF': {
        'additional_multimodal_unit': 'PASS_12', 'PNG_connected_decode': 'PASS',
        'normal_generate_fp_quant': 'PASS', 'vision_circle': 'ERROR_UNSUPPORTED_RANK2_RMS_NORM',
        'independent_text_per_layer': 'FAIL_BASELINE_QUANTIZATION_BOUNDARY_LOSS',
        'full_text': 'FAIL_UNEXPLAINED_BASELINE_PARITY',
        'completed_text_before_after_components': 'EXACT', 'frozen_state_completed_scope': 'PASS',
        'pretrained_evaluation': 'BLOCKED'}
}
(root / 'model_results.json').write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'report': str(report_path), 'model_matrix': str(root / 'model_results.json')}))
