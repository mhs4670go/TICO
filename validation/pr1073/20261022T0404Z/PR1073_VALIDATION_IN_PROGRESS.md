# TICO PR #1073 검증

**진행 중 — 아래 완료 범위와 원본 실패를 기록하며, 전체 실행 종료 후 최종 갱신한다.**

검증 SHA: `6b2f6a0511458d6a750e2a260d706080a78149dd` (`mhs4670go/TICO`, `ea`; Samsung/TICO PR1073 원격 HEAD와 일치). 원래 export baseline은 `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`이며 `/workspace/TICO-export-baseline`에 detached worktree를 만들었다. `61f3a05`/최신 main으로 대체하지 않았다.

제품 소스, baseline 소스, PR metadata를 변경하지 않았다. commit/push/merge 없음. 별도 검증 fixture·스크립트만 결과 디렉터리에 추가한다. Circle reference 실행은 실제 NPU 실행/속도/메모리 또는 bit-exact integer backend 검증이 아니다.

## 환경과 재현

- Python3.12.14, 가시 CPU5개 / cgroup quota4 CPU, RAM약17GiB, GPU없음. 시작 디스크 여유약30GiB.
- 기존 Torch/Transformers 및 pretrained 모델/체크포인트/데이터 없음. project output/models/checkpoints 및 approved HF cache만 검사했다.
- 저장소 표준에 따라 `.venv`에서 `./ccex install --cpu_only`, `./ccex configure test --cpu_only`, `./ccex configure format`을 실행했다. Torch2.13.0+cpu, TorchVision0.28.0+cpu, Transformers5.18.0, circle-schema0.10.1. 선택 평가에 필요한 accelerate1.12.0만 기존 패키지 constraint를 보존하여 추가했다. 시스템 전역 업그레이드 없음.
- 정확한 패키지/실제 `tico.__file__`/cgroup/CPU/RAM/disk는 `environment.json`, `packages.json`, 초기 raw log에 있다. 각 worker는 지정 checkout 소스 경로를 검증한다.
- `commands.sh`와 각 `*.command.json`은 cwd·명령·환경·종료 코드·시간, 대응 `*.log`는 merged stdout/stderr를 보존한다. CLI 실제 `--help`도 기록했다.

## 현재 완료 범위

| 검증 | 결과 | 수준 / 근거 |
|---|---|---|
| 관련 PR test canonical discovery | FAIL:461개, errors6, skip0 | Unit/mock + 일부 real tiny HF. 원본 `changed_unit_discovery.log` |
| 최초 직접 module 호출 | FAIL:314개, errors11 | test-root import 경로 문제8개 포함. 원본 보존 후 표준 discovery로 재실행 |
| 추가 Qwen/Gemma multimodal unit | PASS:23/12개, skip0 | 실제 random tiny HF + 합성 patch. Gemma 내부 unit 명시 활성화 |
| Llama RoPE 전용 | PASS:32개, skip0, exit0 | tiny real HF/Torch Export. `llama-rope-checks.json` |
| 3모델 long eager + frozen-state | PASS | 실제 tiny HF/PTQ. calibration3/export6/eager12+decode4/native32 |
| Qwen/Gemma 생성 PNG + 실제 processor + cached decode | PASS | tiny real HF/PTQ, image prefill + 연결된4 step, benchmark 점수 아님 |
| 작은 pretrained Llama FP/Quant PPL·likelihood | 완료 | 고정 Maykeye revision, 실제 recipe CLI. FP6019.2490 / Quant5889.0615 |
| 작은 pretrained Llama generation | FP PASS / Quant FAIL | pinned Transformers5.18 generation API 누락, baseline fresh PTQ에도 재현 |
| tiny Qwen image generation | FP PASS / Quant FAIL | 같은 메서드 누락. baseline 독립 실제 재현 완료 (같은 기존 호환성 오류) |
| tiny Gemma image generation | FP/Quant PASS | 같은 PNG/processor/greedy2 token, frozen-state 보존 |
| comparator 추가 negative | FAIL:손상된 증거를 성공 처리2건 | 검증 도구 결함. `COMPARATOR_GAPS.md` |
| Owning `./ccex test -k quantization` | FAIL:2670개, errors9, skip141 | 완주. 오류8개는 static 테스트 계약,1개는 45초 child timeout (원래 제한 그대로 양쪽 isolated rerun PASS) |
| 전체 `./ccex test` | INCOMPLETE:1800초 timeout124 | generated op/net396 ok/11 skip 후 quantization 일부 실행, 전체 PASS 아님 |
| 전체 lint/type 최초 실행 | INCOMPLETE:900초 timeout124 | 원본 보존. 변경 파일 검사도600초 timeout124. 표준 전체 재실행은835.68초에 실제 exit1 완료 (UFMT19/Pylint5/mypy59errors+3notes) |
| UFMT 전체 파일 별도 검사 | FAIL:19개 파일 | 새 파일8개+변경 파일11개. baseline 최초 비교는 파일 미선택으로 무효이며 명시적 파일 선택으로 재검증 중. patch 미적용 |
| Llama actual bundle Circle | strict FAIL,6stages×3phases 완료 | host 차이는 HF 기준으로 설명. 모든 동일외부입력 stage replay exact, frozen exact |
| Qwen actual per-layer/vision Circle | strict FAIL,9stages×3phases 완료 | old/new와 eager 전후 graph/host/모든 arrays exact, localTorch parity는 baseline에도 FAIL |
| Gemma vision Circle | ERROR:baseline/candidate 동일 | rank2 RMS_NORM 미지원. text stage/full-text는 별도 독립 실행 |
| Tiny pretrained bundle Circle | ERROR / INCOMPLETE | 원래 fixture는 scale=inf 거부. 별도 cache-calibrated fixture는1200초 timeout,9실제stages/aftereager미완료 |

## 실제 평가 및 상태 보존

작은 pretrained 모델은 `Maykeye/TinyLLama-v0` revision `298338802ab94432b917bcce11382aa151aee50f`의 실제4.62M 파라미터다. 선택한 safetensors/config/tokenizer 파일 약11.8MB만 받았다. multi-GB 모델 다운로드 없음. Llama-3.2 결과가 아니다.

FP/Quant는 같은12개 Wikitext test row, 동일한660-token corpus/hash, tokenizer, float32, seed42, cap128, PPL window64/stride32를 사용했다. calibration32/export16과 eager73-token 입력을 구분했다. 실제 metric 완료 로그와 `pretrained_smoke_results.json`을 확인했다. protocol JSON만으로 완료 판정하지 않았다. `pretrained_protocol_comparison.json`은 허용된 checkpoint/class/mode 차이 외 프로토콜 일치를 확인한다.

PPL은 FP6019.2490234375, Quant5889.0615234375, 차이-130.1875이다. 작은 고정 샘플 smoke이며 전체 benchmark 품질 결론이 아니다. FP/Quant 점수 동일을 요구하지 않았다. 두 fixed likelihood 입력과 FP cached generation8은 완료했으나 Quant generation은 `get_experts_implementation` 누락으로 실패했다. 예외 전후에도 모든 frozen state/config/profile이 동일했다.

생성 PNG 실험의 HF/NO_QUANT 최대오차는 Qwen2.4437904e-6/Gemma1.4305115e-6이며 원래 atol1e-5/rtol1e-4를 유지했다. Quant/FP hidden 변화량은 각각0.26026475/0.20679152이며 품질 점수나 동등성 PASS를 의미하지 않는다. 상세 state hashes·실제 tensor·PNG·processor config는 `EAGER_SUPPLEMENT_RESULTS.md`와 하위 JSON에 있다.

## 현재 재현된 실패와 분류

1. **테스트의 새 static 계약 미반영**: candidate focused discovery 오류6개는 Llama FakeLayerWrapper2개, Llama full static adapter 미준비3개, 신규 Qwen full-text adapter 미준비1개다. Owning 전체 quantization suite에는 attention mask 미제공 Llama 테스트2개가 추가돼 deterministic 오류8개이며,45초 subprocess timeout1개를 포함해 errors9개다. timeout 테스트는 기존45초 제한을 그대로 유지한 isolated candidate/baseline 재실행에서 PASS했지만 원래 ERROR를 지우지 않았다. baseline Llama 해당 module23개, causal-LM module6개와 attention module18개는 같은 Python/deps에서 PASS. 제품/테스트 원본을 수정하지 않았다. `unit_test_analysis.md`에 파일/라인과 미적용 최소 수정안이 있다.
2. **실제 generation 호환성 오류**: Llama Quant wrapper가 Transformers5.18 `generation/utils.py:2480`에서 요구하는 experts accessor를 제공하지 않는다. baseline에서 fresh tiny PTQ로 동일 오류를 재현했으므로 PR 고유 회귀로 단정하지 않는다. Qwen도 candidate와 baseline의 실제 image generate에서 동일 오류를 재현했다 (baseline fresh tiny calibration은 export-only 비교가 아니다). Gemma는 이미 accessor를 위임해 해당 실험이 완료됐다. 외부 evaluator나 product에 임시 patch를 적용하지 않았다.
3. **comparator 증거 완전성 결함**: 양쪽에서 모든 Torch 배열+parity 목록 또는 모든 input 배열을 같이 삭제해도 원본 comparator가 성공 처리했다. `export_regression.py:717` coverage는 Circle output 하나 이상만 검사하고 `:732`의 `all([])`가 누락 parity를 허용한다. 실제 Circle에서는 graph I/O와 모든 output/delta-KV 목록을 별도 audit한다. 이 결함은 RoPE 수치 수정과 무관하다.
4. **Circle strict Torch parity**: baseline 실제 decoder에서 오차가 관측됐다. 큰 populated-cache decode 오차 일부는 초기 tiny fixture가 past-KV observer를 calibration하지 않은 한계로 좁혀졌다. 기존 raw FAIL을 보존하고 cache-bearing calibration이 있는 별도 새 fixture를 만든다. prefill에서는 실제 softmax qparams의42.5 tie에서 Torch42/255(ties-even), Circle43/255(ties-away)로1 quantization step 차이가 재현됐다. 해당 observer/runtime 소스는 baseline/candidate 간 동일하다. `PARITY_DIAGNOSIS.md`에 수치·위치·기존 결과의 bit-exact 재실행 증거를 보존했다. tolerance를 올리지 않는다. 이 분석은 기존 FAIL을 PASS로 바꾸지 않는다.

## 추가 calibration 경로 감사

`CALIBRATION_PATH_AUDIT.md`와 근거 JSON은 관련16개 함수가 baseline/candidate에서 동일함을 확인했다. Qwen 표준 recipe가 cache를 끈 prompt-only calibration을 하고, DynamicCache 경로가 tuple 전용 `obs_past_key/value`를 우회하는 기존 coverage 위험이 있다. 실제 tiny checkpoint에서 미수집 통계/scale를 확인했지만 실제 pretrained 표준 Qwen recipe 수치 실패를 입증한 것으로 표현하지 않는다. Llama는 양수 decode calibration에서 같은 past observer를 사용하며 decode_steps=0은 별도 조건이다. Gemma는 tuple/HF/current KV가 present observer를 공유해 같은 분리 문제로 묶지 않는다. Qwen prefill/delta-K와 Gemma sliding-prefill의 모든 오차를 이 원인으로 설명하지 않는다.

## 미검증 자원 / 범위

기존 pretrained Llama-3.2/Qwen3-VL/Gemma4의 trusted frozen full `nn.Module` CPU float32 checkpoint와 일치하는 원본 revision/tokenizer/processor가 없다. 대형2048 old/new 및 실제 VLM benchmark/Video-MME는 해당 자산 없이는 BLOCKED다. `state_dict`만으로는 현재 runner 요구를 충족하지 않는다. 필요한 profile, context capacity, SHA256, 생성/quantization provenance도 함께 필요하다. 기존 baseline-compatible pretrained checkpoint 비교와 새 baseline-generated tiny PTQ 비교는 별개다.

native2048인 작은 pretrained checkpoint를4096으로 늘리지 않았다. 별도 새 native16384 tiny candidate(22,688 parameters)에서는4096 eager+4cached decode 및 capacity8 full Circle 전후 재수출을 완료했다.117ops/140qparam tensors/5outputs와 모든 배열이 전후 exact이며 frozen state도 보존됐다. strict Torch parity 최대0.009165 차이로 원래 exit1은 유지한다.2048 static이나 pretrained 검증으로 일반화하지 않는다. 전체 자원 목록은 `blocked_resources.json`을 참조한다.

## 부속 증거

`VALIDATION_LEVELS.md`, `EAGER_COVERAGE.md`, `EAGER_SUPPLEMENT_RESULTS.md`, `COMPARATOR_GAPS.md`, `FIXTURE_INFRASTRUCTURE.md`, `unit_test_analysis.md`, `static_checks_summary.json`, 원본 command/log/JSON 및 실제 stage artifacts를 같은 결과 root에 보존한다. 최종 summary와 model별 Circle/full/isolation matrix는 실행 종료 후 갱신한다.
