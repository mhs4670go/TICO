# Wrapper CPU 검증 결과

검증 checkout `${CANDIDATE_REPO}` head `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`, baseline `${BASELINE_REPO}` `261bda28ec55fdac42f59346c950326e4065f8fa`. 공용 CPU venv Python3.12/Torch2.13.0+cpu/Transformers5.18.0 사용. 각 subprocess는 `PYTHONPATH`를 해당 checkout으로 고정했고 각 artifact JSON에 `tico.__file__`을 기록했다. 원본 제품/테스트 코드는 변경하지 않았다.

## 범위와 결과

현재 registry의 50 case 모두 availability=true. 기본 YAML 그대로 CPU/FP32, seed123, tiny profile, calibration_iters3을 유지했다. 각 case builder/input 생성기의 내부 고정 seed(예: assistant2026/2027)도 원본 그대로다. 전체 numerical smoke는 exit0, **50 실행/50 PASS/0 FAIL/0 SKIP**, 10.674초. FP/Quant 실제 forward, 기존 threshold, shape, finite 조건을 원래 runner가 검사했다. `all` mode는 개별 JSON을 저장하지 않으므로 원본 로그와 그 로그를 파싱한 `wrapper-smoke/summary.json`을 근거로 집계했다.

Circle 지원 플래그가 true인 **48 case를 각각 독립 subprocess로 순차 실행**했다. 나머지2 case는 numerical까지 통과했고 일반 Circle export는 문서상 의도적으로 제외했다: `gemma4_text_model`(전용 static adapter 필요), `gemma4_assistant_sparse_head`(host ordered sparse head).

| 구분 | 실제 결과 | 근거 |
|---|---:|---|
| 원래 inspector numerical+export exit | 42 성공,6 실패 | `wrapper-circle/CASE.json`, `artifact-checks.json.inspector_exit_code` |
| 실제 파일 저장 | 42 PASS,6 no-artifact | 파일 경로/bytes/SHA256 기록 |
| 저장 파일 재읽기 + 구조 검사 | 42 PASS | `CircleDocument.load/verify(raise_on_error=True)` |
| 최초 runtime 보조 harness | 41 실행 PASS,1 dict input binder 오류,6 no-artifact | kernel 실패와 binder 오류 구분 필요 |
| 최초 raw input strict parity | 25 PASS,16 FAIL,1 binder 때문에 미완료,6 no-artifact | atol1e-5/rtol1e-4 그대로 |
| 보완한 name-checked dict binder + 동일 encoded ABI replay | runtime42 실행,parity25 PASS/17 FAIL,6 no-artifact 제외 | `wrapper-circle-encoded-replay/summary.json` |
| 최초 전체 topology/state hash | 47 동일,1 달라짐 | Gemma vision profile 생성 분석으로 해소,아래 참조 |

총48 export subprocess 시간 합계452.337초(대기시간 제외). 모든42 runtime은 serialized quantized input ABI의 **FAKE_QUANTIZE CPU reference** 모드였다. 실제 NPU integer 실행이나 bit-exact integer backend 검증을 뜻하지 않는다.

원본 첫 실패/exit/status는 고치지 않았다. 최초 Gemma model runtime FAIL은 production `ModelInputSpec`가 positional dict를 `torch.tensor(dict)`로 처리하려는 보조 harness 바인딩 문제다. 후속 replay는 실제 adapter Python signature를 재귀적으로 펼치고, 만들어진 이름 집합이 serialized Circle input names와 정확히 같을 때만 입력을 연결했다. 이 보완 후 실제 Gemma model reference runtime은 실행되었지만 strict parity는 FAIL이다. 다른 case는 원래 production binder를 그대로 사용했다.

## 관찰 방법과 mock 경계

`export_case.py`는 원래 inspector.main()에 실제 CLI 인자를 전달한다. `_export_circle`과 `tico.convert`를 관찰용 spy로 감싸지만 **원래 함수를 같은 인자로 그대로 호출**한다. 실제 quantization/model forward/export/runtime을 대체하지 않는다. converter의 정확한 입력을 복제해 Torch expected를 계산하고, 실제 저장된 Circle 파일을 다시 읽어 검증/실행했다. 저장/구조/runtime/수치/frozen 결과는 각각 기록했다. `--report-json`은 각 단일 case에만 전달했다.

encoded replay는 기존 KV Circle tests와 같은 규칙으로 serialized integer ABI 입력을 encode한 뒤 **동일한 encoded 값을 Torch와 runtime 양쪽에 제공**했다. observer를 다시 수집하거나 qparam을 바꾸지 않았다. 원본16 raw parity 실패의 원래 입력/Torch 출력 hash는 모두 정확히 재현되었다. 입력 ABI 정렬 후에도 실패가 남아 단순한 입력 인코딩 차이만으로 설명되지 않는다. 추가 Gemma model을 포함해17 실패가 남았다. replay source state는 모두 불변이었다.

## Base 비교

원본 base에서도23 실패 case와3 양성 대표 case(`nn_linear`, `llama_attention_decode`, `gemma4_text_attention_shared_kv_decode`)를 **독립적으로 prepare/calibrate/convert**했다. head checkpoint를 base에 넣지 않았다. 원본 base 결과는3 PASS/23 FAIL,211.419초. 새 checkpoint schema를 읽지 못한 실패와 혼동하지 않는다.

19개의 비교 가능한 저장 pair(양성3+수치실패16) 모두 raw 입력, raw Torch 출력, encoded Torch 출력, encoded Circle 출력의 dtype/shape/bytes hash가 **revision 사이에서 정확히 동일**했다. 세부 관측 grid는 dtype/axis/range/scale/ZP/lock/enabled 및 weight hash로 정규화해 기록했다. 일부 미관측/선택적 observer의 캐시 표현은 새 lifecycle 계약 때문에 달랐다. 파일 전체 SHA는 서로 다르므로 artifact bytes 동일성을 주장하지 않는다. 실제 비교 입력에서의 Torch/Circle 출력 동일성을 근거로 수치실패16건을 기존 문제로 분류한다.

원본 Qwen vision attention base fixture는 Transformers5.18에서 잘못된 positional argument 때문에 실행 전 실패했다. Qwen model/conditional base fixture는 mm_token_type_ids가 없어 reference/calibration 도중 먼저 실패했다. PR은 이 작은 입력 fixture들을 고쳤다. 이3건은 candidate와 같은 의미의 입력만 적용한 별도 base 실행을 추가했고 원본 base 실패는 그대로 보존했다(추가3회21.357초). Qwen vision attention은 같은 raw/encoded 입력·Torch·Circle 출력 hash와 max error8.961232379078865e-05를 재현했다. 따라서17개 saved numerical 실패가 모두 기존 문제로 확인되었다. 나머지 model/conditional은 올바른 mm_token_type_ids 제공 후 candidate와 같은 explicit position_ids compile assertion까지 도달했다. 결과는 `qwen-equivalence-results.json`/`equivalent-revision-comparison.json`에 따로 기록했다.

## 실패 원인별 구분

1. **기존 saved-Circle 수치 차이**: `nn_tied_embedding`, Llama attention prefill/decoder prefill·decode, Qwen text attention prefill·decode/text decoder prefill/vision block, Gemma text attention prefill·K=V prefill/vision patch embedder/vision attention/vision encoder layer/vision model/vision encoder/model. 동일 입력과 출력 bytes의 base 재현으로 확인한16건이다. Qwen vision attention도 corrected-fixture base에서 입력·출력 bytes까지 같은 실패를 확인했다. Llama RoPE SPLIT 등 내부 grid 전이 손실의 구체적 진단은 `../parity-analysis.md`를 참조한다. tolerance를 늘려 PASS로 바꾸지 않았다.
2. **원래 generic smoke export fixture 준비 부족**: Qwen text model은 prepared_static_export가 없고, Qwen model/conditional은 explicit position_ids 없이 컴파일한다. 해당 guard는 base에도 있다. 실제 dedicated staged export 지원 여부는 CPU integration driver 결과로 구분한다.
3. **Gemma top-level export ABI 혼선**: Gemma causal LM/conditional generation case가 top-level HF forward에 precomputed-embedding tuple을 positional로 넘긴다. causal은 Float32 embedding을 token index로 해석해 실패한다. conditional은 잘못 들어간 pixel_values 경로에서 새 올바른 capability guard가 미관측 vision을 거부한다. base도 잘못된 입력 ABI로 export에 실패한다. `gemma_export_review` 산출물에 원인/보완 실행을 기록한다.
4. **assistant fixture capacity**: core는 valid KV10인데 sliding_kv_length8로 export를 요청한다. candidate/base 모두 같은 명확한 capacity 오류. KV를 자르거나 assertion을 완화하지 않았다. capacity16의 별도 profile 보완 실행 결과도 원래 case와 구분한다. candidate/base 보완 결과 및 실제 도달 단계는 `../gemma-export-review/`의 보고서와 JSON을 참조하며 원래48 case 결과에 합산하지 않는다.
5. **Gemma vision 전체 hash 변화의 오판정**: export_invocation 자체가 독립적인 pooler profile observer6개와 mode owner1개를 새로 생성한다. 기존 weight/observer grid/buffer/mode 변경은0, source/profile frozen 검증은 모두 통과했다. 따라서 최초 전체 topology fingerprint FAIL은 그대로 보존하되 frozen grid 변경으로 해석하지 않는다. 증거 `../gemma4-topology-diagnostic.json`.

## 재실행

원본 numerical 명령은 `../runs/wrapper-numerical-all.json`, 각 export 명령은 `../runs/wrapper-circle-CASE.json`에 cwd/env/timeout/log/원래 exit가 있다. 개별 프로세스 안 original inspector exit와 supplemental 실패 exit도 각각 기록했다.

```bash
cd ${CANDIDATE_REPO}
python ${VALIDATION_ROOT}/run.py --name wrapper-repro-llama --timeout 240 -- \
  python ${VALIDATION_ROOT}/wrapper-harness/export_case.py \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case llama_attention_prefill --device cpu \
  --export circle --strict --no-plot \
  --output-dir ${VALIDATION_ROOT}/repro/llama \
  --report-json ${VALIDATION_ROOT}/repro/llama.json
```

다른 checkout/root에서는 run.py의 cwd와 wrapper-harness script의 root 경로를 맞춘 뒤 새 디렉터리에서 실행한다. 전체 loop는 run_exports.py, base loop는 run_baseline.py, saved-file ABI 재실행은 encoded_replay.py를 사용한다. 원본 로그·실패 배열(.npz)·설정 복사·파일 SHA는 이 검증 root 아래 남겼다.
