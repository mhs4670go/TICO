# 축소 evaluation 실행 결과

결론: 실제 CPU FP/Quant public evaluation 경로와 checkpoint round trip은 PASS. 외부 Hugging Face HellaSwag 데이터 접근은 BLOCKED(프록시 CONNECT HTTP 403). 아래 객관식은 **4행 synthetic HellaSwag 형식 fixture**이며 실제 benchmark 데이터 점수 또는 pretrained 품질 비교가 아니다.

## 실제 코드와 입력

- candidate `9f631d6651ce3780e3a8e054ab02910b9bd40fdc`, import `${CANDIDATE_REPO}/tico/...`; source 수정 없음.
- 실제 Transformers LlamaForCausalLM: vocab128, hidden64, 2 layers, 2 attention/1 KV heads, head_dim32, 90,432 parameters, float32, CPU, seed1073.
- `cpu_fixtures.fixture_context` + 실제 `PTQStage.run`의 prepare → calibration(2개×8 tokens, 각 2 decode steps) → convert; `torch.save` / `torch.load(weights_only=False)` → `validate_frozen_model`; 저장 전후 `_state_contract` 정확히 동일.
- checkpoint `quantized_model.pt`, SHA256 `2dca60284360a76bd0229a7dd73db1ad35919310ed48c7a28db81ddfa7235d32`; 생성 방법 `prepare.py`.
- calibration IDs5..12/21..28, 평가 문장 IDs80..107로 분리. 평가 데이터로 observer 업데이트하지 않음.
- 기존 environment Torch2.13.0+cpu/Transformers5.18.0/datasets5.0.1 유지. 선택 평가 패키지에는 repo pin이 없으므로 stable `lm-eval[hf]==0.4.13`을 기존 전체 version constraints로 추가(기존 패키지 upgrade 없음). accelerate1.15.0/peft0.21.2 포함. 설치 후 `pip check` PASS.

## 결과

| 실행 이름(run.py --name) | 실제 경로·입력 | 실제 처리·수치 | 결과/시간 |
|---|---|---|---|
| evaluation-prepare | 실제 PTQ stage와 checkpoint round trip | 2×8 calibration + 2 decode steps, 90,432 params; frozen exact equality | PASS 8.201s |
| evaluation-cli-fp-ppl-final | public evaluate.py --tasks ppl, 실제 Llama adapter/helper/datasets/scoring | 로컬 JSON4행, 28 input / 27 유효 scoring tokens; 3 forwards; PPL127.8993911743164 | PASS 6.201s |
| evaluation-cli-quant-ppl-final | 동일 CLI에 실제 converted --checkpoint 지정 | 동일27 scoring tokens/3 forwards; PPL128.04383850097656 | PASS 7.332s |
| evaluation-cli-fp-mc-final | public --tasks lm_eval → 실제 evaluate_lm_tasks/evaluate_llm_on_tasks/HFLM/simple_evaluate | synthetic HellaSwag 형식4 examples×4 choices=16 requests/forwards,32 continuation tokens; acc=.25/acc_norm=.25 | PASS 13.257s |
| evaluation-cli-quant-mc-final | 동일 경로·동일 fixture·실제 converted checkpoint | 동일16 requests/forwards,32 scoring tokens; acc=.25/acc_norm=.25 | PASS 21.621s |
| evaluation-verify-results | 실제 저장한 evaluator sample 결과와 HFLM token encoding 검증 | 모든 likelihood 유한; task별 sample 평균==summary; FP/Quant forward 입력 hash 목록 정확히 일치 | PASS |
| external-benchmark-access | curl -I --max-time15 https://huggingface.co/datasets/Rowan/hellaswag | CONNECT proxy403, exit56 | BLOCKED |

PPL window16/stride8/context64, batch1. PPL hook이 실제 labels에서 15+8+4=27 scoring tokens를 기록했다. MC에서는 모델에 labels를 넘기지 않으므로 forward hook의 `valid_scoring_tokens=0`은 NLL target 수가 아니다. `verify_results.py`가 저장된 **실제 evaluator sample.arguments**를 같은 genuine `HFLM._encode_pair`로 인코딩하여 16개 continuation×2=32를 확인했다. 모델은 이미 실제 16회 forward를 수행했고 sample.resps는 실제 loglikelihood다.

Quant 평가 전후 frozen contract SHA256 `89f5e837cf2efdc19ad98b6ab7d8e43b0a45d2565254c00378fcc3861b1101c4` 동일. FP는 tensor state_dict digest로 비변경을 확인했다. FP와 Quant 결과/프로토콜 디렉터리가 분리되며 evaluator `use_cache=None`, `cache_requests=False`다. 각 새 프로세스마다 HFLM을 새로 만들며 응답 cache를 공유하지 않는다. seed/데이터/task/토크나이저/context/fewshot0/batch1가 같다.

## harness 경계

`invoke_cli.py`는 `runpy.run_module(...evaluate,run_name='__main__')`로 실제 public CLI를 실행한다. public 옵션·config loader·model/tokenizer loader·--checkpoint loader/validation·adapter/eager policy·평가 helper 모두 원본 코드다. read-only forward hook, metric 반환값 기록, 평가 전후 frozen 검사만 추가한다.

PPL은 loader patch 없이 config의 local JSON directory를 `datasets.load_dataset`으로 직접 읽는다. MC에서는 public Llama CLI에 sample cap이 없으므로 실제 `simple_evaluate` 호출에 local YAML `TaskManager`, limit4, num_fewshot0, seed1073, cache off, bootstrap_iters0을 더해 원래 함수를 호출했다. YAML은 외부 `Rowan/hellaswag` 대신4행 JSON을 지정하고 기존 `lm_eval.tasks.hellaswag.utils.process_docs`, prompt/choices/target, 실제 evaluator metric aggregation을 사용한다. task key `hellaswag`를 로컬에서 덮어쓴 사실은 로그에 명시되어 있다. production HellaSwag metric parser가 실제 evaluator의 `acc,none`/`acc_norm,none`를 읽는 것도 확인했다. 모델 forward·loglikelihood·metric 계산은 mock하지 않았다. 변경 없는 CLI 단독 사용으로 sample-limit가 전달되었다고 주장하지 않는다.

기존 CPU driver에 포함된 실제 Llama/Qwen/Gemma PPL와 generation은 root 결과를 재사용한다. 이 추가 실행은 Llama public CLI/checkpoint/lm-eval gap을 보완한다. 나머지 VLM public benchmark CLI, 외부 실제 HellaSwag/WikiText, 전체 pretrained 품질, NPU/GPU 성능·integer runtime은 검증하지 않았다. tiny random-model 점수로 품질 동등성 또는 회귀를 판단하지 않는다.

최초 FP 시도 2건은 tracing harness의 `_state_contract` FP 오용 때문에 평가 전에 실패했으며 제품 문제가 아니다. 원본 로그와 `HARNESS_CORRECTIONS.md`에 최소 재현·수정 이력을 보존했다. 수정한 harness의 final4회 결과로 판정한다.

## 재현·증거

- 전체 실행 명령/exit/시간: 상위 `runs/evaluation-*.json`; 로그: `logs/evaluation-*.log`.
- 기계 판독 요약: `evaluation/summary.json`; 모델 생성/체크포인트 경계: `fixture-evidence.json`.
- 각 `*-final/evidence.json`: 실제 forward input SHA256, scoring count, metric, state hash, checkpoint, import path.
- 각 `*-mc-final/lm_eval_results.json`: 실제 task config, samples/resps, filtered metric key, summary, evaluator version/seed.
- `*-final/protocol/evaluation_protocol.json`은 실행설정 기록이며 metric 완료 증거와 별도로 유지.
- 소스/fixture: `prepare.py`, `invoke_cli.py`, `verify_results.py`, `config.yaml`, `task-configs/*`, `mc.jsonl`, `ppl-data/test.jsonl`.
- 재실행: `bash ${VALIDATION_ROOT}/evaluation/reproduce.sh` (새 reproduce-* 로그·결과 생성).
