# 재실행

검증 revision은 `revisions.json`에 고정돼 있다. 원본 checkout `${BASELINE_REPO}`는 base를 유지하며, `${CANDIDATE_REPO}`는 detached head다. `run.py`는 원래 child exit code, timeout, 실행시간, 환경, cwd, stdout/stderr 로그를 기록한다. `COMMANDS.sh`/`commands.json`에는 실제 실행한 모든 명령이 있다. 도움말/수집 명령은 실제 실행 coverage에 포함하지 않는다.

현재 환경 재사용:

```bash
source ${VALIDATION_VENV}/bin/activate
cd ${CANDIDATE_REPO}
git rev-parse HEAD HEAD^{tree}
# 9f631d6651ce3780e3a8e054ab02910b9bd40fdc
# c81b5f254023b21bb76f1f6d04405b5c7d3f54b2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' CCEX_RUNTIME=reference PYTHONPATH="$PWD"
python -c 'import tico; print(tico.__file__)'
```

새 환경에서는 `source-context/docs/development.md`, `docs/torch_version_policy.md`와 현재 revision의 installer를 따른다. 이번 기본 환경은 이미 구성되어 있었으므로 재설치하지 않았다. 선택적 lm-eval만 `lm-eval[hf]==0.4.13`을 `evaluation/constraints-existing.txt`로 기존 버전을 고정해 설치했다. 처음 plain lm-eval에는 accelerate가 없어 hf extra를 추가한 이력도 원본 로그에 있다.

```bash
./ccex install --cpu_only
./ccex configure test --cpu_only
./ccex configure format
python -m pip install 'lm-eval[hf]==0.4.13' -c ${VALIDATION_ROOT}/evaluation/constraints-existing.txt
python -m pip check
```

## 기본 suite와 원본 CLI

반드시 새 출력 경로를 사용한다. 아래의 ROOT는 이번 산출물과 다른 디렉터리다.

```bash
ROOT=$(mktemp -d -t pr1097-validation.XXXXXXXX)
mkdir -p "$ROOT"
set -o pipefail
./ccex test -k frozen 2>&1 | tee "$ROOT/focused.log"
./ccex test 2>&1 | tee "$ROOT/full-suite.log"
./ccex format --no-apply-patches 2>&1 | tee "$ROOT/format.log"
python -m tico.quantization.examples.inspector --mode wrapper-smoke --list-cases
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case all --device cpu --export none --strict --no-plot \
  --output-dir "$ROOT/wrapper-smoke" 2>&1 | tee "$ROOT/wrapper.log"
python -m tico.quantization.recipes.debug.cpu_validation \
  --stage all --output "$ROOT/cpu-validation" 2>&1 | tee "$ROOT/cpu-validation.log"
```

`cpu_validation --output`은 **아직 없는 디렉터리**여야 한다. 원본 결과 디렉터리를 재사용하면 의도적으로 거부된다. 이 드라이버는 baseline 비교 모드가 아니라 candidate 내부 검증이다.

## Circle 전체 지원 case 재실행

`wrapper-harness/registry.json`을 현재 source registry와 대조한다. 일반 export 미지원 case를 제외한 case별 실제 명령은 `runs/wrapper-circle-*.json`과 `COMMANDS.sh`에 있다. `export_case.py`는 inspector.main에 동일 CLI 인자를 전달하고 실제 export/convert를 그대로 호출하며, 관측용 spy로 정확한 입력/출력을 확보해 저장 파일의 구조/reference/parity/frozen 비변경을 검사한다.

```bash
python ${VALIDATION_ROOT}/wrapper-harness/export_case.py \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case llama_attention_prefill --device cpu \
  --export circle --strict --no-plot \
  --output-dir "$ROOT/circle/llama_attention_prefill" \
  --report-json "$ROOT/circle/llama_attention_prefill.json"
```

전체 export orchestration은 `wrapper-harness/run_exports.py`다. 새 전체 결과를 만들려면 이 script와 run.py를 새 ROOT로 복사하고 script의 root 상수만 새 경로로 바꾼다. 실제 제품/모델/검증 assertion은 바꾸지 않는다. 또는 기록된 개별 명령의 output 경로를 새 ROOT로 지정한다. 원본 CLI만 실행하려면 `export_case.py` 대신 `-m tico.quantization.examples.inspector`를 쓰되 이는 파일 저장까지만 검증하며 reference/parity evidence를 새로 만들지는 않는다.

## 소량 평가/checkpoint

`evaluation/reproduce.sh`는 fixture 생성 후 public PPL/MC 평가를 새 `reproduce-*` 이름으로 실행한다. 두 번째 같은 재실행에는 새 label을 쓴다. 외부 HF task를 받은 것이 아니라 실제 evaluator에 로컬 synthetic HellaSwag 형식4행만 연결한 것이다. production model forward/metric 계산은 교체하지 않는다.

```bash
bash ${VALIDATION_ROOT}/evaluation/reproduce.sh
```

설정/입력: `evaluation/config.yaml`, `ppl-data/test.jsonl`, `mc.jsonl`, `task-configs/hellaswag.yaml`. 모델은 `evaluation/prepare.py`가 seed1073으로 생성·PTQ·저장·복원한다. 모델/토크나이저/체크포인트는 생성물이며 소스 commit 대상이 아니다. 기존 artifact hash는 `evaluation/fixture-evidence.json`과 최종 artifact manifest에 있다.

## 축소 artifact benchmark

```bash
python -m test.performance.benchmark_conversion_memory \
  --sizes-mib 1 --tables 1 --width 16 --threads 1 --repeat 1 \
  --optimization O1 --input-kind module --workdir "$ROOT/benchmark-module" \
  --keep-outputs --json > "$ROOT/conversion-module.json"
python -m test.performance.benchmark_conversion_memory \
  --sizes-mib 1 --tables 1 --width 16 --threads 1 --repeat 1 \
  --optimization O1 --input-kind pt2 --workdir "$ROOT/benchmark-pt2" \
  --keep-outputs --json > "$ROOT/conversion-pt2.json"
python -m test.performance.benchmark_circle_optimizer \
  ${VALIDATION_ROOT}/wrapper-circle/llama_attention_decode/llama_attention_decode.q.circle \
  --repeat 1 --warmup 0 --json > "$ROOT/optimizer.json"
```

Baseline은 `${BASELINE_REPO}`에서 `PYTHONPATH=${BASELINE_REPO}`로 실행한다. 먼저 `tico.__file__`을 확인한다. `baseline-selective-summary.json`과 해당 run record에 정확한 비교 명령이 있다. 각 revision에서 모델을 독립 생성했으며 새 frozen checkpoint를 구버전에 주입하지 않았다. 새로운 실패를 비교할 때도 동일 원칙을 적용한다.

추가 실패 비교: `baseline-cpu-comparison/commands.json`과 `baseline_cpu_queue.py`는 base 자체 fixture를 만들어 Llama16/Qwen16/Qwen32 및 saved Qwen connected16/32를 순차 실행한다. head 체크포인트를 base에 전달하지 않는다. assistant 추가 용량 검증은 `gemma-export-review/assistant_capacity16.py`와 해당 `runs/assistant-capacity16-circle-r2*.json`을 참조한다. 초기 helper import 오류는 coverage가 아니며 원본 로그와 구분한다.
