# Calibration 경로 읽기 전용 감사

대상 HEAD: `6b2f6a0511458d6a750e2a260d706080a78149dd`.
Baseline: `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`.

**결론:** Qwen3-VL의 static-tuple past observer 미수집 위험을 tiny fixture만의 문제로 한정할 근거는 없다. 표준 recipe도 해당 경로를 calibration하지 않는다. 다만 이번 감사는 소스·기존 출력 증거를 읽은 분석이며, 표준 pretrained recipe를 새로 실행해 수치 실패를 재현한 결과가 아니다. 분류는 **기존 표준 recipe의 조건부 static-cache observer coverage 위험**이다. PR이 새로 도입한 pretrained 회귀나 Qwen strict parity 전체 원인으로 확정하지 않는다.

`calibration_path_audit.command.json`의 exit 0은 분석 완료를 뜻하며 모델 검증 PASS가 아니다. 제품/기존 테스트/체크포인트 변경, calibration, export, 모델 다운로드는 수행하지 않았다.

## 기존 실행으로 입증된 사실

`QWEN_PARITY_DIAGNOSTIC.md/.json`은 같은 tiny frozen checkpoint의 baseline, candidate-before, candidate-after_eager에 대해 실제 Circle 실행을 조사했다. 9-stage old/new 및 eager 전후 배열은 exact이나 decoder Torch–Circle strict parity는 실패했다. Full-model prefill+DynamicCache decode calibration을 수행한 checkpoint에서도 `obs_past_key/value`는 min=+inf/max=-inf, frozen scale 약 1e-8이었다. 이 값은 **그 fixture의 qscheme에서 관측한 값**이며 stock int16 recipe의 qparams가 반드시 같은 값이라고 주장하지 않는다.

이 감사에서는 이 기존 실패를 변경하거나 재분류해 PASS로 만들지 않았다. Prefill과 delta-K에도 오차가 있으므로 미수집 past observer만으로 모든 strict parity 실패를 설명할 수 없다.

## 표준 recipe와 실제 observer 경로

| 모델 | 기본 PTQ calibration | Static past observer와의 관계 | 판단 범위 |
|---|---|---|---|
| Llama | `decode_steps`/`decode_calibration_steps`가 양수면 full-model cached prefill 뒤 manual decode loop. `llama_quantize.yaml`은 8회. | HF Cache에서도 K/V를 추출하며 `obs_past_key/value`를 적용하므로 실제 decode가 있으면 tuple past와 같은 observer를 방문한다. 별도 per-layer static-tuple calibration loop는 아니다. | 양수 decode 경로는 Qwen의 동일한 분리 문제와 다르다. decode_steps=0의 prefill-only 설정에서는 past 통계 수집을 보장하지 않는다. |
| Qwen3-VL | 로드 시 cache를 끄고 processor batch를 각각 full-model에 한 번 전달한다. 표준 adapter에는 decode loop나 static-tuple replay가 없다. | `obs_past_key/value`는 per-layer tuple/list에서만 방문한다. HF Cache 경로는 update 뒤 `obs_present_key/value`만 방문한다. | 기본 prefill뿐 아니라 DynamicCache decode를 추가해도 distinct static-past observer 수집을 보장하지 않는다. |
| Gemma4 | 로드 시 cache를 끄고 processor batch를 각각 한 번 전달한다. 표준 adapter에 별도 decode loop가 없다. | 별도 `obs_past_key/value`가 없고 tuple past, HF Cache update, prefill이 `obs_present_key/value`를 공통 사용한다. Shared-KV도 present observer를 사용한다. | Qwen과 똑같은 distinct-past observer 미수집 구조는 없다. 이것이 모든 cache 분포·sliding/shared-KV profile 또는 Circle 수치 parity의 충분한 calibration 증거는 아니다. |

## 소스 근거

- 공통 PTQ stage는 adapter calibration 직후 convert를 호출한다. 추가 static decode calibration이나 cache qparams 보정 단계가 없다: `tico/quantization/recipes/stages/ptq.py:214`, `:216`.
- Qwen cache 비활성화: `tico/quantization/recipes/adapters/qwen3_vl.py:141`, `:147`; calibration은 batch별 `model(**move_to_device(...))`뿐이다: `:171`, `:183`, `:186`.
- VLM calibration builder는 processor 출력을 반환한다: `recipes/data/vlm.py:81`; 실제 image-question 입력을 append하는 곳은 `tico/quantization/evaluation/vlm_eval_utils.py:1627`; text-only calibration dict는 `:1712`의 `{"input_ids": sample_ids}`다. 여기에는 decoder의 과거 KV를 생성·연결·재생하는 과정이 없다. 외부에서 직접 만든 calibration batch는 이 기본 경로와 별개다.
- Qwen full text model은 cache를 요청하고 입력 cache가 없을 때 HF DynamicCache를 생성한다: `wrapq/wrappers/qwen_vl/quant_text_model.py:530`. Full-model은 그 동일 객체를 각 decoder에 전달한다: `:619`, `:624`.
- Qwen past observer는 tuple/list로 인식되는 입력에만 적용된다: `wrapq/wrappers/qwen_vl/quant_text_attention.py:392`, `:397`, `:404`. HF Cache update 분기는 present observer만 사용한다: `:531`, `:542`. 새로운 K/V와 present는 prefill에서도 관측된다: `:546`, `:763`.
- 반면 실제 Qwen per-layer export decode는 `(past_key, past_value)` tuple을 전달한다: `recipes/export/qwen3_vl.py:454`, `:461`, `:479`. 따라서 calibration과 export가 서로 다른 observer 경로를 방문할 수 있다.
- Stock Qwen recipe의 activation 기본값은 int16이다: `recipes/adapters/qwen3_vl.py:217`; past는 weight/bias가 아니므로 activation role을 상속한다: `config/ptq.py:38`, `:297`. Qwen builder의 기본 layer overrides는 projection/norm 정책이며 cache observer 공유나 미수집 보완을 하지 않는다: `config/builders.py:322`, `:510`. 실제 override로 비양자화 정책 또는 별도 qparams를 지정한 경우에는 조건을 다시 확인해야 한다.
- Llama 실제 cached calibration loop: `recipes/adapters/llama.py:253`, `:278`, `:284`; Cache 객체도 past observer를 통과하는 추출 경로: `wrapq/wrappers/llama/quant_attention.py:480`, `:508`, `:332`. `decode_steps=0`이면 prefill 한 번으로 끝난다: `adapters/llama.py:274`. 기본 export 지향 preset은 `examples/configs/llama_quantize.yaml:23`, `:65`; prefill-only 예시는 `llama_ptq_override_policies.yaml:23`이다.
- Gemma cache 비활성화/일회 forward: `recipes/adapters/gemma4.py:168`, `:201`, `:315`; tuple past도 present observer를 쓰는 코드: `wrapq/wrappers/gemma4/quant_text_attention.py:475`, `:480`, `:492`; shared-KV도 `:790`의 present observer를 사용한다.
- Freeze는 직접 소유 observer 모두에 `compute_qparams()`를 호출하며 이곳에 수집 여부 검사는 없다: `wrapq/wrappers/quant_module_base.py:114`. Affine observer의 초기 통계는 +inf/-inf이고 `compute_qparams` 결과는 qscheme에 의존한다: `wrapq/observers/affine_base.py:52`, `:115`. 따라서 **frozen/scale 필드 존재만으로 calibration coverage 완료를 판단할 수 없다.**
- GPTQ qparam handoff는 기본적으로 weight observer를 대상으로 하므로 cache activation 통계의 대체 증거가 아니다: `recipes/qparams.py:215`, `:218`, `:233`.

## Baseline과의 관계

`inspect_calibration_paths.py`를 정상 Python stdlib만으로 실행해 양쪽의 16개 관련 함수 소스를 비교했다. 모두 AST와 추출된 함수 source가 동일했다. Qwen의 cache 비활성화/calibration/data 전달/config builder 및 static-vs-HF cache observer 분기, Llama calibration/cache 추출, Gemma calibration/cache observer 공유, 공통 PTQ stage가 포함된다. 결과와 줄별 source excerpt는 `CALIBRATION_PATH_AUDIT_EVIDENCE.json`에 있다.

따라서 이 observer 경로 분리는 pre-PR baseline에도 존재한다. PR의 eager context·RoPE 변경이 이를 새로 만들었다는 근거는 없다. 기존 Qwen old/new 출력 exact와도 모순되지 않지만, 그 사실이 strict Torch–Circle 실패를 허용하는 근거는 아니다.

## 남은 재현과 미적용 제안

1. 표준 adapter/PTQStage 경로로 작은 실제 HF Qwen 모델과 고정 in-memory prompt/image batch를 사용해 freeze 직전 `past/new/present` observer의 수집 여부, min/max, qscheme와 dtype을 기록한다. 데이터·모델 다운로드는 필요 없는 형태로 구성할 수 있다. 이 추가 실험은 이번 읽기 전용 감사에서 실행하지 않았다.
2. 동일 생성 코드/seed/입력으로 별도 **새 fixture**에 실제 per-layer tuple past를 calibration하고, 독립된 파일/해시를 부여한다. 기존 frozen 파일을 수정하거나 기존 export-only 비교와 같은 checkpoint로 취급하면 안 된다.
3. 원래 허용 오차 `atol=1e-5`, `rtol=1e-4`로 prefill·empty-past·populated-past와 delta-K/V를 다시 실제 Torch/Circle 비교한다. Past 통계 보완이 populated-cache 차이에 미치는 영향과, past 없는 prefill/delta-K 차이의 원인을 분리한다.
4. 실 pretrained 결과를 판정하려면 해당 revision/profile의 신뢰할 수 있는 frozen full-module checkpoint와 생성/calibration provenance를 확인하고 같은 통계 검사를 한다. 이번에는 이 pretrained recipe 검증을 수행하지 않았으므로 pretrained 품질 실패나 성공을 선언하지 않는다.

미적용 최소 개선 방향은 freeze/export 전에 **실제로 요청될 경로의 활성 quant observer coverage를 검사하고 미수집 시 명확히 거부**하는 것이다. 제품 설계상 수집 경로를 정리할 경우에는 (a) HF Cache와 tuple이 같은 semantic boundary observer를 방문하도록 정합화하거나, (b) 실제 static-tuple decode calibration을 명시적으로 수행하는 방법을 검토할 수 있다. 서로 다른 qscheme/bit-width/override에 대한 호환성을 입증하지 않고 past/present qparams를 복사하거나 checkpoint를 조용히 재양자화하는 수정은 제안하지 않는다. 어느 방법도 이 감사에서 적용하지 않았다.
