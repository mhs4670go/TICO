# Tiny fixture 생성 인프라 수정

첫 Llama fixture는 full `QuantLlamaForCausalLM`을 직접 저장했고 frozen QUANT·CPU float32 조건을 만족했다. 하지만 실제 recipe exporter는 최상위 `PTQWrapper`를 요구한다 (`recipes/export/llama.py:167`의 `model.wrapped.model` 검사). 첫 old/new 실행은 baseline에서 중복 wrapping을 시도하여 `wrap_helper.py:50`의 `AssertionError: The module is already wrapped.`로 종료했다.

- 원본 creator: `create_tiny_frozen_fixture.v1.py`
- 원본 체크포인트: `fixtures/baseline_llama.pt`
- 원본 실패/exit 2: `tiny_llama_old_new.command.json`, `tiny-llama-old-new/summary.json`, `tiny-llama-old-new/baseline.log`
- 최소 수정: 원본 HF model을 직접 Quant 클래스 대신 정상 `PTQWrapper(fp_model, qcfg=...)`로 감싼 후 동일한 calibration/freeze 절차를 수행한다. 제품이나 baseline source는 수정하지 않았다.
- 수정 후 별도 생성 체크포인트: `fixtures/baseline_llama_v2.pt` (새 파일), 생성 명령/로그 `create_baseline_llama_fixture_v2.*`
- 재실행 결과: `tiny-llama-old-new-v2/` (새 output 경로).

각 checkpoint의 생성 코드 SHA256, 출처 checkout SHA, dtype, native capacity, calibration 길이, quantization/vision profile, 전체 HF config 및 checkpoint SHA256은 해당 `.provenance.json`에 저장했다. Tiny random fixture이며 pretrained 품질 검증이 아니다.

추가 revision 및 독립 validation case:

- Qwen 첫 생성은 baseline wrapper가 `vision.grid_thw`의 tuple 형식을 요구하여 실패했다 (`quant_text_model.py:464`). creator에서 tuple로 넘기도록 최소 수정하고 새 `baseline_qwen_v2.pt`를 생성했다. 원본 `create_baseline_qwen_fixture.log`/exit 1과 v2 소스를 보존했다. Export YAML은 정상 list로 직렬화되고 exporter가 profile tuple로 정규화한다.
- 초기 Llama/Qwen fixture는 cache 없는 prefill calibration만 수행했다. Llama 실제 populated-past Torch/Circle 오차에 unobserved past observer가 포함됨을 `PARITY_DIAGNOSIS.md`에서 확인했다. 원본 Llama v2 strict FAIL은 유지한다. NEW creator revision은 freeze 전에 cache-bearing prefill4 + teacher-forced decode2를 calibration한다. Qwen/Gemma의 새 `*_cached.pt`를 양 worker가 동일하게 읽으며 worker는 calibration하지 않는다.
- Gemma opt-in `validate_static_layout=True` case는 실제 `[4,32]` vision feature의 `shape[1]`을 token 수로 읽어 32-vs-4로 실패했다. 해당 제품 코드는 baseline/candidate에 동일하다 (`quant_model.py:229` 전후). 원본 `create_baseline_gemma_fixture.log`/exit 1을 유지한다. 이 case는 PASS로 바꾸지 않는다.
- 별도 Gemma documented-default `validate_static_layout=False` profile에서는 canonical `[x,y]` patch positions를 사용하고, `Gemma4StaticVisionProfile.validate_processor_outputs`, `validate_wrapped_model`, 실제 feature `shape[-2]==4`/`shape[-1]==32`를 명시적으로 검사했다. 위치 순서·grid·placeholder placement·checkpoint profile을 생략하지 않는다. 검사 결과는 `baseline_gemma_cached.provenance.json`의 calibration/manual_vision_contract에 저장했다. 이는 True case의 재분류가 아닌 다른 profile의 독립 실행이다.
- Llama original full-only runner는 baseline adapter가 요구하는 `ctx.calibration_inputs` shape example을 만들지 않아 ERROR/exit 2로 실패했다 (`tiny-llama-full-original/`). `export_regression_full_shape_fixture.py`는 원본 runner 복사본에 Llama `circle_full`일 때만 CPU long zeros shape input을 추가한다. 실제 calibration/observer update는 수행하지 않으며 모든 원래 state fingerprint, strict comparison 및 tolerance를 유지한다. Diff와 원본/복사본 SHA256은 같은 이름의 `.diff`/`.provenance.json`에 저장했다. Baseline 및 제품 소스는 수정하지 않았다.

- Qwen full-text-only original run (`--artifacts '' --include-full-text`)은 실제 conversion/reference 실행 후 `full_text_regression.q.circle` 저장 시 artifact directory가 없어 `FileNotFoundError`로 ERROR/exit2 종료했다 (`tiny-qwen-full-text-png-original/`). `export_regression_full_text_dirs.py`는 원본 복사본의 `_export_full_text_probe` 안에 `path.mkdir(parents=True, exist_ok=True)` 한 줄만 추가한다. 원래 output 경로를 미리 생성하지 않았으며 worker가 선택한 새 phase 안의 저장 디렉터리를 준비한다. 원본/복사본 diff와 SHA256 보존, 원본 comparator/state checks/수치 기준은 변경하지 않았다. 재실행은 `tiny-qwen-full-text-png-dirs-retry/`에 분리한다.

- Gemma full-text-only probe의 directory 수정본 첫 실행은 baseline `QuantGemma4TextModel`의 필수 static mask dict를 제공하지 않아 `NotImplementedError`로 ERROR/exit2 종료했다 (`tiny-gemma-full-text-png-dirs/`). 이 필수 입력 거부는 유지하며 정상 계약이다. 새로운 `export_regression_gemma_full_text_masks.py` 복사본은 checkpoint의 기록된 full/sliding mask template을 선택 seq8로 **trace 전에** 읽어 registered buffers로 복사하고 `attention_mask` dict를 전달한다. Custom position IDs를 전달하거나 live eager RoPE를 호출하지 않는다. 원래 제품 소스·checkpoint·qparams·strict 기준·runtime은 수정하지 않았다. Original/parent/copy diff와 SHA256, host mask shape/dtype/hash/NPZ는 새 결과 경로에 저장했다. 재실행은 `tiny-gemma-full-text-png-masks-retry/`로 분리한다.
