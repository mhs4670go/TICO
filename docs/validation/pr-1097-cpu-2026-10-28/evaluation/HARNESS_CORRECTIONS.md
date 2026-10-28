# 검증 harness 수정 이력

- 최초 FP CLI run `evaluation-cli-fp-ppl`, `evaluation-cli-fp-mc`는 실제 평가에 들어가기 전 tracing harness의 잘못된 전제 때문에 실패했다. `export_regression._state_contract()`는 frozen quantized checkpoint 전용인데 FP 모델에도 호출했다. 제품 오류가 아니다.
- 최초 버전의 해당 두 문장은 `before = _state_contract(ctx.model)` / `after = _state_contract(ctx.model)`였다. 현재는 checkpoint 사용 시에만 원래 함수로 frozen grid와 observer 상태를 비교하고, FP는 tensor state_dict byte digest를 비교한다. `invoke_cli.py`의 원래 공개 CLI/model/evaluator 흐름은 수정하지 않았다.
- 최소 재현: `PYTHONPATH=${CANDIDATE_REPO} ${VALIDATION_VENV}/bin/python -c 'from tico.quantization.recipes.debug.export_regression import _state_contract; from tico.quantization.recipes.debug.cpu_fixtures import llama_model; _state_contract(llama_model())'`는 의도대로 `ValueError: This regression runner requires a frozen quantized checkpoint.`를 발생시킨다.
- 후속 재실행 `*-final`에서는 PPL hook을 실제 production child 한 곳에만 달아 중복 logging을 피하고 실제 labels의 유효 scoring token 수를 직접 기록했다. 최초 Quant PPL/MC는 이미 실제 metric 계산을 성공했지만 최종 보고에는 final evidence를 사용한다.
- assertion/tolerance/skip 또는 제품 코드는 변경하지 않았다.
