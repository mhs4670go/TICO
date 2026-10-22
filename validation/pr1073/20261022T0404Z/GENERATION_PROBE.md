# 실제 VLM image generation 보충 검증

대상: candidate `6b2f6a0511458d6a750e2a260d706080a78149dd`, baseline `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`. pretrained Llama의 normal Quant generate 실패가 발견되어 이 독립 검증을 추가했다. 이전 tiny image-forward/cached-step PASS를 `generate()` 성공으로 일반화하지 않는다.

정상 Transformers5.18.0 `generate()`를 직접 호출했다. 제품 변경, monkeypatch, API bypass, 예외를 성공 처리하는 fallback이 없다. deterministic 저장 PNG와 실제 로컬 HF image processor, 동일 source HF weights, float32, seed1073, 동일 input tensor, greedy/num_beams1/use_cache=True/min_new_tokens2/max_new_tokens2/pad0/eos=None를 사용했다. 두 모델은 random local config이며 pretrained 이미지 benchmark 점수는 아니다.

| 모델 | FP generate | frozen Quant generate | 평가 전후 frozen 상태 |
|---|---|---|---|
| Qwen3-VL candidate | PASS: 2 tokens `[982,579]` | **FAIL**: `get_experts_implementation` 없음 | PASS: raw parameter/buffer/observer/mode/profile 차이 없음 |
| Gemma4 candidate | PASS: 2 tokens `[251,245]` | PASS: 2 tokens `[251,245]` | PASS: raw 차이 없음 |

Gemma FP/Quant token 일치는 관측값이며 요구 assertion이 아니다. Qwen Quant 결과는 생성 완료/metric 완료로 취급하지 않는다. 최초 image prefill은 성공한 뒤 generation decode 준비의 API 요구사항에서 실패했다.

Qwen 실패 위치: `transformers/generation/utils.py:2480`가 `self.get_experts_implementation()`을 호출하며, TICO `tico/quantization/wrapq/wrappers/qwen_vl/quant_for_conditional_generation.py:42`의 GenerationMixin wrapper에는 해당 method가 없다. 이 파일의 git blob은 baseline/candidate 모두 `ceea433c15c26f39e33d977c2161806c0f01772f`다. Gemma conditional wrapper의261-265행은 get/set delegation을 제공한다. 최소 수정안은 Qwen(및 별도 Llama 실패가 확인된 경우 해당 wrapper)에 동등한 get/set delegation을 추가하는 것이지만 **적용하지 않았다**. 이 수정만으로 전체 generation 성공을 단정하지 않으며 수정 후 실제 generation 재검증이 필요하다.

원본 기록: `image_generation_probe.log`, `.command.json`, `image_generation_probe/summary.json`; 모델별 `frozen_full_module.pt`/SHA256, `generation_inputs.pt`, `model_config.json`, `evaluation_protocol.json`, 실제 token `.pt`, 전후 `state.json`.

## 동일 candidate checkpoint의 baseline loading

`qwen_image_generation_baseline`은 같은 Python과 정확히 같은 candidate checkpoint SHA256 `3e3f3a33115fc18420a08e5b2a4894e458f6d87b65fe19b78662c6ba551ca29f`를 읽었다. TICO import는 `/workspace/TICO-export-baseline/tico/__init__.py`로 확인했다.

결과는 exit1 **FAIL**이다. 이 candidate checkpoint에는 legacy `causal_mask_template`가 없어 baseline `quant_text_model.py:208`에서 첫 prefill 중 실패했다. 이 실행으로 baseline의 experts API 실패를 관측했다고 주장하지 않는다. 원본 증거는 `qwen_image_generation_baseline.log`, `.command.json`, `.json`에 별도 보존했다. 후보 checkpoint를 고쳐 baseline에 맞추지 않았다.

## Gemma prepared-static 예외 cleanup의 경계

현재 Gemma wrapper는 `_build_static_export_inputs` owner protocol을 구현하지 않는다. generic `prepared_static_export`(`wrapq/utils/static_export.py:48`,69)는 이 model에서 임시 owner buffer를 만들지 않으므로, 이 context에 예외를 던져도 실제 Gemma 임시 export buffer cleanup 증거가 되지 않는다. 해당 owner cleanup은 **N/A/미검증**으로 유지한다. 기존 Llama/Qwen actual owner의 예외 cleanup PASS와 구분한다.

## Fresh baseline 실제 generation 추가 완료

`qwen_image_generation_fresh_baseline`은 원래 baseline constructor로 legacy mask가 존재하는 정상 wrapper를 만들고 동일 source HF parameter70개(해시 전부 일치), 저장 PNG inputs로 독립 tiny calibration/freeze를 수행했다. FP generate는 동일 `[982,579]` PASS, Quant는 candidate와 동일 experts method 누락으로 실제 FAIL(exit1)했다. 전후 frozen state raw 변경0이다. 따라서 generation API 호환성 오류가 baseline에도 존재함을 실제 실행으로 확인했다. 이것은 동일 frozen checkpoint export-only equivalence가 아니다. 전체 정리와 최소 수정 제안은 `IMAGE_GENERATION_FINDINGS.md`, raw와 원인 분리 JSON은 `image_generation_analysis.json`을 참조한다.
