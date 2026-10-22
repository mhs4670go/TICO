# 실제 image generate 결과와 원인

검증 SHA: candidate `6b2f6a0511458d6a750e2a260d706080a78149dd`, baseline `bb09ca852b0ca1b1a6271cfe1e10c162104c997e`.

| 대상 | FP normal generate | Quant normal generate | frozen state |
|---|---|---|---|
| Qwen3-VL candidate | PASS: 추가2 tokens `[982,579]` | **FAIL**: `get_experts_implementation` 없음 | PASS: raw 변경0 |
| Qwen3-VL fresh baseline | PASS: 동일 `[982,579]` | **FAIL**: 동일 API 오류 | PASS: raw 변경0 |
| Gemma4 candidate | PASS: 추가2 tokens `[251,245]` | PASS: 추가2 tokens `[251,245]` | PASS: raw 변경0 |

위 모델은 random local HF config이다. 실제 저장 PNG→로컬 생성한 실제 HF image processor→vision/text `generate()`를 정상 호출했다. pretrained 이미지 benchmark나 자연 이미지 품질 점수는 아니다. Qwen은 약202만 parameter, Gemma는 약6723만 parameter이다. Gemma는 작은 hidden/layer config이지만 기본 PLE lookup table이 커서 frozen fixture는 약261MiB이며, 가용 RAM 내에서 순차 실행했다.

FP/Quant는 같은 source HF weights·PNG processor tensor·float32·seed1073·prompt token IDs·greedy·beam1·KV-cache on·max/min_new_tokens2·pad0·eosNone를 사용한다. 실제 token 파일, input tensors, full-module checkpoint SHA256, model config와 protocol record가 모델별 디렉터리에 있다. FP/Quant token 동등성을 통과 조건으로 요구하지 않았다. Gemma의 동일 tokens는 관측값이다.

원래 `image_generation_probe`의 종료코드는 **1**이다. Qwen 실패를 Gemma PASS나 앞선 image forward/cached-step PASS로 덮어쓰지 않았다. frozen 비교는 parameter, buffer, observer 통계/qparams/enabled, quant mode/profile을 포함하며 두 candidate와 fresh baseline 모두 raw 차이가 없다.

## 재현되는 원인

Transformers5.18.0 `transformers/generation/utils.py:2480`는 decode 준비 중 `self.get_experts_implementation()`을 호출한다. `tico/quantization/wrapq/wrappers/qwen_vl/quant_for_conditional_generation.py:42`의 GenerationMixin wrapper에는 해당 method가 없다. Qwen의 image prefill은 진행됐지만 생성 loop가 두 token을 완료하지 못한다.

이 파일의 baseline/candidate git blob은 모두 `ceea433c15c26f39e33d977c2161806c0f01772f`다. 아래 fresh baseline 실제 실행에서도 동일 오류가 재현됐으므로, 이번 PR 이전부터 존재한 generation API 호환성 실패로 분류한다. raw 결과는 양쪽 **FAIL**로 유지한다.

최소 제품 수정 제안은 Gemma conditional wrapper `tico/quantization/wrapq/wrappers/gemma4/quant_for_conditional_generation.py:261`의 방식으로 다음 delegation을 Qwen wrapper에 제공하는 것이다. Llama에서 같은 오류가 관측된 경우 해당 wrapper에도 같은 검토가 필요하다. **제품 코드는 수정하지 않았다.** 이 제안 이후의 추가 호환성 문제 유무는 실제 generate 재실행으로 확인해야 한다.

```python
def get_experts_implementation(self):
    return self.module.get_experts_implementation()

def set_experts_implementation(self, experts_implementation):
    return self.module.set_experts_implementation(experts_implementation)
```

## 두 baseline 시도를 분리한 증거

1. `qwen_image_generation_baseline`: candidate가 만든 정확히 같은 full-module checkpoint를 원래 baseline source로 읽어 정상 generate를 호출했다. TICO import는 `/workspace/TICO-export-baseline/tico/__init__.py`, checkpoint SHA256은 candidate와 동일 `3e3f3a33115fc18420a08e5b2a4894e458f6d87b65fe19b78662c6ba551ca29f`다. 그러나 candidate checkpoint에는 legacy static mask가 없어 baseline `quant_text_model.py:208`에서 `causal_mask_template` 누락으로 **먼저 실패**했다. 이 시도로 experts 오류를 확인했다고 주장하지 않는다. 이 후보 checkpoint를 수정해 baseline에 맞추지 않았다.
2. `qwen_image_generation_fresh_baseline`: 같은 source HF weights와 저장 PNG inputs를 사용하되, **원래 baseline constructor로 새 wrapper**를 만들었다. 진짜 legacy mask buffers 세 개가 존재함을 확인하고, 같은 작은 image sample로 독립 calibration/freeze 후 정상 generate했다. FP PASS, Quant 동일 experts API **FAIL**을 실제 확인했다. candidate의 source HF parameter70개 해시가 baseline source HF parameter70개와 정확히 같고 FP tokens도 같다. 평가 전후 모든 frozen 상태는 정확히 보존됐다.

두 번째 실행은 독립적인 tiny baseline PTQ 재현이며 **동일 frozen checkpoint export-only 비교가 아니다**. 이 실행은 preexisting generation API 오류를 확인하기 위한 것이며 Circle graph/export equivalence나 pretrained Quant 품질 비교를 입증하지 않는다.

## 원본 증거

- 분석 JSON: `image_generation_analysis.json` (raw 결과와 원인 분류 분리)
- Candidate: `image_generation_probe.log`, `.command.json`, `image_generation_probe/summary.json`
- Candidate 모델별: `generation_inputs.pt`, `frozen_full_module.pt`, `model_config.json`, `state.json`, FP/Quant token `.pt`, FP/Quant `evaluation_protocol.json`
- 같은 candidate checkpoint의 baseline 시도: `qwen_image_generation_baseline.log`, `.command.json`, `.json`
- 독립 fresh baseline 실제 재현: `qwen_image_generation_fresh_baseline.log`, `.command.json`, `qwen_image_generation_fresh_baseline/summary.json`, `/state.json`
- 실행 코드: `image_generation_probe.py`, `image_generation_baseline.py`, `qwen_generation_fresh_baseline.py`
- 재현 명령: `commands.sh`의 위 세 실행 항목

Gemma wrapper에는 `_build_static_export_inputs` owner가 없어 generic prepared-static context의 임시 owner buffer cleanup 검증은 N/A/미검증으로 유지했다. no-op context의 예외 복구를 실제 Gemma 임시 export state 검증으로 세지 않았다.
