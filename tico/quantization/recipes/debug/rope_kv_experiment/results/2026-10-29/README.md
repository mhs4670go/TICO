# Gemma4 RoPE/KV 2×2 결과

기준은 `54411f0ebf4782988193f1b40beab9d82c0f830d`, 실제 측정 소스는
`279ea8cedc1e8c52b69d5341f6614f760b1171e8`이다. 실험 시작 working tree는
깨끗했다. upstream main도 기준 SHA와 일치했다. origin은 mhs4670go/TICO이며
사용자 변경은 없었다. production wrapper와 공통 exporter/runtime는 수정하지 않았다.

**R1K1을 추가 검증 후보로 권한다. production 기본 정책 교체는 보류한다.**
3개 data seed의 평균 own-cache attention output MAE와 Quantize 수는 개선됐지만
개별 seed에서는 악화도 있었다. 이는 고정 seed-123 random weights를 사용한
실제 Gemma4 attention wrapper 실험이며 pretrained 정확도 근거가 아니다.

| Profile | R0K0 MAE | R1K0 MAE | R0K1 MAE | R1K1 MAE |
|---|---:|---:|---:|---:|
| smoke affine UINT8 | .03583197 | .03570825 | .03580361 | .03567531 |
| UINT8 storage / INT16 compute | .01232118 | .01224048 | .01212827 | .01203907 |

값은 prefill 8 token 후 own-S-cache 128 decode step의 step별 MAE 평균을
3개 calibration/evaluation seed에서 평균한 것이다. weights는 모든 seed/variant에서
같다. mean±SD, RMSE, max error, normalized L2, cosine, prefill/replay, step 32/128,
2×2 effects/interaction은 [전체 수치 표](report.md)와 [JSON](summary.json)에 있다.
표의 max error는 step별 max의 평균이며 전체 최댓값과 구분한다.

- R1은 affine에서 RoPE Quantize 4–6개, mixed에서 4개를 제거했다.
  mixed R0K1→R1K1은 prefill 6→2, decode 8→4이다.
- R1K1의 affine Quantize 수는 0이지만 graph는 여전히 UINT8 양자화 graph이다.
  marker가 tensor qparams로 표현되며 양자화를 끈 결과가 아니다.
- mixed K1 decode는 필요한 past S→C 변환 2개와 S writeout 2개를 유지한다.
  cache는 UINT8이며 decode K/V delta는 합계 128 bytes, past 136 slots는 17,408 bytes다.
- 모든 그룹에서 공통 boundary의 dtype/qscheme/**실제 scale/zero-point**가 일치했다.
  CALIB가 FP를 전달하므로 정책 경로 변경이 공통 observer 통계를 바꾸지 않은 결과다.
  R0 half/cat 및 K0 P 전용 observer는 이 비교에서 제외했다.
- R1은 mixed seed789에서, K1은 mixed seed456에서 평균 오차를 조금 늘렸다.
  3 seed로 유의성이나 항상 개선됨을 주장하지 않는다.

고정 qparam replay는 별도 표다. 동일한 S bytes를 입력하고 local FP reference도
같은 dequantized S past를 받았다. mixed에서 P 제거와 current S 경유 제거를
각각 분리한 6-policy 진단을 수행했다. 평균적으로 current S 경유 제거의 효과가
더 컸지만 seed456에서는 이 변경만으로 MAE가 약 +6.24e-6 악화됐다.
따라서 K 효과 전체를 past P 제거 하나로 해석하지 않는다. affine frozen replay의
동일 출력도 경계 경로가 실행되지 않았다는 뜻이 아니다.

## Saved Circle 검증

네 조합 × 두 profile × 세 seed의 **48 saved Circle**을 실제 bytes에서 다시 읽어
구조/ABI를 검사하고 FAKE_QUANTIZE 실행했다. 독립 production control의 12개
Circle은 별도로 저장/비교했다. R0K1은 모든 그룹에서 production eager 출력,
attention weights, S delta, qparams 및 serialized topology/ABI가 일치했다.
NO_QUANT wrapper도 원래 HF Gemma4 + DynamicCache와 prefill/4 decode에서 비교했다.
모든 variant의 export/evaluation 전후 weights/frozen observer state는 불변이었다.

Strict single-artifact fidelity는 **38/48 PASS, 10/48 FAIL**이다. 원래 tolerance
atol=1e-5, rtol=1e-4를 유지했으며 이 때문에 본 실행 CLI exit code는 **1**이다.
측정 누락/실행 오류를 성공으로 숨긴 결과가 아니다. report 생성은 exit 0이다.

24개 모두 saved prefill→decode 128 step을 실제 연결했다. Circle이 출력한 S delta를
integer bytes로 encode해 저장하고, 그 원본 bytes를 다음 decode의 S ABI로 재입력했다.
같은 Circle cache를 받은 eager 대비 최대 차이는 affine 0, mixed 0.00118606이다.
`validation_status`는 단발 artifact strict 결과와 recurrent 실행 여부를 집계한다.
그 값만으로 recurrent 수치 parity를 주장하지 않으며 별도 오차 표를 확인해야 한다.

두 prefill mismatch를 원인까지 재현했다. 일치하는 입력·dequantized weights·qparams에서
Torch/NumPy FP32 matmul 누산값이 quantization threshold의 양쪽으로 갈라졌다.
mixed의 sine 입력에서는 ties-even/away ABI encoding 차이도 별도로 확인했다.
[검증 기록](validation.json)의 `fidelity_diagnosis`에 script/trace hashes가 있다.
이 두 진단을 모든 recurrent mismatch의 원인이나 domain 손실 부재의 포괄 증명으로
확대하지 않는다. concat serialized grids 및 필요한 변환의 구조 검증은 별도로 통과했다.

NATIVE 48개는 quantized FC integer kernel이 없어 **UNSUPPORTED**다. NPU 실행과
latency는 미검증이다. Quantize 감소를 latency 개선으로 해석하지 않는다.
storage concat은 이 delta ABI에서 host의 S-byte append이며 Circle 안의 storage
CONCAT kernel을 검증한 것은 아니다. compute concat은 serialized C/C→C로 검사했다.

## 범위와 한계

FP hidden trace를 동일하게 공급하면서 각 variant가 자신의 S cache를 누적했다.
전체 모델의 token teacher forcing, hidden feedback, final logits는 평가하지 않았다.
pretrained checkpoint/tokenizer/held-out text가 없어 **cached NLL/PPL은 ASSET_UNAVAILABLE**다.
random-init PPL은 계산하거나 실제 품질 근거로 사용하지 않았다. 기존 W4A16 Gemma4
config는 조사했지만 실제 pretrained 4-way cached NLL adapter 통합/실행은 남았다.

calibration은 모든 조합에 3 samples × (8 prefill + 4 decode)=36 tokens이며
평가는 별도 seed stream의 136 tokens다. calibration 위치 0..11, evaluation 0..135다.
long-context 오차는 단조롭지 않고 공통 cos/sin clipping도 있다. 결과를 보고 calibration을
바꾸지 않았다. 짧은 calibration 범위와 대상 밖 observer 오차를 R/K 효과와 혼동하지 않는다.

K/V와 past/current/concat **오차**는 별도 기록했다. 일반 observer clipping 집계의
`obs_new_*`는 current delta와 past S-domain 선언 호출을 함께 포함하고,
`obs_present_*`는 past/current/concat 호출을 함께 포함하므로 새 delta 전용 또는
current 전용 clipping rate로 해석하면 안 된다. 보충 sampled clipping trace는
[분리된 clipping 요약](cache-clipping-summary.json)과 hashed raw artifact로 제공한다.
보충 실행은 24개 조합을 같은 qparams/frozen state로 재구성하고 step 0/1/32/128의
96개 출력을 main과 exact 비교했다. current S delta, past S 선언, past P, past/current/concat C를
각각 측정했으며 128 step 전체 clipping 집계는 아니다. raw code가 경계를 아주 조금 넘는
FP 표현 오차와 0.5 code 이상 초과도 구분했다. 예를 들어 mixed current S에서 최대
1/64 clipping과 21.49-code 초과, past P에서 최대 약 0.231% clipping이 관찰됐다.
이 sampled 최댓값을 전체 모델 clipping rate로 일반화하지 않는다. Eager trace에는 유효 token만 포함하며 static Circle padding은
mask로 제외했다. sliding-window 경계(12 steps)와 shared-KV producer/consumer
결합은 네 조합의 작은 회귀로 검증했다.

## 재실행과 자료

먼저 repository virtual environment를 활성화한다. 결과 디렉터리는 새 경로여야 한다.

```bash
export ARTIFACT_ROOT="$(mktemp -d)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  tico.quantization.recipes.debug.rope_kv_experiment.runner \
  --output "$ARTIFACT_ROOT/main"
python -m tico.quantization.recipes.debug.rope_kv_experiment.report \
  --input "$ARTIFACT_ROOT/main" --output "$ARTIFACT_ROOT/main-report"
```

특정 조합만 재실행하려면 runner에 `--variant R1K1 --profile u8s16c --seed 456 --steps 32`를
추가한다. 전체 2×2 report는 네 variant가 모두 있어야 생성된다. 실제 pretrained 후속
평가에는 로컬 weights/tokenizer/text를 먼저 제공하고 기존 Gemma4 W4A16 recipe에
이 정책 adapter와 cached teacher-forced NLL 경로를 통합해야 한다. 현재의 uncached PPL
명령을 KV 효과 검증 명령으로 대체해 제시하지 않는다.

[검증 명령/원래 종료 코드](validation.json), [artifact hashes](artifact-manifest.json)를
보존했다. 신규 회귀 31개, 최종 수정 영향 회귀 11개, 기존 Gemma4 회귀 79개,
exporter pass 회귀 166개(기존 skip 2), runtime independence 4개를 실행했다.
최종 formatting/Pylint/mypy는 실제 base를 지정한 lintrunner에서 통과했다.
실패·중단된 개발 실행 로그도 artifacts에 남아 있다. 무관한 전체 suite와 대형 Llama/Qwen
benchmark는 실행하지 않았다.

Circle/NPZ/raw logs, observer 상세와 detail.json은 Git 밖의 ARTIFACT_ROOT에 있다.
파일별 hash는 manifest와 validation에 있으며 repository 사본은 machine-specific
경로를 REPO_ROOT/ARTIFACT_ROOT placeholder로만 바꿨다. 실행 runtime은 TICO source
279ea8c이다. imported module version은 0.2.0, retained distribution metadata는 0.1.0으로
달라 SHA와 Python source digest를 실제 코드 식별자로 사용한다.
