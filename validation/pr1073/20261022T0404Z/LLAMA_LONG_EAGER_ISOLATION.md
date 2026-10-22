# 4096-token eager 이후 실제 full Circle 격리 검증

**원래 strict 실행 결과는 FAIL(exit1)** 이다. 4096-token eager와4회 cached decode는 실제 완료됐고, 전후 Circle의 모든 구성요소/실제 출력은 동일했다. 그러나 요청된 Torch-vs-Circle parity가 불일치하여 전체 isolation PASS로 바꾸지 않았다.

| 확인 항목 | 새 실행 결과 |
|---|---|
| 실제 tiny-HF/PTQ checkpoint | 원래 constructor native capacity16384, llama3 original capacity8192; 새로 생성 |
| calibration/freeze | 실제 text8/6/4, cache prefill4+decode2 후 freeze |
| 긴 eager | prefill4096+연결된 cached decode4 완료; 요구4100 ≤ 원래 기록capacity16384 |
| frozen state | eager 직후 및 양쪽 export 후 모두 보존, differences0 |
| 실제 static export | static capacity8, `tico.convert`/Torch Export/실제 CircleReferenceRuntime full model 양쪽 phase 완료 |
| stage/수치 증거 | phase마다 `model.q.circle` 1개, 117operators/140qparam tensors, input1/output5 완전 검증, 모든 실제 array finite |
| eager 전후 component 동일성 | Circle binary/complete graph contract/constants/qparams/I/O/TorchFX/host metadata/실제 모든 input·Torch·Circle output array 동일 |
| strict Torch/Circle parity | **FAIL**, atol1e-5/rtol1e-4 유지 |
| 전체 원래 comparator | **FAIL**, exit1 유지 |

정적 full Circle의 입력은 int64 `[1,8]`, 출력은 logits `[1,8,64]`와 두 layer의 K/V `[1,1,8,16]`4개다. 5개 Torch golden과 5개 실제 Circle 출력, 하나의 실제 bound input이 두 phase 모두 저장돼 있다. phase마다 수치 probe는 `export_example`1개다. `circle_full` 경로에는 별도 `llama_host.npz`가 없어 host 비교는 빈 manifest의 동일성이다(실제 host arrays 검증은 별도 circle_bundle 결과를 따라야 한다). CircleReferenceRuntime은 `fake_quantize` 모드였다. 이 결과는 NPU 실행이나 bit-exact 정수 backend 검증이 아니다.

Torch/Circle 출력별 최대 절대오차(두 phase 동일):

| 출력 index | max abs | strict parity |
|---|---:|---|
| 0 logits | 0.0055593401193618774 | FAIL |
| 1 layer0 K | 0 | PASS |
| 2 layer0 V | 0.009165078401565552 | FAIL |
| 3 layer1 K | 0.0015604384243488312 | FAIL |
| 4 layer1 V | 0.005374044179916382 | FAIL |

전후 graph와 수치가 정확히 같다는 관측은 긴 eager가 이 static snapshot을 바꾸지 않았다는 좁은 근거다. 원래 baseline과의 동등성이나 Torch/Circle 수치 정확성 PASS는 아니다. 본 실행만으로 parity 오차의 원인을 RoPE 또는 특정 rounding 연산으로 확정하지 않으며, 별도 `PARITY_DIAGNOSIS.md`/원본 수치 분석과 함께 해석해야 한다. tolerance 변경, 제품 수정, 재양자화는 하지 않았다.

## checkpoint와 실행 식별

- 대상 SHA: `6b2f6a0511458d6a750e2a260d706080a78149dd`.
- checkpoint: `fixtures/candidate_llama_native16384.pt`, CPU float32, 336357bytes.
- SHA256: `bc5efaaaf81d4e3d4efdd8b9bbb790971e58a171cdbf26f80577739f2ac602bf`. 실행 후 파일 해시도 동일했다.
- LlamaConfig는 처음부터 native16384로 생성했다. 기존 checkpoint의 config를 늘리지 않았다. 원래 llama3 original capacity8192와 모순되는 작은 native capacity를 사용하지 않았다.
- creator는 안정된 `create_tiny_frozen_fixture.py`를 `create_long_llama_fixture.py`로 그대로 복사했다. 코드 SHA256 `041bb03ca7e9e080eac42672f58062490ee373c9467d2af4cf36d650f60b7dd0`.
- 실제 사용한 runner는 **수정하지 않은** `tico/quantization/recipes/debug/export_regression.py`이며 SHA256 `451508ace4f387be5306fe829a76bfbd5a11249148e80be089e1173e1a851111`이다. copied-runner shape-witness 수정은 이 candidate-only 실행에 필요 없어 사용하지 않았다.
- profile: `npu_export`, parameter22688개/weight tensors21개/observer143개/quant mode62개를 포함하는 작은 random local HF model이며 pretrained checkpoint가 아니다.

## 재현 명령

새 checkpoint 생성과 resource gate 명령은 `commands.sh`의 `create_candidate_llama_native16384`, `long_llama_resource_gate` 항목에 있다. 출력 하위 경로는 실행 전에 만들지 않았다. 생성된 checkpoint/config를 사용한 원래 명령은 다음과 같다(재실행 시 새 output 경로 필요).

```bash
.venv/bin/python tico/quantization/recipes/debug/export_regression.py isolation \
  --candidate-repo . \
  --config out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native16384.export.yaml \
  --checkpoint out/pr1073_validation/20261022T0404Z/fixtures/candidate_llama_native16384.pt \
  --artifacts circle_full --seq-len 8 --eager-seq-len 4096 --decode-steps 4 \
  --require-torch-parity \
  --output out/pr1073_validation/20261022T0404Z/tiny-llama-native16384-long-eager-full
```

실제로는 동일 명령을 logger/resource monitor로 감싸 stdout/stderr/exit/timing/RAM/disk를 기록했다. preflight cgroup 사용량<12GiB를 요구했고 gate를 통과했다. 전체 worker process-tree peak RSS는1374711808bytes(약1.28GiB), 관측 중 최소 host available RAM은14227992576bytes, 실행 후 free disk는29316722688bytes다. 실행 시간178.86s이며 large static lint 작업은 이 stress가 끝날 때까지 defer했다.

## 원본 증거

- `llama_long_eager_full_isolation.log` / `.command.json`
- `llama_long_eager_full_resources.json`, `long_llama_resource_gate.json`
- `fixtures/candidate_llama_native16384.provenance.json` / `.export.yaml`
- `tiny-llama-native16384-long-eager-full/summary.json`: 원래 comparator FAIL
- 같은 run의 `candidate/eager.json`, `candidate/environment.json`, `candidate/quantization_state.json`
- `candidate/before` 및 `candidate/after_eager`: 각각 실제 Circle·snapshot·record `.json`/`.npz`
- `independent_full_evidence.json`: 독립 completeness/finite/전체 component 동일성/strict parity 감사
- `audit_long_llama_full.log` / `.command.json`: 독립 감사 exit1
- `llama_long_eager_analysis.json`: raw 결과와 좁은 전후 동일성 근거를 구분한 분석

이 실행은 >2048 eager 입력의 영향을 검사했지만 **static2048 export**, pretrained Llama-3.2, old/new baseline 비교, VLM 이미지/video isolation은 검증하지 않았다. static artifact의 용량은8이다.
