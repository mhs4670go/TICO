# Gemma4 실제 Circle 실패 artifact

이 디렉터리에는 실제 PNG processor 입력과 frozen tiny HF PTQ checkpoint로 생성한 **진짜 Circle binary**를 보존했다. 원래 regression recorder는 runtime 실행 후에만 저장하므로 unsupported operator 발생 시 `.circle`이 남지 않아, 별도 1회 실제 변환에서 먼저 저장하고 동일 runtime 실패를 재현했다. 원본 실패 결과는 유지했다.

- `vision_prefill.q.circle`: 165504 bytes, SHA256 `4cb5d8deecfef8cf41338211d4d298170048ef0bf64f1893d4b85b6442b0af5d`
- `circle_contract.json`: 실제 직렬화된 graph I/O·topology/options·constant/qparam 계약
- `torch_outputs.npz`: 실제 Torch vision 출력 float32 `[4,32]`, 전 원소 finite
- `frozen_before.json` / `frozen_after.json`: weights·observer stats/qparams·mode fingerprint, 두 값 동일
- `result.json`: 입력/체크포인트/출력 hashes, 실행 상태와 원본 traceback

정상 `tico.convert`, 실제 저장, `CircleDocument.verify`는 완료되었다. 실제 `CircleReferenceRuntime`의 FAKE_QUANTIZE 실행은 op 102의 RMS_NORM에서 다음 오류로 끝났다.

```
UnsupportedCircleOperatorError: RMS_NORM supports rank 3 or 4 inputs, found shape (4, 32).
```

근거: `tico/circle/runtime/kernels/nn.py:619`의 rank guard. Export adapter는 원래 visual soft token을 `[V, text_hidden_size]`로 평탄화하는 계약을 갖는다 (`gemma4/export_adapters.py:385`). Baseline 및 candidate 원래 실행에서도 같은 오류가 관측되었다. 따라서 기존 reference runtime의 rank 지원 제한으로 기록하며, 최종 Circle 실행이나 수치 parity PASS로 보고하지 않는다. Tiny real-HF profile이므로 pretrained Gemma4 또는 NPU 검증도 아니다.

미적용 최소 수정 제안: RMS_NORM의 rank-2 입력 지원을 검토하고, 마지막 축 RMS normalization의 독립 Torch 기준, gamma 계약, fake-quant 및 해당 실제 vision artifact 회귀 테스트를 함께 추가한다. Runtime guard나 kernel은 이번 검증에서 수정하지 않았다. Threshold도 바꾸지 않았다.

재현 생성 명령·stdout/stderr·exit 2는 상위 `preserve_gemma_vision_failure.command.json`과 `.log`, `commands.sh`에 보존했다. `preserve_gemma_vision_failure.py`는 새 output directory를 요구한다.
