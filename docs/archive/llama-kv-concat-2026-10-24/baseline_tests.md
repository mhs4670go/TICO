# Existing baseline tests

All commands ran from `/workspace/TICO` after `source /tmp/tico-venv/bin/activate`, with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. No production source or existing test changes were made. No remote checkpoint/tokenizer/model/dataset was downloaded. Tests use the built-in reference runtime by default.

| Exact test command | Outcome | Log |
|---|---|---|
| `./ccex test -k quantize_per_tensor_retarget` | Passed: 2 tests, 0.892 s, zero skips | `test_retarget.log` |
| `./ccex test -i -k quantization.wrapq.wrappers.llama` | Passed: 55 tests, 2.727 s, zero skips | `test_llama_wrappers.log` |

The generic retargeting baseline exercises public conversion serialization of retargeted concat inputs and rejects a metadata-only Quantize domain change. It proves generic explicit retargeting/serialization, not the observed Llama topology.

The wrapper group includes the closest tiny real `LlamaAttention` and `LlamaDecoderLayer` lifecycle/adapter tests, including local random CPU prefill and cached decode. The NPU percentile attention export test exports both adapters with `torch.export.export(strict=False)` and verifies exact equality against quantized eager adapter outputs. The group includes a real MLP Circle export; its decoder adapter tests verify eager signatures and shapes rather than Circle serialization.

The separate `./ccex test -k test_llama_circle_runtime` command runs real saved stage-bundle exports with local random tiny Llama models (hidden size 8, intermediate 16, 2 attention heads, 1 KV head, head_dim 4, max_seq 8, vocab 32). Its fake-quantized class uses a one-layer PTQ model calibrated through prefill and real cached decode; float tests use two layers. This provides existing Circle export/runtime coverage beyond the wrapper tests. Results follow below once complete.

| Exact test command | Outcome | Log |
|---|---|---|
| `./ccex test -k test_llama_circle_runtime` | Passed: 13 tests, 25.026 s, zero skips | `test_llama_circle_runtime.log` |

This group includes `TestFakeQuantizedLlamaCircleRuntime.test_fake_quantized_teacher_forced_logits_and_generation` and `test_quantized_artifacts_require_explicit_fake_quantize_mode`. The log records the expected `[QuantCheck] WARNING: 3 nodes without qparam detected` during PTQ stage exports. That warning did not cause a skip or failure. The test result does not establish equal cache qparams, target-NPU validity, or bit-exact integer backend behavior.

`git status --short` was empty after these tests. No full-suite, remote-model, CUDA, ONE, or ONERT tests were run.
