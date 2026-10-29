# Exported Circle snapshots

Added following the explicit request to push the exported Circle files. These are the existing measured binaries, copied byte-for-byte; no weights, qparams or exports were regenerated.

This branch contains **20** primary default-O1 files (1,451,632 bytes): BASE10 under `base/` and this branch's variant10 under `b/`. Each implementation includes inspector2/2 G1 prefill/decode, custom24/8 prefill/decode FP32+QUANT, and tiny4/2 prefill/decode FP32+QUANT. The BASE O0 diagnostic is excluded.

`manifest.json` records implementation SHAs, fixture dimensions, byte counts, SHA256 and the previous per-file export/runtime validation. `SHA256SUMS` verifies the committed bytes:

```bash
cd docs/archive/gemma4_gqa_grouping/20261029T070221Z/circle_artifacts
sha256sum -c SHA256SUMS
```

All30 distinct primary source artifacts (BASE/A/B) passed structural reverification before copying. Every custom/tiny source hash and size matched the measured export summary; copied files have identical bytes. This addition does not rerun reference inference, model tests or benchmarks.

Prior runtime status must be read separately from schema verification:

|Implementation|QUANT reference runtime|FP32 reference runtime|
|---|---|---|
|BASE|All6 execute exactly vs ABI-aligned eager|custom/tiny prefill and decode4 fail the known serialized Softmax/PV rank contract|
|A|All6 execute exactly vs ABI-aligned eager|All4 execute|
|B|All6 execute exactly vs ABI-aligned eager|custom/tiny prefill2 execute; decode2 fail the known serialized rank contract|

Quantized execution used the explicit FAKE_QUANTIZE semantic reference. NPU compile and device execution remain NOT_RUN. Failed FP artifacts are retained for diagnosis and labeled in the manifest; they are not presented as deployable/runtime-passing models. See [comparison](../comparison.md) and [export results](../export_summary.md).

The repository normally ignores generated Circle artifacts. This directory's narrow ignore exception implements the user's explicit follow-up request without changing the global ignore rule. No checkpoint, tensor dump, NPU binary or raw log is added.
