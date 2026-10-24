# Source review supporting the observed-graph investigation

This file reports source facts only. They do not establish what a serialized Llama graph contains; use the export snapshots, Circle tables, and runtime results as that evidence.

## Revision and instructions

- Inspected checkout: `c0a3cbcf9c67cb2e760185eddf4d7c71901e534d`.
- Latest commit is `[quantization] Retarget explicit Quantize during qparam propagation (#1080)`.
- Read root `AGENTS.md`, `tico/quantization/AGENTS.md`, and `docs/design.md`, plus the relevant quantization and WrapQ README architecture sections.
- No production or test files changed by this review.

## Real Llama module and adapter semantics

- `tico/quantization/config/llama_attention.py`: default and explicit `npu_export` profile use `scale_fusion="k_proj"`, `rope="pre_negated_sin"`, and `layout="unrolled"`. The reference profile uses batched layout, ordinary HF sine, and no scale fusion.
- `quant_attention.py:249`: pre-negated-sine rotate-half computes `(x2,x1)`. The first half of sine must already be negative. `quant_decoder_layer.py:172-186` builds static cos/sin tables with this exact convention.
- `quant_attention.py:506-512`: tuple past-KV tensors are fake-quantized with `obs_past_key/value` before head extraction. `:624-626` extracts `(B,Kpast,H)` head slices.
- `quant_attention.py:648-661`: prefill present-K/V applies `obs_present_*` to each current head; decode concatenates past and current head at dimension 1, then applies `obs_present_*` to the concatenated result.
- `quant_attention.py:804-833`: current K is RoPE-transformed, then passed to `_build_present_kv_head`; current V is the projection slice, then passed directly to that helper. The same current tensors are separately appended to delta-output lists.
- **The unrolled implementation does not apply `obs_new_*` before the attention Concat.** `quant_attention.py:680-681` applies `obs_new_k/v` to `torch.stack(new_*_parts,dim=1)` only in `_finalize_cache_output`, after the attention path has already used those tensors. In contrast, the batched path applies `obs_new_*` before Concat (`:954-968`). Therefore the requested N grid cannot be assumed to describe the current compute input.
- `quant_attention.py:857-873`: K Concat feeds transposition and QK matmul, V Concat feeds attention-times-V matmul; each KV head is reused for its query-head repetitions.
- `export_adapters.py:350-377` and `:413-441`: actual decoder-layer signatures are `(hidden_states,attention_mask,position_embeddings)` for prefill and the same plus `past_key_value` for decode. The latter two arguments are `(cos,sin)` and `(past_key,past_value)` tuples, respectively. Both adapters set `cache_output_mode="delta"`; with `return_kv=True` the outputs are hidden, new K, new V.
- `quant_decoder_layer.py:341-377`: real decoder-layer path includes input layernorm, wrapped attention, residual addition, post-attention layernorm, MLP and final residual quantization. `:356-364` forwards cache output mode through kwargs to attention. `:399-407` chooses the normal prefill/decode adapters.

## Folding and propagation

- `fold_quant_ops.py:107-188`: `_inherit_qparam_from_view_chain` recognizes affine-preserving slice/reshape/permute/expand chains and same-domain Concat inputs. It seeds view metadata before determining whether a following quantization is a new domain.
- `fold_quant_ops.py:372-400`: if the immediate producer has no existing qparam, an ordinary sole-consumer producer adopts the Q-DQ grid and both Q and DQ disappear. A shared float producer retains explicit Q to avoid quantizing other users.
- `fold_quant_ops.py:407-438`: if the producer already has qparams, equal qparams remove the Q-DQ pair; differing qparams retain explicit Q with its destination metadata and remove DQ. Same dtype alone does not make different scales/zero-points equal.
- `propagate_qparam_forward.py:104-112`: for affine tensors, propagation may replace an existing same-dtype destination qparam; different dtype stops propagation. `:119-159` includes permute/reshape/slice (and listed unary operations). `:171-236` handles Concat: same-domain inputs seed the output, otherwise symmetric INT16 inputs can seed output with the greatest scale. Different UINT8 input domains do not trigger that fallback.
- `propagate_qparam_backward.py:42-116`: new helper retargets an already explicit, private, valid, scalar affine per-tensor Quantize. Its only user must be the propagating node. It updates argument scale/zero-point and output qparam atomically, preserving dtype/range, and never crosses its input domain boundary.
- `propagate_qparam_backward.py:146-158`: every explicit quantizer is a backward boundary, including unsupported overloads; supported private affine per-tensor Quantize may retarget and traversal stops there.
- **`propagate_qparam_backward.py:173-181` permits existing affine producer metadata to be overwritten when dtype matches. It does not use the sole-user restriction for ordinary affine producers/views/placeholders.** This is distinct from explicit Quantize retargeting, which does use the sole-user guard.
- `propagate_qparam_backward.py:183-205`: reverse traversal propagates Concat output metadata into each input, and reshape/permute metadata into their sources. It does not separately traverse slice nodes here.
- `concat_qparam.py`: width-Concat exception requires provenance from TICO Concat/QuantConcat and a rank >=3 penultimate-axis concat. A raw `torch.cat` for Llama sequence concatenation is intentionally excluded even though it is rank 3 and dimension 1.

## Conversion boundary and exact pass schedule

- `tico/utils/convert.py:234-252`: decompose fake-quant (both scalar and tensor-qparam passes), then Torch core decomposition.
- `:258-319`: legalizations and layout conversions run before quantization folding.
- `:327-340`: quantization bundle is `FoldQuantOps -> RemoveWeightDequantOp -> PropagateQParamForward -> PropagateQParamBackward -> LegalizeQuantizedClamp -> QParamSafeConstPropPass -> QuantizeBias -> RemoveUnusedPlaceholder -> InsertQuantizeOnDtypeMismatch`.
- `:355-357`: support/training checks precede serialization.
- `:367-381`: bytes export first prepares graph and serializes with `build_circle_bytes`; only then optionally invokes Circle O1. Thus an FX metadata change before serialization is not an O1 removal.
- `op_quantize_per_tensor.py:37-61,84-85`: serializer rejects disagreement between explicit Quantize arguments and destination tensor scale/zero-point. `:93-108` emits a real Circle QUANTIZE.

## Tests inspected

- `test/quantization/passes/test_quantize_per_tensor_retarget.py`: synthetic double-quantized inputs have initial UINT8 grid `(0.5,127)` and intermediate grids `(0.125,17)` / `(0.25,23)` before a Concat grid `(0.375,29)`. The test requires two real Circle Quantizes from initial grid to Concat grid for **both noopt and O1**. Separate test rejects a metadata-only Quantize destination change. This proves generic behavior, not a Llama topology.
- `test_quant_attention.py` and `test_quant_decoder_layer.py`: tiny locally initialized model tests cover prefill/decode adapters, returned delta shapes, NPU profile settings, sign conventions and lifecycle behavior.
- `test_quantize_attention_decode.py`: includes `test_percentile_npu_profile_exports_prefill_and_decode`, which runs `torch.export.export(...,strict=False)` and checks exact exported-Torch/eager equality; it does not inspect Circle cache ABI.
- `test_quantize_decoder_layer_prefill.py` and `test_quantize_decoder_layer_decode.py`: lifecycle/example tests are internal-test gated; use `--include-internal` when invoking them. They do not by themselves establish Circle graph topology.

## Interpretation guardrails

1. A no-Quantize result may reflect first-boundary folding into producer/input metadata, rather than explicit Quantize retargeting.
2. Concat propagation may change a producer or placeholder grid before Circle serialization. Track the actual input ABI and producer users in snapshots.
3. Delta N observers are output branches in the requested unrolled implementation. Compare actual serialized outputs to actual next-stage inputs, not N/R/C observer names.
4. Only attribute a change to Circle O1 when it is present in the noopt Circle and absent from O1 Circle.
