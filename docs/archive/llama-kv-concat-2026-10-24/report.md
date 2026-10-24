# Tiny Llama KV-cache quantization investigation

The actual decode graph has **Case B on the past inputs and Case C on the current producers**: no explicit QUANTIZE feeds either past/current KV Concat. This is already true in the unoptimized Circle. O1 removes no QUANTIZE. The controlled decode additionally exposes **Case D on the separate delta-output branches**, where singleton Concats retain stale output qparams after their shared input producers adopt the present grid.

The natural exported cache ABI is directly chainable in this experiment. The controlled ABI is not: both prefill and decode delta outputs use N=(0.0625,113), while decode past inputs use C=(0.3125,91). There is no internal decode cache-input requantizer; the host must requantize. Controlled decode also fails exact eager/Circle cache parity. Consequently this result does **not** support treating Llama as evidence that the proposed Gemma4 cleanup preserves explicit requantization or cache ABI/parity.

## Environment and model

- Latest fetched `origin/main` and HEAD: **c0a3cbcf9c67cb2e760185eddf4d7c71901e534d**. Main was fetched before investigation; HEAD matched it throughout.
- Commit title: `[quantization] Retarget explicit Quantize during qparam propagation (#1080)`. The merged change is present, and its generic regression passes.
- Python **3.12.14**; Torch **2.13.0+cpu**; Transformers **5.18.0**; TorchVision **0.28.0+cpu**. Linux x86_64, one Torch CPU thread, deterministic algorithms enabled.
- Read root `AGENTS.md`, `tico/quantization/AGENTS.md`, `docs/design.md`, relevant quantization/WrapQ/Circle guides, test instructions, and every requested source/test file. Source analysis is in Appendix D; the conclusions below come from real exports and execution.
- Real randomly initialized `LlamaDecoderLayer(config, layer_idx=0)`, not a substitute attention-only model. Config: hidden=16, intermediate=32, layers=1, attention heads=2, KV heads=1, head_dim=8, max positions=8, vocab=32, bias=False, dropout=0, eager attention.
- `PTQConfig(model_args={"profile":"npu_export"})`; selected options verified as `layout="unrolled"`, `rope="pre_negated_sin"`, `scale_fusion="k_proj"`.
- Model/calibration seed **20261024**, evaluation seed **1984**. CPU only. No checkpoint, model, tokenizer or dataset download.
- Normal `prepare -> calibration -> convert` lifecycle. Calibration: three seven-token prefill calls, each followed by a real one-token cached decode using that prefill's returned cache. Normal conversion freezes qparams. The controlled scenario is a deep copy of that identical frozen model, with only the six cache observers loaded through `load_qparams(...,lock=True)`; no graph metadata was patched.
- Real adapter signatures: prefill `(hidden_states, attention_mask, (cos,sin))`; decode adds `(past_key,past_value)`. Both use `return_kv=True` and return hidden plus delta K/V.
- Prefill static shapes: hidden `(1,7,16)`, additive causal mask `(1,7,7)`, RoPE `(1,7,8)`. Decode shapes: hidden `(1,1,16)`, zero additive mask `(1,1,8)`, RoPE `(1,1,8)`, each past `(1,1,7,8)`. RoPE uses repeated half-frequency tables and negates the first four sine channels exactly as the NPU convention requires. Decode position is 7. Past test values are deterministic off-grid linspaces to distinguish candidate equations.

## Frozen observer qparams

All six observers are UINT8, scalar per-tensor asymmetric affine, range [0,255], disabled after freeze, wrapper mode QUANT. Natural observers are not externally locked; controlled loaded observers are locked. These are observer values, not assumed serialized boundary values.

| Observer | Natural scale | Natural zero point | Controlled scale | Controlled zero point |
|---|---:|---:|---:|---:|
| obs_past_key | 0.004473801702260971 | 128 | 0.1875 | 127 |
| obs_past_value | 0.010872464627027512 | 116 | 0.1875 | 127 |
| obs_new_k | 0.004473801702260971 | 128 | 0.0625 | 113 |
| obs_new_v | 0.012130454182624817 | 104 | 0.0625 | 113 |
| obs_present_key | 0.004473801702260971 | 128 | 0.3125 | 91 |
| obs_present_value | 0.012130454182624817 | 104 | 0.3125 | 91 |

Define Pk=(0.004473801702260971,128), Pv=(0.012130454182624817,104), controlled R=(0.1875,127), N=(0.0625,113), C=(0.3125,91). Natural Ck=Pk and Cv=Pv.

## Observed FX evolution

Temporary method wrappers captured actual pass invocations without changing pass order, scheduling, arguments, results or graph metadata. Captures: immediately after `torch.export.export`, after each fake-quant decomposition invocation, FoldQuantOps, forward propagation, backward propagation, and immediately before serialization. Every corresponding noopt/O1 FX snapshot is identical. Full extracted subgraph tables with targets, inputs, output nodes, qparams, Quantize arguments and users are in Appendix A; full graph dumps are not pasted.

The cache Concats are `%cat_1` K and `%cat_2` V, `aten.cat.default`, dim=1, `(1,7,8)+(1,1,8)->(1,8,8)`. Output nodes are the cat nodes themselves. After legalization their data inputs are `%reshape_default_21` + `%add` for K and `%reshape_default_22` + `%linear_2` for V.

| Stage | Past path | Current path | Concat/output |
|---|---|---|---|
| torch.export | placeholder -> actual past fake-quant -> head select | K RoPE ADD -> K-rotation fake-quant; V projection -> activation fake-quant -> view/select | cat -> present fake-quant; no TICO qparam metadata yet |
| fake-quant decomposition | Q_R -> DQ_R -> select | original producer Q_P/DQ_P boundaries | cat -> Q_C -> DQ_C; separate stacked delta -> Q_N/DQ_N |
| FoldQuantOps | past Q/DQ folds into placeholder R and head reshape R | original P metadata folds into K ADD / V FC | present Q/DQ folds into cat C; **no input Quantize remains** |
| PropagateQParamForward | preserves R | propagates P along view/delta branches | cat retains C |
| PropagateQParamBackward | overwrites past placeholder and reshape with C | overwrites shared K ADD and V FC with C | main cat inputs/output agree; controlled delta singleton cats retain stale P |
| before serialization | same as backward result | same as backward result | no subsequent repair; output Q_N survives only when distinct |

Natural K past already equals C. Natural V past changes from (0.010872464627027512,116) to Cv=(0.012130454182624817,104). Controlled K and V both change R->C. This is an input-ABI metadata overwrite, not an explicit Quantize argument retarget.

Observer-boundary mapping: decode past fake-quant `_6/_7` folds into input placeholders; present fake-quant `_24/_25` folds into `%cat_1/%cat_2`; new fake-quant `_50/_51` is after stacks on delta branches and becomes either folded metadata (natural) or explicit `%quantize_per_tensor_49/_50` (controlled). In the unrolled path **N is not upstream of attention Concat**.

No cache Quantize scale/zero-point arguments changed during propagation. Decomposed scalar Q nodes have UINT8 dtype, range [0,255], and one DQ user. Surviving controlled decode output Q_N nodes each have one output user. Controlled prefill retains Q_C `_22/_23`, each with two attention consumers, and output Q_N `_47/_48`, each with one output user. The merged generic explicit-Q retarget code is present but is not exercised on these decode Concat inputs.

## Final Circle graphs and O1 comparison

All eight artifacts were parsed with `CircleDocument.from_bytes`, passed `document.verify()` with no issues, and inspected through CircleGraph/inspection APIs. Identification used axis, exact edge shapes, names and dataflow: K feeds transposes then QK matmuls; V feeds attention-times-V matmuls. Operator order was not the identification criterion.

| Scenario/module | Operators noopt -> O1 | Total Q noopt -> O1 | Q feeding KV Concat | All KV-path Q noopt -> O1 |
|---|---:|---:|---:|---:|
| natural prefill | 54 -> 52 | 2 -> 2 | N/A: no past/current concat | 0 -> 0 |
| natural decode | 58 -> 56 | 2 -> 2 | 0 -> 0 | 0 -> 0 |
| controlled prefill | 58 -> 56 | 6 -> 6 | N/A | 4 -> 4 |
| controlled decode | 60 -> 58 | 4 -> 4 | 0 -> 0 | 2 -> 2 |

Controlled prefill has two KV compute Q_C operators and two delta Q_N operators. Controlled decode has only two delta Q_N operators. Every model also has two hidden/layernorm QUANTIZE operators unrelated to cache Concat.

O1 removes exactly two attention-head layout reshapes (`reshape_default_15` [1,Q,2,8] and `_16` [1,Q,16]) and canonicalizes three SPLIT_V operators to SPLIT. **No QUANTIZE disappears.** No cache ABI or cache qparam changes between noopt and O1. There is no basis to say “O1 removed the KV requantization.”

Main decode cache tensors have the following identities in **both scenarios and both optimization settings**. Names use suffixes of the preserved full names; Appendix B contains full names and per-artifact producer/consumer tables.

| Role | Tensor | Name suffix | Shape | Producer | Consumers |
|---|---:|---|---|---|---|
| past K input | 13 | tico::past_key_value_0 | [1,1,7,8] | graph input | op11 RESHAPE |
| past V input | 14 | tico::past_key_value_1 | [1,1,7,8] | graph input | op12 RESHAPE |
| K past Concat input | 27 | reshape_default_21 | [1,7,8] | op11 RESHAPE | op13 CONCATENATION |
| V past Concat input | 28 | reshape_default_22 | [1,7,8] | op12 RESHAPE | op14 CONCATENATION |
| K current Concat input | 26 | add | [1,1,8] | op10 ADD | op13 + delta singleton Concat |
| V current Concat input | 20 | linear_2 | [1,1,8] | op5 FULLY_CONNECTED | op14 + delta singleton Concat |
| K Concat output | 29 | cat_1 | [1,8,8] | op13 CONCATENATION, axis1 | op22/34 TRANSPOSE -> QK matmuls |
| V Concat output | 30 | cat_2 | [1,8,8] | op14 CONCATENATION, axis1 | op26/38 BATCH_MATMUL |

All above tensors are UINT8. Natural K rows all have Pk; natural V rows all have Pv. Every controlled row has C=(0.3125,91). Thus the main KV Concats have consistent input/output grids.

Actual noopt path:

```text
past Circle input [C] -> RESHAPE [C] -----+
                                         +-> KV CONCAT axis1 [C] -> attention
current K ADD / V FC output [C] ----------+
       |
       +-> singleton CAT [P*] -> RESHAPE [P*] -> QUANTIZE [N] -> delta output
           * controlled decode: stale grid, incompatible with C input
```

Actual O1 path:

```text
past Circle input [C] -> RESHAPE [C] -----+
                                         +-> KV CONCAT axis1 [C] -> attention
current K ADD / V FC output [C] ----------+
       |
       +-> singleton CAT [P*] -> RESHAPE [P*] -> QUANTIZE [N] -> delta output
```

The two removed O1 reshapes are elsewhere. Natural delta N=P=C, so its delta Quantize folds away and the starred mismatch does not occur. Controlled prefill instead preserves the original producer P and has separate `P -> Q_C -> attention` and `P -> singleton CAT/reshape -> Q_N -> delta` branches.

## Unexpected delta-branch metadata (Case D)

Controlled decode shared producers each have two users. Backward propagation visits the later delta branch first, then the earlier attention Concat overwrites the shared producers with C; it leaves already-visited delta cats at P, without another forward propagation.

| Branch | Singleton Concat noopt / O1 | Input tensor/grid | Output tensor noopt / O1 and grid |
|---|---|---|---|
| K | op43 / op41 | t26 C | t61 / t59 Pk |
| V | op46 / op44 | t20 C | t64 / t62 Pv |

These single-input Concat operations are value preserving, yet their input/output grids differ. Subsequent reshapes preserve stale P. Structural verification does not validate this semantic invariant.

A native-reference probe extracted each actual singleton Concat without metadata edits. Input code 91 represents real zero in C; the operator copied code 91, but its output metadata decoded that code as **-0.165530663** (Pk) or **-0.157695904** (Pv), in both noopt/O1. This confirms the metadata/operation inconsistency independently of fake-quantized arithmetic. It does not claim target-NPU validation.

## Prefill-to-decode and decode-to-decode ABI

| Scenario, both noopt/O1 | Prefill delta K/V outputs | Decode delta K/V outputs | Decode expected past inputs | Classification |
|---|---|---|---|---|
| natural | K Pk, V Pv | K Pk, V Pv | K Pk, V Pv | **1: identical**, host can copy/append without requantization |
| controlled | K/V N=(.0625,113) | K/V N=(.0625,113) | K/V C=(.3125,91) | **3: host-side requantization required**, no internal cache-input Q |

The controlled case also has the independent Case-D delta-branch issue above; host requantization addresses boundary encoding, not that graph inconsistency or eager parity.

Circle delta-output tensor indices (K,V), noopt / O1: natural prefill (56,58)/(54,56); natural decode (62,64)/(60,62); controlled prefill (59,62)/(57,60); controlled decode (63,66)/(61,64). Shapes are [1,1,7,8] for prefill and [1,1,1,8] for decode. Natural outputs are produced by reshapes; controlled outputs by Q_N. Appendix B records names, producers and every consumer.

## Numerical verification

Executed quantized eager adapters and real serialized Circle via `CircleReferenceRuntime.run(mode=ExecutionMode.FAKE_QUANTIZE,trace=True)`. No tolerance was widened or selected to declare observed mismatches passing. The following values apply independently to both noopt and O1; corresponding noopt/O1 outputs are exactly equal.

| Scenario/module | Hidden max abs (different/elements) | New K max abs (different/elements) | New V max abs (different/elements) |
|---|---:|---:|---:|
| natural prefill | 0 (0/112) | .004473801702260971 (1/56) | 0 (0/56) |
| natural decode | .024770021438598633 (1/16) | 0 (0/8) | 0 (0/8) |
| controlled prefill | 0 (0/112) | 0 (0/56) | 0 (0/56) |
| controlled decode | .024770110845565796 (9/16) | .125 (7/8) | .125 (7/8) |

Exact matches are exact equality. Natural mismatches are approximately one output-grid step and are left as observed differences, without an unproven attribution to rounding. Torch uses nearest-even; Circle reference uses ties away from zero. Controlled delta discrepancies are two N steps, corroborated by producer adoption rather than explained away as floating-point noise.

Independent expected affine fake quantization used `Q(s,z,x)=(clip(round(x/s)+z,0,255)-z)*s`. Runtime input binding rounds floats to the serialized input grid. Therefore the Circle past path is **past already encoded/interpreted as C at its ABI**, numerically Q_C(raw supplied float), not Q_C(Q_R(past)). Controlled actual past equals Q_C(raw) exactly and differs from Q_C(Q_R(raw)) at 8/56 K and 7/56 V entries, max .3125. Natural V differs at 11/56 entries, max .09704363346; natural K R=C.

Reconstructing raw current values from actual serialized K ADD operands and V FC input/dequantized weights shows current tensors equal Q_C(raw) exactly. No upstream Q_N exists. However the supplied C/N ratio is the odd integer 5: the eight observed current values also satisfy Q_C(Q_N(raw)). Numeric samples alone cannot exclude that candidate for this nonsaturating range; the operator graph and tensor qparams establish direct C. An independent saturation witness with the unchanged grids distinguishes the equations: raw=9.125 gives Q_C(raw)=9.0625 versus Q_C(Q_N(raw))=8.75. This witness is algebraic, not an observed full-model activation.

Controlled Circle delta outputs equal Q_N(actual C producer) exactly. Eager deltas equal Q_N(the original P-quantized current head). For K:

```text
Circle: [ .3125, 0,     .3125, 0,     .3125, 0,    0,     0 ]
Eager:  [ .1875,-.0625,.25,  -.0625,.375,  .125,-.125, 0 ]
```

This numerically corroborates that C adoption changes the shared delta-output computation. Appendix C contains complete numerical results, V arrays, independent equations and native extraction probe results.

## Conclusion and limits

- **Explicit Quantize before decode KV Concat? No**, in either scenario, with or without O1.
- **Where is the transition?** Past Q/DQ first folds into input metadata; backward propagation overwrites that input ABI and its reshape with C. Current K ADD/V FC adopt C directly. No explicit cache-Concat-input Quantize is retargeted here.
- **O1 changes the answer? No.** It removes two unrelated reshapes and zero Quantizes.
- **Direct cache chaining?** Natural yes in this calibration; controlled no, requires host N->C requantization. Observer labels alone would have given the wrong ABI conclusion for natural V past.
- **Valid/expected everywhere? No.** Main KV concat grids agree, but controlled decode delta singleton-Concat metadata is inconsistent and delta values diverge from eager.
- **Supports Llama as the model for Gemma4 cleanup? No as evidence of safe explicit requantization, boundary preservation or parity.** It demonstrates metadata adoption and an unexpected shared-branch consequence. Generic retarget regression success does not establish those properties for Llama, and this task makes no Gemma4 change.

All 70 focused existing tests pass with zero skips; the investigation still exposes the controlled discrepancy. No standalone attention export was needed because the required real decoder graph was sufficient. No full suite, remote-model test, CUDA execution, ONE/ONERT or target-NPU backend validation was run. Fake-quantized reference execution is not bit-exact integer-backend validation; native probing was limited to extracted value-preserving Concats. The natural one-step discrepancies were not root-caused further. Current-path candidate equations are not numerically exclusive on the observed nonsaturating values, as explicitly documented.

All scripts, Circle files, traces, JSON and this report are under `/tmp/tico_llama_kv_concat_investigation/`. Production source and existing tests are unchanged; git working tree is clean; no commit, push or PR was created.

Full stage/operator/tensor tables and exact commands/outcomes follow. Artifact filenames are `decoder_{prefill|decode}_{natural|forced}_{noopt|o1}.circle`.



# Appendix A: extracted FX stage evidence

# Observed FX cache subgraphs

The tables below are extracted from real decoder-layer exports by `analyze_fx.py`. Names prefixed `%` are FX tensor nodes. Qparam notation is `dtype(scale,zero_point)`; all domains here are scalar per-tensor affine. `none` is absent TICO qparam metadata, not an assertion that fake-quant is absent. Shapes and Q/DQ arguments supply the pre-fold domain evidence.

## Snapshot schedule and noopt/O1 relationship

- prefill/natural, torch_export: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, DecomposeFakeQuantize: 3 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, DecomposeFakeQuantizeTensorQParams: 2 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, FoldQuantOps: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, PropagateQParamForward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, PropagateQParamBackward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/natural, before_serialization: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, torch_export: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, DecomposeFakeQuantize: 3 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, DecomposeFakeQuantizeTensorQParams: 2 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, FoldQuantOps: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, PropagateQParamForward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, PropagateQParamBackward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- prefill/forced, before_serialization: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, torch_export: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, DecomposeFakeQuantize: 3 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, DecomposeFakeQuantizeTensorQParams: 2 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, FoldQuantOps: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, PropagateQParamForward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, PropagateQParamBackward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/natural, before_serialization: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, torch_export: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, DecomposeFakeQuantize: 3 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, DecomposeFakeQuantizeTensorQParams: 2 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, FoldQuantOps: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, PropagateQParamForward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, PropagateQParamBackward: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.
- decode/forced, before_serialization: 1 invocation(s); noopt/O1 node args, shapes, qparams and users identical: **True**.

Every invocation was inspected. Scalar fake-quant decomposition ran three times (calls 0/1/2); tensor-qparam decomposition ran twice (0/1). Folding, forward propagation, backward propagation, and pre-serialization capture each ran once. For the tables, decomposition means the final tensor-qparam decomposition snapshot, after all observed decomposition restarts. There is no later backward/forward fixed-point repair beyond the captured calls.

## What changed

- Both decode KV Concats are robustly identified as `cat_1` (K) and `cat_2` (V), dimension 1, `(1,7,8)+(1,1,8)->(1,8,8)`. K users are `permute` / `permute_1` leading to the two QK matmuls; V users are `bmm_1` / `bmm_3`. Their output node names are the Concat node names themselves.
- Initial export has no TICO qparam metadata. It does contain actual fake-quant calls on past inputs, K-rotation output, present-K/V Concat outputs, and stacked delta outputs. Decomposition makes those boundaries explicit Q-DQ pairs.
- Folding removes past Q-DQ into each past placeholder and propagates that grid through its legalized head reshape. It folds present Q-DQ into each cache Concat. There is **no explicit Quantize on either decode cache-Concat input immediately after folding**, for either scenario.
- Natural V past grid changes from `uint8(0.010872464627027512,116)` to `uint8(0.012130454182624817,104)` in backward propagation, including the Circle-input placeholder. Natural K past/present/current grids were already identical.
- Forced backward propagation changes both past placeholders and head reshapes from R=`uint8(0.1875,127)` to C=`uint8(0.3125,91)`. It also changes shared K current producer `add` and V current producer `linear_2` to C. These producers each have two users: attention Concat and the delta-output branch. The transition uses metadata overwrites, not Quantize argument retargeting.
- In forced decode, delta-only one-input Concats `cat_6`/`cat_7` and their reshapes retain old natural producer domains Pk=`uint8(0.004473801702260971,128)` / Pv=`uint8(0.012130454182624817,104)` after their sources were changed to C. Thus **a value-preserving one-input CAT now has different input/output grids**. The branch retains explicit output Quantizes to N=`uint8(0.0625,113)`. This inconsistency is already present before Circle serialization.
- Reverse traversal explains the stale branch: later delta `cat_6`/`cat_7` are visited before earlier attention `cat_1`/`cat_2`. The delta branches initially propagate P into the shared producers; then attention Concat overwrites those producers with C. The pass ends without another forward traversal to reconcile delta metadata.
- Controlled prefill preserves separate explicit present-Q_C boundaries (`quantize_per_tensor_22` K and `_23` V); these have two consumers and their upstream P producer domains are preserved. Its delta branch retains explicit Q_N (`_47`/`_48`). There is no past/current cache Concat in prefill. Natural prefill equal-grid present and delta boundaries fold away.
- No existing explicit Quantize scale/zero-point argument changed between Fold, Forward, Backward and Serialization on any cache branch. Explicit retargeting capability is present in the code but is not exercised by the decoder cache-Concat inputs here.

## Observer-boundary mapping

| Logical boundary | Initial export | After folding/final effect |
|---|---|---|
| decode obs_past_key/value | `fake_quantize_per_tensor_affine_6` / `_7` directly on `past_key_value_0` / `_1` | Q-DQ folds into placeholders; head extraction legalizes to `reshape_default_21` / `_22`; backward overwrites R with Concat C when different |
| decode obs_present_key/value | `fake_quantize_per_tensor_affine_24` / `_25` after `cat_1` / `cat_2` | metadata on `cat_1` / `cat_2`; no surviving Q_C operation |
| decode obs_new_k/v | `fake_quantize_per_tensor_affine_50` / `_51` after `stack_2` / `_3` | stack legalizes to one-input cat+reshape; natural equal N folds away; forced Q_N `_49` / `_50` survive after reshape17/18 |
| prefill obs_present_key/value | `fake_quantize_per_tensor_affine_22` / `_23` on current K/V | natural folded; forced `_22` / `_23` survive as explicit Q_C on attention branches |
| prefill obs_new_k/v | `fake_quantize_per_tensor_affine_48` / `_49` after stack2/3 | natural folded; forced Q_N `_47` / `_48` survive |

Trace/provenance corroboration: the raw export cache nodes have `nn_module_stack` including `LlamaDecoderLayer*ExportAdapter`, `QuantLlamaDecoderLayer`, `PTQWrapper`, `QuantLlamaAttention`; past fake-quant stack points at `_normalize_past_key_value` via `quant_attention.py:1043`; unrolled compute/delta boundaries point at `_forward_unrolled` via `:1079`. The trace retained by Torch does not include every inner helper line, so exact observer labels are cross-checked against actual initial graph edges and decomposed numeric values rather than inferred solely from a missing stack substring.

## natural decode: relevant nodes at each stage

### torch_export (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_6 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_7 |
| `fake_quantize_per_tensor_affine_6` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%past_key_value_0', '%to_12', '%to_13', 0, 255]` | `[1, 1, 7, 8]` | none | select_2 |
| `fake_quantize_per_tensor_affine_7` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%past_key_value_1', '%to_14', '%to_15', 0, 255]` | `[1, 1, 7, 8]` | none | select_3 |
| `linear_2` | `linear.default` | `['%fake_quantize_per_tensor_affine_13', '%fake_quantize_per_channel_affine_2']` | `[1, 1, 8]` | none | fake_quantize_per_tensor_affine_14 |
| `fake_quantize_per_tensor_affine_14` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%linear_2', '%to_34', '%to_35', 0, 255]` | `[1, 1, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%fake_quantize_per_tensor_affine_14', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 1, 8]` | none | cat_2, stack_3 |
| `add` | `add.Tensor` | `['%fake_quantize_per_tensor_affine_21', '%fake_quantize_per_tensor_affine_22']` | `[1, 1, 8]` | none | fake_quantize_per_tensor_affine_23 |
| `fake_quantize_per_tensor_affine_23` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%add', '%to_52', '%to_53', 0, 255]` | `[1, 1, 8]` | none | cat_1, stack_2 |
| `select_2` | `select.int` | `['%fake_quantize_per_tensor_affine_6', 1, 0]` | `[1, 7, 8]` | none | cat_1 |
| `select_3` | `select.int` | `['%fake_quantize_per_tensor_affine_7', 1, 0]` | `[1, 7, 8]` | none | cat_2 |
| `cat_1` | `cat.default` | `[['%select_2', '%fake_quantize_per_tensor_affine_23'], 1]` | `[1, 8, 8]` | none | fake_quantize_per_tensor_affine_24 |
| `fake_quantize_per_tensor_affine_24` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%cat_1', '%to_54', '%to_55', 0, 255]` | `[1, 8, 8]` | none | transpose, transpose_1 |
| `cat_2` | `cat.default` | `[['%select_3', '%select_1'], 1]` | `[1, 8, 8]` | none | fake_quantize_per_tensor_affine_25 |
| `fake_quantize_per_tensor_affine_25` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%cat_2', '%to_56', '%to_57', 0, 255]` | `[1, 8, 8]` | none | matmul_1, matmul_3 |
| `stack_2` | `stack.default` | `[['%fake_quantize_per_tensor_affine_23'], 1]` | `[1, 1, 1, 8]` | none | fake_quantize_per_tensor_affine_50 |
| `fake_quantize_per_tensor_affine_50` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_2', '%to_110', '%to_111', 0, 255]` | `[1, 1, 1, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 1, 8]` | none | fake_quantize_per_tensor_affine_51 |
| `fake_quantize_per_tensor_affine_51` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_3', '%to_112', '%to_113', 0, 255]` | `[1, 1, 1, 8]` | none | output |

### DecomposeFakeQuantizeTensorQParams (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_6 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_7 |
| `quantize_per_tensor_default_6` | `qd.quantize_per_tensor.default` | `['%past_key_value_0', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_6 |
| `dequantize_per_tensor_default_6` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_6', 0.004473801702260971, 128, 0, 255]` | `[1, 1, 7, 8]` | none | select_2 |
| `quantize_per_tensor_default_7` | `qd.quantize_per_tensor.default` | `['%past_key_value_1', 0.010872464627027512, 116, 0, 255]` | `None` | none | dequantize_per_tensor_default_7 |
| `dequantize_per_tensor_default_7` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_7', 0.010872464627027512, 116, 0, 255]` | `[1, 1, 7, 8]` | none | select_3 |
| `quantize_per_tensor_default_14` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_14 |
| `dequantize_per_tensor_default_14` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_14', 0.012130454182624817, 104, 0, 255]` | `[1, 1, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%dequantize_per_tensor_default_14', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 1, 8]` | none | cat_2, stack_3 |
| `add` | `add.Tensor` | `['%dequantize_per_tensor_default_21', '%dequantize_per_tensor_default_22']` | `[1, 1, 8]` | none | quantize_per_tensor_default_23 |
| `quantize_per_tensor_default_23` | `qd.quantize_per_tensor.default` | `['%add', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_23 |
| `dequantize_per_tensor_default_23` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_23', 0.004473801702260971, 128, 0, 255]` | `[1, 1, 8]` | none | cat_1, stack_2 |
| `select_2` | `select.int` | `['%dequantize_per_tensor_default_6', 1, 0]` | `[1, 7, 8]` | none | cat_1 |
| `select_3` | `select.int` | `['%dequantize_per_tensor_default_7', 1, 0]` | `[1, 7, 8]` | none | cat_2 |
| `cat_1` | `cat.default` | `[['%select_2', '%dequantize_per_tensor_default_23'], 1]` | `[1, 8, 8]` | none | quantize_per_tensor_default_24 |
| `quantize_per_tensor_default_24` | `qd.quantize_per_tensor.default` | `['%cat_1', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_24 |
| `cat_2` | `cat.default` | `[['%select_3', '%select_1'], 1]` | `[1, 8, 8]` | none | quantize_per_tensor_default_25 |
| `quantize_per_tensor_default_25` | `qd.quantize_per_tensor.default` | `['%cat_2', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_25 |
| `stack_2` | `stack.default` | `[['%dequantize_per_tensor_default_23'], 1]` | `[1, 1, 1, 8]` | none | quantize_per_tensor_default_49 |
| `quantize_per_tensor_default_49` | `qd.quantize_per_tensor.default` | `['%stack_2', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_49 |
| `dequantize_per_tensor_default_49` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_49', 0.004473801702260971, 128, 0, 255]` | `[1, 1, 1, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 1, 8]` | none | quantize_per_tensor_default_50 |
| `quantize_per_tensor_default_50` | `qd.quantize_per_tensor.default` | `['%stack_3', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_50 |
| `dequantize_per_tensor_default_50` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_50', 0.012130454182624817, 104, 0, 255]` | `[1, 1, 1, 8]` | none | output |

### FoldQuantOps (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.010872464627027512,116) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%dequantize_per_channel_2']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.010872464627027512,116) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.012130454182624817,104) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | none | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | none | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | output |

### PropagateQParamForward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.010872464627027512,116) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.010872464627027512,116) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.012130454182624817,104) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | output |

### PropagateQParamBackward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.012130454182624817,104) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | output |

### before_serialization (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.012130454182624817,104) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.004473801702260971,128) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | output |

## natural prefill: relevant nodes at each stage

### torch_export (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%fake_quantize_per_tensor_affine_11', '%fake_quantize_per_channel_affine_2']` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_12 |
| `fake_quantize_per_tensor_affine_12` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%linear_2', '%to_30', '%to_31', 0, 255]` | `[1, 7, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%fake_quantize_per_tensor_affine_12', [1, 7, 1, 8]]` | `[1, 7, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_23, stack_3 |
| `add` | `add.Tensor` | `['%fake_quantize_per_tensor_affine_19', '%fake_quantize_per_tensor_affine_20']` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_21 |
| `fake_quantize_per_tensor_affine_21` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%add', '%to_48', '%to_49', 0, 255]` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_22, stack_2 |
| `fake_quantize_per_tensor_affine_22` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%fake_quantize_per_tensor_affine_21', '%to_50', '%to_51', 0, 255]` | `[1, 7, 8]` | none | transpose, transpose_1 |
| `fake_quantize_per_tensor_affine_23` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%select_1', '%to_52', '%to_53', 0, 255]` | `[1, 7, 8]` | none | matmul_1, matmul_3 |
| `stack_2` | `stack.default` | `[['%fake_quantize_per_tensor_affine_21'], 1]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_48 |
| `fake_quantize_per_tensor_affine_48` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_2', '%to_106', '%to_107', 0, 255]` | `[1, 1, 7, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_49 |
| `fake_quantize_per_tensor_affine_49` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_3', '%to_108', '%to_109', 0, 255]` | `[1, 1, 7, 8]` | none | output |

### DecomposeFakeQuantizeTensorQParams (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%dequantize_per_tensor_default_11', '%dequantize_per_channel_default_2']` | `[1, 7, 8]` | none | quantize_per_tensor_default_12 |
| `dequantize_per_tensor_default_12` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_12', 0.012130454182624817, 104, 0, 255]` | `[1, 7, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%dequantize_per_tensor_default_12', [1, 7, 1, 8]]` | `[1, 7, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 7, 8]` | none | stack_3, quantize_per_tensor_default_23 |
| `add` | `add.Tensor` | `['%dequantize_per_tensor_default_19', '%dequantize_per_tensor_default_20']` | `[1, 7, 8]` | none | quantize_per_tensor_default_21 |
| `quantize_per_tensor_default_21` | `qd.quantize_per_tensor.default` | `['%add', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_21 |
| `dequantize_per_tensor_default_21` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_21', 0.004473801702260971, 128, 0, 255]` | `[1, 7, 8]` | none | stack_2, quantize_per_tensor_default_22 |
| `quantize_per_tensor_default_22` | `qd.quantize_per_tensor.default` | `['%dequantize_per_tensor_default_21', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_22 |
| `quantize_per_tensor_default_23` | `qd.quantize_per_tensor.default` | `['%select_1', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_23 |
| `stack_2` | `stack.default` | `[['%dequantize_per_tensor_default_21'], 1]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_47 |
| `quantize_per_tensor_default_47` | `qd.quantize_per_tensor.default` | `['%stack_2', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_47 |
| `dequantize_per_tensor_default_47` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_47', 0.004473801702260971, 128, 0, 255]` | `[1, 1, 7, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_48 |
| `quantize_per_tensor_default_48` | `qd.quantize_per_tensor.default` | `['%stack_3', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_48 |
| `dequantize_per_tensor_default_48` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_48', 0.012130454182624817, 104, 0, 255]` | `[1, 1, 7, 8]` | none | output |

### FoldQuantOps (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%dequantize_per_channel_2']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_5, bmm_1, bmm_3 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_4, permute, permute_1 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | output |

### PropagateQParamForward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_5, bmm_1, bmm_3 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_4, permute, permute_1 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | output |

### PropagateQParamBackward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_5, bmm_1, bmm_3 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_4, permute, permute_1 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | output |

### before_serialization (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | cat_5, bmm_1, bmm_3 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | cat_4, permute, permute_1 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | output |

## forced decode: relevant nodes at each stage

### torch_export (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_6 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_7 |
| `fake_quantize_per_tensor_affine_6` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%past_key_value_0', '%to_12', '%to_13', 0, 255]` | `[1, 1, 7, 8]` | none | select_2 |
| `fake_quantize_per_tensor_affine_7` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%past_key_value_1', '%to_14', '%to_15', 0, 255]` | `[1, 1, 7, 8]` | none | select_3 |
| `linear_2` | `linear.default` | `['%fake_quantize_per_tensor_affine_13', '%fake_quantize_per_channel_affine_2']` | `[1, 1, 8]` | none | fake_quantize_per_tensor_affine_14 |
| `fake_quantize_per_tensor_affine_14` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%linear_2', '%to_34', '%to_35', 0, 255]` | `[1, 1, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%fake_quantize_per_tensor_affine_14', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 1, 8]` | none | cat_2, stack_3 |
| `add` | `add.Tensor` | `['%fake_quantize_per_tensor_affine_21', '%fake_quantize_per_tensor_affine_22']` | `[1, 1, 8]` | none | fake_quantize_per_tensor_affine_23 |
| `fake_quantize_per_tensor_affine_23` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%add', '%to_52', '%to_53', 0, 255]` | `[1, 1, 8]` | none | cat_1, stack_2 |
| `select_2` | `select.int` | `['%fake_quantize_per_tensor_affine_6', 1, 0]` | `[1, 7, 8]` | none | cat_1 |
| `select_3` | `select.int` | `['%fake_quantize_per_tensor_affine_7', 1, 0]` | `[1, 7, 8]` | none | cat_2 |
| `cat_1` | `cat.default` | `[['%select_2', '%fake_quantize_per_tensor_affine_23'], 1]` | `[1, 8, 8]` | none | fake_quantize_per_tensor_affine_24 |
| `fake_quantize_per_tensor_affine_24` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%cat_1', '%to_54', '%to_55', 0, 255]` | `[1, 8, 8]` | none | transpose, transpose_1 |
| `cat_2` | `cat.default` | `[['%select_3', '%select_1'], 1]` | `[1, 8, 8]` | none | fake_quantize_per_tensor_affine_25 |
| `fake_quantize_per_tensor_affine_25` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%cat_2', '%to_56', '%to_57', 0, 255]` | `[1, 8, 8]` | none | matmul_1, matmul_3 |
| `stack_2` | `stack.default` | `[['%fake_quantize_per_tensor_affine_23'], 1]` | `[1, 1, 1, 8]` | none | fake_quantize_per_tensor_affine_50 |
| `fake_quantize_per_tensor_affine_50` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_2', '%to_110', '%to_111', 0, 255]` | `[1, 1, 1, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 1, 8]` | none | fake_quantize_per_tensor_affine_51 |
| `fake_quantize_per_tensor_affine_51` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_3', '%to_112', '%to_113', 0, 255]` | `[1, 1, 1, 8]` | none | output |

### DecomposeFakeQuantizeTensorQParams (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_6 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_7 |
| `quantize_per_tensor_default_6` | `qd.quantize_per_tensor.default` | `['%past_key_value_0', 0.1875, 127, 0, 255]` | `None` | none | dequantize_per_tensor_default_6 |
| `dequantize_per_tensor_default_6` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_6', 0.1875, 127, 0, 255]` | `[1, 1, 7, 8]` | none | select_2 |
| `quantize_per_tensor_default_7` | `qd.quantize_per_tensor.default` | `['%past_key_value_1', 0.1875, 127, 0, 255]` | `None` | none | dequantize_per_tensor_default_7 |
| `dequantize_per_tensor_default_7` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_7', 0.1875, 127, 0, 255]` | `[1, 1, 7, 8]` | none | select_3 |
| `quantize_per_tensor_default_14` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.012130454182624817, 104, 0, 255]` | `None` | none | dequantize_per_tensor_default_14 |
| `dequantize_per_tensor_default_14` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_14', 0.012130454182624817, 104, 0, 255]` | `[1, 1, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%dequantize_per_tensor_default_14', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 1, 8]` | none | cat_2, stack_3 |
| `add` | `add.Tensor` | `['%dequantize_per_tensor_default_21', '%dequantize_per_tensor_default_22']` | `[1, 1, 8]` | none | quantize_per_tensor_default_23 |
| `quantize_per_tensor_default_23` | `qd.quantize_per_tensor.default` | `['%add', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_23 |
| `dequantize_per_tensor_default_23` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_23', 0.004473801702260971, 128, 0, 255]` | `[1, 1, 8]` | none | cat_1, stack_2 |
| `select_2` | `select.int` | `['%dequantize_per_tensor_default_6', 1, 0]` | `[1, 7, 8]` | none | cat_1 |
| `select_3` | `select.int` | `['%dequantize_per_tensor_default_7', 1, 0]` | `[1, 7, 8]` | none | cat_2 |
| `cat_1` | `cat.default` | `[['%select_2', '%dequantize_per_tensor_default_23'], 1]` | `[1, 8, 8]` | none | quantize_per_tensor_default_24 |
| `quantize_per_tensor_default_24` | `qd.quantize_per_tensor.default` | `['%cat_1', 0.3125, 91, 0, 255]` | `None` | none | dequantize_per_tensor_default_24 |
| `cat_2` | `cat.default` | `[['%select_3', '%select_1'], 1]` | `[1, 8, 8]` | none | quantize_per_tensor_default_25 |
| `quantize_per_tensor_default_25` | `qd.quantize_per_tensor.default` | `['%cat_2', 0.3125, 91, 0, 255]` | `None` | none | dequantize_per_tensor_default_25 |
| `stack_2` | `stack.default` | `[['%dequantize_per_tensor_default_23'], 1]` | `[1, 1, 1, 8]` | none | quantize_per_tensor_default_49 |
| `quantize_per_tensor_default_49` | `qd.quantize_per_tensor.default` | `['%stack_2', 0.0625, 113, 0, 255]` | `None` | none | dequantize_per_tensor_default_49 |
| `dequantize_per_tensor_default_49` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_49', 0.0625, 113, 0, 255]` | `[1, 1, 1, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 1, 8]` | none | quantize_per_tensor_default_50 |
| `quantize_per_tensor_default_50` | `qd.quantize_per_tensor.default` | `['%stack_3', 0.0625, 113, 0, 255]` | `None` | none | dequantize_per_tensor_default_50 |
| `dequantize_per_tensor_default_50` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_50', 0.0625, 113, 0, 255]` | `[1, 1, 1, 8]` | none | output |

### FoldQuantOps (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.1875,127) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.1875,127) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%dequantize_per_channel_2']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.1875,127) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.1875,127) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | none | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | none | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_49 |
| `quantize_per_tensor_49` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_50 |
| `quantize_per_tensor_50` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |

### PropagateQParamForward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.1875,127) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.1875,127) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.1875,127) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.1875,127) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_49 |
| `quantize_per_tensor_49` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_50 |
| `quantize_per_tensor_50` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |

### PropagateQParamBackward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.3125,91) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.3125,91) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.3125,91) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.3125,91) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.3125,91) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.3125,91) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_49 |
| `quantize_per_tensor_49` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_50 |
| `quantize_per_tensor_50` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |

### before_serialization (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `past_key_value_0` | `past_key_value_0` | `[]` | `[1, 1, 7, 8]` | uint8(0.3125,91) | reshape_default_21 |
| `past_key_value_1` | `past_key_value_1` | `[]` | `[1, 1, 7, 8]` | uint8(0.3125,91) | reshape_default_22 |
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 1, 8]` | uint8(0.3125,91) | cat_7, cat_2 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 1, 8]` | uint8(0.3125,91) | cat_6, cat_1 |
| `reshape_default_21` | `reshape.default` | `['%past_key_value_0', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.3125,91) | cat_1 |
| `reshape_default_22` | `reshape.default` | `['%past_key_value_1', [1, 7, 8]]` | `[1, 7, 8]` | uint8(0.3125,91) | cat_2 |
| `cat_1` | `cat.default` | `[['%reshape_default_21', '%add'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `cat_2` | `cat.default` | `[['%reshape_default_22', '%linear_2'], 1]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `permute` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm |
| `bmm_1` | `bmm.default` | `['%_softmax', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `permute_1` | `permute.default` | `['%cat_1', [0, 2, 1]]` | `[1, 8, 8]` | uint8(0.3125,91) | bmm_2 |
| `bmm_3` | `bmm.default` | `['%_softmax_1', '%cat_2']` | `[1, 1, 8]` | uint8(0.008762814104557037,135) | cat_5 |
| `cat_6` | `cat.default` | `[['%add'], 1]` | `[1, 1, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_6', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_49 |
| `quantize_per_tensor_49` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |
| `cat_7` | `cat.default` | `[['%linear_2'], 1]` | `[1, 1, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_7', [1, 1, 1, 8]]` | `[1, 1, 1, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_50 |
| `quantize_per_tensor_50` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 1, 8]` | uint8(0.0625,113) | output |

## forced prefill: relevant nodes at each stage

### torch_export (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%fake_quantize_per_tensor_affine_11', '%fake_quantize_per_channel_affine_2']` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_12 |
| `fake_quantize_per_tensor_affine_12` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%linear_2', '%to_30', '%to_31', 0, 255]` | `[1, 7, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%fake_quantize_per_tensor_affine_12', [1, 7, 1, 8]]` | `[1, 7, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_23, stack_3 |
| `add` | `add.Tensor` | `['%fake_quantize_per_tensor_affine_19', '%fake_quantize_per_tensor_affine_20']` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_21 |
| `fake_quantize_per_tensor_affine_21` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%add', '%to_48', '%to_49', 0, 255]` | `[1, 7, 8]` | none | fake_quantize_per_tensor_affine_22, stack_2 |
| `fake_quantize_per_tensor_affine_22` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%fake_quantize_per_tensor_affine_21', '%to_50', '%to_51', 0, 255]` | `[1, 7, 8]` | none | transpose, transpose_1 |
| `fake_quantize_per_tensor_affine_23` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%select_1', '%to_52', '%to_53', 0, 255]` | `[1, 7, 8]` | none | matmul_1, matmul_3 |
| `stack_2` | `stack.default` | `[['%fake_quantize_per_tensor_affine_21'], 1]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_48 |
| `fake_quantize_per_tensor_affine_48` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_2', '%to_106', '%to_107', 0, 255]` | `[1, 1, 7, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 7, 8]` | none | fake_quantize_per_tensor_affine_49 |
| `fake_quantize_per_tensor_affine_49` | `fake_quantize_per_tensor_affine.tensor_qparams` | `['%stack_3', '%to_108', '%to_109', 0, 255]` | `[1, 1, 7, 8]` | none | output |

### DecomposeFakeQuantizeTensorQParams (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%dequantize_per_tensor_default_11', '%dequantize_per_channel_default_2']` | `[1, 7, 8]` | none | quantize_per_tensor_default_12 |
| `dequantize_per_tensor_default_12` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_12', 0.012130454182624817, 104, 0, 255]` | `[1, 7, 8]` | none | view_2 |
| `view_2` | `view.default` | `['%dequantize_per_tensor_default_12', [1, 7, 1, 8]]` | `[1, 7, 1, 8]` | none | select_1 |
| `select_1` | `select.int` | `['%view_2', 2, 0]` | `[1, 7, 8]` | none | stack_3, quantize_per_tensor_default_23 |
| `add` | `add.Tensor` | `['%dequantize_per_tensor_default_19', '%dequantize_per_tensor_default_20']` | `[1, 7, 8]` | none | quantize_per_tensor_default_21 |
| `quantize_per_tensor_default_21` | `qd.quantize_per_tensor.default` | `['%add', 0.004473801702260971, 128, 0, 255]` | `None` | none | dequantize_per_tensor_default_21 |
| `dequantize_per_tensor_default_21` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_21', 0.004473801702260971, 128, 0, 255]` | `[1, 7, 8]` | none | stack_2, quantize_per_tensor_default_22 |
| `quantize_per_tensor_default_22` | `qd.quantize_per_tensor.default` | `['%dequantize_per_tensor_default_21', 0.3125, 91, 0, 255]` | `None` | none | dequantize_per_tensor_default_22 |
| `quantize_per_tensor_default_23` | `qd.quantize_per_tensor.default` | `['%select_1', 0.3125, 91, 0, 255]` | `None` | none | dequantize_per_tensor_default_23 |
| `stack_2` | `stack.default` | `[['%dequantize_per_tensor_default_21'], 1]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_47 |
| `quantize_per_tensor_default_47` | `qd.quantize_per_tensor.default` | `['%stack_2', 0.0625, 113, 0, 255]` | `None` | none | dequantize_per_tensor_default_47 |
| `dequantize_per_tensor_default_47` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_47', 0.0625, 113, 0, 255]` | `[1, 1, 7, 8]` | none | output |
| `stack_3` | `stack.default` | `[['%select_1'], 1]` | `[1, 1, 7, 8]` | none | quantize_per_tensor_default_48 |
| `quantize_per_tensor_default_48` | `qd.quantize_per_tensor.default` | `['%stack_3', 0.0625, 113, 0, 255]` | `None` | none | dequantize_per_tensor_default_48 |
| `dequantize_per_tensor_default_48` | `qd.dequantize_per_tensor.default` | `['%quantize_per_tensor_default_48', 0.0625, 113, 0, 255]` | `[1, 1, 7, 8]` | none | output |

### FoldQuantOps (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%dequantize_per_channel_2']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_23, cat_5 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_22, cat_4 |
| `quantize_per_tensor_22` | `qd.quantize_per_tensor.default` | `['%add', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `quantize_per_tensor_23` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_47 |
| `quantize_per_tensor_47` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_48 |
| `quantize_per_tensor_48` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |

### PropagateQParamForward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_23, cat_5 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_22, cat_4 |
| `quantize_per_tensor_22` | `qd.quantize_per_tensor.default` | `['%add', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `quantize_per_tensor_23` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_47 |
| `quantize_per_tensor_47` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_48 |
| `quantize_per_tensor_48` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |

### PropagateQParamBackward (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_23, cat_5 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_22, cat_4 |
| `quantize_per_tensor_22` | `qd.quantize_per_tensor.default` | `['%add', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `quantize_per_tensor_23` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_47 |
| `quantize_per_tensor_47` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_48 |
| `quantize_per_tensor_48` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |

### before_serialization (last observed invocation)

| Node | Target | Data inputs / arguments | Shape | TICO qparam | Users |
|---|---|---|---|---|---|
| `linear_2` | `linear.default` | `['%rms_norm', '%_prop_tensor_constant141']` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_23, cat_5 |
| `add` | `add.Tensor` | `['%mul', '%mul_1']` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_22, cat_4 |
| `quantize_per_tensor_22` | `qd.quantize_per_tensor.default` | `['%add', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | permute, permute_1 |
| `quantize_per_tensor_23` | `qd.quantize_per_tensor.default` | `['%linear_2', 0.3125, 91, 0, 255, 'torch.uint8']` | `[1, 7, 8]` | uint8(0.3125,91) | bmm_1, bmm_3 |
| `cat_4` | `cat.default` | `[['%add'], 1]` | `[1, 7, 8]` | uint8(0.004473801702260971,128) | reshape_default_17 |
| `reshape_default_17` | `reshape.default` | `['%cat_4', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.004473801702260971,128) | quantize_per_tensor_47 |
| `quantize_per_tensor_47` | `qd.quantize_per_tensor.default` | `['%reshape_default_17', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |
| `cat_5` | `cat.default` | `[['%linear_2'], 1]` | `[1, 7, 8]` | uint8(0.012130454182624817,104) | reshape_default_18 |
| `reshape_default_18` | `reshape.default` | `['%cat_5', [1, 1, 7, 8]]` | `[1, 1, 7, 8]` | uint8(0.012130454182624817,104) | quantize_per_tensor_48 |
| `quantize_per_tensor_48` | `qd.quantize_per_tensor.default` | `['%reshape_default_18', 0.0625, 113, 0, 255, 'torch.uint8']` | `[1, 1, 7, 8]` | uint8(0.0625,113) | output |

## Explicit cache Quantize argument/user audit

| Export | Stage | Node | Scale | Zero point | Range | Dtype argument | Users |
|---|---|---|---|---|---|---|---|
| decode/forced | FoldQuantOps | quantize_per_tensor_49 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | FoldQuantOps | quantize_per_tensor_50 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | PropagateQParamForward | quantize_per_tensor_49 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | PropagateQParamForward | quantize_per_tensor_50 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | PropagateQParamBackward | quantize_per_tensor_49 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | PropagateQParamBackward | quantize_per_tensor_50 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | before_serialization | quantize_per_tensor_49 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| decode/forced | before_serialization | quantize_per_tensor_50 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | FoldQuantOps | quantize_per_tensor_22 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: permute, permute_1 |
| prefill/forced | FoldQuantOps | quantize_per_tensor_23 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: bmm_1, bmm_3 |
| prefill/forced | FoldQuantOps | quantize_per_tensor_47 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | FoldQuantOps | quantize_per_tensor_48 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | PropagateQParamForward | quantize_per_tensor_22 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: permute, permute_1 |
| prefill/forced | PropagateQParamForward | quantize_per_tensor_23 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: bmm_1, bmm_3 |
| prefill/forced | PropagateQParamForward | quantize_per_tensor_47 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | PropagateQParamForward | quantize_per_tensor_48 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | PropagateQParamBackward | quantize_per_tensor_22 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: permute, permute_1 |
| prefill/forced | PropagateQParamBackward | quantize_per_tensor_23 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: bmm_1, bmm_3 |
| prefill/forced | PropagateQParamBackward | quantize_per_tensor_47 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | PropagateQParamBackward | quantize_per_tensor_48 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | before_serialization | quantize_per_tensor_22 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: permute, permute_1 |
| prefill/forced | before_serialization | quantize_per_tensor_23 | 0.3125 | 91 | [0, 255] | torch.uint8 | 2: bmm_1, bmm_3 |
| prefill/forced | before_serialization | quantize_per_tensor_47 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
| prefill/forced | before_serialization | quantize_per_tensor_48 | 0.0625 | 113 | [0, 255] | torch.uint8 | 1: output |
Natural has no retained cache-branch Quantize after Fold; retained model Quantizes concern hidden/layernorm paths. Forced decode retained Quantizes are delta-output Q_N only; forced prefill retains attention Q_C and delta Q_N. None is an explicit requantizer on a past/current input of the decode KV Concat.



# Appendix B: final Circle tables, every Quantize, ABI and O1

# Circle graph observations

All models loaded with `CircleDocument.from_bytes` and passed `document.verify()` unless issues listed. Cache Concat identification requires axis 1 and exact decode edge shapes; K additionally feeds transpose, V directly feeds attention matmul. Operator order is not used for identification.

| artifact | operators | all QUANTIZE | Concat edge QUANTIZE | delta edge QUANTIZE | all KV edge QUANTIZE |
|---|---:|---:|---:|---:|---:|
| decoder_decode_forced_noopt.circle | 60 | 4 | 0 | 2 | 2 |
| decoder_decode_forced_o1.circle | 58 | 4 | 0 | 2 | 2 |
| decoder_decode_natural_noopt.circle | 58 | 2 | 0 | 0 | 0 |
| decoder_decode_natural_o1.circle | 56 | 2 | 0 | 0 | 0 |
| decoder_prefill_forced_noopt.circle | 58 | 6 | 0 | 2 | 4 |
| decoder_prefill_forced_o1.circle | 56 | 6 | 0 | 2 | 4 |
| decoder_prefill_natural_noopt.circle | 54 | 2 | 0 | 0 | 0 |
| decoder_prefill_natural_o1.circle | 52 | 2 | 0 | 0 | 0 |

## decoder_decode_forced_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 50:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 77 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 59:ADD | GRAPH_OUTPUT |
| delta K | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |
| delta V | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| delta K branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |
| delta K branch | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | 45:QUANTIZE |
| delta K graph output | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| delta V branch | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:CONCATENATION | 47:RESHAPE |
| delta V branch | 65 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 47:RESHAPE | 48:QUANTIZE |
| delta V graph output | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit


Operator 43:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |

Operator 46:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:CONCATENATION | 47:RESHAPE |

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 50:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 45; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | 45:QUANTIZE |
| QUANTIZE output | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |

Operator 48; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 65 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 47:RESHAPE | 48:QUANTIZE |
| QUANTIZE output | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

Operator 51; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 68 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 50:ADD | 51:QUANTIZE, 59:ADD |
| QUANTIZE output | 69 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 51:QUANTIZE | 52:RMS_NORM |

## decoder_decode_forced_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 75 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |
| delta V | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| delta K branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K branch | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| delta K graph output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| delta V branch | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |
| delta V branch | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| delta V graph output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit


Operator 41:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |

Operator 44:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 43; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| QUANTIZE output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

Operator 46; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| QUANTIZE output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 66 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 67 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_decode_natural_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 75 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | GRAPH_OUTPUT |
| delta V | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:RESHAPE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.004473801702260971] | [128] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 45:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.012130454182624817] | [104] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| delta K branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |
| delta K graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 45:CONCATENATION |
| delta V branch | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:CONCATENATION | 46:RESHAPE |
| delta V graph output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 66 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 67 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_decode_natural_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 73 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 55:ADD | GRAPH_OUTPUT |
| delta K | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | GRAPH_OUTPUT |
| delta V | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:RESHAPE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.004473801702260971] | [128] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 43:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.012130454182624817] | [104] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| delta K branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K graph output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 43:CONCATENATION |
| delta V branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 43:CONCATENATION | 44:RESHAPE |
| delta V graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 47; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 64 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 46:ADD | 47:QUANTIZE, 55:ADD |
| QUANTIZE output | 65 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 47:QUANTIZE | 48:RMS_NORM |

## decoder_prefill_forced_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 22:ADD, 34:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 17:MUL, 29:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 18:MUL, 30:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 73 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |
| delta V | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 41:CONCATENATION |
| delta K branch | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K branch | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| delta K graph output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 44:CONCATENATION |
| delta V branch | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |
| delta V branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| delta V graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 11; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 41:CONCATENATION |
| QUANTIZE output | 25 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:QUANTIZE | 20:TRANSPOSE, 32:TRANSPOSE |

Operator 12; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 44:CONCATENATION |
| QUANTIZE output | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_23 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:QUANTIZE | 24:BATCH_MATMUL, 36:BATCH_MATMUL |

Operator 43; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| QUANTIZE output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

Operator 46; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| QUANTIZE output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 64 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 65 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_prefill_forced_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 22:ADD, 34:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 17:MUL, 29:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 18:MUL, 30:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 71 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 55:ADD | GRAPH_OUTPUT |
| delta K | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |
| delta V | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 39:CONCATENATION |
| delta K branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 39:CONCATENATION | 40:RESHAPE |
| delta K branch | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | 41:QUANTIZE |
| delta K graph output | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 42:CONCATENATION |
| delta V branch | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:CONCATENATION | 43:RESHAPE |
| delta V branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 43:RESHAPE | 44:QUANTIZE |
| delta V graph output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 11; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 39:CONCATENATION |
| QUANTIZE output | 25 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:QUANTIZE | 20:TRANSPOSE, 32:TRANSPOSE |

Operator 12; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 42:CONCATENATION |
| QUANTIZE output | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_23 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:QUANTIZE | 24:BATCH_MATMUL, 36:BATCH_MATMUL |

Operator 41; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | 41:QUANTIZE |
| QUANTIZE output | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |

Operator 44; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 43:RESHAPE | 44:QUANTIZE |
| QUANTIZE output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

Operator 47; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 62 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 46:ADD | 47:QUANTIZE, 55:ADD |
| QUANTIZE output | 63 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 47:QUANTIZE | 48:RMS_NORM |

## decoder_prefill_natural_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 44:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 20:ADD, 32:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 15:MUL, 27:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 16:MUL, 28:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 69 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 53:ADD | GRAPH_OUTPUT |
| delta K | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | GRAPH_OUTPUT |
| delta V | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:RESHAPE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 18:TRANSPOSE, 30:TRANSPOSE, 39:CONCATENATION |
| delta K branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 39:CONCATENATION | 40:RESHAPE |
| delta K graph output | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 22:BATCH_MATMUL, 34:BATCH_MATMUL, 41:CONCATENATION |
| delta V branch | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 41:CONCATENATION | 42:RESHAPE |
| delta V graph output | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 44:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 45; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 60 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 44:ADD | 45:QUANTIZE, 53:ADD |
| QUANTIZE output | 61 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 45:QUANTIZE | 46:RMS_NORM |

## decoder_prefill_natural_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 42:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 20:ADD, 32:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 15:MUL, 27:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 16:MUL, 28:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 67 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 51:ADD | GRAPH_OUTPUT |
| delta K | 54 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 38:RESHAPE | GRAPH_OUTPUT |
| delta V | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 40:RESHAPE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 18:TRANSPOSE, 30:TRANSPOSE, 37:CONCATENATION |
| delta K branch | 53 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 37:CONCATENATION | 38:RESHAPE |
| delta K graph output | 54 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 38:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 22:BATCH_MATMUL, 34:BATCH_MATMUL, 39:CONCATENATION |
| delta V branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 39:CONCATENATION | 40:RESHAPE |
| delta V graph output | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 40:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 42:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 43; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 58 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 42:ADD | 43:QUANTIZE, 51:ADD |
| QUANTIZE output | 59 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 43:QUANTIZE | 44:RMS_NORM |

# Cache ABI comparison

| scenario / optimization | prefill K → decode past K | prefill V → decode past V | decode delta K → next past K | decode delta V → next past V |
|---|---|---|---|---|
| natural / noopt | identical (class 1) | identical (class 1) | identical (class 1) | identical (class 1) |
| natural / o1 | identical (class 1) | identical (class 1) | identical (class 1) | identical (class 1) |
| forced / noopt | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} |
| forced / o1 | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} |

# O1 comparison

Every O1 model has exactly two fewer operators. Comparing producer output tensor names identifies the removed operators as the attention-head layout round trip: `reshape_default_15` (shape [1,Q,2,8]) and `reshape_default_16` (shape [1,Q,16]). No QUANTIZE is removed. `SPLIT_V` is canonicalized to `SPLIT` (three instances). Cache input ABI and cache grids do not change. Forced decode retains the delta-branch qparam mismatch in both noopt and O1.

prefill/natural: removed noopt operators 36:RESHAPE (['reshape_default_15']), 37:RESHAPE (['reshape_default_16']).
decode/natural: removed noopt operators 40:RESHAPE (['reshape_default_15']), 41:RESHAPE (['reshape_default_16']).
prefill/forced: removed noopt operators 38:RESHAPE (['reshape_default_15']), 39:RESHAPE (['reshape_default_16']).
decode/forced: removed noopt operators 40:RESHAPE (['reshape_default_15']), 41:RESHAPE (['reshape_default_16']).

# Observed topology and ABI interpretation

The primary past/current decode concatenations are locally well formed: K and V each have identical input/output quantization parameters. Both noopt and O1 implement Case B on the past edge and Case C on the current edge. There is no Circle QUANTIZE on either edge.

```text
noopt, forced decode (main compute path)
  past graph input UINT8 C (.3125,91) -> RESHAPE C --+
                                                   +-> CONCAT axis1 C -> attention
  K ADD / V FULLY_CONNECTED output C --------------+

o1, forced decode (main compute path)
  past graph input UINT8 C (.3125,91) -> RESHAPE C --+
                                                   +-> CONCAT axis1 C -> attention
  K ADD / V FULLY_CONNECTED output C --------------+
```

The controlled model also exposes a separate Case D issue on its delta-output side branch. The shared K ADD / V FULLY_CONNECTED is retargeted to C on the compute edge, while the single-input Concat that implements the stack for the delta branch retains its earlier natural producer grid P. A value-preserving single-input Concat cannot implement the represented P transition. The following reshape retains P and an explicit Quantize then produces N. O1 preserves this mismatch.

```text
forced decode delta branch (both noopt/O1)
  current producer C -> singleton CONCAT output P -> RESHAPE P -> QUANTIZE N -> delta output
                       ^ qparam mismatch

forced prefill (both noopt/O1)
  current producer P --+-> QUANTIZE C -> attention compute
                       +-> singleton CONCAT P -> RESHAPE P -> QUANTIZE N -> delta output
```

Forced prefill output and forced decode delta output are N (.0625,113), whereas decode past inputs demand C (.3125,91). Because decode has no cache-input Quantize, this is ABI class 3: the host must explicitly requantize cache bytes N→C before passing them to decode. Copying UINT8 bytes directly interprets different real values. In this controlled same-scale-for-K/V example the host mapping is `q_C = clamp(round((q_N-113)*0.0625/0.3125)+91, 0, 255)`. Requantization reconciles storage grids but does not repair the separate stale-P metadata on the delta branch.

Natural prefill/delta outputs and natural decode past inputs have identical serialized qparams, so that measured boundary is class 1. This natural equality does not establish that arbitrary independently calibrated or deliberately distinct observer domains are chainable.

The reference fake-quantize runtime can pass values through the singleton Concat despite inconsistent P metadata because value-preserving operations do not round to their declared output grid. Successful structural verification and fake-quantize execution therefore do not validate this quantized storage-semantics mismatch or establish integer backend correctness.



# Appendix C: numerical evidence and probes

# Numerical evidence

Executed with CPU-only `CircleReferenceRuntime(...).run(..., mode=ExecutionMode.FAKE_QUANTIZE, trace=True)` against real serialized bytes. Eager tensors came from the frozen quantized decoder export adapters. No source or test changes were made.

## Exact eager/Circle comparisons

No tolerance was used to label equality. Differences below are observations, not tests declared passing through widened tolerances.

| Artifact | Output | Exact | Different / elements | Max absolute | Mean absolute |
|---|---|---:|---:|---:|---:|
| decoder_prefill_natural_noopt | hidden | True | 0 / 112 | 0 | 0 |
| decoder_prefill_natural_noopt | new_k | False | 1 / 56 | 0.004473801702 | 7.988931611e-05 |
| decoder_prefill_natural_noopt | new_v | True | 0 / 56 | 0 | 0 |
| decoder_prefill_natural_o1 | hidden | True | 0 / 112 | 0 | 0 |
| decoder_prefill_natural_o1 | new_k | False | 1 / 56 | 0.004473801702 | 7.988931611e-05 |
| decoder_prefill_natural_o1 | new_v | True | 0 / 56 | 0 | 0 |
| decoder_decode_natural_noopt | hidden | False | 1 / 16 | 0.02477002144 | 0.00154812634 |
| decoder_decode_natural_noopt | new_k | True | 0 / 8 | 0 | 0 |
| decoder_decode_natural_noopt | new_v | True | 0 / 8 | 0 | 0 |
| decoder_decode_natural_o1 | hidden | False | 1 / 16 | 0.02477002144 | 0.00154812634 |
| decoder_decode_natural_o1 | new_k | True | 0 / 8 | 0 | 0 |
| decoder_decode_natural_o1 | new_v | True | 0 / 8 | 0 | 0 |
| decoder_prefill_forced_noopt | hidden | True | 0 / 112 | 0 | 0 |
| decoder_prefill_forced_noopt | new_k | True | 0 / 56 | 0 | 0 |
| decoder_prefill_forced_noopt | new_v | True | 0 / 56 | 0 | 0 |
| decoder_prefill_forced_o1 | hidden | True | 0 / 112 | 0 | 0 |
| decoder_prefill_forced_o1 | new_k | True | 0 / 56 | 0 | 0 |
| decoder_prefill_forced_o1 | new_v | True | 0 / 56 | 0 | 0 |
| decoder_decode_forced_noopt | hidden | False | 9 / 16 | 0.02477011085 | 0.01393316686 |
| decoder_decode_forced_noopt | new_k | False | 7 / 8 | 0.125 | 0.078125 |
| decoder_decode_forced_noopt | new_v | False | 7 / 8 | 0.125 | 0.0703125 |
| decoder_decode_forced_o1 | hidden | False | 9 / 16 | 0.02477011085 | 0.01393316686 |
| decoder_decode_forced_o1 | new_k | False | 7 / 8 | 0.125 | 0.078125 |
| decoder_decode_forced_o1 | new_v | False | 7 / 8 | 0.125 | 0.0703125 |

All three outputs of each noopt model equal the corresponding O1 output exactly, for both scenarios and both prefill/decode. O1 did not change these numerical findings.

## Independent affine equations

`Q(s,z,x) = (clip(round(x/s)+z,0,255)-z)*s`; the independent NumPy implementation uses float32 and rounds ties away from zero, matching the documented Circle reference arithmetic. Eager Torch fake-quant uses nearest-even. The script does not call the runtime fake-quant helper to obtain expected values.

The fake-quantize runtime explicitly rounds graph inputs to their serialized grid (`executor.py`, graph-input binding). Therefore a real integer Circle ABI in grid C is represented in fake-quant execution by `Q_C(raw supplied float)`. This is input encoding in grid C, not a serialized QUANTIZE operation.

| Forced path (both noopt/O1) | Actual vs Q_C(raw) | Actual vs Q_C(Q_R(raw)) |
|---|---|---|
| Past K | exact | 8/56 differ; max 0.3125 |
| Past V | exact | 7/56 differ; max 0.3125 |

The actual past compute path is ABI-C input → value-preserving reshape → Concat. It performs no R-grid round trip in Circle. In eager, tuple past is first Q_R, then the full concatenation is Q_C. The intentionally off-grid past arrays distinguish these behaviors numerically.

Natural K happened to have R=C for this calibration, so both past equations coincide. Natural V actual matches Q_C(raw) exactly and differs from Q_C(Q_R(raw)) in 11/56 entries (maximum 0.09704363346).

## Current producer equations

For current K, raw values are reconstructed as the two operands of the serialized ADD summed in float32; for current V, raw values are reconstructed from serialized FULLY_CONNECTED input, dequantized weight and bias. This defines “raw” immediately before the final current producer output rounding; earlier observer/producer rounding remains in its operands.

| Forced current path | Actual producer | Actual vs Q_C(raw) | Actual vs Q_C(Q_N(raw)) |
|---|---|---|---|
| K | ADD 10 | exact, all 8 | exact, all 8 |
| V | FULLY_CONNECTED 5 | exact, all 8 | exact, all 8 |

The chosen scale ratio C/N = 0.3125/0.0625 = 5 makes these two candidate equations numerically identical for the sampled nonsaturating values. An odd integer scale ratio has coincident nearest-rounding decision boundaries. These values alone cannot prove the absence of an N round trip. The actual operator/tensor graph supplies that distinction: ADD/FC directly produces a C tensor, no Q_N lies upstream of the KV Concat, and explicit Q_N exists only on the downstream delta-output branch. No qparams were changed to manufacture a numerical separation.

Forced K raw from Circle ADD:

```text
[0.19947826862335205, -0.04683496803045273, 0.23958708345890045, -0.08374613523483276, 0.38486093282699585, 0.12073078751564026, -0.1190124973654747, -0.01993955671787262]
Q_C(raw) = [0.3125, 0.0, 0.3125, 0.0, 0.3125, 0.0, 0.0, 0.0]
```

Forced V raw from Circle FC:

```text
[0.275064617395401, -0.05965232849121094, 0.4499187469482422, -0.3033389151096344, 0.28487828373908997, -1.2063891887664795, 0.8972616195678711, 1.3630772829055786]
Q_C(raw) = [0.3125, 0.0, 0.3125, -0.3125, 0.3125, -1.25, 0.9375, 1.25]
```

## Algebraic witness using the unchanged controlled grids

The supplied C and N grids do not make the two current equations identical for all values: N clips beyond its representable range. An independent scalar diagnostic with `raw=9.125` gives `Q_C(raw)=9.0625`, `Q_N(raw)=8.875`, and `Q_C(Q_N(raw))=8.75`. For `raw=-7.5`, these are `-7.5`, `-7.0625`, and `-7.1875`, respectively. These are algebraic witnesses, not observed full-model activations. The real current ADD/FC tensors lie in the nonsaturating range where both candidate equations happen to agree; graph topology and qparams establish which path is serialized. No observer or model metadata was changed.

## Returned delta caches and metadata mismatch

Forced decode delta outputs equal Q_N(actual C current producer output) exactly. Eager delta outputs equal Q_N(the saved eager current head, already in K-rotation/V-projection grid) exactly. The C producer adoption therefore changes the returned delta values too.

```text
K Circle delta = [0.3125, 0.0, 0.3125, 0.0, 0.3125, 0.0, 0.0, 0.0]
K eager delta  = [0.1875, -0.0625, 0.25, -0.0625, 0.375, 0.125, -0.125, 0.0]
```

```text
V Circle delta = [0.3125, 0.0, 0.3125, -0.3125, 0.3125, -1.25, 0.9375, 1.25]
V eager delta  = [0.25, -0.0625, 0.4375, -0.3125, 0.25, -1.1875, 0.875, 1.375]
```

Forced decode noopt also has singleton CONCATENATION operators on these delta branches. K operator 43 copies current tensor 26 (C=.3125,91) into tensor 61 declared in the old K-rotation grid (.004473801702260971,128); V operator 46 copies tensor 20 (C) into tensor 64 declared in old V-projection grid (.012130454182624817,104). The following reshapes preserve those stale qparams, before explicit Q_N operators 45 and 48. These singleton Concat input/output qparams disagree; the fake-quant runtime treats Concat as value preserving, so its trace still contains C-grid real values. This is a semantic inconsistency in auxiliary delta-output branch metadata even though structural verification succeeds. O1 retains the corresponding mismatches; use the Circle tables for optimized indices.

## Rounding and limits

Torch and Circle use different tie rounding conventions. Small natural discrepancies are reported without assigning an unproven cause: natural prefill K differs at 1/56 entries by one K scale (0.004473801702260971), and natural decode hidden differs at 1/16 entries by approximately one final hidden scale (0.024770021438598633). Forced decode cache differences are much larger than an N-scale rounding tie: maximum 0.125 = two N steps, with 7/8 differing entries for both K/V, corroborated by the C adoption and exact independent equations.

FAKE_QUANTIZE validates semantic reference behavior, not bit-exact integer/NPU behavior. No target backend was run. No arbitrary tolerance was selected to turn mismatches into passes.

## Commands

```bash
cd /workspace/TICO
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/num_probe.py
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/analyze_numerics.py > /tmp/tico_llama_kv_concat_investigation/analyze_numerics.log 2>&1
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/write_numerical_report.py
```

All commands completed successfully. `num_probe.py` prints Circle Concat/Quantize metadata and captured eager head values; `analyze_numerics.py` writes all candidate comparisons, raw reconstructed producer values, samples and noopt/O1 parity to `numerical_analysis.json`; this script writes `numerical_analysis.md`.

## Native singleton-Concat contract probe

Using `extract_by_operator_indices`, extracted only the actual forced-decode singleton delta Concat operators (noopt K43/V46; O1 K41/V44), then ran each extracted artifact in default NATIVE reference mode. No tensor metadata was edited. Input UINT8 code 91 represents real zero in C. Each Concat copies code 91 to its output exactly, but the output metadata decodes it as -0.16553066298365593 for K and -0.15769590437412262 for V, in both noopt and O1. This directly demonstrates that the differing metadata changes the interpretation of copied integers; it is not an internal requantization. The probe contains only value-preserving Concat and makes no claim about backend-specific integer arithmetic.

Command (success):

```bash
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/probe_singleton_concat.py > /tmp/tico_llama_kv_concat_investigation/probe_singleton_concat.log 2>&1
```

Outputs: `singleton_concat_native_probe.json` and four temporary `delta_singleton_{k,v}_forced_{noopt,o1}.circle` artifacts.



# Appendix D: source review

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



# Appendix E: commands and outcomes

# Commands and outcomes

Initial execution directory was /workspace; repository did not exist.

```bash
git clone https://github.com/Samsung/TICO.git /workspace/TICO
git -C /workspace/TICO fetch origin main
git -C /workspace/TICO status --short
git -C /workspace/TICO rev-parse origin/main
```
Succeeded, clean checkout at c0a3cbcf9c67cb2e760185eddf4d7c71901e534d. Clone was already on main at this revision; no checkout/reset was needed.

```bash
mkdir -p /tmp/tico_llama_kv_concat_investigation
python -m venv /tmp/tico-venv
cd /workspace/TICO
source /tmp/tico-venv/bin/activate
./ccex install --cpu_only > /tmp/tico_llama_kv_concat_investigation/install.log 2>&1
./ccex configure test --cpu_only > /tmp/tico_llama_kv_concat_investigation/configure.log 2>&1
python -m pip check
```
Succeeded. Torch 2.13.0+cpu / TorchVision 0.28.0+cpu; repository-pinned Transformers 5.18.0. `pip check`: no broken requirements. Packages/source were downloaded; no model/tokenizer/checkpoint/dataset was downloaded.

Instructions and source inspection used `cat`, `sed -n`, and `rg` over root/quantization/Circle/test AGENTS, docs/design.md, docs/development.md, quantization/WrapQ/Circle READMEs, required wrapper/pass/convert files, observer affine_base.py, Circle runtime executor.py/quantization.py, and existing Llama/retarget tests. `find /workspace -name AGENTS.md -maxdepth 3`, `pwd`, `ls -la /workspace`, and bounded filesystem/environment tool discovery confirmed the empty initial workspace. `/etc/codex/network-policy.json` was read per the cloud-runtime skill. No secrets were printed.

```bash
HF_HUB_OFFLINE=1 python /tmp/tico_llama_kv_concat_investigation/investigate.py > /tmp/tico_llama_kv_concat_investigation/investigate.log 2>&1
```
The first temporary-script run stopped before export: observer API is `compute_qparams()`, not `get_qparams()`. Corrected this temporary script and corrected the runtime mode argument to `run(..., mode=ExecutionMode.FAKE_QUANTIZE)` before re-running. No production/test patch. The final run succeeded, generating 8 Circle files, all stage JSON snapshots, tensors/traces and numerical.json. Final script contains the precise Python API calls and instrumentation; no permanent logging changes.

```bash
python /tmp/tico_llama_kv_concat_investigation/inspect_circle.py
```
Succeeded (incremental run on 5 models, final run on all 8). All `CircleDocument.from_bytes(...).verify()` calls passed with no issues. Full graph inspection saved beside the models.

Existing test commands and detailed outcomes are in baseline_tests.md; all were run from /workspace/TICO with /tmp/tico-venv activated and HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1:

```bash
./ccex test -k quantize_per_tensor_retarget
./ccex test -i -k quantization.wrapq.wrappers.llama
./ccex test -k test_llama_circle_runtime
```
All passed: 2 + 55 + 13 = 70 tests; zero skips.

Analysis commands use the temporary scripts documented in the report. Git hygiene commands:

```bash
git status --porcelain=v1
git rev-parse HEAD origin/main
git log -1 --format=fuller
```
Working tree clean. HEAD and origin/main identical. No commit/push/PR or production/test edit occurred.

Final analysis commands (all successful):

```bash
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/analyze_fx.py
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/num_probe.py
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/analyze_numerics.py
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/write_numerical_report.py
/tmp/tico-venv/bin/python /tmp/tico_llama_kv_concat_investigation/probe_singleton_concat.py
git diff --exit-code
git status --porcelain=v1
```

Final independent Python verification loaded all eight decoder Circle bytes with `CircleDocument.from_bytes`, called `verify()`, and wrote byte counts/SHA256 hashes to artifact_manifest.json. All eight passed again. Git diff/status were empty, exit 0. No generated Circle/PT2/PNG investigation files exist in the checkout; the repository PNGs are pre-existing assets.



# Appendix F: existing regression results

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
