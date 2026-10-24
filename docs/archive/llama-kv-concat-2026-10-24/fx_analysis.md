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
