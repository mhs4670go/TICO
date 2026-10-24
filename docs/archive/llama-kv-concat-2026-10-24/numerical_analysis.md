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
