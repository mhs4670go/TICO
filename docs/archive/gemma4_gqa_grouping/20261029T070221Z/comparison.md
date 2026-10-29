# Gemma4 unrolled GQA grouping comparison — 20261029T070221Z

## Scope and provenance

Samsung/TICO main was fetched once at task start from `https://github.com/Samsung/TICO.git`; fixed BASE is `54411f0ebf4782988193f1b40beab9d82c0f830d`. No later main movement is incorporated. The original `work` checkout was clean and remains untouched. Three worktrees use this one base. Neither variant is based on or merged with the other.

|Implementation|Independent branch|Measured implementation SHA|
|---|---|---|
|BASE|detached main snapshot|54411f0ebf4782988193f1b40beab9d82c0f830d|
|A, PV-only|perf/gemma4-unrolled-pv-grouped|a2bec27766e08a3c79754b61f60e657a7a1b24dd|
|B, QK+PV|perf/gemma4-unrolled-qk-pv-grouped|d191ac62cad9e85cb8d284363d5b32a1f45ddaa7|

The subsequent report commit is intentionally separate; its SHA is reported in the delivery message. Both branches contain this same comparison. All commits use author **and** committer `seongwoo <mhs4670go@naver.com>` plus the repository DCO line. Local Git identity overrides pre-existing system/global identities; no `GIT_AUTHOR_*`/`GIT_COMMITTER_*` overrides were present at inspection. Remote branch names were checked with `ls-remote` and did not already exist. Only explicit named feature refs are pushed to the existing fork remote, `https://github.com/mhs4670go/TICO.git`.

CPU-only synthetic data, no pretrained downloads, no GPU. Python 3.12.14, Torch 2.13.0+cpu, Transformers 5.18.0. `environment.json` and the metric manifests retain host details, import paths, weights/data digests and commit IDs. Every worker is a separate process with cwd/PYTHONPATH pointing to its worktree; assertions and `integration-imports.json` check actual TICO/wrapper source locations. The shared venv is used without changing its editable installation. Full-wrapper, numerical and export fixtures intentionally have different dimensions, but each comparison uses identical weights/data across implementations.

## Code and tensor contract

The requested source paths exist at BASE. BASE already has identity-scaling elimination while retaining both logits fake-quant boundaries, plus persistent KV storage/compute domain handling. No assumed older validation branch was used. Baseline investigation included root/quantization/test AGENTS, development/torch policy/design, WrapQ/lifecycle/export docs, inspector config/cases, tests and CI. The production changes are confined to `_forward_unrolled` and accurate layout documentation. The batched reference path, external settings, projections/norm/RoPE, mask construction, storage/compute conversion, producer/consumer binding, cache ABI and `o_proj` are unchanged.

For `G=Hq/Hkv`, head index is `kv_idx*G+rep_idx`. BASE computes `Q_i[B,T,D] @ K_i^T[B,D,S]`, then `P_i[B,T,S] @ V_i[B,S,D]`. A retains each QK/scale/mask/float32 Softmax/dtype restoration/dropout/softmax observer, concatenates P along M into `[B,G*T,S]`, and runs one PV `[B,G*T,D]`. B concatenates Q into `[B,G*T,D]`, runs QK `[B,G*T,S]`, scale/mask/Softmax per group, and one PV per group. All attention MatMuls stay rank3. Neither variant repeats KV or uses head-axis broadcasting/rank5 GQA.

A slices grouped output rows `[rep*T:(rep+1)*T]`. B views grouped P/output as `[B,G,T,S/D]` and selects head slices. Both stack outputs at axis2 to `[B,T,Hq,D]`, then reshape to `[B,T,Hq*D]` before unchanged `o_proj`; weights stack at axis1 to `[B,Hq,T,S]`. This preserves head-major grouping but token-major final output, including T>1 and non-contiguous inputs. G=1 avoids new concat/reorder. B's mask `[Bm,1,S]` broadcasts directly; `[Bm,T,S]` repeats the complete T-row block G times. It never builds a new GT causal triangle; original padding/sliding/append offsets are retained.

## Observer boundaries and quantization ABI

|Site|BASE / A CALIB calls|B CALIB calls|Shape before grouping / grouped|
|---|---:|---:|---|
|logits_raw, logits, mask_add, softmax|Hq|Hkv|[B,T,S] / [B,G*T,S]|
|attn_out|Hq|Hq|[B,T,D], head order retained|
|attn_weights|1|1|[B,Hq,T,S], return-only|
|attn_out_h|1|1|[B,T,Hq,D]|

`_apply_attention_scale` is reused unchanged: scaling=1 still observes logits_raw and logits; non-unit scaling retains the scale tensor/FQ/multiply. Softmax stays float32 on S with cast back. Training/dropout support remains; B changes dropout call shapes, so identical seeds need not give identical training samples. Fully masked-row policy is unchanged. `obs_attn_weights` is never newly inserted in PV input (the batched path does have that extra input boundary, so it is not the primary QUANT reference).

An initial implementation observed PV only after splitting. Real Circle inspection exposed floating grouped MatMul outputs with missing qparams: markers do not propagate backwards through slices. The final narrow fix uses the **existing** `obs_attn_out` on the unsplit PV result only during frozen scalar affine QUANT, replacing head-local fake-quant rather than adding another round trip. Pointwise scalar FQ commutes with slicing; an actual-lifecycle regression checks exact equality. CALIB continues head-by-head for A and B, preserving PV EMA sample order. G=1 retains its original boundary. No new observer state/name, float island, common pass edit or suppressed required conversion was introduced.

Default PTQ is per-tensor UINT8 MinMax; explicit MinMax repeats it as a control. MinMax combines extrema and is insensitive to grouped invocation except FP roundoff. B's EMA logits/Softmax statistics change because 24 ordered updates become 8 group updates; independent-calibration differences must not be interpreted as a grouping bug. A keeps those updates too. Histogram/percentile were not run: they are not defaults, and the required MinMax/EMA comparison already distinguishes order-dependent calibration. Per-channel/MX are outside the validated optimized Circle scope: B's row axis1 changes T→GT, so BASE per-channel row grids cannot be transplanted unchanged; MX axis reductions can change semantics. Both variants retain head-local eager PV FQ for non-scalar/non-affine observers, but this does not establish their grouped PV Circle qparam ABI. No claim of general per-channel/MX export support is made.

## Numerical experiments

`numerics_benchmark.py` runs actual wrappers with seeded 24/8, D8, hidden32 fixtures. Six cases are decode/append-prefill/B2, each core (including original o_proj for numerical comparisons) and full wrapper. Calibration and evaluation are distinct fixed samples. NO_QUANT uses common initial weights. Fixed-grid evaluation restores BASE whole-module checkpoints through the current lifecycle, calls `validate_frozen_model`, checks exact frozen observer snapshots/values/policies and disabled collection, and asserts no mutation after evaluation. This avoids assuming bare state_dict restores nonregistered references. Independent experiments separately prepare/calibrate/convert each initial model with identical samples/order/seed. Exact equality of all observer grids was verified; the compact metrics preserve all-grid snapshot digests, the seven requested boundary qparam values, frozen flags, calls/shapes and data hashes. Full grids can be regenerated into the external artifact directory.

PEIR follows `tico/quantization/evaluation/metric.py`: `max(abs(reference-target))/(max(reference)-min(reference))`, denominator1 for zero interval. The function returns a fraction; percent is fraction×100. Tables below state percent explicitly. Relative error is L2(error)/L2(reference), avoiding unstable elementwise division near zero. Metrics retain mean/max absolute errors per tensor/case. Worst-case columns can come from different cases and are not a single output distribution. Existing FP tolerance remains `atol=rtol=1e-5`; no tolerance was raised. Numerical equivalence was assessed in eval with dropout0.

B product was first committed as `386a4ecc776cfe39ab8b467c646db9254bd199d6`; `d191ac62` fixes only a test typing issue and retains the actual output projection in that test. Numerics/benchmarks ran at clean `d191ac62`; export source identity confirms byte-identical product code between those two commits.

|Implementation / comparison|Worst mean abs output|Worst max abs output|Worst relative L2|Worst PEIR %|
|---|---:|---:|---:|---:|
|base / FP vs BASE|0|0|0|0|
|base / frozen default vs BASE|0|0|0|0|
|base / independent default vs BASE|0|0|0|0|
|base / independent EMA vs BASE|0|0|0|0|
|a / FP vs BASE|0|0|0|0|
|a / frozen default vs BASE|0|0|0|0|
|a / independent default vs BASE|0|0|0|0|
|a / independent EMA vs BASE|0|0|0|0|
|b / FP vs BASE|7.24103302e-08|3.57627869e-07|2.29796407e-07|1.53007044e-05|
|b / frozen default vs BASE|0|0|0|0|
|b / independent default vs BASE|2.51748133e-08|1.1920929e-07|8.38969885e-08|6.08174547e-06|
|b / independent EMA vs BASE|0.128855194|0.373196483|0.524122207|35.353531|

All table values below are worst-case across six cases (decode/append/B2 × core/full). PEIR is percent. Default is UINT8 MinMax; explicit MinMax is repeated as a control. FP32; eval/dropout=0; one CPU thread. Actual per-case mean/max/relative L2 and PEIR are retained in JSON/CSV.

|Implementation|Experiment/policy|Output max abs vs BASE|Output PEIR % vs BASE|Weights max abs vs BASE|
|---|---|---:|---:|---:|
|base|NO_QUANT|0|0|0|
|base|fixed_vs_base/default|0|0|0|
|base|fixed_vs_base/minmax|0|0|0|
|base|fixed_vs_base/ema|0|0|0|
|base|independent_vs_base/default|0|0|0|
|base|independent_vs_base/minmax|0|0|0|
|base|independent_vs_base/ema|0|0|0|
|a|NO_QUANT|0|0|0|
|a|fixed_vs_base/default|0|0|0|
|a|fixed_vs_base/minmax|0|0|0|
|a|fixed_vs_base/ema|0|0|0|
|a|independent_vs_base/default|0|0|0|
|a|independent_vs_base/minmax|0|0|0|
|a|independent_vs_base/ema|0|0|0|
|b|NO_QUANT|3.5762787e-07|1.5300704e-05|4.1723251e-07|
|b|fixed_vs_base/default|0|0|0|
|b|fixed_vs_base/minmax|0|0|0|
|b|fixed_vs_base/ema|0|0|0|
|b|independent_vs_base/default|1.1920929e-07|6.0817455e-06|0|
|b|independent_vs_base/minmax|1.1920929e-07|6.0817455e-06|0|
|b|independent_vs_base/ema|0.37319648|35.353531|0.40353376|

|Implementation|Policy|Fixed output max abs vs FP|Fixed PEIR % vs FP|Independent output max abs vs FP|Independent PEIR % vs FP|
|---|---|---:|---:|---:|---:|
|base|default|0.19506332|8.0308215|0.19506332|8.0308215|
|base|minmax|0.19506332|8.0308215|0.19506332|8.0308215|
|base|ema|0.639801|31.849582|0.639801|31.849582|
|a|default|0.19506332|8.0308215|0.19506332|8.0308215|
|a|minmax|0.19506332|8.0308215|0.19506332|8.0308215|
|a|ema|0.639801|31.849582|0.639801|31.849582|
|b|default|0.19506332|8.0308215|0.19506326|8.0308204|
|b|minmax|0.19506332|8.0308215|0.19506326|8.0308204|
|b|ema|0.639801|31.849582|0.38277853|18.452894|

B independent EMA first diverges at logits_raw, before Softmax/PV; changed upstream grids include logits_raw/logits/mask_add/softmax. The worst output difference vs BASE is0.37319648, weights0.40353376. For this synthetic data B EMA error versus the common FP reference is smaller than BASE EMA (max0.38277853 vs0.639801); this is not a real-model quality claim. Default/explicit MinMax differs only in the downstream o_proj.act_out grid at FP rounding scale. In fixed-qparam evaluation all three policies produced exact BASE outputs, weights and canonicalized intermediate boundary tensors. No rounding crossing was observed for these seeds; it is not a proof for every input.

The separate D64 benchmark fixture extends the outside-timing numerical check to S128/2048 and prefill32/128. A remains exactly BASE. B full-wrapper FP max is8.64267e-7, projection-excluded core max3.90038e-6, attention-weight max1.07288e-6 (decodeS2048). Fixed QUANT output/weights/cache remain exactly BASE. Snapshot and weight hashes match every worker and are unchanged by evaluation. Numerical comparisons here do not include calibration/setup time.

## Test and export validation

|Command group|BASE PASS/FAIL/SKIP|A PASS/FAIL/SKIP|B PASS/FAIL/SKIP|Exit|
|---|---|---|---|---|
|Full Gemma4 unittest discovery|354/0/68 (422 run)|361/0/68 (429 run)|363/0/68 (431 run)|0 each|
|Related config/lifecycle/frozen export/domain tests|61/0/0|61/0/0|61/0/0|0 each|
|Explicit internal tiny attention prepare/calibrate/convert|1/0/0|1/0/0|1/0/0|0 each|
|Final variant core regression module|N/A|6/0/0|7/0/0|0 each|
|New shared cached/model integration module|N/A|2/0/0|2/0/0|0 each|
|Frozen observer unit module (extra BASE check)|31/0/0|not repeated|not repeated|0|

Counts overlap (new integration is included in full discovery), so do not sum them. All68 skips are existing internal smoke opt-ins, not Gemma import failures. The tiny relevant internal attention test was explicitly enabled and passed. Final focused regressions include tests added/fixed after full discovery imported its test modules; full discovery ran stable final product code, and the final focused runs validate the final test versions. Exact commands/timings are in `integration-validation.json`. Product/test UFMT+MYPY selected-file checks pass both branches, final A new-core all-linter check passes; initial B all-linter check had no Pylint/UFMT errors and only a test MYPY issue fixed in d191ac62. All eight final reproduction Python scripts passed repository selected-file UFMT/MYPY, Pylint jobs1 (10/10), and syntax checks after annotation-only fixes; details and the separate imported-code direct-mypy diagnostics are recorded in log-summary.md. No claim is made that full repository formatting/static analysis or the entire non-Gemma test suite ran.

Pre-edit focused baseline tests passed (A's broad attention pattern33PASS/4SKIP; B's focused attention/export19PASS). No new failures remain in final executed unit/integration suites. Earlier experimental missing-qparam/equal-domain failures motivated the PV marker fix; those superseded runs are not reported as final PASS. Initial full A/B pre-fix runs and a duplicate A cache-only run were interrupted, excluded. A colliding BASE related log was discarded and uniquely rerun clean. `log-summary.md` records these exclusions.

Coverage includes24/8, original small fixtures, G1, KVhead1, B1/B2, decode/prefill/append-prefill, supplied additive masks with batch1/B and query1/T, explicit head/token patterns, non-contiguous QKV, full/sliding/padding, unit/nonunit scaling, owning/k_eq_v/shared consumers, UINT8 storage and INT16 attention computation, weights/o_proj/cache order and actual cached model decode. Training/dropout execution and original shape rejection contracts remain tested. Reference runtime tests include serialized prefill plus two chained decode steps and producer-bound shared-KV domains.

|Validation level|Status|
|---|---|
|Default inspector tiny2/2,D32 prefill+decode Circle CLI|6/6 success across BASE/A/B; all6 EP + saved-Circle FAKE_QUANTIZE runtime comparisons exact|
|Additional24/8 and4/2,D4, FP/QUANT, prefill/decode EP|24/24 eager-versus-exported execution exact|
|Additional fixtures actual Circle generation + schema verify|24/24 success|
|Additional QUANT Circle FAKE_QUANTIZE reference runtime|12/12 success, hidden/key/value exact vs ABI-aligned eager|
|Additional FP Circle NATIVE reference runtime|6 success /6 FAIL: BASE prefill/decode and B decode for both fixtures|
|NPU compile / actual NPU execution and latency|NOT_RUN: compiler/toolchain not installed|

Circle generation alone is not runtime validation. The FP failures are inherited serialized Softmax rank4 `[1,1,M,S]` feeding PV with rank3 output metadata `[1,M,D]`; the reference runtime correctly rejects it. BASE also fails with O0. A FP prefill/decode and B FP prefill execute (24/8 max output error5.96e-7 prefill, A decode1.94e-7). This issue was not hidden by common pass/runtime changes. QUANT semantic reference is explicitly FAKE_QUANTIZE, not bit-exact integer NPU execution. See `export_notes.md` and `export_summary.json` for exact failures, shape/dtype/qparam values and artifact hashes.

### Measured graph structure

The four projection FULLY_CONNECTED operations are excluded from QK/PV counts. EP and final Circle both show BASE24/24, A24/8, B8/8. Every attention MatMul in EP and QUANT Circle has rank3 operands/outputs; the failed FP Circle cases have the serialized Softmax/PV rank defect described above. Grouped M isG*T, K/V are not repeated, and no head-axis broadcasting/rank5 path is introduced. For the extra fixture T=S=3 prefill and T1/S8 decode, D4: BASE QK `[1,T,4]@[1,4,S]`, PV `[1,T,S]@[1,S,4]`; A PV uses `[1,3*T,S]`; B QK uses `[1,3*T,4]`, PV `[1,3*T,S]`. Full per-op shapes/qparams are regenerated by the export script.

|24/8 QUANT Circle op|BASE prefill/decode|A prefill/decode|B prefill/decode|
|---|---:|---:|---:|
|QK MatMul|24/24|24/24|8/8|
|PV MatMul|24/24|8/8|8/8|
|Concat|3/5|11/13|12/13|
|Reshape|48/51|48/51|80/83|
|Transpose|27/24|27/24|11/8|
|StridedSlice|40/40|64/64|64/64|
|Softmax|24/24|24/24|8/8|
|Quantize|10/12|10/12|10/12|
|Dequantize|0/0|0/0|0/0|
|FLOAT activations / missing data qparams|0/0|0/0|0/0|

Auxiliary counts cover the full wrapper, including unchanged cache/RoPE/projection plumbing. Control/index/shape tensors correctly remain unquantized. All new concat/reshape/slice data operands and outputs are UINT8 affine per-tensor in the default export fixture, matching their connected producer grids. Representative Q/K/V/raw-logits/P/PV scales and zero points are preserved in `export_notes.md`; no extra Quantize/Requantize or Q/DQ round trip was needed after the local fix. NPU kernel counts cannot be inferred from Python or graph operator counts.

## CPU measurement and cost

One intra/inter-op thread, fixed seed4670, FP32 eval/dropout0, B1,Hq24/Hkv8,D64, hidden64. Core calls the production `_forward_unrolled` with o_proj replaced by Identity solely for timing; full-wrapper includes projections/norm/RoPE/mask/cache append/o_proj. QUANT uses the identical BASE frozen whole-module checkpoint with exact snapshot and weight digest verification. Warmup10 per scope/mode/case/round,60 timed calls per round; two serial reverse-order rounds BASE,A,B,B,A,BASE give120 pooled samples. Calibration/setup/export/file I/O/reference checks are outside timing. No other tests/lint/export jobs ran during benchmark. Median/p10/p90 and per-round medians are retained; no CI performance threshold is added. Shared cloud CPU scheduling noise remains possible.

CPU microseconds; pooled median [p10,p90] from120 calls (2 rounds ×60), warmup10 per round. One thread, B1 Hq24/Hkv8 D64, full-wrapper hidden_size64. Core excludes o_proj. No concurrent benchmark workloads.

|Case/mode|Scope|BASE µs|A µs|B µs|A BASE/A ratio|B BASE/B ratio|
|---|---|---:|---:|---:|---:|---:|
|T1_S128/NO_QUANT|core_projection_excluded|634.56 [588.73,748.55]|684.79 [634.92,766.20]|503.61 [460.31,532.40]|0.927|1.260|
|T1_S128/NO_QUANT|full_wrapper|1192.05 [1149.03,1326.60]|1235.85 [1190.48,1324.89]|1071.52 [1018.20,1446.60]|0.965|1.112|
|T1_S128/QUANT|core_projection_excluded|2043.62 [1981.14,2254.34]|1942.45 [1880.59,2081.03]|1128.21 [1092.20,1274.22]|1.052|1.811|
|T1_S128/QUANT|full_wrapper|7546.52 [7392.60,8081.44]|7370.37 [7282.45,7668.61]|6562.75 [6450.67,6949.58]|1.024|1.150|
|T1_S2048/NO_QUANT|core_projection_excluded|1488.55 [1427.93,1690.02]|1515.08 [1448.18,1645.43]|1219.55 [1175.13,1274.30]|0.982|1.221|
|T1_S2048/NO_QUANT|full_wrapper|2961.97 [2871.12,3237.81]|3073.09 [2907.02,3326.07]|2746.98 [2653.52,3004.11]|0.964|1.078|
|T1_S2048/QUANT|core_projection_excluded|4339.09 [4250.77,5360.36]|4190.37 [4092.46,4464.62]|3199.83 [3139.01,3345.00]|1.035|1.356|
|T1_S2048/QUANT|full_wrapper|44574.82 [43822.39,46137.31]|44958.15 [43935.76,49199.54]|43649.66 [43073.52,45679.20]|0.991|1.021|
|T32_S128/NO_QUANT|core_projection_excluded|1243.03 [1200.09,1348.57]|1287.19 [1239.23,1375.25]|1015.95 [976.54,1129.56]|0.966|1.224|
|T32_S128/NO_QUANT|full_wrapper|2140.77 [2091.43,2238.08]|2178.98 [2116.63,2275.79]|1949.37 [1891.39,2047.58]|0.982|1.098|
|T32_S128/QUANT|core_projection_excluded|6024.69 [5944.77,6417.36]|5867.12 [5799.14,6305.39]|4900.98 [4845.80,5196.72]|1.027|1.229|
|T32_S128/QUANT|full_wrapper|15521.32 [15161.49,16381.67]|15466.04 [15054.92,16547.74]|14339.77 [14083.99,14909.06]|1.004|1.082|
|T128_S128/NO_QUANT|core_projection_excluded|2970.46 [2915.47,3094.28]|3065.73 [2994.57,3298.53]|2758.97 [2686.34,2969.90]|0.969|1.077|
|T128_S128/NO_QUANT|full_wrapper|4485.00 [4389.10,4606.92]|4562.99 [4440.96,4795.52]|4269.78 [4119.86,4577.68]|0.983|1.050|
|T128_S128/QUANT|core_projection_excluded|17666.74 [17525.02,18246.32]|17691.09 [17449.29,18155.24]|16641.45 [16401.24,17369.27]|0.999|1.062|
|T128_S128/QUANT|full_wrapper|37725.77 [37314.54,38832.29]|37853.22 [37082.90,39684.31]|36397.69 [36093.97,37246.82]|0.997|1.036|

A is consistently slower in NO_QUANT here (core roughly2–8%, full roughly2–4%), despite fewer PV ops. QUANT A core improves about3–5% for shorter shapes but full-wrapper gains are small or negative and T128 is neutral. B improves NO_QUANT core1.077–1.260x/full1.050–1.112x and QUANT core1.062–1.811x/full1.021–1.150x in these measured cases. Small differences with overlapping quantiles should be treated as noise, not guaranteed gains. CPU fake-quant and Python dispatch cost are substantial; none of these ratios establishes NPU acceleration.

Arithmetic is unchanged: QK MAC = PV MAC = `B*Hq*T*S*D`; total MAC is `2*B*Hq*T*S*D`, approximately `4*B*Hq*T*S*D` FLOPs. Grouping changes M and potential KV reuse. A adds P concat writes `B*Hq*T*S` elements across groups; B adds Q concat writes `B*Hq*T*D` elements and one mask concat of `Bm*G*T*S` elements only when query mask hasT rows. Head splitting can retain the grouped output storage via views; final stack/layout materializes as before. Reshape is not assumed zero-copy for arbitrary strides.

|FP32 static size, G3/B1/D64|Qcat / grouped PV output per group|A Pcat per group|B maskcat (Bm1)|
|---|---:|---:|---:|
|T1,S128|768 B|1,536 B|0|
|T1,S2048|768 B|24,576 B|0|
|T128,S128|98,304 B|196,608 B|196,608 B|

These are static tensor-size estimates, not measured peak allocator memory or DRAM traffic. Fake-quant intermediates, old/new loop variable references, copying/reordering and lowering can increase transient memory. Reduced graph op count does not imply reduced FLOPs or memory traffic.

## Recommendation and remaining validation

A is the conservative alternative when preserving existing head-ordered calibration (especially EMA) is the priority; it meets PV graph reduction but offers little CPU benefit and sometimes costs more due to concat/slicing. B is the stronger CPU candidate and reduces both attention MatMul and Softmax dispatch, with more reshapes and changed upstream calibration. For NPU evaluation, first compile both QUANT exports on the actual target, compare grouping-M scheduling, kernel reports, SRAM/live buffers and measured latency; do not choose from CPU ratios alone. For B compare fresh calibration/model quality against BASE and frozen-qparam controls, especially EMA. A avoids that upstream calibration change.

Unverified: NPU compiler/device execution and performance; real pretrained-model quality; general per-channel/MX optimized Circle ABI; histogram/percentile observers; complete Cartesian combinations; every possible input near quantization rounding thresholds; full repository CI suite/static check. FP serialized Softmax rank failures need a separate narrow conversion issue. None was suppressed or presented as a pass.

## Reproduction

Use an existing compatible CPU Python environment (`PYTHON_BIN` absolute path). Create three detached worktrees with exactly the implementation commits above; leave report/final commits out of numerical timing metadata. Export scripts accept arbitrary worktree paths; the benchmark coordinator expects names `gemma4-base`, `gemma4-a`, `gemma4-b` under one root.

```bash
BASE_SHA=54411f0ebf4782988193f1b40beab9d82c0f830d
A_SHA=a2bec27766e08a3c79754b61f60e657a7a1b24dd
B_SHA=d191ac62cad9e85cb8d284363d5b32a1f45ddaa7
RUN_ROOT=$(mktemp -d)
OUT=$(mktemp -d)
TOOLS=$(pwd)/docs/archive/gemma4_gqa_grouping/20261029T070221Z
PYTHON_BIN=/absolute/path/to/venv/bin/python
git worktree add --detach "$RUN_ROOT/gemma4-base" "$BASE_SHA"
git worktree add --detach "$RUN_ROOT/gemma4-a" "$A_SHA"
git worktree add --detach "$RUN_ROOT/gemma4-b" "$B_SHA"
for impl in base a b; do
  (
    cd "$RUN_ROOT/gemma4-$impl"
    export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$PYTHON_BIN" "$TOOLS/numerics_benchmark.py" --mode numerics \
      --impl "$impl" --out "$OUT/numerics/$impl" --base-out "$OUT/numerics/base"
  )
done
"$PYTHON_BIN" "$TOOLS/summarize_numerics.py" "$OUT/numerics" --output "$OUT/numerics_summary"
PYTHON_BIN="$PYTHON_BIN" bash "$TOOLS/run_exports.sh" \
  "$RUN_ROOT/gemma4-base" "$RUN_ROOT/gemma4-a" "$RUN_ROOT/gemma4-b" "$OUT/export"
# Stop other CPU workloads before this command.
"$PYTHON_BIN" "$TOOLS/run_benchmarks.py" --python "$PYTHON_BIN" \
  --worktrees-root "$RUN_ROOT" --output "$OUT/benchmark" --repeats 60
"$PYTHON_BIN" "$TOOLS/summarize_benchmarks.py" "$OUT/benchmark" --output "$OUT/benchmark_summary"
```

`run_exports.sh` components were executed individually in this run and the combined script was syntax checked. The export collector records per-stage errors and may exit0 after collecting failures; inspect the runtime status fields, not just process status. Raw detailed graph/tensor records, Circle binaries/checkpoints and timing samples are intentionally external to Git; the scripts regenerate them.

Tests (run separately from benchmarks in each worktree, with cwd/PYTHONPATH set as above):

```bash
python -m unittest discover -s test/quantization/wrapq/wrappers/gemma4 -v
python -m unittest test.quantization.config.test_gemma4_attention \
  test.quantization.wrapq.test_lifecycle test.quantization.wrapq.test_frozen_state_export \
  test.quantization.passes.test_split_concat_domain_export \
  test.quantization.passes.test_mixed_dtype_concat_export -v
RUN_INTERNAL_TESTS=1 python -m unittest \
  test.quantization.wrapq.wrappers.gemma4.test_quantize_text_attention -v
# A / B respectively:
python -m unittest test.quantization.wrapq.wrappers.gemma4.test_unrolled_pv_grouping -v
python -m unittest test.quantization.wrapq.wrappers.gemma4.test_unrolled_qk_pv_grouping -v
```

Inspector starting command actually used (also decode with the corresponding case):

```bash
python -m tico.quantization.examples.inspector \
 --config tico/quantization/examples/configs/wrapper_smoke.yaml --mode wrapper-smoke \
 --case gemma4_text_attention_prefill --export circle --no-plot --strict \
 --output-dir "$OUT/inspector-prefill"
```

For local target compilation, first confirm the installed compiler options. The repository's `tico/quantization/evaluation/executor/triv24_executor.py` uses `-o OUTPUT INPUT.circle`. Example after regenerating separate artifacts (NOT executed in cloud):

```bash
NPU_COMPILER=/usr/share/one/backends/triv24/bin/triv24-compile
"$NPU_COMPILER" --help
for impl in base a b; do
  for mode in prefill decode; do
    "$NPU_COMPILER" -o "$OUT/$impl-$mode.tvn" \
      "$OUT/export/$impl/custom/h24_${mode}_quant.circle" \
      > "$OUT/$impl-$mode.compile.log" 2>&1
  done
done
```

Keep compiler reports for each variant and compare shapes/kernel allocation/latency using the installed toolchain documentation. No unverified extra compiler flags are assumed.
