**Gemma4 RoPE/KV domain 2×2 measurements**

Base `54411f0ebf4782988193f1b40beab9d82c0f830d`; measured HEAD `279ea8cedc1e8c52b69d5341f6614f760b1171e8`; Python source SHA256 `490abfbad6938e323d7f732ec6f1b61d191f8a36ace403931a2df00f6b8edabe`.

Profiles: affine, u8s16c; seeds: [123, 456, 789]; 8-token prefill + 128 fixed-hidden decode steps. Values below are means ± sample standard deviation across seeds. These are random-initialized attention fixtures, not pretrained language-model accuracy.

Fresh calibration runs each variant independently from common FP weights and disjoint fixed calibration/evaluation PRNG streams. All use three calibration prefills and four real-cache decode calibration reads per prefill. Compute/storage format equality does not imply equal numerical qparams. Full qparams, statistics, clipping, endpoint occupancy, and per-step boundary metrics remain in the hashed artifacts.

| Profile | Variant | Prefill MAE | Own-cache decode MAE | FP-cache replay MAE | Decode RMSE | Mean max error | Mean normalized L2 | Mean cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| affine | R0K0 | 0.0173238 ± 0.00511 | 0.035832 ± 0.0021 | 0.030468 ± 0.00328 | 0.0446101 ± 0.00256 | 0.117459 ± 0.005 | 0.109221 ± 0.00608 | 0.992002 ± 0.00162 |
| affine | R1K0 | 0.0173238 ± 0.00511 | 0.0357082 ± 0.00216 | 0.0305015 ± 0.00321 | 0.0444591 ± 0.00264 | 0.11703 ± 0.00535 | 0.108918 ± 0.00634 | 0.992008 ± 0.00162 |
| affine | R0K1 | 0.0173238 ± 0.00511 | 0.0358036 ± 0.0021 | 0.0304279 ± 0.00326 | 0.0445787 ± 0.00255 | 0.117313 ± 0.00495 | 0.109134 ± 0.00612 | 0.992015 ± 0.00164 |
| affine | R1K1 | 0.0173238 ± 0.00511 | 0.0356753 ± 0.00215 | 0.0304614 ± 0.00319 | 0.0444237 ± 0.00264 | 0.116884 ± 0.00532 | 0.108823 ± 0.00638 | 0.992022 ± 0.00164 |
| u8s16c | R0K0 | 0.00581592 ± 0.00047 | 0.0123212 ± 0.00406 | 0.0111101 ± 0.00383 | 0.0153228 ± 0.00516 | 0.0387272 ± 0.0134 | 0.0370354 ± 0.0122 | 0.998386 ± 0.00159 |
| u8s16c | R1K0 | 0.00581592 ± 0.00047 | 0.0122405 ± 0.00414 | 0.0111163 ± 0.00384 | 0.0152201 ± 0.00527 | 0.0384239 ± 0.0137 | 0.0367248 ± 0.0125 | 0.998408 ± 0.00161 |
| u8s16c | R0K1 | 0.00336971 ± 0.000536 | 0.0121283 ± 0.00418 | 0.0108655 ± 0.00391 | 0.0150787 ± 0.00531 | 0.0380998 ± 0.0137 | 0.0364764 ± 0.0126 | 0.998403 ± 0.00161 |
| u8s16c | R1K1 | 0.00336971 ± 0.000536 | 0.0120391 ± 0.00428 | 0.0108639 ± 0.00394 | 0.014967 ± 0.00543 | 0.0377709 ± 0.014 | 0.0361474 ± 0.013 | 0.998425 ± 0.00163 |

Own-cache rollout appends only each variant's emitted S delta. Replay reinjects the common FP past values at every step and declares each variant's S input domain. Both compare against the original FP wrapper/cache trace; the fixed-qparam table below instead uses a local FP reference receiving the same dequantized S past.

| Profile | Variant | Decode step | Own-cache MAE | Replay MAE |
|---|---|---:|---:|---:|
| affine | R0K0 | 1 | 0.0164718 ± 0.00853 | 0.0221066 ± 0.0148 |
| affine | R0K0 | 32 | 0.0592268 ± 0.063 | 0.0220286 ± 0.0167 |
| affine | R0K0 | 128 | 0.0491355 ± 0.0331 | 0.0536534 ± 0.0474 |
| affine | R1K0 | 1 | 0.0164718 ± 0.00853 | 0.0221066 ± 0.0148 |
| affine | R1K0 | 32 | 0.0592268 ± 0.063 | 0.0220286 ± 0.0167 |
| affine | R1K0 | 128 | 0.0489221 ± 0.0328 | 0.0536534 ± 0.0474 |
| affine | R0K1 | 1 | 0.0164718 ± 0.00853 | 0.0221066 ± 0.0148 |
| affine | R0K1 | 32 | 0.0592268 ± 0.063 | 0.0220286 ± 0.0167 |
| affine | R0K1 | 128 | 0.0491355 ± 0.0331 | 0.0536534 ± 0.0474 |
| affine | R1K1 | 1 | 0.0164718 ± 0.00853 | 0.0221066 ± 0.0148 |
| affine | R1K1 | 32 | 0.0592268 ± 0.063 | 0.0220286 ± 0.0167 |
| affine | R1K1 | 128 | 0.0489221 ± 0.0328 | 0.0536534 ± 0.0474 |
| u8s16c | R0K0 | 1 | 0.00809358 ± 0.00798 | 0.00595658 ± 0.00345 |
| u8s16c | R0K0 | 32 | 0.0252722 ± 0.0247 | 0.00440756 ± 0.000511 |
| u8s16c | R0K0 | 128 | 0.0280114 ± 0.0183 | 0.0222474 ± 0.0207 |
| u8s16c | R1K0 | 1 | 0.00779651 ± 0.00747 | 0.00585104 ± 0.00326 |
| u8s16c | R1K0 | 32 | 0.0252722 ± 0.0247 | 0.00440756 ± 0.000511 |
| u8s16c | R1K0 | 128 | 0.0280114 ± 0.0183 | 0.0222474 ± 0.0207 |
| u8s16c | R0K1 | 1 | 0.00760447 ± 0.00802 | 0.00539026 ± 0.00353 |
| u8s16c | R0K1 | 32 | 0.0247069 ± 0.0254 | 0.00396595 ± 0.000331 |
| u8s16c | R0K1 | 128 | 0.0280413 ± 0.0183 | 0.0223376 ± 0.0209 |
| u8s16c | R1K1 | 1 | 0.00727462 ± 0.00745 | 0.00523589 ± 0.00326 |
| u8s16c | R1K1 | 32 | 0.0247069 ± 0.0254 | 0.00396595 ± 0.000331 |
| u8s16c | R1K1 | 128 | 0.0280413 ± 0.0183 | 0.0223376 ± 0.0209 |

| Profile | Variant | Quantize prefill/decode | Regions decode: RoPE / past / current / other | S dtype; K+V delta bytes | FAKE single prefill/decode | Recurrent max error | NATIVE |
|---|---|---|---|---|---|---:|---|
| affine | R0K0 | [(4, 4), (4, 6), (6, 6)] | [(4, 0, 0, 0), (4, 2, 0, 0), (6, 0, 0, 0)] | UINT8; 128 | FAIL/PASS, PASS/PASS | 0 | UNSUPPORTED |
| affine | R1K0 | [(0, 0), (0, 2)] | [(0, 0, 0, 0), (0, 2, 0, 0)] | UINT8; 128 | FAIL/PASS, PASS/PASS | 0 | UNSUPPORTED |
| affine | R0K1 | [(4, 4), (6, 6)] | [(4, 0, 0, 0), (6, 0, 0, 0)] | UINT8; 128 | FAIL/PASS, PASS/PASS | 0 | UNSUPPORTED |
| affine | R1K1 | [(0, 0)] | [(0, 0, 0, 0)] | UINT8; 128 | FAIL/PASS, PASS/PASS | 0 | UNSUPPORTED |
| u8s16c | R0K0 | [(8, 10), (8, 11)] | [(4, 2, 4, 0), (4, 3, 4, 0)] | UINT8; 128 | PASS/PASS | 0.00118606 | UNSUPPORTED |
| u8s16c | R1K0 | [(4, 6), (4, 7)] | [(0, 2, 4, 0), (0, 3, 4, 0)] | UINT8; 128 | PASS/PASS | 0.00118606 | UNSUPPORTED |
| u8s16c | R0K1 | [(6, 8)] | [(4, 2, 0, 2)] | UINT8; 128 | FAIL/FAIL, FAIL/PASS, PASS/PASS | 0.00118606 | UNSUPPORTED |
| u8s16c | R1K1 | [(2, 4)] | [(0, 2, 0, 2)] | UINT8; 128 | FAIL/FAIL, FAIL/PASS, PASS/PASS | 0.00118606 | UNSUPPORTED |

Graph counts come from saved/reloaded Circle bytes. Region counts use transparent paths adjacent to serialized RoPE/KV concats; the current region includes a storage Quantize when it lies on that path. NATIVE quantized arithmetic is backend-defined and unsupported by the reference runtime. FAKE_QUANTIZE executes dequantized arithmetic with serialized grids; it does not establish bit-exact NPU behavior. Node-count reductions are not latency measurements.

| Profile | R effect at K0 | R effect at K1 | K effect at R0 | K effect at R1 | Interaction |
|---|---:|---:|---:|---:|---:|
| affine | -0.000123724 ± 8.9e-05 | -0.000128298 ± 9.48e-05 | -2.83604e-05 ± 4.91e-05 | -3.29346e-05 ± 5.7e-05 | -4.57425e-06 ± 7.92e-06 |
| u8s16c | -8.06965e-05 ± 0.000124 | -8.92079e-05 ± 0.000139 | -0.000192903 ± 0.000361 | -0.000201414 ± 0.000376 | -8.51143e-06 ± 1.59e-05 |

Effects use paired mean decode MAE differences, with negative values indicating lower error. Interaction is E11−E10−E01+E00. K changes both past P removal and current S bypass; these main-table effects do not isolate the two causes.

| Fixed common qparams: profile | Variant | Local FP output MAE | Fully FP output MAE |
|---|---|---:|---:|
| affine | R0K0 | 0.0208629 ± 0.0148 | 0.0221066 ± 0.0148 |
| affine | R1K0 | 0.0208629 ± 0.0148 | 0.0221066 ± 0.0148 |
| affine | R0K1 | 0.0208629 ± 0.0148 | 0.0221066 ± 0.0148 |
| affine | R1K1 | 0.0208629 ± 0.0148 | 0.0221066 ± 0.0148 |
| u8s16c | R0K0 | 0.00355748 ± 0.00132 | 0.00595658 ± 0.00345 |
| u8s16c | R1K0 | 0.00344744 ± 0.00117 | 0.00585104 ± 0.00326 |
| u8s16c | R0K1 | 0.00255818 ± 0.000736 | 0.00539026 ± 0.00353 |
| u8s16c | R1K1 | 0.0023266 ± 0.000339 | 0.00523589 ± 0.00326 |

| Frozen KV decomposition | Past P removal at serial current | Current S removal at serial past | Both removed | Interaction |
|---|---:|---:|---:|---:|
| affine | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| u8s16c | -4.77808e-05 ± 8.28e-05 | -0.00090674 ± 0.000819 | -0.000999298 ± 0.000874 | -4.4777e-05 ± 7.76e-05 |

KV auxiliary decomposition status: MEASURED. These are common-qparam single-step differences, not fresh-calibration model NLL. Common fresh boundary dtype/qscheme/scale/zero-point equality across all variants: True; numeric K/V S equality: True. Dedicated R0 half/cat and K0 P are excluded from the common-boundary intersection.

Pretrained cached NLL/PPL: **ASSET_UNAVAILABLE**. No random-init PPL is reported. Native integer/NPU execution and latency are unverified. Descriptive seed variation provides no significance claim.

Reproduce measurements using the recorded command in `summary.json` source metadata; regenerate this report with `python -m tico.quantization.recipes.debug.rope_kv_experiment.report --input <artifacts> --output <new-report-dir>`. `artifact-manifest.json` records 332 source artifact sizes/hashes. Raw logs, Circle binaries and tensor arrays stay outside Git.
