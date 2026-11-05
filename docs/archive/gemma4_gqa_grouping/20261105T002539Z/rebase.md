# Independent Gemma4 branch rebase — 2026-11-05

Samsung/TICO main was fetched once and fixed at `42779caf9229ac176b47bfa6e341de84b07e8622` for both branches. The original checkout and historical BASE worktree remain untouched. Both feature worktrees were clean before rebase; no stash/reset was used.

Both alternatives were independently rebased with `git rebase --onto 42779caf9229ac176b47bfa6e341de84b07e8622 54411f0ebf4782988193f1b40beab9d82c0f830d`. There were no conflicts, dropped commits, code fixes or merges. `git range-diff` reports every replayed commit patch-equivalent. A/B implementations remain separate. All replayed authors and committers are `seongwoo <mhs4670go@naver.com>`.

|Variant|Previous published head|Rebased head tested before this note|Rebased product implementation|
|---|---|---|---|
|A|7353bcc2b7b5f6f8ddc532b57a764e5b439269ac|7d9ad1f6af6565a634f640270ad16c69461767d1|e4853c7cc52cebd7941bddd1b6da7eb7bbbe7948|
|B|66a803fc76f228f6e521e944d2e0788bc268678b|4c6eb7d56a1180982c76d7befe089eeb9e643866|f4bd75af474a0dc5f4657d6f13976cbf51a210c0|

The subsequent metadata commit changes only this note/JSON and historical-notice links. Its final SHA is in the delivery message; product code remains the tested code above.

## Publication and recovery

The earlier explicit prohibition on force-push remains in effect. Rebase cannot update the existing remote refs as a fast-forward, so the original published refs are preserved and the targets for ordinary push are:

- A: `perf/gemma4-unrolled-pv-grouped-rebased-20261105`
- B: `perf/gemma4-unrolled-qk-pv-grouped-rebased-20261105`

The existing local feature branch names contain the rebased histories. Original tips also have local recovery refs:

- `backup/gemma4-unrolled-pv-grouped-pre-rebase-20261105T002539Z`
- `backup/gemma4-unrolled-qk-pv-grouped-pre-rebase-20261105T002539Z`

No main push, PR, merge or force-push is part of this update.

## Validation performed

|Variant|Focused tests PASS/FAIL/SKIP|Fresh24/8 QUANT Circle prefill/decode|
|---|---|---|
|A|46/0/0 (39 core/integration/RoPE +7 KV Circle)|2/2 PASS|
|B|47/0/0 (same five modules, with7 B grouping cases)|2/2 PASS|

All test processes exited0. Tests execute actual wrappers, fresh prepare/calibrate/freeze, shared/sliding cached model decode, new upstream RoPE norm-domain inheritance, UINT8/INT16 cache storage/compute, and serialized Circle reference parity. Import diagnostics confirm each process used its own worktree's TICO. The source audit confirms grouping is the only wrapper-method delta from upstream; upstream RoPE/lifecycle changes are intact. Exact commands, durations, imports and range-diff records are in [validation.json](validation.json).

Fresh export used the existing small Hq24/Hkv8/D4 fixture, B1, prefill T=S3 and decode T1/S8 with7 past entries. A/B weights, calibration order and evaluation hashes match. Eager versus torch.export is exact for all4 cases; actual Circle generation/schema verification passes; FAKE_QUANTIZE semantic reference returns hidden/key/value exactly equal to ABI-aligned eager. All attention MatMul operands/outputs are rank3; QK/PV counts remain A24/8 and B8/8. Float activation count and missing-data-qparam count are0. See [circle_smoke.json](circle_smoke.json) for shapes, exact domains and hashes.

Both variants now have2 Quantize/0 Dequantize in prefill and4/0 in decode in this fixture. Remaining Quantize operations are genuine K/V storage-to-compute grid conversions; identity-domain conversions are0. The old October counts must not be carried forward: upstream RoPE norm-domain inheritance removes the former internal RoPE requantizations. No conversion was removed by this rebase patch, and no extra grouped-concat conversion appeared.

The fresh smoke was run as follows in each rebased worktree, using a separate process and output path:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  /path/to/venv/bin/python  docs/archive/gemma4_gqa_grouping/20261029T070221Z/export_check.py  --quant quant --output /path/to/new-output
```

NPU compilation/device execution, performance benchmarks, independent-observer accuracy comparisons, full repository CI and FP Circle runtime were not rerun. This update does not claim new performance/quality results.

## Historical report, checkpoints and Circle snapshots

The October comparison remains tied to BASE `54411f0ebf4782988193f1b40beab9d82c0f830d` and its measured A/B SHAs. Original metrics, qparams and committed Circle payloads are preserved byte-for-byte. Only a notice link was added to the historical comparison/artifact README. The original report table was not relabeled as measurements of the new main.

Upstream `f8e9b03` changes default RoPE norm-domain inheritance and frozen-checkpoint policy. Do not load an October frozen checkpoint and interpret it as a calibrated rebased model. Reprepare and recalibrate fresh modules under the new policy; all validation in this update did so.

Committed Circle snapshots under `20261029T070221Z/circle_artifacts/` remain historical October exports, including their recorded FP runtime limitations. They were not replaced or certified for the rebased code. Fresh QUANT smoke binaries were generated outside Git for validation, with their hashes recorded here; the command above regenerates them. No large logs/checkpoints or new binary files are added by this metadata commit.
