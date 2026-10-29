# Executed commands and log summary

Commands used `/workspace/TICO/.venv/bin/python` in a separate process per worktree with explicit `PYTHONPATH=/workspace/gemma4-{base,a,b}`, matching cwd, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Benchmarks additionally set Torch intra/inter-op threads1; measured imports and implementation SHAs are in the JSON manifests. Raw logs stay outside Git under the execution directory, not required to reproduce the checks.

|Log / command|Exit|Result|
|---|---:|---|
|base-gemma4.log: `python -m unittest discover -s test/quantization/wrapq/wrappers/gemma4 -v`|0|422 run,354PASS/0FAIL/68SKIP,568.329s|
|a-gemma4.log: same discovery|0|429 run,361PASS/0FAIL/68SKIP,444.943s|
|b-gemma4.log: same discovery|0|431 run,363PASS/0FAIL/68SKIP,442.771s|
|base-related-clean.log: related5 modules listed below|0|61PASS/0FAIL/0SKIP,291.477s|
|a-related.log: same related modules|0|61PASS/0FAIL/0SKIP,383.036s|
|b-related.log: same related modules|0|61PASS/0FAIL/0SKIP,375.102s|
|base-observer-frozen.log: `python -m unittest test.quantization.wrapq.observers.test_frozen_state -v`|0|31PASS/0FAIL/0SKIP,3.248s|
|{base,a,b}-internal-attention.log: `RUN_INTERNAL_TESTS=1 python -m unittest test.quantization.wrapq.wrappers.gemma4.test_quantize_text_attention -v`|0 each|1PASS each|
|a-grouping-final.log: `python -m unittest test.quantization.wrapq.wrappers.gemma4.test_unrolled_pv_grouping -v`|0|6PASS/0FAIL/0SKIP|
|b-grouping-final.log: `python -m unittest test.quantization.wrapq.wrappers.gemma4.test_unrolled_qk_pv_grouping -v`|0|7PASS/0FAIL/0SKIP|
|a/b-integration.log: `python -m unittest test.quantization.wrapq.wrappers.gemma4.test_grouped_attention_integration -v`|0 each|2PASS each, included in discovery|
|a-final-ufmt-mypy.log / b-lint-final.log: `lintrunner --take UFMT,MYPY` with all5 changed implementation/test files explicitly listed|0 each|No lint issues|
|a-lint-final.log: final A new-core test full lintrunner|0|No lint issues|
|b-lint.log: initial B all-linter check|1|Only MYPY test Identity assignment issue; no UFMT/Pylint errors; fixed in d191ac62 and exact final focused/MYPY rerun passes|
|a-lint.log: initial A all-linter check|1|Only two MYPY variable annotations; fixed before implementation commit; final selected checks pass|
|Numerical worker (each implementation): `numerics_benchmark.py --mode numerics`|0 each|NO_QUANT, frozen BASE grids, independent default/MinMax/EMA; identical weights/data and freeze checks passed|
|Benchmark coordinator `run_benchmarks.py ... --repeats 60`|0|6/6 serial workers exit0; no overlapping compute jobs; all weights/frozen qparams identical, unmodified|
|Inspector CLI prefill/decode `--export circle --strict --no-plot`|0 each|6/6 actual CLI exports; follow-up EP + saved Circle runtime6/6 exact|
|export_check.py, custom24/8 and4/2, each implementation|0 each collector|24/24 EP exact;24/24 Circle generated/verified; QUANT runtime12/12 exact; FP runtime6PASS/6FAIL (collector status is not test status)|
|Compiler presence (`command -v triv24-compile`, known file path)|not found|NPU compile/execution NOT_RUN|
|Environment `python -m pip check`|0|No broken requirements|

Related5 modules:

```bash
python -m unittest test.quantization.config.test_gemma4_attention \
 test.quantization.wrapq.test_lifecycle test.quantization.wrapq.test_frozen_state_export \
 test.quantization.passes.test_split_concat_domain_export \
 test.quantization.passes.test_mixed_dtype_concat_export -v
```

Repository lintrunner uses the existing UFMT/Pylint/MYPY configuration. All changed product/test files received UFMT+MYPY; Pylint's initial runs found no product/test diagnostics except the separately fixed MYPY issues. Research-script checks are reported in comparison.md; they are distinct from a repository-wide ccex static run.

The full discovery test count differs from the eventual final test count: test discovery imports modules before execution. A final scalar frozen group marker regression was added after discovery loaded its test module; B's test-only projection typing fix also occurred after discovery. Stable final product code was used for all reported full suites, and final focused6/7 tests were explicitly rerun. No counts from two invocations are merged or inflated.

Excluded runs: initial pre-fix A/B full discoveries were interrupted and restarted; a duplicate A post-fix cache-only suite was stopped with143 because stable full discovery covered it; a BASE related-log filename collision was resolved by interrupting both initial related runs and a unique clean rerun. The initial A KV-only test exposed real floating grouped PV/qparam propagation and redundant-Quantize issues before the final marker fix; these are not presented as passing results. Final full cache tests pass. Existing FP serialized Softmax rank failures remain recorded in export_summary.json and are not hidden by the collectors' successful process exit.

Large Circle/checkpoint/tensor dumps and raw logs are intentionally not committed. Export metadata includes hashes; all raw artifacts can be regenerated with the scripts beside this file.

Final report-script Pylint: `python -m pylint --jobs=1 --rcfile=.pylintrc docs/archive/gemma4_gqa_grouping/20261029T070221Z/*.py`, exit0,10.00/10 (all8 final Python scripts).

Final report-script static checks after annotation-only fixes:

- All8 final Python scripts: UFMT + repository selected-file MYPY adapters exit0 (export4 and numerical4 separate invocations), py_compile/syntax PASS.
- All8 final scripts: Pylint jobs1 exit0,10.00/10 (export4 and numerical4 separate final invocations).
- Initial combined report-script selected MYPY run exited1 for missing heterogeneous JSON/Counters annotations and registry return narrowing; all were corrected without ignores or runtime/metric changes.
- A separate direct mypy invocation on numerical scripts reported4 diagnostics in unmodified imported `tico/utils/mx` code. The repository's normal selected-file adapter filters diagnostics outside selected paths; its final selected-file checks pass. This is not a full-tree type-check claim.
- Reproduction scripts changed only type annotations/casts after measurement; timed code and metric values are unchanged. `run_exports.sh` passes `bash -n`. Final JSON parsing/Python syntax and byte-identical report-file checks across branches pass.
