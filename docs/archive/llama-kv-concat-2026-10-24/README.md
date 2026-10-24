# Tiny Llama KV-cache Circle topology investigation

This is an archived investigation of Samsung/TICO revision
`c0a3cbcf9c67cb2e760185eddf4d7c71901e534d`, fetched from `origin/main` on
2026-10-24. It records observed behavior at that revision, not a specification or
an assertion about later revisions.

Start with [the complete report](report.md), which includes all supporting
reports as appendices and can be shared as a single document.

Individual reports:

- [FX graph stages](fx_analysis.md)
- [Circle operators, tensors, Quantize inventory and cache ABI](circle_tables.md)
- [Numerical comparisons and native singleton-Concat probes](numerical_analysis.md)
- [Source review](source_review.md)
- [Commands and outcomes](commands.md)
- [Existing regression tests](baseline_tests.md)

The observed decode topology has no explicit Quantize feeding the main K/V
Concats: past input metadata and current producer metadata adopt the present
grid. O1 does not remove a cache Quantize. The controlled scenario exposes a
non-chainable cache ABI and stale metadata on separate delta-output branches.
See the report for numerical evidence and limitations.

The investigation originally left the working tree unchanged. These Markdown
reports were subsequently committed on a separate branch at the user's explicit
request for sharing. Historical statements about no commit or push describe the
original investigation phase. No production source, tests, generated Circle
models, PT2 files, tensor traces, or execution dumps are included in this archive.
Temporary artifact paths in the reports document the original execution and are
not repository dependencies.
