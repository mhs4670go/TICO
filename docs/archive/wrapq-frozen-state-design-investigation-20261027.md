# WrapQ frozen-state ownership: design investigation

## 1. Executive summary

**Recommendation:** make common WrapQ infrastructure own the transition from collected
statistics or imported qparams to validated, persistent frozen observer state. Give
one lifecycle coordinator authority to traverse, validate, and commit a model freeze.
Wrappers declare their active quantization sites and semantic constraints; they must
not implement independent versions of generic readiness, freezing, or grid persistence.

This is a correctness problem, demonstrated on canonical Samsung `main`, not a proposal
to remove code for appearance. A public `prepare` → `convert` sequence with no activation
calibration produces a structurally valid Circle linear model whose UINT8 input and
output scales are `9.99999993922529e-09`. Separately, imported locked grids disappear
in `state_dict`, Percentile ignores the common lock before consulting its reservoir,
failed freezes leave partially converted models, and nested wrappers are frozen
multiple times. Model-local checks repair only some of these failures.

The recommended design is deliberately narrower than redesigning graph quantization:
preserve current NPU tensor-domain lowering and final Circle contracts for valid
models. Reject missing/invalid calibration before export, preserve exact frozen grids
across supported saves, and make conversion failure recoverable. Estimated size is
medium-to-large: roughly five substantive review slices plus integration/docs, mostly
common infrastructure and model-hook migration, with no new dependency or model asset.

This report is the only intended repository change. It does not implement the design.

## 2. Repository baseline

| Item | Observation |
|---|---|
| Supplied checkout | Branch `work`, HEAD `9343af8419d5873260a027a3564eb3493c3e0428` |
| Initial status | Clean; no unrelated local changes |
| Configured origin | `https://github.com/mhs4670go/TICO.git`; its remote main was the supplied HEAD |
| Canonical repository queried | `https://github.com/Samsung/TICO.git` |
| Latest main observed, 2026-10-27 UTC | `126d7087a13af2ba7fa39d9e1493f813bfd39430` |
| Relationship | Supplied HEAD is its ancestor, six commits behind |
| Investigation checkout | Detached worktree at canonical main; original branch/HEAD unchanged |
| Python/dependencies | Python 3.12.14; Torch 2.13.0+cpu; Transformers 5.18.0; circle-schema 0.10.1; NumPy 2.5.4; editable TICO metadata 0.1.0 |
| Dependency health | `python -m pip check`: no broken requirements |
| Limits | CPU only, CUDA unavailable; no checkpoint/dataset downloads or external ONE/NPU compiler used; pytest absent, unittest available |
| Network | Managed restricted HTTP proxy, GitHub allowed, no VPN/credentials configured; canonical fetch succeeded |

All code references below describe **126d708**, unless explicitly identified as
history. In particular, `b002982` is included: first fake-quant markers annotate NPU
tensor domains. A design based only on supplied HEAD's float-preservation rules would
be stale. The other new commits are `a2d084e`, `1986844`, `264afd6`, `652d2cc`, and
`126d708`; none repairs the selected lifecycle issue.

Context read across the investigation: root and applicable quantization/pass/Circle/test
`AGENTS.md`; root README; `docs/{README,design,development,requirements,system_test,
torch_version_policy}.md`; WrapQ and quantization READMEs; recipe, adapter, stage,
export, debug/wrapper-smoke and example/config guides; Circle artifact/runtime guide.
These establish subsystem ownership, static stage boundaries, CPU synthetic tests,
and the distinction between structural verification and numerical/backend validation.
No separate contribution guide was found by the repository guide search. Existing
dated investigation convention was absent beyond `docs/archive/project-initiation/`;
this report follows the requested archive naming convention.

## 3. Candidate list and ranking

Rank prioritizes demonstrated impact, recurrence, a precise invariant, validation,
ownership reduction, then feasibility. Sizes are qualitative engineering estimates,
not measured line counts.

| Rank / candidate | Responsibility, subsystems and evidence | Failure / workaround / reach | Size, validation, migration risk |
|---|---|---|---|
| **1. Authoritative frozen observer lifecycle** | `ObserverBase`, `AffineObserverBase`, `PercentileObserver`, `QuantModuleBase`, `PTQQuantizer`, public convert, Gemma/Qwen cache validators and Gemma extra state. Missing stats accepted; lock/state persistence lost; multiple traversal and partial commit reproduced. | Invalid grids become real Circle ABI; loaded GPTQ grids change or cannot freeze; recovery fails. Gemma copies generic frozen grids and checks them locally; Qwen checks ranges at execution/export. Affects generic layers and all model families. | Medium-to-large, highly testable with tiny CPU fixtures. Moderate checkpoint/private-extension migration risk. Exactly one selected topic. |
| **2. Established graph domains versus propagation constraints** | `FoldQuantOps._inherit_qparam_from_view_chain`, `PropagateQParamForward/Backward`, `InsertQuantizeOnDtypeMismatch`, Quantize visitor, fixed schedule in `utils/convert.py`. Affine backward propagation replaces existing same-dtype grids; MX has stronger existing-domain/shared-user checks. | Actual Circle concat probe overwrites independent `.125` input/output domain with `.5`. Fold repeats view-chain inference to compensate ordering; `c0a3cbc` fixes one explicit-Quantize case. All quantized models. | Large, domain algebra and close nonmatching cases needed. High export compatibility risk after `b002982`; marker semantics must be resolved independently. Strong runner-up, but broader than lifecycle and less isolated. |
| **3. Stage ABI authority split between observer manifests and artifacts** | `recipes/export/{llama,llama_bundle,qwen3_vl,gemma4}.py`, runtime Llama consumer, static export adapters, staged tests. Llama inspects serialized roles; Gemma `_observer_manifest` reads private observer grids; Qwen lacks equivalent general bundle manifest. | Potential mismatch between Python grid declarations and final pass-derived domains; weak mocked exporter tests do not prove chainability. Current mitigation is family-specific assertions and Llama manifest inspection. All staged exports; strongest runtime consumer is Llama. | Large. CPU artifact inspection is feasible, but no uniformly supported cross-family runtime. Moderate/high manifest migration risk. Cross-layer risk confirmed; a present broken bundle was not established. |
| **4. Cache policy legality duplicated by model family** | Llama/Qwen/Gemma attention cache validation and static profiles. Llama permits sequence-axis per-channel cache policy that calibrates prefill but fails decode; Qwen rejects that policy; Gemma supports only a narrower affine subset. | Tiny Llama B=1,H=2,KV=1,D=4,S=3 prefill → S=1 decode fails scale-length validation with channel_axis=2. Storage/compute fixes landed in three separate commits. A shared token-locality checker could prevent drift, but Gemma restrictions and shared-KV semantics differ legitimately. | Medium. Tiny attention fixtures can validate; moderate config rejection risk. Narrower reach than candidate 1; do not unify supported scheme sets merely for consistency. |

Candidate 1 wins because several independent public-path failures share one missing
owner and one testable contract. Candidate 2 has direct artifact evidence too, but
requires a separate decision about backend domain unification. Candidates 3 and 4
remain useful follow-up work; this implementation must not absorb them wholesale.

## 4. Selected design problem

The word **frozen** currently has no single authoritative meaning. It can mean:
`_mode == QUANT`, observers disabled, nonempty `_cached_scale`, valid collected ranges,
a locked externally supplied grid, or a model-specific cache check. Those predicates
are neither equivalent nor committed together.

Required invariant:

> For every enabled quantization site admitted by a converted model's supported
> execution capabilities, the site's quantization semantics are valid and stable.
> An affine site has a validated grid derived from sufficient statistics or explicitly
> imported compatible qparams; Identity and MX have explicit non-affine readiness.
> Freeze either commits the complete required set once, or changes none of it.
> Saving/restoring frozen state preserves its exact grid, policy, provenance and lock.

An optional capability can be unavailable after calibration, but attempting it must
fail before quantization/export. Missing stats must never be turned into a floor
scale and advertised as readiness. Valid all-zero calibration is different from no
calibration and retains the existing numerical floor rules.

## 5. Evidence

### E1: Missing activation calibration reaches the final artifact

On latest main, prepare a bias-free 2×2 `nn.Linear`, then convert without a calibration
forward. `QuantLinear.enable_calibration` observes weights immediately; its input and
output activation ranges remain `(+inf,-inf)`. `AffineObserverBase.compute_qparams`
does not reject these sentinels. In the asymmetric path its negative range is clamped
to `1e-8`; zero point becomes zero. `has_qparams` checks only nonempty scale.

Observed artifact, default Circle O1 enabled:

| Tensor | Final dtype | Scale | Zero point |
|---|---|---|---|
| Linear input | UINT8 | `9.99999993922529e-09` | 0 |
| Linear output | UINT8 | `9.99999993922529e-09` | 0 |
| Weight | UINT8, per-channel axis 0 | approximately `[.0019607844,.0049019610]` | `[0,51]` |

One `FULLY_CONNECTED`, no additional arithmetic operator; `CircleDocument.verify()`
returns `VerificationReport(issues=())`. Structural validity cannot establish that
the grid has a calibration origin. Signed symmetric unobserved ranges can instead
produce infinite scales; the issue is not limited to unsigned floor clamping.

### E2: Frozen grids are not recomputable from persisted generic state

`AffineObserverBase` registers `_cached_scale/_cached_zp` as nonpersistent because
they supposedly can be recomputed; `_qparams_locked`, enabled flags, and wrapper mode
are ordinary attributes. A MinMax observer loaded with scale `.3`, zp `7`, lock true
saves only `min_val/max_val`. Fresh load loses both grid and lock. With no stats,
recompute gives `(1e-8,0)`. A calibrated `QuantLinear` with imported weight scales
`[.3,.3]` changed after state-dict restore/refreeze to approximately
`[.00095699396,.00260338443]` in the deterministic seed-71 probe.

Percentile is a second counterexample to recomputability: its reservoir and clipping
bounds are not persistent. A frozen percentile state dict cannot regenerate its grid
from global min/max. Further, even a fresh `PercentileObserver.load_qparams(...,
lock=True)` followed by `compute_qparams` raises `No calibration values were
collected`: the subclass consults `_clipping_bounds` **before** the base lock check.
This is a current failure without any serialization step.

Whole-module `torch.save/torch.load(weights_only=False)` preserves these buffers and
attributes today; it is the recipe checkpoint format and must not be confused with
state dict. A loaded Percentile whole-module object can fake-quantize an imported
grid but still fails if a caller recomputes/freezes it without samples.

### E3: Repeated lifecycle work and non-atomic failure

`PTQQuantizer.prepare/convert` calls the lifecycle at the root and again at every
`QuantModuleBase` in `model.modules()`. The base methods already recurse. A nested
root/middle/leaf probe recorded freeze calls **2/3/4**. The old local-observer cleanup
did not remove the quantizer's overlapping traversal.

A two-observer module whose second observer raises during qparam computation is left
in QUANT; its first observer is disabled with cached qparams, the second disabled
without qparams. Public `convert` also deletes `tico_quantizer` before calling the
quantizer: failure destroys the public retry handle. This is a reachable failure
pattern with Percentile, not only with the injected failing observer used to isolate it.

### E4: Model-local repairs expose the absent common contract

- Gemma4 attention implements its own `freeze_qparams`, `_has_cache_statistics`,
  `_validate_frozen_observer`, and `get_extra_state/set_extra_state` (roughly lines
  570–844). These mix general finite/positive/grid checks with legitimate Gemma cache
  policy and source-binding checks. They persist four cache grids, not every generic
  activation/weight grid.
- Qwen `require_calibrated_cache` (roughly 550–596) checks persistent ranges and cached
  grids during QUANT execution/export. It requires statistics even when a compatible
  imported grid might suffice; Gemma explicitly permits loaded grids. Its generic
  grid checks are independent from the base freeze.
- Llama's active observer filtering was improved in `b704b57`, but it lacks the same
  readiness guard; a tiny uncalibrated attention freeze succeeds.
- Qwen lists `obs_logits_raw/obs_scale` even when `scale_fusion='k_norm'` skips them.
  Llama filters those sites. A new strict freeze cannot simply reject every allocated
  observer without migrating active-site declarations.

### E5: History and tests explain recurrence

`5d1e089` (#733) separated local observer ownership from recursive lookup to avoid
double processing. Blame shows base recursion from `d122de81`, and the overlapping
quantizer loop from `a33008d2`, remained. `8da8a14` (#866) added locks only in the
affine base. `82e2ef5` (#877) canonicalized imported per-channel shapes. Neither
established a subtype-wide or persistence contract. `42b796b` (#1038) fixed EMA
statistics aliasing but not frozen snapshot aliasing/persistence.

Recent `e1caa21`, `b704b57`, `5620abe`, `9343af8` repaired cache domains locally.
They correctly separate storage and compute but leave their validity tied to
different lifecycle predicates. Repeating local guards does not repair generic
Linear, Percentile, recursive conversion, or state-dict loading.

Existing tests pass despite E1–E3. `TestQuantModuleBase.test_mode_cycle` explicitly
freezes without collecting anything and merely asserts mode/`has_qparams`; its
expectation encodes the bug and must be replaced. `test_extra_repr` also freezes an
uncalibrated dummy. Affine lock tests cover one instance, not subtype-wide lock
handling or save/load. Gemma source/cache tests protect useful semantic contracts,
but do not prove all observers survive restore. Details and commands are in §17.

## 6. Current behavior and data flow

Example path (same core infrastructure for Llama/Qwen/Gemma projection wrappers):

```text
public prepare(config)
  -> PTQQuantizer / PTQWrapHelper / PTQWrapper and registered wrappers
  -> enable_calibration (recursive + global traversal)
       resets observers; parameter wrappers collect fixed weights
  -> optional recipes.qparams.inject_gptq_qparams: load_qparams(lock=True)
  -> calibration forwards: QuantModuleBase._fq -> collect, return float input
  -> public convert deletes quantizer handle
  -> freeze_qparams sets QUANT, disables and computes observers, recurses
  -> QUANT forward: fake_quant uses cached buffers; MX uses runtime scaling
  -> torch.export, fake-quant decompositions -> Q/DQ with qparam constants
  -> FoldQuantOps: first marker becomes QPARAM_KEY domain metadata
  -> RemoveWeightDequantOp: quantized constant data + weight QPARAM
  -> forward/backward propagation -> clamp -> safe constant propagation
     -> bias quantization -> unused placeholders -> dtype bridges
  -> CircleSubgraph.add_tensor_from_node reads QPARAM for final dtype/grid
  -> serialization -> default Circle O1 -> artifact/runtime input contract
```

Graph passes do not calibrate and cannot reconstruct omitted observation. Their
ability to propagate/transform grids must not be used to infer readiness. There is
no calibration-provenance field in Circle, nor should one be required to repair E1.
The missing check belongs before export.

Current first-marker NPU annotation semantics are important: intermediate FX
`meta['val']` may remain floating; final Circle tensors are integer by QPARAM.
Real subsequent domain transitions must remain explicit Quantize where the current
lowering requires them. Intermediate Python numerical parity alone cannot establish
the final ABI, hence E1 inspects serialized bytes.

For cache paths, current projected KV branches independently to `new_k/new_v`
storage S and `present_key/present_value` compute C. Past S reads convert to C;
concatenation/attention consumes C; persistent deltas and full caches stay S.
Prefill and decode must reuse exact producer storage grids. Llama HF DynamicCache
and StaticCache mutation, Qwen multimodal/static geometry, and Gemma shared producer
bindings are semantic contracts, not generic observer arithmetic.

## 7. Current responsibility map

| Invariant/transformation | Current owner and other writers | Readers / ordering / assumptions | Failure and tests |
|---|---|---|---|
| Active local sites | Wrapper `_all_observers`; conditional family filtering; attrs may remain dormant | Base lifecycle, introspection, site control; debug names are not necessarily module-tree names | Inactive sites freeze or missing site is hidden; attention-specific tests |
| Tree lifecycle traversal | Base recursive methods **and** PTQQuantizer global loop | Parameter collection overrides call super; arbitrary plain containers are only reached by global loop | Repeated reset/compute; #733 tests check lookup, not whole traversal counts |
| Statistics sufficiency | Affine assumes ranges; Percentile reservoir check; Gemma/Qwen cache checks | Must precede deriving qparams; imported locks should bypass collection | Missing ranges become floor/infinite scale; mode-cycle test accepts it |
| Grid computation/import | Affine compute/load; Percentile overrides compute; GPTQ transfer | Calibration then locked injection then activation collection then freeze | Lock bypass inconsistency; per-instance affine tests only |
| Frozen validity | `has_qparams`, fake_quant; Gemma/Qwen duplicate validators | Consumers assume cached values are finite/positive and shape compatible | Nonempty is misleading readiness; cache checks cover subsets |
| Freeze commit and mode | Base and Gemma override; public convert removes handle early | Child failure occurs after parent mutations; repeated passes assumed harmless | Partial QUANT, disabled observers, no retry handle; no atomicity test |
| Frozen persistence | Generic persistent ranges, transient grid; whole-module pickle; Gemma extra state | Recompute assumed possible; source bindings loaded in expected order | Lost locks/grids; percentile reservoir absent; Gemma cache-only tests |
| Q/DQ and graph domain lowering | Fake-quant decomposition, Fold, weight DQ removal, propagation | Valid qparams supplied; fixed schedule; FX dtype differs from ABI | Invalid-origin but well-formed grid survives; pass tests prove local transforms |
| Circle dtype/grid | Serializer QPARAM conversion; artifact optimization preserves/transforms tensor contracts | Final metadata, not source observer or FX float dtype | Structural verifier cannot identify missing calibration; Circle executor tests cover calibrated cases |
| Stage ABI | Family exporters, frozen observer consumers, final artifact readers/manifests | Static shapes, qparam reuse, producer-consumer identity | Independent manifest authority risk (candidate 3); staged cache tests |
| Export-generated constants | Gemma vision pooler export adapter resets/collects/computes pool-weight observer in QUANT | Profile creates a new deterministic constant after calibration | Mutates frozen source state; requires explicit export-local finalization, not a global ban on new qparams |

Duplicated ownership is clearest in traversal and generic grid validation/persistence.
There is circular reasoning in QUANT ⇒ ready ⇒ `has_qparams` ⇒ compute succeeded,
when compute itself never established statistics readiness. Freeze ordering is
observable on errors. Gemma source load ordering is a separate semantic dependency
that must remain validated rather than hidden by general qparam restoration.

## 8. Root structural cause

The interface bundles observation, qparam derivation, mutable caching, and execution
under `compute_qparams`, without a readiness or commit contract. Subclasses can do
work before the base lock branch. `QuantModuleBase` treats a successfully returned
tuple as valid and changes mode before asking for it. Serialization assumes cached
grids are disposable derivatives, though imported and percentile grids are not.

Model wrappers compensate by inspecting private fields. The same backend-wide
condition therefore has different owners depending on which cache family or export
entrypoint executes it. A traversal fix, an extra `isfinite`, or persisting scales
alone would each leave other parts of the same frozen-state invariant broken.

## 9. Required external contracts

Preserve for valid inputs/configurations:

- Public `prepare(model, quant_config, args=None, kwargs=None, inplace=True)` and
  `convert(model, inplace=True)` signatures and algorithm dispatch. Keep GPTQ's
  existing in-place requirement; this work targets WrapQ PTQ lifecycle.
- YAML/CLI keys, supported affine/MX/Identity policies, dtype/range/rounding,
  calibrated numerical behavior, GPTQ scale adjustments and observer `fp_name` mapping.
- `prepare -> inject and lock compatible GPTQ weights -> calibrate -> convert`;
  loaded weight grids remain exact while activation observers collect.
- Recipe whole-module `.pt` save/load and independently supported state-dict paths,
  with explicit version/migration errors for unrecoverable old state.
- Current Circle tensor domain lowering, default O1, serialized I/O order, names,
  shapes, dtypes, qparams, and S/C cache ABI for equivalent valid frozen state.
- Prefill/decode, cache-disabled inference, StaticCache/DynamicCache behavior,
  Gemma shared-KV and external float-cache binding restrictions, per-layer export,
  and the distinction between reference fake-quant execution and integer backend proof.

Intentional behavior changes: missing required calibration, invalid imported qparams,
incompatible policy restores, and partial/ambiguous frozen checkpoints fail clearly;
they must not silently acquire new grids. Frozen state-dict restores become complete.
Do not preserve the stale README example `convert(model, quant_config)`: the second
argument is currently `inplace`, not a supported configuration overload.

## 10. Target responsibility map

| Responsibility | Sole owner | Other stages may do |
|---|---|---|
| Declare active/conditional sites, names and execution capability groups | Model/operator wrapper | Common collector enumerates declarations, never guesses from observer attribute names |
| Statistics collection and algorithm-specific range derivation | Observer subtype | Parameter preparation hook observes deterministic parameters; adapters supply calibration inputs |
| Validate affine policy/grid, imported-grid precedence, snapshot serialization | Common affine observer state layer | Algorithm hooks propose candidate grids; consumers request read-only snapshots |
| Identity/MX readiness and policy | Their observer subtype under common protocol | No invented affine sentinel grid used as readiness evidence |
| Traverse, build freeze plan, validate all required sites, commit modes/flags once | Common WrapQ lifecycle coordinator | Public/base methods delegate; model hooks validate plans without mutation |
| Cache token locality, scheme subset, shared source ownership and binding | Family wrapper/profile | Reuse common grid validation; express dependency references, do not copy producer ownership |
| Artifact-local constant grid creation | Export lowering that creates that constant | Finalize a separate local observer/snapshot via common validator; do not reset source observers |
| Q/DQ to final graph domains | Existing graph quantization pipeline | Read frozen inputs, perform documented transformations, validate resulting domain contracts |
| Circle tensor ABI | Existing serializer and artifact validators | Read final graph domains, never recompute model qparams |

## 11. Recommended design

### 11.1 Common observer state and candidate protocol

Introduce a small typed state representation in existing `wrapq/observers/` or one
adjacent common module. Names here are design vocabulary, not required public APIs:

- `FrozenAffineState`: detached owned scale/zp tensors, dtype/qmin/qmax, scheme,
  granularity/axis, provenance (`calibrated` or `imported`), lock, fake-quant bypass
  state, schema version.
  Represent clipping provenance sufficiently for diagnostics; do not persist a whole
  Percentile reservoir merely to execute a frozen model.
- Identity state: explicitly bypassed/float policy; MX state: supported dynamic
  microscaling policy (format, axis, rounding, exponent method), without affine stats.
- Observer operations: derive a **candidate without mutation**, validate it, install
  it, retrieve frozen state, reset for a new calibration cycle, save/load its schema.
  The common affine layer chooses a locked imported grid before invoking any subtype
  statistics hook. Percentile supplies clipping bounds through that hook.

Affine validation must check nonempty compatible scale/zp shapes, finite positive
scales, integer/in-range zp, policy/range compatibility and channel-axis meaning.
Each site descriptor supplies expected tensor rank/channel extent: parameter shape
is known at prepare; activation extent comes from its observed schema or an explicit
wrapper boundary contract. Per-channel imported grids with unknown extent remain
unready until the site can validate compatibility; never guess it from qparam length
or tensor rank alone.
Reject fractional zp before conversion to integer rather than silently truncating.
Preserve current valid scale dtype and canonical 1D per-channel behavior. Statistics
validation distinguishes unobserved, degenerate finite, invalid nonfinite, and
algorithm-specific sufficient data. Preserve Percentile's documented finite-sample
filtering policy; do not change collection mathematics as a side effect.

Cache or caller access must not mutate a frozen state through a shared tensor alias.
Clone imported tensors on ownership transfer; `.to()` must retain proper buffers and
device movement. A dataclass marked frozen is insufficient to make tensors immutable.
Private `_cached_*` may be transitional accessors, then removed from production
consumers; a pure explicit recomputation API must not overwrite a committed grid.
Legacy observer helper calls may delegate to validated candidate/finalization logic,
but `compute_qparams` must not retain subtype-dependent lock semantics.

### 11.2 One coordinator, staged freeze, explicit hooks

The coordinator walks actual `nn.Module` ownership through **all** containers,
including arbitrary plain modules, deduplicating module and observer identities.
Do not simply delete the global loop: `_child_quant_modules` currently descends only
ModuleList/ModuleDict/Sequential, so the global loop also compensates for a traversal
hole. Keep transparent PTQWrapper debug naming separate from ownership traversal.

`QuantModuleBase.enable_calibration/freeze_qparams` remain convenient entrypoints
delegating to the coordinator on their subtree. PTQQuantizer calls it once for any
root, including an ordinary module. Replace recursive subclass lifecycle overrides
with local hooks: reset/parameter collection, site declarations, plan validation.
The coordinator first resets the selected sites, then invokes local parameter hooks
once. Preserve parameter-statistics timing and GPTQ injection after prepare.

Freeze performs four steps:

1. Enumerate stable site identities, active/capability requirements and external-source
   dependencies; reject missing or conflicting ownership.
2. Derive/validate candidates without changing modes, flags, cached grids or samples.
3. Run wrapper semantic checks against that complete plan (including borrowed producer
   grids). No qparam computation or mutation in these checks.
4. Install validated snapshots and disable collection, then enter QUANT. Repeated
   freeze is idempotent unless an explicit new calibration/reset began.

No full model deepcopy is needed for transactionality: stage small qparam tensors
and scalar state only. Commit must use prevalidated non-failing operations or restore
the small changed state if an installation hook fails. Public convert removes the
quantizer handle only after success; `inplace=False` must not strip the original
prepared model's handle. Test both paths, without broadening unrelated quantizer APIs.

### 11.3 Active and conditional sites

Migrate structural inactivity before enforcing strict readiness: fused scale sites,
unused RoPE paths, Gemma shared consumers versus KV producers, identity/bypass sites.
Inactive means unreachable under declared semantics, not merely never observed.
Do not globally filter out every unobserved observer; that would preserve E1.

Cache-disabled workflows require explicit conditional readiness. Wrappers declare a
cache-output capability group separate from ordinary no-cache attention. If its
required observations are missing, ordinary calibrated no-cache inference can remain
available, but enabling cache output, decode or a cache-bearing export must fail
with the missing site/group before execution. A model may be QUANT with an explicitly
unavailable capability, not with silently fabricated grids. Required default linear
activation sites are never optional. Preserve public signatures; prefer deriving
groups from existing wrapper/export modes rather than adding YAML switches.

Capability requirements must describe actual **reads, computations, writes and
returns**, not simply `use_cache`. `use_cache=False` can suppress writes/returns
while past/shared KV remains an attention input. `present_*` compute grids are core
attention requirements. Qwen currently applies `new_*` fake quant even without a
returned cache; Gemma producer paths also compute storage branches, whereas Llama
unrolled storage collection occurs in output finalization. An optional unavailable
storage branch may be skipped only when proven unconsumed for that execution path;
valid calibrated paths must retain their existing branch/numerical behavior. Tests
must cover `use_cache=False` with and without past/shared inputs.

Partially calibrated optional groups must be unavailable as a whole. Export adapters
request readiness for their exact prefill/decode group. Qwen's finite-range heuristic
is replaced by common calibrated-or-imported provenance, subject to family policy;
valid loaded qparams alone are allowed, not unvalidated hand-edited caches. Gemma
continues to reject unsupported MX/per-channel cache conversion. This does not adopt
candidate 4's full shared cache-legality abstraction.

### 11.4 Persistence and loading

Persist exact frozen states, observer policy, lock/provenance and wrapper mode/group
readiness in a versioned schema. Statistics alone are insufficient. Load supports
dynamic per-channel buffer sizes and validates policy before admitting execution.
Validate references and cross-module capability readiness after the whole subtree
has loaded; avoid dependent checks on a half-loaded producer. Never infer frozen mode
from finite min/max or silently accept invalid frozen state through `strict=False`.

Frozen snapshot restoration does not promise mid-calibration resumption: Percentile
reservoir/RNG resumption is separate, and calibration-only state lacking required
samples must explicitly require recalibration. Legacy min/max-only states can restore
as **not frozen** and be recalibrated; they cannot claim exact imported/Percentile
restoration. Legacy whole-module objects containing valid cached grids may be upgraded
through an explicit validator at the load boundary. Unsupported or ambiguous versions
must raise actionable reprepare/recalibrate errors, never fall back to new grids.

Gemma keeps source-binding metadata, source policy compatibility and obsolete-cache
rejection. Remove its independent generic grid-copy schema once the common snapshot
loader is authoritative; a documented adapter may read version-2 Gemma cache state
into common state if all required information exists. Version-2 Gemma state dicts
contain only cache frozen grids: recovering those grids alone cannot promote the
whole model to QUANT when exact non-cache snapshots are absent. Keep it unready and
require recalibration; never recompute the other grids from ranges and advertise an
exact frozen restore. Do not revive removed `past_*`
observer state or interpret it as S/C. Old objects missing necessary information fail.

### 11.5 Export and future model support

Export consumes ready snapshots and does not recalibrate source-model activations.
Gemma pool weights are a legitimate export-profile-generated constant: give the
adapter a separate local observer/snapshot, observe that deterministic constant, and
finalize through the same validator. Source snapshots must remain unchanged across
profile exports, including failures. Migrate smoke tooling that calls compute directly
to explicit fixture finalization. No production graph pass learns model names.

A new model implements active sites, capability groups, parameter collection and
semantic plan validation. It uses common snapshots for weights/activations, references
shared producers instead of copying grids, and gains common contract tests. It does
not implement generic freeze recursion, finite-scale checks or generic serialization.

## 12. Alternatives considered

| Alternative | Ownership / complexity | Correctness and order | Compatibility / tests / extension / migration |
|---|---|---|---|
| Add guards to each wrapper and persist more local extra state | Wrappers keep readiness/persistence; small initial patches, growing maintenance | Fixes cache cases but misses generic Linear/Percentile, duplicate traversal and partial failure; checks still occur after mutation or only at export | Lowest initial migration; tests proliferate by family; new models must rediscover rules. Rejected: repeats demonstrated history. |
| Infer readiness and reconstruct all qparams at export | Exporter/graph passes own finalization; moderate-to-large changes | Cannot recover imported grids or Percentile clipping from ranges; eager QUANT remains unsafe; export mode/order changes calibration semantics | Breaks save/reload and non-export inference; weak independently testable contract; every export path must implement it. Rejected. |
| Persist grid buffers and validate in existing recursive freeze | Observer owns values, wrappers/quantizer retain traversal; medium patch | Improves restore but still computes/mutates before full-model validation; local overrides can violate lock; same observer may reset repeatedly | Smaller migration but leaves failure recovery and once-only responsibility unresolved. Useful intermediate commit only, not final architecture. |
| **Recommended: observer-owned snapshots + common freeze coordinator** | Observer owns validity/serialization; coordinator owns transaction; wrappers own site semantics. Medium-to-large coherent change | Explicit prepare/inject/calibrate/freeze order, independent candidates, one commit, no graph-pass dependency for readiness | Private hooks migrate; versioned checkpoints; shared CPU contract suite; future wrappers declare rather than reimplement. Highest one-time migration, strongest recurrence reduction. |

The recommended design is the smallest architecture that resolves all parts of the
selected invariant; it is not the smallest textual patch.

## 13. Compatibility and migration policy

Do not promise byte-identical pickle encodings or preserve private observer fields.
Do promise exact frozen tensor grid values/policy and equivalent artifact semantics
for valid models. Compare artifact contracts rather than byte ordering when internal
buffer layout changes. Keep current float-versus-quantized NPU domain semantics.

Migration must distinguish: fresh prepare; imported locked grids; valid calibrated
frozen checkpoints; legacy range-only state dicts; obsolete cache-domain checkpoints;
and explicitly stateless Identity/MX. No single permissive `has_qparams` fallback.
Update model validators to call the common protocol while retaining semantic checks.
Keep an explicitly bounded legacy reader if it can prove exact restoration; otherwise
fail with reprepare/recalibrate guidance. Document each accepted schema version.

Local preparation hooks in NN Conv/Linear/Embedding/PReLU/LayerNorm/RMSNorm and Gemma
embedding/vision modules must migrate together with the coordinator. Out-of-tree
private lifecycle overrides need a documented migration to hooks. Detect unsupported
legacy mutating observer/lifecycle overrides and reject with hook migration guidance;
never invoke them inside the supposedly pure freeze plan. Existing public
adapter/quantizer registries and GPTQ name matching remain.

## 14. Detailed implementation slices

Each slice should merge with passing focused tests; do not land failing regression
tests alone. Exact new filenames are flexible within established owners.

### Slice 1 — Common candidate validation and affine lock precedence

- Objective: establish read/derive/install primitives and validation without yet
  changing model-wide traversal.
- Files/symbols: `observers/{base,affine_base,percentile,identity,mx}.py`; introduce
  common state types here, not in exporters. Separate Percentile clipping derivation
  from lock selection. Validate imports and opt-in candidate operations here. Keep a
  temporary adapter for existing model-freeze callers until active-site migration;
  broad missing-statistics rejection at model freeze activates only in slice 3.
- API/ownership: common affine entrypoint honors imported lock before subtype hooks;
  candidate validation is nonmutating; Identity/MX readiness explicit.
- Tests: unobserved vs finite zero range; signed/unsigned; NaN/inf; bad scale/zp,
  shape/axis; imported tensor alias isolation; all affine subtypes honor locks;
  candidate failure leaves state unchanged. Test unobserved rejection through the new
  candidate protocol; replace old empty-model-freeze assertions in slice 3. Do not
  break dormant-site model flows before slice 2 is complete.
- Artifact check: calibrated 2×2 Linear keeps existing input/output/weight grids;
  direct invalid imports cannot reach Circle. No need for large models.
- Dependency: none. Rollback: helper introduction can be reverted before migration;
  do not keep a second unchecked import path for compatibility.

### Slice 2 — Active declarations and local lifecycle hooks

- Objective: inventory every wrapper lifecycle override and migrate local preparation
  while preserving supported execution paths.
- Files/symbols: `QuantModuleBase`, `PTQWrapper`, `control.iter_quantization_sites`,
  NN and ops wrappers with `enable_calibration`; Llama/Qwen/Gemma `_all_observers`,
  fused scale paths, shared-KV and vision/embedding preparation hooks.
- API/ownership: declarations expose required versus conditional sites and semantic
  dependency references. Preserve debug names and `fp_name`. Local hooks never recurse.
- Tests: arbitrary plain-module container; shared module/observer aliases; transparent
  wrapper names; each parameter collected once; fused/nonfused scale; no-cache groups;
  Gemma source-consumer site membership; existing quantization-site controls.
- Artifact check: tiny calibrated prefill/decode and cache-disabled wrappers export
  with the same shapes/dtypes/grids as baseline; no inactive observer is required.
- Dependency: slice 1. Failure/rollback: keep old base entrypoints delegating through
  transitional hooks until slice 3; never filter missing stats as a substitute for
  declaring structural inactivity.

### Slice 3 — Once-only transactional model freeze

- Objective: make common coordinator the sole traversal and commit owner.
- Files/symbols: `wrapq/quantizer.py` prepare/convert, `QuantModuleBase` entrypoints,
  a small common lifecycle module if needed, `public_interface.convert`, Gemma custom
  freeze and Qwen calibrated-cache checks.
- API/ownership: enumerate/plan/validate/commit once; readiness request for capability
  groups; repeated freeze idempotent. Preserve retry handle until success; preserve
  original prepared object in non-in-place conversion.
- Tests: nested counts become one, plain container coverage, failure at parent/child/
  last observer leaves all modes/flags/grids unchanged, retry succeeds after fixing
  input; imported lock unchanged; required versus unavailable optional capability;
  real Percentile failure in public path; no calibration Linear rejects before QUANT.
- Artifact check: E1 can no longer emit a Circle file; calibrated baseline still
  has expected UINT8/INT16 boundaries. Inspect default-O1 artifact, not just FX.
- Dependency: 1–2. Delete old overlapping global+recursive loops and Gemma generic
  freeze loop in this slice. Failure consideration: transaction stores only qparam
  state, not large model copies; semantic hooks must be pure.

### Slice 4 — Persist exact frozen state and validate loads

- Objective: preserve committed semantics and establish explicit legacy policy.
- Files/symbols: observer state-dict save/load hooks and variable buffer handling;
  wrapper mode/capability restoration; Gemma `get_extra_state/set_extra_state` and
  source post-load validation; recipe checkpoint loading/validation entrypoints.
- API/ownership: one versioned observer schema, deferred cross-module reference check,
  explicit whole-module upgrade and legacy range-only treatment.
- Tests: calibrated MinMax/EMA/Percentile and imported locks, per-channel shapes,
  CPU/device movement, strict and non-strict incompatible policy loads, corrupt
  frozen fields/version, Identity/MX, whole-module roundtrip, Gemma producer-consumer
  load/binding mismatch; calibration-only Percentile requires recalibration;
  fake-quant bypass state roundtrips, and enabling an unready bypassed site fails
  before execution.
- Artifact check: export before and after reload; compare every public boundary's
  dtype/shape/grid and quantized constants, prefill→decode chain and outputs. Preserve
  loaded GPTQ grids exactly, not merely close numerical outputs.
- Dependency: 1–3. Delete Gemma's independent generic grid-copy writer, retain bounded
  legacy reader only if exact restoration is provable. Rollback requires explicit
  schema-version handling; do not rewrite user's checkpoints in place.

### Slice 5 — Snapshot consumers and export-local constants

- Objective: remove private cache-field reads and post-freeze source mutation.
- Files/symbols: `recipes/export/gemma4._observer_manifest`, assistant LM-head extraction,
  `debug/{cpu_validation,export_regression}.py`, wrapper-smoke Gemma fixture finalizers,
  `QuantGemma4VisionPooler.as_prefill_export_adapter`, cache export readiness methods.
- API/ownership: manifests/inspection read snapshots; generated pool constant gets
  its own local finalized grid. Model-specific policy/source validation remains.
- Tests: source snapshot fingerprints unchanged after each export/profile/failure;
  actual per-layer artifact chaining; generated pool constants numerically unchanged;
  no execution/export caller computes source qparams lazily.
- Artifact check: inspect actual Circle stage tensor grids and explicit Quantize
  transitions, default O1, two small vision profiles for export-local isolation.
- Dependency: 3–4. Delete transitional private-field consumers after migration; avoid
  accidentally moving candidate 3's complete manifest redesign into this slice.

### Slice 6 — Cross-family integration, documentation and removal audit

- Objective: make the contract executable for maintainers/new wrappers.
- Files: WrapQ/quantization/recipe export and stage READMEs, example/config references
  where readiness/checkpoint behavior is described, relevant tests in existing owners.
  Correct stale public convert examples. Update `docs/design.md` lifecycle boundary.
- Tests: full affected unit groups and non-model suite; matrix below, oldest supported
  Torch smoke in CI. Replace implementation-only mode/empty-cache assertions rather
  than preserving them with fake stats. Retain genuine S/C/source/ABI tests.
- Artifact check: baseline-versus-target small family fixtures, save/load and per-layer
  prefill/decode. Record structural, fake-quant numerical and backend results separately.
- Dependency: all earlier slices. Success gate: no independent production generic
  freeze/persistence path and no hidden recalibration during execution. If a supported
  variant cannot be proven, block merge or make its unsupported status explicit;
  do not broaden skips, weaken tolerances, or add a silent fallback.

## 15. Validation matrix

All required structural fixtures below run on CPU, need no network, model weights,
dataset assets, GPU, ONE installation or NPU compiler. Construct Hugging Face configs
locally. An optional downstream backend run is separate evidence, not a prerequisite
for proving the selected invariant.

| Validation | Smallest fixture and property proved | Expected result | CPU / assets |
|---|---|---|---|
| Statistics readiness | Scalar/vector MinMax, EMA, Percentile: untouched, empty, zero, constant, nonfinite | Missing required stats reject; valid finite degenerate ranges preserve old math; subtype-specific finite filtering preserved | Yes / none |
| Imported grid contract | Each affine subtype; scalar and C=2 grids, bad scales/zp/policy/axes, aliased input tensors | Compatible locked state bypasses statistics exactly; invalid input rejects before mutation; later input edits cannot alter grid | Yes / none |
| Stateless modes | Identity and MX on a tiny 2D tensor, fake-quant enable/bypass toggles | No fabricated affine readiness; existing float/MX math preserved; policy validation explicit | Yes / none |
| Once-only traversal | Three nested modules, ordinary container, tied child, aliased observer, PTQWrapper | Each site reset/parameter-collected/frozen once; names and fp mappings stable | Yes / none |
| Atomic freeze/public recovery | Two sites/child with one Percentile missing samples, both inplace settings | No partial modes/flags/grids; public handle retained on failure; successful retry; original preserved for inplace=False | Yes / none |
| Serialization | C=2 QuantLinear, imported MinMax/EMA/Percentile; fresh receiver, malformed/legacy state; whole-module BytesIO | Exact scale/zp/lock/policy/readiness roundtrip; corrupt/ambiguous frozen loads reject even nonstrict; calibration resume not implicitly promised | Yes / none |
| GPTQ handoff | Existing synthetic folded-linear and attention handoff fixtures | Compatible locked weights unchanged through calibration/freeze/reload; non-target activations collect; missing requested reuse still errors | Yes / none |
| No-calibration export regression | Deterministic 2×2 bias-free Linear from E1 | Public convert fails before QUANT/artifact creation; positive calibrated control exports | Yes / none |
| Calibrated Circle contract | Same Linear; UINT8 and signed INT16 policy, C=2 weight grid | Deserialize final bytes: exact dtype/grid/axis, FC topology, no unexpected conversion ops; verify succeeds; fake-quant parity uses measured grid rounding tolerance | Yes / none |
| Family readiness smoke | Local tiny Llama/Qwen text/Gemma4 configs, one attention/layer; fusion on/off | Active required sites ready; fused dormant sites excluded; preserve semantic family differences | Yes / none |
| Cache capabilities | Same fixtures, cache off/on, prefill S=3 and decode S=1 | No-cache inference remains supported, including past/shared input reads where supported; unready cache group rejected before use/export; prefill-only with cache enabled can freeze storage without decode stats | Yes / none |
| Persistent cache ABI | Tiny Llama and Gemma producer DynamicCache/StaticCache, Qwen supported cache paths, delta/present modes and no-write past reads | Persistent values remain S, compute values C; capacity/update semantics and exact prefill/decode boundary grids unchanged | Yes / none |
| Shared Gemma KV | Two compatible producer/consumer layers plus explicit float-source variant | Producer state borrowed, not recalibrated/copied independently; wrong binding/policy restore rejects; local constraints preserved | Yes / none |
| Per-layer Circle chaining | One tiny decoder, prefill and decode exports, before/after save/load | Inspect serialized cache and hidden boundary dtypes/shapes/scales/zp; append S delta and reuse without host requantization; compare staged outputs with same frozen eager fixture | Yes / none |
| Q/DQ topology | Two intentionally distinct frozen S/C grids | Final Circle Quantize nodes represent required transitions; no accidental replacement of final grid by an FX float dtype; follow current b002982 domain policy | Yes / none |
| Generated constants | Tiny Gemma pooler, two static profiles, success/failing export | Correct profile-local pool grid; source observer snapshot unchanged; verify and compare resulting pool computation | Yes / none |
| Inspection tools | Snapshot-based manifest, debug fingerprint and CircleDocument inspection | No inspection recomputation/mutation; metadata matches inspected artifacts where promised | Yes / none |
| Broad regression/style | Focused groups, `./ccex test`, `./ccex format --no-apply-patches`, `pip check`; supported oldest/default Torch smoke | No widened skips/tolerances; public examples/signatures correct; import/runtime independence retained | Yes / none |
| Optional target backend | Any existing supported deployment fixture/toolchain supplied by maintainer | Separately document compiler acceptance/integer accuracy; never claim fake-quant reference is bit-exact | Backend-specific / optional |

Numerical tolerances must come from existing operator/grid rounding error, not be
increased to conceal changed qparams. Exact equality applies to persisted grids,
locks, integer metadata, interface order and rejection behavior. Use source snapshot
fingerprints and final artifact inspections together: each proves a different boundary.

## 16. Risks and open assumptions

**Confirmed facts:** E1–E4; repeated traversal; current subtype lock failure; min/max-only
state dicts cannot restore imported/Percentile grids; whole-module pickle is the recipe
format; Gemma shared-KV requires explicit binding; current backend marker semantics
differ from supplied HEAD. These are reproduced or directly inspected, not hypotheses.

**Behavioral risks:** strict validation will expose intentionally dormant and conditionally
unused observers. Active declarations and conditional capability enforcement must land
before broad rejection. Parameter collection hooks must retain current timing. Reset
must explicitly clear imported locks for a fresh calibration cycle as it does today;
fake-quant toggles must remain independent of collection and not erase snapshots.
Persist their numerical bypass policy. Enabling a previously bypassed site without
a valid snapshot must reject before execution rather than manufacture readiness.

**Export risks:** a snapshot layer must not override graph-domain lowering or infer ABI
from FX float types. Existing candidate-2 domain propagation can complicate parity;
record it separately rather than adjusting lifecycle grids to compensate. Generated
profile constants need their own finalize owner. No manifest format revision is
required just to replace private reads with equivalent snapshot data.

**Migration risks:** old generic state dicts lack sufficient provenance, so exact frozen
restore cannot be guaranteed. Old whole-module pickle classes may lack new attributes;
handle them at explicit load boundaries. New schemas must handle dynamic buffer shape,
device mapping, strict/non-strict policy mismatch and source bindings. Do not silently
reinterpret obsolete past-cache observers. Out-of-tree private overrides need migration.

**Performance risks:** clone small qparams, not weights/calibration reservoirs; avoid
duplicating candidate tensors per alias. Once-only traversal removes redundant work.
Do not run Python tensor-valued validation on every traced forward; validate/commit
before export, with cheap ready-state guards and capability checks at entry boundaries.
Freeze planning must not build full model copies or require GPU synchronization per
element beyond existing qparam validation needs.

**Unsupported schemes:** do not invent affine state for MX, extend Gemma cache schemes,
or claim new groupwise/GPTQ support. Validators should be extensible by observer
capability, but reject policies whose shape/storage contract is not known.

**Implementation-time confirmations still needed:**

1. Exhaustive site/capability inventory for every wrapper variant, including shared
   module aliases, fusion choices and optional multimodal branches; tiny tests covered
   representative paths, not every registered wrapper.
2. Exact compatibility surface for externally serialized whole-module objects and
   out-of-tree observer subclasses. No real user checkpoint archive was available.
3. Final schema placement/names and the smallest safe version-2 Gemma legacy adapter.
   Prove missing state cannot be accepted by `strict=False`.
4. Torch export interaction with new ready guards/buffer state on all supported Torch
   families; only default 2.13 CPU was exercised here.
5. Conditional capability grouping must preserve cache-disabled workflows without
   treating missing required calibration as inactivity. Do not leave this implicit.
6. Commit/install hook failure handling and post-load cross-source validation must be
   covered by real failure tests; the report proposes an architecture, not proof that
   an unimplemented transaction is infallible.

Large-model quality, memory/performance benchmarks, integer NPU execution and all
supported Torch versions were not run. They are not needed to reproduce this issue,
but downstream performance/backend claims require their own evidence.

## 17. Relevant files, symbols, tests, commits and experiments

### Source and test index

| Area | Files / symbols to start from |
|---|---|
| Public lifecycle | `tico/quantization/public_interface.py:convert`; `wrapq/quantizer.py:PTQQuantizer.prepare/convert` |
| Local lifecycle | `wrapq/wrappers/quant_module_base.py:enable_calibration,freeze_qparams,_child_quant_modules,_all_observers`; `ptq_wrapper.py` |
| State algorithms | `wrapq/observers/{base,affine_base,minmax,ema,percentile,identity,mx}.py`; affine `load_qparams,has_qparams,compute_qparams,fake_quant`; Percentile `_clipping_bounds,compute_qparams` |
| Handoff/control | `recipes/qparams.py:inject_gptq_qparams`; `recipes/stages/ptq.py`; `wrapq/control.py:iter_quantization_sites` |
| Families | `wrapq/wrappers/llama/quant_attention.py`; `qwen_vl/quant_text_attention.py:require_calibrated_cache,_all_observers`; `gemma4/quant_text_attention.py:freeze_qparams,_validate_frozen_observer,get_extra_state,set_extra_state,cache_storage_observers` |
| Export consumers | `recipes/export/{checkpoint,gemma4,gemma4_assistant,llama,llama_bundle,qwen3_vl}.py`; `gemma4/quant_vision_pooler.py:as_prefill_export_adapter`; debug CPU/export-regression and wrapper-smoke cases |
| Artifact path | `tico/passes/decompose_fake_quantize*.py`; `tico/quantization/passes/{fold_quant_ops,remove_weight_dequant_op,propagate_qparam_forward,propagate_qparam_backward,quantize_bias,insert_quantize_on_dtype_mismatch}.py`; `tico/utils/convert.py`; `tico/serialize/circle_graph.py:add_tensor_from_node` |
| Existing contract tests | `test/quantization/wrapq/observers/test_{affine_base,percentile,minmax,base,identity,mx}.py`; `wrappers/test_quant_module_base.py`; family attention and Gemma `test_past_kv_observers.py,test_text_model_cache_sources.py`; `test/quantization/recipes/test_{qparams,ptq_qparam_handoff,folded_linear_qparam_handoff,llama_attention_qparam_handoff}.py`; `test/unit_test/quantization/test_circle_executor.py` |

Paths in the source index abbreviated after `tico/quantization/` or
`test/quantization/wrapq/` as appropriate. Family semantic tests are to retain/extend;
uncalibrated mode-cycle assertions are to replace. Mocked export tests can validate
routing but do not replace artifact-level requirements.

Relevant history: `d122de81` original recursive lifecycle; `a33008d2` quantizer loop;
`5d1e089` local/recursive ownership (#733); `8da8a14` locking (#866); `82e2ef5`
per-channel import canonicalization (#877); `42b796b` EMA copy (#1038); `e1caa21`
Gemma past reads (#1079); `c0a3cbc` Quantize propagation retarget (#1080);
`8974e59` static smoke export contracts (#1084); `b704b57` Llama S/C (#1083);
`5620abe` Qwen S/C (#1087); `9343af8` Gemma S/C (#1086); `b002982` NPU domain
folding (#1088); `126d708` exact saturation (#1090).

### Executed commands and outcomes

Actual investigation paths were the supplied repository `/workspace/TICO`, detached
worktree `/tmp/tico-design-main`, and interpreter `/workspace/TICO/.venv/bin/python`.
For reproducibility below, `REPO` denotes the supplied checkout and `MAIN_TREE` the
detached worktree; these variables are command-local conveniences, not project config.

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
git ls-remote origin refs/heads/main
git ls-remote https://github.com/Samsung/TICO.git refs/heads/main
git fetch https://github.com/Samsung/TICO.git main
git log --oneline HEAD..FETCH_HEAD
git merge-base HEAD FETCH_HEAD
git worktree add --detach /tmp/tico-design-main 126d7087a13af2ba7fa39d9e1493f813bfd39430
```

Repository discovery used `rg --files` for AGENTS/README/contribution/design files,
`rg -n` for TODO/FIXME/workaround/deprecated, lifecycle overrides, qparam writes,
serialization and tests. Source/history review used `git log -25 --oneline --` on
observer/base paths, `git show --stat` on the commits above, and
`git blame -L 105,145 .../quant_module_base.py` plus `git blame -L 65,101 .../quantizer.py`.
These showed the two traversal owners survived #733.

```bash
REPO=/workspace/TICO
MAIN_TREE=/tmp/tico-design-main
"$REPO/.venv/bin/python" --version
"$REPO/.venv/bin/python" -m pip check
cd "$MAIN_TREE"
PYTHONPATH="$MAIN_TREE" "$REPO/.venv/bin/python" -m unittest \
  test.quantization.wrapq.observers.test_affine_base \
  test.quantization.wrapq.observers.test_percentile \
  test.quantization.wrapq.wrappers.test_quant_module_base \
  test.quantization.recipes.test_qparams \
  test.quantization.recipes.test_ptq_qparam_handoff -q
```

Result: **65 tests passed**. Additional independently executed groups: 41 observer/base
tests passed; 105 affine/percentile plus Llama/Qwen/Gemma attention tests passed; 35
affine/MinMax/base/Identity/MX tests passed. Groups overlap; these are not additive
coverage counts. One existing MX list-indexing deprecation warning appeared. A package
metadata query including pytest failed because pytest is absent; a corrected query
returned the versions in §2. No dependency installation was necessary.

Temporary CPU probes executed outside the repository:

```bash
PYTHONPATH=/tmp/tico-design-main /workspace/TICO/.venv/bin/python /tmp/tico_wrapper_lifecycle_probe.py
PYTHONPATH=/tmp/tico-design-main /workspace/TICO/.venv/bin/python /tmp/tico_freeze_probe.py
PYTHONPATH=/tmp/tico-design-main /workspace/TICO/.venv/bin/python /tmp/tico_concat_probe.py
```

They produced E1–E3 and candidate-2 evidence. Nothing generated was placed in the
repository. To avoid depending on those temporary files, the central probes can be
recreated with the following self-contained script against the investigated commit:

```python
import copy
import torch
import tico
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.quantization import prepare, convert
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.quantizer import PTQQuantizer
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

# Missing calibration becomes serialized ABI, default O1 enabled.
m = torch.nn.Linear(2, 2, bias=False).eval()
with torch.no_grad():
    m.weight.copy_(torch.tensor([[.25, .5], [-.25, 1.]]))
q = convert(prepare(m, PTQConfig())).eval()
cm = tico.convert(q, (torch.tensor([[.125, .375]]),))
doc = CircleDocument.from_bytes(cm.circle_binary)
print(doc.verify())
runtime = CircleReferenceRuntime(doc)
for t in runtime.program.tensors:
    print(t.index, t.name, t.dtype, t.quantization)

# Imported grids disappear in generic state dict; Percentile ignores lock first.
o = MinMaxObserver(name="imported")
o.load_qparams(torch.tensor(.3), torch.tensor(7), lock=True)
r = MinMaxObserver(name="imported")
r.load_state_dict(copy.deepcopy(o.state_dict()))
print(list(o.state_dict()), r.has_qparams, r._qparams_locked)
print(o.compute_qparams(), r.compute_qparams())
p = PercentileObserver(name="imported")
p.load_qparams(torch.tensor(.3), torch.tensor(7), lock=True)
try:
    p.compute_qparams()
except RuntimeError as e:
    print(type(e).__name__, str(e))

class Counter(QuantModuleBase):
    def __init__(self, child=None):
        super().__init__()
        self.obs = MinMaxObserver(name="x")
        self.calls = 0
        if child is not None:
            self.child = child
    def _all_observers(self):
        return (self.obs,)
    def forward(self, x):
        return self._fq(x, self.obs)
    def freeze_qparams(self):
        self.calls += 1
        super().freeze_qparams()

leaf = Counter()
mid = Counter(leaf)
root = Counter(mid)
PTQQuantizer(PTQConfig()).convert(root)
print(root.calls, mid.calls, leaf.calls)  # 2, 3, 4 on investigated main

class FailObserver(MinMaxObserver):
    def compute_qparams(self):
        raise RuntimeError("intentional invalid boundary")

class Pair(Counter):
    def __init__(self):
        super().__init__()
        self.bad = FailObserver(name="bad")
    def _all_observers(self):
        return (self.obs, self.bad)

pair = Pair()
pair.enable_calibration()
pair.obs.collect(torch.tensor([1.]))
try:
    pair.freeze_qparams()
except RuntimeError:
    print(pair._mode, pair.obs.enabled, pair.obs.has_qparams,
          pair.bad.enabled, pair.bad.has_qparams)
    # QUANT, False, True, False, False: partial commit
```

Additional here-doc probes used only CPU tensors: MinMax/Percentile collect
`[-1,1,2]`, import `(.125,7)`, then state-dict restore/recompute; whole-module BytesIO
roundtrip of imported MinMax/Percentile; tiny Llama prefill S=3/decode S=1 with
sequence-axis per-channel storage overrides. These corroborate E2 and candidate 4.
No large-model accuracy result is claimed. Final report checks: Markdown parsed with `markdown_it` (CommonMark plus tables);
all 18 required sections present, Implementation Contract last, no trailing whitespace;
the embedded Python evidence block was extracted to `/tmp/tico_report_evidence.py`
and executed against latest main, reproducing E1–E3 and the Percentile lock error.
`git diff --check`, the untracked-file whitespace check
`git diff --no-index --check /dev/null <report>`, and final changed-file/branch
checks passed. The temporary detached worktree was removed after validation;
the original checkout remains on its original branch and commit. Repository lintrunner
configuration matches only Python/Python stub files, so no production format/static
check was needed for this Markdown-only diff. No production refactor, whole non-model
suite, or backend validation was performed for this documentation-only change.

## 18. Implementation Contract

Use this section as the primary input to the implementation task; §§11–15 specify
the mechanics and concrete slices.

**Must remain unchanged**

1. Public prepare/convert signatures, CLI/YAML keys, supported model families and
   quantization schemes, valid calibrated math and GPTQ handoff ordering/names.
2. Exact loaded compatible locked grids; collection of unrelated activation stats;
   explicit reset starts a fresh calibration cycle and clears old locks.
3. Current final Circle ABI and NPU domain-marker semantics for equivalent valid
   frozen state, including default O1 and explicit real domain transitions.
4. Prefill/decode S/C cache ABI, per-layer chaining, no-cache inference, supported
   StaticCache/DynamicCache mutation, and Gemma source-binding/unsupported-policy
   restrictions. Do not unify distinct model semantics.

**Must change**

1. Required unobserved/invalid affine sites must reject before QUANT/artifact creation.
   Zero-range finite calibration remains valid. Identity/MX readiness is explicit.
2. Imported lock precedence applies to every affine subtype, including Percentile.
3. Lifecycle traversal touches each owned active site once, through arbitrary module
   containers and aliases. Failure leaves modes, flags, grids and retry handles intact.
4. Frozen state saves/restores exact qparams/policy/lock/provenance/readiness; no
   silent recomputation from incomplete state. Reject incompatible/corrupt frozen
   state even under non-strict loads. Define explicit legacy migration/errors.
5. Optional unready capabilities reject before use; they do not invalidate otherwise
   supported calibrated no-cache execution or masquerade as calibrated grids.

**Selected ownership**

Observer common infrastructure owns validated frozen snapshots and persistence;
subtypes own only statistics derivation or stateless policy. One WrapQ coordinator
owns traversal, validation plan and commit. Wrappers own active/conditional sites,
parameter collection, cache/source semantics. Exporters consume snapshots and own
only newly generated export-local constant grids. Existing graph lowering and Circle
serialization own final domains and ABI, not model calibration readiness.

**Private APIs that may change**

`_all_observers`, child traversal helpers, lifecycle overrides/hooks, mutable cached
fields, compute/import internals, model-specific generic validation/persistence,
and fixture finalization APIs. Preserve stable public behavior and diagnostic naming;
do not retain duplicate unsafe paths merely to preserve private structure.

**Old paths to remove**

Overlapping recursive/global lifecycle loops; Gemma's independent generic freeze and
grid-copy writer; subtype computation before common lock selection; nonempty-scale
as the sole readiness predicate; lazy source-state recomputation/reset in execution
or export; tests that bless empty-calibration freeze. Keep semantic source binding
and obsolete-cache rejection. Any legacy reader must be bounded and prove exactness.

**Required tests and artifact inspection**

Implement the CPU matrix in §15, especially missing-versus-zero stats, subtype locks,
once-only arbitrary-container traversal, public failure/retry in both in-place modes,
exact state roundtrip, shared-source restores, structural inactivity and conditional
capability checks. Export tiny calibrated Linear and Llama/Qwen/Gemma stages before
and after restore. Deserialize real default-O1 Circle bytes; inspect boundary
dtype/shape/scale/zp/axis and Quantize topology; verify structure and separately run
reference fake-quant numerical/staged parity. E1 must fail before producing an artifact.
Run focused tests, full affected groups, the non-model suite and format/static checks.

**Success criteria**

Every admitted active site has one validated stable owner; failed freezes have no
partial effect; locked grids survive calibration/reload unchanged; valid outputs and
serialized contracts are preserved; required missing statistics cannot produce a
valid-looking exported model; a new wrapper needs no custom generic freeze or grid
serialization. All migrated consumers use the common protocol and all semantic family
constraints retain explicit tests.

**Prohibited shortcuts**

Do not suppress missing stats by using a floor/default grid, silently skip every
unobserved site, collect after conversion to repair readiness, infer frozen state
from min/max alone, deepcopy whole models to simulate a transaction, add model-name
branches to generic infrastructure, restore stale float-bridge semantics, hide errors
with fallbacks/skips/tolerances, or substitute FX assertions/mock exports for Circle
inspection. Do not claim integer backend proof from fake-quant reference execution.

**Known open assumptions**

Finalize the complete active/capability inventory, schema/legacy adapter details,
out-of-tree override migration, supported-Torch export behavior and transaction
failure mechanics during implementation. No external checkpoint archive or NPU
toolchain was tested. These uncertainties do not invalidate the demonstrated issue;
they are explicit gates for the corresponding slices, not permission for silent
compatibility guesses.
