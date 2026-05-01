# SMT Implementation Team Plan

## Goal

This document is the implementation-facing plan for bringing Ora SMT verification to a "100%" state.

In this context, "100%" does **not** mean "every construct is encoded exactly no matter the cost."
It means:

- no known silent unsoundness
- no known fail-open behavior
- all remaining non-exact areas are explicit, tested, and intentionally fail-closed
- real-contract `UNKNOWN` hotspots are either eliminated or reduced to clearly isolated solver limits
- sequential verification is the semantic reference
- parallel verification is allowed only when it is provably equivalent to sequential on the same prepared queries

This plan is meant for multiple implementers working in parallel. Each workstream below includes:

- why it matters
- current status
- anchor repros
- likely implementation files
- done criteria
- anti-goals

## Current Branch Status

The current branch is materially ahead of the older SMT audit. The following major areas are already closed or mostly closed:

- transient storage is modeled as tracked SMT state
- `UNKNOWN` fails closed
- mid-proof degradation now fails closed
- major structured `ora.try_stmt` families are covered
- major canonical `scf.for` and `scf.while` families are covered
- guard clauses exist as a language feature with:
  - runtime enforcement
  - proof integration
  - proven-guard elimination
- scoped and structured struct metadata recovery is much broader than before
- sequential vs parallel equivalence has targeted probe coverage
- loop invariant post checking is not dead code; it is now a coverage/harness item

Recent commits that materially changed the SMT baseline:

- `133d1abf` `Fail closed degraded verifier queries`
- `3212e58f` `Fail closed swapped-compare decrement while results`
- `a4ec66ed` `Propagate path assumptions into guard proofs`
- `90ab9f21` `Propagate scoped struct metadata through SMT summaries`
- `65c5263b` `Encode affine carried scf.while results exactly`
- `ec69324b` `Carry fallthrough path assumptions into SMT obligations`
- `9c747454` `Guard helper summary obligations by branch path`

## Definition Of Done

Ora SMT is "done" when all of the following are true:

1. There is no known silent unsoundness on the current branch.
2. There is no known fail-open verifier path.
3. Remaining degradation paths are explicit, tested, and intentionally conservative.
4. Real contract probes do not have unexplained `UNKNOWN` hotspots in known tractable patterns.
5. Sequential and parallel verification are equivalent on prepared queries across the maintained probe set.
6. Audit and design docs match branch reality.

## Workstream 1: Eliminate Live `UNKNOWN` Hotspots

### Why This Matters

The remaining top-priority issue is not broad soundness anymore. It is verifier incompleteness in concrete real-contract cases. These `UNKNOWN`s block trust because the verifier cannot classify them as either:

- real bugs (`SAT`)
- or proven-safe (`UNSAT`)

The most important current example is the `openStream` family.

### Current Status

This workstream is closed on the current branch.

The narrow repro:

- [open_stream_add_unknown.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/open_stream_add_unknown.ora)

and the corresponding real contract:

- [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora)

no longer have the earlier unexplained `UNKNOWN` hotspot.

### Anchor Repros

- [open_stream_add_unknown.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/open_stream_add_unknown.ora)
- [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora)

### Likely Files

- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)
- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [expr_lowering.zig](/Users/logic/Ora/Ora/src/hir/expr_lowering.zig)

### Implementation Focus

- normalize `@XXWithOverflow` result-branch assumptions before they reach unrelated later obligations
- flatten or replace accumulated fallthrough ladders when the proof only needs the local branch fact
- preserve exact semantics; only query shape may change

### Done Means

- the narrow repro no longer returns `UNKNOWN`
- the corresponding `erc20_stream_core` obligations move to `UNSAT` or real `SAT`
- no regression in arithmetic or summary correctness tests

### Anti-Goals

- do not patch contracts just to hide SMT incompleteness
- do not add heuristic rewrites that are not bitvector-equivalent
- do not accept slower global query handling if it only shifts the timeout elsewhere

## Workstream 2: Finish Non-Canonical Loop Exactness

### Why This Matters

Canonical loops are in much better shape. The remaining loop frontier is now a small fail-closed boundary, not a general open-ended promise of loop exactness:

- non-canonical `scf.while`
- only concrete `scf.for` result/state cases that come with a real degradation repro

This work determines whether remaining loop degradations are:

- genuinely unavoidable for now
- or just missing exact recognizers

### Current Status

Improved already:

- canonical counting `scf.for`
- canonical counting `scf.while`
- bounded finite cases
- no-write loop state preservation
- affine carried `scf.while` result exactness

Still live:

- `scf.while result requires loop summary`
- general loop state summary non-exactness, only when backed by a concrete repro

On the current branch, the explicit residual anchors are:

- malformed/unsummarizable `scf.while` result fallback with no condition/yield semantics to recover
- unsigned swapped-compare decrement `scf.while`
- signed swapped-compare decrement `scf.while`

These are intentionally fail-closed and now regression-tested with exact degradation reasons. The swapped-decrement forms are not a target for a local recognizer unless we first add a sound modular trip-count/ranking proof: the loop control variable moves away from the bound, so routing them through the canonical decrement closed form would be unsound.

`scf.for` has exact coverage for zero-iteration, identity, increment, decrement, positive-step, finite unroll, multi-result derived, geometric constant-multiplier carried updates, and known-callee summary families. Add more `scf.for` work only with a concrete source/MLIR repro.

### Anchor Repros

- direct encoder regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)
- soundness examples under [ora-example/smt/soundness](/Users/logic/Ora/Ora/ora-example/smt/soundness)

### Likely Files

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### Implementation Focus

- identify additional closed-form result families
- identify additional exact no-write or result-independent state families
- keep the remaining boundary explicit and fail-closed

### Done Means

- every remaining loop degradation reason is either:
  - covered by an exact recognizer
  - or deliberately preserved with a named regression
- the remaining loop matrix is small and stable

### Anti-Goals

- do not claim general loop exactness without a sound summary model
- do not reintroduce optimistic exactness for swapped/decrement/control-misaligned loops

## Workstream 3: Finish Interprocedural Summary Exactness

### Why This Matters

Known-callee summaries have improved a lot, but helper-heavy code still exposes residual exactness gaps. This is one of the main remaining sources of:

- opaque UF fallback
- lost state precision
- degraded result/state summaries

### Current Status

Closed or improved:

- direct path-assumption propagation into helper obligations
- many structured `try`, `switch`, `execute_region`, and deferred-return paths
- better write-set and state replay coverage
- compiler-lowered sema summaries now attach tracked `ora.read_slots` and `ora.write_slots` metadata to HIR functions
- the encoder treats explicit slot metadata as the authoritative tracked-state set for known-callee read/write recovery

Still intentionally fail-closed in the encoder:

- `failed to recover known callee write set exactly`
- `failed to recover known callee read set exactly`

On the current branch, the direct anchors that remain in-tree are intentional opaque-case boundaries:

- known callee with unknown write set delegating to an unresolved writer
- known callee with unknown read set delegating to an unresolved reader

The broad helper-summary bucket is no longer the right mental model. Compiler-generated slot summaries are exact for tracked storage/transient state; remaining failures are metadata-absent or genuinely unresolved callee boundaries.

### Anchor Repros

- helper-heavy regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)
- app probes:
  - [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora)
  - [defi_lending_pool.ora](/Users/logic/Ora/Ora/ora-example/apps/defi_lending_pool.ora)

### Likely Files

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)
- [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### Implementation Focus

- reduce exact-known-callee failures one degradation reason at a time
- improve exact write-set recovery
- improve exact returned-path replay when the callee is pure and structured

### Done Means

- common helper shapes no longer fall back to opaque summaries
- remaining UF fallbacks are rare, deliberate, and regression-tested

### Anti-Goals

- do not "fix" helper summaries by silently inlining unsafely
- do not weaken alias/write-set distinctions just to preserve exactness

## Workstream 4: Finish Struct And Frame Exactness

### Why This Matters

Struct/frame exactness directly affects:

- untouched-field preservation
- state/frame reasoning
- helper-return exactness

Large improvements have landed already, but the final residual cases are still important because frame precision is easy to overstate.

### Current Status

Improved already:

- source-metadata recovery
- scoped `ora.struct.decl` recovery
- structured-control result recovery
- call-path metadata recovery
- execute-region / switch / branch-return recovery

Remaining boundary:

- truly absent metadata
- any residual frame constraints that cannot be reconstructed from source or scope

### Anchor Repros

- struct tests in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### Likely Files

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)
- possibly MLIR/C API glue if more metadata must be preserved earlier

### Implementation Focus

- recover more metadata only when it is really available in scope or source
- otherwise make the fail-closed boundary explicit

### Done Means

- no silent partial-frame behavior remains
- metadata-miss cases are either:
  - newly exact
  - or explicitly degraded with tests

### Anti-Goals

- do not invent metadata
- do not assume field order or names without reliable source information

## Workstream 5: Finish Residual `ora.try_stmt` Exactness

### Why This Matters

`ora.try_stmt` exactness used to be a major risk area. It is now much better, but there is still a residual tail of nested/live cases. These should be closed if they are structurally recoverable and left fail-closed otherwise.

### Current Status

Already covered in large part:

- direct unwrap cases
- always-catching cases
- equivalent branches
- state-only merges
- `scf.if`
- `ora.conditional_return`
- `ora.switch`
- `ora.switch_expr`
- finite/single-iteration loop interactions
- nested self-caught and composed catch-predicate families
- nested escaping result summaries when both the try-region value and the catch reachability predicate are recoverable

Remaining gap:

- harder nested/live try flows not reducible to the current exact predicate families
- opaque/disconnected catch conditions where no exact predicate can be recovered without inventing semantics

### Anchor Repros

- `try` regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### Likely Files

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### Implementation Focus

- search for remaining degradations triggered by real structured control
- add exact handling only when catch reachability and yielded values are fully recoverable

### Done Means

- only intentionally hard live-try residuals remain
- all remaining degradation reasons are narrow and justified

### Anti-Goals

- do not broaden try exactness by ignoring escaping catches
- do not treat nested caught flows as harmless unless that is semantically proven

## Workstream 6: Coverage, Harness, And Reliability

### Why This Matters

Even when the core SMT logic is right, confidence can decay if coverage is narrow or drifted. This workstream ensures the branch remains trustworthy as it evolves.

### Current Status

Closed on the current branch:

- soundness examples are in the compiler harness
- complex app probes exist
- sequential/parallel equivalence has direct prepared-query coverage
- the maintained probe set is representative of the current branch
- refined aggregate probes now cover both struct fields and ADT payloads
- release artifact smoke passed on the maintained artifact corpus

Ongoing maintenance:

- keep docs aligned with branch reality when new SMT work lands
- prevent stale audit claims from reappearing as open blockers

### Anchor Repros

- [open_stream_add_unknown.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/open_stream_add_unknown.ora)
- [fail_loop_invariant_post.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/fail_loop_invariant_post.ora)
- [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora)
- [defi_lending_pool.ora](/Users/logic/Ora/Ora/ora-example/apps/defi_lending_pool.ora)
- [erc20_bitfield_comptime_generics.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_bitfield_comptime_generics.ora)

### Likely Files

- [compiler.test.zig](/Users/logic/Ora/Ora/src/compiler.test.zig)
- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)
- audit docs under [docs/compiler](/Users/logic/Ora/Ora/docs/compiler)

### Implementation Focus

- keep sequential as the semantic reference
- require parallel to match sequential on the same prepared queries
- keep soundness examples and app probes alive in the regular harness

### Done Means

- complex probes stay green or fail for known, classified reasons
- no stale audit statements contradict the current branch
- prepared-query sequential/parallel equivalence remains enforced

### Anti-Goals

- do not let parallel become "close enough"
- do not treat one-off local contract checks as sufficient coverage

## Recommended Team Split

## Responsibilities

### Core SMT Owners

The core SMT owners are responsible for correctness, soundness, and final trust decisions.

This group owns:

- exactness classification
- soundness boundaries
- fail-closed behavior
- semantic review of tractability rewrites
- final integration signoff for SMT-core changes
- sequential verification as the semantic reference
- approval of any parallel execution behavior that claims semantic equivalence

In practice, this means the core SMT owners review and approve any change that affects:

- path assumptions
- overflow encoding
- loop summaries
- `ora.try_stmt` summaries
- interprocedural summary semantics
- degradation boundaries
- verifier success/failure/unknown behavior

### SMT Execution Team

The SMT execution team is responsible for parallel execution within the correctness bar set by the core SMT owners.

This group owns:

- bounded repro reduction
- regression expansion
- harness coverage
- metadata preservation plumbing
- known-callee fallback cleanup
- tractability experiments
- probe maintenance
- documentation updates aligned with accepted changes

The SMT execution team should assume:

- throughput is delegated
- semantic authority is not

### Core SMT Owners Also Implement

The core SMT owners are not review-only.

They should directly implement changes when:

- a workstream is on the critical path
- the change is too semantics-sensitive to delegate safely
- a delegated implementation needs correction during review
- the execution team is blocked or moving too slowly on a correctness-critical item

### Working Rule

The intended operating model is:

- the SMT execution team moves work quickly inside a scoped bucket
- the core SMT owners make the final decision on whether a change is:
  - exact
  - bounded exact
  - degraded
  - or unacceptable

This is especially important for any change that could silently widen the trusted verifier surface.

### 1. Solver / Tractability Owner

Focus:

- `UNKNOWN` reduction
- `@XXWithOverflow` path-shape normalization
- real-contract timeout hot paths

Primary files:

- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)
- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [expr_lowering.zig](/Users/logic/Ora/Ora/src/hir/expr_lowering.zig)

### 2. Loop Exactness Owner

Focus:

- non-canonical `scf.while`
- concrete loop-result exactness repros, not stale fallback names

Primary files:

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig)

### 3. Interprocedural Summary Owner

Focus:

- known-callee residual fallbacks
- write-set recovery
- result/state replay exactness

Primary files:

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)

### 4. Struct / Frame Owner

Focus:

- absent-metadata boundaries
- frame completeness
- metadata preservation across lowering boundaries

Primary files:

- [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig)
- MLIR/C API glue if needed

### 5. Reliability / Harness Owner

Focus:

- app probes
- soundness corpus
- audit/doc alignment
- sequential/parallel equivalence

Primary files:

- [compiler.test.zig](/Users/logic/Ora/Ora/src/compiler.test.zig)
- [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig)
- [docs/compiler](/Users/logic/Ora/Ora/docs/compiler)

## Recommended Execution Order

### Phase 1

Eliminate live `UNKNOWN` hotspots.

Primary target:

- [open_stream_add_unknown.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/open_stream_add_unknown.ora)

### Phase 2

Finish residual loop-result exactness.

Primary target:

- remaining intentional `scf.while result requires loop summary` boundaries

### Phase 3

Finish interprocedural summary residuals.

Primary target:

- known-callee state/result exactness fallbacks

### Phase 4

Finish struct/frame residuals.

Primary target:

- truly absent metadata cases

### Phase 5

Final reliability pass.

Primary target:

- app probes
- soundness corpus
- sequential/parallel equivalence
- audit/doc consistency

Status: closed for the current branch on 2026-05-01.

Closeout evidence:

- app probes: `complex SMT app probes match between sequential and parallel verification`
- soundness corpus: `conditional_return_split`, `overflow_mul_constant`, `switch_arm_path_predicates`, and `fail_loop_invariant_post` stay in the regular compiler harness
- refined aggregate corpus: `refinement_struct_field_proof.ora` and `refinement_adt_payload_proof.ora` prove end-to-end and match sequential/parallel verification
- fail-closed corpus: `SMT degradation probes fail closed in sequential and parallel verification`
- release smoke: `./scripts/release_artifact_smoke.sh /tmp/ora-release-smoke-phase5`
- full validation: `zig build test-compiler -Dskip-mlir=true --summary failures` and `zig build test --summary failures`

## Active Workboard

As of 2026-05-01, Phase 5 is closed and the ADT phase work is not blocked on the remaining SMT exactness items. The only SMT work that should stay active is the residual, explicitly classified exactness work below. New SMT work should either attach to one of these rows with a concrete repro or open a new row with a degradation reason and exit criterion.

| Workstream | Owner Type | Current Status | Anchor Repro | Blocking Files | Exit Criteria |
| --- | --- | --- | --- | --- | --- |
| Live `UNKNOWN` elimination | Core SMT owners + SMT execution solver/tractability owner | Done | [open_stream_add_unknown.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/open_stream_add_unknown.ora), [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora) | [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig), [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [expr_lowering.zig](/Users/logic/Ora/Ora/src/hir/expr_lowering.zig) | No unexplained `UNKNOWN` in the narrow repro; corresponding real-contract obligations resolve to `UNSAT` or real `SAT`. |
| Non-canonical loop exactness | Core SMT owners + SMT execution loop owner | Residual active; currently limited to intentional `scf.while` fail-closed boundaries unless a concrete `scf.for` repro appears | direct loop regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | Remaining loop degradations are either exact or explicitly preserved with named regressions and rationale. |
| Interprocedural summary exactness | Core SMT owners + SMT execution interprocedural summaries owner | Closed for compiler-generated tracked-state summaries; residual only for metadata-absent or genuinely unresolved callees | helper-heavy regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig), [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora), [defi_lending_pool.ora](/Users/logic/Ora/Ora/ora-example/apps/defi_lending_pool.ora) | [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig), [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | Compiler-lowered read/write slot summaries stay exact; metadata-absent unresolved callees remain explicit fail-closed boundaries. |
| Struct/frame residuals | Core SMT owners + SMT execution struct/frame owner | Closed for known metadata-backed product values; metadata-miss cases intentionally fail closed | struct regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig), fail-closed guards in [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig) | [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig), [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig) | No silent partial-frame behavior; product metadata loss aborts verification for both field update and field extract. |
| Untyped SMT AST coercion fallback | Core SMT owners | Closed for current source repros; remaining direct helper use is defensive/testing-only or legacy untyped API surface | [map_of_arrays.ora](/Users/logic/Ora/Ora/ora-example/corpus/mixing/map_of_arrays.ora), tensor insert/extract regression in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | Source-typed paths use typed coercion; unsupported sort-pair coercion remains fail-closed instead of silently manufacturing mismatched ASTs. |
| Residual `ora.try_stmt` exactness | Core SMT owners + SMT execution interprocedural summaries owner | Residual active; narrowed to exact branch/result summaries | `try` regressions in [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | [encoder.zig](/Users/logic/Ora/Ora/src/z3/encoder.zig), [encoder.test.zig](/Users/logic/Ora/Ora/src/z3/encoder.test.zig) | Remaining live-try degradations are narrow, justified, and regression-tested. |
| Guard/context propagation | Core SMT owners + SMT execution solver/tractability owner | Done | guard regressions in [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig) | [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig) | Applicable path facts discharge guards consistently without cross-branch leakage. |
| Harness / reliability / equivalence | SMT execution reliability/harness owner with core SMT owner review | Done for current branch; maintenance only | [fail_loop_invariant_post.ora](/Users/logic/Ora/Ora/ora-example/smt/soundness/fail_loop_invariant_post.ora), [erc20_stream_core.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_stream_core.ora), [defi_lending_pool.ora](/Users/logic/Ora/Ora/ora-example/apps/defi_lending_pool.ora), [erc20_bitfield_comptime_generics.ora](/Users/logic/Ora/Ora/ora-example/apps/erc20_bitfield_comptime_generics.ora), [refinement_struct_field_proof.ora](/Users/logic/Ora/Ora/ora-example/corpus/types/refinement/refinement_struct_field_proof.ora), [refinement_adt_payload_proof.ora](/Users/logic/Ora/Ora/ora-example/corpus/types/refinement/refinement_adt_payload_proof.ora) | [compiler.test.match.zig](/Users/logic/Ora/Ora/src/compiler.test.match.zig), [compiler.test.verification.zig](/Users/logic/Ora/Ora/src/compiler.test.verification.zig), [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig), [docs/compiler](/Users/logic/Ora/Ora/docs/compiler) | Probe set remains live, parallel matches sequential on prepared queries, release smoke passes, and docs stay aligned with branch reality. |

## What We Should Not Do

- Do not widen exactness by weakening semantics.
- Do not patch contracts to hide verifier bugs.
- Do not treat stale audit items as live blockers.
- Do not let parallel semantics drift away from sequential semantics.
- Do not accept unclassified degradation reasons.

## Immediate Next Actions

1. Keep the remaining `scf.while` non-canonical degradations explicit, reason-checked, and small.
2. Only reopen interprocedural or `ora.try_stmt` work if a new non-opaque, structurally recoverable degradation appears.
3. Leave opaque/disconnected helper and catch-condition cases explicitly fail-closed unless there is a semantics-preserving recovery path.
4. Treat the ADT phases as closed unless a new regression contradicts the phase exit criteria.
5. Keep this document current when new SMT work changes the maintained probe set.
