# Fixed-Array Bounds Soundness

Status: IN PROGRESS — CONTRACT APPROVED; TASKS 1–2 CLOSED. This is an S1
compiler-correctness project, not a loop-proof capability extension.
Fixed-array loop artifact authority remains blocked until this contract is
implemented and independently reviewed.

## Objective

Make every fixed-array read and write obey the language rule:

```text
index < element_count
```

A statically known out-of-bounds access is rejected during compilation. A
runtime-dependent out-of-bounds access reverts in every build mode, including
`--no-verify`. Formal verification proves the same bounds predicate from the
same canonical compiler data; it does not invent a verifier-only restriction.

This project fixes checked indexing. It does not arm fixed-array loop artifact
authority.

## Current Defect

The shipping compiler violates the documented checked-indexing contract in
three connected places:

- `values[7]` on `[T; 4]` compiles without a diagnostic;
- a runtime fixed-array index lowers to address arithmetic and a load or store
  without a bounds assertion, allowing an out-of-range storage write to modify
  an adjacent slot; and
- the canonical loop timeline records `.fixedIndex` with an index expression
  and type but no array extent, while Lean checks only unsignedness.

The existing `fixed_storage_loop_auto.ora` probe demonstrates the formal gap:
the timeline contains `.fixedIndex i`, but no bounds safety checkpoint. The
emitted SIR computes the element slot and performs `sir.sstore` directly.

This is a language and compiler soundness defect. It is not acceptable to
repair only the verifier or only verified builds.

## Verification Boundary and Claim

For every fixed-array access that reaches executable canonical Ora MLIR:

1. the compiler knows the exact positive element count from the resolved array
   type;
2. the access index has the canonical unsigned index representation;
3. a runtime safety assertion for `index < element_count` dominates the access;
4. an out-of-bounds execution reaches the ordinary assertion-revert path
   before any load or store;
5. the assertion remains executable in `--no-verify` builds; and
6. formal projection records the exact same index, extent, and safety condition
   in canonical LoopVC data.

The formal claim is made about canonical Ora MLIR. Source-to-canonical-MLIR
lowering and later lowering remain trusted compiler components, but their
agreement is tested structurally and through executable EVM conformance.

Bounds safety is separate from transition existence. An out-of-bounds access
must not be modeled as an absent storage transition, because that would make a
universal transition obligation vacuous. The proof obligation is:

```text
reachable_prefix -> index < element_count
```

Only after that condition holds may the load or store transition execute.

## Semantic Model

The Lean model lands before compiler changes. It extends the shared checked-
operation semantics rather than adding a loop-specific theorem.

The Lean-first model must represent, without depending on emitted compiler
rows:

- an extent-bearing fixed-index semantic input;
- the canonical unsigned index value;
- `FixedIndexSafe index extent`, denoting `index.toNat < extent`;
- a safety checkpoint distinct from the following load or store transition;
  and
- a fixed-index location whose extent is part of the checked canonical data.

Required generic facts include:

- index zero is safe exactly when the extent is positive;
- `extent - 1` is safe for every positive extent;
- `extent` and larger indices are unsafe;
- safety is identical for reads and writes;
- read and write accesses share the same bounds condition;
- an unsafe index cannot satisfy the safety obligation; and
- later timeline facts cannot discharge an earlier bounds checkpoint.

No second timeline executor, storage model, or loop-induction theorem may be
introduced. Existing `TimelineSafe`, `TimelineExec`, and
`verified_of_induction` remain the only corresponding semantic structures.

## Compiler Rules

### Constant indices

Sema rejects a fixed-array index when its compile-time value is greater than
or equal to the resolved element count. The diagnostic is source-anchored and
names the index and array extent. This applies to reads and writes.

Failure to evaluate an otherwise valid runtime expression is not treated as
zero and is not accepted as proof of safety. Such expressions continue to the
runtime-check path.

### Runtime indices

All fixed-array read and write lowering paths call one shared bounds-check
operation. The check compares the canonical unsigned index with the exact
element count and emits the ordinary assertion operation immediately before
the access.

The assertion is language behavior:

- it exists under `--no-verify`;
- Z3 may prove it, but proof does not by itself authorize its removal;
- a Lean loop certificate grants no runtime-check-erasure authority; and
- wrapping address arithmetic is never used as a substitute for a bounds
  proof.

No load/store lowering path may bypass the shared check. Any representation
for which the element count cannot be recovered fails compilation with a named
reason.

## Canonical Formal Data

The canonical fixed-index projection carries the exact array extent in
addition to the existing index formula and type. The extent comes from the
resolved shaped type, not from source text, a declaration string, or a default.

The semantic projector and independent inventory collector both obtain and
record the extent. Their rows must agree exactly on:

- storage root;
- projection position and kind;
- index source identity;
- index type;
- element type; and
- element count.

Missing, zero, inconsistent, reordered, or fabricated extent information
fails closed. The new field participates in deterministic LoopVC hashing and
Lean/Zig formal lockstep.

The bounds checkpoint is projected from the real compiler assertion. Z3 may
not synthesize `index < extent` independently when the canonical assertion is
absent. Lean and Z3 consume the same prefix-staged condition.

## Task Structure

### Task 1 — Lean checked-index semantics

- [x] Add an extent-bearing fixed-index semantic input without changing the
  emitted row schema in this task.
- [x] Define the shared bounds-safety predicate.
- [x] Keep safety separate from transition existence.
- [x] Prove the generic boundary and read/write-equivalence facts.
- [x] Pin prefix staging with an adversarial later-contradiction example.
- [x] Build and axiom-audit the Lean package.

Task 1 is Lean-only and introduces no compiler or artifact authority. It lands
as a dedicated commit after review.

### Task 2 — Constant out-of-bounds rejection

- [x] Reject constant out-of-bounds reads in sema.
- [x] Reject constant out-of-bounds writes in sema.
- [x] Accept indices `0` and `extent - 1`.
- [x] Preserve runtime-dependent indices for Task 3 checking.
- [x] Emit one source-anchored diagnostic per invalid access.

Task 2 changes compilation behavior and lands separately after focused
positive and negative corpus tests.

### Task 3 — Runtime bounds enforcement

- [ ] Route every fixed-array read and write through one shared check emitter.
- [ ] Emit `index < extent` before the access.
- [ ] Keep the assertion in `--no-verify` output.
- [ ] Prove through EVM execution that an out-of-bounds storage write reverts
  and leaves the adjacent slot unchanged.
- [ ] Pin in-bounds reads and writes in both verified and unverified builds.

Task 3 is not complete from MLIR text alone; executable conformance evidence is
mandatory.

### Task 4 — Canonical LoopVC integration

- [ ] Add the exact extent to fixed-index schema, emission, and hashing.
- [ ] Extend the semantic projector and independent inventory in lockstep.
- [ ] Project the real bounds assertion as a prefix-staged safety checkpoint.
- [ ] Encode the checkpoint identically in canonical Z3 and Lean.
- [ ] Prove that a safe projected index denotes the same element location used
  by the storage action.
- [ ] Reject missing, malformed, or inconsistent extent/checkpoint data by
  name.
- [ ] Extend formal-sync snapshots and executable lockstep mutations.

Task 4 does not remove `storage_projection_authority_unreviewed`.

### Task 5 — Corpus and deterministic regeneration

- [ ] Add unrelated fixed-array contracts with different roots, element
  counts, and read/write expressions.
- [ ] Pin constant and runtime indices at `0`, `extent - 1`, `extent`, and a
  much larger value.
- [ ] Pin local and persistent-storage fixed arrays where supported.
- [ ] Regenerate affected Ora/SIR checks only after reviewing the semantic
  difference.
- [ ] Rebaseline bytecode-size and gas data with measured explanations.
- [ ] Confirm deterministic LoopVC and formal artifacts across two runs.

Generated differences are part of this fix and may not be hidden by weakening
checks or excluding fixtures.

### Task 6 — Closure audit

- [ ] Focused sema, lowering, Lean, Z3, and conformance tests pass.
- [ ] Constant-OOB negative corpus checks pass with the named diagnostic.
- [ ] The no-verify EVM corruption probe now reverts and preserves its
  sentinel slot.
- [ ] Verifier mutation tests reject a removed or altered bounds checkpoint.
- [ ] `check-formal-sync` passes.
- [ ] The full repository gate passes on the exact closure tree.
- [ ] An independent reviewer confirms the source, MLIR, LoopVC, Z3, Lean,
  SIR, and EVM chain for one runtime-indexed access.

Only after Task 6 closes may a separate project propose fixed-array loop
artifact authority.

## Required Tests

### Compile-time cases

- valid constant read and write at index `0`;
- valid constant read and write at `extent - 1`;
- invalid constant read and write at `extent`;
- invalid constant read and write above `extent`;
- nested/grouped constant expressions are evaluated consistently; and
- a runtime-dependent index is not misclassified as a constant.

### Runtime cases

- in-bounds dynamic read returns the selected element;
- in-bounds dynamic write changes only the selected element;
- out-of-bounds dynamic read reverts;
- out-of-bounds dynamic write reverts before modifying storage;
- an adjacent persistent storage root retains its value after the rejected
  write; and
- verified and `--no-verify` builds agree on all cases.

### Formal and adversarial cases

- the existing safe `i < extent` loop discharges the bounds checkpoint;
- an invariant insufficient to establish bounds fails closed;
- removing the assertion prevents fixed-array formal support;
- changing the projected extent, index identity, type, or projection tag
  fails the dual-inventory comparison;
- a later false path cannot prove an earlier bounds checkpoint;
- an unsigned/signed mismatch fails rather than coercing silently; and
- fixed-array artifact authority remains blocked throughout this project.

## Gatekeeper Controls

The project is not complete unless the test suite makes these regressions
observable:

- any fixed-array access lowered without a dominating shared bounds assertion;
- any constant out-of-bounds access reaching MLIR;
- any LoopVC fixed-index projection without an exact positive extent;
- any verifier-only bounds premise without a matching compiler assertion;
- any mismatch between semantic projection and independent inventory; and
- any proof result that erases the runtime check without separately reviewed
  optimization authority.

Negative tests assert the named failure or revert behavior. A generic nonzero
compiler exit is insufficient evidence.

## Growth Envelope

Production code means tracked files under `formal/Ora` and `src`; tests,
fixtures, generated check files, and documentation are reported separately.

- Task 1 Lean semantics: expected `+80` to `+160` net lines.
- Task 2 sema rejection: expected `+50` to `+120` net lines.
- Task 3 runtime enforcement: expected `+70` to `+150` net lines.
- Task 4 formal integration: expected `+140` to `+260` net lines.
- Tasks 5 and 6 repairs: expected `+0` to `+80` net production lines.
- Expected total: `+340` to `+770` net production lines.
- Hard architecture-review stop: `+900` net production lines.

Crossing the stop requires net-negative consolidation before further
production growth. The limit is not raised inside this project. If one shared
runtime check cannot cover both reads and writes, or if Lean and Z3 require
different extent/bounds representations, implementation stops and returns to
design review.

## Commit and Review Discipline

Each task is a separate review unit:

```text
proposal -> discussion -> implementation -> focused validation -> approval
         -> commit
```

No task begins before its predecessor is approved and committed. Task 1 is
Lean-only. The full repository gate runs once on the exact closure tree, while
each task runs its focused tests; any shared obligation-schema or Env change
also runs the complete compiler test suite and formal synchronization before
approval.

## Non-Goals

This project does not:

- arm fixed-array loop artifact authority;
- prove fixed-array storage preservation or postconditions automatically;
- add dynamic arrays or change slice semantics;
- authorize bounds-check erasure;
- prove physical storage-layout preservation;
- add loop termination; or
- change mapping-key or nested-child-loop authority.

## Completion Condition

The project closes only when constant out-of-bounds accesses are rejected,
runtime out-of-bounds accesses revert before memory or storage mutation in all
build modes, canonical LoopVC contains the exact extent and prefix-staged
bounds checkpoint, both provers consume that same data, the regression and
mutation battery is green, and the exact-tree full gate and independent
chain-of-custody review pass.
