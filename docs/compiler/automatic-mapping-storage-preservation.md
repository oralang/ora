# Automatic Mapping-Storage Preservation Proofs

Status: CLOSED WITHOUT AUTOMATIC AUTHORITY. Task 1 landed in `6514687d8`.
Task 2 triggered the project's stop condition and was reverted. Mapping
storage-preservation claims continue to require an external Lean certificate.

## Objective

Automatically construct the Lean portion of an already supported
mapping-storage loop certificate for this canonical claim:

```ora
requires write_key != observed_key

while condition
    invariant values[observed_key] == old(values[observed_key])
{
    values[write_key] = value;
    // existing supported scalar/control work
}
```

The compiler already projects and denotes this LoopVC, canonical Z3 already
checks it, and a user-authored external Lean certificate can authorize it.
This project replaces the repeated certificate proof with repository-owned
proof construction through the stable `by ora_loop_auto` interface.

This is proof construction, not new proof authority. Canonical Z3 and the Lean
kernel remain independently mandatory before bytecode emission.

## Verification Claim

For a LoopVC inside the boundary below, the generated proof establishes the
existing `InductionObligations`:

- the invariant holds in the loop's initial state;
- every reachable safety checkpoint holds at its exact timeline prefix;
- every backedge preserves both the scalar invariant and the declared
  mapping-location preservation invariant; and
- every exit satisfies its existing target assertion.

Mapping preservation is derived from the exact canonical timeline:

1. each read contributes no write footprint;
2. each mapping write resolves through the ordinary typed location evaluator;
3. every resolved write is covered by the declared singleton write frame;
4. the explicit function assumption proves that the observed and written
   mapping locations are distinct; and
5. the existing `TimelineExec.preserves_outside` theorem proves that the
   observed location retains its entry value.

The proof must use the same LoopVC terms, timeline events, variable identities,
types, and assumption row already checked by the certificate digest. It may not
reconstruct these facts from source syntax.

## Initial Supported Boundary

The first automatic boundary requires all of the following:

- one storage-preservation invariant of the form
  `root[observed_key] == old(root[observed_key])`;
- `observed_key` is a context-stable function parameter;
- every persistent write in the loop timeline is to
  `root[write_key]`, where `write_key` is a context-stable function parameter;
- observed and write keys have the same supported Ora value type;
- one canonical function assumption states `write_key != observed_key`;
- timeline reads and repeated writes are allowed when their existing ordering
  and identity checks succeed;
- all scalar state, control edges, safety checkpoints, initialization, and exit
  assertions are already inside the current `ora_loop_auto` boundary; and
- preheader execution, when present, is already supported by the closed
  preheader-trace project and introduces no new authority in this project.

The following remain unsupported by this automatic proof:

- missing, indirect, differently typed, or derived key-disequality evidence;
- a loop-carried or storage-read-result mapping key;
- fixed-array, field-plus-key, or multiple mapping projections;
- more than one observed preservation location or more than one write frame;
- a write to the observed location anywhere in the timeline;
- a storage-mutating child summary;
- a formula outside the existing canonical storage denotation; and
- any loop already excluded by the scalar/control or storage authority gates.

Unsupported cases retain the external-certificate path when their LoopVC is
denotable. Otherwise they fail closed under the existing named reason.

## Architecture Constraints

The implementation must be a transfer/combinator over the existing
`InductionObligations`. A refinement-style theorem is acceptable; a second
loop-induction rule is never acceptable.

The implementation must also satisfy these constraints:

- `verified_of_induction` remains the only loop-induction architecture;
- `TimelineExec` and the existing storage action semantics remain the only
  executor and store-transition model;
- generated modules retain the stable `by ora_loop_auto` proof body;
- the tactic consumes canonical LoopVC data rather than source shapes;
- no Zig source-shape recognizer or fixture-specific proof writer is added;
- no new LoopVC field, query kind, certificate schema, or digest exception is
  introduced;
- no compiler-generated Lean module is imported by committed Lean code;
- proof failure, unsupported shape, timeout, parse failure, or checker failure
  blocks automatic authorization; and
- this certificate grants no runtime-check-erasure or optimization authority.

The planner's assumption restriction is narrowed only for the reviewed shape:
exactly one assumption row containing one disequality between two
context-stable function-parameter identities, where those identities are
precisely the projection keys of the singleton write frame and observed
location. Any other assumption content, count, type, identity, or formula
shape retains `automatic_assumptions_require_external_certificate`. The tactic
may consume only that exact digest-bound assumption row; it may not infer,
combine, normalize, or discard other function assumptions.

If implementation requires a source-shape recognizer, positional event IDs,
hard-coded term IDs, fixture names, or a second generated proof script, work
stops and the design returns to review.

## Task Structure

### Task 1 — Lean frame bridge

- [x] Prove that reads contribute no write footprint.
- [x] Bind a write footprint to the location returned by the ordinary location
  evaluator.
- [x] Lift state-polymorphic per-event frame coverage over the whole timeline.
- [x] Compose complete frame coverage and location disjointness into storage
  preservation.
- [x] Prove that a context-stable key survives loop-local and timeline-read
  binding.
- [x] Prove typed one-key mapping resolution through the ordinary evaluator.
- [x] Pin repeated-write preservation and the later-observed-write
  counterexample in Lean.

Task 1 changes no compiler authority. It is complete only after reviewer
approval and a dedicated commit.

### Task 2 — Repository tactic wiring

Closed through the stop-control path on 2026-09-05. The attempted
implementation did not produce a kernel-accepted certificate and reached
`+816` uncommitted production lines, exceeding the entire project's `+400`
hard limit. The experiment was removed in full. The checklist remains
unchecked because automatic proof construction was not implemented.

- [ ] Teach `ora_loop_auto` to recognize the denoted preservation obligation
  from the goal and canonical LoopVC data.
- [ ] Establish context-variable freshness from canonical variable and read
  identities.
- [ ] Establish each event's frame using the Task 1 bridge.
- [ ] Bind key disjointness to the exact canonical assumption row.
- [ ] Narrow the Zig planner's assumption restriction only for the exact
  single-disequality shape defined above; every nonmatching assumption set
  retains `automatic_assumptions_require_external_certificate`.
- [ ] Compose storage preservation with the existing scalar/control
  `InductionObligations` proof.
- [ ] Leave all nonmatching goals open so Lean rejects the automatic proof.

Task 2 may not add a Zig recognizer or emit a fixture-shaped Lean proof.

The observed failure was fail-closed: the generated proof left a required
post-timeline invariant goal open, Lean introduced `sorryAx`, and the axiom
audit rejected the proof. No certificate or bytecode was emitted. Continuing
the same approach would duplicate the existing scalar/control proof planner
inside a second storage-specific tactic, contrary to the architecture and
growth constraints above.

### Task 3 — Production acceptance evidence

- [ ] Positive: the canonical distinct-key fixture automatically receives a
  kernel-checked certificate and emits bytecode.
- [ ] Missing inequality: removing the `requires` fact blocks automatic proof
  construction and emits no bytecode.
- [ ] Same key: writing the observed key refutes preservation and emits no
  bytecode.
- [ ] Later observed write: a disjoint write followed by a write to the
  observed key fails frame coverage and emits no bytecode.
- [ ] Additional assumption: adding an unrelated second `requires` clause on
  an untouched parameter retains
  `automatic_assumptions_require_external_certificate`; it is not silently
  consumed, dropped, or passed to the tactic.
- [ ] The external-certificate route remains accepted and digest-bound.
- [ ] Two consecutive successful runs produce byte-identical authority
  artifacts.

The negative tests must retain the preservation invariant. Removing the claim
does not count as a negative preservation test.

### Task 4 — Closure audit

- [ ] Focused Lean and compiler tests pass.
- [ ] `lake build` and `check-formal-sync` pass.
- [ ] Relevant canonical-Z3, verifier-mutation, and artifact-authority checks
  pass.
- [ ] The full repository gate passes on the exact closure tree.
- [ ] An independent review confirms that the kernel-checked module contains
  the digest-bound mapping timeline and assumption evidence.
- [ ] The final report states the exact automatic boundary and every remaining
  exclusion.

## Growth Envelope

The baseline is commit `3b09746a5`. Production code means committed files
under `formal/Ora` and `src`; tests, fixtures, and documentation are reported
separately.

- Task 1 semantic bridge: measured at +145 production lines.
- Task 2 tactic wiring: expected +80 to +160 production lines.
- Tasks 3 and 4 authority/closure repairs: expected +0 to +60 production lines.
- Expected total: +225 to +365 production lines.
- Hard review stop: +400 net production lines from the baseline.

Crossing +400 requires net-negative consolidation before another production
line is added. The limit is not raised inside this project. If the first
positive production certificate does not work within the envelope, the
automatic project stops; the already sound external-certificate boundary
remains unchanged.

## Completion Condition

This project is complete only when all four CLI cases exercise the production
binary, the positive case emits bytecode through canonical Z3 plus a
kernel-checked Lean certificate, every negative emits zero bytecode, the full
gate is green on the exact tree, and the reviewer confirms the certificate
chain of custody.

Completion does not claim automatic proof support for fixed arrays,
loop-carried keys, read-result keys, multiple projections, storage-mutating
children, or arbitrary storage formulas.

## Closure Outcome

The automatic project did not meet its completion condition and is closed by
its explicit stop rule. The committed result is limited to the reusable Task 1
frame-preservation bridge. The following behavior is unchanged:

- canonical Z3 and the Lean kernel remain mandatory for storage-loop
  certificates;
- the reviewed external-certificate path remains accepted and digest-bound;
- automatic planning still rejects assumption-carrying mapping-preservation
  loops with `automatic_assumptions_require_external_certificate`; and
- no new artifact-emission or runtime-check-erasure authority was introduced.

Any future automatic mapping-preservation project requires a new reviewed
contract with a compact proof-composition seam; it must not resume the reverted
storage-specific proof-script approach.
