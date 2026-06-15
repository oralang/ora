# Conformance Findings Ledger

Execution- and mutation-discovered compiler/EVM findings, one entry each.
Corpus `.ora` and `.spec.toml` files carry only `// see FINDINGS.md#<id>`
pointers, never finding prose.

When a finding is fixed: flip its characterization rows to the intended
behavior, run `zig build test-conformance`, and mark the entry FIXED with the
commit.

---

## F-001 — `requires` not boundary-enforced; discharged checks erased

- **Status:** FIXED
- **Severity:** S1
- **Owner:** compiler+smt-audit
- **What:** public `requires` clauses were verification-only assumptions and did
  not guard ABI entrypoints. Fixed by emitting executable runtime assertions
  alongside SMT preconditions.
- **Repro/fixed rows:** `tests/conformance/arithmetic_checked_revert.*`.

## F-002 — catch-path unpacking of aggregate error-union payloads

- **Status:** FIXED
- **Severity:** S1
- **Owner:** compiler
- **What:** `catch` after a `try` on aggregate-success error unions dereferenced
  scalar error IDs as aggregate payload pointers. Fixed by distinguishing
  pointer-backed payloads from scalar error words.
- **Repro/fixed rows:** `error_union_wide_carrier.err_default()` and
  `error_union_local_aggregate.err_marker()`.

## F-003 — lib/evm panics on huge-offset memory gas computation

- **Status:** FIXED
- **Severity:** S2
- **Owner:** lib/evm
- **What:** hostile huge-offset memory operations could panic lib/evm gas math
  instead of returning a clean EVM failure. Fixed by bounding memory expansion
  accounting so huge `mload`/`mstore`/`mstore8`/`mcopy` cases return `OutOfGas`
  without panicking.
- **Proof:** `zig build test-evm --summary all` covers the huge-offset memory
  handlers.

## F-004 — tuple return from inside try/catch fails OraToSIR

- **Status:** FIXED
- **Severity:** S3
- **Owner:** compiler
- **What:** returning a `(u256, string)` tuple from inside a try/catch block
  failed lowering. Fixed and covered by a content-bearing tuple return from the
  catch path.
- **Proof:** `error_union_local_aggregate.catch_content()` returns
  `(uint256,string)` and passes both lib/evm conformance and Anvil differential
  execution.

## F-005 — trait-impl methods could not be called from contract functions

- **Status:** FIXED
- **Severity:** S2
- **Owner:** compiler
- **What:** contract-context calls to monomorphized trait impl methods failed to
  reference a materialized function. Covered now by
  `trait_impl_contract_call.*`.

## F-006 — narrow error-union dispatcher returns lost selectors

- **Status:** FIXED
- **Severity:** S1
- **Owner:** compiler
- **What:** public narrow error-union return paths lost custom error selector
  data and had untrusted success encoding. Covered now by
  `dispatcher_narrow_error_union_return.*`.

## F-007 — conformance harness did not thread metered gas

- **Status:** FIXED
- **Severity:** S3
- **Owner:** test-harness
- **What:** `gas_max` initially had no teeth because gas-used accounting read as
  zero. Covered now by the conformance gas ceiling self-test.

## F-008 — events emitted no topic0 signature hash

- **Status:** FIXED
- **Severity:** S2
- **Owner:** compiler
- **What:** events originally emitted no `topic0`, making them non-filterable by
  standard ABI tooling. Covered now by `events_multifield.*`.

## F-009 — signed narrow integer ABI returns are zero-extended

- **Status:** FIXED
- **Severity:** S1
- **Owner:** compiler
- **What:** `get_delta() -> i16` returned the low 16-bit two's-complement value
  zero-extended as `65529`, while Solidity ABI `int16` return values must be
  sign-extended to 256 bits. Fixed by routing public scalar returns through the
  shared ABI static-word materializer and tightening the in-process conformance
  decoder to reject non-canonical signed return words.
- **Proof:** `zig build test-conformance --summary all` catches the canonical
  ABI word requirement, and `zig build test-conformance-anvil --summary all`
  now reports `54 specs | 54 agree | 0 divergences | 0 errored`.

## F-010 — loop invariant postcondition accepts body corruption

- **Status:** FIXED
- **Severity:** S1
- **Owner:** smt-audit
- **What:** the bounded mutation probe
  `MutationLoopInvariant.countTo` verified and emitted artifacts when the loop
  body changed from `counter = counter + 1` to `counter = counter + 2`, even
  though the function claims `ensures(counter == n)`. Fixed by checking natural
  loop-exit invariant states separately from the backedge step.
- **Proof:** `verify_mutations.py` now includes
  `loop_invariant_body_corruption`, and `zig build check-verifier-mutations`
  rejects it.

## F-011 — state invariant preservation accepts weakened preconditions

- **Status:** FIXED
- **Severity:** S1
- **Owner:** smt-audit
- **What:** a mutation probe for a contract state invariant verified and emitted
  artifacts after weakening the precondition that should make the invariant
  provable. Minimal shape:
  `storage var x: u256 = 0; invariant bounded(x <= 100); set(v)` writes
  `x = v`. The original `requires(v <= 100)` verifies, but the mutant
  `requires(true)` also verifies even though `v = 101` violates the invariant
  after the write. Fixed by re-encoding global contract invariants after each
  verified function body has updated the verifier storage model.
- **Proof:** `verify_mutations.py` now includes
  `state_invariant_precondition_removed`, and
  `zig build check-verifier-mutations` rejects it.
