# What Ora Proves

Ora's verifier is verification-first, not magic. A successful full verification
report means every emitted proof obligation was discharged by Z3 with no
recorded degradation, soundness loss, vacuity, or `UNKNOWN`. It does not
independently certify that the encoder's model of each construct is faithful —
that the model is faithful is a trusted, separately-audited assumption (see
[Trusted Assumptions](#trusted-assumptions)). Anything outside the modeled
surface is either reported as reduced/degraded verification or left as an
explicit open limit.

## Success Means

A report may claim `success: true` only when all of these hold:

- verification ran in `full` mode
- storage-state tracking was enabled
- call-summary verification was enabled
- the encoder did not record degradation or soundness loss
- vacuity checks did not find contradictory assumptions
- Z3 did not return `UNKNOWN` for any required proof obligation
- verifier errors are empty

The JSON report also exposes `verification_trust` so CI can distinguish full,
reduced, vacuous-risk, and degraded runs without reconstructing the gate.

## Proven Today

Ora currently proves these properties when the program stays inside the modeled
surface:

| Property | SMT evidence | Boundary |
|----------|--------------|----------|
| Checked integer arithmetic | Overflow and underflow checks are emitted as proof obligations. | Only modeled paths in a successful full verification run. |
| Division and remainder | Zero-divisor checks are emitted as proof obligations. | Same modeled-path boundary as arithmetic checks. |
| Closed refinement guard lexicon | Built-in refinement guards are proven or kept as runtime guards. | Arbitrary user-defined predicates are not extensible refinement guards. |
| Function contracts | `requires` clauses constrain the verified body and are enforced at boundaries; `ensures`, `assert`, loop `invariant`, and callee preconditions become SMT obligations. | Proof strength depends on the user's specs and invariants. |
| `old(expr)` | `old()` denotes function-entry state, including inside loop invariants. | Loop-entry snapshots require explicit locals such as `let start = x`. |
| Call-summary `old()` | Pure-call and heavy call-summary paths rebind materialized storage to call-site state. | Only applies to modeled summaries and resolved callees. |
| Storage frames | Resolved callees and known write sets preserve unmodified storage. | Unknown writes fail closed or record soundness loss. |
| Unresolved state-changing calls | The verifier refuses to silently preserve storage. | No post-state facts are proved without a sound summary. |
| Unresolved `staticcall` | Modeled as no-write with opaque return values. | No semantic facts about return values without a summary. |
| Trusted extern summaries | `staticcall fn` summaries can specify return facts; framed `call fn` summaries can preserve caller storage. | Trusted boundary; user `modifies` exists for current-contract storage paths, but richer extern-summary effects remain future work. |
| Effectful loops | Supplied invariants prove entry, preservation, body safety, and loop-post obligations. | Missing invariants fail proof rather than implying induction. |
| Loop body safety | Inductive-iteration body obligations, such as checked arithmetic, are discharged. | Applies to the invariant path the verifier can encode. |
| EVM environment reads | Selected environment values are stable symbolic values within a query. | The verifier does not model every environmental or gas-sensitive behavior. |
| Runtime `keccak256` | Modeled as a deterministic uninterpreted function. | No collision-resistance or cryptographic axiom is assumed. |

The closed refinement guard lexicon currently includes `MinValue`, `MaxValue`,
`InRange`, `NonZeroAddress`, `NonZero`, and `BasisPoints`.

Ora also has refinement types such as `Exact` and `Scaled`. Those are
type-system and representation facts unless a specific construct emits an
active guard or SMT obligation. They are not automatically part of the closed
runtime/SMT guard lexicon above.

## Not Proven Today

These are not current Ora verifier claims:

- unresolved external contract return values are not semantically specified;
  semantic return facts require explicit trusted extern summaries
- unresolved state-changing external calls do not prove post-state properties;
  they degrade or fail closed
- trusted `call fn` summaries do not yet support arbitrary user-declared
  external effects; current `modifies` support is limited to v1 current-contract
  storage paths and framed summary preservation
- `delegatecall` is intentionally absent and must not inherit `staticcall`
  no-write semantics if added later
- `keccak256` collision resistance is not assumed; the verifier only assumes
  deterministic equality for identical inputs
- precompile cryptographic validity, gas, failure, and exceptional behavior are
  not modeled
- ABI byte layout for opaque boundary values is not proven by the SMT model
- final EVM bytecode equivalence is not yet proven against the SMT model
- bytecode-level differential tests cover representative storage layouts and
  multi-contract behavior, but they are not a complete symbolic bytecode proof
- user-defined refinement predicates outside the closed guard lexicon are not
  supported as arbitrary extensible refinement guards
- custom trigger inference for user `forall`/`exists` remains future work

## Soundness Escape Hatches

These settings intentionally reduce what the verifier can claim:

- `--no-verify` — disables Z3 verification entirely in emit/report modes
- `--verify=basic` — runs reduced (non-`full`) verification

`ora build` rejects `--no-verify`; emit-mode output or reports produced through
reduced verification settings must not be presented as fully verified. Full
verification reports require `full` mode and record the reduced trust state when
that bar is not met.

Parallel SMT execution is not implemented; the verifier runs in a single Z3
context. There is no cross-context parallel path.

## Trusted Assumptions

The verifier still relies on implementation trust:

- Z3 is trusted as the SMT backend.
- Ora-to-MLIR lowering is trusted to preserve source semantics.
- The MLIR-to-SMT encoder is trusted to model each construct faithfully. A
  modeling error that makes a satisfiable obligation appear `UNSAT` (a *false
  UNSAT*) would erase a runtime guard that should have been kept. This surface is
  audited construct-by-construct and gated by oracle and differential execution
  tests, but it is implementation trust, not a theorem.
- Sema-emitted metadata such as external call kind and computed write slots is
  trusted, with cross-checks where implemented.
- The current storage-layout manifest checks the Ora-to-SIR boundary; full
  verifier-to-bytecode equivalence remains Pillar 4 work.
- `@lock` reentrancy protection is a sema/effect-system guarantee, not an SMT
  theorem.

## Runtime Bytecode Evidence

Ora also maintains deployed-bytecode smoke tests for properties that are outside
the SMT proof claim but still matter for the source-to-EVM boundary:

- scalar storage, mappings, nested mappings, slices, structs, packed fields, and
  common nested compositions use Solidity-compatible storage slots in deployed
  bytecode
- public ABI transfer covers representative static words, fixed bytes, and
  structured payloads used by those storage fixtures
- multi-contract fixtures cover storage namespace isolation, return-data
  decoding, try/catch paths, revert rollback, and gas-exhausted external-call
  failure slices

These tests are regression evidence for emitted bytecode. They do not replace a
formal SMT-vs-EVM transition equivalence proof.

## Reading Reports

Prefer the JSON report for automation:

- `verification.success` is the high-level gate.
- `verification.verification_trust` explains whether the run was full, reduced,
  vacuous-risk, or degraded.
- `summary.soundness_losses` lists structured soundness-loss reasons.
- `query_fragment_counts` shows how many obligations used `QF_BV`,
  `QF_BV+Array`, `AUFBV`, `AUFBV+Quantifiers`, `unknown`, or `other`.
- each query records its fragment, solver logic, status, vacuity result, and
  optional model/proof payload.

Current `summary.soundness_losses` labels are:

<!-- CI guard: soundness-loss labels in this section must stay backtick-wrapped. -->

- `user_disabled_state_verification`
- `user_disabled_call_verification`
- `missing_type_metadata`
- `missing_product_metadata`
- `missing_control_flow_summary`
- `unsupported_operation`
- `unsupported_error_encoding`
- `unsupported_sort_coercion`
- `unresolved_callee`
- `inexact_call_summary`
- `inexact_state_summary`
- `internal_encoding_failure`
- `soundness_loss_cap_exceeded`

`soundness_loss_cap_exceeded` is a truncation marker, not a separate proof
failure. It means the report reached its bounded soundness-loss list size and
additional specific labels were omitted. Precision-note reports use the same
convention with `precision_note_cap_exceeded`.
