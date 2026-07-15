# `formal/` — Ora's Lean 4 development

This directory contains Ora's machine-checked Lean definitions and proofs.
`lake build` elaborates the `Ora` library, and the Lean kernel checks every
theorem imported by `Ora.lean`.

## Scope

Ora is not a verified compiler. The results in this directory apply to specific
model fragments and generated compiler facts.

The development does not contain a complete Ora operational semantics, a
whole-language type-soundness theorem, or a proof that source-to-bytecode
lowering preserves semantics.

## Current Verification Surfaces

| Surface | Implementation | Limitation |
|---|---|---|
| Type model | Regions, primitive and composite types, declarations, structural equality and assignability, well-formedness, refinements, effects, and a small typing-judgment skeleton | The model is not connected bidirectionally to compiler type acceptance and has no runtime semantics |
| Snapshot syncs | Compiler-emitted data rows are regenerated and checked against handwritten Lean specifications | Selected finite surfaces only; not arbitrary compiler behavior |
| Obligation semantics | A supported manifest fragment has fail-closed Lean denotation and reusable bit-vector, storage, resource, and agreement theorems | Partial fragment; unsupported terms denote to `none` and cannot be proved |
| Userland proof acceptance | Selected plain Z3-`UNKNOWN` obligations can be discharged by audited Lean proofs and unblock artifact emission | Lean proofs cannot override `SAT`, degraded, vacuous, or unsupported rows and cannot erase runtime guards |
| Dispatcher verification | Repository snapshots check planner facts; contract builds with `--lean-proofs` check the concrete SIR dispatcher and, when bytecode is emitted, bind a validated backend report to the bytecode | Per-contract translation validation of known dispatcher shapes, not a general SIR-to-EVM correctness theorem |
| Resource model | Abstract create, destroy, and move operations have guard, conservation, self-move, and frame theorems | The abstract model is not a complete proof of every emitted resource operation |

## Established Results

The `Ora` library currently includes proofs or kernel-checked sync theorems for:

- reflexivity, equality characterization, and decidable equality for the modeled
  `Ty.beq` relation;
- reflexivity and transitivity of the modeled structural `Ty.assignable`
  relation;
- selected compiler type-relation rows agreeing with the Lean relations;
- structural `WF Γ t` and declaration-environment `DeclEnvWF Γ` predicates,
  with projection lemmas for well-formed declarations;
- refinement registry/coherence facts and denotational containment for the
  implemented refinement fragment;
- EVM-width `U256` arithmetic facts, including wrapping subtraction and
  totalized division/remainder edge cases;
- fail-closed denotation of unsupported obligation syntax;
- agreement, storage-disjointness, and obligation-totality fixture rows;
- dispatcher strategy, planner, and concrete table facts represented by the
  generated snapshots;
- resource conservation under the stated guards, self-move identity, and frame
  properties for untouched places.

These results apply to the definitions and generated rows imported by
`formal/Ora.lean`. The sync checks connect selected compiler facts to those
theorems; they do not model the complete compiler.

## Limitations

The following theorems or connections do not currently exist:

- no theorem equating compiler type acceptance with `WF`;
- no complete connection between compiler assignability and `Ty.assignable`;
- no general theorem that modeled assignability preserves `WF`;
- no operational semantics for the complete Ora language;
- no preservation, progress, or no-stuck theorem for Ora programs;
- no source-to-HIR, HIR-to-MLIR, MLIR-to-SIR, or SIR-to-EVM correctness proof;
- no versioned post-state storage semantics for written roots;
- no Lean proof lane for loop invariant step relations;
- no canonical SMT byte-equivalence result for every verification query;
- no proof that every compiler-generated resource goal corresponds to a
  meaningful live proof row.

## Verification Workflows

The verification system has two workflows.

### Repository sync gate

```sh
zig build check-formal-sync
```

This command:

1. regenerates the eight committed snapshots under `Ora/Generated/`;
2. rejects generated files containing proof code rather than data;
3. fails if regenerated data differs from the committed snapshots;
4. runs `lake build` over the complete `Ora` library;
5. audits theorem dependencies and rejects disallowed axioms.

The full pre-push bar includes this gate:

```sh
zig build gate
```

The gate verifies that the selected generated data agrees with the checked-in
Lean specifications. It does not verify unmodeled compiler passes.

### Contract proof workflow

Contract compilation with `--lean-proofs` first runs the normal Z3 verification
path. For eligible unresolved obligations, the userland Lean gate checks the
exact emitted manifest target, proof-target agreement, theorem elaboration, and
axiom policy. A successful proof may unblock artifact emission for that target.

The same mode also checks the concrete dispatcher extracted from SIR. If
bytecode is emitted, the dispatcher certificate is bound to the backend report
and final bytecode hashes.

Runtime guard erasure uses a separate authorization path. Only clean,
identity-matched Z3 `GuardViolate` evidence can authorize erasure. A userland
Lean proof cannot add an id to `proven_guard_ids`, erase a guard, or mark
resource runtime checks as proved.

Focused proof-checker tests are available through:

```sh
zig build test-proof-check
```

Lean is invoked as a proof checker; it is not linked into the `ora` executable.

## Trust Assumptions

The verification system trusts the following components outside the Lean
kernel:

- compiler extraction of manifests, identities, snapshots, and dispatcher
  facts;
- the correspondence between extracted facts and the compiler IR they describe;
- Z3 and Ora's SMT encoder for obligations discharged by Z3;
- the hand-maintained relation between Lean denotation and the live SMT encoding
  outside the required canonical crosscheck fragment;
- lowering and legalization not covered by dispatcher template validation;
- cryptographic hashes used to bind proof certificates to artifacts.

Unsupported or malformed proof data blocks proof acceptance and artifact
emission. The trusted components listed above are not formally verified.

## Layout

```text
formal/
  lean-toolchain
  lakefile.toml
  Ora.lean
  Ora/
    Types/          # type universe, WF, equality, assignability, effects
    Spec/           # handwritten expected compiler facts
    Generated/      # compiler-emitted data-only snapshots
    Obligation/     # manifest, U256 semantics, denotation, agreement theorems
    Resource/       # abstract resource model and theorems
    Dispatcher*.lean
    Sinora*.lean
    *Sync.lean      # trusted checks over generated data
```

The generated files are data, not proofs. Handwritten `*Sync.lean` modules
decode that data and state the checked propositions.

## Standalone Lean Build

```sh
cd formal
lake build
```

The toolchain is pinned by `lean-toolchain`. The project currently uses Lean
core and does not depend on Mathlib.

## Roadmap

The main semantic dependency chain is:

1. **Type meta-theory bridge.** Add the missing Bool/Prop bridges and prove
   composition results such as assignability preserving well-formedness for the
   explicitly modeled relation.
2. **Core operational semantics.** Define values, environments, store/world,
   lockset, and evaluation for a precisely named core fragment.
3. **Core soundness.** Prove preservation and progress/no-stuck for that fragment,
   including its effect and lock rules.
4. **Versioned storage semantics.** Model pre-state and post-state identities for
   written roots.
5. **Loop proof lane.** Add proof-targetable step relations only after versioned
   storage and loop-carried identities are available.
6. **Compiler connection.** Add translation validation or template proofs that
   connect selected compiler IR fragments to the semantics.
7. **Resource surface cleanup.** Bind meaningful resource goals to live rows and
   demote accounting-only goals.

Each milestone needs a named fragment, theorem statements, negative tests, and
gate membership before documentation may present it as a guarantee.
