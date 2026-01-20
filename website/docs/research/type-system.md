---
sidebar_position: 3
---

# Type System

Ora’s type system is region-aware, refinement-based, and effect-conscious. It is
built to make storage semantics explicit and make proofs first-class.

## Core ideas

- **Regions as types**: storage, memory, transient, calldata are explicit.
- **Refinement types**: value constraints live in types and are enforced.
- **Hybrid static/dynamic**: prove when possible, guard when not.
- **Effect visibility**: writes are tracked and surfaced in typing judgment.
- **Error unions**: explicit success/error channels.

## Formal foundations

The formal definition lives in the v0.11 PDF, with a calculi‑level description
of regions, refinements, effects, and lock discipline:

- [Type System Specification v0.11 PDF](/Ora%20Type%20System%20Specification%20v0.11.pdf)

## Implemented baseline

The implemented baseline is captured in `TYPE_SYSTEM_STATE.md` and includes:

- Core types and error unions (`!T | Err1 | Err2`).
- Flow‑sensitive refinements for branch conditions.
- SMT‑only assumptions for ambiguous control flow.
- `requires`, `ensures`, `invariant`, `assume`, `assert` typing.

## Example: region + refinement intent

```ora
contract Vault {
    storage var balances: map[NonZeroAddress, u256];

    pub fn credit(user: NonZeroAddress, amount: MinValue<u256, 1>) {
        balances[user] = balances[user] + amount;
    }
}
```

## Typing judgment

The core typing judgment tracks storage layout, context, lockset, and effects:

```
Σ; Γ; Λ ⊢ e : σ ! ϵ
```

- `Σ` storage layout
- `Γ` typing context
- `Λ` lockset
- `σ` located type
- `ϵ` effect (e.g., Pure or Writes)

## Implementation details

- Type resolution and refinement validation live in `src/ast/type_resolver/**`.
- Flow‑sensitive refinements are applied in the resolver for branch conditions.
- SMT‑only assumptions are emitted as `assume` statements when refinements
  cannot be inferred safely.
- Refinement guards are emitted during MLIR lowering and can be proven by the
  SMT pass to avoid runtime checks.

## Evidence

- `TYPE_SYSTEM_STATE.md`
- `docs/Ora Type System Specification v0.11.pdf`
- `website/docs/design-documents/type-system-v0.1.md`
