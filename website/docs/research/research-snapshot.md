---
sidebar_position: 2
---

# Research Snapshot

Current research foundation of Ora. Not a
roadmap; it is a compact view of what is specified, implemented, and actively
investigated.

## Core research pillars

### Type System

- **Specified**: Region-aware, refinement-based calculus (v0.11 PDF, `ora-2`).
- **Implemented**: Phase‑1 baseline in `TYPE_SYSTEM_STATE.md`.
- **Next**: Refinement propagation across arithmetic and control-flow joins.

### Comptime

- **Specified**: Comptime-first reasoning as a design principle.
- **Implemented**: Pure comptime evaluation during type resolution.
- **Next**: Comptime function execution and partial evaluation.

### SMT Verification

- **Specified**: Proof-first, runtime-fallback model.
- **Implemented**: Z3 pass over Ora MLIR with counterexamples.
- **Next**: Default guard discharge and SMT-driven pruning.

### Refinement Types

- **Specified**: Hybrid static–dynamic refinements with SMT proofs.
- **Implemented**: Guard insertion and refinement validation in the front end.
- **Next**: Rewrap semantics and full SMT integration.

## Evidence (selected)

- `docs/Ora Type System Specification v0.11.pdf`
- `docs/formal-specs/ora-2.md`
- `TYPE_SYSTEM_STATE.md`
- `docs/compiler/comptime-and-smt.md`
- `docs/compiler/refinement-types-strategy.md`
- `docs/compiler/formal-verification.md`
- `docs/compiler/onboarding/09-z3-smt.md`

## Read next

- [Type System](./type-system)
- [Comptime](./comptime)
- [SMT Verification](./smt-verification)
- [Refinement Types](./refinement-types)
- [Compiler Architecture](./compiler-architecture)
