---
sidebar_position: 4
---

# Comptime

Ora is comptime-first: as much reasoning as possible happens at
compile time, with SMT as the fallback and runtime checks as the last resort.

Source: `docs/compiler/comptime-and-smt.md`.

## Implemented today

- Comptime evaluation is used during type resolution and refinement validation.
- It evaluates pure, side-effect-free expressions and constants.
- It helps skip guards when refinements are provably satisfied by constants.

## Implementation details

- Refinement validation: `src/ast/type_resolver/refinements/**`.
- Guard skipping: `src/ast/type_resolver/core/statement.zig`.
- SMT-only assumptions emitted for branch constraints that cannot refine types.

## Scope boundaries

- No general execution of user-defined functions at comptime.
- No storage reads/writes or effectful evaluation.
- No symbolic execution across control flow.

## Research direction

The design goal is a hermetic, deterministic comptime model that supports:

- Pure comptime execution without IO or host leakage.
- Explicit comptime blocks and parameters.
- Partial evaluation before SMT to reduce proof cost.

## Evidence

- `docs/compiler/comptime-and-smt.md` (implementation state and roadmap)

## References

- `docs/compiler/comptime-and-smt.md`
- `docs/formal-specs/ora-2.md` (design intent)
