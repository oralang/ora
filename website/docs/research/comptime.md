---
sidebar_position: 4
---

# Comptime

Ora is comptime-first: as much reasoning as possible happens at compile time, with SMT as the fallback and runtime checks as the last resort.

## Implemented today

- **Constant folding** — Pure, side-effect-free expressions are evaluated at compile time during the fold pass (`src/ast/type_resolver/comptime_fold.zig`). Calls to functions (including generic functions) are folded when all arguments are compile-time known.
- **Comptime type parameters** — `comptime T: type` in function and struct definitions; type is a first-class value at compile time. Used for [generics](/docs/generics): generic functions and structs are monomorphized per instantiation.
- **Type resolution and refinements** — Comptime evaluation is used during type resolution and refinement validation; it helps skip guards when refinements are provably satisfied by constants.
- **Refinement validation** — `src/ast/type_resolver/refinements/**`.
- **Guard skipping** — `src/ast/type_resolver/core/statement.zig`; SMT-only assumptions for branch constraints that cannot refine types.

## Scope boundaries

- No storage reads/writes or effectful evaluation at comptime.
- No symbolic execution across control flow.
- User-defined functions are executed at comptime only when invoked in constant contexts (all arguments known); there is no general “run any function at comptime” mode.

## Research direction

The design goal is a hermetic, deterministic comptime model that supports:

- Pure comptime execution without IO or host leakage.
- Explicit comptime blocks and parameters (beyond current use in generics).
- Partial evaluation before SMT to reduce proof cost.

## References

- [Generics (comptime type parameters)](/docs/generics)
- `docs/compiler/comptime-and-smt.md` (implementation state and roadmap)
- `docs/formal-specs/ora-2.md` (design intent)
