---
sidebar_position: 5
---

# Asuka v0.1 — Release Notes

Asuka v0.1 is Ora's first release milestone: a coherent front-end pipeline with
clear language semantics and a working backend path to EVM bytecode.

## What shipped

- **Front-end pipeline**: lexer, parser, type resolution, Ora MLIR emission.
- **Regions**: storage, memory, calldata, transient — with compiler-enforced transition rules.
- **Type system**: refinement types, error unions, structs, enums, tuples, bitfields, generics.
- **Traits**: declaration, `impl` blocks with `self` receivers, bounded generics, ghost specs, comptime methods.
- **Verification**: `requires`/`ensures`/`invariant`/`guard`/`assume`/`assert`, ghost state, Z3 SMT integration, counterexamples.
- **Arithmetic**: checked by default, wrapping operators, overflow-reporting builtins, signed integers.
- **Backend**: Ora MLIR → Sensei-IR (SIR) → EVM bytecode, end-to-end compilation.
- **Tooling**: `ora fmt` (canonical formatter), `ora init` (project scaffolding), `ora debug` (source-level EVM debugger).
- **Imports**: `comptime const` imports, `ora.toml` project configuration, cycle detection.

## What's next

- Stronger diagnostics and error messages
- Deeper verification: loops, quantifiers, interprocedural analysis
- Backend parity coverage for all supported constructs
- Tooling stabilization and editor integration

## Contributing priorities

- Backend lowering and legalization
- Error handling and diagnostics
- Tests for refinements, effects, and regions
- Documentation updates
