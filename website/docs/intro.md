---
sidebar_position: 1
---

# Introduction to Ora

> **Asuka v0.2** | **Proof-carrying contracts** | **Contributors Welcome**

Ora is a smart contract language and compiler focused on precise semantics,
explicit memory regions, and verification-friendly design. The compiler is
built in Zig and lowers through Ora MLIR to Sensei-IR (SIR) on the backend
path to EVM bytecode.

This site has two complementary tracks:

- **Practical docs** for writing Ora and using the compiler today.
- **Research docs** that capture the academic and architectural foundations.

## What Ora is

Ora is not a Solidity clone. It is a language for contracts where the compiler
is part of the trust story. The design favors explicitness: regions, effects,
ADT shapes, ABI layouts, and refinement constraints are surfaced and checked
early.

## Current status

Ora has reached **Asuka v0.2**. This release expands the language and compiler
around proof-carrying contracts: first-class `Result<T, E>` / error-union
values, unified ADTs, Z3 verification reports with vacuity/degradation
surfacing, runtime ABI encode/decode support, dynamic public returns, hardened
extern traits, source-level debugging, LSP production work, compiler metrics,
and CFG tooling.

The Asuka track is still where Ora's language surface evolves, but v0.2 is a
release milestone: valid examples should compile, unsupported features should
fail closed, and the docs should describe compiler reality.

## Research focus

We document research work in-progress as first-class artifacts:

- [Type System](./research/type-system)
- [Comptime](./research/comptime)
- [SMT Verification](./research/smt-verification)
- [Refinement Types](./research/refinement-types)
- [Formal Verification](./formal-verification)

## How to read this documentation

- [Getting Started](./getting-started) to build the compiler and run your first file.
- [Language Basics](./language-basics) for core syntax and types.
- [Imports and Modules](./imports) for multi-file projects and `ora.toml`.
- [Examples](./examples) for working patterns and caveats.
- [Compiler Field Guide](./compiler/field-guide) for contributors and new compiler engineers.
- [Research](./research) for the academic lens: formal verification, type system
  strategy, and compiler architecture.

## Contributing

Ora is a research-grade compiler that benefits from tests, docs, and minimal
reproducers as much as from compiler changes. If you want to help, start with:

- `CONTRIBUTING.md` in the repo
- the Compiler Field Guide
- small, well-scoped documentation fixes
