---
sidebar_position: 1
---

# Introduction to Ora

> **Pre-ASUKA Alpha** | **Research-driven** | **Contributors Welcome**

Ora is an experimental smart contract language and compiler focused on precise
semantics, explicit memory regions, and verification-friendly design. The
compiler is built in Zig and lowers through Ora MLIR to Sensei-IR (SIR) on the
backend path to EVM bytecode.

This site has two complementary tracks:

- **Practical docs** for writing Ora and using the compiler today.
- **Research docs** that capture the academic and architectural foundations.

## What Ora is

Ora is not a Solidity clone. A language for contracts where the
compiler is a first-class research artifact. The design favors explicitness:
regions, effects, and refinement constraints are surfaced and checked early.

## Current status

Ora is in **pre-release alpha** on the ASUKA track. The front-end pipeline
(lexing, parsing, type resolution, and Ora MLIR emission) is active and under
constant iteration. Backend lowering to Sensei-IR (SIR) and EVM is active.

Expect breaking changes. Nothing here is production-ready.

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
- [Examples](./examples) for working patterns and caveats.
- [Compiler Field Guide](./compiler/field-guide/index) for contributors and new compiler engineers.
- [Research](./research/index) for the academic lens: formal verification, type system
  strategy, and compiler architecture.

## Contributing

Ora is a research-grade compiler that benefits from tests, docs, and minimal
reproducers as much as from compiler changes. If you want to help, start with:

- `CONTRIBUTING.md` in the repo
- the Compiler Field Guide
- small, well-scoped documentation fixes
