# Introduction
EVM-IR Specification v0.1  
Status: Draft  
Audience: Compiler engineers, VM architects, language designers

---

## Purpose of This Specification

This document defines **EVM-IR**, a low-level, SSA-based intermediate representation
for compiling high-level smart contract languages to Ethereum Virtual Machine (EVM)
bytecode.

EVM-IR is designed to:

- Serve as a **common backend target** for multiple languages (Ora and others)
- Provide a **precise, analyzable model** of EVM-style execution
- Enable **optimizations and formal reasoning** at a level higher than bytecode
- Remain **EVM-aware** without mirroring the EVM’s stack machine structure
- Be **simple to implement** in existing compiler toolchains (e.g., MLIR dialects)

EVM-IR is *not* a source language, runtime, or bytecode format. It is a compiler
IR that sits between high-level language frontends and final EVM code generation.

---

## Design Goals

EVM-IR is guided by the following design goals:

### Language-Agnostic

EVM-IR must not embed assumptions from a single language (e.g. Ora, Solidity, Vyper).
Any frontend that can lower its semantics to a:

- typed, SSA-based control-flow graph
- without composite types or high-level exceptions

should be able to target EVM-IR.

### EVM-Aware but Not Stack-Shaped

The EVM is a stack machine. EVM-IR is **not**.

Instead, EVM-IR:

- Models EVM concepts explicitly (storage, memory, calldata, transient storage)
- Uses **SSA values and typed operations** instead of implicit stacks
- Defers stack concerns to a later **stackification** phase

This separation makes IR easier to reason about, optimize, and verify.

### Canonical SSA Form

EVM-IR is always in a structured SSA form with:

- Typed SSA values
- Basic blocks with explicit terminators
- No critical edges
- No unreachable blocks after legalization
- No PHI operations in the final canonical form (state merges use memory)

A separate **legalizer** pass normalizes arbitrary EVM-IR into this canonical form.

### Deterministic Lowering to Bytecode

Given a canonical EVM-IR module, lowering to EVM bytecode should be:

- **Deterministic**: no hidden semantics or non-local choices
- **Predictable**: small IR changes produce small codegen changes
- **Reasonable**: preserves obvious performance expectations

To achieve this, the specification defines:

- Allowed operations and their semantics
- Control-flow and structural invariants
- Stackification rules (SSA → stack machine)

### Explicit, Minimal Semantics

EVM-IR avoids embedding high-level language features such as:

- exceptions
- generics
- inheritance
- traits / interfaces
- language-specific error models

Instead, it exposes a minimal set of operations that can encode these features
using control flow, memory, storage, and calls.

---

## Non-Goals

EVM-IR intentionally does *not* attempt to:

- Define a new source language or syntax
- Replace the Ethereum ABI specification
- Define gas pricing or performance guarantees
- Encode every EVM quirk as a first-class concept
- Serve as a human-friendly assembly language
- Model non-EVM backends (e.g., WASM) directly

Language frontends are responsible for:

- Type checking and high-level type systems
- Ownership / borrowing / capability models
- Semantic checks (e.g. overflow policies)
- Composite type lowering (structs, arrays, enums)
- High-level optimizations

EVM-IR assumes that all high-level decisions have been made before lowering.

---

## Position in the Compilation Pipeline

EVM-IR sits in the middle of the compilation pipeline:

```text
 High-Level Language (Ora, etc.)
        │
        ▼
  Frontend IR / AST / MLIR dialect
        │
 (language-specific lowering)
        ▼
        EVM-IR
        │
   [Legalizer Pass]
        ▼
  Canonical EVM-IR (SSA)
        │
   [Stackifier Pass]
        ▼
   Stack-Oriented IR
        │
   [Backend / Codegen]
        ▼
     EVM Bytecode
```

Key properties:

- **Frontends** lower their own constructs to EVM-IR, not directly to bytecode.
- **Legalizer** ensures EVM-IR satisfies canonical constraints (no PHI, no composites, normalized CFG).
- **Stackifier** converts SSA into linear, stack-based code suitable for bytecode emission.
- **Bytecode generator** encodes the stack-level program into raw EVM opcodes (out of scope for v0.1).

---

## Module and Function Model

An EVM-IR compilation unit is a **module**.

A module contains:

- **Functions**: code bodies with typed parameters and return types
- **Global metadata**: debug info, ABI information, target properties
- **Optionally**: declarations of external functions or runtime hooks

Each function:

- Has a name (e.g., `@transfer` or `@__entry`)
- Has a signature: list of parameter types and result types
- Contains a non-empty set of **basic blocks**
- Has exactly one **entry block**
- Uses SSA values produced by operations within the function

Basic blocks:

- Contain a sequence of operations
- End with a **terminator** (branch, conditional branch, return, revert, etc.)
- May have zero or more **predecessors**
- Have no implicit control-flow edges

---

## Canonical IR Requirements (High-Level)

While the detailed canonicalization rules are specified in the **Legalizer** section,
the following high-level requirements hold for canonical EVM-IR:

1. **SSA Form**  
   Each value is assigned exactly once and used any number of times.
   No mutable variables; state is represented via memory, storage, or transient storage.

2. **No PHI Nodes in Final Form**  
   IR may temporarily use PHI-like constructs (MLIR block arguments, etc.),  
   but the legalizer must ultimately lower all control-flow merges to explicit
   memory operations (e.g., stores in predecessors, loads in merge blocks).

3. **Explicit Control Flow**  
   Every block ends with a terminator. No fall-through without an explicit branch.

4. **No Composite Types**  
   Structs, tuples, arrays, and maps are not first-class types in EVM-IR.
   They must be lowered to primitive values and memory operations.

5. **Well-Formed CFG**  
   No unreachable blocks. No critical edges that violate canonicalization rules.

6. **Well-Typed Operations**  
   Every operation must satisfy type rules defined in the **Operations** section.
   Implicit conversions are not allowed.

---

## EVM Awareness and Address Spaces

Although EVM-IR is not a stack machine, it is explicitly aware of the distinct
EVM address spaces:

- **Memory**: transient per-call, word-addressable, zero-initialized
- **Storage**: persistent contract key-value store
- **Calldata**: read-only call input buffer
- **Transient Storage**: EIP-1153 temporary key-value store, cleared at end of transaction
- **Code**: read-only contract code region for `EXTCODE*` operations

These are expressed via **typed pointers** with address space identifiers, rather
than via raw numeric offsets alone. This allows:

- static verification of address space usage
- more robust lowering to EVM opcodes
- analysis and optimization passes to reason about memory vs. storage vs. transient vs. calldata

---

## Relationship With Frontends (e.g., Ora)

EVM-IR is intentionally **not tied** to the Ora language, but Ora is a primary
expected frontend.

Frontends are responsible for:

- Introducing EVM-IR functions that implement their language functions
- Lowering language-level constructs (errors, results, enums, structs, etc.)
  into memory and control flow
- Providing ABI metadata used by the ABI lowering stage
- Injecting verification-friendly patterns if desired

The IR itself does not know about:

- Ora’s refinement types
- Ora’s proof obligations
- Language-level error unions

Those are erased or encoded into EVM-IR primitives before or during lowering.

---

## ABI and Entry Semantics (High-Level View)

EVM-IR supports:

- A special **entry dispatcher** function that:
  - examines calldata
  - extracts the 4-byte selector
  - routes execution to the appropriate function body
  - handles unknown selectors and ETH-only transfers

- Per-function **ABI decode** logic that:
  - reads argument words from calldata
  - decodes static types into SSA values
  - decodes dynamic ABI types into memory regions

- Per-function **ABI encode** logic that:
  - writes return values to memory in ABI layout
  - returns a pointer + length pair via `return` / `revert`

Details are provided in the **ABI Lowering** section, but the introduction notes that
EVM-IR is designed to act as the point where ABI obligations become explicit.

---

## Debug Information and Tooling

EVM-IR defines a minimal, language-agnostic debug model:

- Source locations (file, line, column)
- Variable metadata (name, type, location)
- Scope information (function/block scopes)

The specification also describes how to export this information into a format
compatible with the **ethdebug/format** project, allowing integration with:

- debuggers
- tracers
- on-chain analysis tools

Details are provided in the **Debug Information** section.

---

## Stackification and Backend

EVM-IR is not directly executable. It must be lowered to:

1. **Canonical EVM-IR (SSA)** — after legalization
2. **Stack-Oriented IR** — via the stackifier
3. **EVM bytecode** — via a backend encoder (out of scope for v0.1)

The stackifier is responsible for:

- choosing frame layouts for stack and memory
- scheduling instructions according to stack constraints
- introducing DUP/SWAP operations when necessary
- ensuring efficient code generation

The backend then maps Stack IR to raw opcodes.

---

## Document Structure

This specification is split into the following standalone sections:

- `types.md` — Type system (primitive types, pointers, address spaces)
- `ops.md` — Operations (semantics, type rules, constraints)
- `legalizer.md` — Canonical form, legalizer rules, CFG normalization
- `stackifier.md` — SSA → stack machine transformation
- `debug.md` — Debug metadata and ethdebug integration
- `abi-lowering.md` — ABI decoding, encoding, dispatcher, fallback/receive
- `examples.md` — Worked examples of complete lowering pipelines
- `appendix.md` — Reference tables, glossary, and auxiliary material

Each file can be read independently, but together they describe the full EVM-IR model.
