# EVM-IR Specification v0.1

**Status:** Draft  
**Audience:** Compiler engineers, VM architects, language designers

---

# Introduction

## Purpose of This Specification

This document defines **EVM-IR**, a low-level, SSA-based intermediate representation
for compiling high-level smart contract languages to Ethereum Virtual Machine (EVM)
bytecode.

EVM-IR is designed to:

- Serve as a **common backend target** for multiple languages (Ora and others)
- Provide a **precise, analyzable model** of EVM-style execution
- Enable **optimizations and formal reasoning** at a level higher than bytecode
- Remain **EVM-aware** without mirroring the EVM's stack machine structure
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

- Ora's refinement types
- Ora's proof obligations
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

# EVM‑IR Type System Specification

This document defines the **complete type system** for the EVM‑IR dialect.  
Types are designed to be:

- **Fully deterministic**  
- **Lowerable to raw EVM semantics**  
- **SSA-compatible**  
- **Address‑space aware**  
- **Front‑end independent**  

EVM‑IR must express *all* values that can appear during EVM execution while maintaining a clean and analyzable static representation.

---

## Design Principles

### Minimal but Sufficient  
The EVM has only one real data type: **256‑bit stack words**.  
EVM‑IR exposes a richer static type system for:

- safety  
- optimization  
- cross‑dialect analysis  
- debug info  
- ABI lowering  

…but **all types must map cleanly to EVM stack words**, memory bytes, or structured memory regions.

### No Composite Types  
To keep the IR canonical and close to EVM semantics:

❌ No structs  
❌ No tuples  
❌ No arrays as first‑class values  
❌ No arbitrary-width integers  

Front‑ends (like Ora) must lower composites into memory regions *before* emitting EVM‑IR.

### Strongly Typed Pointers  
Pointers exist only with explicit address spaces:

- memory (0)  
- storage (1)  
- calldata (2)  
- transient storage (3)  
- code (4)  

---

## Primitive Types

### `u256`
- Arbitrary EVM stack word  
- Represents unsigned integers modulo 2²⁵⁶  
- Used for:
  - arithmetic
  - indexing
  - storage keys
  - ABI values

This is the *dominant* type in EVM‑IR.

### `u160`
- 160‑bit value  
- Canonical representation for EVM addresses  
- Codegen guarantees zero‑extension to 256 bits when needed.

### `bool`
- Logical value restricted to `{0, 1}`  
- Codegen enforces normalization when necessary  
- Represented as a 256‑bit stack value where `0 = false`, `1 = true`

### `u32`, `u8`
These narrower integer types are necessary for:

- byte/word operations  
- calldata slicing  
- memory offsets  
- hashing preparation  

Representation:
- always **zero‑extended** to 256-bit stack words
- do *not* carry sign information
- survive optimization passes intact

---

## Pointer Types

Pointer syntax:

```
ptr<addrspace>
ptr<addrspace, size=N>
```

The `addrspace` parameter is required and specifies which EVM address space the pointer references.

The `size` parameter is **optional** and specifies the size (in bytes) of the memory region the pointer references. This is useful for:
- Frame layout optimization in the stackifier
- Type safety verification (bounds checking)
- Static analysis and optimization

When `size` is omitted, the pointer size is unknown or not tracked at the type level.

### Address Space Table

| Address Space | Meaning | EVM backing | Notes |
|--------------|---------|--------------|-------|
| `0` | Memory | `MLOAD / MSTORE` | Byte-addressable, uses `ptr<0>` |
| `1` | Storage | `SLOAD / SSTORE` | Uses `u256` keys, not pointers |
| `2` | Calldata | `CALLDATALOAD` | Read-only, uses `ptr<2>` |
| `3` | Transient | `TLOAD / TSTORE` | EIP-1153, uses `u256` keys |
| `4` | Code | `CODESIZE / CODECOPY` | Read-only code region |

### Memory Pointers
- represent byte-addressable memory addresses  
- arithmetic permitted via `evm.ptr_add`
- EVM memory operations (`MLOAD`/`MSTORE`) can operate on any byte offset (alignment not enforced by EVM)
- Size metadata (when present) helps with frame layout and bounds checking

Example:
```
%p = evm.alloca 32 : ptr<0, size=32>  // allocates 32 bytes, pointer tracks size
%q = evm.ptr_add %p, %offset : ptr<0>  // result pointer may not preserve size
```

### Storage Keys
- Storage uses `u256` keys, not pointers  
- Storage keys are represented as `u256` values, not `ptr<1>`  
- Arithmetic must be expressed explicitly  
- Compiler uses Keccak when lowering composite offsets

### Calldata Pointers
Represent byte offsets into calldata buffer.

### Transient Storage Keys
- Transient storage uses `u256` keys, not pointers  
- Transient storage keys are represented as `u256` values, not `ptr<3>`  
- For EIP‑1153 transient storage operations

### Code Pointers
Represent offsets into the contract code region.

---

## Special Types

### `void`
Used for:
- operations that do not produce SSA values
- pure side‑effect operations

### `unreachable`
Represents IR regions that cannot execute.  
Used by:
- legalizer
- control‑flow verification
- optimization passes

---

## Type Rules

### Arithmetic
- `u256 op u256 → u256`
- Narrow types (`u8`, `u32`, `u160`) **automatically zero-extend** to `u256` when used in arithmetic operations
- `bool` is treated as `u256` during arithmetic but must be re‑normalized before branch conditions
- Operations like `evm.mload8` return `u8`, which is then zero-extended to `u256` when used in subsequent operations

### Comparisons
Comparisons always produce `bool`.

```
u256 < u256 → bool
u256 == u256 → bool
...
```

### Pointers
Pointer arithmetic requires explicit operations:

```
%p2 = evm.ptr_add %p1, %offset
```

No implicit pointer addition.

### Type Casting

**Implicit casts** (automatic zero-extension):
- `u8 → u32 → u256` - when used in operations expecting wider types
- `u160 → u256` - when used in arithmetic or comparisons
- `bool → u256` - when used in arithmetic operations

**Explicit casts** required to go *down* in width:
- `u256 → u160` - truncation (e.g., for address operations)
- `u256 → u32` - truncation (e.g., for byte operations)
- `u256 → u8` - truncation (e.g., for byte extraction)

Explicit casts are performed via bitwise operations (e.g., `evm.and` with a mask) or dedicated cast operations if provided by the frontend.

---

## Type Safety Invariants

The IR must satisfy:

### All values are explicitly typed  
### All pointer ops respect address space rules  
### No composite types appear  
### Control flow must not change types at merge points  
### Frontends must perform layout lowering before EVM‑IR  
### Legalizer enforces canonical form  

---

## Printing and Parsing Format

Examples:

```
%a = evm.add %x : u256, %y : u256
%p = evm.alloca : ptr<0>
%q = evm.ptr_add %p, %offset : ptr<0>
%flag = evm.eq %lhs, %rhs : bool
```

Pointer syntax:

```
ptr<addrspace>
ptr<addrspace, size=N>
```

Examples:
- `ptr<0>` - memory pointer (size unknown)
- `ptr<0, size=32>` - memory pointer to 32-byte region
- `ptr<2>` - calldata pointer
- `ptr<2, size=64>` - calldata pointer to 64-byte region
- Note: Storage uses `u256` keys, not pointers

---

## Future Extensions

Planned optional type extensions:

- typed memory regions
- fat pointers with bounds information
- packed integer types
- internal IR-only struct wrappers for ABI lowering

(Not part of v0.1, but reserved.)

---

## Open Questions for Discussion

### Fat Pointers

**Question:** Should EVM-IR support **fat pointers** that carry both address and bounds information?

#### Current State

Currently, EVM-IR supports:
- Thin pointers: `ptr<addrspace>` or `ptr<addrspace, size=N>` (size is optional type metadata)
- Size metadata is optional and stored in the type, not at runtime

#### Fat Pointer Proposal

Fat pointers would bundle pointer and bounds information together as a runtime value:

```
fat_ptr<addrspace> = (ptr, base, size)
```

Where:
- `ptr` - the actual pointer value
- `base` - base address of the allocation
- `size` - size of the allocated region

#### Discussion Points

**Arguments FOR fat pointers:**
1. **Runtime bounds checking** - enables dynamic bounds verification
2. **Memory safety** - can detect out-of-bounds accesses at runtime
3. **Debugging** - easier to track memory regions during execution
4. **Formal verification** - bounds information available for proof systems
5. **ABI encoding** - could help with dynamic array bounds tracking

**Arguments AGAINST fat pointers:**
1. **EVM doesn't support it** - EVM has no native fat pointer operations
2. **Performance overhead** - storing bounds at runtime uses extra memory/stack slots
3. **Complexity** - adds complexity to stackifier (must handle multi-word values)
4. **Not canonical** - EVM-IR aims to be close to EVM semantics
5. **Alternative approaches** - static analysis can provide bounds checking without runtime cost

#### Alternative Approaches

1. **Static bounds tracking** - Use type-level size metadata (`ptr<0, size=N>`) with static analysis
2. **Separate bounds values** - Keep pointer and size as separate SSA values, not bundled
3. **Hybrid approach** - Support both thin and fat pointers, let frontend choose

#### Questions for Discussion

- Should fat pointers be a first-class type, or just a pattern using multiple SSA values?
- If implemented, should they be optional (opt-in) or required for certain operations?
- How would fat pointers interact with `evm.ptr_add` and pointer arithmetic?
- Would fat pointers survive through the stackifier, or be lowered to separate values?
- Are there specific use cases (e.g., ABI dynamic arrays) that would benefit most?

**Status:** Open for discussion. No decision required for v0.1.

---

## Summary

EVM‑IR's type system:

- provides static analysis benefits  
- ensures deterministic lowering to EVM  
- is compatible with MLIR's SSA model  
- imposes no constraints that would harm optimization  
- is strict enough to prevent undefined IR states  

This provides a complete v0.1 specification of the type system used throughout the EVM‑IR dialect.

---

# EVM‑IR Operation Set Specification (v0.1)

This document defines the **complete operation set** for the EVM‑IR dialect.  
All operations follow the design principles:

- SSA form  
- Deterministic semantics  
- No side effects outside defined address spaces  
- Directly lowerable to canonical EVM bytecode  
- Free of high‑level language semantics  

This is the authoritative definition of all v0.1 ops.

---

## Operation Categories

EVM‑IR operations are grouped as follows:

1. Arithmetic & Bitwise  
2. Comparison  
3. Constants  
4. Memory  
5. Storage  
6. Transient Storage  
7. Calldata  
8. Code Introspection  
9. Pointer Operations  
10. Control Flow  
11. Environment  
12. External Calls  
13. Hashes  
14. Logging  
15. Contract Creation  
16. Self-Destruct  
17. Debug Ops (optional)

---

## Arithmetic Operations

All arithmetic ops operate on `u256` unless specified.

```
%r = evm.add %a, %b : u256
%r = evm.sub %a, %b : u256
%r = evm.mul %a, %b : u256
%r = evm.div %a, %b : u256     // unsigned
%r = evm.sdiv %a, %b : u256    // signed
%r = evm.mod %a, %b : u256
%r = evm.smod %a, %b : u256
%r = evm.addmod %a, %b, %m : u256
%r = evm.mulmod %a, %b, %m : u256
%r = evm.exp %a, %b : u256
```

---

## Bitwise Operations

```
%r = evm.and %a, %b : u256
%r = evm.or %a, %b : u256
%r = evm.xor %a, %b : u256
%r = evm.not %a : u256
%r = evm.shl %shift, %value : u256
%r = evm.shr %shift, %value : u256
%r = evm.sar %shift, %value : u256
```

---

## Comparison Operations

All comparisons produce `bool`.

```
%r = evm.eq %a, %b : bool
%r = evm.lt %a, %b : bool
%r = evm.gt %a, %b : bool
%r = evm.slt %a, %b : bool
%r = evm.sgt %a, %b : bool
```

---

## Constants

```
%v = evm.constant 0x1234 : u256
%v = evm.constant 1 : bool
```

---

## Memory Operations

### Allocation

```
%p = evm.alloca : ptr<0>
%p = evm.alloca <size> : ptr<0, size=<size>>
```

Allocates memory in the linear memory segment.  
- If size is provided, the resulting pointer type includes size metadata: `ptr<0, size=N>`
- If size is omitted, the pointer type is `ptr<0>` (size unknown)
- Offset selection is handled by the stackifier's frame layout phase
- Size metadata helps the stackifier optimize frame layout and enables bounds checking

### Load / Store

```
%r = evm.mload %p : u256
evm.mstore %p, %value : void
```

### Byte operations

```
%r = evm.mload8 %p : u8
evm.mstore8 %p, %v : void
```

---

## Storage Operations

```
%r = evm.sload %key : u256
evm.sstore %key, %value : void
```

Storage keys are `u256`.

---

## Transient Storage (EIP‑1153)

```
%r = evm.tload %key : u256
evm.tstore %key, %value : void
```

---

## Calldata Operations

```
%r = evm.calldataload %ptr : u256
%r = evm.calldatasize : u256
%r = evm.calldatacopy %dst, %src, %len : void
```

---

## Code Introspection

```
%r = evm.codesize : u256
evm.codecopy %dst, %src, %len : void
```

---

## Pointer Operations

```
%q = evm.ptr_add %p, %offset : ptr<AS>
```

`AS` inherited from `%p`.

---

## Control Flow

### Unconditional Branch

```
evm.br ^dest
```

### Conditional Branch

```
evm.condbr %cond, ^if_true, ^if_false
```

### Switch

```
evm.switch %scrut, default ^dflt
    case 0 → ^bb0
    case 1 → ^bb1
```

### Return

```
evm.return %ptr, %len : void
```

Returns data from memory region `[%ptr, %ptr + %len)` to the caller.  
Both `%ptr` and `%len` are `u256` values. The pointer must be a memory pointer (`ptr<0>`).

### Revert

```
evm.revert %ptr, %len : void
```

Reverts execution and returns data from memory region `[%ptr, %ptr + %len)` to the caller.  
Both `%ptr` and `%len` are `u256` values. The pointer must be a memory pointer (`ptr<0>`).

### Unreachable

```
evm.unreachable
```

---

## Environment Operations

```
%a = evm.address : u160
%v = evm.balance %addr : u256
%o = evm.origin : u160
%c = evm.caller : u160
%v = evm.callvalue : u256
%p = evm.gasprice : u256
%g = evm.gas : u256
%b = evm.blockhash %n : u256
%t = evm.timestamp : u256
%n = evm.number : u256
%g = evm.gaslimit : u256
%c = evm.coinbase : u160
%cid = evm.chainid : u256      // EIP-1344
%bf = evm.basefee : u256      // EIP-1559
```

---

## External Calls

```
%success, %retlen = evm.call %gas, %addr, %value, %inptr, %inlen, %outptr, %outlen
%success, %retlen = evm.staticcall %gas, %addr, %inptr, %inlen, %outptr, %outlen
%success, %retlen = evm.delegatecall %gas, %addr, %inptr, %inlen, %outptr, %outlen
```

- `%gas`: `u256` - gas limit for the call
- `%addr`: `u160` - target address
- `%value`: `u256` - ETH value to send (only for `evm.call`)
- `%inptr`: `ptr<0>` - memory pointer to input data
- `%inlen`: `u256` - length of input data
- `%outptr`: `ptr<0>` - memory pointer where return data will be written
- `%outlen`: `u256` - maximum length of return data buffer

Returns:
- `%success`: `bool` - `1` if call succeeded, `0` if it reverted
- `%retlen`: `u256` - actual length of return data written to `%outptr`

Return data is written to memory at `%outptr`. The return type is a tuple lowered into memory in canonical IR.

---

## Hashing

```
%r = evm.keccak256 %ptr, %len : u256
```

Computes Keccak-256 hash of memory region `[%ptr, %ptr + %len)`.  
`%ptr` must be a memory pointer (`ptr<0>`), `%len` is `u256`.

---

## Logging Operations

EVM logging operations emit events:

```
evm.log0 %ptr, %len : void
evm.log1 %ptr, %len, %topic0 : void
evm.log2 %ptr, %len, %topic0, %topic1 : void
evm.log3 %ptr, %len, %topic0, %topic1, %topic2 : void
evm.log4 %ptr, %len, %topic0, %topic1, %topic2, %topic3 : void
```

- `%ptr`: `ptr<0>` - memory pointer to log data
- `%len`: `u256` - length of log data
- `%topic0` through `%topic3`: `u256` - indexed event topics

Logs data from memory region `[%ptr, %ptr + %len)` with up to 4 indexed topics.

---

## Contract Creation

```
%addr = evm.create %value, %ptr, %len : u160
%addr = evm.create2 %value, %ptr, %len, %salt : u160
```

- `%value`: `u256` - ETH value to send to new contract
- `%ptr`: `ptr<0>` - memory pointer to initialization code
- `%len`: `u256` - length of initialization code
- `%salt`: `u256` - salt for CREATE2 (deterministic address)

Returns:
- `%addr`: `u160` - address of created contract, or `0` if creation failed

---

## Self-Destruct

```
evm.selfdestruct %addr : void
```

Destroys the current contract and sends remaining balance to `%addr` (`u160`).

---

## Debug Operations

Optional, ignored during codegen:

```
evm.dbg.trace %value
evm.dbg.label %id
```

---

## Verification Rules

The verifier enforces:

- Correct operand types  
- Correct address space usage  
- Canonical form rules  
- No composite types  
- No implicit pointer arithmetic  
- No user‑level control over stack behavior  

---

## Summary

This file defines the **full EVM‑IR operation set** needed to lower high‑level languages cleanly into a canonical IR before stackification and final codegen.

It is complete for v0.1.

---

# EVM‑IR Legalizer Specification (v0.1)

The **Legalizer** transforms arbitrary well‑typed EVM‑IR into **canonical EVM‑IR**, a restricted form required before stackification and code generation.

The legalizer runs after the frontend emits EVM‑IR but before optimization and stackification.

---

## Purpose of the Legalizer

The legalizer ensures:

- All IR obeys **canonical control flow rules**
- **No PHI nodes** or block arguments remain  
- All control‑flow merges occur through **explicit memory operations**
- All operations are **structurally valid**
- No illegal or undefined patterns reach the backend
- Memory and pointer semantics are correct
- Address space usage is valid
- Switch statements are normalized
- Unreachable blocks are removed
- Frame layout constraints are prepared for stackifier

---

## Canonical Form Requirements

Canonical EVM‑IR must satisfy all rules:

1. **No PHIs** (not allowed in EVM‑IR v0.1)
2. **No block arguments**  
3. **No critical edges**
4. **All branches must target explicit basic blocks**
5. **Each block must end with exactly one terminator**
6. **Switch operations must be normalized**
7. **All pointer arithmetic must be explicit**
8. **Composite types must already be lowered**
9. **All unreachable code eliminated**

If any rule is violated, the legalizer rewrites the IR.

---

## PHI Elimination Strategy (Memory‑Based)

Because EVM is a stack machine, PHI nodes cannot survive past this phase.

Given:

```
%x = phi [ %v1, ^bb1 ], [ %v2, ^bb2 ]
```

The legalizer rewrites as:

```
%slot = evm.alloca : ptr<0>

^bb1:
    evm.mstore %slot, %v1
    evm.br ^merge

^bb2:
    evm.mstore %slot, %v2
    evm.br ^merge

^merge:
    %x = evm.mload %slot : u256
```

All PHIs are eliminated using temporary memory slots.

---

## Switch Normalization

A switch must lower to a sequence of cascaded compares:

```
evm.switch %v, default ^d
  case 0 → ^b0
  case 1 → ^b1
```

Legalizer rewrites to:

```
%is0 = evm.eq %v, %c0
evm.condbr %is0, ^b0, ^test1
^test1:
%is1 = evm.eq %v, %c1
evm.condbr %is1, ^b1, ^d
```

Canonical form has **no multi-case switch ops**.

---

## Control‑Flow Normalization

Rules enforced:

- No critical edges  
- All branches become explicit basic blocks  
- Empty blocks collapsed unless they serve as jump targets  
- Every block must have **one terminator**, not zero or multiple  

Example fix:

```
^bb:
    %c = evm.eq %x, %y
    evm.condbr %c, ^bb_true, ^bb_false
    evm.mstore %p, %x   // illegal after terminator
```

Legalizer moves trailing ops into a new block:

```
^bb:
    %c = evm.eq %x, %y
    evm.condbr %c, ^bb_true, ^bb_fix

^bb_fix:
    evm.mstore %p, %x
    evm.br ^bb_false
```

---

## Unreachable Block Removal

Blocks reached only through `evm.unreachable` or not reached at all are removed.

Algorithm:

1. Build reverse postorder  
2. Mark reachable blocks  
3. Delete all unmarked blocks  
4. Clean up dangling branches  

---

## Memory and Pointer Verification

The legalizer enforces:

- No pointer arithmetic except through `evm.ptr_add`
- Memory pointers must be `ptr<0>`
- Transient storage keys must be `u256` (not pointers)
- Storage keys must be `u256` (not pointers)
- Code pointers must use address space 4
- Calldata pointers must not be mutated

Violations are rewritten or rejected.

---

## Composite Type Elimination

Front-end must eliminate structs/arrays before EVM‑IR.

If composites still appear:

❌ Reject with diagnostic  
or  
✔ Lower via auto‑generated memory layout (frontend recovery mode)

---

## Terminator Repair

Rules:

- Blocks must end in one of:  
  `evm.br`, `evm.condbr`, `evm.return`, `evm.revert`, `evm.unreachable`
- If block ends in a non‑terminator op, legalizer inserts:
  `evm.unreachable`

Example repair:

```
^bb:
    evm.mstore %p, %v
```

becomes:

```
evm.mstore %p, %v
evm.unreachable
```

---

## Legalizer Processing Order

1. Validate IR invariants  
2. Eliminate PHI nodes  
3. Normalize switch ops  
4. Normalize control flow  
5. Canonicalize terminators  
6. Remove unreachable blocks  
7. Enforce type and pointer invariants  
8. Re-run canonicalization if necessary  

The legalizer may run multiple rounds until a fixpoint is reached.

---

## Examples

### Example 1 — PHI removal

Input (pre-legalizer IR with block arguments):

```
^entry:
    %v1 = evm.constant 10 : u256
    %v2 = evm.constant 20 : u256
    evm.condbr %c, ^a, ^b
^a:
    evm.br ^merge(%v1)
^b:
    evm.br ^merge(%v2)
^merge(%x : u256):
    evm.return %x : u256
```

Output (canonical form after legalizer):

```
%slot = evm.alloca : ptr<0>

^entry:
    %v1 = evm.constant 10 : u256
    %v2 = evm.constant 20 : u256
    evm.condbr %c, ^a, ^b

^a:
    evm.mstore %slot, %v1 : void
    evm.br ^merge

^b:
    evm.mstore %slot, %v2 : void
    evm.br ^merge

^merge:
    %x = evm.mload %slot : u256
    evm.return %x : u256
```

The legalizer eliminates the block argument `%x` by:
1. Allocating a temporary memory slot
2. Each predecessor stores its value to the slot
3. The merge block loads from the slot

---

## Summary

The legalizer enforces:

- **No PHIs**  
- **Fully explicit control-flow**  
- **Canonical memory and pointer operations**  
- **Normalized switch structures**  
- **Safe and deterministic IR for stackifier**  

This is the final IR form allowed into the EVM‑IR optimization and stackification phases.

---

# EVM‑IR Stackifier Specification (v0.1)

The **Stackifier** lowers canonical EVM‑IR (SSA form) into a **linear stack‑machine representation** suitable for direct EVM bytecode emission.  
It is the *most critical* backend phase: it removes SSA, eliminates registers, assigns memory frame locations, schedules stack operations, and linearizes control flow.

This document defines the complete stackifier pipeline, algorithms, constraints, and required invariants.

---

## Purpose of the Stackifier

The stackifier transforms canonical EVM‑IR into a form in which:

- All SSA values become **stack values**, **memory slots**, or **immediate constants**
- Control flow is linearized into blocks with explicit jump destinations
- All IR operations are converted into **EVM stack instructions** (`PUSH`, `DUP`, `SWAP`, arithmetic ops, etc.)
- All local variables are assigned offsets in a single **stack frame**
- The resulting representation is suitable for direct bytecode generation

---

## Stack IR Model

After stackification, each basic block is lowered to a sequence:

```
<JUMPDEST label>
<STACK OPS>
<EVM INSTRUCTIONS>
<JUMP or RETURN or REVERT>
```

This representation is *not* SSA.

### Stack IR Value Kinds

A value in Stack IR can be:

- **StackSlot(N)** — lives at depth N in the EVM stack  
- **MemorySlot(offset)** — spilled to memory  
- **Immediate(value)** — generated via PUSH  
- **Void** — operations like store or branch

---

## Pipeline Overview

The stackifier runs in 6 stages:

1. **Block Linearization**  
2. **Lifetime Analysis**  
3. **Frame Layout Allocation**  
4. **SSA to Stack Value Assignment**  
5. **Instruction Scheduling & Stack Shaping**  
6. **Terminator Lowering**  

Each stage is detailed below.

---

## Stage 1 — Block Linearization

Canonical IR contains branches between basic blocks; the stackifier:

### Assigns numeric labels
Every block receives a deterministic integer label:

```
^entry → L0
^loop  → L1
^exit  → L2
```

### Constructs a linear order
A depth‑first ordering is used unless critical for performance.  
Jump destinations correspond 1‑to‑1 with labels.

---

## Stage 2 — Lifetime Analysis

The goal is to classify SSA values as:

1. **Stack‑resident**
2. **Memory‑resident**
3. **Immediate**

### Stack‑resident rules
A value may remain on the stack if:

- It is used in LIFO order
- It does not escape to another block  
- It does not have conflicting lifetime with later values

### Memory‑resident rules
Values must be spilled to memory if:

- They are used across blocks  
- They are live at merge points  
- They appear in loops with backward edges  
- They interfere with other stack values in a way that cannot be resolved by DUP/SWAP efficiently

### Immediate rules
Constants become `PUSH` instructions.

---

## Stage 3 — Frame Layout Allocation

The stack frame is a region of linear memory where the stackifier emits `evm.alloca` slots.

Algorithm:

1. Walk IR and collect all memory allocations  
2. Assign each a static offset  
3. Merge non‑overlapping lifetimes to reuse space (optional optimization)  
4. Emit base pointer (usually 0)  
5. Translate:
   ```
   evm.alloca : ptr<0>
   ```
   into:
   ```
   MemorySlot(offset)
   ```

The frame grows monotonically.

---

## Stage 4 — SSA Value Assignment

Each SSA value is assigned one of:

- **Kept on the stack**  
- **Spilled to memory**  
- **Reconstructed from memory when needed**

### Stack slots
The top of the stack (TOS) corresponds to the most recently produced value.

Example:

```
%x = evm.add %a, %b
```

Becomes:

```
<load a>     // push
<load b>     // push
ADD          // pops 2, pushes 1
```

### Memory spills
If `%x` is spilled:

```
<compute x>
<store x to memory slot S>
```

Later loads reintroduce it:

```
LOAD S
```

---

## Stage 5 — Instruction Scheduling & Stack Shaping

This is the core of stackification.

### Stack Discipline

The stackifier must insert:

- `DUPn` to duplicate stack values required later  
- `SWAPn` to reorder stack for correct operand positions  
- `POP` to discard unused results  
- `PUSH` for immediates  
- `MLOAD` / `MSTORE` for memory traffic  

### Operand Ordering

Given an op:

```
%r = evm.sub %a, %b
```

Stack order must be:

```
push a
push b
SUB
```

The stackifier computes if:

- `a` and `b` are on the stack
- `DUP` or `SWAP` can expose them
- Memory loads are needed

### Stack Shaping Examples

#### Example: simple arithmetic

SSA:

```
%t1 = evm.add %x, %y
%t2 = evm.mul %t1, %z
```

Stack IR:

```
PUSH x
PUSH y
ADD
PUSH z
MUL
```

Redundant DUPs avoided by stack checker.

#### Example: spilled value

```
<t1 computed>
STORE [slot0]
...
LOAD [slot0]
```

---

## Stage 6 — Terminator Lowering

### Branches

```
evm.br ^L1
→
JUMP to L1
```

### Conditional branches

```
evm.condbr %cond, ^L1, ^L2
```

Lowered:

```
<compute cond>
PUSH L1
JUMPI
PUSH L2
JUMP
```

### Returns

```
evm.return %value
```

→

```
<value on stack>
RETURN
```

### Reverts

```
evm.revert %ptr, %len
```

→

```
<ptr>
<len>
REVERT
```

### Unreachable

```
evm.unreachable
```

→

```
INVALID
```

---

## Stackifier Invariants

The resulting STACK‑IR must satisfy:

1. Stack height never negative  
2. Stack height remains ≤ 1024 **per EVM rules**  
3. No uninitialized stack values  
4. No dead stack state before terminators  
5. All computed values eventually consumed  
6. All JUMP destinations correspond to valid labels  
7. All memory offsets constant-folded

---

## Examples

### Example 1 — Basic function

EVM‑IR:

```
%x = evm.add %a, %b
evm.return %x
```

Stack IR:

```
PUSH a
PUSH b
ADD
RETURN
```

---

### Example 2 — If/Else

EVM‑IR:

```
%c = evm.eq %x, %y
evm.condbr %c, ^then, ^else
^then:
    evm.return %a
^else:
    evm.return %b
```

Stack IR:

```
; entry
PUSH x
PUSH y
EQ
PUSH L_then
JUMPI
PUSH L_else
JUMP

L_then:
PUSH a
RETURN

L_else:
PUSH b
RETURN
```

---

## Summary

The stackifier:

- Removes SSA  
- Assigns all values to stack or memory  
- Eliminates PHIs (legalizer already removes)  
- Schedules stack operations  
- Linearizes control flow  
- Produces a ready‑to‑encode stack program  

This is the version‑complete, canonical specification for the Stackifier in EVM‑IR v0.1.

---

# Debug Information Specification (v0.1)

This document defines how **debug information** is represented, preserved, and emitted in EVM‑IR v0.1.  
Debug metadata is optional, zero‑overhead for codegen, and compatible with the emerging **ethdebug** standard.

---

## Purpose of Debug Information

Debug info allows:

- Source‑level debugging  
- Stack traces  
- Breakpoints  
- Stepping through EVM execution  
- Variable inspection (stack/memory/storage)  
- Mapping bytecode back to high‑level code  

Debug metadata **must not** affect semantics or optimization.

---

## Design Constraints

Debug metadata must be:

- **Non‑semantic** — must not change program meaning  
- **Opaque to optimizations** — passes must ignore or preserve it  
- **Recoverable** — must survive lowering phases unless explicitly discarded  
- **Compatible** with ethdebug  
- **Compact** — avoid bloating IR

---

## Debug Metadata Types

EVM‑IR defines the following debug constructs:

1. **Source Location** (`!loc`)  
2. **Scope Information** (`!scope`)  
3. **Variable Metadata** (`!var`)  
4. **Debug Ops** (`evm.dbg.*`)  
5. **Function Metadata** (`!fn`)  
6. **Compilation Unit Metadata** (`!cu`)  

These closely mirror LLVM DI constructs but simplified for EVM.

---

## Source Location Metadata (`!loc`)

Represents positions in original source files:

```
!loc = location(file="contract.ora", line=12, col=5)
```

Attached to any EVM‑IR operation:

```
%v = evm.add %a, %b : u256 loc(!loc)
```

Used by:

- step‑through debugging  
- breakpoint setting  
- source‑level error reporting  

---

## Scope Metadata (`!scope`)

Defines lexical scopes:

```
!scope0 = scope(parent = null)
!scope1 = scope(parent = !scope0)
```

Control:

- stack tracing  
- variable visibility  

A function defines its own top‑level scope.

---

## Variable Metadata (`!var`)

Maps high‑level variables to:

- SSA values  
- stack slots  
- memory slots  
- storage keys  

Format:

```
!v = var(name="balance", type="u256", scope=!scope1)
```

Attached via debug ops or metadata tables.

Frontends must update variable info during lowering.

---

## Debug Operations

Debug ops do not affect semantics and may be removed without changing program meaning.

### `evm.dbg.trace`

Emit a trace event for debugging:

```
evm.dbg.trace %value
```

Lowered to:

- ethdebug event  
- OR internal annotation (ignored by bytecode)

### `evm.dbg.label`

Useful for marking positions:

```
evm.dbg.label "loop_start"
```

---

## Function Metadata (`!fn`)

Describes:

- function name  
- source span  
- parameter info  
- return info  

Example:

```
!f = fn(name="transfer", file="token.ora", line=23, col=1)
```

Used by stack traces.

---

## Compilation Unit Metadata (`!cu`)

Represents:

- version info  
- compiler  
- module origin  
- timestamp  

Example:

```
!cu = cu(language="Ora", producer="ora-compiler 0.1")
```

---

## Mapping Debug Info Through Lowering

### EVM‑IR Legalizer

The legalizer:

- preserves debug metadata  
- duplicates metadata when splitting blocks  
- assigns merged metadata on PHI elimination  

### Stackifier

Stackifier:

- maps SSA values → stack/memory slots  
- emits variable location tables  
- rewrites source locations as bytecode offsets  

### Bytecode Emission (Out of Scope)

Final bytecode:

- emits ETHDEBUG tables  
- maps byte offsets → source locations  
- maps variables → addresses (stack/memory/storage)  
- encodes call frame/scope info  

---

## ETHDEBUG Mapping

EVM‑IR guarantees compatibility with:

https://github.com/ethdebug/format

Fields mapped:

- `line`
- `column`
- `file`
- `offset`
- `length`
- `stack_mapping`
- `memory_mapping`
- `storage_mapping`
- `function_name`
- `scope_chain`

The stackifier helps compute stack and memory mapping tables.

---

## Variable Location Tracking

Variables can live in:

- stack  
- memory  
- storage  
- transient storage  

Rules:

- If spilled → memory tracking entry  
- If SLOAD'd → storage tracking entry  
- If kept on stack → stack-depth entry  
- If copied → alias updated  

---

## Invalidation Rules

Debug info is dropped only when:

- the optimizer proves it is unreachable  
- metadata becomes contradictory  
- operations are folded into constants  

Otherwise it must survive all passes.

---

## Examples

### Example — Annotated Arithmetic

```
!loc_add = location(file="f.ora", line=5, col=12)
%x = evm.add %a, %b loc(!loc_add)
```

### Example — Variable Mapping

```
!v = var(name="counter", type="u256", scope=!scope0)
%slot = evm.alloca
evm.dbg.trace %slot loc(!loc_var)
```

---

## Summary

This debug specification provides:

- full variable tracking  
- source mapping  
- scope representation  
- stepping / tracing capability  
- ethdebug compatibility  
- zero semantic impact  

It completes the debugging infrastructure for EVM‑IR v0.1.

---

# ABI Lowering Specification (v0.1)

This document defines how ABI‑level concepts are lowered into **canonical EVM‑IR**, independent of any high‑level language (Ora, Vyper‑like frontends, custom DSLs).

ABI lowering is the process of handling:

- function dispatch  
- argument decoding  
- return encoding  
- static call‑frame layout  
- fallback / receive behavior  
- error bubbling  
- ABI‑compliant revert payloads  

This is the "boundary layer" between contract execution and the EVM call interface.

---

## Goals

ABI lowering must:

- produce deterministic and portable IR  
- avoid Solidity‑specific semantics  
- handle multi‑entry and single‑entry languages  
- support tuples, structs, and arrays *before* lowering  
- produce IR in canonical form (ready for legalizer + stackifier)  
- interoperate with standard Ethereum tooling  

It must *not* depend on any specific frontend.

---

## ABI Model Overview

EVM‑IR ABI lowering assumes:

1. Every contract has an entrypoint labeled `^entry`.  
2. Calldata begins at offset `0`.  
3. If the contract uses selector‑based dispatch:
   - the first 4 bytes define the selector  
   - arguments follow according to ABI layout  
4. If selectorless (e.g., Vyper, single‑function languages), calldata is decoded directly  
5. Fallback / Receive logic are optional but supported  
6. Return values must be encoded into memory, then returned

---

## Contract Entrypoint

Every contract begins execution at `^entry`.

ABI lowering inserts:

```
^entry:
    %size = evm.calldatasize
    <selector detection and dispatch>
```

The lowering differs depending on:

- selector‑based ABI (Solidity‑style)
- single‑entry ABI (Vyper‑style)
- custom ABI (user‑defined)

We describe the **generalized ABI**, suitable for all.

---

## Function Selector Dispatch

### Extract selector

To extract the 4-byte function selector from calldata:

The frontend must provide a calldata base pointer at offset 0. This is typically done by:
1. Frontend creating a calldata pointer constant at offset 0, or
2. Using a special operation to obtain the calldata base pointer

Example (assuming frontend provides calldata base):
```
%calldata_base = <calldata pointer at offset 0> : ptr<2>
%sig_raw = evm.calldataload %calldata_base : u256   ; loads 32 bytes from offset 0
%shift = evm.constant 224 : u256
%sig = evm.shr %shift, %sig_raw : u256              ; extract top 4 bytes (selector)
```

Note: The exact mechanism for creating calldata pointers is frontend-specific. The key requirement is that `%calldata_base` is a `ptr<2>` pointing to offset 0 in calldata.

### Match selector

Lowering emits:

```
evm.switch %sig, default ^fallback
  case 0x12345678 → ^fn0
  case 0x87654321 → ^fn1
  ...
```

The legalizer rewrites this into normalized conditional branches.

---

## Calldata Decoding

ABI lowering takes each function argument (already laid out by the frontend) and emits:

### Static argument load

```
%offset_const = evm.constant <offset> : u256
%p = evm.ptr_add %calldata_base, %offset_const : ptr<2>
%arg = evm.calldataload %p : u256
```

Where `%calldata_base` is a `ptr<2>` (calldata pointer) at offset 0, and `<offset>` is the byte offset of the argument in calldata (typically 4 + N*32 for the Nth argument after the selector).

### Dynamic arrays and bytes

ABI lowering expands:

```
offset → load offset → load length → copy bytes
```

Everything must eventually be lowered into:

- `evm.calldataload`
- `evm.calldatacopy`
- memory operations

### Structs / Tuples

Structs must be:

- analyzed by the frontend  
- flattened into memory layout  
- expanded into individual loads  

ABI lowering does **not** handle composite types; it only consumes flattened values.

---

## Preparing a Function Frame

Each function lowers into a block:

```
^fn0:
    <argument decoding>
    <body>
```

ABI lowering:

- allocates memory slots for arguments (`evm.alloca`)
- stores decoded args into memory
- loads them into SSA only when needed

Example:

```
%slot0 = evm.alloca
evm.mstore %slot0, %arg0
```

---

## Return Encoding

Return values are written to a contiguous memory region:

```
%retbuf = evm.alloca : ptr<0>
evm.mstore %retbuf, %value : void
%size = evm.constant <size_in_bytes> : u256
evm.return %retbuf, %size : void
```

For multi-value returns, values are laid out sequentially in memory according to ABI encoding rules.

ABI flattening ensures:

- multi‑value returns are laid out tightly
- arrays/bytes include length prefix
- user types are fully pre‑lowered

---

## Revert Payloads

ABI‑compatible revert payload:

```
0x08c379a0 ++ length ++ message
```

ABI lowering emits:

```
%ptr = evm.alloca
evm.mstore %ptr, <error signature>
evm.mstore %ptr+4, <string length>
evm.mstore %ptr+36, <string bytes>
evm.revert %ptr, %len
```

---

## Fallback and Receive

We define language‑neutral semantics:

### Fallback Block

A fallback handler is any block that executes when:

- selector is zero OR
- selector doesn't match any known entry

ABI lowering emits:

```
default → ^fallback
```

### Receive Block

Executed if calldata is empty (`calldatasize = 0`) *and* `callvalue > 0`.

Example:

```
%sz = evm.calldatasize
%val = evm.callvalue
%is_empty = evm.eq %sz, 0
%has_value = evm.gt %val, 0
%cond = evm.and %is_empty, %has_value
evm.condbr %cond, ^receive, ^fallback
```

These are optional for languages that do not define them.

---

## Error Bubbling (Staticcall / Call Return)

When lowering:

```
%success, %retptr = evm.call ...
```

ABI lowering ensures error propagation:

```
evm.condbr %success, ^ok, ^bubble

^bubble:
    ; revert with returned payload
    ; %retptr is the memory pointer where return data was written
    ; %retlen is the length of return data (from evm.call return)
    evm.revert %retptr, %retlen : void
```

---

## Summary

ABI lowering in EVM‑IR v0.1 provides:

- dispatch logic  
- selector parsing  
- argument extraction  
- return encoding  
- revert formatting  
- fallback/receive logic  
- call bubbling  

It acts as the contract boundary layer, ensuring all entrypoints produce canonical EVM‑IR suitable for legalization and stackification.

---

# EVM‑IR Worked Examples (v0.1)

This document contains **concrete, end‑to‑end examples** of EVM‑IR in use:

- direct IR for simple functions
- pre‑legalizer vs canonical IR
- stackification sketches
- ABI lowering and selector dispatch
- storage interaction
- basic conditional logic

These are **illustrative**, not exhaustive, and are intended to guide implementers.

---

## Simple Pure Function: `add(a, b) -> u256`

### High‑Level Intent

```ora
fn add(a: u256, b: u256) -> u256 {
    return a + b;
}
```

### EVM‑IR Function (Canonical Form)

```mlir
func @add(%a : u256, %b : u256) -> u256 {
^entry:
    %sum = evm.add %a, %b : u256
    %retbuf = evm.alloca : ptr<0>
    evm.mstore %retbuf, %sum : void
    %size = evm.constant 32 : u256
    evm.return %retbuf, %size : void
}
```

Notes:

- No memory needed (no spills).
- Single basic block.
- Already canonical.

---

## Storage Read/Write Example

### High‑Level Intent

```ora
// storage slot[0] holds a u256 counter
fn increment() {
    let current = S[0];
    S[0] = current + 1;
}
```

### EVM‑IR

```mlir
func @increment() {
^entry:
    %slot_key = evm.constant 0 : u256
    %cur      = evm.sload %slot_key : u256
    %one      = evm.constant 1 : u256
    %next     = evm.add %cur, %one : u256
    evm.sstore %slot_key, %next : void
    %retbuf = evm.alloca : ptr<0>
    %size = evm.constant 0 : u256
    evm.return %retbuf, %size : void
}
```

Notes:

- Direct storage access via `sload` / `sstore`.
- Keys are modeled as `u256` constants.

---

## Conditional Branching Example

### High‑Level Intent

```ora
fn max(a: u256, b: u256) -> u256 {
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}
```

### EVM‑IR (Pre‑Legalizer, With Block Args)

```mlir
func @max(%a : u256, %b : u256) -> u256 {
^entry:
    %ge = evm.sgt %a, %b : bool   // treating as signed or unsigned per frontend choice
    evm.condbr %ge, ^then(%a), ^else(%b)

^then(%x : u256):
    %retbuf = evm.alloca : ptr<0>
    evm.mstore %retbuf, %x : void
    %size = evm.constant 32 : u256
    evm.return %retbuf, %size : void

^else(%y : u256):
    %retbuf2 = evm.alloca : ptr<0>
    evm.mstore %retbuf2, %y : void
    %size2 = evm.constant 32 : u256
    evm.return %retbuf2, %size2 : void
}
```

### EVM‑IR (Canonical Form After Legalizer)

Legalizer eliminates block arguments and PHIs via memory:

```mlir
func @max(%a : u256, %b : u256) -> u256 {
^entry:
    %slot = evm.alloca : ptr<0>

    %ge = evm.sgt %a, %b : bool
    evm.condbr %ge, ^then, ^else

^then:
    evm.mstore %slot, %a : void
    evm.br ^merge

^else:
    evm.mstore %slot, %b : void
    evm.br ^merge

^merge:
    %result = evm.mload %slot : u256
    %retbuf = evm.alloca : ptr<0>
    evm.mstore %retbuf, %result : void
    %size = evm.constant 32 : u256
    evm.return %retbuf, %size : void
}
```

---

## ABI‑Lowered Function With Selector Dispatch

### High‑Level Intent

```ora
// External function:
// fn setValue(x: u256)
```

Assume ABI function signature:

- name: `setValue`
- selector: `0x55241077` (example)

### EVM‑IR Module Skeleton

```mlir
module {
  func @__entry() {
  ^entry:
      %cd_size = evm.calldatasize : u256

      // load selector word - frontend provides calldata base pointer at offset 0
      %calldata_base = <calldata pointer at offset 0> : ptr<2>
      %word0 = evm.calldataload %calldata_base : u256
      %shift = evm.constant 224 : u256
      %sel = evm.shr %shift, %word0 : u256

      evm.switch %sel, default ^fallback
        case 0x55241077 : ^dispatch_setValue

  ^dispatch_setValue:
      // args start at offset 4 bytes
      %arg_off = evm.constant 4 : u256
      %arg_ptr = evm.ptr_add %calldata_base, %arg_off : ptr<2>
      %arg0 = evm.calldataload %arg_ptr : u256

      // call body with arg0 (pre-legalizer: uses block argument)
      evm.br ^setValue_body(%arg0)

  ^fallback:
      // unknown selector → revert
      %retbuf = evm.alloca : ptr<0>
      %size = evm.constant 0 : u256
      evm.revert %retbuf, %size : void
  }

  // the logical function body (pre-legalizer form with block argument)
  func @setValue_body(%x : u256) {
  ^entry:
      // Implementation goes here; for example, store into slot 0:
      %slot = evm.constant 0 : u256
      evm.sstore %slot, %x : void
      %retbuf = evm.alloca : ptr<0>
      %size = evm.constant 0 : u256
      evm.return %retbuf, %size : void
  }
}
```

Legalizer will remove block arguments as in previous examples.

---

## Receive / Fallback Scenario

### Intention

Contract wants:

- If `calldata` is empty and `callvalue > 0` → execute receive handler
- Else if selector matches known function → dispatch
- Else → fallback revert

### EVM‑IR

```mlir
func @__entry() {
^entry:
    %cd_size = evm.calldatasize : u256
    %val     = evm.callvalue : u256
    %zero    = evm.constant 0 : u256

    %is_empty  = evm.eq %cd_size, %zero : bool
    %has_value = evm.sgt %val, %zero : bool
    %do_recv   = evm.and %is_empty, %has_value : bool

    evm.condbr %do_recv, ^receive, ^dispatch_or_fallback

^receive:
    // simple receive logic (e.g., log)
    %retbuf = evm.alloca : ptr<0>
    %size = evm.constant 0 : u256
    evm.return %retbuf, %size : void

^dispatch_or_fallback:
    %calldata_base = <calldata pointer at offset 0> : ptr<2>
    %word0 = evm.calldataload %calldata_base : u256
    %shift = evm.constant 224 : u256
    %sel = evm.shr %shift, %word0 : u256

    evm.switch %sel, default ^fallback
      case 0x12345678 : ^dispatch_fn

^dispatch_fn:
    // decode arguments, call function body
    evm.br ^fn_body

^fn_body:
    // ...
    %retbuf = evm.alloca : ptr<0>
    %size = evm.constant 0 : u256
    evm.return %retbuf, %size : void

^fallback:
    %retbuf = evm.alloca : ptr<0>
    %size = evm.constant 0 : u256
    evm.revert %retbuf, %size : void
}
```

---

## Simple Example With Transient Storage (EIP‑1153)

### High‑Level Intent

```ora
// Use transient storage as a per‑call scratch space
fn tempCounter() {
    let key = 0;
    let cur = T[key]; // transient
    T[key] = cur + 1;
}
```

### EVM‑IR

```mlir
func @tempCounter() {
^entry:
    %key  = evm.constant 0 : u256
    %cur  = evm.tload %key : u256
    %one  = evm.constant 1 : u256
    %next = evm.add %cur, %one : u256
    evm.tstore %key, %next : void
    %retbuf = evm.alloca : ptr<0>
    %size = evm.constant 0 : u256
    evm.return %retbuf, %size : void
}
```

---

## From Canonical EVM‑IR to Stack‑Shaped Sketch

Using the earlier `max` function's canonical IR:

```mlir
func @max(%a : u256, %b : u256) -> u256 {
^entry:
    %slot = evm.alloca : ptr<0>

    %ge = evm.sgt %a, %b : bool
    evm.condbr %ge, ^then, ^else

^then:
    evm.mstore %slot, %a : void
    evm.br ^merge

^else:
    evm.mstore %slot, %b : void
    evm.br ^merge

^merge:
    %result = evm.mload %slot : u256
    %retbuf = evm.alloca : ptr<0>
    evm.mstore %retbuf, %result : void
    %size = evm.constant 32 : u256
    evm.return %retbuf, %size : void
}
```

A sketch of the corresponding stack‑shaped program:

```text
; assume a, b are loaded appropriately as inputs

; entry:
PUSH a
PUSH b
SGT               ; stack: [cond]
PUSH L_then
JUMPI
PUSH L_else
JUMP

L_then:
PUSH a
MSTORE [slot]     ; store a to memory
PUSH L_merge
JUMP

L_else:
PUSH b
MSTORE [slot]
PUSH L_merge
JUMP

L_merge:
MLOAD [slot]      ; push result
MSTORE [retbuf]   ; store result to return buffer
PUSH 32           ; return size
PUSH [retbuf]     ; return pointer
RETURN
```

(Exact instruction ordering is determined by the stackifier and frame layout logic.)

---

## Summary

This examples section demonstrates:

- How high‑level semantics map to EVM‑IR
- How the legalizer rewrites PHI‑like patterns into memory
- How ABI entrypoints (selectors, fallback, receive) look in IR
- How transient storage is modeled
- How stackification conceptually transforms IR to a stack machine

These examples can be expanded into executable tests in the EVM‑IR test suite and used as reference patterns for frontend implementers.

---

# Appendix (v0.1)

This appendix contains supporting material for the EVM‑IR v0.1 specification.

---

## A. Address Space Table

| ID | Name        | Description                          |
|----|-------------|--------------------------------------|
| 0  | memory      | Linear temporary memory              |
| 1  | storage     | Persistent key–value storage         |
| 2  | calldata    | Read‑only external call buffer       |
| 3  | transient   | EIP‑1153 transient storage           |
| 4  | code        | Immutable contract bytecode region   |

---

## B. Naming Conventions

- Functions use `func @name`
- Blocks use `^label`
- Types lowercase: `u256`, `u160`, `bool`
- IR ops prefixed with `evm.`

---

## C. Minimal Grammar Sketch

```
module      ::= { func }
func        ::= "func" "@" ident "(" args ")" ("->" type)? block
args        ::= [ arg { "," arg } ]
arg         ::= "%" ident ":" type
block       ::= { label ":" inst* }
inst        ::= result "=" op operands | op operands
result      ::= "%" ident
op          ::= ident
operands    ::= operand { "," operand }
operand     ::= result | const | ptr
```

---

## D. Canonical Form Summary

1. All blocks must end with a terminator  
2. No PHI nodes (replaced by memory merges)  
3. All memory allocations hoisted  
4. Switch lowered to structured branches  
5. No composite values in SSA  
6. Pure SSA for arithmetic/logical ops  
7. Control flow explicit  

---

## E. Future Work (Non‑Normative)

- gas metadata  
- optimization passes  
- symbolic interpreter  

---

## F. Open Questions

For open design questions and discussion topics, see:

- **Fat Pointers** - See the "Open Questions for Discussion" section in the Type System chapter  
  Discussion about whether EVM-IR should support fat pointers with runtime bounds information.

---

## G. Change Log (v0.1)

Initial complete specification and IR definition.

