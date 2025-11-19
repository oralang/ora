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

# Design Principles

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

# Primitive Types

## `u256`
- Arbitrary EVM stack word  
- Represents unsigned integers modulo 2²⁵⁶  
- Used for:
  - arithmetic
  - indexing
  - storage keys
  - ABI values

This is the *dominant* type in EVM‑IR.

## `u160`
- 160‑bit value  
- Canonical representation for EVM addresses  
- Codegen guarantees zero‑extension to 256 bits when needed.

## `bool`
- Logical value restricted to `{0, 1}`  
- Codegen enforces normalization when necessary  
- Represented as a 256‑bit stack value where `0 = false`, `1 = true`

## `u32`, `u8`
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

# Pointer Types

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

## Address Space Table

| Address Space | Meaning | EVM backing | Notes |
|--------------|---------|--------------|-------|
| `0` | Memory | `MLOAD / MSTORE` | Byte-addressable, uses `ptr<0>` |
| `1` | Storage | `SLOAD / SSTORE` | Uses `u256` keys, not pointers |
| `2` | Calldata | `CALLDATALOAD` | Read-only, uses `ptr<2>` |
| `3` | Transient | `TLOAD / TSTORE` | EIP-1153, uses `u256` keys |
| `4` | Code | `CODESIZE / CODECOPY` | Read-only code region |

## Memory Pointers
- represent byte-addressable memory addresses  
- arithmetic permitted via `evm.ptr_add`
- EVM memory operations (`MLOAD`/`MSTORE`) can operate on any byte offset (alignment not enforced by EVM)
- Size metadata (when present) helps with frame layout and bounds checking

Example:
```
%p = evm.alloca 32 : ptr<0, size=32>  // allocates 32 bytes, pointer tracks size
%q = evm.ptr_add %p, %offset : ptr<0>  // result pointer may not preserve size
```

## Storage Keys
- Storage uses `u256` keys, not pointers  
- Storage keys are represented as `u256` values, not `ptr<1>`  
- Arithmetic must be expressed explicitly  
- Compiler uses Keccak when lowering composite offsets

## Calldata Pointers
Represent byte offsets into calldata buffer.

## Transient Storage Keys
- Transient storage uses `u256` keys, not pointers  
- Transient storage keys are represented as `u256` values, not `ptr<3>`  
- For EIP‑1153 transient storage operations

## Code Pointers
Represent offsets into the contract code region.

---

# Special Types

## `void`
Used for:
- operations that do not produce SSA values
- pure side‑effect operations

## `unreachable`
Represents IR regions that cannot execute.  
Used by:
- legalizer
- control‑flow verification
- optimization passes

---

# Type Rules

## Arithmetic
- `u256 op u256 → u256`
- Narrow types (`u8`, `u32`, `u160`) **automatically zero-extend** to `u256` when used in arithmetic operations
- `bool` is treated as `u256` during arithmetic but must be re‑normalized before branch conditions
- Operations like `evm.mload8` return `u8`, which is then zero-extended to `u256` when used in subsequent operations

## Comparisons
Comparisons always produce `bool`.

```
u256 < u256 → bool
u256 == u256 → bool
...
```

## Pointers
Pointer arithmetic requires explicit operations:

```
%p2 = evm.ptr_add %p1, %offset
```

No implicit pointer addition.

## Type Casting

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

# Type Safety Invariants

The IR must satisfy:

### All values are explicitly typed  
### All pointer ops respect address space rules  
### No composite types appear  
### Control flow must not change types at merge points  
### Frontends must perform layout lowering before EVM‑IR  
### Legalizer enforces canonical form  

---

# Printing and Parsing Format

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

# Future Extensions

Planned optional type extensions:

- typed memory regions
- fat pointers with bounds information
- packed integer types
- internal IR-only struct wrappers for ABI lowering

(Not part of v0.1, but reserved.)

---

# Open Questions for Discussion

## Fat Pointers

**Question:** Should EVM-IR support **fat pointers** that carry both address and bounds information?

### Current State

Currently, EVM-IR supports:
- Thin pointers: `ptr<addrspace>` or `ptr<addrspace, size=N>` (size is optional type metadata)
- Size metadata is optional and stored in the type, not at runtime

### Fat Pointer Proposal

Fat pointers would bundle pointer and bounds information together as a runtime value:

```
fat_ptr<addrspace> = (ptr, base, size)
```

Where:
- `ptr` - the actual pointer value
- `base` - base address of the allocation
- `size` - size of the allocated region

### Discussion Points

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

### Alternative Approaches

1. **Static bounds tracking** - Use type-level size metadata (`ptr<0, size=N>`) with static analysis
2. **Separate bounds values** - Keep pointer and size as separate SSA values, not bundled
3. **Hybrid approach** - Support both thin and fat pointers, let frontend choose

### Questions for Discussion

- Should fat pointers be a first-class type, or just a pattern using multiple SSA values?
- If implemented, should they be optional (opt-in) or required for certain operations?
- How would fat pointers interact with `evm.ptr_add` and pointer arithmetic?
- Would fat pointers survive through the stackifier, or be lowered to separate values?
- Are there specific use cases (e.g., ABI dynamic arrays) that would benefit most?

**Status:** Open for discussion. No decision required for v0.1.

---

# Summary

EVM‑IR’s type system:

- provides static analysis benefits  
- ensures deterministic lowering to EVM  
- is compatible with MLIR’s SSA model  
- imposes no constraints that would harm optimization  
- is strict enough to prevent undefined IR states  

This file provides a complete v0.1 specification of the type system used throughout the EVM‑IR dialect.
