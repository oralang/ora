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

This is the “boundary layer” between contract execution and the EVM call interface.

---

# Goals

ABI lowering must:

- produce deterministic and portable IR  
- avoid Solidity‑specific semantics  
- handle multi‑entry and single‑entry languages  
- support tuples, structs, and arrays *before* lowering  
- produce IR in canonical form (ready for legalizer + stackifier)  
- interoperate with standard Ethereum tooling  

It must *not* depend on any specific frontend.

---

# ABI Model Overview

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

# Contract Entrypoint

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

# Function Selector Dispatch

## Extract selector

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

## Match selector

Lowering emits:

```
evm.switch %sig, default ^fallback
  case 0x12345678 → ^fn0
  case 0x87654321 → ^fn1
  ...
```

The legalizer rewrites this into normalized conditional branches.

---

# Calldata Decoding

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

# Preparing a Function Frame

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

# Return Encoding

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

# Revert Payloads

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

# Fallback and Receive

We define language‑neutral semantics:

### Fallback Block

A fallback handler is any block that executes when:

- selector is zero OR
- selector doesn’t match any known entry

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

# Error Bubbling (Staticcall / Call Return)

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

# Summary

ABI lowering in EVM‑IR v0.1 provides:

- dispatch logic  
- selector parsing  
- argument extraction  
- return encoding  
- revert formatting  
- fallback/receive logic  
- call bubbling  

It acts as the contract boundary layer, ensuring all entrypoints produce canonical EVM‑IR suitable for legalization and stackification.
