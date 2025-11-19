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

# Operation Categories

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

# Arithmetic Operations

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

# Bitwise Operations

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

# Comparison Operations

All comparisons produce `bool`.

```
%r = evm.eq %a, %b : bool
%r = evm.lt %a, %b : bool
%r = evm.gt %a, %b : bool
%r = evm.slt %a, %b : bool
%r = evm.sgt %a, %b : bool
```

---

# Constants

```
%v = evm.constant 0x1234 : u256
%v = evm.constant 1 : bool
```

---

# Memory Operations

## Allocation

```
%p = evm.alloca : ptr<0>
%p = evm.alloca <size> : ptr<0, size=<size>>
```

Allocates memory in the linear memory segment.  
- If size is provided, the resulting pointer type includes size metadata: `ptr<0, size=N>`
- If size is omitted, the pointer type is `ptr<0>` (size unknown)
- Offset selection is handled by the stackifier's frame layout phase
- Size metadata helps the stackifier optimize frame layout and enables bounds checking

## Load / Store

```
%r = evm.mload %p : u256
evm.mstore %p, %value : void
```

## Byte operations

```
%r = evm.mload8 %p : u8
evm.mstore8 %p, %v : void
```

---

# Storage Operations

```
%r = evm.sload %key : u256
evm.sstore %key, %value : void
```

Storage keys are `u256`.

---

# Transient Storage (EIP‑1153)

```
%r = evm.tload %key : u256
evm.tstore %key, %value : void
```

---

# Calldata Operations

```
%r = evm.calldataload %ptr : u256
%r = evm.calldatasize : u256
%r = evm.calldatacopy %dst, %src, %len : void
```

---

# Code Introspection

```
%r = evm.codesize : u256
evm.codecopy %dst, %src, %len : void
```

---

# Pointer Operations

```
%q = evm.ptr_add %p, %offset : ptr<AS>
```

`AS` inherited from `%p`.

---

# Control Flow

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

# Environment Operations

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

# External Calls

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

# Hashing

```
%r = evm.keccak256 %ptr, %len : u256
```

Computes Keccak-256 hash of memory region `[%ptr, %ptr + %len)`.  
`%ptr` must be a memory pointer (`ptr<0>`), `%len` is `u256`.

---

# Logging Operations

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

# Contract Creation

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

# Self-Destruct

```
evm.selfdestruct %addr : void
```

Destroys the current contract and sends remaining balance to `%addr` (`u160`).

---

# Debug Operations

Optional, ignored during codegen:

```
evm.dbg.trace %value
evm.dbg.label %id
```

---

# Verification Rules

The verifier enforces:

- Correct operand types  
- Correct address space usage  
- Canonical form rules  
- No composite types  
- No implicit pointer arithmetic  
- No user‑level control over stack behavior  

---

# Summary

This file defines the **full EVM‑IR operation set** needed to lower high‑level languages cleanly into a canonical IR before stackification and final codegen.

It is complete for v0.1.
