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

Using the earlier `max` function’s canonical IR:

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
