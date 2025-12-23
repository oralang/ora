# EVM‑IR Legalizer Specification (v0.1)

The **Legalizer** transforms arbitrary well‑typed EVM‑IR into **canonical EVM‑IR**, a restricted form required before stackification and code generation.

The legalizer runs after the frontend emits EVM‑IR but before optimization and stackification.

---

# Purpose of the Legalizer

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

# Canonical Form Requirements

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

# PHI Elimination Strategy (Memory‑Based)

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

# Switch Normalization

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

# Control‑Flow Normalization

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

# Unreachable Block Removal

Blocks reached only through `evm.unreachable` or not reached at all are removed.

Algorithm:

1. Build reverse postorder  
2. Mark reachable blocks  
3. Delete all unmarked blocks  
4. Clean up dangling branches  

---

# Memory and Pointer Verification

The legalizer enforces:

- No pointer arithmetic except through `evm.ptr_add`
- Memory pointers must be `ptr<0>`
- Transient storage keys must be `u256` (not pointers)
- Storage keys must be `u256` (not pointers)
- Code pointers must use address space 4
- Calldata pointers must not be mutated

Violations are rewritten or rejected.

---

# Composite Type Elimination

Front-end must eliminate structs/arrays before EVM‑IR.

If composites still appear:

❌ Reject with diagnostic  
or  
✔ Lower via auto‑generated memory layout (frontend recovery mode)

---

# Terminator Repair

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

# Legalizer Processing Order

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

# Examples

## Example 1 — PHI removal

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

# Summary

The legalizer enforces:

- **No PHIs**  
- **Fully explicit control-flow**  
- **Canonical memory and pointer operations**  
- **Normalized switch structures**  
- **Safe and deterministic IR for stackifier**  

This is the final IR form allowed into the EVM‑IR optimization and stackification phases.
