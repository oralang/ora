# EVM‑IR Stackifier Specification (v0.1)

The **Stackifier** lowers canonical EVM‑IR (SSA form) into a **linear stack‑machine representation** suitable for direct EVM bytecode emission.  
It is the *most critical* backend phase: it removes SSA, eliminates registers, assigns memory frame locations, schedules stack operations, and linearizes control flow.

This document defines the complete stackifier pipeline, algorithms, constraints, and required invariants.

---

# Purpose of the Stackifier

The stackifier transforms canonical EVM‑IR into a form in which:

- All SSA values become **stack values**, **memory slots**, or **immediate constants**
- Control flow is linearized into blocks with explicit jump destinations
- All IR operations are converted into **EVM stack instructions** (`PUSH`, `DUP`, `SWAP`, arithmetic ops, etc.)
- All local variables are assigned offsets in a single **stack frame**
- The resulting representation is suitable for direct bytecode generation

---

# Stack IR Model

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

# Pipeline Overview

The stackifier runs in 6 stages:

1. **Block Linearization**  
2. **Lifetime Analysis**  
3. **Frame Layout Allocation**  
4. **SSA to Stack Value Assignment**  
5. **Instruction Scheduling & Stack Shaping**  
6. **Terminator Lowering**  

Each stage is detailed below.

---

# Stage 1 — Block Linearization

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

# Stage 2 — Lifetime Analysis

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

# Stage 3 — Frame Layout Allocation

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

# Stage 4 — SSA Value Assignment

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

# Stage 5 — Instruction Scheduling & Stack Shaping

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

# Stage 6 — Terminator Lowering

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

# Stackifier Invariants

The resulting STACK‑IR must satisfy:

1. Stack height never negative  
2. Stack height remains ≤ 1024 **per EVM rules**  
3. No uninitialized stack values  
4. No dead stack state before terminators  
5. All computed values eventually consumed  
6. All JUMP destinations correspond to valid labels  
7. All memory offsets constant-folded

---

# Examples

## Example 1 — Basic function

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

## Example 2 — If/Else

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

# Summary

The stackifier:

- Removes SSA  
- Assigns all values to stack or memory  
- Eliminates PHIs (legalizer already removes)  
- Schedules stack operations  
- Linearizes control flow  
- Produces a ready‑to‑encode stack program  

This is the version‑complete, canonical specification for the Stackifier in EVM‑IR v0.1.
