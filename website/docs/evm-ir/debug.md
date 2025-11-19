# Debug Information Specification (v0.1)

This document defines how **debug information** is represented, preserved, and emitted in EVM‑IR v0.1.  
Debug metadata is optional, zero‑overhead for codegen, and compatible with the emerging **ethdebug** standard.

---

# Purpose of Debug Information

Debug info allows:

- Source‑level debugging  
- Stack traces  
- Breakpoints  
- Stepping through EVM execution  
- Variable inspection (stack/memory/storage)  
- Mapping bytecode back to high‑level code  

Debug metadata **must not** affect semantics or optimization.

---

# Design Constraints

Debug metadata must be:

- **Non‑semantic** — must not change program meaning  
- **Opaque to optimizations** — passes must ignore or preserve it  
- **Recoverable** — must survive lowering phases unless explicitly discarded  
- **Compatible** with ethdebug  
- **Compact** — avoid bloating IR

---

# Debug Metadata Types

EVM‑IR defines the following debug constructs:

1. **Source Location** (`!loc`)  
2. **Scope Information** (`!scope`)  
3. **Variable Metadata** (`!var`)  
4. **Debug Ops** (`evm.dbg.*`)  
5. **Function Metadata** (`!fn`)  
6. **Compilation Unit Metadata** (`!cu`)  

These closely mirror LLVM DI constructs but simplified for EVM.

---

# Source Location Metadata (`!loc`)

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

# Scope Metadata (`!scope`)

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

# Variable Metadata (`!var`)

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

# Debug Operations

Debug ops do not affect semantics and may be removed without changing program meaning.

## `evm.dbg.trace`

Emit a trace event for debugging:

```
evm.dbg.trace %value
```

Lowered to:

- ethdebug event  
- OR internal annotation (ignored by bytecode)

## `evm.dbg.label`

Useful for marking positions:

```
evm.dbg.label "loop_start"
```

---

# Function Metadata (`!fn`)

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

# Compilation Unit Metadata (`!cu`)

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

# Mapping Debug Info Through Lowering

## EVM‑IR Legalizer

The legalizer:

- preserves debug metadata  
- duplicates metadata when splitting blocks  
- assigns merged metadata on PHI elimination  

## Stackifier

Stackifier:

- maps SSA values → stack/memory slots  
- emits variable location tables  
- rewrites source locations as bytecode offsets  

## Bytecode Emission (Out of Scope)

Final bytecode:

- emits ETHDEBUG tables  
- maps byte offsets → source locations  
- maps variables → addresses (stack/memory/storage)  
- encodes call frame/scope info  

---

# ETHDEBUG Mapping

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

# Variable Location Tracking

Variables can live in:

- stack  
- memory  
- storage  
- transient storage  

Rules:

- If spilled → memory tracking entry  
- If SLOAD’d → storage tracking entry  
- If kept on stack → stack-depth entry  
- If copied → alias updated  

---

# Invalidation Rules

Debug info is dropped only when:

- the optimizer proves it is unreachable  
- metadata becomes contradictory  
- operations are folded into constants  

Otherwise it must survive all passes.

---

# Examples

## Example — Annotated Arithmetic

```
!loc_add = location(file="f.ora", line=5, col=12)
%x = evm.add %a, %b loc(!loc_add)
```

## Example — Variable Mapping

```
!v = var(name="counter", type="u256", scope=!scope0)
%slot = evm.alloca
evm.dbg.trace %slot loc(!loc_var)
```

---

# Summary

This debug specification provides:

- full variable tracking  
- source mapping  
- scope representation  
- stepping / tracing capability  
- ethdebug compatibility  
- zero semantic impact  

It completes the debugging infrastructure for EVM‑IR v0.1.
