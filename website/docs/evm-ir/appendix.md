# Appendix (v0.1)

This appendix contains supporting material for the EVM‑IR v0.1 specification.

---

# A. Address Space Table

| ID | Name        | Description                          |
|----|-------------|--------------------------------------|
| 0  | memory      | Linear temporary memory              |
| 1  | storage     | Persistent key–value storage         |
| 2  | calldata    | Read‑only external call buffer       |
| 3  | transient   | EIP‑1153 transient storage           |
| 4  | code        | Immutable contract bytecode region   |

---

# B. Naming Conventions

- Functions use `func @name`
- Blocks use `^label`
- Types lowercase: `u256`, `u160`, `bool`
- IR ops prefixed with `evm.`

---

# C. Minimal Grammar Sketch

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

# D. Canonical Form Summary

1. All blocks must end with a terminator  
2. No PHI nodes (replaced by memory merges)  
3. All memory allocations hoisted  
4. Switch lowered to structured branches  
5. No composite values in SSA  
6. Pure SSA for arithmetic/logical ops  
7. Control flow explicit  

---

# E. Future Work (Non‑Normative)

- gas metadata  
- optimization passes  
- symbolic interpreter  

---

# F. Open Questions

For open design questions and discussion topics, see:

- **Fat Pointers** - Section 9.1 in `01-types.md`  
  Discussion about whether EVM-IR should support fat pointers with runtime bounds information.

---

# G. Change Log (v0.1)

Initial complete specification and IR definition.
