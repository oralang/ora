# The Compiler Shape (without compiler jargon)

This chapter answers: “Who does what?” without requiring prior compiler knowledge.

## 2.1 The pipeline as promises

Think of each phase as a team member with a job and a promise:

### Lexer
- **Input:** source text  
- **Output:** tokens (+ location info)  
- **Promise:** the parser receives a stable token stream

### Parser
- **Input:** tokens  
- **Output:** AST with spans  
- **Promise:** later phases can walk a structured tree and point diagnostics at precise source ranges

### Semantics Phase 1 (symbol collection)
- **Input:** AST  
- **Output:** symbol table + scopes  
- **Promise:** name lookup is stable (what `value` refers to, which `inc` overload, etc.)

### Type resolver
- **Input:** AST + symbol table  
- **Output:** the same AST, now annotated with `TypeInfo`  
- **Promise:** lowering doesn’t guess types — it reads them from the AST

### Optional Semantics Phase 2 (extra validation)
- **Input:** typed AST  
- **Output:** additional checks (often wired through tests)  
- **Promise:** catches semantic rules that don’t fit nicely into typing

### MLIR lowering
- **Input:** typed AST  
- **Output:** Ora MLIR module  
- **Promise:** downstream passes/verification operate on a consistent IR

### Verification & passes
- **Input:** Ora MLIR  
- **Output:** verified, optimized IR; optionally SMT results  
- **Promise:** invalid programs are rejected early; valid programs are normalized for the backend

### Ora → Sensei-IR (SIR) conversion
- **Input:** Ora MLIR  
- **Output:** Sensei-IR (SIR) MLIR  
- **Promise:** whatever reaches Sensei-IR (SIR) is in the “legal” subset supported by the backend

## 2.2 The one design choice to remember: typed AST

Ora’s AST is not just syntax.
Many nodes carry `TypeInfo` fields that begin unknown and are filled during type resolution.

That means:
- a lot of “lowering bugs” are actually “typing bugs”
- if an IR op has the wrong type, check the typed AST first

## 2.3 Where to stand when you’re confused

If you feel lost:
- go back to the artifact ladder (Tokens → AST → Typed AST → MLIR → Sensei-IR (SIR))
- run the phase you think is failing in isolation
- inspect the artifact

That workflow is the real onboarding.
