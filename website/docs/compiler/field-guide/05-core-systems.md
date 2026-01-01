# The Core Systems (Frontend → Semantics → Types → MLIR)

This chapter is the “where things live” chapter.

> **Lane A:** You mainly need this when you’re writing tests/diagnostics.  
> **Lane B:** You will use this daily.

---

## 5.1 Frontend: Lexer + Parser + AST

### Lexer
Responsibilities:
- tokenization
- keyword recognition
- literal scanning
- span tracking
- recovery (keep going after errors)

Key locations:
- `src/lexer.zig`
- `src/lexer/scanners/`

### Parser
Responsibilities:
- build AST nodes
- attach spans
- basic structural validation
- error recovery

Key locations:
- `src/parser.zig` and `src/parser/*`

### AST
Responsibilities:
- define node types
- arena allocation
- serialization/debug prints

Key locations:
- `src/ast/` (including arena + serializer)

---

## 5.2 Semantics: symbol tables and scopes

Responsibilities:
- collect declarations
- build symbol table
- manage scoping rules
- prepare for type resolution

Key locations:
- `src/semantics.zig`
- `src/semantics/*`

---

## 5.3 Types: `TypeInfo` + type resolver

Responsibilities:
- annotate AST nodes with `TypeInfo`
- enforce typing rules
- enforce rules around refinements and error unions
- produce precise diagnostics when typing fails

Key locations:
- `src/ast/type_resolver/`
- (and related structures in AST/type modules)

Practical note:
- if MLIR lowering “mysteriously” fails, check whether TypeInfo is missing or wrong

---

## 5.4 MLIR lowering: typed AST → Ora IR

Responsibilities:
- translate typed nodes to MLIR ops
- attach attributes and types
- produce structured lowering errors
- produce IR that can be verified + transformed

Key locations:
- `src/mlir/`
- `src/mlir/lower.zig`

---

## 5.5 Verification, passes, and conversion: MLIR → MLIR

Responsibilities:
- verify dialect invariants
- canonicalize/optimize
- convert Ora → SIR as a legality boundary
- optionally run SMT verification passes

Key locations:
- `src/mlir/verification.zig`
- `src/mlir/pass_manager.zig`
- `src/mlir/ora/lowering/OraToSIR/`
- `src/z3/*`

---

## 5.6 Where to start reading code (recommended)

If you’re new:
1) CLI entrypoints in `src/main.zig`  
2) lexer + parser to understand artifacts  
3) type resolver (TypeInfo flow)  
4) lowering (how typed nodes become ops)

That order matches how you’ll debug in real life.
