# Appendix D — Where to Add Tests

The best tests are:
- minimal
- phase-local
- stable over refactors

## What to test by phase

### Lexer tests
- tokenization of tricky literals
- keyword vs identifier ambiguity
- span correctness

### Parser tests
- AST shape for specific constructs
- recovery behavior after an error
- precedence/associativity for expressions

### Type resolver tests
- correct `TypeInfo` annotation
- correct diagnostics for invalid programs
- rules like “error unions must be unwrapped with `try`”

### Lowering (MLIR) tests
- emitted ops contain expected shapes
- storage load/store appears when expected
- specification ops are emitted/attached correctly

### Verification tests
- illegal IR is rejected early
- legal IR passes verifier

### Conversion (Ora → Sensei-IR (SIR)) tests
- specific ops are convertible
- missing patterns fail loudly with clear diagnostics

### SMT/Z3 tests
- constraints encode correctly
- counterexample models are stable and meaningful
