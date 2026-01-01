# Appendix C — Add a Feature Checklist (canonical)

Use this checklist when adding a new language feature. The goal is end-to-end coherence.

1) **Grammar**
   - update grammar files
2) **Lexer**
   - keyword/literal/token scanning
3) **Parser**
   - parse rule(s) + AST node construction
4) **AST**
   - node definitions + spans
5) **Semantics**
   - symbol collection / scoping (if needed)
6) **Type resolution**
   - `TypeInfo` rules + diagnostics
7) **MLIR lowering**
   - emit Ora ops for new node forms
8) **Verification**
   - dialect invariants (if needed)
9) **Ora → SIR conversion**
   - conversion patterns for legality boundary
10) **Tests**
   - minimal repros + regression tests (lexer/parser/types/lowering/conversion)

**Important:** Do not skip tests. Features without tests become “unreviewable debt”.
