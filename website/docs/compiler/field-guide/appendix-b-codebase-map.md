# Appendix B — Codebase Map (quick reference)

This is a navigation aid. If you’re unsure where something lives, start here.

## CLI / Orchestration
- `src/main.zig`

## Frontend
- Lexer: `src/lexer.zig`, `src/lexer/scanners/`
- Parser: `src/parser.zig`, `src/parser/*`
- AST: `src/ast/*`

## Semantics and Types
- Semantics: `src/semantics.zig`, `src/semantics/*`
- Type resolver: `src/ast/type_resolver/*`

## MLIR
- Lowering orchestrator: `src/mlir/lower.zig`
- MLIR modules: `src/mlir/*`
- Verification: `src/mlir/verification.zig`
- Pass pipeline: `src/mlir/pass_manager.zig`
- Ora → SIR conversion patterns: `src/mlir/ora/lowering/OraToSIR/*`

## SMT / Z3
- Z3 integration: `src/z3/*` (notably `encoder.zig`)

> Tip: if you’re lost, follow the artifact ladder and search for the function that prints that artifact.
