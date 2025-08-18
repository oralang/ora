## Ora Semantics Component: Detailed Plan

This plan outlines the full semantics component based on the current grammar (`GRAMMAR.bnf`) and existing code (parser, AST, types). It is grounded in concrete files and rules.

### Quality bar and goals
- Production-ready, modern compiler standards:
  - Deterministic diagnostics (stable ordering and IDs), rich messages with notes and fix hints
  - Robust recovery and safety (non-panicking traversal, throttled error floods)
  - Performance discipline (no quadratic hot paths, bounded recursion, zero leaks under GPA)
  - Incremental hooks and CST integration (stable node IDs, byte-precise spans)
  - Comprehensive typing and validations (calls/operators/switch/regions/immutability/errors/logs/contracts)
  - Strong test suite (valid/invalid fixtures, fuzzing, snapshot diagnostics, coverage metrics)

### Architecture alignment with existing semantics (code-bin/semantics)
- Reuse the proven module structure from `code-bin/semantics`:
  - core/driver, diagnostics, state/coverage, contract/function/expression analyzers, import analyzer, switch analyzer, memory region validator, immutable tracker, utils.
- Replace `src/typer.zig` dependency with `src/ast/type_info.zig`:
  - All type checks should use `TypeInfo`/`OraType` (equals/hash already added) instead of `typer.OraType`.
  - If backend needs `typer`, add a thin converter (later).
- Port analyzers into `src/semantics/*`:
  - `src/semantics/core.zig`, `diagnostics.zig`, `state.zig`, `contract_analyzer.zig`, `function_analyzer.zig`, `expression_analyzer.zig`, `import_analyzer.zig`, `switch_analyzer.zig`, `memory_region_validator.zig`, `immutable_tracker.zig`, `utils.zig`.
- Keep existing safety/coverage patterns from old code (safeAnalyzeNode, validation coverage) and adapt to our AST spans.

### 0) Foundation and integration
Status: Done
- Entry point: `src/semantics.zig`
  - Function: `analyze(nodes: []const ast.AstNode) -> SemanticsResult` [Done]
  - Return: `diagnostics: []Diagnostic`, `symbols: SymbolTable`, optional semantic model [Done]
- Data structures:
  - `SymbolTable` with parent link; entries contain: name, kind (Contract, Struct, Enum, Log, Error, Fn, Var, Param, Module), `TypeInfo`, mutability, region, span. [Done]
- Diagnostics: use `ast.SourceSpan` (file_id + byte_offset) for precision. [Done]

### 1) Symbol collection and scoping
Status: Partially Done
- Top-level scope: collect `contract`, `fn` (incl. `inline`), `struct`, `enum`, `log`, `error`, imports. [Done]
- Contract scope: state variables (respect `storage/memory/tstore`), functions, nested types. [Done]
- Function scope: parameters, local variables and struct destructuring (`let .{...} = expr;`). [Done (params + locals), TODO (struct destructuring binder)]
- Imports (GRAMMAR 42–45): `@import("path");` and `const name = @import("path");` create module/alias symbols (resolution of module contents is deferred). [Done]
- Rules: redeclaration errors, unknown identifier diagnostics on use. [Done]
  - Files: implemented in `src/semantics/collect.zig` and wired via `src/semantics/core.zig`.

### 2) Type inference and assignment rules
Status: Partially Done
- Expression typing via `TypeInfo`/`OraType` (`src/ast/type_info.zig`):
  - Literals map to primitive types; identifiers resolve to symbol types. [Done]
  - Field access: resolve against `anonymous_struct` fields; named struct resolution TBD. [Done (anonymous_struct), TODO (named structs)]
  - Index access: array/slice/mapping element type propagation. [Done]
  - Calls: callee function type, arity checking, param type match, return type propagation. [Partial: return type propagation for identifier callees; arity/param checks TODO]
  - Operators: arithmetic (+, -, *, /, %, **), comparisons (==, !=, <, <=, >, >=), logical (&&, ||), bitwise (|, &, ^, <<, >>) with numeric/boolean rules. [TODO]
  - `@cast(Type, expr)` (GRAMMAR 339–340): propagate target type. [Done]
- Statements:
  - Assignment `lvalue = expr;` (GRAMMAR 209): enforce lvalue on LHS; basic type compatibility RHS→LHS. [Done]
  - Compound assignment (GRAMMAR 212): numeric-only first cut; mutability check for identifiers. [Done]
  - Return: ensure expression type matches function return `->` (GRAMMAR 76). [Done]
  - `log` call: match declared `log` signature; check `indexed` markers parsed in `src/parser/declaration_parser.zig`. [Done]
- Error unions (`!T | E | ...`): propagate success type `T` and error members; functions returning unions validate return statements accordingly. [Partial (success type via try; return validation implemented)] [Done]
  - Files: implement in `src/semantics.zig`.

### 3) Switch typing and exhaustiveness
- Switch statement/expression (GRAMMAR 253–273):
  - Infer scrutinee type once.
  - Patterns: range (270) endpoints must type-check with scrutinee; else must be last.
  - Switch expression: all arm expressions unify to a common type (first cut: equal types); emit mismatch diagnostics.
  - Exhaustiveness rules:
    - Enums: cover all variants or include `else`.
    - Bool: arms for `true` and `false` or include `else`.
    - Integers/strings: allow ranges/values; require `else` unless statically exhaustive.
  - Parser enforces arm separation and block-vs-expression modes (see `src/parser/common_parsers.zig`).

### 4) Function contracts
- `requires(...)` and `ensures(...)` (GRAMMAR 79–83): type-check to `bool`; attach to function metadata for future verification.
- Validate `old()` usage only inside `ensures`.

### 5) Move statement (GRAMMAR 218–221)
- `move amount from src to dest;`
  - amount: numeric; `src`/`dest`: storage paths or valid lvalues; ensure `dest` is assignable; regions consistent.
  - Validate move does not violate immutability or region rules; emit targeted diagnostics.

### 6) Regions, mutability, lvalues
Status: Partially Done
- Memory regions: `storage`/`memory`/`tstore`/`stack` (GRAMMAR 99). Validate declarations and accesses per context (contract/function/global). [TODO]
- Mutability: `let` immutable; `var` mutable; `const`/`immutable` require initializer; params respect `mut` flag. [Done (identifier writes), TODO (const/immutable/init rules)]
- Lvalues: enforce via `isLValue` (Identifier/FieldAccess/IndexAccess) with diagnostics. [Done]
- Region transitions and assignments validated (target vs source region rules). [TODO]

### 7) Errors and logs
- Error declarations (GRAMMAR 130–131): recorded as tags/types; used in error unions.
- Logs (GRAMMAR 124; extended for `indexed`): verify argument count/types and record indexed flags.
  - Enforce `indexed` only in allowed positions; ensure argument types match declared field types.

### 8) Modules/imports
Status: Partially Done
- Record module symbols for `@import(...)` and aliases via `const name = @import(...)`. [Done]
- Defer module content resolution; allow qualified name placeholders. [Done]
- Additional validation (cycles, duplicates across files). [TODO]

### 9) Diagnostics
Status: TODO
- Deterministic ordering by `file_id` then `byte_offset`.
- Categories and codes: NameError, TypeError, SemanticError, RegionError, ContractError.
- Messages include expected vs found types; spans from AST tokens (`spanFromToken`).
- Recovery toggles and error-flood throttling for IDE scenarios.

### 10) Architecture and performance
- Two phases per compilation unit:
  1) Collect symbols (top-down scopes).
  2) Type-check/validate.
- Iterative traversal for deep expressions if needed; otherwise recursive is OK to start.
- Safety wrapper for analyze steps; bound recursion; avoid quadratic joins.
- Zero allocations leaks with GPA; reuse buffers.

### 11) Testing plan
- Unit tests for:
  - Name resolution (unknown symbol, shadowing, redeclaration).
  - Assignment typing (including casts and const/immutable mutation errors).
  - Return type matching and missing return.
  - Switch typing (arm type mismatch, invalid ranges, missing else where needed).
  - Error union returns (`!T | E`) correctness.
  - Log calls with `indexed` flags.
  - Move statement constraints.
  - Requires/ensures must be boolean.
- Fixtures:
  - `tests/fixtures/semantics/valid/*.ora` and `.../invalid/*.ora`.
  - Integrate semantics test step into `build.zig` alongside parser/lexer tests.
- Fuzzing seeds for expressions/statements; snapshot expected diagnostics (stable order).
- Coverage metrics for analyzers and rule exercise counts.

### 12) Implementation steps
1. Create/extend `src/semantics/*` mirroring `code-bin/semantics` modules; expand `core.zig` driver with analysis state, recovery toggles, coverage counters. [Partial]
2. Add `type_integration.zig` (on `TypeInfo`/`OraType`): struct-field resolution (named/anonymous), function type construction, numeric compatibility, call typing, cast validation. [TODO]
3. Extend `state.zig` (symbol kinds/types/mutability/region); keep `SymbolTable` APIs stable. [Done]
4. Implement/extend `collectSymbols(nodes)` with redeclaration tests and module/alias symbols. [Done]
5. Name resolution helpers (`resolveName`) for identifiers and qualified names. [TODO]
6. Expression typing: literals/ident/field/index/calls/@cast, operators (arithmetic/logical/bitwise/comparisons/exponent), error union flow (`try`, returns). [Partial]
7. Assignment checks: lvalue + type compatibility + mutability + region assignment validation. [Partial]
8. Returns and contracts: return type checks; `requires`/`ensures` must be boolean; `old()` limited to `ensures`. [Partial]
9. Switch analyzer: scrutinee type, range validation, arm type unification, exhaustiveness/else. [TODO]
10. Import analyzer: record modules and aliases; shallow validation only. [TODO]
11. Memory region validator and immutable tracker: validate declarations, transitions, and writes. [TODO]
12. Logs: validate argument count/types and `indexed` flags against declarations. [TODO]
13. Diagnostics/safety: centralized diagnostics, deterministic ordering, safe traversal, error-flood throttling. [TODO]
14. Coverage/performance hooks and metrics. [TODO]
15. Tests: add `tests/semantics/*` valid/invalid, fuzz, snapshots; integrate in `build.zig`. [TODO]

### Production hardening checklist
- Deterministic diagnostics; stable ordering by `file_id` then `byte_offset`.
- Coverage stats (nodes analyzed, rules exercised) preserved from old framework.
- Error recovery toggles (as in old `semantics_recovery.zig`) for IDE tolerance.
- Performance: avoid excessive allocations; reuse buffers; short-circuit on floods.
- Clear separation between analysis (AST/TypeInfo) and backend.

### Incremental and tooling integration
- Stable node IDs and CST-backed spans for incremental re-analysis
- Mapper CST → AST to anchor diagnostics and editor tooling
- Flags to enable/disable recovery and incremental behavior

### Definition of Done (Semantics)
- All modules listed above implemented or stubbed with explicit TODOs not blocking type/safety rules
- Deterministic diagnostics with IDs; recovery toggle; no leaks under test
- Test suite with valid/invalid fixtures, snapshots, and minimal fuzz seeds; coverage counters present
- Integrated into `zig build test`; CI green

### Current preconditions met
- Types unified (`src/root.zig` exports `ast.type_info.OraType`, `TypeInfo`).
- L-value validator skeleton exists in `src/semantics.zig`.
- Arrays carry sizes (`OraType.array = { elem, len }`); parser stores size.
- Result[T,E] removed; error unions `!T | E` are the single error mechanism.
- Grammar updated for `indexed` in logs.



Prioritized next steps (low churn → core coverage)
1) Type integration utilities on TypeInfo/OraType:
named struct field resolution, function type construction, numeric compat, call typing.
Files: src/semantics/type_integration.zig, extend src/semantics/expression_analyzer.zig.
2) Switch analyzer:
unify arm types, validate ranges and else.
File: src/semantics/switch_analyzer.zig.
3) Requires/ensures boolean checks; log call checking with indexed:
Files: src/semantics/function_analyzer.zig, new src/semantics/log_analyzer.zig or fold into statement analyzer.
4) Immutability and memory regions (minimal pass):
Introduce immutable_tracker.zig (constructor-only writes, initialization), and memory_region_validator.zig (basic rules).
Wire into variable declarations and assignments.
5) Diagnostics + safety wrapper:
diagnostics.zig, errors.zig, memory_safety.zig, and integrate with core.zig to ensure stable, categorized diagnostics and safe traversal.
6) Imports/builtins:
import_analyzer.zig (shallow), builtin_functions.zig (at least @cast enforcement).
7) Tests:
Add tests/semantics/valid/* and invalid/* for each area above; wire into build.zig.
This brings us to feature parity with the old stack’s essentials while staying aligned with the new TypeInfo model. If you want, I can start with step 1 (type integration + call typing) and keep tests/builds green after each commit.