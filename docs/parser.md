Here’s a concise, actionable to-do list of parser/lexer/AST alignment items. Each item has Issue/Decision, Actions, Acceptance, Touches.

1) DoubleMap keyword bug - DOne
- Issue: `isKeyword` checks `.Doublemap` (typo), actual enum is `.DoubleMap`.
- Actions: Fix case in `isKeyword`.
- Acceptance: Build passes; no false negatives on `doublemap`.
- Touches: `src/lexer.zig` (isKeyword switch).

2) Shift token normalization – Done
- Decision: Keep lexer tokens as `.LessLess`/`.GreaterGreater` only; map to AST `.LeftShift`/`.RightShift` in parser.
- Actions: Removed `TokenType.LeftShift/RightShift`; ensured scanner emits `.LessLess`/`.GreaterGreater`; updated `isOperator` accordingly; parser mapping already handled.
- Acceptance: `<<`/`>>` recognized and parsed; no dead enum variants. Build green.
- Touches: `src/lexer.zig` (TokenType, isOperator), `src/parser/expression_parser.zig` (mapping uses `.LessLess`/`.GreaterGreater`).

3) Function return type syntax – Done (Arrow)
- Decision: Use grammar `-> Type` after parameter list.
- Actions: Implemented Arrow handling in `DeclarationParser`; C-style return removed.
- Acceptance: Fixtures compile using `->`; one canonical form remains.
- Touches: `src/parser/declaration_parser.zig`, tests, `GRAMMAR.bnf`, docs.

4) Switch arms: commas and semicolons – Done
- Actions: Statement arms now require `;` for expression bodies (mode-aware parseSwitchBody). Optional commas between arms supported.
- Touches: `src/parser/common_parsers.zig`, `src/parser/{statement,expression}_parser.zig`.

5) Switch condition required – Done
- Actions: Both statement and expression parsers now require `switch (expr)`.
- Touches: `src/parser/statement_parser.zig`, `src/parser/expression_parser.zig`.

6) Requires/ensures semicolons unified - Done
- Issue: Statements reject `;`, function-header clauses allow optional `;`.
- Actions: Disallow `;` in both locations.
- Acceptance: `requires(...)`/`ensures(...)` with semicolons are rejected; without semicolons parse fine.
- Touches: `src/parser/declaration_parser.zig`, tests, docs.

7) Import/log/error semicolon policy – Done (enforce `;`)
- Actions: Now require `;` after `@import(...)`, `const name = @import(...)`, `log ...;`, and `error ...;`.
- Touches: `src/parser/declaration_parser.zig`, `src/parser/statement_parser.zig`.

8) Type syntax unification: arrays/slices – Done (keep `[T; N]` + `slice[T]`) -Done
- Actions: Updated grammar to `[T; N]` for fixed arrays and `slice[T]` for dynamic sequences. Parser already matches.
- Touches: `GRAMMAR.bnf`.

9) Type unions with “|” (and error unions) - done
- Issue: Grammar has `type ("|" type)+` and `!type`; parser only supports `!type`.
- Actions: Implement `|` union parsing in `TypeParser`; update `TypeInfo` if needed.
- Acceptance: `u32 | bool`, `!T | E` parse to proper `TypeInfo`.
- Touches: `src/parser/type_parser.zig`, `src/ast/type_info.zig`, tests, docs.

10) Anonymous struct types - Done
- Issue: Grammar allows `struct { a: T, ... }` as a type; parser lacks it.
- Actions: Implement anonymous struct type parser; represent in `TypeInfo`.
- Acceptance: `struct { a: u32 }` usable in params/fields; tests.
- Touches: `src/parser/type_parser.zig`, `src/ast/type_info.zig`, tests.

11) Cast syntax consolidation – Done
- Decision: Keep `@cast(Type, expr)`.
- Actions: Removed lexer `as` keyword; parser already implements `@cast(...)`. Next: remove any references to `as` in fixtures/docs.
- Touches: `src/lexer.zig`, docs/tests.

12) Exponent operator “**” – Done (keep)
- Actions: Parser precedence includes right-associative `**`; AST has `StarStar`; fixtures updated to include exponent precedence cases. Codegen will map to Yul EXP.
- Touches: `src/parser/expression_parser.zig`, `src/ast/expressions.zig`, tests.

13) Switch pattern expression breadth - Done
- Issue: Range patterns in common parser look ahead for ints/identifiers; grammar allows any expression.
- Actions: Relax lookahead to rely on expression parser whenever `...` appears.
- Acceptance: Patterns like `(a+b)...f(x)` parse; tests added.
- Touches: `src/parser/common_parsers.zig`, tests.

14) Import alias via const - Done
- Issue: Implementation supports `const name = @import("path")`; grammar only has `@import("path");`.
- Decision: Keep feature (document + add grammar) or remove.
- Actions: Align grammar/docs or gate in parser.
- Acceptance: Grammar, docs, parser consistent; fixtures updated.
- Touches: `GRAMMAR.bnf`, `docs`, `src/parser/declaration_parser.zig`, tests.

15) Tuple destructuring - Done
- Issue: Parser supports `let (a, b) = ...` but grammar only shows `let .{ ... } = ...`.
- Decision: Keep and document or remove/gate.
- Actions: Align grammar or remove tuple form.
- Acceptance: One canonical destructuring form; tests.
- Touches: `src/parser/declaration_parser.zig`, `src/parser/statement_parser.zig`, `docs`, tests.

16) isOperator completeness - Done
- Issue: Ensure `isOperator` recognizes exactly the token variants the lexer emits (post item 2).
- Actions: Update `isOperator` after shift token normalization; verify includes all applicable operators (`<<`, `>>`, `...`, `->`, compound assigns).
- Acceptance: No operator token missed by `isOperator`.
- Touches: `src/lexer.zig`.

17) Switch expression/statement arm separation - Done
- Issue: Statement arms may use blocks; expression arms must be expressions.
- Actions: Keep shared body parser but add mode to enforce semicolon for statement arms and block-only/expr distinction per grammar.
- Acceptance: Invalid forms rejected with helpful messages; tests.
- Touches: `src/parser/common_parsers.zig`, `src/parser/{statement,expression}_parser.zig`, tests.

18) Tests and fixtures
- Actions: Add/adjust tests for all decisions above (parser and lexer), especially:
  - switch commas/semicolons/condition
  - function return syntax
  - type unions/anonymous struct types
  - array/slice syntax
  - cast/exponent decisions
  - tuple destructuring policy
- Acceptance: Test suite green with updated fixtures.
- Touches: `tests/fixtures/parser/**`, `tests/lexer/**`.

19) Docs and grammar updates
- Actions: Update `GRAMMAR.bnf`, `docs/grammar-implementation-sync.md`, any website docs to match final decisions.
- Acceptance: Docs reflect implemented syntax; no stale examples.
- Touches: `GRAMMAR.bnf`, `docs/**`, `website/**`.

- Quick references to current code spots
  - DoubleMap typo:
    ```3369:3394:/src/lexer.zig
    pub fn isKeyword(token_type: TokenType) bool { ... .Doublemap ... }
    ```
  - Shift token duplication:
    ```1147:1151:/src/lexer.zig
    LessLess, GreaterGreater, LeftShift, RightShift
    ```
  - Function return style (current C-style in fixtures):
    ```1678:1686:/tests/lexer/lexer_test_suite.zig
    pub fn function{d}(...) u256 { ... }
    ```
  - Switch arm parsing (no commas/semicolon enforcement):
    ```186:218:/src/parser/common_parsers.zig```
  - Requires/ensures semicolon mismatch:
    ```223:254:/src/parser/declaration_parser.zig```
    ```287:305:/src/parser/statement_parser.zig```
  - Array/slice parser:
    ```174:201:/src/parser/type_parser.zig```
  - Cast via @cast:
    ```1181:1213:/src/parser/expression_parser.zig```

- If you want, I can start by fixing items 1, 2, 4, and 6 (small, low-risk), then move to the larger design decisions (3, 8, 9, 10, 11, 12, 14, 15).


Yes. Here’s a focused add-on roadmap to bring the parser and AST to production quality. Same format: Issue/Decision, Actions, Acceptance, Touches.

20) Lossless parsing (comments/trivia) - Done
- Issue: Comments/whitespace are dropped; needed for formatting/tools.
- Actions: Capture leading/trailing trivia per token; optionally expose a CST or attach trivia to AST nodes.
- Acceptance: Round-trip formatter preserves comments/spacing.
- Touches: src/lexer.zig, src/parser/*, src/ast/*, docs, tests.

21) Error-tolerant AST nodes
- Issue: Hard errors abort parse; IDEs need partial AST.
- Actions: Introduce ErrorNode/ErrorType sentinels; ensure downstream passes skip or degrade gracefully.
- Acceptance: Broken sources still build an AST with localized error nodes; IDE features continue.
- Touches: src/ast/{expressions,statements}.zig, src/parser/*, tests.

22) Structured diagnostics with fix-its
- Issue: Errors lack fix suggestions and secondary ranges.
- Actions: Standardize diagnostic codes, categories, primary/secondary ranges, quick-fixes (e.g., “insert ‘;’”, “add ‘)’”).
- Acceptance: Tests assert presence of codes and fix-its; CLI and LSP output render labels.
- Touches: src/parser/common.zig, parser error sites, diagnostics infra, tests, docs.

23) Production-grade error recovery
- Issue: Recovery is generic; needs production heuristics.
- Actions: Synchronization sets per nonterminal; recover at “safe” tokens; rainbow-delim repair; special cases for dangling else, missing ‘)’/’]’/’}’.
- Acceptance: Representative broken snippets recover to expected AST shapes; perf stable.
- Touches: src/parser/common.zig, statement/expression/type parsers, tests.

24) Incremental parsing
- Issue: Full reparse on edit.
- Actions: Track node spans + stable IDs + cheap subtree hashing; implement reparse windows; API for reusing unchanged subtrees.
- Acceptance: Edits reparse subtrees only; benchmark shows wins on medium files.
- Touches: src/parser/*, src/ast/* (NodeId, parent links), tests, docs.

25) Parent links and stable NodeId
- Issue: No upward navigation or stable IDs.
- Actions: Attach parent pointer (or index) and 64-bit NodeId deterministic by position; ensure maintained through edits.
- Acceptance: Visitor can navigate up; NodeId stable across no-op formatting.
- Touches: src/ast/*, builders in parser, tests.

26) Source positions: byte offsets + fileId - DOne
- Issue: Only line/column/len; no file association; tabs/multibyte not handled.
- Actions: Store fileId + byte offset/length; keep line/col as derived; normalize tabs policy.
- Acceptance: Accurate mapping in LSP; multi-file projects supported.
- Touches: src/lexer.zig, src/ast/types.zig (SourceSpan), all parser span constructors, tests.

27) Performance hardening and limits
- Issue: Potential deep recursion, OOM, pathological inputs.
- Actions: Depth/size limits (nesting, params, case count); iterative parses for deep expressions; pooled small vectors; profile hot paths; arena stats.
- Acceptance: Fuzz and microbench pass; no stack overflow on deep nests; memory caps enforced.
- Touches: src/parser/*, src/ast/*, tests/benches.

28) Unicode and identifier policy - Done
- Decision: Keep ASCII-only or adopt Unicode identifiers/escapes.
- Actions: If Unicode: implement identifier classes, escapes in strings/chars; normalize; security review (confusables).
- Acceptance: Tests covering NFC, escapes, edge cases; documented policy.
- Touches: src/lexer.zig, docs, tests.

29) Attributes/annotations spec - Done
- Decision: No general attributes in v1. The `@` sigil is reserved for the existing builtins only:
  - `@import("...");` statement
  - `@lock(expr);` / `@unlock(expr);` statements
  - `@cast(Type, expr)` expression
  There is no `@payable`, `@inline`, etc. `inline` remains a keyword modifier before `fn`.
- Actions:
  - Grammar: Keep only the three builtin forms above; do not add a generic `@identifier(...)` production.
  - Parser: If an `@` is encountered not starting one of the known forms, emit “Unknown @ directive” and recover.
  - Tests: Add invalid fixtures for `@payable` and `@inline` before functions, and a stray `@foo(1);` statement.
- Acceptance: Invalid `@` usages are rejected with diagnostics; docs clearly state no attributes.
- Touches: Grammar, parser diagnostics, tests.

30) Anonymous struct type + union type design (type_info) - DOne
- Issue: Adding types (struct literal, unions) needs robust TypeInfo modeling.
- Actions: Ensure TypeInfo handles field lists, union members, formatting, deinit; add hashing/equality.
- Acceptance: Pretty-print and equality tests for complex types pass.
- Touches: src/ast/type_info.zig, type_parser, tests.

31) AST validation pass
- Issue: Some syntactic invariants leak to semantics.
- Actions: Add a fast AST validator (e.g., lvalue-only LHS, switch arms constraints, break label scoping).
- Acceptance: Validator finds issues on crafted invalid ASTs; integrated to compiler pipeline.
- Touches: src/ast/*, new validator module, tests.

32) Pretty-printer and stable formatter
- Issue: No official formatter; needed for dev UX/tests.
- Actions: Implement pretty-printer from AST (uses trivia if lossless); snapshot tests.
- Acceptance: Formatter round-trips most code; stable output under reformat.
- Touches: src/ast_serializer.zig or new formatter, tests, docs.

33) Fuzzing and corpus tests
- Issue: No fuzz input coverage.
- Actions: Hook libFuzzer/AFL; seed with fixtures; sanitizer builds; CI job.
- Acceptance: Sustained fuzz run w/o crashes; bug repro reduction pipeline.
- Touches: scripts/, build, tests, CI.

34) LSP-ready symbol extraction hooks
- Issue: IDE features need quick symbol/outline info.
- Actions: Add traversal helpers: collect declarations, spans, doc comments; expose stable API.
- Acceptance: Prototype LSP server consumes APIs; latency acceptable.
- Touches: src/ast/ast_visitor.zig, helpers, docs.

35) Parser API ergonomics and modes
- Issue: One-shot parse only.
- Actions: Expose parseModule, parseExpr, parseType; strict vs tolerant modes; config object.
- Acceptance: REPL/tests use parseExpr; strict mode halts on first error.
- Touches: src/parser/mod.zig, public API, docs.

36) Memory lifecycle robustness
- Issue: Deinit paths complex; risk of leaks/double frees.
- Actions: Unit tests for deinit on every node variant; arena lifetime rules; valgrind/ASAN checks.
- Acceptance: Leak/UB-free under stress tests.
- Touches: src/ast/*.zig, tests.

37) Versioned grammar/feature flags
- Issue: Evolving language needs gating.
- Actions: Add language version to parser config; gate syntax (e.g., **, const import alias) behind flags.
- Acceptance: Tests cover new/old versions; helpful diagnostics on gated features.
- Touches: src/parser/common.zig (config), parsers, docs.

38) Consistency checks and codegen hooks
- Issue: Some constructs span both parser and later phases.
- Actions: Add hooks/markers for lowering (e.g., switch desugaring, move semantics tags) in AST.
- Acceptance: Lowering pipeline can rely on consistent AST metadata.
- Touches: src/ast/*.zig, docs.

If you want, I can append these to docs/parser.md under a “Production-readiness tasks” section, then start with low-risk wins (20, 22, 23 basics, 26, 27) and create stubs for NodeId/parent links (25).

- Added a comprehensive task list covering lossless/CST, tolerant AST and recovery, diagnostics with fix-its, incremental parsing, NodeId/parent links, span/file model, perf and limits, Unicode policy, formal attributes, TypeInfo robustness, AST validation, formatter, fuzzing, LSP hooks, API modes, memory lifecycle, versioning, and lowering hooks.