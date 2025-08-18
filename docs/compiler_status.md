## Ora Compiler Status and Semantics Readiness

This document summarizes the current implementation status of the Ora compiler frontend (lexer, parser, AST, CST, and types), with direct references to code and `GRAMMAR.bnf`, and proposes a prioritized list of gaps before or during semantics integration.

### Lexer: `src/lexer.zig`
- Status
  - Tokens and keyword set align with grammar, including operators and builtins:
    - `@` token (`TokenType.At`), `import` keyword (`TokenType.Import`), shifts `<<`/`>>` (`.LessLess`/`.GreaterGreater`), exponent `**` (`.StarStar`), arrow `->` (`.Arrow`), and range `...` (`.DotDotDot`). See `src/lexer.zig` (grep: At at ~1180, Import at ~1454).
  - `isOperator` includes arithmetic, comparison, logical, bitwise, shifts, compound assigns, `->`, `...`.
  - ASCII-only policy enforced; non-ASCII invalids covered by fixtures: `tests/fixtures/lexer/error_cases/non_ascii_identifier.ora`, `non_ascii_string.ora`.
  - Trivia captured (leading/trailing) with lossless reconstruction supported by `src/lossless_printer.zig`.
- Gaps: None blocking semantics.

### Parser Core and Modules
- Top-Level: `src/parser/parser_core.zig`
  - Supports: `@import("...");`, `const name = @import("...");`, `contract`, `fn` (with `inline`), `var/let/const/immutable`, `struct`, `enum`, `log`, `error`.
  - Unknown top-level `@` rejected with diagnostic: “Unknown @ directive at top-level; only @import is supported.”
  - CST integration creates top-level `CstKind` nodes.

- Statements: `src/parser/statement_parser.zig`
  - Implements grammar statements (see `GRAMMAR.bnf` 178–246), including:
    - `@lock(expr);` and `@unlock(expr);` (grammar 239–241); unknown `@` statements rejected with recovery.
    - Switch statements require `switch (expr)` with arm separation handled by common parser.
    - Move statement parsing is present (`move amount from src to dest;`, grammar 218–221).

- Declarations: `src/parser/declaration_parser.zig`
  - Function declarations follow grammar 64–83: `inline` before `fn`; `pub` visibility; return type via `->` (implemented at lines ~205–213); `requires`/`ensures` without semicolons.
  - Imports: `@import("path");` and `const name = @import("path");` (grammar 42–45), implemented in `parseImport`/`parseConstImport`.
  - Struct, enum, log, and error declarations per grammar (109–131; 124–126).
  - Unknown `@` before declarations rejected: “attributes are not supported”.

- Expressions: `src/parser/expression_parser.zig`
  - Switch expressions require `switch (expr)`; arms use `.ExpressionArm` mode (no block bodies).
  - `@cast(Type, expr)` implemented (grammar 339–340). Unknown `@` builtins rejected (“Unknown builtin function”).
  - Exponentiation `**` supported (tests present under parser fixtures for precedence).

- Types: `src/parser/type_parser.zig`
  - Primitives, `map[K,V]` (grammar 162), `doublemap[K1,K2,V]` (164), arrays `[T; N]` (168), `slice[T]` (170).
  - Error union restriction: `|` is only allowed when left side is an error union `!T`; otherwise diagnostic at lines ~91–101. Produces `_union` with `!T` as first member.
  - Anonymous struct types: `struct { name: Type, ... }` (172–176) implemented (L123–149), producing `OraType.anonymous_struct`.
  - `void` only valid in return type position (L68–76), per grammar 160–161.

- Common Parsers: `src/parser/common_parsers.zig`
  - Switch body handling with mode separation: `.StatementArm` requires trailing `;` for expression bodies; `.ExpressionArm` disallows blocks.
  - Range patterns accept general expressions `(a+b)...f(x)` (grammar 270).

### AST and Type System
- Spans: `src/ast/types.zig`
  - `SourceSpan` contains `file_id`, `byte_offset`, `line`, `column`, `length`, `lexeme`.

- Unified Type Info: `src/ast/type_info.zig`
  - `TypeInfo` with `TypeCategory` and `OraType` union:
    - Covers primitives and complex types: `struct_type`, `enum_type`, `contract_type`, `array` (elem ptr), `slice` (elem ptr), `mapping`, `double_map`, `tuple`, `function`, `error_union` (!T), `_union` (error-union members), `anonymous_struct`, `module`.
  - Memory management: `deinitTypeInfo`/`deinitOraType` free nested structures including `_union`, `anonymous_struct`, `mapping`, `double_map`, `tuple`, `function`, `result`.
  - Pretty-print: `OraType.render(writer)` renders structural source-like representations for types (anonymous structs, error unions, mappings, tuples, functions, etc.).
  - Tests: `tests/type_info_render_and_eq_test.zig` asserts render for anonymous struct, error-union union, mapping; includes a basic structural equality helper.

- AST Nodes: `src/ast/expressions.zig`, `src/ast/statements.zig`
  - Include nodes for Lock/Unlock; parser populates these. Visitors (`src/ast/ast_visitor.zig`) include `visitLock` hooks.

### CST and Lossless
- CST: `src/cst.zig`
  - Thin CST supports top-level nodes and token stream; builder integrated in `parser_core.zig` and CLI (`src/main.zig`) with `--no-cst` flag.
- Lossless: `src/lossless_printer.zig` reconstructs byte-for-byte from tokens+trivia; tests in `tests/lossless_roundtrip_test.zig` pass.

### Grammar Sync: `GRAMMAR.bnf`
- Implemented decisions reflected:
  - Return types via `->` (76), `requires/ensures` no `;` (79–83), arrays `[T; N]` (168), `slice[T]` (170), error unions `! T ("|" type)+` (151), anonymous struct types (172–176), `@import` and const import (42–45), switch rules and arm separation (253–273).
- Note: The top-level list still shows `const_declaration` (31) though code unifies const into `variable_declaration` (97–102). Consider removing `const_declaration` for consistency.

### Tests and Build Integration
- Build (`build.zig`) runs:
  - Lexer suite (`tests/lexer/lexer_test_suite.zig`) and fixtures.
  - Parser invalid fixtures (`tests/parser/parser_invalid_fixture_suite.zig`).
  - CST tests (`tests/cst_token_stream_test.zig`, `tests/cst_parser_top_level_test.zig`).
  - Lossless/byte-offset/doc-comments tests.
  - Type info render/equality tests (`tests/type_info_render_and_eq_test.zig`).

---

## Semantics Readiness and Priority Gaps

1) Unify type representation for semantics - DOne
   - Issue: There are two `OraType` definitions: the unified one in `src/ast/type_info.zig` and a separate one in `src/typer.zig` (see `src/root.zig` re-exports `pub const OraType = typer.OraType;`).
   - Risk: Divergence and duplicated logic.
   - Plan: Make semantics operate on `ast.type_info.OraType`/`TypeInfo`. Add a conversion layer to `typer.OraType` only where required by backend; or refactor `typer` to consume `ast.type_info`.
   - Priority: High.

2) Equality and hashing for `ast.type_info.OraType`
   - Need deep structural `equals(a,b)` and a stable `hash` across variants (mapping/double_map/tuple/function/_union/anonymous_struct).
   - Tests can follow `tests/type_info_render_and_eq_test.zig` style.
   - Priority: High.

3) Array size representation
   - Grammar `[T; integer_literal]` (168). Current `OraType.array: *const OraType` doesn’t carry length.
   - Plan: change to `array: struct { elem: *const OraType, len: u64 }`; update `deinit`/`render` and `type_parser.zig` to parse and store size.
   - Priority: Medium.

4) Result vs error unions — DECIDED: Remove Result - Done
   - We removed `Result[T,E]` parsing and the corresponding `ast.type_info` variant. Use error unions `!T | E` exclusively.
   - Priority: Done.

5) L-value validation - Done
   - Grammar defines `lvalue` (311–318). Parser enforces at syntax level in statements, but add an AST validation pass to ensure assignability for complex cases.
   - Priority: Medium.

6) Switch typing and exhaustiveness
   - Semantics should type-check patterns vs scrutinee, handle `else`, and detect overlaps.
   - Priority: Medium–High.

7) Function contracts
   - Parse-time `requires`/`ensures` exist; semantics should type-check these expressions and store/validate.
   - Priority: Medium.

8) Docs/attachments
   - `src/doc_comments.zig` and `src/doc_attach.zig` can map docs to AST; integrate if semantics/symbol tables require docs.
   - Priority: Low.

9) CST breadth
   - CST currently top-level only; sufficient for now. Extend if the formatter or tooling needs deeper CST. Not required for semantics.
   - Priority: Low.

---

## Recommended Next Steps
1. Implement `equals` and `hash` for `ast.type_info.OraType` and `TypeInfo` (add tests). - DOne
2. Add array size to `OraType.array` and propagate in `src/parser/type_parser.zig`.
3. Scaffold `src/semantics.zig`:
   - `analyze(nodes: []const ast.AstNode) -> { diagnostics, symbol_table }`.
   - Build symbol tables; enforce const/immutable initialization per `GRAMMAR.bnf` 95–102.
   - Type-check assignments (lvalue), switch cases, and function pre/postconditions.
4. Decide unification strategy between `ast.type_info` and `typer` and implement a conversion or refactor path.

This status is grounded in the referenced code files and the current `GRAMMAR.bnf` rules. Update this document as we complete each gap.


### Blockers and low-hanging tasks before semantics


- Blocker: unify type representation Done
  - Files: `src/ast/type_info.zig` (unified `TypeInfo`/`OraType`) vs `src/typer.zig` (separate `OraType`); `src/root.zig` re-exports `pub const OraType = typer.OraType` (L78–79).
  - Action: Choose `ast/type_info` for semantics; add conversion to `typer.OraType` for backend or refactor `typer` to consume `ast/type_info`.
  - Priority: High.

- Low-hanging: spans use `file_id = 0` - Done
  - File: `src/parser/common.zig` `ParserCommon.makeSpan` (L9–19) hardcodes `.file_id = 0`; many call sites rely on it.
  - Action: Replace calls with `BaseParser.spanFromToken(token)` (L156–166) or pass `file_id` into `makeSpan` and update usages (e.g., in `src/parser/declaration_parser.zig`).
  - Priority: High value, low risk.

- Low-hanging: top-level `inline fn` - Done
  - File: `src/parser/parser_core.zig` top-level functions (L141–149) check `.Pub` or `.Fn` only.
  - Action: Allow `.Inline` too: `if (self.check(.Pub) or self.check(.Fn) or self.check(.Inline))`.
  - Priority: Low.

- Low-hanging: error declaration semicolon policy - DOne
  - Grammar: `error_declaration ... ";"` (GRAMMAR.bnf L131).
  - File: `src/parser/declaration_parser.zig` `parseErrorDecl` uses optional `;` (L791 `_ = self.base.match(.Semicolon)`).
  - Action: Enforce required `;` in both top-level and contract scopes.
  - Priority: Low.

- Low-hanging: `log` field `indexed` pseudo-keyword vs grammar - Done
  - Grammar: `log_declaration ::= "log" identifier "(" parameter_list? ")" ";"` (L125); no `indexed` modifier.
  - File: `src/parser/declaration_parser.zig` supports `indexed` before fields (L499–516).
  - Action: Remove `indexed` support for now or add it to grammar; prefer removal to keep spec tight.
  - Priority: Low.

- Medium: array size in type representation
  - Grammar: `[type; integer_literal]` (L168).
  - File: `src/ast/type_info.zig` has `OraType.array: *const OraType` (no length); `src/parser/type_parser.zig` parses arrays but drops the size.
  - Action: Change to `array: struct { elem: *const OraType, len: u64 }`; update deinit/render and `parseArrayType` to carry size.
  - Priority: Medium.

- Spec mismatch: parameter default values
  - Grammar: `parameter ::= identifier ":" type` (L74); no defaults.
  - File: `src/parser/declaration_parser.zig` supports defaults in `parseParameterWithDefaults` (L632–644).
  - Action: Remove defaults or amend grammar; prefer removal for v1.
  - Priority: Low.

- Spec cleanup: `const_declaration` in grammar
  - Grammar: top-level list includes `const_declaration` (L31) but we unified to `variable_declaration` (L97–102).
  - Action: Remove `const_declaration` from `GRAMMAR.bnf`.
  - Priority: Low.

- Tests: top-level unknown `@`
  - Parser: `parser_core.zig` errors on unknown top-level `@` (L108–127).
  - Action: Add invalid fixture under `tests/fixtures/parser_invalid/top_level/bad_unknown_at.ora` to lock behavior.
  - Priority: Low.


