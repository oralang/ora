## Extending the Ora Compiler (junior-friendly)

This guide shows how to add features to the Ora compiler, with just enough Zig to be productive. Keep it simple, follow the steps, and use the checklist at the end.

### What the compiler does (in one minute)
- **lex**: turns text into tokens (`src/lexer.zig`)
- **parse**: turns tokens into AST nodes (`src/parser/…` + `src/ast/…`)
- **semantics**: verifies and annotates AST (`src/semantics/…`)
- **serialize/codegen**: converts AST to other forms (`src/ast/ast_serializer.zig`, codegen under `code-bin/`)

How to run locally:
```sh
cd /Users/logic/Ora/Ora
zig build                  # builds
zig-out/bin/ora parse ora-example/expressions/basic_expressions.ora
zig-out/bin/ora ast ora-example/enums/basic_enums.ora -o build
```

### Where things live
- `src/main.zig`: CLI. Calls lexer → parser → (optionally) serializer.
- `src/lexer.zig`: tokenization. Where new keywords/operators are recognized.
- `src/ast.zig`: central AST types and re-exports. Top-level `AstNode` union.
- `src/ast/ast_arena.zig`: arena allocator for AST memory. Do all AST allocations here.
- `src/ast/ast_builder.zig`: helpers to construct AST safely.
- `src/parser/`: split by concern
  - `parser_core.zig`: main `Parser`
  - `expression_parser.zig`, `statement_parser.zig`, `declaration_parser.zig`, `type_parser.zig`
- `src/semantics/`: analyses and checks (names, types, validations)
- `ora-example/`: tiny language examples you can run
- `tests/`: Zig tests for the frontend

## Bootstrap playbooks (follow these step-by-step)

These are concrete, end-to-end guides you can copy each time. Do them in order and use the checklist.

### Playbook A: Rename an identifier (keyword or builtin name)
Goal: rename keyword `log` → `event` (example).

1) Find where it’s recognized
```sh
rg "TokenType.*Log|\\blog\\b" src -n
```
- Expect hits in `src/lexer.zig` (keyword table) and parsers handling logs.

2) Lexer: update keyword mapping
- In `src/lexer.zig`, find the keyword table/switch mapping strings → `TokenType`.
- Replace the entry for `"log"` with `"event"` (keep the same token type if semantics are same, or introduce a new `TokenType.Event` if you want a new token).

3) Parser: update the branch that consumes the token
- In `src/parser/declaration_parser.zig` or `statement_parser.zig`, find the code that checks for `TokenType.Log` or matches the lexeme `"log"`.
- Update it to accept the new token/lexeme and still build the same AST node (e.g., `LogDeclNode` or `Log` statement).

4) Examples/tests: update inputs
- Fix examples in `ora-example/logs/` to use the new spelling.
- Adjust tests in `tests/` to expect the new token path.

5) Run
```sh
zig build
zig-out/bin/ora parse ora-example/logs/basic_logs.ora
```

6) Checklist
- [ ] Keyword mapping updated in lexer
- [ ] Parser branches updated
- [ ] Examples/tests pass

Notes:
- If you created a new `TokenType`, ensure all `switch` statements over tokens are updated (compiler will point to missing cases).
- If this changes AST shape, update `ast_serializer` and any visitors.

### Playbook B: Add a new operator with precedence
Goal: add `%` as modulo, mapped to `BinaryOp.Mod`.

1) Lexer
- Add a token for `%` in `src/lexer.zig` scanning logic.
- Ensure it doesn’t conflict with multi-char operators (e.g., `%=`, if supported).

2) AST
- Add `.Mod` to `BinaryOp` (likely in `src/ast/expressions.zig`).
- If you find a `switch` on `BinaryOp` elsewhere, add the new case.

3) Parser
- In `src/parser/expression_parser.zig`, place `%` at the same precedence level as `*` and `/`.
- If the parser uses precedence functions (e.g., `parseMultiplicative()`), add `%` there.
- If it’s Pratt/precedence-climbing, map the `%` token to the correct binding power.

4) Tests/examples
- Add an example file: `ora-example/expressions/mod.ora` with `let x = 7 % 3;`.
- Add/extend a test asserting the AST contains `BinaryOp.Mod`.

5) Run
```sh
zig build
zig-out/bin/ora parse ora-example/expressions/mod.ora
```

6) Checklist
- [ ] Token recognized
- [ ] AST enum updated
- [ ] Parser precedence updated
- [ ] Tests/examples added

### Playbook C: Create a new statement node (end-to-end)
Goal: add `defer { ... }` statement.

1) AST node
- In `src/ast/statements.zig`, define a struct, e.g. `DeferNode { body: BlockNode, span: SourceSpan }`.
- In `src/ast.zig`, add `.Defer: DeferNode` inside the `AstNode` union and update any `deinitAstNode` logic if needed.

2) Parser
- In `src/parser/statement_parser.zig`, in the statement dispatch, match the `defer` keyword/token.
- Parse a block and construct `AstNode{ .Statement = ... }` or `AstNode{ .Defer = ... }` depending on your AST design.
- Allocate via arena: `const node = try arena.createNode(ast.StmtNode); node.* = ast.StmtNode{ .Defer = ... };`

3) Semantics
- In `src/semantics/…`, validate scope rules or placement rules for `defer`.

4) Builder (optional convenience)
- In `src/ast/ast_builder.zig`, add a helper `deferStmt` mirroring other statement helpers.

5) Serializer (if applicable)
- In `src/ast/ast_serializer.zig`, add the new case so JSON export works.

6) Example + test
- Add `ora-example/control_flow/defer.ora`.
- Write a test in `tests/` asserting the node appears.

7) Run
```sh
zig build && zig build test
```

### Playbook D: Introduce/rename an AST field safely
Goal: add `visibility` to `FunctionNode` or rename an existing field.

1) Update the struct
- Edit `src/ast.zig` (and/or `src/ast/*`) to add/rename the field.

2) Fix all construction sites
- Search for allocations/assignments of that node:
```sh
rg "FunctionNode|\.Function\s*=|\.visibility\s*=|\.name\s*=|builder\.addFunction" src -n
```
- Update code to set the new field, using a sensible default if needed.

3) Update serializers and visitors
- Update `src/ast/ast_serializer.zig` and any visitors or printers to include the field.

4) Parser/Builtin defaults
- If parser builds the node, provide default values or read from syntax.

5) Tests/examples
- Adjust any failing tests to account for the new field.

### Deeper notes (how to find the right place)

- Token → Parser → AST map:
  - Start in lexer for token symbol/keyword
  - Then search in `parser/` for where it’s consumed
  - Confirm which AST node is produced (check `src/ast/…` and `src/ast.zig`)

- Precedence changes:
  - Look for functions like `parseUnary`, `parseMultiplicative`, `parseAdditive`, or a Pratt table
  - Keep operators with the same math precedence together

- Using the arena correctly:
  - Single node: `const n = try arena.createNode(T); n.* = T{ ... };`
  - Arrays: `const arr = try arena.createSlice(T, count); @memcpy(arr, src);`
  - Strings: `const s = try arena.createString(src_bytes);`
  - Never free arena-allocated AST manually; arena lifetime owns it

- Builder helpers:
  - `src/ast/ast_builder.zig` has helpers like `identifier`, `binary`, `block`, `whileStmt`, etc.
  - Preferred for constructing complex nodes; it adds basic validation and diagnostics.

### Useful searches
```sh
# Find where a token is handled
rg "TokenType\W+Identifier" src -n
rg "\.Identifier\b" src -n

# Find parser entry points
rg "pub const Parser" src/parser -n
rg "fn parse" src/parser -n

# Find AST union cases
rg "union\(enum\).*AstNode" -n src
rg "\.Contract|\.Function|\.StructDecl|\.EnumDecl" src -n
```

### Common gotchas (deeper)
- Update all `switch`es over enums/unions when adding tags (the compiler will hint missing cases).
- When adding tokens, check for multi-char neighbors (e.g., `=`, `==`, `=>`) to avoid mis-scans.
- Keep parser error messages precise; juniors benefit from clear diagnostics.
- Tests flaking? Recheck precedence/associativity and source spans assigned to nodes.

### Minimal Zig you need
- **struct**: data container
  ```zig
  const Point = struct { x: i32, y: i32 };
  ```
- **enum** and **union(enum)**: tagged unions (used for AST)
  ```zig
  const Node = union(enum) { Int: i64, Str: []const u8 };
  ```
- **errors**: `fn f() !T` returns `T` or error. Use `try`.
  ```zig
  fn read() !u8 { return 42; }
  const v = try read();
  ```
- **defer**: always runs on scope exit.
  ```zig
  defer some.cleanup();
  ```
- **slices & pointers**: `[]T` is slice, `*T` is pointer.
- **allocators**: we use the arena’s allocator for AST: `arena.allocator()`.
- **comptime** params: used in helpers like `createNode(self, comptime T: type)`.

### How to add a new language feature (pattern)
Example: add a binary operator `mod` with syntax `a % b`.

1) Tokens (lexer)
- Update `src/lexer.zig`:
  - Add a `TokenType` for `%` if missing
  - Teach the scanner to emit that token when it sees `%`

2) AST shape
- Update `src/ast/expressions.zig` (or where `BinaryOp` lives):
  - Add `Mod` to `BinaryOp`
- Nothing else changes if binary nodes already support all operators.

3) Parser
- In `src/parser/expression_parser.zig`:
  - Ensure `%` is recognized at the right precedence (same as `*`/`/` typically)
  - Map the `%` token to `BinaryOp.Mod`

4) Semantics (if needed)
- In `src/semantics/…` add any checks (e.g., `%` only for integers)

5) Examples + tests
- Add a small example under `ora-example/expressions/` using `%`
- Add/extend a test in `tests/` ensuring the AST contains the expected node/enum variant

6) Run
```sh
zig build
zig-out/bin/ora parse ora-example/expressions/your_new_example.ora
```

### How to add a new statement (pattern)
Example: add `defer { … }` statement.

1) AST node
- Define a new node in `src/ast/statements.zig` and add a case in `src/ast.zig`’s `AstNode` union.

2) Parser
- Extend `src/parser/statement_parser.zig` to recognize the keyword and build the node.
- Use the arena: `const node = try arena.createNode(ast.StmtNode);` and set `node.* = …`.

3) Semantics
- Add validations if any (scoping, allowed locations, etc.).

4) Example + tests
- Add a tiny `.ora` in `ora-example/` and a test in `tests/`.

### Arena usage rules (important)
- Allocate any AST struct or slice with the arena:
  - `try arena.createNode(T)` for a single node
  - `try arena.createSlice(T, n)` for arrays
  - `try arena.createString(bytes)` for owned strings
- Do not free AST pieces manually; they are freed when the arena is `deinit()`-ed.
- Don’t store pointers to AST beyond the arena’s lifetime.

### Coding style we follow (quick)
- Clear names: functions are verbs, variables are nouns
- Handle errors explicitly with `!` and `try`
- Keep control flow shallow; use early returns
- Prefer arena allocations for AST; avoid mixing allocators
- Keep files small and focused (lexer vs parsers vs ast vs semantics)

### Step-by-step: add a keyword
Goal: add keyword `event`.

1) Lexer (`src/lexer.zig`)
- Add `TokenType.Event`
- In the identifier/keyword table, map `"event"` to `TokenType.Event`

2) Parser (`src/parser/declaration_parser.zig`)
- In the top-level declaration switch, add a branch for `event`
- Build a `LogDecl` or a new node if it’s different

3) AST (`src/ast.zig` and `src/ast/statements.zig` or `…/declarations.zig`)
- Ensure the node type exists in `AstNode` union and has a struct definition

4) Semantics (`src/semantics/…`)
- Validate names, fields, and constraints

5) Example + test
- Add `ora-example/logs/basic_logs.ora`-like file using `event`
- Add a small test asserting the AST has the right node

### Useful commands
```sh
# Build everything
zig build

# Run CLI
zig-out/bin/ora lex ora-example/functions/basic_functions.ora
zig-out/bin/ora parse ora-example/statements/contract_declaration.ora
zig-out/bin/ora ast ora-example/enums/basic_enums.ora -o build

# Run tests
zig build test
```

### Troubleshooting
- “Cannot allocate” → ensure you’re using the arena’s allocator
- “Invalid tag” or switch fallthrough → update all `switch`es when adding enum/union tags
- “Use after free” → don’t keep AST pointers past arena `deinit()`
- Parser loops/recursion → spot precedence/associativity mistakes in `expression_parser.zig`

### Checklist (use this for any change)
- [ ] Tokens recognized in `src/lexer.zig`
- [ ] AST nodes/enums updated in `src/ast/…` and `src/ast.zig`
- [ ] Parser builds the right nodes in `src/parser/…`
- [ ] Semantics validate and/or annotate in `src/semantics/…`
- [ ] Examples added under `ora-example/`
- [ ] Tests updated in `tests/`
- [ ] `zig build && zig-out/bin/ora parse …` succeeds

If unsure, find a similar feature in the codebase and mirror its pattern.


