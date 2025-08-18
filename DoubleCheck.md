## Double-check: Lexer, Parser, and AST vs GRAMMAR.bnf / GRAMMAR.ebnf

Scope: Cross-verify every grammar element against the current implementation in `src/lexer.zig`, `src/parser/**`, and `src/ast/**`. Status values: Implemented, Implemented with differences, Partially implemented, Not implemented.

### Top-level program structure
- program ::= top_level_declaration*
  - Implemented: Yes. Parsed in `Parser.parse()` by looping `parseTopLevel()`.
  - Cite:
    ```1:69:src/parser/parser_core.zig
    pub fn parse(self: *Parser) ParserError![]AstNode {
        var nodes = std.ArrayList(AstNode).init(self.arena.allocator());
        defer nodes.deinit();
        while (!self.isAtEnd()) {
            if (self.check(.Eof)) break;
            const node = try self.parseTopLevel();
            try nodes.append(node);
        }
        return nodes.toOwnedSlice();
    }
    ```

- top_level_declaration (BNF lists: contract_declaration | function_declaration | const_declaration | struct_declaration | enum_declaration | log_declaration | import_declaration | error_declaration)
  - Implemented: Mostly. Parser supports contract, function, variable (covers const/immutable), struct, enum, log, import. Error declarations appear supported inside contracts and as statements, not clearly at top-level.
  - Note: BNF still lists `const_declaration`; EBNF says it was merged into `VariableDeclaration`. Code follows the unified variable direction (no separate `const` declaration).
  - Cite:
    ```70:132:src/parser/parser_core.zig
    fn parseTopLevel(self: *Parser) ParserError!AstNode {
        if (self.match(.At)) { ... parseImport() }
        if (self.check(.Const)) { ... parseConstImport() }
        if (self.match(.Contract)) { ... parseContract() }
        if (self.check(.Pub) or self.check(.Fn)) { ... parseFunction() }
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) { ... parseVariableDecl() }
        if (self.match(.Struct)) { ... parseStruct() }
        if (self.match(.Enum)) { ... parseEnum() }
        if (self.match(.Log)) { ... parseLogDecl() }
        return self.errorAtCurrent("Expected top-level declaration");
    }
    ```

### Import declarations
- import_declaration ::= "@" "import" "(" string_literal ")" ";"
  - Lexer: `@` as `.At`, `import` keyword present.
  - Parser: Implemented, but semicolon is not enforced for `@import(...)`. For `const name = @import(...)`, semicolon is optional.
  - Status: Implemented with differences (semicolon handling).
  - Cite:
    ```2550:2660:src/lexer.zig
    fn scanAtDirective(self: *Lexer) ... if (std.mem.eql(u8, directive_name, "import")) { ... add .At then .Import }
    ```
    ```42:59:src/parser/declaration_parser.zig
    pub fn parseImport(self: *DeclarationParser) !ast.AstNode {
        const import_token = self.base.previous();
        _ = try self.base.consume(.Import, "Expected 'import' after '@'");
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'import'");
        const path_token = try self.base.consume(.StringLiteral, "Expected string path");
        _ = try self.base.consume(.RightParen, "Expected ')' after import path");
        return ast.AstNode{ .Import = ast.ImportNode{ ... } };
    }
    ```

### Contract declarations
- contract_declaration ::= "contract" identifier "{" contract_member* "}"
  - Implemented: Yes. Also supports ad hoc `extends`/`implements` as plain identifiers (not keywords).
  - Status: Implemented with differences (extends/implements are not keywords).
  - Cite:
    ```104:158:src/parser/declaration_parser.zig
    pub fn parseContract(self: *DeclarationParser) !ast.AstNode { ... optional 'extends'/'implements' by identifier ... parseContractMember()* ... }
    ```

- contract_member ::= variable_declaration | function_declaration | log_declaration | struct_declaration | enum_declaration | error_declaration
  - Implemented: Yes.
  - Cite:
    ```530:592:src/parser/declaration_parser.zig
    pub fn parseContractMember(self: *DeclarationParser) !ast.AstNode { ... Function | VariableDecl | ErrorDecl | LogDecl | StructDecl | EnumDecl ... }
    ```

### Function declarations
- function_declaration ::= inline_modifier? visibility? "fn" function_name "(" parameter_list? ")" return_type? requires_clause* ensures_clause* block
  - Inline/visibility: Implemented (`inline`, `pub`).
  - Name: identifier or `init`: Implemented.
  - Parameters: Implemented (`name: type`, defaults supported).
  - Return type: Grammar uses `-> type`. Parser accepts space-separated type (e.g., `fn f() u32 {}`) and defaults to void; it does not require/consume `->`.
  - requires/ensures: Implemented with parentheses. Parser permits optional semicolons; grammar says no semicolons.
  - Status: Implemented with differences (return type syntax and semicolons).
  - Cite:
    ```160:276:src/parser/declaration_parser.zig
    // parseFunction: handles Inline, Pub, Fn, name (identifier|init), params, return type via TypeParser without '->', requires/ensures with optional ';', and body
    ```

- parameter ::= identifier ":" type
  - Implemented: Yes (`parseParameter`, `parseParameterWithDefaults`).

- requires_clause / ensures_clause
  - Implemented: Yes; semicolons optional (difference vs grammar).
  - Cite:
    ```219:259:src/parser/declaration_parser.zig
    while (self.base.match(.Requires)) { ... _ = self.base.match(.Semicolon); }
    while (self.base.match(.Ensures)) { ... _ = self.base.match(.Semicolon); }
    ```

- quantified_expression, quantifier ::= "forall" | "exists"
  - Lexer/AST: Tokens exist; AST node `QuantifiedExpr` exists.
  - Parser: No parsing support found for `forall/exists` forms.
  - Status: Partially implemented (lexer+AST only).
  - Cite:
    ```1098:1102:src/lexer.zig
    // Forall, Exists, Where tokens
    ```
    ```178:192:src/ast/expressions.zig
    pub const QuantifiedExpr = struct { ... }
    ```
    ```1:300:src/parser/expression_parser.zig
    // no handling for Forall/Exists tokens
    ```

### Variable declarations
- variable_declaration ::= memory_region? variable_kind identifier (":" type)? ("=" expression)? ";"
  - Implemented: Yes. Supports `storage|memory|tstore` and `var|let|const|immutable`. Requires type annotation or initializer; allows tuple destructuring.
  - Semicolon: Optional in parser; grammar requires it.
  - Status: Implemented with differences (semicolon optional; requires type-or-init).
  - Cite:
    ```673:753:src/parser/declaration_parser.zig
    fn parseVariableDeclWithLock(...) { ... name : type? or '=' ... _ = self.base.match(.Semicolon); }
    ```

### Struct declarations
- struct_declaration, struct_member
  - Implemented: Yes; fields use `name: type;`.
  - Cite:
    ```283:333:src/parser/declaration_parser.zig
    pub fn parseStruct(...) { ... field_name ':' type ';' ... }
    ```

### Enum declarations
- enum_declaration, enum_member
  - Implemented: Yes; optional base type with `:`, optional `= expression`, commas, and implicit values.
  - Cite:
    ```379:483:src/parser/declaration_parser.zig
    pub fn parseEnum(...) { ... optional ':' type ... variant '=' expression? ... }
    ```

### Log declarations (events)
- log_declaration ::= "log" identifier "(" parameter_list? ")" ";"
  - Implemented: Yes; supports `indexed` qualifier; semicolon optional (difference).
  - Status: Implemented with differences (semicolon optional).
  - Cite:
    ```486:529:src/parser/declaration_parser.zig
    pub fn parseLogDecl(...) { ... _ = self.base.match(.Semicolon); }
    ```

### Error declarations
- error_declaration ::= "error" identifier ("(" parameter_list? ")")? ";"
  - Implemented: Yes; optional params, semicolon optional.
  - Status: Implemented with differences (semicolon optional).
  - Cite:
    ```792:821:src/parser/declaration_parser.zig
    fn parseErrorDecl(...) { ... parameters? ... _ = self.base.match(.Semicolon); }
    ```

### Type system
- error_prefix_type ::= "!" type
  - Implemented: Yes (`parseErrorUnionType` handles `!T`).
  - Cite:
    ```292:307:src/parser/type_parser.zig
    fn parseErrorUnionType(...) { ... category = .ErrorUnion ... }
    ```

- error_union_type ::= type ("|" type)+
  - Implemented: No (no union type with `|` in parser). Lexer provides `|` as bitwise operator only.
  - Status: Not implemented.

- primitive_type (u8..i256, bool, address, string, bytes, void)
  - Implemented: Yes.
  - Cite:
    ```50:76:src/parser/type_parser.zig
    if (self.base.match(.U8)) ... if (self.base.match(.Void)) { ... }
    ```

- map_type ::= "map" "[" type "," type "]"
  - Implemented: Yes.
  - Cite:
    ```104:133:src/parser/type_parser.zig
    fn parseMapType(...) { ... }
    ```

- doublemap_type ::= "doublemap" "[" type "," type "," type "]"
  - Implemented: Yes.
  - Cite:
    ```135:172:src/parser/type_parser.zig
    fn parseDoubleMapType(...) { ... }
    ```

- array_type (Revised): "[" Expression "]" type | "[]" type
  - Implemented: No. Parser uses older form `[Type; N]` for fixed-size and `[Type]` for dynamic slice, mapped to `Slice` category.
  - Status: Not implemented (for revised syntax).
  - Cite:
    ```174:211:src/parser/type_parser.zig
    fn parseArrayType(...) { // parses [T; N] and [T] -> Slice }
    ```

- slice_type ::= "slice" "[" type "]"
  - Implemented: Yes.
  - Cite:
    ```241:258:src/parser/type_parser.zig
    fn parseSliceType(...) { ... }
    ```

- anonymous_struct_type ::= "struct" "{" ... "}"
  - Implemented: No (as a type). Anonymous struct literals exist as expressions, not as a type in `TypeParser`.
  - Status: Not implemented.

### Statements
- statement set (variable_declaration, assignment, compound_assignment, destructuring_assignment, move_statement, expression_statement, if/while/for, switch_statement, return, break, continue, log, lock, unlock, try, block)
  - General: `StatementParser.parseStatement()` dispatches across all; many are implemented.
  - Cite:
    ```42:63:src/parser/statement_parser.zig
    pub fn parseStatement(self: *StatementParser) !ast.StmtNode { ... }
    ```

- assignment_statement ::= lvalue "=" expression ";"
  - Parser: Assignments are expressions; `parseAssignment()` validates LHS as LValue. Semicolon enforced only when used as a statement; expression statements accept `;` optionally.
  - Status: Implemented with differences (semicolon optional at statement level in many places).
  - Cite:
    ```91:117:src/parser/expression_parser.zig
    if (self.base.match(.Equal)) { validateLValue ... return .Assignment }
    ```

- compound_assignment_statement ::= lvalue compound_operator expression ";"
  - Implemented (both as expression form and dedicated statement recognizer). Semicolon enforced in statement parser path.
  - Status: Implemented.
  - Cite:
    ```119:159:src/parser/expression_parser.zig
    if (self.base.match(.PlusEqual) ... ) return .CompoundAssignment
    ```
    ```807:855:src/parser/statement_parser.zig
    fn tryParseCompoundAssignmentStatement(...) { ... ';' required }
    ```

- destructuring_assignment ::= "let" "." "{" ... "}" "=" expression ";"
  - Implemented: Yes (struct pattern). Semicolon optional.
  - Cite:
    ```354:450:src/parser/statement_parser.zig
    fn tryParseDestructuringAssignment(...) { ... 'let .{ ... } = expr' ... _ = self.base.match(.Semicolon); }
    ```

- move_statement (Revised): "move" expression "from" expression "to" expression ";"
  - Implemented: Old syntax only: `expr from source -> dest : amount;`. There is no `move` keyword and no `to` keyword.
  - Status: Not implemented (revised form); legacy form present.
  - Cite:
    ```453:503:src/parser/statement_parser.zig
    fn tryParseMoveStatement(...) { expr ... 'from' ... '->' ... ':' ... }
    ```

- expression_statement ::= expression ";"
  - Implemented: Yes (semicolon handling tends to be optional across statements).

- if_statement, while_statement
  - Implemented: Yes.

- for_statement ::= "for" "(" expression ")" "|" identifier ("," identifier)? "|" statement
  - Implemented: Yes; body is parsed as a block (subset of statement).
  - Cite:
    ```652:657:src/parser/statement_parser.zig
    return ast.StmtNode{ .ForLoop = ... .body = body }
    ```

- switch_statement
  - Implemented: Yes; supports patterns (literal, range with `...`, enum variants) and `else`. Optional commas between arms. Default case must be a block due to AST shape.
  - Status: Implemented with differences (default expression body not supported).
  - Cite:
    ```660:721:src/parser/statement_parser.zig
    fn parseSwitchStatement(...) { ... parseSwitchPattern ... parseSwitchBody ... default_case block only }
    ```

- return_statement ::= "return" expression? ";"
  - Implemented: Yes; semicolon optional.
  - Cite:
    ```200:238:src/parser/statement_parser.zig
    fn parseReturnStatement(...) { ... _ = self.base.match(.Semicolon); }
    ```

- break_statement ::= "break" (":" identifier expression?)? ";"
  - Implemented with differences: Parser also allows `break <expr>` without label; semicolon optional.
  - Cite:
    ```723:753:src/parser/statement_parser.zig
    // optional ':' label, optional value, ';' optional
    ```

- continue_statement ::= "continue" (":" identifier)? ";"
  - Implemented: Yes; semicolon optional.

- log_statement ::= "log" identifier "(" expression_list? ")" ";"
  - Implemented: Yes; semicolon optional.
  - Cite:
    ```240:267:src/parser/statement_parser.zig
    fn parseLogStatement(...) { ... _ = self.base.match(.Semicolon); }
    ```

- lock_statement / unlock_statement
  - unlock_statement: Implemented as `@unlock(expr);` (semicolon optional).
  - lock_statement: Not implemented as a standalone statement. `@lock` is only handled as an annotation before variable declarations.
  - Cite:
    ```505:520:src/parser/statement_parser.zig
    fn tryParseUnlockAnnotation(...) { '@unlock(' expr ')' }
    ```
    ```326:344:src/parser/statement_parser.zig
    fn tryParseLockAnnotation(...) { only as prefix to variable decl }
    ```

- try_statement ::= "try" expression ("catch" ("|" identifier "|")? block)?
  - Implemented: Different shape. Parser has a statement form `try { ... } catch (...) { ... }` and an expression form `try <expr>`. The exact grammar form (try <expr> catch block) is not matched for statements.
  - Status: Implemented with differences.
  - Cite:
    ```773:802:src/parser/statement_parser.zig
    fn parseTryStatement(...) { try-block form }
    ```
    ```908:920:src/parser/expression_parser.zig
    if (self.base.match(.Try)) { return ast.ExprNode{ .Try = ... } }
    ```

- block / labeled_block
  - Implemented: Yes (block nodes; labeled blocks supported in parser).

### Switch expressions
- switch_expression ::= "switch" "(" expression ")" "{" switch_expr_arm* "}"
  - Implemented: Yes; similar to statement form. Default case again must be a block due to AST structure.
  - Cite:
    ```1091:1130:src/parser/expression_parser.zig
    fn parseSwitchExpression(...) { ... }
    ```

### Expressions and precedence
- assignment_expression, logical_or_expression (||), logical_and_expression (&&), bitwise (|, ^, &), equality (==, !=), relational (<, <=, >, >=), additive (+, -), multiplicative (*, /, %), unary ((! | - | +)*), postfix, primary
  - Implemented: Yes for the full chain except unary `+` (not handled).
  - LValue rule: Implemented via explicit validation before `=` and compound assignments.
  - Additional operator: `**` exponentiation is supported (not in grammar).
  - Status: Implemented with differences (unary `+` missing; extra `**`).
  - Cite:
    ```494:520:src/parser/expression_parser.zig
    fn parseUnary(...) { handles '!' and '-' only }
    ```
    ```467:491:src/parser/expression_parser.zig
    fn parseExponent(...) { matches '**' right-associative }
    ```

### Postfix operators
- postfix: .identifier, [expression], (argument_list?)
  - Implemented: Yes (field access, indexing, call).
  - Cite:
    ```600:633:src/parser/expression_parser.zig
    // handles '[' index ']', '.' field, and casts via 'as'
    ```

### Primary expressions
- literal, identifier, "(" expression ")", old(expr), comptime block, cast_expression, error_expression, quantified_expression, anonymous_struct_literal, switch_expression, array_literal
  - Literals/identifier/parentheses: Implemented.
  - old(expr): Implemented.
  - comptime block: Implemented.
  - cast_expression: Grammar uses `@cast(Type, Expr)`. Parser uses `expr as Type`. Lexer has `@` built-in functions but not `@cast` support; builtin table includes only `@div...` variants.
  - error_expression: Implemented as `error.Name`.
  - quantified_expression: Not implemented in parser.
  - anonymous_struct_literal: Implemented as `.{ field = value }` (without leading dot on field names). Grammar requires `.{ .field = value }` (dot before field); difference.
  - switch_expression: Implemented.
  - array_literal: Implemented.
  - Status: Implemented with differences / partial (cast and quantified, and anonymous struct fields syntax).
  - Cite:
    ```950:959:src/parser/expression_parser.zig
    // comptime block
    ```
    ```961:971:src/parser/expression_parser.zig
    // error.Name
    ```
    ```625:665:src/parser/expression_parser.zig
    // 'expr as Type' cast
    ```
    ```1356:1396:src/parser/expression_parser.zig
    // anonymous struct literal expects 'field = value'
    ```

### Literals
- integer_literal, string_literal, boolean_literal, address_literal, hex_literal
  - Lexer: Implemented (also supports BinaryLiteral and Character/Raw strings beyond grammar).
  - Address literal: Lexer interprets `0x` followed by hex; if exactly 40 hex digits, it becomes AddressLiteral. Underscores are accepted in scanning, which deviates from grammar (grammar disallows underscores).
  - Status: Implemented with differences (extra literal forms; underscores allowed in hex/address).
  - Cite:
    ```2291:2327:src/lexer.zig
    fn scanHexLiteral(...) { ... digit_count ... if (digit_count == 40) addAddressToken() else addHexToken() }
    ```

### Identifiers
- identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
  - Implemented: Yes in lexer and used pervasively.

### Additional observations vs grammar
- Semicolons: The revised grammar removes optional semicolons in many places; the parser generally treats semicolons as optional (imports, requires/ensures, log/error decls, return/break/continue, etc.).
- Move statement: Grammar update not reflected; code still uses legacy `from -> :` pattern without `move`/`to`.
- Return type arrow: Grammar mandates `->` for function return types; parser does not use `->`.
- Array type syntax: Grammar’s `[N]T` / `[]T` not supported; code uses `[T; N]` and `[T]`.
- Quantifiers: Tokens and AST exist, parsing is missing.
- Anonymous struct type: Missing in type parser; anonymous struct literal field syntax differs (no leading dot).
- Try statement shape: Grammar uses `try <expr> [catch ... block]`; code also supports a statement-form try/catch block.
- Switch default arm: Parser requires block for default; grammar allows expression.
- Lock statement: Only `@unlock(expr)` exists as a statement; `@lock(expr)` is treated as a variable decl annotation.
- Operators: Unary `+` missing; exponentiation `**` added (not in grammar).

### Summary status by category
- Top-level: Implemented (minor differences around `const` naming and import semicolon).
- Declarations: Implemented with differences (return type arrow, semicolons).
- Types: Partially implemented (no union `|` types; different array syntax; no anonymous struct types).
- Statements: Largely implemented with differences (move/lock/try shapes; semicolons optional; break with value without label allowed).
- Expressions: Implemented with differences (cast syntax, quantifiers missing, anonymous struct fields syntax, unary `+` missing, extra `**`).
- Literals: Implemented plus extras (binary/char/raw); address/hex underscore tolerance differs.




Consistent naming between suite names, function names, and test case labels.
Consider normalizing expected token counts across tests (some use tokens.len < 2, others subtract EOF—both fine, but standardize).
Building large strings with allocPrint in a loop is fine with an arena, but you can reduce allocations by formatting into a reusable buffer or using writer to large_file.

Gaps to consider covering
Strings/escapes: Escaped quotes, backslashes, newline escapes, invalid escape sequences, unterminated with backslash at EOL.
Comments: Nested comments (if supported), unterminated mixed with code, comments after tokens same-line, doc comments differences.

Whitespace/tabs: Column correctness with tabs, CRLF handling.


Fixtures: You already have tests/fixtures/lexer. Incorporate them into assertions to ensure real sample programs tokenize without diagnostics.




Custom runner not used by Zig test runner