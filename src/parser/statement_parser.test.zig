// ============================================================================
// Statement Parser Tests
// ============================================================================
// Tests for parsing statements from tokens
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const StatementParser = ora_root.parser.parser_mod.StatementParser;
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;

// ============================================================================
// Variable Declaration Tests
// ============================================================================

test "statements: variable declaration with type" {
    const allocator = testing.allocator;
    const source = "var x: u256;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .VariableDecl);
    if (stmt == .VariableDecl) {
        try testing.expect(std.mem.eql(u8, "x", stmt.VariableDecl.name));
    }
}

test "statements: variable declaration with initializer" {
    const allocator = testing.allocator;
    const source = "var x: u256 = 100;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .VariableDecl);
    if (stmt == .VariableDecl) {
        try testing.expect(std.mem.eql(u8, "x", stmt.VariableDecl.name));
        try testing.expect(stmt.VariableDecl.value != null);
    }
}

test "statements: let declaration with type inference" {
    const allocator = testing.allocator;
    const source = "let x = 100;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .VariableDecl);
    if (stmt == .VariableDecl) {
        try testing.expect(std.mem.eql(u8, "x", stmt.VariableDecl.name));
        try testing.expect(stmt.VariableDecl.value != null);
    }
}

test "statements: storage variable declaration" {
    const allocator = testing.allocator;
    const source = "storage var counter: u256;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .VariableDecl);
    if (stmt == .VariableDecl) {
        try testing.expect(std.mem.eql(u8, "counter", stmt.VariableDecl.name));
        try testing.expectEqual(ast.Statements.MemoryRegion.Storage, stmt.VariableDecl.region);
    }
}

// ============================================================================
// Return Statement Tests
// ============================================================================

test "statements: return without value" {
    const allocator = testing.allocator;
    const source = "return;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Return);
    // return without value - value field may be optional or a special variant
    // check that it's a return statement
}

test "statements: return with value" {
    const allocator = testing.allocator;
    const source = "return 42;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Return);
    if (stmt == .Return) {
        try testing.expect(stmt.Return.value != null);
    }
}

test "statements: return with expression" {
    const allocator = testing.allocator;
    const source = "return x + y;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Return);
    if (stmt == .Return) {
        try testing.expect(stmt.Return.value != null);
        if (stmt.Return.value) |value| {
            try testing.expect(value == .Binary);
        }
    }
}

// ============================================================================
// Control Flow Tests
// ============================================================================

test "statements: if statement without else" {
    const allocator = testing.allocator;
    const source = "if (x > 0) { return 1; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .If);
    if (stmt == .If) {
        try testing.expect(stmt.If.else_branch == null);
    }
}

test "statements: if statement with else" {
    const allocator = testing.allocator;
    const source = "if (x > 0) { return 1; } else { return 0; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .If);
    if (stmt == .If) {
        try testing.expect(stmt.If.else_branch != null);
    }
}

test "statements: while loop" {
    const allocator = testing.allocator;
    const source = "while (x < 10) { x = x + 1; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .While);
    // while loop - verify it's a while statement
}

test "statements: for loop" {
    const allocator = testing.allocator;
    const source = "for (items) |item| { sum = sum + item; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .ForLoop);
    // for loop - verify it's a for loop statement
}

test "statements: break statement" {
    const allocator = testing.allocator;
    const source = "break;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Break);
}

test "statements: continue statement" {
    const allocator = testing.allocator;
    const source = "continue;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Continue);
}

// ============================================================================
// Expression Statement Tests
// ============================================================================

test "statements: expression statement (function call)" {
    const allocator = testing.allocator;
    const source = "foo();";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Expr);
    if (stmt == .Expr) {
        try testing.expect(stmt.Expr == .Call);
    }
}

test "statements: expression statement (assignment)" {
    const allocator = testing.allocator;
    const source = "x = 42;";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const stmt = try stmt_parser.parseStatement();

    try testing.expect(stmt == .Expr);
    if (stmt == .Expr) {
        try testing.expect(stmt.Expr == .Assignment);
    }
}

// ============================================================================
// Block Tests
// ============================================================================

test "statements: empty block" {
    const allocator = testing.allocator;
    const source = "{ }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const block = try stmt_parser.parseBlock();

    try testing.expect(block.statements.len == 0);
}

test "statements: block with statements" {
    const allocator = testing.allocator;
    const source = "{ var x: u256 = 1; var y: u256 = 2; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var stmt_parser = StatementParser.init(tokens, &arena);

    const block = try stmt_parser.parseBlock();

    try testing.expect(block.statements.len == 2);
}
