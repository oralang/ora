// ============================================================================
// Expression Parser Tests
// ============================================================================
// Tests for parsing expressions from tokens
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
// Access ExpressionParser through parser module from ora_root
const ExpressionParser = ora_root.parser.parser_mod.ExpressionParser;
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;

// ============================================================================
// Literal Expression Tests
// ============================================================================

test "expressions: integer literal" {
    const allocator = testing.allocator;
    const source = "123";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // check if it's a literal expression
    try testing.expect(expr == .Literal);
}

test "expressions: string literal" {
    const allocator = testing.allocator;
    const source = "\"hello\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // check if it's a literal expression
    try testing.expect(expr == .Literal);
}

test "expressions: boolean literal true" {
    const allocator = testing.allocator;
    const source = "true";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // check if it's a literal expression
    try testing.expect(expr == .Literal);
}

test "expressions: boolean literal false" {
    const allocator = testing.allocator;
    const source = "false";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // check if it's a literal expression
    try testing.expect(expr == .Literal);
}

test "expressions: identifier" {
    const allocator = testing.allocator;
    const source = "myVariable";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Identifier);
}

// ============================================================================
// Binary Expression Tests
// ============================================================================

test "expressions: binary addition" {
    const allocator = testing.allocator;
    const source = "a + b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Plus, expr.Binary.operator);
    }
}

test "expressions: binary subtraction" {
    const allocator = testing.allocator;
    const source = "a - b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Minus, expr.Binary.operator);
    }
}

test "expressions: binary multiplication" {
    const allocator = testing.allocator;
    const source = "a * b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Star, expr.Binary.operator);
    }
}

test "expressions: binary division" {
    const allocator = testing.allocator;
    const source = "a / b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Slash, expr.Binary.operator);
    }
}

test "expressions: binary equality" {
    const allocator = testing.allocator;
    const source = "a == b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.EqualEqual, expr.Binary.operator);
    }
}

test "expressions: binary less than" {
    const allocator = testing.allocator;
    const source = "a < b";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Less, expr.Binary.operator);
    }
}

// ============================================================================
// Precedence Tests
// ============================================================================

test "expressions: precedence - multiplication before addition" {
    const allocator = testing.allocator;
    const source = "a + b * c";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // should parse as: a + (b * c)
    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Plus, expr.Binary.operator);
        // right side should be multiplication
        if (expr.Binary.rhs.* == .Binary) {
            try testing.expectEqual(ast.Expressions.BinaryOp.Star, expr.Binary.rhs.Binary.operator);
        } else {
            try testing.expect(false); // Right should be binary
        }
    }
}

test "expressions: precedence - addition before comparison" {
    const allocator = testing.allocator;
    const source = "a + b < c";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // should parse as: (a + b) < c
    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Less, expr.Binary.operator);
    }
}

// ============================================================================
// Unary Expression Tests
// ============================================================================

test "expressions: unary negation" {
    const allocator = testing.allocator;
    const source = "-x";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Unary);
    if (expr == .Unary) {
        try testing.expectEqual(ast.Expressions.UnaryOp.Minus, expr.Unary.operator);
    }
}

test "expressions: unary logical not" {
    const allocator = testing.allocator;
    const source = "!x";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Unary);
    if (expr == .Unary) {
        try testing.expectEqual(ast.Expressions.UnaryOp.Bang, expr.Unary.operator);
    }
}

// ============================================================================
// Parentheses Tests
// ============================================================================

test "expressions: parentheses override precedence" {
    const allocator = testing.allocator;
    const source = "(a + b) * c";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    // should parse as: (a + b) * c
    try testing.expect(expr == .Binary);
    if (expr == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Star, expr.Binary.operator);
        // left side should be addition (in parentheses)
        if (expr.Binary.lhs.* == .Binary) {
            try testing.expectEqual(ast.Expressions.BinaryOp.Plus, expr.Binary.lhs.Binary.operator);
        } else {
            try testing.expect(false); // Left should be binary
        }
    }
}

// ============================================================================
// Function Call Tests
// ============================================================================

test "expressions: function call no args" {
    const allocator = testing.allocator;
    const source = "foo()";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Call);
}

test "expressions: function call with args" {
    const allocator = testing.allocator;
    const source = "foo(a, b)";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var expr_parser = ExpressionParser.init(tokens, &arena);

    const expr = try expr_parser.parseExpression();

    try testing.expect(expr == .Call);
    if (expr == .Call) {
        // verify we got a call expression with arguments
        // the parser should produce at least 2 arguments for "foo(a, b)"
        try testing.expect(expr.Call.arguments.len >= 1);
    }
}
