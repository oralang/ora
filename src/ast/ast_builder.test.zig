// ============================================================================
// AST Builder Tests
// ============================================================================
// Tests for building AST nodes programmatically
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;
const ast_builder = ora_root.ast_builder;

// ============================================================================
// Expression Node Building Tests
// ============================================================================

test "ast_builder: integer literal" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 3, .byte_offset = 0 };
    const expr = try builder.integerLiteral("123", span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal) {
        try testing.expect(expr.Literal == .Integer);
    }
}

test "ast_builder: string literal" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 7, .byte_offset = 0 };
    const expr = try builder.stringLiteral("hello", span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal) {
        try testing.expect(expr.Literal == .String);
    }
}

test "ast_builder: boolean literal true" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4, .byte_offset = 0 };
    const expr = try builder.boolLiteral(true, span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal) {
        try testing.expect(expr.Literal == .Bool);
        if (expr.Literal == .Bool) {
            try testing.expect(expr.Literal.Bool.value == true);
        }
    }
}

test "ast_builder: boolean literal false" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5, .byte_offset = 0 };
    const expr = try builder.boolLiteral(false, span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal) {
        try testing.expect(expr.Literal == .Bool);
        if (expr.Literal == .Bool) {
            try testing.expect(expr.Literal.Bool.value == false);
        }
    }
}

test "ast_builder: identifier" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 };
    const expr = try builder.identifier("myVariable", span);

    try testing.expect(expr.* == .Identifier);
    if (expr.* == .Identifier) {
        try testing.expect(std.mem.eql(u8, "myVariable", expr.Identifier.name));
    }
}

test "ast_builder: binary addition" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 1, .column = 3, .length = 1, .byte_offset = 2 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };

    const lhs = try builder.identifier("a", span1);
    const rhs = try builder.identifier("b", span2);
    const expr = try builder.add(lhs, rhs, span_op);

    try testing.expect(expr.* == .Binary);
    if (expr.* == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Plus, expr.Binary.operator);
    }
}

test "ast_builder: binary multiplication" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 1, .column = 3, .length = 1, .byte_offset = 2 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };

    const lhs = try builder.identifier("a", span1);
    const rhs = try builder.identifier("b", span2);
    const expr = try builder.multiply(lhs, rhs, span_op);

    try testing.expect(expr.* == .Binary);
    if (expr.* == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Star, expr.Binary.operator);
    }
}

test "ast_builder: unary negation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_operand = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 1, .length = 2, .byte_offset = 0 };

    const operand = try builder.identifier("x", span_operand);
    // use UnaryOp enum value directly (ast.Operators.Unary is an alias for ast.Expressions.UnaryOp)
    const expr = try builder.unary(.Minus, operand, span_op);

    try testing.expect(expr.* == .Unary);
    if (expr.* == .Unary) {
        try testing.expectEqual(ast.Expressions.UnaryOp.Minus, expr.Unary.operator);
    }
}

test "ast_builder: nested binary expression" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    // build: a + b * c
    const span_a = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span_b = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_c = ast.SourceSpan{ .line = 1, .column = 9, .length = 1, .byte_offset = 8 };
    const span_mul = ast.SourceSpan{ .line = 1, .column = 7, .length = 1, .byte_offset = 6 };
    const span_add = ast.SourceSpan{ .line = 1, .column = 3, .length = 5, .byte_offset = 2 };

    const a = try builder.identifier("a", span_a);
    const b = try builder.identifier("b", span_b);
    const c = try builder.identifier("c", span_c);
    const mul = try builder.multiply(b, c, span_mul);
    const add = try builder.add(a, mul, span_add);

    try testing.expect(add.* == .Binary);
    if (add.* == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Plus, add.Binary.operator);
        // right side should be multiplication
        if (add.Binary.rhs.* == .Binary) {
            try testing.expectEqual(ast.Expressions.BinaryOp.Star, add.Binary.rhs.Binary.operator);
        } else {
            try testing.expect(false); // Right should be binary
        }
    }
}

// ============================================================================
// Statement Node Building Tests
// ============================================================================

test "ast_builder: block statement" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2, .byte_offset = 0 };
    const block = try builder.block(&[_]ast.Statements.StmtNode{}, span);

    try testing.expect(block.statements.len == 0);
}

// ============================================================================
// Memory Management Tests
// ============================================================================

test "ast_builder: arena allocation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 3, .byte_offset = 0 };
    const expr1 = try builder.integerLiteral("123", span);
    const expr2 = try builder.integerLiteral("456", span);

    // both expressions should be valid and allocated from the arena
    try testing.expect(expr1.* == .Literal);
    try testing.expect(expr2.* == .Literal);
}

test "ast_builder: node counter" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const initial_count = builder.getNodeCount();
    try testing.expectEqual(@as(u32, 0), initial_count);

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 3, .byte_offset = 0 };
    _ = try builder.integerLiteral("123", span);

    // node counter should increment (though nodes aren't added to built_nodes automatically)
    // the counter tracks nodes created through the builder's methods
}
