// ============================================================================
// AST Expressions Tests
// ============================================================================
// Tests for expression node creation, structure, and source span preservation
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;
const ast_builder = ora_root.ast_builder;

// ============================================================================
// Expression Node Creation Tests
// ============================================================================

test "expressions: integer literal creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 3, .byte_offset = 0 };
    const expr = try builder.integerLiteral("42", span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal) {
        try testing.expect(expr.Literal == .Integer);
        if (expr.Literal == .Integer) {
            try testing.expect(std.mem.eql(u8, "42", expr.Literal.Integer.value));
            try testing.expectEqual(span.line, expr.Literal.Integer.span.line);
            try testing.expectEqual(span.column, expr.Literal.Integer.span.column);
        }
    }
}

test "expressions: string literal creation" {
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
        if (expr.Literal == .String) {
            try testing.expect(std.mem.eql(u8, "hello", expr.Literal.String.value));
            try testing.expectEqual(span.line, expr.Literal.String.span.line);
        }
    }
}

test "expressions: boolean literal creation" {
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
            try testing.expectEqual(span.line, expr.Literal.Bool.span.line);
        }
    }
}

test "expressions: identifier creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const expr = try builder.identifier("x", span);

    try testing.expect(expr.* == .Identifier);
    if (expr.* == .Identifier) {
        try testing.expect(std.mem.eql(u8, "x", expr.Identifier.name));
        try testing.expectEqual(span.line, expr.Identifier.span.line);
        try testing.expectEqual(span.column, expr.Identifier.span.column);
    }
}

test "expressions: binary expression creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 3, .length = 1, .byte_offset = 2 };

    const lhs = try builder.identifier("a", span1);
    const rhs = try builder.identifier("b", span2);
    const expr = try builder.add(lhs, rhs, span_op);

    try testing.expect(expr.* == .Binary);
    if (expr.* == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Plus, expr.Binary.operator);
        try testing.expect(expr.Binary.lhs.* == .Identifier);
        try testing.expect(expr.Binary.rhs.* == .Identifier);
        try testing.expectEqual(span_op.line, expr.Binary.span.line);
    }
}

test "expressions: unary expression creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_operand = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };

    const operand = try builder.identifier("x", span_operand);
    const expr = try builder.unary(.Minus, operand, span_op);

    try testing.expect(expr.* == .Unary);
    if (expr.* == .Unary) {
        try testing.expectEqual(ast.Expressions.UnaryOp.Minus, expr.Unary.operator);
        try testing.expect(expr.Unary.operand.* == .Identifier);
        try testing.expectEqual(span_op.line, expr.Unary.span.line);
    }
}

test "expressions: nested binary expression" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 1, .column = 3, .length = 1, .byte_offset = 2 };
    const span3 = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_op1 = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };
    const span_op2 = ast.SourceSpan{ .line = 1, .column = 4, .length = 1, .byte_offset = 3 };

    const a = try builder.identifier("a", span1);
    const b = try builder.identifier("b", span2);
    const c = try builder.identifier("c", span3);

    const inner = try builder.add(a, b, span_op1);
    const outer = try builder.multiply(inner, c, span_op2);

    try testing.expect(outer.* == .Binary);
    if (outer.* == .Binary) {
        try testing.expectEqual(ast.Expressions.BinaryOp.Star, outer.Binary.operator);
        try testing.expect(outer.Binary.lhs.* == .Binary);
        try testing.expect(outer.Binary.rhs.* == .Identifier);
    }
}

// ============================================================================
// Source Span Preservation Tests
// ============================================================================

test "expressions: source span preserved in literals" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 5, .column = 10, .length = 2, .byte_offset = 100 };
    const expr = try builder.integerLiteral("99", span);

    try testing.expect(expr.* == .Literal);
    if (expr.* == .Literal and expr.Literal == .Integer) {
        try testing.expectEqual(@as(u32, 5), expr.Literal.Integer.span.line);
        try testing.expectEqual(@as(u32, 10), expr.Literal.Integer.span.column);
        try testing.expectEqual(@as(u32, 2), expr.Literal.Integer.span.length);
        try testing.expectEqual(@as(u32, 100), expr.Literal.Integer.span.byte_offset);
    }
}

test "expressions: source span preserved in identifiers" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 3, .column = 7, .length = 3, .byte_offset = 50 };
    const expr = try builder.identifier("foo", span);

    try testing.expect(expr.* == .Identifier);
    try testing.expectEqual(@as(u32, 3), expr.Identifier.span.line);
    try testing.expectEqual(@as(u32, 7), expr.Identifier.span.column);
    try testing.expectEqual(@as(u32, 3), expr.Identifier.span.length);
}

// ============================================================================
// Expression Structure Tests
// ============================================================================

test "expressions: binary expression structure" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 3, .length = 1, .byte_offset = 2 };

    const lhs = try builder.integerLiteral("10", span1);
    const rhs = try builder.integerLiteral("20", span2);
    const expr = try builder.add(lhs, rhs, span_op);

    try testing.expect(expr.* == .Binary);
    if (expr.* == .Binary) {
        // check lhs structure
        try testing.expect(expr.Binary.lhs.* == .Literal);
        if (expr.Binary.lhs.* == .Literal and expr.Binary.lhs.Literal == .Integer) {
            try testing.expect(std.mem.eql(u8, "10", expr.Binary.lhs.Literal.Integer.value));
        }

        // check rhs structure
        try testing.expect(expr.Binary.rhs.* == .Literal);
        if (expr.Binary.rhs.* == .Literal and expr.Binary.rhs.Literal == .Integer) {
            try testing.expect(std.mem.eql(u8, "20", expr.Binary.rhs.Literal.Integer.value));
        }
    }
}

test "expressions: unary expression structure" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_operand = ast.SourceSpan{ .line = 1, .column = 2, .length = 1, .byte_offset = 1 };
    const span_op = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };

    const operand = try builder.integerLiteral("5", span_operand);
    const expr = try builder.unary(.Minus, operand, span_op);

    try testing.expect(expr.* == .Unary);
    if (expr.* == .Unary) {
        try testing.expect(expr.Unary.operand.* == .Literal);
        if (expr.Unary.operand.* == .Literal and expr.Unary.operand.Literal == .Integer) {
            try testing.expect(std.mem.eql(u8, "5", expr.Unary.operand.Literal.Integer.value));
        }
    }
}

// ============================================================================
// Expression Type Information Tests
// ============================================================================

test "expressions: type info present in expressions" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const expr = try builder.identifier("x", span);

    try testing.expect(expr.* == .Identifier);
    // type info should be present (even if unknown initially)
    _ = expr.Identifier.type_info;
}

test "expressions: binary expression has type info" {
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
        // type info should be present
        _ = expr.Binary.type_info;
    }
}
