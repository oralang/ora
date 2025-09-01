const std = @import("std");
const lib = @import("ora");

test "quantified expression AST creation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a forall expression: forall x: u256 => (x > 0)
    const variable_type = lib.ast.type_info.CommonTypes.u256_type();

    // Create body expression: x > 0
    const x_ident = try lib.ast.Expressions.createIdentifier(allocator, "x", lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 1,
        .byte_offset = 0,
    });

    const zero_literal = try lib.ast.Expressions.createUntypedIntegerLiteral(allocator, "0", lib.ast.SourceSpan{
        .line = 1,
        .column = 5,
        .length = 1,
        .byte_offset = 4,
    });

    const body_expr = try lib.ast.Expressions.createBinaryExpr(allocator, x_ident, .Greater, zero_literal, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 5,
        .byte_offset = 0,
    });

    // Create quantified expression
    const quantified_expr = try lib.ast.Expressions.createQuantifiedExpr(allocator, .Forall, "x", variable_type, null, // no condition
        body_expr, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 20,
        .byte_offset = 0,
    });

    // Verify the quantified expression was created correctly
    try std.testing.expect(quantified_expr.* == .Quantified);

    const quant = quantified_expr.Quantified;
    try std.testing.expect(std.mem.eql(u8, quant.variable, "x"));
    try std.testing.expect(quant.quantifier == .Forall);
    try std.testing.expect(quant.condition == null); // No condition specified
    try std.testing.expect(quant.variable_type.ora_type.? == .u256);

    // Verify the body expression is a binary expression with Greater operator
    try std.testing.expect(quant.body.* == .Binary);
    const binary = quant.body.Binary;
    try std.testing.expect(binary.operator == .Greater);

    // Verify the left operand is an identifier "x"
    try std.testing.expect(binary.lhs.* == .Identifier);
    try std.testing.expect(std.mem.eql(u8, binary.lhs.Identifier.name, "x"));

    // Verify the right operand is a literal "0"
    try std.testing.expect(binary.rhs.* == .Literal);
    try std.testing.expect(binary.rhs.Literal == .Integer);
    try std.testing.expect(std.mem.eql(u8, binary.rhs.Literal.Integer.value, "0"));

    std.debug.print("Quantified expression AST creation test passed\n", .{});
}

test "quantified expression with condition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a forall expression with condition: forall x: u256 where x > 0 => (x < 100)
    const variable_type = lib.ast.type_info.CommonTypes.u256_type();

    // Create condition: x > 0
    const x_ident = try lib.ast.Expressions.createIdentifier(allocator, "x", lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 1,
        .byte_offset = 0,
    });

    const zero_literal = try lib.ast.Expressions.createUntypedIntegerLiteral(allocator, "0", lib.ast.SourceSpan{
        .line = 1,
        .column = 5,
        .length = 1,
        .byte_offset = 4,
    });

    const condition_expr = try lib.ast.Expressions.createBinaryExpr(allocator, x_ident, .Greater, zero_literal, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 5,
        .byte_offset = 0,
    });

    // Create body: x < 100
    const x_ident2 = try lib.ast.Expressions.createIdentifier(allocator, "x", lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 1,
        .byte_offset = 0,
    });

    const hundred_literal = try lib.ast.Expressions.createUntypedIntegerLiteral(allocator, "100", lib.ast.SourceSpan{
        .line = 1,
        .column = 5,
        .length = 3,
        .byte_offset = 4,
    });

    const body_expr = try lib.ast.Expressions.createBinaryExpr(allocator, x_ident2, .Less, hundred_literal, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 8,
        .byte_offset = 0,
    });

    // Create quantified expression with condition
    const quantified_expr = try lib.ast.Expressions.createQuantifiedExpr(allocator, .Forall, "x", variable_type, condition_expr, body_expr, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 30,
        .byte_offset = 0,
    });

    // Verify the quantified expression was created correctly
    try std.testing.expect(quantified_expr.* == .Quantified);

    const quant = quantified_expr.Quantified;
    try std.testing.expect(std.mem.eql(u8, quant.variable, "x"));
    try std.testing.expect(quant.quantifier == .Forall);
    try std.testing.expect(quant.condition != null); // Condition specified
    try std.testing.expect(quant.variable_type.ora_type.? == .u256);

    // Verify the condition expression is a binary expression with Greater operator
    try std.testing.expect(quant.condition.?.* == .Binary);
    const condition_binary = quant.condition.?.Binary;
    try std.testing.expect(condition_binary.operator == .Greater);

    // Verify the body expression is a binary expression with Less operator
    try std.testing.expect(quant.body.* == .Binary);
    const body_binary = quant.body.Binary;
    try std.testing.expect(body_binary.operator == .Less);

    std.debug.print("Quantified expression with condition test passed\n", .{});
}
