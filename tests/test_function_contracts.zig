const std = @import("std");
const ora = @import("ora");

test "function contract verification with requires and ensures" {
    const allocator = std.testing.allocator;

    // Create a simple condition expression for requires clause
    const requires_lhs = try ora.ast.expressions.createIdentifier(allocator, "x", .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 });
    defer allocator.destroy(requires_lhs);
    const requires_rhs = try ora.ast.expressions.createUntypedIntegerLiteral(allocator, "0", .{ .line = 1, .column = 5, .length = 1, .byte_offset = 0 });
    defer allocator.destroy(requires_rhs);
    const requires_condition = try ora.ast.expressions.createBinaryExpr(allocator, requires_lhs, .Greater, requires_rhs, .{ .line = 1, .column = 1, .length = 5, .byte_offset = 0 });
    defer allocator.destroy(requires_condition);

    // Create a simple condition expression for ensures clause
    const ensures_lhs = try ora.ast.expressions.createIdentifier(allocator, "result", .{ .line = 2, .column = 1, .length = 6, .byte_offset = 0 });
    defer allocator.destroy(ensures_lhs);
    const ensures_rhs = try ora.ast.expressions.createUntypedIntegerLiteral(allocator, "0", .{ .line = 2, .column = 10, .length = 1, .byte_offset = 0 });
    defer allocator.destroy(ensures_rhs);
    const ensures_condition = try ora.ast.expressions.createBinaryExpr(allocator, ensures_lhs, .Greater, ensures_rhs, .{ .line = 2, .column = 1, .length = 10, .byte_offset = 0 });
    defer allocator.destroy(ensures_condition);

    // Create function parameters
    const param = ora.ast.ParameterNode{
        .name = "x",
        .type_info = ora.ast.type_info.TypeInfo.explicit(.Integer, .u256, .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 }),
        .is_mutable = false,
        .default_value = null,
        .span = .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 },
    };

    // Create function body (empty block for this test)
    const body = ora.ast.statements.BlockNode{
        .statements = &[_]ora.ast.statements.StmtNode{},
        .span = .{ .line = 3, .column = 1, .length = 10, .byte_offset = 0 },
    };

    // Create function with contracts
    var params = [_]ora.ast.ParameterNode{param};
    var requires_clauses = [_]*ora.ast.expressions.ExprNode{requires_condition};
    var ensures_clauses = [_]*ora.ast.expressions.ExprNode{ensures_condition};
    const function = ora.ast.FunctionNode{
        .name = "test_function",
        .parameters = params[0..],
        .return_type_info = ora.ast.type_info.TypeInfo.explicit(.Integer, .u256, .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 }),
        .body = body,
        .visibility = .Public,
        .attributes = &[_]u8{},
        .is_inline = false,
        .requires_clauses = requires_clauses[0..],
        .ensures_clauses = ensures_clauses[0..],
        .span = .{ .line = 1, .column = 1, .length = 50, .byte_offset = 0 },
    };

    // Verify the function was created correctly
    try std.testing.expect(std.mem.eql(u8, function.name, "test_function"));
    try std.testing.expect(function.parameters.len == 1);
    try std.testing.expect(std.mem.eql(u8, function.parameters[0].name, "x"));
    try std.testing.expect(function.requires_clauses.len == 1);
    try std.testing.expect(function.ensures_clauses.len == 1);
    try std.testing.expect(function.visibility == .Public);
    try std.testing.expect(!function.is_inline);

    // Verify the requires clause
    const requires_expr = function.requires_clauses[0];
    try std.testing.expect(requires_expr.* == .Binary);
    const requires_binary = requires_expr.Binary;
    try std.testing.expect(requires_binary.operator == .Greater);
    try std.testing.expect(requires_binary.lhs.* == .Identifier);
    try std.testing.expect(requires_binary.rhs.* == .Literal);

    // Verify the ensures clause
    const ensures_expr = function.ensures_clauses[0];
    try std.testing.expect(ensures_expr.* == .Binary);
    const ensures_binary = ensures_expr.Binary;
    try std.testing.expect(ensures_binary.operator == .Greater);
    try std.testing.expect(ensures_binary.lhs.* == .Identifier);
    try std.testing.expect(ensures_binary.rhs.* == .Literal);
}

test "function contract verification context" {
    const allocator = std.testing.allocator;

    // Create verification context
    var context = ora.ast.verification.VerificationContext.init(allocator);
    defer context.deinit();

    // Add verification attributes for function contracts
    try context.addAttribute(ora.ast.verification.VerificationAttribute{
        .attr_type = .Precondition,
        .name = "ora.requires",
        .value = "x > 0",
        .span = .{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 },
    });

    try context.addAttribute(ora.ast.verification.VerificationAttribute{
        .attr_type = .Postcondition,
        .name = "ora.ensures",
        .value = "result > 0",
        .span = .{ .line = 2, .column = 1, .length = 15, .byte_offset = 0 },
    });

    // Verify context contains the expected attributes
    try std.testing.expect(context.current_attributes.items.len == 2);
    try std.testing.expect(context.current_attributes.items[0].attr_type == .Precondition);
    try std.testing.expect(context.current_attributes.items[1].attr_type == .Postcondition);
    try std.testing.expect(std.mem.eql(u8, context.current_attributes.items[0].name.?, "ora.requires"));
    try std.testing.expect(std.mem.eql(u8, context.current_attributes.items[1].name.?, "ora.ensures"));
    try std.testing.expect(std.mem.eql(u8, context.current_attributes.items[0].value.?, "x > 0"));
    try std.testing.expect(std.mem.eql(u8, context.current_attributes.items[1].value.?, "result > 0"));
}
