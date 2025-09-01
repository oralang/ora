const std = @import("std");
const ora = @import("ora");

test "verification attributes creation" {
    // Test creating verification attributes
    const allocator = std.testing.allocator;

    // Create a quantified metadata
    const metadata = try allocator.create(ora.ast.verification.QuantifiedMetadata);
    defer allocator.destroy(metadata);
    metadata.* = ora.ast.verification.QuantifiedMetadata.init(.Forall, "x", ora.ast.type_info.TypeInfo.explicit(.Integer, .u256, .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 }), .{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 });

    // Create verification attributes
    var attributes = std.ArrayList(ora.ast.verification.VerificationAttribute).init(allocator);
    defer attributes.deinit();

    try attributes.append(ora.ast.verification.VerificationAttribute{
        .attr_type = .Quantified,
        .name = "ora.quantified",
        .value = "true",
        .span = .{ .line = 1, .column = 1, .length = 15, .byte_offset = 0 },
    });

    try attributes.append(ora.ast.verification.VerificationAttribute{
        .attr_type = .Assertion,
        .name = "ora.assertion",
        .value = "invariant",
        .span = .{ .line = 2, .column = 1, .length = 15, .byte_offset = 0 },
    });

    // Verify the attributes were created correctly
    try std.testing.expect(attributes.items.len == 2);
    try std.testing.expect(attributes.items[0].attr_type == .Quantified);
    try std.testing.expect(attributes.items[1].attr_type == .Assertion);
    try std.testing.expect(std.mem.eql(u8, attributes.items[0].name.?, "ora.quantified"));
    try std.testing.expect(std.mem.eql(u8, attributes.items[1].name.?, "ora.assertion"));

    // Test verification context
    var context = ora.ast.verification.VerificationContext.init(allocator);
    defer context.deinit();

    try context.addAttribute(attributes.items[0]);
    try context.addAttribute(attributes.items[1]);

    try std.testing.expect(context.current_attributes.items.len == 2);
    try std.testing.expect(context.mode == .None);
}

test "quantified expression with verification metadata" {
    const allocator = std.testing.allocator;

    // Create a simple body expression
    const body = try ora.ast.expressions.createUntypedIntegerLiteral(allocator, "0", .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 });
    defer allocator.destroy(body);

    // Create verification metadata
    const metadata = try allocator.create(ora.ast.verification.QuantifiedMetadata);
    defer allocator.destroy(metadata);
    metadata.* = ora.ast.verification.QuantifiedMetadata.init(.Forall, "x", ora.ast.type_info.TypeInfo.explicit(.Integer, .u256, .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 }), .{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 });

    // Create verification attributes
    var attributes = std.ArrayList(ora.ast.verification.VerificationAttribute).init(allocator);
    defer attributes.deinit();

    try attributes.append(ora.ast.verification.VerificationAttribute{
        .attr_type = .Quantified,
        .name = "ora.quantified",
        .value = "true",
        .span = .{ .line = 1, .column = 1, .length = 15, .byte_offset = 0 },
    });

    // Create quantified expression with verification metadata
    const quantified_expr = try ora.ast.expressions.createQuantifiedExprWithVerification(allocator, .Forall, "x", ora.ast.type_info.TypeInfo.explicit(.Integer, .u256, .{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 }), null, // no condition
        body, .{ .line = 1, .column = 1, .length = 20, .byte_offset = 0 }, metadata, attributes.items);
    defer allocator.destroy(quantified_expr);

    // Verify the expression was created correctly
    try std.testing.expect(quantified_expr.* == .Quantified);
    const quant = quantified_expr.Quantified;
    try std.testing.expect(quant.quantifier == .Forall);
    try std.testing.expect(std.mem.eql(u8, quant.variable, "x"));
    try std.testing.expect(quant.verification_metadata != null);
    try std.testing.expect(quant.verification_attributes.len == 1);
    try std.testing.expect(quant.verification_attributes[0].attr_type == .Quantified);
}
