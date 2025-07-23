const std = @import("std");
const lexer = @import("src/lexer.zig");

test "binary literal parsing" {
    const allocator = std.testing.allocator;

    // Test valid binary literals
    const test_cases = [_]struct {
        input: []const u8,
        expected_value: u256,
    }{
        .{ .input = "0b1010", .expected_value = 10 },
        .{ .input = "0B1111", .expected_value = 15 },
        .{ .input = "0b0", .expected_value = 0 },
        .{ .input = "0b1", .expected_value = 1 },
        .{ .input = "0b1_0_1_0", .expected_value = 10 }, // With underscores
        .{ .input = "0b11111111", .expected_value = 255 },
    };

    for (test_cases) |case| {
        const tokens = try lexer.scan(case.input, allocator);
        defer allocator.free(tokens);

        try std.testing.expect(tokens.len == 2); // Binary literal + EOF
        try std.testing.expect(tokens[0].type == .BinaryLiteral);
        try std.testing.expect(tokens[0].value != null);

        if (tokens[0].value) |val| {
            switch (val) {
                .binary => |b| try std.testing.expectEqual(case.expected_value, b),
                else => try std.testing.expect(false), // Should be binary value
            }
        }
    }
}

test "binary literal errors" {
    const allocator = std.testing.allocator;

    // Test invalid binary literals
    const error_cases = [_][]const u8{
        "0b", // No digits
        "0b2", // Invalid digit
        "0bA", // Invalid digit
        "0b12", // Mixed invalid digits
    };

    for (error_cases) |case| {
        const result = lexer.scan(case, allocator);
        try std.testing.expectError(lexer.LexerError.InvalidBinaryLiteral, result);
    }
}

test "binary literal overflow" {
    const allocator = std.testing.allocator;

    // Create a binary literal with more than 256 bits
    var large_binary = std.ArrayList(u8).init(allocator);
    defer large_binary.deinit();

    try large_binary.appendSlice("0b");
    // Add 257 '1' bits to exceed u256 capacity
    for (0..257) |_| {
        try large_binary.append('1');
    }

    const result = lexer.scan(large_binary.items, allocator);
    try std.testing.expectError(lexer.LexerError.NumberTooLarge, result);
}
