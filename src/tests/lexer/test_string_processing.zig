const std = @import("std");
const lexer = @import("src/lexer.zig");

test "string processing engine" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test regular string literal
    const source1 = "\"hello world\"";
    const tokens1 = try lexer.scan(source1, allocator);
    defer allocator.free(tokens1);
    
    try std.testing.expect(tokens1.len == 2); // string + EOF
    try std.testing.expect(tokens1[0].type == .StringLiteral);
    try std.testing.expectEqualStrings("hello world", tokens1[0].lexeme);

    // Test raw string literal
    const source2 = "r\"raw string with \\n\"";
    const tokens2 = try lexer.scan(source2, allocator);
    defer allocator.free(tokens2);
    
    try std.testing.expect(tokens2.len == 2); // raw string + EOF
    try std.testing.expect(tokens2[0].type == .RawStringLiteral);
    try std.testing.expectEqualStrings("raw string with \\n", tokens2[0].lexeme);

    // Test character literal
    const source3 = "'a'";
    const tokens3 = try lexer.scan(source3, allocator);
    defer allocator.free(tokens3);
    
    try std.testing.expect(tokens3.len == 2); // char + EOF
    try std.testing.expect(tokens3[0].type == .CharacterLiteral);
    try std.testing.expectEqualStrings("a", tokens3[0].lexeme);
    if (tokens3[0].value) |val| {
        try std.testing.expect(val.character == 'a');
    }

    // Test character literal with escape sequence
    const source4 = "'\\n'";
    const tokens4 = try lexer.scan(source4, allocator);
    defer allocator.free(tokens4);
    
    try std.testing.expect(tokens4.len == 2); // char + EOF
    try std.testing.expect(tokens4[0].type == .CharacterLiteral);
    try std.testing.expectEqualStrings("\\n", tokens4[0].lexeme);
    if (tokens4[0].value) |val| {
        try std.testing.expect(val.character == '\n');
    }
}