const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "StringPool basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var pool = lexer.StringPool.init(allocator);
    defer pool.deinit();

    // Test interning the same string multiple times
    const str1 = try pool.intern("hello");
    const str2 = try pool.intern("hello");
    const str3 = try pool.intern("world");

    // Same strings should return the same pointer
    try testing.expect(str1.ptr == str2.ptr);
    try testing.expect(str1.ptr != str3.ptr);

    // Content should be correct
    try testing.expectEqualStrings("hello", str1);
    try testing.expectEqualStrings("hello", str2);
    try testing.expectEqualStrings("world", str3);

    // Pool should contain 2 unique strings
    try testing.expectEqual(@as(u32, 2), pool.count());
}

test "StringPool hash function" {
    const hash1 = lexer.StringPool.hash("hello");
    const hash2 = lexer.StringPool.hash("hello");
    const hash3 = lexer.StringPool.hash("world");

    // Same strings should have same hash
    try testing.expectEqual(hash1, hash2);
    // Different strings should have different hashes (very likely)
    try testing.expect(hash1 != hash3);
}

test "StringPool clear functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var pool = lexer.StringPool.init(allocator);
    defer pool.deinit();

    // Add some strings
    _ = try pool.intern("hello");
    _ = try pool.intern("world");
    try testing.expectEqual(@as(u32, 2), pool.count());

    // Clear the pool
    pool.clear();
    try testing.expectEqual(@as(u32, 0), pool.count());

    // Should be able to intern again after clearing
    const str = try pool.intern("hello");
    try testing.expectEqualStrings("hello", str);
    try testing.expectEqual(@as(u32, 1), pool.count());
}

test "Lexer with string interning enabled" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = hello; let y = hello; let z = world;";

    var config = lexer.LexerConfig.default();
    config.enable_string_interning = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Find the identifier tokens
    var hello_tokens: u32 = 0;
    var hello_ptr: ?[*]const u8 = null;
    var world_ptr: ?[*]const u8 = null;

    for (tokens) |token| {
        if (token.type == .Identifier) {
            if (std.mem.eql(u8, token.lexeme, "hello")) {
                hello_tokens += 1;
                if (hello_ptr == null) {
                    hello_ptr = token.lexeme.ptr;
                } else {
                    // All "hello" tokens should have the same pointer due to interning
                    try testing.expect(hello_ptr.? == token.lexeme.ptr);
                }
            } else if (std.mem.eql(u8, token.lexeme, "world")) {
                world_ptr = token.lexeme.ptr;
            }
        }
    }

    // Should have found 2 "hello" tokens
    try testing.expectEqual(@as(u32, 2), hello_tokens);

    // "hello" and "world" should have different pointers
    try testing.expect(hello_ptr != null);
    try testing.expect(world_ptr != null);
    try testing.expect(hello_ptr.? != world_ptr.?);
}

test "Lexer with string interning disabled" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = hello; let y = hello;";

    var config = lexer.LexerConfig.default();
    config.enable_string_interning = false;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Find the identifier tokens
    var hello_count: u32 = 0;

    for (tokens) |token| {
        if (token.type == .Identifier and std.mem.eql(u8, token.lexeme, "hello")) {
            hello_count += 1;
            // Content should be correct
            try testing.expectEqualStrings("hello", token.lexeme);
        }
    }

    // Should have found 2 "hello" tokens
    try testing.expectEqual(@as(u32, 2), hello_count);
}

test "String interning with keywords" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = let; fn test() { fn(); }";

    var config = lexer.LexerConfig.default();
    config.enable_string_interning = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Find the "let" and "fn" tokens
    var let_ptrs = std.ArrayList([*]const u8).init(allocator);
    defer let_ptrs.deinit();
    var fn_ptrs = std.ArrayList([*]const u8).init(allocator);
    defer fn_ptrs.deinit();

    for (tokens) |token| {
        if (std.mem.eql(u8, token.lexeme, "let")) {
            try let_ptrs.append(token.lexeme.ptr);
        } else if (std.mem.eql(u8, token.lexeme, "fn")) {
            try fn_ptrs.append(token.lexeme.ptr);
        }
    }

    // Should have found multiple instances of each keyword
    try testing.expect(let_ptrs.items.len >= 2);
    try testing.expect(fn_ptrs.items.len >= 2);

    // All instances of the same keyword should have the same pointer due to interning
    for (let_ptrs.items[1..]) |ptr| {
        try testing.expect(let_ptrs.items[0] == ptr);
    }
    for (fn_ptrs.items[1..]) |ptr| {
        try testing.expect(fn_ptrs.items[0] == ptr);
    }
}
