const std = @import("std");
const lexer = @import("src/lexer.zig");

test "comprehensive lexer functionality with enhanced tokens" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const source = 
        \\contract MyContract {
        \\    let x = 42;
        \\    let name = "Alice";
        \\    let flag = true;
        \\    let addr = 0x1234567890123456789012345678901234567890;
        \\}
    ;
    
    const tokens = try lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    // Verify we have tokens
    try std.testing.expect(tokens.len > 0);
    
    // Check that all tokens have proper source ranges
    for (tokens) |token| {
        // All tokens should have valid source ranges
        try std.testing.expect(token.range.start_line >= 1);
        try std.testing.expect(token.range.start_column >= 1);
        try std.testing.expect(token.range.end_line >= token.range.start_line);
        try std.testing.expect(token.range.start_offset <= token.range.end_offset);
        
        // Legacy fields should be populated for backward compatibility
        try std.testing.expect(token.line >= 1);
        try std.testing.expect(token.column >= 1);
        
        // Check specific token values
        switch (token.type) {
            .StringLiteral => {
                try std.testing.expect(token.value != null);
                if (token.value) |val| {
                    switch (val) {
                        .string => |s| {
                            try std.testing.expectEqualStrings(token.lexeme, s);
                        },
                        else => try std.testing.expect(false),
                    }
                }
            },
            .True => {
                try std.testing.expect(token.value != null);
                if (token.value) |val| {
                    switch (val) {
                        .boolean => |b| try std.testing.expect(b == true),
                        else => try std.testing.expect(false),
                    }
                }
            },
            .False => {
                try std.testing.expect(token.value != null);
                if (token.value) |val| {
                    switch (val) {
                        .boolean => |b| try std.testing.expect(b == false),
                        else => try std.testing.expect(false),
                    }
                }
            },
            else => {
                // Other tokens may or may not have values
            }
        }
    }
    
    // Verify specific tokens exist
    var found_contract = false;
    var found_string = false;
    var found_boolean = false;
    var found_address = false;
    
    for (tokens) |token| {
        switch (token.type) {
            .Contract => found_contract = true,
            .StringLiteral => found_string = true,
            .True => found_boolean = true,
            .AddressLiteral => found_address = true,
            else => {},
        }
    }
    
    try std.testing.expect(found_contract);
    try std.testing.expect(found_string);
    try std.testing.expect(found_boolean);
    try std.testing.expect(found_address);
}

test "source range accuracy across multiple lines" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const source = 
        \\let x = 1;
        \\let y = 2;
        \\let z = 3;
    ;
    
    const tokens = try lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    // Find tokens on different lines
    var line1_tokens: u32 = 0;
    var line2_tokens: u32 = 0;
    var line3_tokens: u32 = 0;
    
    for (tokens) |token| {
        if (token.range.start_line == 1) line1_tokens += 1;
        if (token.range.start_line == 2) line2_tokens += 1;
        if (token.range.start_line == 3) line3_tokens += 1;
    }
    
    // Each line should have tokens (let, identifier, =, number, ;)
    try std.testing.expect(line1_tokens > 0);
    try std.testing.expect(line2_tokens > 0);
    try std.testing.expect(line3_tokens > 0);
}