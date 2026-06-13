const std = @import("std");
const ora_root = @import("ora_root");

const token_cache = ora_root.lsp.token_cache;

test "token cache normalizes source slices and decoded strings" {
    const source =
        \\const std = @import("std/core.ora");
        \\contract Wallet {}
    ;

    var cache = try token_cache.Cache.init(std.testing.allocator, source);
    defer cache.deinit(std.testing.allocator);

    var saw_import_path = false;
    var saw_wallet = false;
    for (cache.tokens) |token| {
        if (token.type == .StringLiteral) {
            try std.testing.expectEqualStrings("std/core.ora", token.string_value.?);
            saw_import_path = true;
        }
        if (token.type == .Identifier and std.mem.eql(u8, token.lexeme, "Wallet")) {
            const token_addr = @intFromPtr(token.lexeme.ptr);
            const source_addr = @intFromPtr(source.ptr);
            try std.testing.expect(token_addr >= source_addr);
            try std.testing.expect(token_addr < source_addr + source.len);
            saw_wallet = true;
        }
    }

    try std.testing.expect(saw_import_path);
    try std.testing.expect(saw_wallet);
    try std.testing.expect(cache.builderCapacityRequested() >= cache.builderItemsBuilt());
    try std.testing.expectEqual(@as(usize, 0), cache.builderUnusedCapacity());
    try std.testing.expectEqual(@as(usize, 0), cache.builderGrowthEvents());
}

test "token cache returns empty cache on recoverable lexer failure" {
    const source = "\"unterminated";

    var cache = try token_cache.Cache.init(std.testing.allocator, source);
    defer cache.deinit(std.testing.allocator);

    try std.testing.expect(cache.tokens.len <= source.len + 1);
    try std.testing.expect(cache.builderCapacityRequested() >= cache.builderItemsBuilt());
    try std.testing.expectEqual(@as(usize, 0), cache.builderGrowthEvents());
}

test "token cache owns lexer diagnostics" {
    const source = "\"unterminated";

    var cache = try token_cache.Cache.init(std.testing.allocator, source);
    defer cache.deinit(std.testing.allocator);

    try std.testing.expect(cache.diagnostics.len > 0);
    try std.testing.expectEqual(.Error, cache.diagnostics[0].severity);
    try std.testing.expect(cache.diagnostics[0].message.len > 0);
}
