const std = @import("std");
const document_symbol = @import("document_symbol.zig");
const ora_root = @import("ora_root");

const line_index = ora_root.lsp.line_index;
const semantic_index = ora_root.lsp.semantic_index;

test "lsp document symbol: builds nested LSP document symbols" {
    const source =
        \\contract Wallet {
        \\    storage var balance: u256;
        \\    pub fn deposit(amount: u256) -> u256 {
        \\        return amount;
        \\    }
        \\}
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
    ;

    var index = try semantic_index.indexDocument(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const symbols = try document_symbol.build(response_arena.allocator(), source, &lines, .utf8, &index);

    try std.testing.expectEqual(@as(usize, 2), symbols.len);
    try std.testing.expectEqualStrings("Wallet", symbols[0].name);
    try std.testing.expect(symbols[0].children != null);
    try std.testing.expectEqual(@as(usize, 2), symbols[0].children.?.len);
    try std.testing.expectEqualStrings("balance", symbols[0].children.?[0].name);
    try std.testing.expectEqualStrings("deposit", symbols[0].children.?[1].name);
    try std.testing.expectEqualStrings("helper", symbols[1].name);
}

test "lsp document symbol: converts byte ranges to utf16" {
    const source = "/* é */ pub fn helper() -> u256 { return 1; }";

    var index = try semantic_index.indexDocument(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const symbols = try document_symbol.build(response_arena.allocator(), source, &lines, .utf16, &index);

    const helper_offset = std.mem.indexOf(u8, source, "helper") orelse return error.ExpectedHelper;
    const expected_start = lines.offsetToPosition(source, @intCast(helper_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(helper_offset + "helper".len), .utf16);

    try std.testing.expectEqual(@as(usize, 1), symbols.len);
    try std.testing.expectEqualStrings("helper", symbols[0].name);
    try std.testing.expectEqual(expected_start.character, symbols[0].selectionRange.start.character);
    try std.testing.expectEqual(expected_end.character, symbols[0].selectionRange.end.character);
}
