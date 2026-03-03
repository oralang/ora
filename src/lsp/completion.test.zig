const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;

fn hasLabel(items: []const completion.Item, label: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item.label, label)) return true;
    }
    return false;
}

fn positionAfterNth(source: []const u8, needle: []const u8, nth: usize) !frontend.Position {
    var from: usize = 0;
    var seen: usize = 0;
    while (true) {
        const rel = std.mem.indexOfPos(u8, source, from, needle) orelse return error.TestExpectedEqual;
        if (seen == nth) return byteIndexToPosition(source, rel + needle.len);
        seen += 1;
        from = rel + needle.len;
    }
}

fn byteIndexToPosition(source: []const u8, byte_index: usize) frontend.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    for (source[0..byte_index]) |byte| {
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    return .{ .line = line, .character = character };
}

test "lsp completion: includes keyword suggestions by prefix" {
    const source =
        \\pub fn run() -> u256 {
        \\    ret
        \\}
    ;

    const position = try positionAfterNth(source, "ret", 0);

    const items = try completion.completionAt(testing.allocator, source, position);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "return"));
}

test "lsp completion: includes symbols from semantic index" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return hel; }
    ;

    const position = try positionAfterNth(source, "hel", 0);

    const items = try completion.completionAt(testing.allocator, source, position);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "helper"));
}

test "lsp completion: non-matching prefix returns empty" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return zzz; }
    ;

    const position = try positionAfterNth(source, "zzz", 0);

    const items = try completion.completionAt(testing.allocator, source, position);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expectEqual(@as(usize, 0), items.len);
}
