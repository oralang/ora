const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const rename = ora_root.lsp.rename;
const frontend = ora_root.lsp.frontend;

fn positionOfNth(source: []const u8, needle: []const u8, nth: usize) !frontend.Position {
    var from: usize = 0;
    var seen: usize = 0;
    while (true) {
        const rel = std.mem.indexOfPos(u8, source, from, needle) orelse return error.TestExpectedEqual;
        if (seen == nth) return byteIndexToPosition(source, rel);
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

test "lsp rename: includes declaration and references in file" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return helper(1) + helper(2); }
    ;

    const query = try positionOfNth(source, "helper", 1);

    const ranges = try rename.renameRangesAt(testing.allocator, source, query);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 3), ranges.len);
}

test "lsp rename: unknown symbol returns empty" {
    const source = "pub fn run() -> u256 { return missing; }";
    const query = try positionOfNth(source, "missing", 0);

    const ranges = try rename.renameRangesAt(testing.allocator, source, query);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 0), ranges.len);
}

test "lsp rename: identifier validation" {
    try testing.expect(rename.isValidIdentifier("new_name"));
    try testing.expect(rename.isValidIdentifier("newName2"));

    try testing.expect(!rename.isValidIdentifier(""));
    try testing.expect(!rename.isValidIdentifier("2name"));
    try testing.expect(!rename.isValidIdentifier("new-name"));
}
