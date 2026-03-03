const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const references = ora_root.lsp.references;
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

test "lsp references: includes declaration and call sites" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn main() -> u256 {
        \\    let a: u256 = helper(1);
        \\    return helper(a);
        \\}
    ;

    const query = try positionOfNth(source, "helper", 2);

    const ranges = try references.referencesAt(testing.allocator, source, query, true);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 3), ranges.len);
}

test "lsp references: excludes declaration when requested" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn main() -> u256 {
        \\    let a: u256 = helper(1);
        \\    return helper(a);
        \\}
    ;

    const query = try positionOfNth(source, "helper", 1);

    const ranges = try references.referencesAt(testing.allocator, source, query, false);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 2), ranges.len);
}

test "lsp references: respects local scope shadowing" {
    const source =
        \\pub fn run() -> u256 {
        \\    let amount: u256 = 1;
        \\    if (true) {
        \\        let amount: u256 = 2;
        \\        return amount;
        \\    }
        \\    return amount;
        \\}
    ;

    const query = try positionOfNth(source, "amount", 3);

    const ranges = try references.referencesAt(testing.allocator, source, query, true);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 2), ranges.len);
}

test "lsp references: unknown symbol returns empty" {
    const source = "pub fn run() -> u256 { return missing; }";
    const query = try positionOfNth(source, "missing", 0);

    const ranges = try references.referencesAt(testing.allocator, source, query, true);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 0), ranges.len);
}

test "lsp references: parse failure returns empty" {
    const source = "@import(\"std\");";
    const query = try positionOfNth(source, "import", 0);

    const ranges = try references.referencesAt(testing.allocator, source, query, true);
    defer testing.allocator.free(ranges);

    try testing.expectEqual(@as(usize, 0), ranges.len);
}
