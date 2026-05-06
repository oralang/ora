//! Parser for debugger `:break` command arguments.
//!
//! Accepts `<line> [when <expr>] [hit <n>]` in either suffix order.
//! The parser is purely functional and side-effect-free, so it can be
//! unit-tested without touching the rest of the TUI.

const std = @import("std");

pub const ParsedBreakpoint = struct {
    line: u32,
    condition: ?[]const u8 = null,
    hit_target: ?u32 = null,
};

/// Parse `<line> [when <expr>] [hit <n>]`. Either order of suffixes
/// is accepted. Returned slices are borrowed from `rest`; the caller
/// dupes if they need to outlive it.
pub fn parse(rest: []const u8) !ParsedBreakpoint {
    const trimmed = std.mem.trim(u8, rest, " \t");
    if (trimmed.len == 0) return error.InvalidArguments;

    var head_end: usize = 0;
    while (head_end < trimmed.len and trimmed[head_end] != ' ' and trimmed[head_end] != '\t') head_end += 1;
    const line_text = trimmed[0..head_end];
    const line_value: u32 = if (std.mem.indexOfScalar(u8, line_text, ':')) |colon|
        try std.fmt.parseUnsigned(u32, std.mem.trim(u8, line_text[colon + 1 ..], " \t"), 10)
    else
        try std.fmt.parseUnsigned(u32, line_text, 10);

    var result = ParsedBreakpoint{ .line = line_value };
    var cursor = std.mem.trim(u8, trimmed[head_end..], " \t");
    while (cursor.len > 0) {
        if (std.mem.startsWith(u8, cursor, "when ") or std.mem.startsWith(u8, cursor, "when\t")) {
            const after = std.mem.trim(u8, cursor[5..], " \t");
            if (std.mem.indexOf(u8, after, " hit ")) |split_at| {
                result.condition = std.mem.trim(u8, after[0..split_at], " \t");
                cursor = std.mem.trim(u8, after[split_at + 1 ..], " \t");
            } else {
                result.condition = after;
                cursor = "";
            }
        } else if (std.mem.startsWith(u8, cursor, "hit ") or std.mem.startsWith(u8, cursor, "hit\t")) {
            const after = std.mem.trim(u8, cursor[4..], " \t");
            if (std.mem.indexOf(u8, after, " when ")) |split_at| {
                result.hit_target = try std.fmt.parseUnsigned(u32, std.mem.trim(u8, after[0..split_at], " \t"), 10);
                cursor = std.mem.trim(u8, after[split_at + 1 ..], " \t");
            } else {
                result.hit_target = try std.fmt.parseUnsigned(u32, after, 10);
                cursor = "";
            }
        } else {
            return error.InvalidArguments;
        }
    }
    return result;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "breakpoint args: bare line" {
    const r = try parse("42");
    try testing.expectEqual(@as(u32, 42), r.line);
    try testing.expectEqual(@as(?[]const u8, null), r.condition);
    try testing.expectEqual(@as(?u32, null), r.hit_target);
}

test "breakpoint args: file:line accepted" {
    const r = try parse("foo.ora:17");
    try testing.expectEqual(@as(u32, 17), r.line);
}

test "breakpoint args: when predicate" {
    const r = try parse("5 when n == 0");
    try testing.expectEqual(@as(u32, 5), r.line);
    try testing.expectEqualStrings("n == 0", r.condition.?);
    try testing.expectEqual(@as(?u32, null), r.hit_target);
}

test "breakpoint args: hit count" {
    const r = try parse("11 hit 4");
    try testing.expectEqual(@as(u32, 11), r.line);
    try testing.expectEqual(@as(?u32, 4), r.hit_target);
    try testing.expectEqual(@as(?[]const u8, null), r.condition);
}

test "breakpoint args: when then hit" {
    const r = try parse("7 when amount > 0 hit 2");
    try testing.expectEqual(@as(u32, 7), r.line);
    try testing.expectEqualStrings("amount > 0", r.condition.?);
    try testing.expectEqual(@as(?u32, 2), r.hit_target);
}

test "breakpoint args: hit then when" {
    const r = try parse("7 hit 2 when amount > 0");
    try testing.expectEqual(@as(u32, 7), r.line);
    try testing.expectEqualStrings("amount > 0", r.condition.?);
    try testing.expectEqual(@as(?u32, 2), r.hit_target);
}

test "breakpoint args: empty errors" {
    try testing.expectError(error.InvalidArguments, parse(""));
}

test "breakpoint args: trailing garbage errors" {
    try testing.expectError(error.InvalidArguments, parse("5 garbage"));
}

test "breakpoint args: non-numeric line errors" {
    try testing.expectError(error.InvalidCharacter, parse("five"));
}
