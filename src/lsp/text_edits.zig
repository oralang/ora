const std = @import("std");

const Allocator = std.mem.Allocator;

pub const PositionEncoding = enum {
    utf8,
    utf16,
    utf32,
};

pub const Position = struct {
    line: u32,
    character: u32,
};

pub const Range = struct {
    start: Position,
    end: Position,
};

pub const Change = union(enum) {
    full: struct {
        text: []const u8,
    },
    incremental: struct {
        range: Range,
        text: []const u8,
    },
};

pub const ApplyError = error{
    InvalidRange,
} || Allocator.Error;

pub fn applyChangesAlloc(allocator: Allocator, initial_text: []const u8, changes: []const Change) ApplyError![]u8 {
    return applyChangesAllocWithEncoding(allocator, initial_text, changes, .utf16);
}

pub fn applyChangesAllocWithEncoding(
    allocator: Allocator,
    initial_text: []const u8,
    changes: []const Change,
    encoding: PositionEncoding,
) ApplyError![]u8 {
    var current = try allocator.dupe(u8, initial_text);
    errdefer allocator.free(current);

    for (changes) |change| {
        const next = try applySingleChangeAlloc(allocator, current, change, encoding);
        allocator.free(current);
        current = next;
    }

    return current;
}

pub fn applySingleChangeAlloc(
    allocator: Allocator,
    text: []const u8,
    change: Change,
    encoding: PositionEncoding,
) ApplyError![]u8 {
    switch (change) {
        .full => |full| {
            return allocator.dupe(u8, full.text);
        },
        .incremental => |edit| {
            const start_offset = positionToOffset(text, edit.range.start, encoding) orelse return error.InvalidRange;
            const end_offset = positionToOffset(text, edit.range.end, encoding) orelse return error.InvalidRange;
            if (end_offset < start_offset) return error.InvalidRange;

            const prefix = text[0..start_offset];
            const suffix = text[end_offset..];
            const new_len = prefix.len + edit.text.len + suffix.len;
            const next = try allocator.alloc(u8, new_len);

            @memcpy(next[0..prefix.len], prefix);
            @memcpy(next[prefix.len .. prefix.len + edit.text.len], edit.text);
            @memcpy(next[prefix.len + edit.text.len ..], suffix);

            return next;
        },
    }
}

fn positionToOffset(text: []const u8, position: Position, encoding: PositionEncoding) ?usize {
    var line: u32 = 0;
    var index: usize = 0;

    while (line < position.line) : (line += 1) {
        const newline = std.mem.indexOfScalarPos(u8, text, index, '\n') orelse return null;
        index = newline + 1;
    }

    const line_end = std.mem.indexOfScalarPos(u8, text, index, '\n') orelse text.len;

    if (encoding == .utf8) {
        const line_len = line_end - index;
        if (position.character > line_len) return null;
        return index + position.character;
    }

    var cursor = index;
    var units: u32 = 0;
    while (units < position.character) {
        if (cursor >= line_end) return null;

        const byte_len = std.unicode.utf8ByteSequenceLength(text[cursor]) catch return null;
        if (cursor + byte_len > line_end) return null;
        const codepoint = std.unicode.utf8Decode(text[cursor .. cursor + byte_len]) catch return null;

        const step_units: u32 = switch (encoding) {
            .utf16 => if (codepoint <= 0xFFFF) 1 else 2,
            .utf32 => 1,
            .utf8 => unreachable,
        };

        const next_units = std.math.add(u32, units, step_units) catch return null;
        if (next_units > position.character) return null;

        units = next_units;
        cursor += byte_len;
    }

    return cursor;
}
