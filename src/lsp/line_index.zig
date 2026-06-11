const std = @import("std");

const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const text_edits = @import("text_edits.zig");

const Allocator = std.mem.Allocator;

pub const LineIndex = struct {
    line_starts: []u32,
    line_ascii: []bool,
    source_len: u32,

    pub fn init(allocator: Allocator, source: []const u8) !LineIndex {
        if (source.len > std.math.maxInt(u32)) return error.SourceTooLarge;

        var starts = std.ArrayList(u32){};
        errdefer starts.deinit(allocator);
        var ascii = std.ArrayList(bool){};
        errdefer ascii.deinit(allocator);

        try starts.append(allocator, 0);
        try ascii.append(allocator, true);
        var current_line: usize = 0;
        for (source, 0..) |byte, index| {
            if (byte >= 0x80) ascii.items[current_line] = false;
            if (byte == '\n') {
                try starts.append(allocator, @intCast(index + 1));
                try ascii.append(allocator, true);
                current_line += 1;
            }
        }

        return .{
            .line_starts = try starts.toOwnedSlice(allocator),
            .line_ascii = try ascii.toOwnedSlice(allocator),
            .source_len = @intCast(source.len),
        };
    }

    pub fn deinit(self: *LineIndex, allocator: Allocator) void {
        allocator.free(self.line_starts);
        allocator.free(self.line_ascii);
        self.* = .{
            .line_starts = &.{},
            .line_ascii = &.{},
            .source_len = 0,
        };
    }

    pub fn estimatedByteSize(self: *const LineIndex) usize {
        return self.line_starts.len * @sizeOf(u32) + self.line_ascii.len * @sizeOf(bool);
    }

    pub fn positionToOffset(
        self: *const LineIndex,
        source: []const u8,
        line: u32,
        character: u32,
        encoding: text_edits.PositionEncoding,
    ) ?u32 {
        std.debug.assert(source.len == self.source_len);

        if (line >= self.line_starts.len) return null;

        const line_start = self.line_starts[line];
        const line_end = self.lineEnd(line);

        if (encoding == .utf8) {
            const line_len = line_end - line_start;
            if (character > line_len) return null;
            return line_start + character;
        }

        const line_len = line_end - line_start;
        if (self.line_ascii[line]) {
            if (character > line_len) return null;
            return line_start + character;
        }
        if (character <= line_len) {
            const ascii_end = line_start + character;
            if (isAsciiRange(source, line_start, ascii_end)) {
                return ascii_end;
            }
        }

        var cursor: u32 = line_start;
        var units: u32 = 0;
        while (units < character) {
            if (cursor >= line_end) return null;

            const byte_len = std.unicode.utf8ByteSequenceLength(source[cursor]) catch return null;
            if (cursor + byte_len > line_end) return null;
            const codepoint = std.unicode.utf8Decode(source[cursor .. cursor + byte_len]) catch return null;

            const step_units = unitsForCodepoint(codepoint, encoding);
            const next_units = std.math.add(u32, units, step_units) catch return null;
            if (next_units > character) return null;

            units = next_units;
            cursor += @intCast(byte_len);
        }

        return cursor;
    }

    pub fn offsetToPosition(
        self: *const LineIndex,
        source: []const u8,
        offset: u32,
        encoding: text_edits.PositionEncoding,
    ) frontend.Position {
        std.debug.assert(source.len == self.source_len);

        const clamped_offset = @min(offset, self.source_len);
        const line_no = self.lineForOffset(clamped_offset);
        const line_start = self.line_starts[line_no];
        const character = self.countLineUnits(source, line_no, line_start, clamped_offset, encoding);
        return .{
            .line = @intCast(line_no),
            .character = character,
        };
    }

    pub fn textRangeToRange(
        self: *const LineIndex,
        source: []const u8,
        range: compiler.TextRange,
        encoding: text_edits.PositionEncoding,
    ) frontend.Range {
        return .{
            .start = self.offsetToPosition(source, range.start, encoding),
            .end = self.offsetToPosition(source, range.end, encoding),
        };
    }

    fn lineEnd(self: *const LineIndex, line: u32) u32 {
        const next_line: usize = @as(usize, line) + 1;
        if (next_line < self.line_starts.len) {
            return self.line_starts[next_line] - 1;
        }
        return self.source_len;
    }

    fn lineForOffset(self: *const LineIndex, offset: u32) usize {
        var low: usize = 0;
        var high: usize = self.line_starts.len;

        while (low < high) {
            const mid = low + (high - low) / 2;
            if (self.line_starts[mid] <= offset) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        return if (low == 0) 0 else low - 1;
    }

    fn countLineUnits(
        self: *const LineIndex,
        source: []const u8,
        line_no: usize,
        line_start: u32,
        offset: u32,
        encoding: text_edits.PositionEncoding,
    ) u32 {
        if (encoding == .utf8 or self.line_ascii[line_no]) return offset - line_start;
        if (isAsciiRange(source, line_start, offset)) return offset - line_start;

        var cursor = line_start;
        var units: u32 = 0;
        while (cursor < offset) {
            const byte_len = std.unicode.utf8ByteSequenceLength(source[cursor]) catch {
                units += 1;
                cursor += 1;
                continue;
            };
            if (cursor + byte_len > offset) break;

            const codepoint = std.unicode.utf8Decode(source[cursor .. cursor + byte_len]) catch {
                units += 1;
                cursor += 1;
                continue;
            };
            units += unitsForCodepoint(codepoint, encoding);
            cursor += @intCast(byte_len);
        }

        return units;
    }
};

fn isAsciiRange(source: []const u8, start: u32, end: u32) bool {
    const clamped_start = @min(@as(usize, @intCast(start)), source.len);
    const clamped_end = @min(@as(usize, @intCast(end)), source.len);
    if (clamped_start > clamped_end) return false;
    for (source[clamped_start..clamped_end]) |byte| {
        if (byte >= 0x80) return false;
    }
    return true;
}

fn unitsForCodepoint(codepoint: u21, encoding: text_edits.PositionEncoding) u32 {
    return switch (encoding) {
        .utf8 => unreachable,
        .utf16 => if (codepoint <= 0xFFFF) 1 else 2,
        .utf32 => 1,
    };
}
