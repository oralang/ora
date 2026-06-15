const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const uri_ranges = @import("uri_ranges.zig");

const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;

const types = lsp.types;

test "lsp uri ranges: converts indexed byte ranges and preserves raw fallback" {
    const source = "/* é */ pub fn read() -> u256 { return 1; }";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);
    const indexed = uri_ranges.IndexedSource{ .source = source, .line_index = &lines };
    const byte_range = rangeFor(source, &lines, "read");

    const converted = uri_ranges.byteRangeToLsp(indexed, .utf16, byte_range);
    const expected_offset = std.mem.indexOf(u8, source, "read") orelse return error.ExpectedRead;
    const expected = lines.offsetToPosition(source, @intCast(expected_offset), .utf16);

    try std.testing.expectEqual(expected.line, converted.start.line);
    try std.testing.expectEqual(expected.character, converted.start.character);

    const raw = frontend.Range{
        .start = .{ .line = 20, .character = 3 },
        .end = .{ .line = 20, .character = 8 },
    };
    const raw_converted = uri_ranges.byteRangeToLsp(null, .utf16, raw);
    try std.testing.expectEqual(raw.start.line, raw_converted.start.line);
    try std.testing.expectEqual(raw.start.character, raw_converted.start.character);
    try std.testing.expectEqual(raw.end.line, raw_converted.end.line);
    try std.testing.expectEqual(raw.end.character, raw_converted.end.character);
}

test "lsp uri ranges: materializes locations and range slices from indexed source" {
    const source = "let marker = \"é\"; let amount = amount + 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);
    const indexed = uri_ranges.IndexedSource{ .source = source, .line_index = &lines };
    const first = rangeFor(source, &lines, "amount");
    const second_start = (std.mem.indexOf(u8, source, "amount +") orelse return error.ExpectedSecondAmount);
    const second = frontend.Range{
        .start = lines.offsetToPosition(source, @intCast(second_start), .utf8),
        .end = lines.offsetToPosition(source, @intCast(second_start + "amount".len), .utf8),
    };
    const ranges = [_]frontend.Range{ first, second };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var locations = std.ArrayList(types.Location){};
    try locations.ensureTotalCapacity(arena, ranges.len);
    uri_ranges.appendLocationsAssumeCapacity(&locations, "file:///main.ora", &ranges, indexed, .utf16);

    const expected_first_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedFirstAmount;
    const expected_first = lines.offsetToPosition(source, @intCast(expected_first_offset), .utf16);
    try std.testing.expectEqual(@as(usize, 2), locations.items.len);
    try std.testing.expectEqualStrings("file:///main.ora", locations.items[0].uri);
    try std.testing.expectEqual(expected_first.character, locations.items[0].range.start.character);

    const converted = try uri_ranges.rangesToLsp(arena, indexed, .utf16, &ranges);
    try std.testing.expectEqual(@as(usize, 2), converted.len);
    try std.testing.expectEqual(locations.items[1].range.start.character, converted[1].start.character);
}

test "lsp uri ranges: converts lsp ranges to byte ranges and fills rename edits" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);
    const indexed = uri_ranges.IndexedSource{ .source = source, .line_index = &lines };
    const byte_range = rangeFor(source, &lines, "amount");
    const lsp_range = uri_ranges.byteRangeToLsp(indexed, .utf16, byte_range);

    const round_trip = uri_ranges.lspRangeToByte(indexed, .utf16, lsp_range) orelse return error.ExpectedRoundTrip;
    try std.testing.expectEqual(byte_range.start.character, round_trip.start.character);
    try std.testing.expectEqual(byte_range.end.character, round_trip.end.character);

    var edits = [_]types.TextEdit{undefined};
    uri_ranges.fillRenameTextEdits(&edits, indexed, &.{byte_range}, "balance", .utf16);
    try std.testing.expectEqual(lsp_range.start.character, edits[0].range.start.character);
    try std.testing.expectEqualStrings("balance", edits[0].newText);

    const raw_lsp = types.Range{
        .start = .{ .line = 4, .character = 5 },
        .end = .{ .line = 4, .character = 9 },
    };
    const raw_byte = uri_ranges.lspRangeToByte(null, .utf16, raw_lsp) orelse return error.ExpectedRawRange;
    try std.testing.expectEqual(raw_lsp.start.line, raw_byte.start.line);
    try std.testing.expectEqual(raw_lsp.end.character, raw_byte.end.character);
}

fn rangeFor(source: []const u8, lines: *const line_index.LineIndex, needle: []const u8) frontend.Range {
    const start = std.mem.indexOf(u8, source, needle).?;
    return .{
        .start = lines.offsetToPosition(source, @intCast(start), .utf8),
        .end = lines.offsetToPosition(source, @intCast(start + needle.len), .utf8),
    };
}
