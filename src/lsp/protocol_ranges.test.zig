const std = @import("std");
const lsp = @import("lsp");
const protocol_ranges = @import("protocol_ranges.zig");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;

test "lsp protocol ranges: converts byte ranges to utf16" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const expected_start = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf16);

    const converted = protocol_ranges.byteRangeToLspOrRaw(source, &lines, .utf16, .{
        .start = lines.offsetToPosition(source, @intCast(amount_offset), .utf8),
        .end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf8),
    });

    try std.testing.expectEqual(expected_start.line, converted.start.line);
    try std.testing.expectEqual(expected_start.character, converted.start.character);
    try std.testing.expectEqual(expected_end.line, converted.end.line);
    try std.testing.expectEqual(expected_end.character, converted.end.character);
}

test "lsp protocol ranges: raw fallback preserves invalid internal range" {
    const source = "pub fn run() -> u256 { return 1; }";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const raw = frontend.Range{
        .start = .{ .line = 20, .character = 3 },
        .end = .{ .line = 20, .character = 8 },
    };
    const converted = protocol_ranges.byteRangeToLspOrRaw(source, &lines, .utf16, raw);

    try std.testing.expectEqual(raw.start.line, converted.start.line);
    try std.testing.expectEqual(raw.start.character, converted.start.character);
    try std.testing.expectEqual(raw.end.line, converted.end.line);
    try std.testing.expectEqual(raw.end.character, converted.end.character);
}

test "lsp protocol ranges: converts lsp position to internal byte position" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const lsp_position = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const byte_position = protocol_ranges.lspPositionToBytePosition(
        source,
        &lines,
        .utf16,
        .{ .line = lsp_position.line, .character = lsp_position.character },
    ) orelse return error.ExpectedPosition;
    const expected = lines.offsetToPosition(source, @intCast(amount_offset), .utf8);

    try std.testing.expectEqual(expected.line, byte_position.line);
    try std.testing.expectEqual(expected.character, byte_position.character);
}

test "lsp protocol ranges: converts lsp ranges to internal byte ranges" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const start_utf16 = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const end_utf16 = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf16);

    const converted = protocol_ranges.lspRangeToByte(source, &lines, .utf16, .{
        .start = .{ .line = start_utf16.line, .character = start_utf16.character },
        .end = .{ .line = end_utf16.line, .character = end_utf16.character },
    }) orelse return error.ExpectedRange;
    const expected_start = lines.offsetToPosition(source, @intCast(amount_offset), .utf8);
    const expected_end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf8);

    try std.testing.expectEqual(expected_start.line, converted.start.line);
    try std.testing.expectEqual(expected_start.character, converted.start.character);
    try std.testing.expectEqual(expected_end.line, converted.end.line);
    try std.testing.expectEqual(expected_end.character, converted.end.character);
}

test "lsp protocol ranges: rejects reversed lsp ranges" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const start_utf16 = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const end_utf16 = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf16);

    try std.testing.expect(protocol_ranges.lspRangeToByte(source, &lines, .utf16, .{
        .start = .{ .line = end_utf16.line, .character = end_utf16.character },
        .end = .{ .line = start_utf16.line, .character = start_utf16.character },
    }) == null);
}

test "lsp protocol ranges: converts compiler text ranges" {
    const source = "/* é */ pub fn helper() -> u256 { return 1; }";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const helper_offset = std.mem.indexOf(u8, source, "helper") orelse return error.ExpectedHelper;
    const converted = protocol_ranges.textRangeToLsp(source, &lines, .utf16, compiler.TextRange{
        .start = @intCast(helper_offset),
        .end = @intCast(helper_offset + "helper".len),
    });
    const expected_start = lines.offsetToPosition(source, @intCast(helper_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(helper_offset + "helper".len), .utf16);

    try std.testing.expectEqual(expected_start.character, converted.start.character);
    try std.testing.expectEqual(expected_end.character, converted.end.character);
}
