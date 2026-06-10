const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");

const code_lens = ora_root.lsp.code_lens;
const line_index = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const BuildResult = struct {
    items: ?[]types.CodeLens,
    string_bytes: usize,
};

pub fn build(
    arena: Allocator,
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    lenses: []const code_lens.VerificationLens,
) !?[]types.CodeLens {
    return (try buildWithStats(arena, source, lines, encoding, lenses)).items;
}

pub fn buildWithStats(
    arena: Allocator,
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    lenses: []const code_lens.VerificationLens,
) !BuildResult {
    if (lenses.len == 0) return .{ .items = null, .string_bytes = 0 };

    const result = try arena.alloc(types.CodeLens, lenses.len);
    var string_bytes: usize = 0;
    for (lenses, 0..) |lens, index| {
        string_bytes = addSat(string_bytes, lens.title.len);
        string_bytes = addSat(string_bytes, "ora.verify".len);
        result[index] = .{
            .range = protocol_ranges.byteRangeToLspOrRaw(source, lines, encoding, lens.range),
            .command = .{
                .title = try arena.dupe(u8, lens.title),
                .command = "ora.verify",
            },
        };
    }

    return .{ .items = result, .string_bytes = string_bytes };
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
