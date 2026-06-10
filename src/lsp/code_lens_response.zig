const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");

const code_lens = ora_root.lsp.code_lens;
const line_index = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    lenses: []const code_lens.VerificationLens,
) !?[]types.CodeLens {
    if (lenses.len == 0) return null;

    const result = try arena.alloc(types.CodeLens, lenses.len);
    for (lenses, 0..) |lens, index| {
        result[index] = .{
            .range = protocol_ranges.byteRangeToLspOrRaw(source, lines, encoding, lens.range),
            .command = .{
                .title = try arena.dupe(u8, lens.title),
                .command = "ora.verify",
            },
        };
    }

    return result;
}
