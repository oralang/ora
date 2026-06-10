const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");

const inlay_hints = ora_root.lsp.inlay_hints;
const line_index = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    hints: []const inlay_hints.InlayHint,
) !?[]types.InlayHint {
    if (hints.len == 0) return null;

    const result = try arena.alloc(types.InlayHint, hints.len);
    for (hints, 0..) |hint, index| {
        const position = protocol_ranges.bytePositionToLsp(source, lines, encoding, hint.position) orelse return null;
        result[index] = .{
            .position = position,
            .label = .{ .string = hint.label },
            .kind = switch (hint.kind) {
                .type_hint => .Type,
                .parameter_hint => .Parameter,
            },
            .paddingLeft = hint.padding_left,
            .paddingRight = hint.padding_right,
        };
    }

    return result;
}
