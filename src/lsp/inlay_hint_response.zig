const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const inlay_hints = ora_root.lsp.inlay_hints;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const BuildResult = struct {
    items: ?[]types.InlayHint,
    string_bytes: usize,
};

pub fn build(
    arena: Allocator,
    hints: []const inlay_hints.InlayHint,
) !?[]types.InlayHint {
    return (try buildWithStats(arena, hints)).items;
}

pub fn buildWithStats(
    arena: Allocator,
    hints: []const inlay_hints.InlayHint,
) !BuildResult {
    if (hints.len == 0) return .{ .items = null, .string_bytes = 0 };

    const result = try arena.alloc(types.InlayHint, hints.len);
    var string_bytes: usize = 0;
    for (hints, 0..) |hint, index| {
        string_bytes = addSat(string_bytes, hint.label.len);
        result[index] = .{
            .position = .{ .line = hint.position.line, .character = hint.position.character },
            .label = .{ .string = try arena.dupe(u8, hint.label) },
            .kind = switch (hint.kind) {
                .type_hint => .Type,
                .parameter_hint => .Parameter,
            },
            .paddingLeft = hint.padding_left,
            .paddingRight = hint.padding_right,
        };
    }

    return .{ .items = result, .string_bytes = string_bytes };
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
