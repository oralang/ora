const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const folding = ora_root.lsp.folding;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(arena: Allocator, ranges: []const folding.FoldingRange) !?[]types.FoldingRange {
    if (ranges.len == 0) return null;

    const result = try arena.alloc(types.FoldingRange, ranges.len);
    for (ranges, 0..) |range, index| {
        result[index] = .{
            .startLine = range.start_line,
            .endLine = range.end_line,
            .kind = switch (range.kind) {
                .region => .region,
                .comment => .comment,
                .imports => .imports,
            },
        };
    }

    return result;
}
