const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const line_index_api = ora_root.lsp.line_index;
const protocol_ranges = @import("protocol_ranges.zig");
const semantic_index = ora_root.lsp.semantic_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    index: *const semantic_index.SemanticIndex,
) ![]types.DocumentSymbol {
    const symbols = index.symbols;
    if (symbols.len == 0) return try arena.alloc(types.DocumentSymbol, 0);

    const child_lists = try arena.alloc(std.ArrayList(usize), symbols.len);
    defer {
        for (child_lists) |*list| list.deinit(arena);
        arena.free(child_lists);
    }
    for (child_lists) |*list| list.* = .{};

    var roots = std.ArrayList(usize){};
    defer roots.deinit(arena);

    for (symbols, 0..) |symbol, symbol_index| {
        if (symbol.parent) |parent_index| {
            if (parent_index < symbols.len) {
                try child_lists[parent_index].append(arena, symbol_index);
            } else {
                try roots.append(arena, symbol_index);
            }
        } else {
            try roots.append(arena, symbol_index);
        }
    }

    const result = try arena.alloc(types.DocumentSymbol, roots.items.len);
    for (roots.items, 0..) |symbol_index, i| {
        result[i] = try buildRecursive(arena, source, line_index, encoding, index, child_lists, symbol_index);
    }
    return result;
}

fn buildRecursive(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    index: *const semantic_index.SemanticIndex,
    child_lists: []const std.ArrayList(usize),
    symbol_index: usize,
) !types.DocumentSymbol {
    const symbols = index.symbols;
    const symbol = symbols[symbol_index];
    const child_indices = child_lists[symbol_index].items;
    const children = if (child_indices.len > 0) blk: {
        const result = try arena.alloc(types.DocumentSymbol, child_indices.len);
        for (child_indices, 0..) |child_index, i| {
            result[i] = try buildRecursive(arena, source, line_index, encoding, index, child_lists, child_index);
        }
        break :blk result;
    } else &[_]types.DocumentSymbol{};

    return .{
        .name = try arena.dupe(u8, symbol.name),
        .detail = if (symbol.detail) |detail| try arena.dupe(u8, detail) else null,
        .kind = @enumFromInt(semantic_index.toLspKind(symbol.kind)),
        .range = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, symbol.range),
        .selectionRange = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, symbol.selection_range),
        .children = children,
    };
}
