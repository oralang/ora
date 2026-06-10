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
    uri: types.DocumentUri,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    index: *const semantic_index.SemanticIndex,
    position: types.Position,
) !?[]types.CallHierarchyItem {
    const byte_position = protocol_ranges.lspPositionToBytePosition(source, line_index, encoding, position) orelse return null;
    const symbol_index = semantic_index.findSymbolAtPosition(index.symbols, byte_position) orelse return null;
    const symbol = index.symbols[symbol_index];
    if (symbol.kind != .function and symbol.kind != .method) return null;

    const range = protocol_ranges.byteRangeToLsp(source, line_index, encoding, symbol.range) orelse return null;
    const selection_range = protocol_ranges.byteRangeToLsp(source, line_index, encoding, symbol.selection_range) orelse return null;

    const result = try arena.alloc(types.CallHierarchyItem, 1);
    result[0] = .{
        .name = try arena.dupe(u8, symbol.name),
        .kind = @enumFromInt(semantic_index.toLspKind(symbol.kind)),
        .uri = try arena.dupe(u8, uri),
        .range = range,
        .selectionRange = selection_range,
        .detail = if (symbol.detail) |detail| try arena.dupe(u8, detail) else null,
    };
    return result;
}
