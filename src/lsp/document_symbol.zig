const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const line_index_api = ora_root.lsp.line_index;
const protocol_ranges = @import("protocol_ranges.zig");
const semantic_index = ora_root.lsp.semantic_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const FlatBuildResult = struct {
    items: []types.SymbolInformation,
    string_bytes: usize,
};

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
    const children: ?[]types.DocumentSymbol = if (child_indices.len > 0) blk: {
        const result = try arena.alloc(types.DocumentSymbol, child_indices.len);
        for (child_indices, 0..) |child_index, i| {
            result[i] = try buildRecursive(arena, source, line_index, encoding, index, child_lists, child_index);
        }
        break :blk result;
    } else null;

    return .{
        .name = symbol.name,
        .detail = symbol.detail,
        .kind = @enumFromInt(semantic_index.toLspKind(symbol.kind)),
        .range = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, symbol.range),
        .selectionRange = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, symbol.selection_range),
        .children = children,
    };
}

pub fn buildFlat(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    uri: []const u8,
    index: *const semantic_index.SemanticIndex,
) !FlatBuildResult {
    const symbols = index.symbols;
    if (symbols.len == 0) {
        return .{
            .items = try arena.alloc(types.SymbolInformation, 0),
            .string_bytes = 0,
        };
    }

    const uri_copy = try arena.dupe(u8, uri);
    const items = try arena.alloc(types.SymbolInformation, symbols.len);
    var item_count: usize = 0;
    var string_bytes: usize = 0;

    if (index.symbol_position_order.len == symbols.len) {
        for (index.symbol_position_order) |raw_symbol_index| {
            const symbol_index: usize = raw_symbol_index;
            if (symbol_index >= symbols.len) continue;
            appendFlatSymbol(
                &items[item_count],
                source,
                line_index,
                encoding,
                uri_copy,
                symbols,
                symbol_index,
                &string_bytes,
            );
            item_count += 1;
        }
    } else {
        for (symbols, 0..) |_, symbol_index| {
            appendFlatSymbol(
                &items[item_count],
                source,
                line_index,
                encoding,
                uri_copy,
                symbols,
                symbol_index,
                &string_bytes,
            );
            item_count += 1;
        }
    }

    return .{
        .items = items[0..item_count],
        .string_bytes = string_bytes,
    };
}

fn appendFlatSymbol(
    item: *types.SymbolInformation,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    uri: []const u8,
    symbols: []const semantic_index.Symbol,
    symbol_index: usize,
    string_bytes: *usize,
) void {
    const symbol = symbols[symbol_index];
    const container_name = if (symbol.parent) |parent_index|
        if (parent_index < symbols.len) symbols[parent_index].name else null
    else
        null;

    string_bytes.* = addSat(string_bytes.*, symbol.name.len);
    string_bytes.* = addSat(string_bytes.*, uri.len);
    if (container_name) |value| string_bytes.* = addSat(string_bytes.*, value.len);

    item.* = .{
        .name = symbol.name,
        .kind = @enumFromInt(semantic_index.toLspKind(symbol.kind)),
        .location = .{
            .uri = uri,
            .range = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, symbol.selection_range),
        },
        .containerName = container_name,
    };
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
