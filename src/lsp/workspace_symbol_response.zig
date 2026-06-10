const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");

const line_index_api = ora_root.lsp.line_index;
const semantic_index = ora_root.lsp.semantic_index;
const text_edits = ora_root.lsp.text_edits;
const workspace_index = ora_root.lsp.workspace_index;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const AppendStats = struct {
    count: usize = 0,
    string_bytes: usize = 0,
};

pub fn appendOpenDocumentSymbols(
    arena: Allocator,
    symbols: *std.ArrayList(types.SymbolInformation),
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    uri: []const u8,
    index: *const semantic_index.SemanticIndex,
    query: []const u8,
) !AppendStats {
    var stats: AppendStats = .{};
    for (index.symbols) |sym| {
        if (sym.parent != null) continue;
        if (query.len > 0 and !fuzzyMatch(sym.name, query)) continue;

        stats.count += 1;
        stats.string_bytes = addSat(stats.string_bytes, sym.name.len);
        stats.string_bytes = addSat(stats.string_bytes, uri.len);
        if (sym.detail) |detail| stats.string_bytes = addSat(stats.string_bytes, detail.len);

        try symbols.append(arena, .{
            .name = try arena.dupe(u8, sym.name),
            .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
            .location = .{
                .uri = try arena.dupe(u8, uri),
                .range = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, sym.selection_range),
            },
            .containerName = if (sym.detail) |d| try arena.dupe(u8, d) else null,
        });
    }
    return stats;
}

pub fn appendEntrySymbols(
    arena: Allocator,
    symbols: *std.ArrayList(types.SymbolInformation),
    uri: []const u8,
    entry: *const workspace_index.FileEntry,
    query: []const u8,
    range_converter: anytype,
) !AppendStats {
    var stats: AppendStats = .{};
    for (entry.rootSymbolIndexes()) |symbol_index| {
        const sym = entry.symbols[symbol_index];
        if (query.len > 0 and !fuzzyMatch(sym.name, query)) continue;

        stats.count += 1;
        stats.string_bytes = addSat(stats.string_bytes, sym.name.len);
        stats.string_bytes = addSat(stats.string_bytes, uri.len);
        if (sym.detail) |detail| stats.string_bytes = addSat(stats.string_bytes, detail.len);

        try symbols.append(arena, .{
            .name = try arena.dupe(u8, sym.name),
            .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
            .location = .{
                .uri = try arena.dupe(u8, uri),
                .range = try range_converter.byteRangeToLsp(uri, sym.selection_range),
            },
            .containerName = if (sym.detail) |d| try arena.dupe(u8, d) else null,
        });
    }
    return stats;
}

pub fn matchingEntrySymbolCount(entry: *const workspace_index.FileEntry, query: []const u8) usize {
    var count: usize = 0;
    for (entry.rootSymbolIndexes()) |symbol_index| {
        const sym = entry.symbols[symbol_index];
        if (query.len > 0 and !fuzzyMatch(sym.name, query)) continue;
        count += 1;
    }
    return count;
}

fn fuzzyMatch(name: []const u8, query: []const u8) bool {
    var query_index: usize = 0;
    for (name) |char| {
        if (query_index >= query.len) break;
        if (std.ascii.toLower(char) == std.ascii.toLower(query[query_index])) query_index += 1;
    }
    return query_index == query.len;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
