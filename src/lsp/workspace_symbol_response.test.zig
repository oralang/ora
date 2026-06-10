const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");
const workspace_symbol_response = @import("workspace_symbol_response.zig");

const call_hierarchy = ora_root.lsp.call_hierarchy;
const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;
const references = ora_root.lsp.references;
const semantic_index = ora_root.lsp.semantic_index;
const workspace_index = ora_root.lsp.workspace_index;

const types = lsp.types;

const PassthroughRanges = struct {
    pub fn byteRangeToLsp(_: *const @This(), _: []const u8, source_range: frontend.Range) !types.Range {
        return protocol_ranges.rawRange(source_range);
    }
};

test "lsp workspace symbol response: appends cached open-document root symbols" {
    const source = "/* é */ contract Wallet {}";
    const wallet_offset = std.mem.indexOf(u8, source, "Wallet") orelse return error.ExpectedWallet;
    var symbols_data = [_]semantic_index.Symbol{
        openSymbol("Wallet", .contract, null, null, mkRange(0, @intCast(wallet_offset), 0, @intCast(wallet_offset + "Wallet".len))),
        openSymbol("deposit", .function, null, 0, mkRange(0, @intCast(wallet_offset + 8), 0, @intCast(wallet_offset + 15))),
    };
    const index: semantic_index.SemanticIndex = .{
        .symbols = symbols_data[0..],
        .parse_succeeded = true,
    };

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var results = std.ArrayList(types.SymbolInformation){};
    const stats = try workspace_symbol_response.appendOpenDocumentSymbols(
        arena,
        &results,
        source,
        &lines,
        .utf16,
        "file:///open.ora",
        &index,
        "Wa",
    );

    const expected_start = lines.offsetToPosition(source, @intCast(wallet_offset), .utf16);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expectEqual(@as(usize, 1), stats.count);
    try std.testing.expect(stats.string_bytes > 0);
    try std.testing.expectEqualStrings("Wallet", results.items[0].name);
    try std.testing.expectEqualStrings("file:///open.ora", results.items[0].location.uri);
    try std.testing.expectEqual(expected_start.character, results.items[0].location.range.start.character);
}

test "lsp workspace symbol response: appends matching root symbols" {
    var symbols_data = [_]workspace_index.Symbol{
        symbol("ColdRoot", .contract, null, null, mkRange(1, 0, 1, 8)),
        symbol("NestedMethod", .function, "ColdRoot", 0, mkRange(2, 8, 2, 20)),
        symbol("Wallet", .contract, null, null, mkRange(5, 0, 5, 6)),
    };
    const entry = testEntry(symbols_data[0..]);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ranges = PassthroughRanges{};

    var results = std.ArrayList(types.SymbolInformation){};
    const stats = try workspace_symbol_response.appendEntrySymbols(arena, &results, "file:///cold.ora", &entry, "CR", &ranges);

    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expectEqual(@as(usize, 1), stats.count);
    try std.testing.expect(stats.string_bytes > 0);
    try std.testing.expectEqualStrings("ColdRoot", results.items[0].name);
    try std.testing.expectEqual(types.SymbolKind.Class, results.items[0].kind);
    try std.testing.expectEqualStrings("file:///cold.ora", results.items[0].location.uri);
    try std.testing.expectEqual(@as(u32, 1), results.items[0].location.range.start.line);
}

test "lsp workspace symbol response: appends all root symbols for empty query" {
    var symbols_data = [_]workspace_index.Symbol{
        symbol("ColdRoot", .contract, null, null, mkRange(1, 0, 1, 8)),
        symbol("NestedMethod", .function, "ColdRoot", 0, mkRange(2, 8, 2, 20)),
        symbol("Wallet", .contract, null, null, mkRange(5, 0, 5, 6)),
    };
    const entry = testEntry(symbols_data[0..]);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ranges = PassthroughRanges{};

    var results = std.ArrayList(types.SymbolInformation){};
    const stats = try workspace_symbol_response.appendEntrySymbols(arena, &results, "file:///cold.ora", &entry, "", &ranges);

    try std.testing.expectEqual(@as(usize, 2), results.items.len);
    try std.testing.expectEqual(@as(usize, 2), stats.count);
    try std.testing.expect(stats.string_bytes > 0);
    try std.testing.expectEqualStrings("ColdRoot", results.items[0].name);
    try std.testing.expectEqualStrings("Wallet", results.items[1].name);
}

fn openSymbol(
    name: []const u8,
    kind: semantic_index.SymbolKind,
    detail: ?[]const u8,
    parent: ?usize,
    selection_range: frontend.Range,
) semantic_index.Symbol {
    return .{
        .name = name,
        .detail = detail,
        .kind = kind,
        .range = selection_range,
        .selection_range = selection_range,
        .parent = parent,
    };
}

fn symbol(
    name: []const u8,
    kind: semantic_index.SymbolKind,
    detail: ?[]const u8,
    parent: ?usize,
    selection_range: frontend.Range,
) workspace_index.Symbol {
    return .{
        .name = name,
        .detail = detail,
        .kind = kind,
        .range = selection_range,
        .selection_range = selection_range,
        .parent = parent,
    };
}

fn mkRange(start_line: u32, start_character: u32, end_line: u32, end_character: u32) frontend.Range {
    return .{
        .start = .{ .line = start_line, .character = start_character },
        .end = .{ .line = end_line, .character = end_character },
    };
}

fn testEntry(symbols: []workspace_index.Symbol) workspace_index.FileEntry {
    return .{
        .arena = undefined,
        .uri = "file:///cold.ora",
        .version = 1,
        .generation = 1,
        .is_cold = false,
        .features = workspace_index.FeatureSet.symbols_only,
        .line_index = testLineIndex(),
        .symbols = symbols,
        .root_symbol_indexes = @constCast(&[_]u32{ 0, 2 }),
        .callable_symbol_indexes = @constCast(&[_]u32{1}),
        .imports = &[_]workspace_index.Import{},
        .occurrences = &[_]references.Occurrence{},
        .imported_members = &[_]references.ImportedMemberOccurrence{},
        .call_edges = &[_]call_hierarchy.CallEdge{},
        .byte_size = 0,
    };
}

fn testLineIndex() line_index.LineIndex {
    return .{ .line_starts = @constCast(&[_]u32{0}), .source_len = 0 };
}
