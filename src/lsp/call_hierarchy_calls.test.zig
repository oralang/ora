const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const call_hierarchy_calls = @import("call_hierarchy_calls.zig");

const call_hierarchy = ora_root.lsp.call_hierarchy;
const frontend = ora_root.lsp.frontend;
const protocol_ranges = @import("protocol_ranges.zig");
const references = ora_root.lsp.references;
const line_index = ora_root.lsp.line_index;
const workspace_index = ora_root.lsp.workspace_index;

const types = lsp.types;

const PassthroughRanges = struct {
    arena: std.mem.Allocator,

    pub fn byteRangeToLsp(_: *const @This(), _: []const u8, source_range: frontend.Range) !types.Range {
        return protocol_ranges.rawRange(source_range);
    }

    pub fn rangesToLsp(self: *const @This(), _: []const u8, ranges: []const frontend.Range) ![]types.Range {
        const converted = try self.arena.alloc(types.Range, ranges.len);
        for (ranges, 0..) |source_range, index| converted[index] = protocol_ranges.rawRange(source_range);
        return converted;
    }
};

test "lsp call hierarchy calls: builds incoming calls from workspace entry edges" {
    var symbols = [_]workspace_index.Symbol{
        callable("read", mkRange(0, 0, 3, 1), mkRange(0, 7, 0, 11)),
        callable("helper", mkRange(5, 0, 7, 1), mkRange(5, 7, 5, 13)),
    };
    var edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = mkRange(2, 11, 2, 17) },
    };
    const entry = testEntry(&symbols, &edges);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const converter = PassthroughRanges{ .arena = arena };

    var calls = std.ArrayList(types.CallHierarchyIncomingCall){};
    try call_hierarchy_calls.appendIncomingMatches(arena, &calls, "file:///main.ora", &entry, "helper", &converter);

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("read", calls.items[0].from.name);
    try std.testing.expectEqual(@as(usize, 1), calls.items[0].fromRanges.len);
    try std.testing.expectEqual(@as(u32, 2), calls.items[0].fromRanges[0].start.line);
}

test "lsp call hierarchy calls: groups incoming ranges by caller" {
    var symbols = [_]workspace_index.Symbol{
        callable("read", mkRange(0, 0, 4, 1), mkRange(0, 7, 0, 11)),
        callable("helper", mkRange(6, 0, 8, 1), mkRange(6, 7, 6, 13)),
    };
    var edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = mkRange(2, 24, 2, 30) },
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = mkRange(3, 11, 3, 17) },
    };
    const entry = testEntry(&symbols, &edges);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const converter = PassthroughRanges{ .arena = arena };

    var calls = std.ArrayList(types.CallHierarchyIncomingCall){};
    try call_hierarchy_calls.appendIncomingMatches(arena, &calls, "file:///main.ora", &entry, "helper", &converter);

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("read", calls.items[0].from.name);
    try std.testing.expectEqual(@as(usize, 2), calls.items[0].fromRanges.len);
    try std.testing.expectEqual(@as(usize, 1), call_hierarchy_calls.incomingMatchCount(&entry, "helper"));
}

test "lsp call hierarchy calls: filters incoming imported calls by target path and range" {
    var symbols = [_]workspace_index.Symbol{
        callable("read", mkRange(0, 0, 5, 1), mkRange(0, 7, 0, 11)),
        callable("other", mkRange(7, 0, 9, 1), mkRange(7, 7, 7, 12)),
    };
    const imported_range = mkRange(2, 20, 2, 26);
    const local_same_name_range = mkRange(3, 11, 3, 17);
    const other_import_range = mkRange(8, 20, 8, 26);
    var edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = imported_range },
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = local_same_name_range },
        .{ .caller_symbol_index = 1, .callee_name = "helper", .range = other_import_range },
    };
    var imported_members = [_]references.ImportedMemberOccurrence{
        .{
            .imported_path = "/tmp/lib.ora",
            .alias = "lib",
            .member_name = "helper",
            .range = imported_range,
        },
        .{
            .imported_path = "/tmp/other.ora",
            .alias = "other",
            .member_name = "helper",
            .range = other_import_range,
        },
    };
    const entry = testEntryWithImportedMembers(&symbols, &edges, &imported_members);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const converter = PassthroughRanges{ .arena = arena };

    var calls = std.ArrayList(types.CallHierarchyIncomingCall){};
    try call_hierarchy_calls.appendIncomingImportedMatches(
        arena,
        &calls,
        "file:///main.ora",
        &entry,
        "/tmp/lib.ora",
        "helper",
        &converter,
    );

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("read", calls.items[0].from.name);
    try std.testing.expectEqual(@as(usize, 1), calls.items[0].fromRanges.len);
    try std.testing.expectEqual(@as(u32, 2), calls.items[0].fromRanges[0].start.line);
    try std.testing.expectEqual(@as(u32, 20), calls.items[0].fromRanges[0].start.character);
}

test "lsp call hierarchy calls: builds outgoing calls from caller range" {
    var symbols = [_]workspace_index.Symbol{
        callable("read", mkRange(0, 0, 3, 1), mkRange(0, 7, 0, 11)),
        callable("helper", mkRange(5, 0, 7, 1), mkRange(5, 7, 5, 13)),
    };
    var edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = mkRange(2, 11, 2, 17) },
    };
    const entry = testEntry(&symbols, &edges);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const converter = PassthroughRanges{ .arena = arena };

    const calls = (try call_hierarchy_calls.outgoingCalls(
        arena,
        "file:///main.ora",
        &entry,
        symbols[0].range,
        &converter,
    )) orelse return error.ExpectedOutgoingCalls;

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("helper", calls[0].to.name);
    try std.testing.expectEqual(@as(usize, 1), calls[0].fromRanges.len);
    try std.testing.expectEqual(@as(u32, 2), calls[0].fromRanges[0].start.line);
}

test "lsp call hierarchy calls: resolves imported outgoing targets" {
    var symbols = [_]workspace_index.Symbol{
        callable("read", mkRange(0, 0, 3, 1), mkRange(0, 7, 0, 11)),
    };
    const member_range = mkRange(2, 15, 2, 21);
    var edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "helper", .range = member_range },
    };
    var imported_members = [_]references.ImportedMemberOccurrence{
        .{
            .imported_path = "/tmp/lib.ora",
            .alias = "lib",
            .member_name = "helper",
            .range = member_range,
        },
    };
    const entry = testEntryWithImportedMembers(&symbols, &edges, &imported_members);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const converter = PassthroughRanges{ .arena = arena };
    var resolver = ImportedResolver{};

    const calls = (try call_hierarchy_calls.outgoingCallsWithResolver(
        arena,
        "file:///main.ora",
        &entry,
        symbols[0].range,
        &converter,
        &resolver,
    )) orelse return error.ExpectedOutgoingCalls;

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("helper", calls[0].to.name);
    try std.testing.expectEqualStrings("file:///tmp/lib.ora", calls[0].to.uri);
    try std.testing.expectEqual(@as(usize, 1), calls[0].fromRanges.len);
    try std.testing.expectEqual(@as(u32, 2), calls[0].fromRanges[0].start.line);
}

const ImportedResolver = struct {
    pub fn resolve(_: *@This(), occurrence: references.ImportedMemberOccurrence) !?call_hierarchy_calls.ResolvedTarget {
        if (!std.mem.eql(u8, occurrence.imported_path, "/tmp/lib.ora")) return null;
        if (!std.mem.eql(u8, occurrence.member_name, "helper")) return null;
        return .{
            .uri = "file:///tmp/lib.ora",
            .symbol = callable("helper", mkRange(0, 0, 2, 1), mkRange(0, 7, 0, 13)),
        };
    }
};

fn callable(name: []const u8, symbol_range: frontend.Range, selection_range: frontend.Range) workspace_index.Symbol {
    return .{
        .name = name,
        .detail = null,
        .kind = .function,
        .range = symbol_range,
        .selection_range = selection_range,
        .parent = null,
    };
}

fn mkRange(start_line: u32, start_character: u32, end_line: u32, end_character: u32) frontend.Range {
    return .{
        .start = .{ .line = start_line, .character = start_character },
        .end = .{ .line = end_line, .character = end_character },
    };
}

fn testEntry(symbols: []workspace_index.Symbol, edges: []call_hierarchy.CallEdge) workspace_index.FileEntry {
    return .{
        .arena = undefined,
        .uri = "file:///main.ora",
        .version = 1,
        .generation = 1,
        .is_cold = false,
        .features = workspace_index.FeatureSet.calls,
        .line_index = testLineIndex(),
        .symbols = symbols,
        .root_symbol_indexes = @constCast(&[_]u32{ 0, 1 }),
        .callable_symbol_indexes = @constCast(&[_]u32{ 0, 1 }),
        .imports = &[_]workspace_index.Import{},
        .occurrences = &[_]references.Occurrence{},
        .imported_members = &[_]references.ImportedMemberOccurrence{},
        .call_edges = edges,
        .byte_size = 0,
    };
}

fn testEntryWithImportedMembers(
    symbols: []workspace_index.Symbol,
    edges: []call_hierarchy.CallEdge,
    imported_members: []references.ImportedMemberOccurrence,
) workspace_index.FileEntry {
    return .{
        .arena = undefined,
        .uri = "file:///main.ora",
        .version = 1,
        .generation = 1,
        .is_cold = false,
        .features = .{ .call_edges = true, .imported_members = true },
        .line_index = testLineIndex(),
        .symbols = symbols,
        .root_symbol_indexes = @constCast(&[_]u32{0}),
        .callable_symbol_indexes = @constCast(&[_]u32{0}),
        .imports = &[_]workspace_index.Import{},
        .occurrences = &[_]references.Occurrence{},
        .imported_members = imported_members,
        .call_edges = edges,
        .byte_size = 0,
    };
}

fn testLineIndex() line_index.LineIndex {
    return .{ .line_starts = @constCast(&[_]u32{0}), .source_len = 0 };
}
