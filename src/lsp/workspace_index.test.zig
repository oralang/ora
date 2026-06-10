const std = @import("std");
const ora_root = @import("ora_root");

const workspace_index = ora_root.lsp.workspace_index;
const frontend = ora_root.lsp.frontend;
const semantic_index = ora_root.lsp.semantic_index;
const references = ora_root.lsp.references;
const call_hierarchy = ora_root.lsp.call_hierarchy;
const workspace = ora_root.lsp.workspace;
const line_index = ora_root.lsp.line_index;

test "workspace index stores generation-stamped entries and invalidates them" {
    const allocator = std.testing.allocator;

    const semantic_symbols = [_]semantic_index.Symbol{
        .{
            .name = "Main",
            .kind = .contract,
            .range = range(0, 0, 5),
            .selection_range = range(0, 9, 13),
        },
        .{
            .name = "value",
            .detail = "() -> u256",
            .kind = .method,
            .range = range(1, 4, 31),
            .selection_range = range(1, 11, 16),
            .parent = 0,
        },
    };
    var occurrences = [_]references.Occurrence{
        .{ .name = "value", .range = range(1, 11, 16), .definition_range = range(1, 11, 16) },
        .{ .name = "value", .range = range(2, 15, 20), .definition_range = range(1, 11, 16) },
    };
    var imported_members = [_]references.ImportedMemberOccurrence{
        .{ .imported_path = "/tmp/lib.ora", .alias = "Lib", .member_name = "value", .range = range(3, 19, 24) },
    };
    var call_edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 1, .callee_name = "helper", .range = range(2, 15, 21) },
    };
    const imports = [_]workspace.ResolvedImport{
        .{ .specifier = "./lib.ora", .alias = "Lib", .resolved_path = "/tmp/lib.ora" },
    };

    const occurrence_index = references.OccurrenceIndex{ .occurrences = occurrences[0..] };
    const imported_member_index = references.ImportedMemberIndex{ .occurrences = imported_members[0..] };
    const call_edge_index = call_hierarchy.CallEdgeIndex{ .edges = call_edges[0..] };
    var lines = try line_index.LineIndex.init(allocator, "contract Main {\n}\n");
    defer lines.deinit(allocator);

    var index = workspace_index.Index.init(allocator);
    defer index.deinit();

    {
        var entry = try workspace_index.FileEntry.init(
            allocator,
            "file:///tmp/main.ora",
            7,
            1,
            false,
            .{ .occurrences = true, .imported_members = true, .call_edges = true },
            &lines,
            &semantic_symbols,
            &imports,
            &occurrence_index,
            &imported_member_index,
            &call_edge_index,
        );
        errdefer entry.deinit();
        try index.upsert(entry);
    }

    const fresh = index.getFresh(
        "file:///tmp/main.ora",
        7,
        1,
        .{ .occurrences = true, .imported_members = true },
    ).?;
    try std.testing.expectEqual(@as(usize, 2), fresh.symbols.len);
    try std.testing.expectEqual(@as(usize, 1), fresh.rootSymbolIndexes().len);
    try std.testing.expectEqual(@as(u32, 0), fresh.rootSymbolIndexes()[0]);
    try std.testing.expectEqual(@as(usize, 1), fresh.callableSymbolIndexes().len);
    try std.testing.expectEqual(@as(u32, 1), fresh.callableSymbolIndexes()[0]);
    try std.testing.expectEqual(@as(usize, 1), fresh.callableSymbolIndexByRange(range(1, 4, 31)).?);
    try std.testing.expectEqualStrings("value", fresh.callableSymbolNamed("value").?.name);
    try std.testing.expectEqual(@as(usize, 1), fresh.imports.len);
    try std.testing.expectEqualStrings("/tmp/lib.ora", fresh.imports[0].resolved_path);
    try std.testing.expectEqualStrings("value", fresh.occurrenceAt(.{ .line = 2, .character = 16 }).?.name);
    try std.testing.expectEqual(@as(usize, 1), fresh.callEdgeIndex().edges.len);
    try std.testing.expect(fresh.builderItemsBuilt() > 0);
    try std.testing.expect(fresh.builderCapacityRequested() >= fresh.builderItemsBuilt());
    try std.testing.expectEqual(fresh.builderCapacityRequested() - fresh.builderItemsBuilt(), fresh.builderUnusedCapacity());
    try std.testing.expectEqual(@as(usize, 0), fresh.builderGrowthEvents());
    try std.testing.expectEqual(fresh.builderCapacityRequested(), index.builderCapacityRequested());
    try std.testing.expectEqual(fresh.builderItemsBuilt(), index.builderItemsBuilt());
    try std.testing.expectEqual(fresh.builderUnusedCapacity(), index.builderUnusedCapacity());
    try std.testing.expectEqual(fresh.builderGrowthEvents(), index.builderGrowthEvents());
    try std.testing.expect(fresh.sideMapItemsBuilt() > 0);
    try std.testing.expect(fresh.sideMapCapacityRequested() >= fresh.sideMapItemsBuilt());
    try std.testing.expectEqual(fresh.sideMapCapacityRequested() - fresh.sideMapItemsBuilt(), fresh.sideMapUnusedCapacity());
    try std.testing.expectEqual(@as(usize, 0), fresh.sideMapGrowthEvents());
    try std.testing.expectEqual(fresh.sideMapCapacityRequested(), index.sideMapCapacityRequested());
    try std.testing.expectEqual(fresh.sideMapItemsBuilt(), index.sideMapItemsBuilt());
    try std.testing.expectEqual(fresh.sideMapUnusedCapacity(), index.sideMapUnusedCapacity());
    try std.testing.expectEqual(fresh.sideMapGrowthEvents(), index.sideMapGrowthEvents());
    try std.testing.expect(fresh.internedStringItemsBuilt() > 0);
    try std.testing.expect(fresh.internedStringCapacityRequested() >= fresh.internedStringItemsBuilt());
    try std.testing.expectEqual(fresh.internedStringCapacityRequested() - fresh.internedStringItemsBuilt(), fresh.internedStringUnusedCapacity());
    try std.testing.expect(fresh.internedStringGrowthEvents() > 0);
    try std.testing.expectEqual(fresh.internedStringCapacityRequested(), index.internedStringCapacityRequested());
    try std.testing.expectEqual(fresh.internedStringItemsBuilt(), index.internedStringItemsBuilt());
    try std.testing.expectEqual(fresh.internedStringUnusedCapacity(), index.internedStringUnusedCapacity());
    try std.testing.expectEqual(fresh.internedStringGrowthEvents(), index.internedStringGrowthEvents());
    try std.testing.expectEqual(@as(usize, 0), index.coldEntryCount());
    try std.testing.expectEqual(@as(usize, 0), index.coldInternedStringCapacityRequested());

    try std.testing.expect(index.getFresh("file:///tmp/main.ora", 7, 2, .{}) == null);

    index.invalidate("file:///tmp/main.ora");
    try std.testing.expect(index.getFresh("file:///tmp/main.ora", 7, 1, .{}) == null);
}

test "workspace index evicts least-recently-used entries by byte budget" {
    const allocator = std.testing.allocator;

    var index = workspace_index.Index.initWithBudget(allocator, std.math.maxInt(usize));
    defer index.deinit();

    _ = try upsertSymbolEntry(&index, allocator, "file:///tmp/a.ora", "alpha", 1);
    _ = try upsertSymbolEntry(&index, allocator, "file:///tmp/b.ora", "beta", 1);

    index.max_bytes = index.current_bytes;

    try std.testing.expect(index.getFresh("file:///tmp/a.ora", 1, 1, .{}) != null);
    _ = try upsertSymbolEntry(&index, allocator, "file:///tmp/c.ora", "beta", 1);

    try std.testing.expect(index.getFresh("file:///tmp/a.ora", 1, 1, .{}) != null);
    try std.testing.expect(index.getFresh("file:///tmp/b.ora", 1, 1, .{}) == null);
    try std.testing.expect(index.getFresh("file:///tmp/c.ora", 1, 1, .{}) != null);
    try std.testing.expectEqual(@as(usize, 1), index.evictions);
    try std.testing.expect(index.current_bytes <= index.max_bytes);
}

test "workspace index keeps newest entry when one entry exceeds byte budget" {
    const allocator = std.testing.allocator;

    var index = workspace_index.Index.initWithBudget(allocator, 1);
    defer index.deinit();

    _ = try upsertSymbolEntry(&index, allocator, "file:///tmp/oversized.ora", "oversized_symbol", 1);

    try std.testing.expect(index.getFresh("file:///tmp/oversized.ora", 1, 1, .{}) != null);
    try std.testing.expectEqual(@as(usize, 0), index.evictions);
    try std.testing.expect(index.current_bytes > index.max_bytes);
}

test "workspace index interns duplicate retained strings inside entries" {
    const allocator = std.testing.allocator;

    const semantic_symbols = [_]semantic_index.Symbol{
        .{
            .name = "value",
            .detail = "() -> u256",
            .kind = .function,
            .range = range(0, 0, 20),
            .selection_range = range(0, 7, 12),
        },
        .{
            .name = "value",
            .detail = "() -> u256",
            .kind = .function,
            .range = range(2, 0, 20),
            .selection_range = range(2, 7, 12),
        },
    };
    var occurrences = [_]references.Occurrence{
        .{ .name = "value", .range = range(0, 7, 12), .definition_range = range(0, 7, 12) },
        .{ .name = "value", .range = range(3, 11, 16), .definition_range = range(0, 7, 12) },
    };
    var imported_members = [_]references.ImportedMemberOccurrence{
        .{ .imported_path = "/tmp/lib.ora", .alias = "Lib", .member_name = "value", .range = range(4, 19, 24) },
    };
    var call_edges = [_]call_hierarchy.CallEdge{
        .{ .caller_symbol_index = 0, .callee_name = "value", .range = range(3, 11, 16) },
    };
    const imports = [_]workspace.ResolvedImport{
        .{ .specifier = "./lib.ora", .alias = "Lib", .resolved_path = "/tmp/lib.ora" },
    };

    const occurrence_index = references.OccurrenceIndex{ .occurrences = occurrences[0..] };
    const imported_member_index = references.ImportedMemberIndex{ .occurrences = imported_members[0..] };
    const call_edge_index = call_hierarchy.CallEdgeIndex{ .edges = call_edges[0..] };
    var lines = try line_index.LineIndex.init(allocator, "pub fn value() {}\n");
    defer lines.deinit(allocator);

    var entry = try workspace_index.FileEntry.init(
        allocator,
        "file:///tmp/main.ora",
        1,
        1,
        false,
        .{ .occurrences = true, .imported_members = true, .call_edges = true },
        &lines,
        &semantic_symbols,
        &imports,
        &occurrence_index,
        &imported_member_index,
        &call_edge_index,
    );
    defer entry.deinit();

    try std.testing.expect(entry.interned_string_bytes > 0);
    try std.testing.expect(entry.interned_string_count > 0);
    try std.testing.expect(entry.duplicate_string_bytes_saved > 0);
    try std.testing.expect(entry.internedStringCapacityRequested() >= entry.internedStringItemsBuilt());
    try std.testing.expectEqual(entry.internedStringCapacityRequested() - entry.internedStringItemsBuilt(), entry.internedStringUnusedCapacity());
    try std.testing.expect(entry.internedStringGrowthEvents() > 0);
    try std.testing.expectEqual(entry.symbols[0].name.ptr, entry.symbols[1].name.ptr);
    try std.testing.expectEqual(entry.symbols[0].name.ptr, entry.occurrences[0].name.ptr);
    try std.testing.expectEqual(entry.symbols[0].name.ptr, entry.imported_members[0].member_name.ptr);
    try std.testing.expectEqual(entry.symbols[0].name.ptr, entry.call_edges[0].callee_name.ptr);
    try std.testing.expectEqual(entry.imports[0].resolved_path.ptr, entry.imported_members[0].imported_path.ptr);
}

test "workspace index reports cold entry string interning totals" {
    const allocator = std.testing.allocator;

    const semantic_symbols = [_]semantic_index.Symbol{
        .{
            .name = "ColdRoot",
            .kind = .contract,
            .range = range(0, 0, 10),
            .selection_range = range(0, 0, 8),
        },
        .{
            .name = "coldCall",
            .detail = "() -> u256",
            .kind = .function,
            .range = range(1, 0, 18),
            .selection_range = range(1, 7, 15),
            .parent = 0,
        },
    };
    var lines = try line_index.LineIndex.init(allocator, "contract ColdRoot {}\n");
    defer lines.deinit(allocator);

    var index = workspace_index.Index.init(allocator);
    defer index.deinit();

    var entry = try workspace_index.FileEntry.init(
        allocator,
        "file:///tmp/cold.ora",
        0,
        1,
        true,
        workspace_index.FeatureSet.symbols_only,
        &lines,
        &semantic_symbols,
        &.{},
        null,
        null,
        null,
    );
    errdefer entry.deinit();

    const entry_bytes = entry.byte_size;
    const interned_bytes = entry.interned_string_bytes;
    const interned_count = entry.interned_string_count;
    const duplicate_bytes_saved = entry.duplicate_string_bytes_saved;
    const interner_capacity = entry.internedStringCapacityRequested();
    const interner_items = entry.internedStringItemsBuilt();
    const interner_unused = entry.internedStringUnusedCapacity();
    const interner_growth = entry.internedStringGrowthEvents();

    try index.upsert(entry);

    try std.testing.expectEqual(@as(usize, 1), index.coldEntryCount());
    try std.testing.expectEqual(entry_bytes, index.coldBytes());
    try std.testing.expectEqual(interned_bytes, index.coldInternedStringBytes());
    try std.testing.expectEqual(interned_count, index.coldInternedStringCount());
    try std.testing.expectEqual(duplicate_bytes_saved, index.coldDuplicateStringBytesSaved());
    try std.testing.expectEqual(interner_capacity, index.coldInternedStringCapacityRequested());
    try std.testing.expectEqual(interner_items, index.coldInternedStringItemsBuilt());
    try std.testing.expectEqual(interner_unused, index.coldInternedStringUnusedCapacity());
    try std.testing.expectEqual(interner_growth, index.coldInternedStringGrowthEvents());
    try std.testing.expect(index.coldInternedStringCapacityRequested() >= index.coldInternedStringItemsBuilt());
    try std.testing.expectEqual(
        index.coldInternedStringCapacityRequested() - index.coldInternedStringItemsBuilt(),
        index.coldInternedStringUnusedCapacity(),
    );
    try std.testing.expect(index.coldInternedStringGrowthEvents() > 0);
}

fn upsertSymbolEntry(
    index: *workspace_index.Index,
    allocator: std.mem.Allocator,
    uri: []const u8,
    name: []const u8,
    version: i32,
) !usize {
    const semantic_symbols = [_]semantic_index.Symbol{
        .{
            .name = name,
            .kind = .function,
            .range = range(0, 0, 10),
            .selection_range = range(0, 4, 4 + @as(u32, @intCast(name.len))),
        },
    };
    var lines = try line_index.LineIndex.init(allocator, "pub fn x() {}\n");
    defer lines.deinit(allocator);

    var entry = try workspace_index.FileEntry.init(
        allocator,
        uri,
        version,
        1,
        false,
        workspace_index.FeatureSet.symbols_only,
        &lines,
        &semantic_symbols,
        &.{},
        null,
        null,
        null,
    );
    errdefer entry.deinit();

    const byte_size = entry.byte_size;
    try index.upsert(entry);
    return byte_size;
}

fn range(line: u32, start: u32, end: u32) frontend.Range {
    return .{
        .start = .{ .line = line, .character = start },
        .end = .{ .line = line, .character = end },
    };
}
