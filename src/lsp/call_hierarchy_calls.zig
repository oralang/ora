const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const call_hierarchy = ora_root.lsp.call_hierarchy;
const semantic_index = ora_root.lsp.semantic_index;
const references = ora_root.lsp.references;
const workspace_index = ora_root.lsp.workspace_index;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const ResolvedTarget = struct {
    uri: []const u8,
    symbol: workspace_index.Symbol,
};

pub fn appendIncomingMatches(
    arena: Allocator,
    calls: *std.ArrayList(types.CallHierarchyIncomingCall),
    uri: []const u8,
    entry: *const workspace_index.FileEntry,
    target_name: []const u8,
    range_converter: anytype,
) !void {
    const call_edge_index = entry.callEdgeIndex();
    const caller_groups = try call_edge_index.collectIncomingCallerRanges(arena, target_name);
    defer {
        for (caller_groups) |group| arena.free(group.ranges);
        arena.free(caller_groups);
    }

    for (caller_groups) |caller_group| {
        const symbol_index = caller_group.caller_symbol_index;
        if (symbol_index >= entry.symbols.len) continue;
        if (!isCallableSymbolIndex(entry, symbol_index)) continue;
        const sym = entry.symbols[symbol_index];

        try calls.append(arena, .{
            .from = .{
                .name = try arena.dupe(u8, sym.name),
                .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
                .uri = try arena.dupe(u8, uri),
                .range = try range_converter.byteRangeToLsp(uri, sym.range),
                .selectionRange = try range_converter.byteRangeToLsp(uri, sym.selection_range),
                .detail = if (sym.detail) |d| try arena.dupe(u8, d) else null,
            },
            .fromRanges = try range_converter.rangesToLsp(uri, caller_group.ranges),
        });
    }
}

pub fn appendIncomingImportedMatches(
    arena: Allocator,
    calls: *std.ArrayList(types.CallHierarchyIncomingCall),
    uri: []const u8,
    entry: *const workspace_index.FileEntry,
    target_path: []const u8,
    target_name: []const u8,
    range_converter: anytype,
) !void {
    var caller_counts = std.ArrayList(IncomingCallerCount){};
    defer {
        for (caller_counts.items) |count| arena.free(count.ranges);
        caller_counts.deinit(arena);
    }

    for (entry.call_edges) |edge| {
        if (importedMemberForCallPath(entry, edge, target_path, target_name) == null) continue;

        if (findIncomingCallerCount(caller_counts.items, edge.caller_symbol_index)) |index| {
            caller_counts.items[index].count += 1;
        } else {
            try caller_counts.append(arena, .{
                .caller_symbol_index = edge.caller_symbol_index,
                .count = 1,
            });
        }
    }

    for (caller_counts.items) |*caller_count| {
        caller_count.ranges = try arena.alloc(frontend.Range, caller_count.count);
    }

    for (entry.call_edges) |edge| {
        if (importedMemberForCallPath(entry, edge, target_path, target_name) == null) continue;
        const count_index = findIncomingCallerCount(caller_counts.items, edge.caller_symbol_index) orelse continue;
        const caller_count = &caller_counts.items[count_index];
        caller_count.ranges[caller_count.filled] = edge.range;
        caller_count.filled += 1;
    }

    for (caller_counts.items) |caller_count| {
        const symbol_index = caller_count.caller_symbol_index;
        if (symbol_index >= entry.symbols.len) continue;
        if (!isCallableSymbolIndex(entry, symbol_index)) continue;
        const sym = entry.symbols[symbol_index];

        try calls.append(arena, .{
            .from = .{
                .name = try arena.dupe(u8, sym.name),
                .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
                .uri = try arena.dupe(u8, uri),
                .range = try range_converter.byteRangeToLsp(uri, sym.range),
                .selectionRange = try range_converter.byteRangeToLsp(uri, sym.selection_range),
                .detail = if (sym.detail) |d| try arena.dupe(u8, d) else null,
            },
            .fromRanges = try range_converter.rangesToLsp(uri, caller_count.ranges[0..caller_count.filled]),
        });
    }
}

pub fn incomingMatchCount(
    entry: *const workspace_index.FileEntry,
    target_name: []const u8,
) usize {
    const call_edge_index = entry.callEdgeIndex();
    return call_edge_index.incomingCallerCount(target_name);
}

fn isCallableSymbolIndex(entry: *const workspace_index.FileEntry, symbol_index: usize) bool {
    for (entry.callableSymbolIndexes()) |raw_index| {
        if (@as(usize, raw_index) == symbol_index) return true;
    }
    return false;
}

pub fn outgoingCalls(
    arena: Allocator,
    caller_uri: []const u8,
    entry: *const workspace_index.FileEntry,
    caller_range: frontend.Range,
    range_converter: anytype,
) !?[]types.CallHierarchyOutgoingCall {
    var resolver = NoImportedTargetResolver{};
    return outgoingCallsWithResolver(arena, caller_uri, entry, caller_range, range_converter, &resolver);
}

pub fn outgoingCallsWithResolver(
    arena: Allocator,
    caller_uri: []const u8,
    entry: *const workspace_index.FileEntry,
    caller_range: frontend.Range,
    range_converter: anytype,
    imported_resolver: anytype,
) !?[]types.CallHierarchyOutgoingCall {
    const caller_symbol_index = entry.callableSymbolIndexByRange(caller_range) orelse return null;
    const call_edge_index = entry.callEdgeIndex();
    const targets = try call_edge_index.collectUniqueOutgoing(arena, caller_symbol_index);
    defer arena.free(targets);
    if (targets.len == 0) return null;

    var calls = std.ArrayList(types.CallHierarchyOutgoingCall){};
    try calls.ensureTotalCapacity(arena, targets.len);
    for (targets) |target| {
        const resolved = if (entry.callableSymbolNamed(target.callee_name)) |sym|
            ResolvedTarget{ .uri = caller_uri, .symbol = sym.* }
        else if (importedMemberForCall(entry, target)) |imported_member|
            (try imported_resolver.resolve(imported_member)) orelse continue
        else
            continue;

        const from_ranges = try range_converter.rangesToLsp(caller_uri, &.{target.range});
        calls.appendAssumeCapacity(.{
            .to = .{
                .name = try arena.dupe(u8, resolved.symbol.name),
                .kind = @enumFromInt(semantic_index.toLspKind(resolved.symbol.kind)),
                .uri = try arena.dupe(u8, resolved.uri),
                .range = try range_converter.byteRangeToLsp(resolved.uri, resolved.symbol.range),
                .selectionRange = try range_converter.byteRangeToLsp(resolved.uri, resolved.symbol.selection_range),
                .detail = if (resolved.symbol.detail) |d| try arena.dupe(u8, d) else null,
            },
            .fromRanges = from_ranges,
        });
    }

    if (calls.items.len == 0) return null;
    return try calls.toOwnedSlice(arena);
}

const NoImportedTargetResolver = struct {
    pub fn resolve(_: *@This(), _: references.ImportedMemberOccurrence) !?ResolvedTarget {
        return null;
    }
};

fn importedMemberForCall(
    entry: *const workspace_index.FileEntry,
    target: call_hierarchy.CallEdge,
) ?references.ImportedMemberOccurrence {
    for (entry.imported_members) |imported_member| {
        if (!std.mem.eql(u8, imported_member.member_name, target.callee_name)) continue;
        if (!rangesEqual(imported_member.range, target.range)) continue;
        return imported_member;
    }
    return null;
}

fn importedMemberForCallPath(
    entry: *const workspace_index.FileEntry,
    target: call_hierarchy.CallEdge,
    target_path: []const u8,
    target_name: []const u8,
) ?references.ImportedMemberOccurrence {
    if (!std.mem.eql(u8, target.callee_name, target_name)) return null;
    const imported_member = importedMemberForCall(entry, target) orelse return null;
    if (!std.mem.eql(u8, imported_member.imported_path, target_path)) return null;
    return imported_member;
}

const IncomingCallerCount = struct {
    caller_symbol_index: usize,
    count: usize,
    filled: usize = 0,
    ranges: []frontend.Range = &.{},
};

fn findIncomingCallerCount(counts: []const IncomingCallerCount, caller_symbol_index: usize) ?usize {
    for (counts, 0..) |count, index| {
        if (count.caller_symbol_index == caller_symbol_index) return index;
    }
    return null;
}

fn rangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}
