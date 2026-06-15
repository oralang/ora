const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const call_hierarchy = ora_root.lsp.call_hierarchy;
const semantic_index = ora_root.lsp.semantic_index;
const token_cache = ora_root.lsp.token_cache;
const test_analysis = @import("test_analysis.zig");

fn findSymbolIndex(symbols: []const semantic_index.Symbol, name: []const u8, kind: semantic_index.SymbolKind) ?usize {
    for (symbols, 0..) |symbol, index| {
        if (symbol.kind == kind and std.mem.eql(u8, symbol.name, name)) return index;
    }
    return null;
}

test "lsp call hierarchy: call edge index records function calls by caller" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\pub fn read(input: u256) -> u256 {
        \\    return helper(input);
        \\}
    ;

    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    var cache = try token_cache.Cache.init(testing.allocator, source);
    defer cache.deinit(testing.allocator);

    var edges = try call_hierarchy.CallEdgeIndex.init(testing.allocator, cache.tokens, index.symbols);
    defer edges.deinit(testing.allocator);

    try testing.expectEqual(cache.tokens.len, edges.builderCapacityRequested());
    try testing.expect(edges.builderItemsBuilt() > 0);
    try testing.expect(edges.builderUnusedCapacity() > 0);
    try testing.expect(edges.builderGrowthEvents() > 0);

    const read_index = findSymbolIndex(index.symbols, "read", .function) orelse return error.TestExpectedEqual;
    try testing.expect(edges.hasIncomingName(read_index, "helper"));

    const incoming = try edges.collectIncomingRanges(testing.allocator, read_index, "helper");
    defer testing.allocator.free(incoming);
    try testing.expectEqual(@as(usize, 1), incoming.len);

    const incoming_groups = try edges.collectIncomingCallerRanges(testing.allocator, "helper");
    defer {
        for (incoming_groups) |group| testing.allocator.free(group.ranges);
        testing.allocator.free(incoming_groups);
    }
    try testing.expectEqual(@as(usize, 1), incoming_groups.len);
    try testing.expectEqual(read_index, incoming_groups[0].caller_symbol_index);
    try testing.expectEqual(@as(usize, 1), incoming_groups[0].ranges.len);
    try testing.expectEqual(@as(usize, 1), edges.incomingCallerCount("helper"));

    const outgoing = try edges.collectUniqueOutgoing(testing.allocator, read_index);
    defer testing.allocator.free(outgoing);
    try testing.expectEqual(@as(usize, 1), outgoing.len);
    try testing.expectEqualStrings("helper", outgoing[0].callee_name);
}

test "lsp call hierarchy: outgoing call edges are unique by callee name" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\pub fn read(input: u256) -> u256 {
        \\    let first: u256 = helper(input);
        \\    return helper(first);
        \\}
    ;

    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    var cache = try token_cache.Cache.init(testing.allocator, source);
    defer cache.deinit(testing.allocator);

    var edges = try call_hierarchy.CallEdgeIndex.init(testing.allocator, cache.tokens, index.symbols);
    defer edges.deinit(testing.allocator);

    const read_index = findSymbolIndex(index.symbols, "read", .function) orelse return error.TestExpectedEqual;
    const outgoing = try edges.collectUniqueOutgoing(testing.allocator, read_index);
    defer testing.allocator.free(outgoing);

    try testing.expectEqual(@as(usize, 1), outgoing.len);
    try testing.expectEqualStrings("helper", outgoing[0].callee_name);
}

test "lsp call hierarchy: incoming caller groups keep all call ranges for one caller" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\pub fn read(input: u256) -> u256 {
        \\    let first: u256 = helper(input);
        \\    return helper(first);
        \\}
    ;

    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    var cache = try token_cache.Cache.init(testing.allocator, source);
    defer cache.deinit(testing.allocator);

    var edges = try call_hierarchy.CallEdgeIndex.init(testing.allocator, cache.tokens, index.symbols);
    defer edges.deinit(testing.allocator);

    const read_index = findSymbolIndex(index.symbols, "read", .function) orelse return error.TestExpectedEqual;
    const incoming_groups = try edges.collectIncomingCallerRanges(testing.allocator, "helper");
    defer {
        for (incoming_groups) |group| testing.allocator.free(group.ranges);
        testing.allocator.free(incoming_groups);
    }

    try testing.expectEqual(@as(usize, 1), incoming_groups.len);
    try testing.expectEqual(read_index, incoming_groups[0].caller_symbol_index);
    try testing.expectEqual(@as(usize, 2), incoming_groups[0].ranges.len);
    try testing.expectEqual(@as(usize, 1), edges.incomingCallerCount("helper"));
}
