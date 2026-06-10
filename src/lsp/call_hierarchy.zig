const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");
const token_cache = @import("token_cache.zig");

const Allocator = std.mem.Allocator;

pub const CallEdge = struct {
    caller_symbol_index: usize,
    callee_name: []const u8,
    range: frontend.Range,
};

pub const IncomingCallerRanges = struct {
    caller_symbol_index: usize,
    ranges: []frontend.Range,
};

pub const CallEdgeIndex = struct {
    edges: []CallEdge = &.{},
    reserved_capacity: usize = 0,
    builder_growth_events: usize = 0,

    pub fn init(
        allocator: Allocator,
        tokens: []const token_cache.Token,
        symbols: []const semantic_index.Symbol,
    ) !CallEdgeIndex {
        var edges = std.ArrayList(CallEdge){};
        errdefer edges.deinit(allocator);
        const reserved_capacity = tokens.len;
        try edges.ensureTotalCapacity(allocator, reserved_capacity);

        for (symbols, 0..) |symbol, symbol_index| {
            if (!isCallable(symbol.kind)) continue;

            var token_index: usize = 0;
            while (token_index < tokens.len) : (token_index += 1) {
                const token = tokens[token_index];
                if (token.type != .Identifier) continue;
                if (token_index + 1 >= tokens.len or tokens[token_index + 1].type != .LeftParen) continue;

                const range = tokenSelectionRange(token);
                if (!positionInRange(range.start, symbol.range)) continue;
                if (positionInRange(range.start, symbol.selection_range)) continue;

                edges.appendAssumeCapacity(.{
                    .caller_symbol_index = symbol_index,
                    .callee_name = token.lexeme,
                    .range = range,
                });
            }
        }

        return .{
            .edges = try edges.toOwnedSlice(allocator),
            .reserved_capacity = reserved_capacity,
        };
    }

    pub fn deinit(self: *CallEdgeIndex, allocator: Allocator) void {
        allocator.free(self.edges);
        self.* = .{};
    }

    pub fn hasIncomingName(self: *const CallEdgeIndex, caller_symbol_index: usize, callee_name: []const u8) bool {
        for (self.edges) |edge| {
            if (edge.caller_symbol_index == caller_symbol_index and std.mem.eql(u8, edge.callee_name, callee_name)) {
                return true;
            }
        }
        return false;
    }

    pub fn estimatedByteSize(self: *const CallEdgeIndex) usize {
        return bytesFor(CallEdge, self.edges.len);
    }

    pub fn builderCapacityRequested(self: *const CallEdgeIndex) usize {
        return self.reserved_capacity;
    }

    pub fn builderItemsBuilt(self: *const CallEdgeIndex) usize {
        return self.edges.len;
    }

    pub fn builderUnusedCapacity(self: *const CallEdgeIndex) usize {
        return if (self.reserved_capacity > self.edges.len) self.reserved_capacity - self.edges.len else 0;
    }

    pub fn builderGrowthEvents(self: *const CallEdgeIndex) usize {
        return self.builder_growth_events;
    }

    pub fn collectIncomingRanges(
        self: *const CallEdgeIndex,
        allocator: Allocator,
        caller_symbol_index: usize,
        callee_name: []const u8,
    ) ![]frontend.Range {
        var ranges = std.ArrayList(frontend.Range){};
        errdefer ranges.deinit(allocator);
        try ranges.ensureTotalCapacity(allocator, self.matchingIncomingCount(caller_symbol_index, callee_name));

        for (self.edges) |edge| {
            if (edge.caller_symbol_index != caller_symbol_index) continue;
            if (!std.mem.eql(u8, edge.callee_name, callee_name)) continue;
            ranges.appendAssumeCapacity(edge.range);
        }

        return ranges.toOwnedSlice(allocator);
    }

    pub fn collectIncomingCallerRanges(
        self: *const CallEdgeIndex,
        allocator: Allocator,
        callee_name: []const u8,
    ) ![]IncomingCallerRanges {
        var counts = std.ArrayList(IncomingCount){};
        defer counts.deinit(allocator);

        for (self.edges) |edge| {
            if (!std.mem.eql(u8, edge.callee_name, callee_name)) continue;

            if (findIncomingCount(counts.items, edge.caller_symbol_index)) |index| {
                counts.items[index].count += 1;
            } else {
                try counts.append(allocator, .{
                    .caller_symbol_index = edge.caller_symbol_index,
                    .count = 1,
                });
            }
        }

        const groups = try allocator.alloc(IncomingCallerRanges, counts.items.len);
        errdefer allocator.free(groups);

        var initialized: usize = 0;
        errdefer {
            for (groups[0..initialized]) |group| allocator.free(group.ranges);
        }

        for (counts.items, 0..) |count, index| {
            groups[index] = .{
                .caller_symbol_index = count.caller_symbol_index,
                .ranges = try allocator.alloc(frontend.Range, count.count),
            };
            initialized += 1;
        }

        for (self.edges) |edge| {
            if (!std.mem.eql(u8, edge.callee_name, callee_name)) continue;
            const index = findIncomingCount(counts.items, edge.caller_symbol_index) orelse continue;
            const fill_index = counts.items[index].filled;
            groups[index].ranges[fill_index] = edge.range;
            counts.items[index].filled += 1;
        }

        return groups;
    }

    pub fn incomingCallerCount(self: *const CallEdgeIndex, callee_name: []const u8) usize {
        var count: usize = 0;
        for (self.edges, 0..) |edge, edge_index| {
            if (!std.mem.eql(u8, edge.callee_name, callee_name)) continue;
            if (hasEarlierIncomingCaller(self.edges[0..edge_index], edge.caller_symbol_index, callee_name)) continue;
            count += 1;
        }
        return count;
    }

    pub fn collectUniqueOutgoing(
        self: *const CallEdgeIndex,
        allocator: Allocator,
        caller_symbol_index: usize,
    ) ![]CallEdge {
        var edges = std.ArrayList(CallEdge){};
        errdefer edges.deinit(allocator);

        for (self.edges) |edge| {
            if (edge.caller_symbol_index != caller_symbol_index) continue;
            if (containsCallee(edges.items, edge.callee_name)) continue;
            try edges.append(allocator, edge);
        }

        return edges.toOwnedSlice(allocator);
    }

    fn matchingIncomingCount(self: *const CallEdgeIndex, caller_symbol_index: usize, callee_name: []const u8) usize {
        var count: usize = 0;
        for (self.edges) |edge| {
            if (edge.caller_symbol_index == caller_symbol_index and std.mem.eql(u8, edge.callee_name, callee_name)) {
                count += 1;
            }
        }
        return count;
    }
};

const IncomingCount = struct {
    caller_symbol_index: usize,
    count: usize,
    filled: usize = 0,
};

fn findIncomingCount(counts: []const IncomingCount, caller_symbol_index: usize) ?usize {
    for (counts, 0..) |count, index| {
        if (count.caller_symbol_index == caller_symbol_index) return index;
    }
    return null;
}

fn hasEarlierIncomingCaller(edges: []const CallEdge, caller_symbol_index: usize, callee_name: []const u8) bool {
    for (edges) |edge| {
        if (edge.caller_symbol_index == caller_symbol_index and std.mem.eql(u8, edge.callee_name, callee_name)) return true;
    }
    return false;
}

fn containsCallee(edges: []const CallEdge, callee_name: []const u8) bool {
    for (edges) |edge| {
        if (std.mem.eql(u8, edge.callee_name, callee_name)) return true;
    }
    return false;
}

fn isCallable(kind: semantic_index.SymbolKind) bool {
    return kind == .function or kind == .method;
}

fn tokenSelectionRange(token: token_cache.Token) frontend.Range {
    const start_line = if (token.line > 0) token.line - 1 else 0;
    const start_char = if (token.column > 0) token.column - 1 else 0;
    const lexeme_len = std.math.cast(u32, token.lexeme.len) orelse std.math.maxInt(u32);
    const end_char = std.math.add(u32, start_char, lexeme_len) catch std.math.maxInt(u32);

    return .{
        .start = .{ .line = start_line, .character = start_char },
        .end = .{ .line = start_line, .character = end_char },
    };
}

fn positionInRange(pos: frontend.Position, range: frontend.Range) bool {
    if (pos.line < range.start.line) return false;
    if (pos.line > range.end.line) return false;
    if (pos.line == range.start.line and pos.character < range.start.character) return false;
    if (pos.line == range.end.line and pos.character > range.end.character) return false;
    return true;
}

fn bytesFor(comptime T: type, len: usize) usize {
    return std.math.mul(usize, @sizeOf(T), len) catch std.math.maxInt(usize);
}
