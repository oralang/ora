/// Source map for mapping EVM bytecode PCs to source file locations.
/// Used by the debugger to provide source-level stepping.
const std = @import("std");

pub const SourceMap = struct {
    pub const LineProvenance = enum {
        direct,
        synthetic,
        mixed,

        pub fn label(value: LineProvenance) []const u8 {
            return switch (value) {
                .direct => "direct",
                .synthetic => "synthetic",
                .mixed => "direct+synthetic",
            };
        }
    };

    pub const StatementKind = enum {
        runtime,
        runtime_guard,

        pub fn fromString(text: []const u8) ?StatementKind {
            if (std.mem.eql(u8, text, "runtime")) return .runtime;
            if (std.mem.eql(u8, text, "runtime_guard")) return .runtime_guard;
            return null;
        }

        pub fn asString(kind: StatementKind) []const u8 {
            return switch (kind) {
                .runtime => "runtime",
                .runtime_guard => "runtime_guard",
            };
        }
    };

    runtime_start_pc: ?u32 = null,
    entries: []Entry,
    allocator: std.mem.Allocator,

    pub const Entry = struct {
        /// Serialized SIR op index backing this PC mapping, if available.
        idx: ?u32 = null,
        /// Bytecode program counter offset
        pc: u32,
        /// Source file path
        file: []const u8,
        /// 1-based line number in source
        line: u32,
        /// 1-based column number in source
        col: u32,
        /// Stable source-level statement identity, when available.
        statement_id: ?u32 = null,
        /// Provenance statement identity for synthetic or moved regions.
        origin_statement_id: ?u32 = null,
        /// Compiler-emitted contiguous execution region for this lowered run.
        execution_region_id: ?u32 = null,
        /// 1-based occurrence count of this statement's run within the function.
        statement_run_index: ?u32 = null,
        /// Serialized function name for this lowered region, when available.
        function: ?[]const u8 = null,
        /// 1-based line number in emitted textual .sir, when available.
        sir_line: ?u32 = null,
        /// Whether this entry is from compiler-generated expansion instead of a
        /// directly serialized source op.
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
        /// Whether this entry appears earlier in execution order than its
        /// statement-id ordering suggests.
        is_hoisted: bool = false,
        /// Whether the same statement produces multiple statement-boundary
        /// regions in backend order.
        is_duplicated: bool = false,
        /// Whether this PC is the start of a source-level statement.
        /// The debugger stops here during step-over (not on every opcode).
        is_statement: bool,
        /// Runtime statement classification, when available.
        kind: ?StatementKind = null,
    };

    /// Look up the source location for a given PC.
    /// Returns the entry with the highest PC <= the query PC.
    pub fn getEntry(self: *const SourceMap, pc: u32) ?*const Entry {
        if (self.entries.len == 0) return null;

        // Binary search for the last entry where entry.pc <= pc
        var lo: usize = 0;
        var hi: usize = self.entries.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.entries[mid].pc <= pc) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == 0) return null;
        return &self.entries[lo - 1];
    }

    /// Get the first statement-boundary entry for a given source line.
    ///
    /// Entries are sorted by PC, not by source location, so matches for a single
    /// source line may not be contiguous. This helper therefore returns only the
    /// first statement entry for the line.
    pub fn getFirstStatementEntryForLine(self: *const SourceMap, file: []const u8, line: u32) ?*const Entry {
        for (self.entries) |*entry| {
            if (entry.line == line and entry.is_statement and std.mem.eql(u8, entry.file, file)) {
                return entry;
            }
        }
        return null;
    }

    /// Compatibility wrapper for older callers.
    /// Returns a single-element slice containing the first statement entry.
    pub fn getPcsForLine(self: *const SourceMap, file: []const u8, line: u32) []const Entry {
        if (self.getFirstStatementEntryForLine(file, line)) |entry| {
            const idx = @intFromPtr(entry) - @intFromPtr(self.entries.ptr);
            const entry_index = @divExact(idx, @sizeOf(Entry));
            return self.entries[entry_index .. entry_index + 1];
        }
        return &[_]Entry{};
    }

    /// Get the first statement PC for a given source line (for setting breakpoints).
    pub fn getFirstPcForLine(self: *const SourceMap, file: []const u8, line: u32) ?u32 {
        return if (self.getFirstStatementEntryForLine(file, line)) |entry| entry.pc else null;
    }

    pub fn getStatementKindForLine(self: *const SourceMap, file: []const u8, line: u32) ?StatementKind {
        for (self.entries) |*entry| {
            if (entry.line != line) continue;
            if (!std.mem.eql(u8, entry.file, file)) continue;
            if (!entry.is_statement) continue;
            return entry.kind;
        }
        return null;
    }

    pub fn getLineProvenance(self: *const SourceMap, file: []const u8, line: u32) ?LineProvenance {
        var saw_direct = false;
        var saw_synthetic = false;
        for (self.entries) |*entry| {
            if (entry.line != line) continue;
            if (!std.mem.eql(u8, entry.file, file)) continue;
            if (!entry.is_statement) continue;
            if (entry.is_synthetic)
                saw_synthetic = true
            else
                saw_direct = true;
        }
        if (saw_direct and saw_synthetic) return .mixed;
        if (saw_direct) return .direct;
        if (saw_synthetic) return .synthetic;
        return null;
    }

    pub fn getFirstLineForStatementId(self: *const SourceMap, file: []const u8, statement_id: u32) ?u32 {
        for (self.entries) |*entry| {
            if (!std.mem.eql(u8, entry.file, file)) continue;
            if (entry.statement_id != statement_id) continue;
            return entry.line;
        }
        return null;
    }

    pub fn getFirstLineForOriginStatementId(self: *const SourceMap, file: []const u8, origin_statement_id: u32) ?u32 {
        for (self.entries) |*entry| {
            if (!std.mem.eql(u8, entry.file, file)) continue;
            if (entry.origin_statement_id != origin_statement_id) continue;
            return entry.line;
        }
        return null;
    }

    pub fn hasAnyEntryForLine(self: *const SourceMap, file: []const u8, line: u32) bool {
        for (self.entries) |*entry| {
            if (entry.line == line and std.mem.eql(u8, entry.file, file)) return true;
        }
        return false;
    }

    /// Load a source map from JSON.
    /// Expected format:
    /// {
    ///   "version": 1,
    ///   "sources": ["main.ora"],
    ///   "entries": [
    ///     { "pc": 0, "src": 0, "line": 3, "col": 5, "stmt": true },
    ///     ...
    ///   ]
    /// }
    pub fn loadFromJson(allocator: std.mem.Allocator, json_bytes: []const u8) !SourceMap {
        const parsed = try std.json.parseFromSlice(JsonSourceMap, allocator, json_bytes, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        const json_entries = parsed.value.entries;
        const entries = try allocator.alloc(Entry, json_entries.len);

        for (json_entries, 0..) |je, i| {
            const file_path = if (je.file) |file|
                file
            else if (je.src) |src_index|
                if (src_index < parsed.value.sources.len) parsed.value.sources[src_index] else return error.InvalidSourceMap
            else
                return error.InvalidSourceMap;
            entries[i] = .{
                .idx = je.idx,
                .pc = je.pc,
                .file = try allocator.dupe(u8, file_path),
                .line = je.line,
                .col = je.col,
                .statement_id = je.statement_id,
                .origin_statement_id = je.origin_statement_id,
                .execution_region_id = je.execution_region_id,
                .statement_run_index = je.statement_run_index,
                .function = if (je.function) |name| try allocator.dupe(u8, name) else null,
                .sir_line = je.sir_line,
                .is_synthetic = je.is_synthetic,
                .synthetic_index = je.synthetic_index,
                .synthetic_count = je.synthetic_count,
                .synthetic_path = je.synthetic_path,
                .is_hoisted = je.is_hoisted,
                .is_duplicated = je.is_duplicated,
                .is_statement = je.stmt,
                .kind = if (je.kind) |kind_text| StatementKind.fromString(kind_text) else null,
            };
        }

        return .{
            .runtime_start_pc = parsed.value.runtime_start_pc,
            .entries = entries,
            .allocator = allocator,
        };
    }

    /// Create a source map from a pre-built entry list (for testing or programmatic use).
    pub fn fromEntries(allocator: std.mem.Allocator, entries: []const Entry) !SourceMap {
        const owned = try allocator.alloc(Entry, entries.len);
        for (entries, 0..) |e, i| {
            owned[i] = .{
                .idx = e.idx,
                .pc = e.pc,
                .file = try allocator.dupe(u8, e.file),
                .line = e.line,
                .col = e.col,
                .statement_id = e.statement_id,
                .origin_statement_id = e.origin_statement_id,
                .execution_region_id = e.execution_region_id,
                .statement_run_index = e.statement_run_index,
                .function = if (e.function) |name| try allocator.dupe(u8, name) else null,
                .sir_line = e.sir_line,
                .is_synthetic = e.is_synthetic,
                .synthetic_index = e.synthetic_index,
                .synthetic_count = e.synthetic_count,
                .synthetic_path = e.synthetic_path,
                .is_hoisted = e.is_hoisted,
                .is_duplicated = e.is_duplicated,
                .is_statement = e.is_statement,
                .kind = e.kind,
            };
        }
        return .{
            .runtime_start_pc = null,
            .entries = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SourceMap) void {
        for (self.entries) |entry| {
            self.allocator.free(entry.file);
            if (entry.function) |function_name| self.allocator.free(function_name);
        }
        self.allocator.free(self.entries);
    }

    const JsonSourceMap = struct {
        version: u32 = 1,
        runtime_start_pc: ?u32 = null,
        sources: []const []const u8 = &.{},
        entries: []const JsonEntry = &.{},
    };

    const JsonEntry = struct {
        idx: ?u32 = null,
        pc: u32,
        file: ?[]const u8 = null,
        src: ?usize = null,
        line: u32,
        col: u32,
        statement_id: ?u32 = null,
        origin_statement_id: ?u32 = null,
        execution_region_id: ?u32 = null,
        statement_run_index: ?u32 = null,
        function: ?[]const u8 = null,
        sir_line: ?u32 = null,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
        is_hoisted: bool = false,
        is_duplicated: bool = false,
        stmt: bool,
        kind: ?[]const u8 = null,
    };
};

test "SourceMap.getEntry binary search" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .statement_id = 10, .origin_statement_id = 10, .function = "foo", .sir_line = 10, .is_statement = true, .kind = .runtime },
        .{ .idx = 1, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .statement_id = 11, .origin_statement_id = 11, .function = "foo", .sir_line = 11, .is_statement = true, .kind = .runtime },
        .{ .idx = 2, .pc = 12, .file = "test.ora", .line = 3, .col = 1, .statement_id = 12, .origin_statement_id = 12, .function = "foo", .sir_line = 12, .is_statement = true, .kind = .runtime },
        .{ .idx = 3, .pc = 20, .file = "test.ora", .line = 4, .col = 1, .statement_id = 13, .origin_statement_id = 13, .function = "foo", .sir_line = 13, .is_statement = true, .kind = .runtime },
    };
    var sm = try SourceMap.fromEntries(allocator, &entries);
    defer sm.deinit();

    // Exact match
    try std.testing.expectEqual(@as(u32, 1), sm.getEntry(0).?.line);
    try std.testing.expectEqual(@as(u32, 2), sm.getEntry(5).?.line);

    // Between entries: should return the previous entry
    try std.testing.expectEqual(@as(u32, 1), sm.getEntry(3).?.line);
    try std.testing.expectEqual(@as(u32, 2), sm.getEntry(10).?.line);
    try std.testing.expectEqual(@as(u32, 3), sm.getEntry(15).?.line);

    // Past last entry
    try std.testing.expectEqual(@as(u32, 4), sm.getEntry(100).?.line);
}

test "SourceMap.getFirstPcForLine" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .statement_id = 1, .origin_statement_id = 1, .function = "foo", .sir_line = 4, .is_statement = true, .kind = .runtime },
        .{ .idx = 1, .pc = 3, .file = "test.ora", .line = 1, .col = 5, .statement_id = 1, .origin_statement_id = 1, .function = "foo", .sir_line = 5, .is_statement = false, .kind = null },
        .{ .idx = 2, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .statement_id = 2, .origin_statement_id = 2, .function = "foo", .sir_line = 6, .is_statement = true, .kind = .runtime },
    };
    var sm = try SourceMap.fromEntries(allocator, &entries);
    defer sm.deinit();

    try std.testing.expectEqual(@as(u32, 0), sm.getFirstPcForLine("test.ora", 1).?);
    try std.testing.expectEqual(@as(u32, 5), sm.getFirstPcForLine("test.ora", 2).?);
    try std.testing.expect(sm.getFirstPcForLine("test.ora", 99) == null);
}

test "SourceMap.loadFromJson" {
    const allocator = std.testing.allocator;
    const json =
        \\{"version":1,"entries":[
        \\  {"idx":7,"pc":0,"file":"main.ora","line":3,"col":5,"statement_id":9,"origin_statement_id":9,"function":"foo","sir_line":17,"is_synthetic":true,"synthetic_index":1,"synthetic_count":3,"synthetic_path":"0.2/1.3","is_hoisted":true,"is_duplicated":true,"stmt":true,"kind":"runtime_guard"},
        \\  {"idx":8,"pc":6,"file":"main.ora","line":4,"col":5,"statement_id":10,"origin_statement_id":10,"function":"foo","sir_line":18,"stmt":true,"kind":"runtime"}
        \\]}
    ;
    var sm = try SourceMap.loadFromJson(allocator, json);
    defer sm.deinit();

    try std.testing.expectEqual(@as(usize, 2), sm.entries.len);
    try std.testing.expectEqual(@as(u32, 0), sm.entries[0].pc);
    try std.testing.expectEqual(@as(?u32, 7), sm.entries[0].idx);
    try std.testing.expectEqual(@as(u32, 3), sm.entries[0].line);
    try std.testing.expectEqual(@as(?u32, 17), sm.entries[0].sir_line);
    try std.testing.expectEqual(@as(?u32, 9), sm.entries[0].statement_id);
    try std.testing.expectEqual(@as(?u32, 9), sm.entries[0].origin_statement_id);
    try std.testing.expectEqualStrings("foo", sm.entries[0].function.?);
    try std.testing.expect(sm.entries[0].is_synthetic);
    try std.testing.expectEqual(@as(?u32, 1), sm.entries[0].synthetic_index);
    try std.testing.expectEqual(@as(?u32, 3), sm.entries[0].synthetic_count);
    try std.testing.expectEqualStrings("0.2/1.3", sm.entries[0].synthetic_path.?);
    try std.testing.expect(sm.entries[0].is_hoisted);
    try std.testing.expect(sm.entries[0].is_duplicated);
    try std.testing.expect(sm.entries[0].is_statement);
    try std.testing.expectEqual(SourceMap.StatementKind.runtime_guard, sm.entries[0].kind.?);
    try std.testing.expectEqualStrings("main.ora", sm.entries[0].file);
}

test "SourceMap.fromEntries preserves synthetic_path" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{
            .idx = 0,
            .pc = 0,
            .file = "test.ora",
            .line = 1,
            .col = 1,
            .statement_id = 1,
            .origin_statement_id = 1,
            .is_statement = true,
            .is_synthetic = true,
            .synthetic_index = 1,
            .synthetic_count = 2,
            .synthetic_path = "0.2/1.2",
            .kind = .runtime,
        },
    };
    var sm = try SourceMap.fromEntries(allocator, &entries);
    defer sm.deinit();

    try std.testing.expectEqualStrings("0.2/1.2", sm.entries[0].synthetic_path.?);
}

test "SourceMap.getLineProvenance" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 10, .col = 1, .statement_id = 1, .origin_statement_id = 1, .is_statement = true, .is_synthetic = false, .kind = .runtime },
        .{ .idx = 1, .pc = 1, .file = "test.ora", .line = 11, .col = 1, .statement_id = 2, .origin_statement_id = 2, .is_statement = true, .is_synthetic = true, .kind = .runtime },
        .{ .idx = 2, .pc = 2, .file = "test.ora", .line = 12, .col = 1, .statement_id = 3, .origin_statement_id = 3, .is_statement = true, .is_synthetic = false, .kind = .runtime },
        .{ .idx = 3, .pc = 3, .file = "test.ora", .line = 12, .col = 5, .statement_id = 4, .origin_statement_id = 4, .is_statement = true, .is_synthetic = true, .kind = .runtime },
    };
    var sm = try SourceMap.fromEntries(allocator, &entries);
    defer sm.deinit();

    try std.testing.expectEqual(SourceMap.LineProvenance.direct, sm.getLineProvenance("test.ora", 10).?);
    try std.testing.expectEqual(SourceMap.LineProvenance.synthetic, sm.getLineProvenance("test.ora", 11).?);
    try std.testing.expectEqual(SourceMap.LineProvenance.mixed, sm.getLineProvenance("test.ora", 12).?);
    try std.testing.expect(sm.getLineProvenance("test.ora", 99) == null);
}
