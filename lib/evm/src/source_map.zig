/// Source map for mapping EVM bytecode PCs to source file locations.
/// Used by the debugger to provide source-level stepping.
const std = @import("std");

pub const SourceMap = struct {
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
        /// 1-based line number in emitted textual .sir, when available.
        sir_line: ?u32 = null,
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
                .sir_line = je.sir_line,
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
                .sir_line = e.sir_line,
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
        sir_line: ?u32 = null,
        stmt: bool,
        kind: ?[]const u8 = null,
    };
};

test "SourceMap.getEntry binary search" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .sir_line = 10, .is_statement = true, .kind = .runtime },
        .{ .idx = 1, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .sir_line = 11, .is_statement = true, .kind = .runtime },
        .{ .idx = 2, .pc = 12, .file = "test.ora", .line = 3, .col = 1, .sir_line = 12, .is_statement = true, .kind = .runtime },
        .{ .idx = 3, .pc = 20, .file = "test.ora", .line = 4, .col = 1, .sir_line = 13, .is_statement = true, .kind = .runtime },
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
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .sir_line = 4, .is_statement = true, .kind = .runtime },
        .{ .idx = 1, .pc = 3, .file = "test.ora", .line = 1, .col = 5, .sir_line = 5, .is_statement = false, .kind = null },
        .{ .idx = 2, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .sir_line = 6, .is_statement = true, .kind = .runtime },
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
        \\  {"idx":7,"pc":0,"file":"main.ora","line":3,"col":5,"sir_line":17,"stmt":true,"kind":"runtime_guard"},
        \\  {"idx":8,"pc":6,"file":"main.ora","line":4,"col":5,"sir_line":18,"stmt":true,"kind":"runtime"}
        \\]}
    ;
    var sm = try SourceMap.loadFromJson(allocator, json);
    defer sm.deinit();

    try std.testing.expectEqual(@as(usize, 2), sm.entries.len);
    try std.testing.expectEqual(@as(u32, 0), sm.entries[0].pc);
    try std.testing.expectEqual(@as(?u32, 7), sm.entries[0].idx);
    try std.testing.expectEqual(@as(u32, 3), sm.entries[0].line);
    try std.testing.expectEqual(@as(?u32, 17), sm.entries[0].sir_line);
    try std.testing.expect(sm.entries[0].is_statement);
    try std.testing.expectEqual(SourceMap.StatementKind.runtime_guard, sm.entries[0].kind.?);
    try std.testing.expectEqualStrings("main.ora", sm.entries[0].file);
}
