/// Source map for mapping EVM bytecode PCs to source file locations.
/// Used by the debugger to provide source-level stepping.
const std = @import("std");

pub const SourceMap = struct {
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
        /// Whether this PC is the start of a source-level statement.
        /// The debugger stops here during step-over (not on every opcode).
        is_statement: bool,
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
                .is_statement = je.stmt,
            };
        }

        return .{
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
                .is_statement = e.is_statement,
            };
        }
        return .{
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
        stmt: bool,
    };
};

test "SourceMap.getEntry binary search" {
    const allocator = std.testing.allocator;
    const entries = [_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .is_statement = true },
        .{ .idx = 1, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .is_statement = true },
        .{ .idx = 2, .pc = 12, .file = "test.ora", .line = 3, .col = 1, .is_statement = true },
        .{ .idx = 3, .pc = 20, .file = "test.ora", .line = 4, .col = 1, .is_statement = true },
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
        .{ .idx = 0, .pc = 0, .file = "test.ora", .line = 1, .col = 1, .is_statement = true },
        .{ .idx = 1, .pc = 3, .file = "test.ora", .line = 1, .col = 5, .is_statement = false },
        .{ .idx = 2, .pc = 5, .file = "test.ora", .line = 2, .col = 1, .is_statement = true },
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
        \\  {"idx":7,"pc":0,"file":"main.ora","line":3,"col":5,"stmt":true},
        \\  {"idx":8,"pc":6,"file":"main.ora","line":4,"col":5,"stmt":true}
        \\]}
    ;
    var sm = try SourceMap.loadFromJson(allocator, json);
    defer sm.deinit();

    try std.testing.expectEqual(@as(usize, 2), sm.entries.len);
    try std.testing.expectEqual(@as(u32, 0), sm.entries[0].pc);
    try std.testing.expectEqual(@as(?u32, 7), sm.entries[0].idx);
    try std.testing.expectEqual(@as(u32, 3), sm.entries[0].line);
    try std.testing.expect(sm.entries[0].is_statement);
    try std.testing.expectEqualStrings("main.ora", sm.entries[0].file);
}
