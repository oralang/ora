const std = @import("std");

const Allocator = std.mem.Allocator;

pub const PendingFullDiagnostic = struct {
    version: i32,
    generation: u64,
};

pub const Debouncer = struct {
    allocator: Allocator,
    pending: std.StringHashMap(PendingFullDiagnostic),
    scheduled: usize = 0,
    canceled: usize = 0,
    flushed: usize = 0,
    cleared: usize = 0,

    pub fn init(allocator: Allocator) Debouncer {
        return .{
            .allocator = allocator,
            .pending = std.StringHashMap(PendingFullDiagnostic).init(allocator),
        };
    }

    pub fn deinit(self: *Debouncer) void {
        var it = self.pending.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.pending.deinit();
        self.* = undefined;
    }

    pub fn schedule(self: *Debouncer, uri: []const u8, pending: PendingFullDiagnostic) !void {
        if (self.pending.getPtr(uri)) |existing| {
            if (existing.version != pending.version or existing.generation != pending.generation) {
                self.canceled = addSat(self.canceled, 1);
            }
            existing.* = pending;
            self.scheduled = addSat(self.scheduled, 1);
            return;
        }

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);
        try self.pending.put(uri_copy, pending);
        self.scheduled = addSat(self.scheduled, 1);
    }

    pub fn flush(self: *Debouncer, uri: []const u8) bool {
        if (self.take(uri)) |_| {
            self.flushed = addSat(self.flushed, 1);
            return true;
        }
        return false;
    }

    pub fn clear(self: *Debouncer, uri: []const u8) bool {
        if (self.take(uri)) |_| {
            self.cleared = addSat(self.cleared, 1);
            return true;
        }
        return false;
    }

    pub fn pendingCount(self: *Debouncer) usize {
        return self.pending.count();
    }

    fn take(self: *Debouncer, uri: []const u8) ?PendingFullDiagnostic {
        const removed = self.pending.fetchRemove(uri) orelse return null;
        self.allocator.free(removed.key);
        return removed.value;
    }
};

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

test "diagnostic debouncer coalesces pending full diagnostics by uri" {
    var debouncer = Debouncer.init(std.testing.allocator);
    defer debouncer.deinit();

    try debouncer.schedule("file:///contract.ora", .{ .version = 1, .generation = 1 });
    try std.testing.expectEqual(@as(usize, 1), debouncer.pendingCount());
    try std.testing.expectEqual(@as(usize, 1), debouncer.scheduled);
    try std.testing.expectEqual(@as(usize, 0), debouncer.canceled);

    try debouncer.schedule("file:///contract.ora", .{ .version = 2, .generation = 2 });
    try std.testing.expectEqual(@as(usize, 1), debouncer.pendingCount());
    try std.testing.expectEqual(@as(usize, 2), debouncer.scheduled);
    try std.testing.expectEqual(@as(usize, 1), debouncer.canceled);

    try std.testing.expect(debouncer.flush("file:///contract.ora"));
    try std.testing.expectEqual(@as(usize, 0), debouncer.pendingCount());
    try std.testing.expectEqual(@as(usize, 1), debouncer.flushed);
}
