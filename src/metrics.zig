//! Compiler Metrics & Timing
//!
//! Lightweight instrumentation for measuring each compilation phase.
//! Inspired by Clang's -ftime-report and Rust's -Z time-passes.
//!
//! Usage:
//!   var m = Metrics.init(true);
//!   m.begin("lexing");
//!   // ... do work ...
//!   m.endWith(123);
//!   m.report(stdout);

const std = @import("std");
const allocation_stats = @import("lsp/allocation_stats.zig");

pub const Phase = struct {
    name: []const u8,
    wall_ns: u64 = 0,
    invocations: u64 = 0,
    work_count: u64 = 0,
    alloc_calls: u64 = 0,
    bytes_allocated: u64 = 0,
};

pub const Metrics = struct {
    phases: [max_phases]Phase = undefined,
    count: usize = 0,
    timer: std.Io.Clock.Timestamp = undefined,
    total_timer: std.Io.Clock.Timestamp = undefined,
    active: bool = false,
    active_name: []const u8 = "",
    alloc_stats: ?*const allocation_stats.Stats = null,
    begin_alloc_calls: usize = 0,
    begin_bytes_allocated: usize = 0,
    enabled: bool,

    const max_phases = 64;

    pub fn init(enabled: bool) Metrics {
        var m = Metrics{ .enabled = enabled };
        if (enabled) {
            m.total_timer = nowTimestamp();
        }
        return m;
    }

    pub fn setAllocationStats(self: *Metrics, stats: *const allocation_stats.Stats) void {
        self.alloc_stats = stats;
    }

    /// Start timing a named phase.
    pub fn begin(self: *Metrics, name: []const u8) void {
        if (!self.enabled) return;
        std.debug.assert(!self.active);
        self.active = true;
        self.active_name = name;
        if (self.alloc_stats) |stats| {
            self.begin_alloc_calls = stats.alloc_calls;
            self.begin_bytes_allocated = stats.bytes_allocated;
        }
        self.timer = nowTimestamp();
    }

    /// End timing the current phase.
    pub fn end(self: *Metrics) void {
        self.endWith(0);
    }

    /// End timing the current phase and add its structural work count.
    pub fn endWith(self: *Metrics, work_count: u64) void {
        if (!self.enabled) return;
        std.debug.assert(self.active);
        self.active = false;
        const elapsed = elapsedNs(self.timer);
        const index = self.phaseIndex(self.active_name) orelse blk: {
            if (self.count >= max_phases) return;
            const next = self.count;
            self.phases[next] = .{ .name = self.active_name };
            self.count += 1;
            break :blk next;
        };
        self.phases[index].wall_ns = addSat(self.phases[index].wall_ns, elapsed);
        self.phases[index].work_count = addSat(self.phases[index].work_count, work_count);
        self.phases[index].invocations = addSat(self.phases[index].invocations, 1);
        if (self.alloc_stats) |stats| {
            self.phases[index].alloc_calls = addSat(self.phases[index].alloc_calls, deltaU64(stats.alloc_calls, self.begin_alloc_calls));
            self.phases[index].bytes_allocated = addSat(self.phases[index].bytes_allocated, deltaU64(stats.bytes_allocated, self.begin_bytes_allocated));
        }
    }

    /// Add a deterministic, non-timed counter to the same metrics table. This
    /// is used for optimizer/pass counters that are already collected by the
    /// lower layer and should be reported beside phase timings.
    pub fn addCounter(self: *Metrics, name: []const u8, value: u64) void {
        if (!self.enabled) return;
        const index = self.phaseIndex(name) orelse blk: {
            if (self.count >= max_phases) return;
            const next = self.count;
            self.phases[next] = .{ .name = name };
            self.count += 1;
            break :blk next;
        };
        self.phases[index].work_count = addSat(self.phases[index].work_count, value);
        self.phases[index].invocations = addSat(self.phases[index].invocations, 1);
    }

    /// Drop an active phase after an error without recording partial data.
    pub fn cancel(self: *Metrics) void {
        if (!self.enabled) return;
        if (self.active) {
            self.active = false;
            self.active_name = "";
            self.begin_alloc_calls = 0;
            self.begin_bytes_allocated = 0;
        }
    }

    /// Convenience: time a phase given as a closure.
    pub fn measure(self: *Metrics, name: []const u8, comptime func: anytype, args: anytype) @TypeOf(@call(.auto, func, args)) {
        self.begin(name);
        const result = @call(.auto, func, args);
        self.end();
        return result;
    }

    /// Print a table like Clang's -ftime-report.
    pub fn report(self: *Metrics, writer: anytype) !void {
        if (!self.enabled or self.count == 0) return;

        const total_ns = elapsedNs(self.total_timer);
        const total_ms = floatMs(total_ns);

        try writer.print("\n===-------------------------------------------------------------------------===\n", .{});
        try writer.print("                         Ora Compiler Metrics\n", .{});
        try writer.print("===-------------------------------------------------------------------------===\n", .{});
        try writer.print("  {s:<22}  {s:>10}  {s:>6}  {s:>6}  {s:>10}  {s:>12}  {s:>8}  {s:>10}  {s:>12}  {s:>12}\n", .{ "Phase", "Time", "%", "Runs", "Work", "ns/work", "Allocs", "A/work", "Bytes", "B/work" });
        try writer.print("  {s:->22}  {s:->10}  {s:->6}  {s:->6}  {s:->10}  {s:->12}  {s:->8}  {s:->10}  {s:->12}  {s:->12}\n", .{ "", "", "", "", "", "", "", "", "", "" });

        for (self.phases[0..self.count]) |phase| {
            const ms = floatMs(phase.wall_ns);
            const pct = if (total_ns > 0) @as(f64, @floatFromInt(phase.wall_ns)) / @as(f64, @floatFromInt(total_ns)) * 100.0 else 0.0;
            const ns_per_work = if (phase.work_count > 0) @as(f64, @floatFromInt(phase.wall_ns)) / @as(f64, @floatFromInt(phase.work_count)) else 0.0;
            const allocs_per_work = if (phase.work_count > 0) @as(f64, @floatFromInt(phase.alloc_calls)) / @as(f64, @floatFromInt(phase.work_count)) else 0.0;
            const bytes_per_work = if (phase.work_count > 0) @as(f64, @floatFromInt(phase.bytes_allocated)) / @as(f64, @floatFromInt(phase.work_count)) else 0.0;
            try writer.print(
                "  {s:<22}  {d:>8.3}ms  {d:>5.1}%  {d:>6}  {d:>10}  {d:>12.1}  {d:>8}  {d:>10.3}  {d:>12}  {d:>12.1}\n",
                .{ phase.name, ms, pct, phase.invocations, phase.work_count, ns_per_work, phase.alloc_calls, allocs_per_work, phase.bytes_allocated, bytes_per_work },
            );
        }

        try writer.print("  {s:->22}  {s:->10}  {s:->6}  {s:->6}  {s:->10}  {s:->12}  {s:->8}  {s:->10}  {s:->12}  {s:->12}\n", .{ "", "", "", "", "", "", "", "", "", "" });
        try writer.print("  {s:<22}  {d:>8.3}ms\n", .{ "Total", total_ms });
        try writer.print("===-------------------------------------------------------------------------===\n", .{});
    }

    pub fn reportJson(self: *Metrics, writer: anytype) !void {
        if (!self.enabled) {
            try writer.writeAll("{\"enabled\":false,\"phases\":[]}\n");
            return;
        }

        const total_ns = self.total_timer.read();
        try writer.print("{{\"enabled\":true,\"total_wall_ns\":{d},\"phases\":[", .{total_ns});
        for (self.phases[0..self.count], 0..) |phase, index| {
            if (index != 0) try writer.writeByte(',');
            try writer.writeAll("{\"name\":");
            try writeJsonString(writer, phase.name);
            try writer.print(
                ",\"wall_ns\":{d},\"invocations\":{d},\"work_count\":{d},\"alloc_calls\":{d},\"bytes_allocated\":{d}}}",
                .{ phase.wall_ns, phase.invocations, phase.work_count, phase.alloc_calls, phase.bytes_allocated },
            );
        }
        try writer.writeAll("]}\n");
    }

    fn phaseIndex(self: *const Metrics, name: []const u8) ?usize {
        for (self.phases[0..self.count], 0..) |phase, index| {
            if (std.mem.eql(u8, phase.name, name)) return index;
        }
        return null;
    }

    fn floatMs(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
    }
};

fn nowTimestamp() std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .awake);
}

fn elapsedNs(start: std.Io.Clock.Timestamp) u64 {
    const end = nowTimestamp();
    const elapsed = std.Io.Clock.Timestamp.durationTo(start, end);
    if (elapsed.raw.nanoseconds <= 0) return 0;
    return std.math.cast(u64, elapsed.raw.nanoseconds) orelse std.math.maxInt(u64);
}

fn addSat(a: u64, b: u64) u64 {
    return std.math.add(u64, a, b) catch std.math.maxInt(u64);
}

fn deltaU64(current: usize, previous: usize) u64 {
    if (current <= previous) return 0;
    return std.math.cast(u64, current - previous) orelse std.math.maxInt(u64);
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (byte < 0x20) {
                    try writer.print("\\u{x:0>4}", .{byte});
                } else {
                    try writer.writeByte(byte);
                }
            },
        }
    }
    try writer.writeByte('"');
}

test "metrics aggregates repeated phase names" {
    var metrics = Metrics.init(true);
    metrics.begin("syntax");
    metrics.endWith(2);
    metrics.begin("syntax");
    metrics.endWith(3);
    metrics.begin("hir-lower");
    metrics.endWith(1);

    try std.testing.expectEqual(@as(usize, 2), metrics.count);
    try std.testing.expectEqualStrings("syntax", metrics.phases[0].name);
    try std.testing.expectEqual(@as(u64, 2), metrics.phases[0].invocations);
    try std.testing.expectEqual(@as(u64, 5), metrics.phases[0].work_count);
    try std.testing.expectEqualStrings("hir-lower", metrics.phases[1].name);
    try std.testing.expectEqual(@as(u64, 1), metrics.phases[1].invocations);
}

test "metrics aggregates deterministic counters" {
    var metrics = Metrics.init(true);
    metrics.addCounter("mlir.inline", 3);
    metrics.addCounter("mlir.inline", 4);
    metrics.addCounter("mlir.const-dedup", 2);

    try std.testing.expectEqual(@as(usize, 2), metrics.count);
    try std.testing.expectEqualStrings("mlir.inline", metrics.phases[0].name);
    try std.testing.expectEqual(@as(u64, 2), metrics.phases[0].invocations);
    try std.testing.expectEqual(@as(u64, 7), metrics.phases[0].work_count);
    try std.testing.expectEqualStrings("mlir.const-dedup", metrics.phases[1].name);
    try std.testing.expectEqual(@as(u64, 1), metrics.phases[1].invocations);
    try std.testing.expectEqual(@as(u64, 2), metrics.phases[1].work_count);
}

test "metrics records allocation deltas for active phases" {
    var stats: allocation_stats.Stats = .{};
    var metrics = Metrics.init(true);
    metrics.setAllocationStats(&stats);

    metrics.begin("resolve");
    stats.alloc_calls += 2;
    stats.bytes_allocated += 128;
    metrics.endWith(4);

    metrics.begin("resolve");
    stats.alloc_calls += 3;
    stats.bytes_allocated += 64;
    metrics.endWith(2);

    try std.testing.expectEqual(@as(usize, 1), metrics.count);
    try std.testing.expectEqual(@as(u64, 5), metrics.phases[0].alloc_calls);
    try std.testing.expectEqual(@as(u64, 192), metrics.phases[0].bytes_allocated);
    try std.testing.expectEqual(@as(u64, 6), metrics.phases[0].work_count);
}

test "metrics writes machine-readable JSON report" {
    var metrics = Metrics.init(true);
    metrics.begin("ast-lower");
    metrics.endWith(7);

    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(std.testing.allocator);
    try metrics.reportJson(json.writer(std.testing.allocator));

    try std.testing.expect(std.mem.containsAtLeast(u8, json.items, 1, "\"enabled\":true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, json.items, 1, "\"name\":\"ast-lower\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, json.items, 1, "\"work_count\":7"));
}
