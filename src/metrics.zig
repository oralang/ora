//! Compiler Metrics & Timing
//!
//! Lightweight instrumentation for measuring each compilation phase.
//! Inspired by Clang's -ftime-report and Rust's -Z time-passes.
//!
//! Usage:
//!   var m = Metrics.init();
//!   m.begin("lexing");
//!   // ... do work ...
//!   m.end();
//!   m.report(stdout);

const std = @import("std");

pub const Phase = struct {
    name: []const u8,
    wall_ns: u64,
};

pub const Metrics = struct {
    phases: [max_phases]Phase = undefined,
    count: usize = 0,
    timer: std.time.Timer = undefined,
    total_timer: std.time.Timer = undefined,
    active: bool = false,
    enabled: bool,

    const max_phases = 32;

    pub fn init(enabled: bool) Metrics {
        var m = Metrics{ .enabled = enabled };
        if (enabled) {
            m.total_timer = std.time.Timer.start() catch unreachable;
        }
        return m;
    }

    /// Start timing a named phase.
    pub fn begin(self: *Metrics, name: []const u8) void {
        if (!self.enabled) return;
        std.debug.assert(!self.active);
        self.active = true;
        self.timer = std.time.Timer.start() catch unreachable;
        if (self.count < max_phases) {
            self.phases[self.count] = .{ .name = name, .wall_ns = 0 };
        }
    }

    /// End timing the current phase.
    pub fn end(self: *Metrics) void {
        if (!self.enabled) return;
        std.debug.assert(self.active);
        self.active = false;
        const elapsed = self.timer.read();
        if (self.count < max_phases) {
            self.phases[self.count].wall_ns = elapsed;
            self.count += 1;
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

        const total_ns = self.total_timer.read();
        const total_ms = floatMs(total_ns);

        try writer.print("\n===-------------------------------------------------------------------------===\n", .{});
        try writer.print("                         Ora Compiler Metrics\n", .{});
        try writer.print("===-------------------------------------------------------------------------===\n", .{});
        try writer.print("  {s:<30}  {s:>10}  {s:>6}\n", .{ "Phase", "Time", "%" });
        try writer.print("  {s:->30}  {s:->10}  {s:->6}\n", .{ "", "", "" });

        for (self.phases[0..self.count]) |phase| {
            const ms = floatMs(phase.wall_ns);
            const pct = if (total_ns > 0) @as(f64, @floatFromInt(phase.wall_ns)) / @as(f64, @floatFromInt(total_ns)) * 100.0 else 0.0;
            try writer.print("  {s:<30}  {d:>8.3}ms  {d:>5.1}%\n", .{ phase.name, ms, pct });
        }

        try writer.print("  {s:->30}  {s:->10}  {s:->6}\n", .{ "", "", "" });
        try writer.print("  {s:<30}  {d:>8.3}ms\n", .{ "Total", total_ms });
        try writer.print("===-------------------------------------------------------------------------===\n", .{});
    }

    fn floatMs(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
    }
};
