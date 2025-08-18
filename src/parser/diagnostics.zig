const std = @import("std");

// Global switch to enable/disable parser diagnostics to stderr
pub var enable_stderr_diagnostics: bool = true;

pub inline fn print(comptime fmt: []const u8, args: anytype) void {
    if (enable_stderr_diagnostics) std.debug.print(fmt, args);
}
