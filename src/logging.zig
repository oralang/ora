// ============================================================================
// Logging Utilities
// ============================================================================
//
// Centralized logging helpers for user-facing diagnostics and debug traces.
// Debug output is gated behind a runtime flag.
//
// ============================================================================

const std = @import("std");

var debug_enabled: bool = false;

pub fn setDebugEnabled(enabled: bool) void {
    debug_enabled = enabled;
}

pub fn isDebugEnabled() bool {
    return debug_enabled;
}

fn emit(prefix: []const u8, comptime fmt: []const u8, args: anytype) void {
    if (prefix.len > 0) {
        std.debug.print("{s}: ", .{prefix});
    }
    std.debug.print(fmt, args);
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (!debug_enabled) return;
    emit("debug", fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    emit("info", fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    emit("warning", fmt, args);
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    emit("error", fmt, args);
}

pub fn help(comptime fmt: []const u8, args: anytype) void {
    emit("help", fmt, args);
}
