const std = @import("std");

pub const Severity = enum {
    warning,
    err,
};

pub const Diagnostic = struct {
    severity: Severity,
    line: u32,
    column: u32,
    message: []const u8,
    hasError: bool = false,
};

pub const Bag = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(Diagnostic) = .empty,

    pub fn init(allocator: std.mem.Allocator) Bag {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Bag) void {
        for (self.items.items) |item| {
            self.allocator.free(item.message);
        }
        self.items.deinit(self.allocator);
    }

    pub fn add(
        self: *Bag,
        severity: Severity,
        line: u32,
        column: u32,
        comptime fmt: []const u8,
        args: anytype,
    ) !void {
        const message = try std.fmt.allocPrint(self.allocator, fmt, args);
        try self.items.append(self.allocator, .{
            .severity = severity,
            .line = line,
            .column = column,
            .message = message,
        });
    }

    pub fn errorAt(self: *Bag, line: u32, column: u32, comptime fmt: []const u8, args: anytype) !void {
        try self.add(.err, line, column, fmt, args);
    }

    pub fn warningAt(self: *Bag, line: u32, column: u32, comptime fmt: []const u8, args: anytype) !void {
        try self.add(.warning, line, column, fmt, args);
    }

    pub fn hasErrors(self: *const Bag) bool {
        for (self.items.items) |item| {
            if (item.severity == .err) return true;
        }
        return false;
    }

    pub fn writeTo(self: *const Bag, writer: anytype) !void {
        for (self.items.items) |item| {
            const tag = switch (item.severity) {
                .warning => "warning",
                .err => "error",
            };
            try writer.print("{d}:{d}: {s}: {s}\n", .{ item.line, item.column, tag, item.message });
        }
    }
};
