const std = @import("std");

pub const Severity = enum {
    warning,
    err,

    fn label(self: Severity) []const u8 {
        return switch (self) {
            .warning => "warning",
            .err => "error",
        };
    }
};

pub const Diagnostic = struct {
    // 1-based source location in the SIR text. Sinora diagnostics intentionally
    // stay line-oriented because the parser is line-oriented and the Plank SIR
    // oracle reports line-level failures too.
    severity: Severity,
    line: u32,
    column: u32,
    // Owned by the diagnostic bag that produced it. Snapshots copied out of a
    // bag (see oracle.zig) duplicate this slice and must free it separately.
    message: []const u8,
};

pub const Bag = struct {
    // The bag owns every diagnostic message. The caller owns the bag itself and
    // controls when all messages are freed via deinit().
    allocator: std.mem.Allocator,
    items: std.ArrayList(Diagnostic) = .empty,
    // Maintained on append so hasErrors() is O(1). Parser/legalizer/codegen
    // paths call hasErrors() often at phase boundaries and after speculative
    // unsupported-shape probes.
    error_count: usize = 0,

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
        // Format once and store the owned message. Diagnostics must survive
        // parser arena teardown and may be cloned into corpus comparison records.
        const message = try std.fmt.allocPrint(self.allocator, fmt, args);
        errdefer self.allocator.free(message);
        try self.items.append(self.allocator, .{
            .severity = severity,
            .line = line,
            .column = column,
            .message = message,
        });
        if (severity == .err) self.error_count += 1;
    }

    pub fn errorAt(self: *Bag, line: u32, column: u32, comptime fmt: []const u8, args: anytype) !void {
        try self.add(.err, line, column, fmt, args);
    }

    pub fn warningAt(self: *Bag, line: u32, column: u32, comptime fmt: []const u8, args: anytype) !void {
        try self.add(.warning, line, column, fmt, args);
    }

    pub fn hasErrors(self: *const Bag) bool {
        return self.error_count != 0;
    }

    pub fn writeTo(self: *const Bag, writer: anytype) !void {
        for (self.items.items) |item| {
            try writer.print("{d}:{d}: {s}: {s}\n", .{ item.line, item.column, item.severity.label(), item.message });
        }
    }
};

test "diagnostic bag tracks errors without scanning warnings" {
    var bag = Bag.init(std.testing.allocator);
    defer bag.deinit();

    try bag.warningAt(4, 2, "ignored shape {s}", .{"x"});
    try std.testing.expect(!bag.hasErrors());
    try std.testing.expectEqual(@as(usize, 0), bag.error_count);

    try bag.errorAt(5, 9, "bad operand {d}", .{7});
    try std.testing.expect(bag.hasErrors());
    try std.testing.expectEqual(@as(usize, 1), bag.error_count);
    try std.testing.expectEqual(@as(usize, 2), bag.items.items.len);
}

test "diagnostic bag renders stable line-oriented output" {
    var bag = Bag.init(std.testing.allocator);
    defer bag.deinit();

    try bag.warningAt(1, 3, "heads up", .{});
    try bag.errorAt(2, 5, "missing block '{s}'", .{"join"});

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try bag.writeTo(&output.writer);

    try std.testing.expectEqualStrings(
        "1:3: warning: heads up\n2:5: error: missing block 'join'\n",
        output.written(),
    );
}
