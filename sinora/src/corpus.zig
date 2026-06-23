const std = @import("std");

const oracle = @import("oracle.zig");

pub const Summary = struct {
    total: usize = 0,
    expected: usize = 0,
    unexpected: usize = 0,
    failed: usize = 0,
    both_accept_codegen_pending: usize = 0,
    both_accept_bytecode_equal: usize = 0,
    bytecode_mismatch: usize = 0,
    zig_codegen_rejected: usize = 0,
    zig_rejected: usize = 0,
    rust_rejected: usize = 0,
    both_rejected: usize = 0,
    first_unexpected_path: ?[]const u8 = null,
    first_unexpected_classification: ?oracle.Classification = null,
    first_failed_path: ?[]const u8 = null,
    first_failed_error: ?[]const u8 = null,

    pub fn deinit(self: *Summary, allocator: std.mem.Allocator) void {
        if (self.first_unexpected_path) |path| allocator.free(path);
        if (self.first_failed_path) |path| allocator.free(path);
        if (self.first_failed_error) |message| allocator.free(message);
        self.* = undefined;
    }

    pub fn ok(self: Summary) bool {
        return self.unexpected == 0 and self.failed == 0;
    }

    fn addClassification(self: *Summary, classification: oracle.Classification) void {
        switch (classification) {
            .both_accept_codegen_pending => self.both_accept_codegen_pending += 1,
            .both_accept_bytecode_equal => self.both_accept_bytecode_equal += 1,
            .bytecode_mismatch => self.bytecode_mismatch += 1,
            .zig_codegen_rejected => self.zig_codegen_rejected += 1,
            .zig_rejected => self.zig_rejected += 1,
            .rust_rejected => self.rust_rejected += 1,
            .both_rejected => self.both_rejected += 1,
        }
    }

    fn noteUnexpected(
        self: *Summary,
        allocator: std.mem.Allocator,
        path: []const u8,
        classification: oracle.Classification,
    ) !void {
        self.unexpected += 1;
        if (self.first_unexpected_path == null) {
            self.first_unexpected_path = try allocator.dupe(u8, path);
            self.first_unexpected_classification = classification;
        }
    }

    fn noteFailure(self: *Summary, allocator: std.mem.Allocator, path: []const u8, err: anyerror) !void {
        self.failed += 1;
        if (self.first_failed_path == null) {
            self.first_failed_path = try allocator.dupe(u8, path);
            self.first_failed_error = try allocator.dupe(u8, @errorName(err));
        }
    }
};

pub const Record = struct {
    path: []const u8,
    classification: oracle.Classification,
    zig_accepted: bool,
    rust_accepted: bool,
    codegen_accepted: ?bool = null,
    bytecode_equal: ?bool = null,
    diagnostics: usize = 0,
};

pub const RecordFilter = enum {
    all,
    pending_only,
};

pub fn expectedLabel(mode: oracle.BackendMode) []const u8 {
    return switch (mode) {
        .debug => oracle.Classification.both_accept_bytecode_equal.label(),
        .release, .release_generic => "both-accept-bytecode-equal|both-accept-codegen-pending",
    };
}

pub fn isExpectedClassification(mode: oracle.BackendMode, classification: oracle.Classification) bool {
    return switch (mode) {
        .debug => classification == .both_accept_bytecode_equal,
        // For both release backends an honest codegen gap (pending) is allowed;
        // an emitted-but-wrong bytecode (mismatch) is not. This is what makes
        // the generic gate ratchet: it goes green only when zero fixtures emit
        // wrong bytes, which is the precondition for deleting the legacy path.
        .release, .release_generic => classification == .both_accept_bytecode_equal or classification == .both_accept_codegen_pending,
    };
}

pub fn compareDirectory(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
    rust_plank_path: []const u8,
    mode: oracle.BackendMode,
) !Summary {
    var sink: NoopRecordSink = .{};
    return compareDirectoryWithSink(allocator, io, root_path, rust_plank_path, mode, &sink);
}

pub fn compareDirectoryTextRecords(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
    rust_plank_path: []const u8,
    mode: oracle.BackendMode,
    filter: RecordFilter,
    writer: anytype,
) !Summary {
    var sink = TextRecordSink(@TypeOf(writer)){ .writer = writer, .filter = filter };
    return compareDirectoryWithSink(allocator, io, root_path, rust_plank_path, mode, &sink);
}

fn compareDirectoryWithSink(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
    rust_plank_path: []const u8,
    mode: oracle.BackendMode,
    sink: anytype,
) !Summary {
    var paths: std.ArrayList([]const u8) = .empty;
    defer {
        freeStringList(allocator, paths.items);
        paths.deinit(allocator);
    }
    try collectSirFiles(allocator, io, root_path, &paths);
    std.mem.sort([]const u8, paths.items, {}, lessThanString);

    var summary: Summary = .{};
    errdefer summary.deinit(allocator);

    for (paths.items) |path| {
        summary.total += 1;
        var comparison = oracle.compareSirFile(allocator, io, path, rust_plank_path, mode) catch |err| {
            try summary.noteFailure(allocator, path, err);
            continue;
        };
        defer {
            oracle.freeDiagnostics(allocator, comparison.zig.diagnostics);
            comparison.deinit(allocator);
        }

        summary.addClassification(comparison.classification);
        try sink.record(recordFromComparison(path, comparison, mode));
        if (isExpectedClassification(mode, comparison.classification)) {
            summary.expected += 1;
        } else {
            try summary.noteUnexpected(allocator, path, comparison.classification);
        }
    }

    return summary;
}

const NoopRecordSink = struct {
    fn record(_: *NoopRecordSink, _: Record) !void {}
};

fn TextRecordSink(comptime Writer: type) type {
    return struct {
        writer: Writer,
        filter: RecordFilter,

        fn record(self: *@This(), item: Record) !void {
            if (!recordMatchesFilter(item, self.filter)) return;
            try writeRecordText(self.writer, item);
        }
    };
}

fn recordMatchesFilter(record: Record, filter: RecordFilter) bool {
    return switch (filter) {
        .all => true,
        .pending_only => record.classification == .both_accept_codegen_pending,
    };
}

fn recordFromComparison(path: []const u8, comparison: oracle.Comparison, mode: oracle.BackendMode) Record {
    const bytecode = switch (mode) {
        .debug => comparison.debug_bytecode,
        .release, .release_generic => comparison.release_bytecode,
    };
    var record: Record = .{
        .path = path,
        .classification = comparison.classification,
        .zig_accepted = comparison.zig.accepted,
        .rust_accepted = if (comparison.rust) |rust| rust.accepted else false,
        .diagnostics = comparison.zig.diagnostics.len,
    };
    if (bytecode) |result| {
        record.codegen_accepted = result.zig_accepted;
        record.bytecode_equal = result.equal;
    }
    return record;
}

pub fn writeRecordText(writer: anytype, record: Record) !void {
    try writer.print(
        "{s}\tclassification={s}\tzig={}\trust={}\tcodegen=",
        .{ record.path, record.classification.label(), record.zig_accepted, record.rust_accepted },
    );
    if (record.codegen_accepted) |accepted| {
        try writer.print("{}", .{accepted});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll("\tbytecode_equal=");
    if (record.bytecode_equal) |equal| {
        try writer.print("{}", .{equal});
    } else {
        try writer.writeAll("null");
    }
    try writer.print("\tdiagnostics={d}\n", .{record.diagnostics});
}

pub fn writeSummaryText(
    writer: anytype,
    summary: Summary,
    mode: oracle.BackendMode,
    expected: []const u8,
) !void {
    if (summary.first_unexpected_path) |path| {
        try writer.print("first_unexpected_path={s}\n", .{path});
        try writer.print("first_unexpected_classification={s}\n", .{summary.first_unexpected_classification.?.label()});
    }
    if (summary.first_failed_path) |path| {
        try writer.print("first_failed_path={s}\n", .{path});
        try writer.print("first_failed_error={s}\n", .{summary.first_failed_error.?});
    }
    try writer.print("mode={s}\n", .{mode.label()});
    try writer.print("expected={s}\n", .{expected});
    try writer.print("total={d}\n", .{summary.total});
    try writer.print("expected_count={d}\n", .{summary.expected});
    try writer.print("unexpected_count={d}\n", .{summary.unexpected});
    try writer.print("failed={d}\n", .{summary.failed});
    try writer.print("both_accept_codegen_pending={d}\n", .{summary.both_accept_codegen_pending});
    try writer.print("both_accept_bytecode_equal={d}\n", .{summary.both_accept_bytecode_equal});
    try writer.print("bytecode_mismatch={d}\n", .{summary.bytecode_mismatch});
    try writer.print("zig_codegen_rejected={d}\n", .{summary.zig_codegen_rejected});
    try writer.print("zig_rejected={d}\n", .{summary.zig_rejected});
    try writer.print("rust_rejected={d}\n", .{summary.rust_rejected});
    try writer.print("both_rejected={d}\n", .{summary.both_rejected});
}

pub fn writeSummaryJson(
    writer: anytype,
    summary: Summary,
    mode: oracle.BackendMode,
    expected: []const u8,
) !void {
    try writer.print(
        \\{{"mode":"{s}","expected":"{s}","total":{d},"expected_count":{d},"unexpected_count":{d},"failed":{d},"classifications":{{"both_accept_codegen_pending":{d},"both_accept_bytecode_equal":{d},"bytecode_mismatch":{d},"zig_codegen_rejected":{d},"zig_rejected":{d},"rust_rejected":{d},"both_rejected":{d}}}
    , .{
        mode.label(),
        expected,
        summary.total,
        summary.expected,
        summary.unexpected,
        summary.failed,
        summary.both_accept_codegen_pending,
        summary.both_accept_bytecode_equal,
        summary.bytecode_mismatch,
        summary.zig_codegen_rejected,
        summary.zig_rejected,
        summary.rust_rejected,
        summary.both_rejected,
    });
    if (summary.first_unexpected_path) |path| {
        try writer.print(
            ",\"first_unexpected\":{{\"path\":\"{s}\",\"classification\":\"{s}\"}}",
            .{ path, summary.first_unexpected_classification.?.label() },
        );
    }
    if (summary.first_failed_path) |path| {
        try writer.print(
            ",\"first_failed\":{{\"path\":\"{s}\",\"error\":\"{s}\"}}",
            .{ path, summary.first_failed_error.? },
        );
    }
    try writer.writeAll("}\n");
}

fn collectSirFiles(allocator: std.mem.Allocator, io: std.Io, root_path: []const u8, out: *std.ArrayList([]const u8)) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".sir")) continue;
        const full_path = try std.fs.path.join(allocator, &.{ root_path, entry.path });
        try out.append(allocator, full_path);
    }
}

fn freeStringList(allocator: std.mem.Allocator, list: []const []const u8) void {
    for (list) |item| allocator.free(item);
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

test "expected corpus classification follows backend mode" {
    try std.testing.expectEqualStrings("both-accept-bytecode-equal", expectedLabel(.debug));
    try std.testing.expectEqualStrings("both-accept-bytecode-equal|both-accept-codegen-pending", expectedLabel(.release));
    try std.testing.expect(isExpectedClassification(.debug, .both_accept_bytecode_equal));
    try std.testing.expect(!isExpectedClassification(.debug, .both_accept_codegen_pending));
    try std.testing.expect(isExpectedClassification(.release, .both_accept_bytecode_equal));
    try std.testing.expect(isExpectedClassification(.release, .both_accept_codegen_pending));
    try std.testing.expect(!isExpectedClassification(.release, .bytecode_mismatch));
}

test "summary tracks classifications and first unexpected" {
    var summary: Summary = .{};
    defer summary.deinit(std.testing.allocator);

    summary.addClassification(.bytecode_mismatch);
    try summary.noteUnexpected(std.testing.allocator, "bad.sir", .bytecode_mismatch);
    try summary.noteUnexpected(std.testing.allocator, "second.sir", .zig_rejected);

    try std.testing.expect(!summary.ok());
    try std.testing.expectEqual(@as(usize, 1), summary.bytecode_mismatch);
    try std.testing.expectEqual(@as(usize, 2), summary.unexpected);
    try std.testing.expectEqualStrings("bad.sir", summary.first_unexpected_path.?);
    try std.testing.expectEqual(oracle.Classification.bytecode_mismatch, summary.first_unexpected_classification.?);
}

test "record text includes classification and codegen state" {
    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();

    try writeRecordText(&buffer.writer, .{
        .path = "case.sir",
        .classification = .both_accept_codegen_pending,
        .zig_accepted = true,
        .rust_accepted = true,
        .diagnostics = 0,
    });

    try std.testing.expectEqualStrings(
        "case.sir\tclassification=both-accept-codegen-pending\tzig=true\trust=true\tcodegen=null\tbytecode_equal=null\tdiagnostics=0\n",
        buffer.written(),
    );
}

test "record filter keeps only codegen pending records" {
    const pending: Record = .{
        .path = "pending.sir",
        .classification = .both_accept_codegen_pending,
        .zig_accepted = true,
        .rust_accepted = true,
    };
    const equal: Record = .{
        .path = "equal.sir",
        .classification = .both_accept_bytecode_equal,
        .zig_accepted = true,
        .rust_accepted = true,
    };

    try std.testing.expect(recordMatchesFilter(pending, .all));
    try std.testing.expect(recordMatchesFilter(equal, .all));
    try std.testing.expect(recordMatchesFilter(pending, .pending_only));
    try std.testing.expect(!recordMatchesFilter(equal, .pending_only));
}
