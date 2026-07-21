const std = @import("std");
const testing = std.testing;

const runner = @import("runner.zig");
const types = @import("types.zig");

const manifest_path = types.CONFORMANCE_DIR_REL ++ "/verified_build_manifest.tsv";
const schema = "verified-conformance-manifest-v2";

const ManifestEntry = struct {
    class: runner.VerifiedBuildClass,
    spec_name: []const u8,
    rejection_reason: ?runner.VerifiedRejectionReason,
};

const Manifest = struct {
    buffer: []u8,
    entries: []ManifestEntry,

    fn deinit(self: *Manifest, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.buffer);
        self.* = undefined;
    }
};

test "verified conformance manifest pins all three build classes" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var manifest = try loadManifest(allocator, manifest_path);
    defer manifest.deinit(allocator);

    const specs = try runner.collectSpecNames(allocator, types.CONFORMANCE_DIR_REL);
    defer runner.freeStringList(allocator, specs);
    try testing.expectEqual(specs.len, manifest.entries.len);

    var counts = [_]usize{ 0, 0, 0 };
    for (manifest.entries, specs) |entry, spec_name| {
        try testing.expectEqualStrings(spec_name, entry.spec_name);
        counts[@intFromEnum(entry.class)] += 1;

        const stem = spec_name[0 .. spec_name.len - ".spec.toml".len];
        const source_name = try std.fmt.allocPrint(allocator, "{s}.ora", .{stem});
        defer allocator.free(source_name);
        const source_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, source_name });
        defer allocator.free(source_path);
        const spec_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, spec_name });
        defer allocator.free(spec_path);

        const observed = try runner.classifyVerifiedBuild(allocator, source_path, spec_path);
        if (observed.class != entry.class or observed.rejection_reason != entry.rejection_reason) {
            std.debug.print("verified conformance drift: {s}: manifest={s}/{s}, observed={s}/{s}\n", .{
                spec_name,
                @tagName(entry.class),
                rejectionReasonName(entry.rejection_reason),
                @tagName(observed.class),
                rejectionReasonName(observed.rejection_reason),
            });
            return error.VerifiedConformanceClassDrift;
        }

        if (entry.class == .rewritten) {
            runner.runConformanceSpecWithExtraArgs(allocator, source_path, spec_path, &.{}) catch |err| {
                std.debug.print("verified rewritten conformance spec failed: {s}\n", .{spec_path});
                return err;
            };
        }
    }

    try testing.expectEqual(@as(usize, 59), counts[@intFromEnum(runner.VerifiedBuildClass.same_bytecode)]);
    try testing.expectEqual(@as(usize, 11), counts[@intFromEnum(runner.VerifiedBuildClass.rewritten)]);
    try testing.expectEqual(@as(usize, 24), counts[@intFromEnum(runner.VerifiedBuildClass.unverifiable)]);
}

test "resource rewrite executes with proved runtime checks kept" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const source_path = types.CONFORMANCE_DIR_REL ++ "/resource_basic_transfer.ora";
    const spec_path = types.CONFORMANCE_DIR_REL ++ "/resource_basic_transfer.spec.toml";
    try testing.expectEqual(
        runner.VerifiedBuildClass.rewritten,
        (try runner.classifyVerifiedBuild(testing.allocator, source_path, spec_path)).class,
    );
    try runner.runConformanceSpecWithExtraArgs(testing.allocator, source_path, spec_path, &.{"--keep-proved-checks"});
}

test "checked shift bounds execute in verified and no-verify builds" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const source_path = types.CONFORMANCE_DIR_REL ++ "/checked_shift_verified.ora";
    const spec_path = types.CONFORMANCE_DIR_REL ++ "/checked_shift_verified.spec.toml";
    try runner.runConformanceSpecWithExtraArgs(testing.allocator, source_path, spec_path, &.{"--no-verify"});
    try runner.runConformanceSpecWithExtraArgs(testing.allocator, source_path, spec_path, &.{});
}

test "verified conformance manifest parser rejects unordered membership" {
    const text =
        schema ++ "\n" ++
        "same-bytecode\tz.spec.toml\t-\n" ++
        "rewritten\ta.spec.toml\t-\n";
    try testing.expectError(error.ManifestNotSorted, parseManifestEntries(testing.allocator, text));
}

test "verified conformance manifest requires class-C reasons" {
    const missing_reason = schema ++ "\nunverifiable\ta.spec.toml\t-\n";
    try testing.expectError(error.MissingUnverifiableReason, parseManifestEntries(testing.allocator, missing_reason));

    const reason_on_verified = schema ++ "\nsame-bytecode\ta.spec.toml\tcontract_invariant_unproved\n";
    try testing.expectError(error.UnexpectedVerifiedReason, parseManifestEntries(testing.allocator, reason_on_verified));
}

fn loadManifest(allocator: std.mem.Allocator, path: []const u8) !Manifest {
    const buffer = try std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), path, allocator, std.Io.Limit.limited(1024 * 1024));
    errdefer allocator.free(buffer);
    return .{
        .buffer = buffer,
        .entries = try parseManifestEntries(allocator, buffer),
    };
}

fn parseManifestEntries(allocator: std.mem.Allocator, text: []const u8) ![]ManifestEntry {
    var entries: std.ArrayList(ManifestEntry) = .empty;
    errdefer entries.deinit(allocator);

    var lines = std.mem.splitScalar(u8, text, '\n');
    const first = lines.next() orelse return error.MissingManifestSchema;
    if (!std.mem.eql(u8, first, schema)) return error.InvalidManifestSchema;

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const class_text = fields.next() orelse return error.InvalidManifestEntry;
        const spec_name = fields.next() orelse return error.InvalidManifestEntry;
        const reason_text = fields.next() orelse return error.InvalidManifestEntry;
        if (fields.next() != null) return error.InvalidManifestEntry;
        if (!std.mem.endsWith(u8, spec_name, ".spec.toml")) return error.InvalidManifestEntry;

        const class: runner.VerifiedBuildClass = if (std.mem.eql(u8, class_text, "same-bytecode"))
            .same_bytecode
        else if (std.mem.eql(u8, class_text, "rewritten"))
            .rewritten
        else if (std.mem.eql(u8, class_text, "unverifiable"))
            .unverifiable
        else
            return error.InvalidManifestClass;

        const rejection_reason: ?runner.VerifiedRejectionReason = if (class == .unverifiable)
            try parseRejectionReason(reason_text)
        else blk: {
            if (!std.mem.eql(u8, reason_text, "-")) return error.UnexpectedVerifiedReason;
            break :blk null;
        };

        if (entries.items.len != 0 and !std.mem.lessThan(u8, entries.items[entries.items.len - 1].spec_name, spec_name)) {
            return error.ManifestNotSorted;
        }
        try entries.append(allocator, .{
            .class = class,
            .spec_name = spec_name,
            .rejection_reason = rejection_reason,
        });
    }
    if (entries.items.len == 0) return error.EmptyManifest;
    return entries.toOwnedSlice(allocator);
}

fn parseRejectionReason(text: []const u8) !runner.VerifiedRejectionReason {
    if (std.mem.eql(u8, text, "-")) return error.MissingUnverifiableReason;
    inline for (std.meta.fields(runner.VerifiedRejectionReason)) |field| {
        if (std.mem.eql(u8, text, field.name)) return @enumFromInt(field.value);
    }
    return error.InvalidUnverifiableReason;
}

fn rejectionReasonName(reason: ?runner.VerifiedRejectionReason) []const u8 {
    return if (reason) |value| @tagName(value) else "-";
}
