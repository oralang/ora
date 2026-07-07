//! Measurement report for canonical-Z3/live-Z3 parity.
//!
//! This is diagnostic-only. It compares the canonical adapter against the live
//! prepared-query hashes already overlaid onto a formal obligation set, without
//! changing proof acceptance, guard erasure, or artifact policy.

const std = @import("std");
const obligation = @import("obligation.zig");
const obligation_to_z3 = @import("obligation_to_z3.zig");
const z3_verification = @import("ora_z3_verification");

const shape_count = std.meta.fields(obligation_to_z3.CanonicalPromotionShape).len;
const unpromoted_index = shape_count;

const Bucket = struct {
    total: u32 = 0,
    live_rows: u32 = 0,
    matched: u32 = 0,
    mismatched: u32 = 0,
    unavailable: u32 = 0,
    no_live_row: u32 = 0,
};

const Summary = struct {
    queries: u32 = 0,
    promoted: u32 = 0,
    live_rows: u32 = 0,
    matched: u32 = 0,
    mismatched: u32 = 0,
    unavailable: u32 = 0,
    no_live_row: u32 = 0,
    required: u32 = 0,
};

const UnsupportedReasonCounts = [std.meta.fields(obligation_to_z3.CanonicalUnsupportedReason).len]u32;

const HashErrorCount = struct {
    name: []const u8,
    count: u32,
};

const HashErrorCounts = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(HashErrorCount) = .empty,

    fn deinit(self: *HashErrorCounts) void {
        for (self.items.items) |item| self.allocator.free(item.name);
        self.items.deinit(self.allocator);
        self.* = undefined;
    }

    fn add(self: *HashErrorCounts, name: []const u8) !void {
        for (self.items.items) |*item| {
            if (std.mem.eql(u8, item.name, name)) {
                item.count +|= 1;
                return;
            }
        }
        try self.items.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, name),
            .count = 1,
        });
    }

    fn sort(self: *HashErrorCounts) void {
        std.mem.sort(HashErrorCount, self.items.items, {}, struct {
            fn lessThan(_: void, lhs: HashErrorCount, rhs: HashErrorCount) bool {
                return std.mem.lessThan(u8, lhs.name, rhs.name);
            }
        }.lessThan);
    }
};

fn shapeName(shape: ?obligation_to_z3.CanonicalPromotionShape) []const u8 {
    return if (shape) |value| @tagName(value) else "unpromoted";
}

fn bucketIndex(shape: ?obligation_to_z3.CanonicalPromotionShape) usize {
    return if (shape) |value| @intFromEnum(value) else unpromoted_index;
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
            else => if (byte < 0x20) {
                try writer.print("\\u{x:0>4}", .{byte});
            } else {
                try writer.writeByte(byte);
            },
        }
    }
    try writer.writeByte('"');
}

fn writeBucket(writer: anytype, bucket: Bucket) !void {
    try writer.print(
        "{{\"total\":{d},\"live_rows\":{d},\"matched\":{d},\"mismatched\":{d},\"unavailable\":{d},\"no_live_row\":{d}}}",
        .{ bucket.total, bucket.live_rows, bucket.matched, bucket.mismatched, bucket.unavailable, bucket.no_live_row },
    );
}

fn writeSummary(writer: anytype, summary: Summary) !void {
    try writer.print(
        "{{\"queries\":{d},\"promoted\":{d},\"live_rows\":{d},\"matched\":{d},\"mismatched\":{d},\"unavailable\":{d},\"no_live_row\":{d},\"required\":{d}}}",
        .{
            summary.queries,
            summary.promoted,
            summary.live_rows,
            summary.matched,
            summary.mismatched,
            summary.unavailable,
            summary.no_live_row,
            summary.required,
        },
    );
}

fn bumpUnavailable(
    summary: *Summary,
    buckets: *[shape_count + 1]Bucket,
    shape: ?obligation_to_z3.CanonicalPromotionShape,
) void {
    summary.unavailable +|= 1;
    buckets[bucketIndex(shape)].unavailable +|= 1;
}

fn bumpNoLiveRow(
    summary: *Summary,
    buckets: *[shape_count + 1]Bucket,
    shape: ?obligation_to_z3.CanonicalPromotionShape,
) void {
    summary.no_live_row +|= 1;
    buckets[bucketIndex(shape)].no_live_row +|= 1;
}

pub fn writeJson(
    writer: anytype,
    allocator: std.mem.Allocator,
    source_file: []const u8,
    set: obligation.ObligationSet,
) !void {
    var summary: Summary = .{};
    var buckets: [shape_count + 1]Bucket = .{Bucket{}} ** (shape_count + 1);
    var unsupported_reasons: UnsupportedReasonCounts = .{0} ** std.meta.fields(obligation_to_z3.CanonicalUnsupportedReason).len;
    var hash_errors = HashErrorCounts{ .allocator = allocator };
    defer hash_errors.deinit();

    var canonical_context: ?z3_verification.Z3Context = null;
    defer if (canonical_context) |*context| context.deinit();

    try writer.writeAll("{\"schema\":\"ora.canonical_z3.measure.v1\",\"source_file\":");
    try writeJsonString(writer, source_file);
    try writer.writeAll(",\"queries\":[");

    var first_query = true;
    for (set.queries) |query| {
        const shape = obligation_to_z3.queryCanonicalPromotionShape(set, query);
        const index = bucketIndex(shape);
        const required = obligation_to_z3.queryCanonicalRequiredModePromoted(set, query);

        summary.queries +|= 1;
        buckets[index].total +|= 1;
        if (shape != null) summary.promoted +|= 1;
        if (required) summary.required +|= 1;

        const has_live_row = query.smtlib_hash != null;
        if (has_live_row) {
            summary.live_rows +|= 1;
            buckets[index].live_rows +|= 1;
        }

        var outcome: []const u8 = "unavailable";
        var reason: ?[]const u8 = null;
        var canonical_hash: ?u64 = null;
        var canonical_constraint_count: ?u32 = null;

        switch (obligation_to_z3.queryCanonicalSupport(set, query)) {
            .supported => {
                if (!has_live_row) {
                    outcome = "no_live_row";
                    reason = "missing_live_prepared_row";
                    bumpNoLiveRow(&summary, &buckets, shape);
                } else {
                    if (canonical_context == null) {
                        canonical_context = try z3_verification.Z3Context.init(allocator);
                    }
                    var adapter = obligation_to_z3.Adapter.init(&canonical_context.?, allocator, set);
                    const canonical = adapter.queryHashForRow(query) catch |err| {
                        outcome = "unavailable";
                        reason = @errorName(err);
                        try hash_errors.add(reason.?);
                        bumpUnavailable(&summary, &buckets, shape);
                        if (!first_query) try writer.writeByte(',');
                        first_query = false;
                        try writeQueryRecord(
                            writer,
                            query,
                            shape,
                            required,
                            outcome,
                            reason,
                            null,
                            null,
                        );
                        continue;
                    };
                    canonical_hash = canonical.smtlib_hash;
                    canonical_constraint_count = canonical.constraint_count;
                    if (canonical.constraint_count == query.constraint_count and canonical.smtlib_hash == query.smtlib_hash.?) {
                        outcome = "match";
                        summary.matched +|= 1;
                        buckets[index].matched +|= 1;
                    } else {
                        outcome = "mismatch";
                        summary.mismatched +|= 1;
                        buckets[index].mismatched +|= 1;
                    }
                }
            },
            .unsupported => |unsupported| {
                reason = @tagName(unsupported);
                unsupported_reasons[@intFromEnum(unsupported)] +|= 1;
                bumpUnavailable(&summary, &buckets, shape);
            },
        }

        if (!first_query) try writer.writeByte(',');
        first_query = false;
        try writeQueryRecord(
            writer,
            query,
            shape,
            required,
            outcome,
            reason,
            canonical_hash,
            canonical_constraint_count,
        );
    }

    hash_errors.sort();

    try writer.writeAll("],\"summary\":");
    try writeSummary(writer, summary);

    try writer.writeAll(",\"shapes\":{");
    inline for (std.meta.fields(obligation_to_z3.CanonicalPromotionShape), 0..) |field, i| {
        if (i != 0) try writer.writeByte(',');
        try writeJsonString(writer, field.name);
        try writer.writeByte(':');
        try writeBucket(writer, buckets[i]);
    }
    if (shape_count != 0) try writer.writeByte(',');
    try writeJsonString(writer, "unpromoted");
    try writer.writeByte(':');
    try writeBucket(writer, buckets[unpromoted_index]);
    try writer.writeByte('}');

    try writer.writeAll(",\"unsupported_reasons\":{");
    inline for (std.meta.fields(obligation_to_z3.CanonicalUnsupportedReason), 0..) |field, i| {
        if (i != 0) try writer.writeByte(',');
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{unsupported_reasons[i]});
    }
    try writer.writeByte('}');

    try writer.writeAll(",\"hash_errors\":{");
    for (hash_errors.items.items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writeJsonString(writer, item.name);
        try writer.print(":{d}", .{item.count});
    }
    try writer.writeAll("}}\n");
}

fn writeQueryRecord(
    writer: anytype,
    query: obligation.VerificationQuery,
    shape: ?obligation_to_z3.CanonicalPromotionShape,
    required: bool,
    outcome: []const u8,
    reason: ?[]const u8,
    canonical_hash: ?u64,
    canonical_constraint_count: ?u32,
) !void {
    try writer.print("{{\"id\":{d},\"kind\":", .{query.id});
    try writeJsonString(writer, @tagName(query.kind));
    try writer.writeAll(",\"shape\":");
    try writeJsonString(writer, shapeName(shape));
    try writer.print(",\"required\":{},\"outcome\":", .{required});
    try writeJsonString(writer, outcome);
    try writer.writeAll(",\"reason\":");
    if (reason) |value| {
        try writeJsonString(writer, value);
    } else {
        try writer.writeAll("null");
    }
    try writer.print(",\"constraint_count\":{d},\"smtlib_hash\":", .{query.constraint_count});
    if (query.smtlib_hash) |hash| {
        try writer.print("\"0x{x}\"", .{hash});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"canonical_constraint_count\":");
    if (canonical_constraint_count) |count| {
        try writer.print("{d}", .{count});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"canonical_smtlib_hash\":");
    if (canonical_hash) |hash| {
        try writer.print("\"0x{x}\"", .{hash});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeByte('}');
}

test "canonical Z3 measurement writes empty report" {
    var out = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer out.deinit();

    try writeJson(&out.writer, std.testing.allocator, "empty.ora", .{});
    const actual = try out.toOwnedSlice();
    defer std.testing.allocator.free(actual);

    try std.testing.expect(std.mem.indexOf(u8, actual, "\"schema\":\"ora.canonical_z3.measure.v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, actual, "\"queries\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, actual, "\"unpromoted\"") != null);
}
