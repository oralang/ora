//! Versioned JSON proof-row manifest parser.
//!
//! The manifest is an input to `proof_check`; it is not trusted. The checker
//! still validates query ids, obligation ids, theorem names, hashes, and Lean's
//! kernel/audit result before any row can affect artifact emission.

const std = @import("std");
const obligation = @import("obligation.zig");

pub const schema_version: u32 = 1;

pub const ProofRow = struct {
    query_id: obligation.Id,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id = &.{},
    module_name: []const u8,
    theorem_name: []const u8,
    path: ?[]const u8 = null,
    content_sha256: ?[]const u8 = null,
};

pub const ParseResult = struct {
    arena: std.heap.ArenaAllocator,
    rows: []const ProofRow,

    pub fn deinit(self: *ParseResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const ManifestJson = struct {
    schema_version: u32,
    proofs: []const ProofRowJson,
};

const ProofRowJson = struct {
    query_id: obligation.Id,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id = &.{},
    module_name: []const u8,
    theorem_name: []const u8,
    path: ?[]const u8 = null,
    content_sha256: ?[]const u8 = null,
};

pub fn parseProofManifestJson(
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !ParseResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const owned_bytes = try arena_allocator.dupe(u8, bytes);
    const parsed = try std.json.parseFromSliceLeaky(ManifestJson, arena_allocator, owned_bytes, .{
        .ignore_unknown_fields = false,
    });
    if (parsed.schema_version != schema_version) return error.UnsupportedProofManifestSchema;

    const rows = try arena_allocator.alloc(ProofRow, parsed.proofs.len);
    for (parsed.proofs, rows) |row, *out| {
        out.* = .{
            .query_id = row.query_id,
            .obligation_ids = row.obligation_ids,
            .assumption_ids = row.assumption_ids,
            .module_name = row.module_name,
            .theorem_name = row.theorem_name,
            .path = row.path,
            .content_sha256 = row.content_sha256,
        };
    }

    return .{
        .arena = arena,
        .rows = rows,
    };
}

test "parse proof manifest json" {
    const source =
        \\{
        \\  "schema_version": 1,
        \\  "proofs": [
        \\    {
        \\      "query_id": 7,
        \\      "obligation_ids": [2, 3],
        \\      "assumption_ids": [1],
        \\      "module_name": "Ora.UserProofs",
        \\      "theorem_name": "Ora.UserProofs.transfer",
        \\      "path": "formal/Ora/UserProofs.lean",
        \\      "content_sha256": null
        \\    }
        \\  ]
        \\}
    ;

    var parsed = try parseProofManifestJson(std.testing.allocator, source);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.rows.len);
    try std.testing.expectEqual(@as(obligation.Id, 7), parsed.rows[0].query_id);
    try std.testing.expectEqualSlices(obligation.Id, &.{ 2, 3 }, parsed.rows[0].obligation_ids);
    try std.testing.expectEqualSlices(obligation.Id, &.{1}, parsed.rows[0].assumption_ids);
    try std.testing.expectEqualStrings("Ora.UserProofs", parsed.rows[0].module_name);
    try std.testing.expectEqualStrings("Ora.UserProofs.transfer", parsed.rows[0].theorem_name);
    try std.testing.expectEqualStrings("formal/Ora/UserProofs.lean", parsed.rows[0].path.?);
    try std.testing.expect(parsed.rows[0].content_sha256 == null);
}

test "proof manifest schema version is required" {
    try std.testing.expectError(error.MissingField, parseProofManifestJson(std.testing.allocator,
        \\{"proofs":[]}
    ));
    try std.testing.expectError(error.UnsupportedProofManifestSchema, parseProofManifestJson(std.testing.allocator,
        \\{"schema_version":2,"proofs":[]}
    ));
}
