//! Shared catalog and audit index for compiler-produced formal artifacts.
//!
//! Artifact ids, suffixes, build layout, certificate classification, and index
//! ordering live here. The formal artifact index is an audit record only; it
//! reports authorization decisions made elsewhere and never grants authority.

const std = @import("std");

pub const index_schema_version: u32 = 2;

pub const ArtifactId = enum {
    smt_report_markdown,
    smt_report_json,
    guard_proof_sidecar,
    lean_obligations_source,
    lean_proof_certificate,
    dispatcher_proof_certificate,
    canonical_z3_measurement,
    source_accounting_report,
    formal_artifact_index,
};

pub const ArtifactKind = enum {
    report,
    proof_sidecar,
    proof_source,
    certificate,
    measurement,
    index,
};

pub const ArtifactOwner = enum {
    z3_userland,
    lean_userland,
    dispatcher_kernel,
    source_accounting_kernel,
    compiler,
};

pub const StagingDestination = enum {
    verify,
};

pub const ArtifactDefinition = struct {
    id: ArtifactId,
    suffix: []const u8,
    kind: ArtifactKind,
    owner: ArtifactOwner,
    schema_version: ?u32 = null,
    staging_destination: StagingDestination = .verify,
    optional: bool = true,
    retain_on_failure: bool,
    integrity_bound: bool = false,
    include_in_index: bool = true,
};

/// Stable artifact order. Do not reorder without a schema/migration review.
pub const artifacts = [_]ArtifactDefinition{
    .{ .id = .smt_report_markdown, .suffix = ".smt.report.md", .kind = .report, .owner = .z3_userland, .retain_on_failure = true },
    .{ .id = .smt_report_json, .suffix = ".smt.report.json", .kind = .report, .owner = .z3_userland, .schema_version = 1, .retain_on_failure = true },
    .{ .id = .guard_proof_sidecar, .suffix = ".proof.json", .kind = .proof_sidecar, .owner = .z3_userland, .schema_version = 1, .retain_on_failure = false },
    .{ .id = .lean_obligations_source, .suffix = ".lean.obligations.lean", .kind = .proof_source, .owner = .lean_userland, .retain_on_failure = true },
    .{ .id = .lean_proof_certificate, .suffix = ".lean.proof.json", .kind = .certificate, .owner = .lean_userland, .schema_version = 1, .retain_on_failure = false, .integrity_bound = true },
    .{ .id = .dispatcher_proof_certificate, .suffix = ".lean.dispatcher.proof.json", .kind = .certificate, .owner = .dispatcher_kernel, .schema_version = 1, .retain_on_failure = false, .integrity_bound = true },
    .{ .id = .canonical_z3_measurement, .suffix = ".canonical-z3.measure.json", .kind = .measurement, .owner = .z3_userland, .schema_version = 1, .retain_on_failure = true },
    .{ .id = .source_accounting_report, .suffix = ".formal.accounting.json", .kind = .report, .owner = .source_accounting_kernel, .schema_version = 1, .retain_on_failure = true, .integrity_bound = true },
    .{ .id = .formal_artifact_index, .suffix = ".formal.artifacts.json", .kind = .index, .owner = .compiler, .schema_version = index_schema_version, .optional = false, .retain_on_failure = true, .include_in_index = false },
};

pub const suffixes: [artifacts.len][]const u8 = blk: {
    var result: [artifacts.len][]const u8 = undefined;
    for (artifacts, 0..) |artifact, index| result[index] = artifact.suffix;
    break :blk result;
};

pub fn definition(id: ArtifactId) *const ArtifactDefinition {
    return switch (id) {
        .smt_report_markdown => &artifacts[0],
        .smt_report_json => &artifacts[1],
        .guard_proof_sidecar => &artifacts[2],
        .lean_obligations_source => &artifacts[3],
        .lean_proof_certificate => &artifacts[4],
        .dispatcher_proof_certificate => &artifacts[5],
        .canonical_z3_measurement => &artifacts[6],
        .source_accounting_report => &artifacts[7],
        .formal_artifact_index => &artifacts[8],
    };
}

pub fn filename(allocator: std.mem.Allocator, stem: []const u8, id: ArtifactId) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, definition(id).suffix });
}

pub const GateId = enum {
    z3_userland,
    lean_userland,
    dispatcher_kernel,
    source_accounting_kernel,
};

pub const GateDomain = enum {
    userland,
    compiler_kernel,
};

pub const GatePhase = enum {
    smt_verification,
    proof_artifact,
    sir_backend_binding,
    source_accounting,
};

pub const BlockingPolicy = enum {
    fallback_allowed,
    blocking_when_requested,
    always_blocking,
};

pub const GateDefinition = struct {
    id: GateId,
    domain: GateDomain,
    phase: GatePhase,
    blocking_policy: BlockingPolicy,
};

/// Stable gate order used by every index, including failure indexes.
pub const gates = [_]GateDefinition{
    .{ .id = .z3_userland, .domain = .userland, .phase = .smt_verification, .blocking_policy = .fallback_allowed },
    .{ .id = .lean_userland, .domain = .userland, .phase = .proof_artifact, .blocking_policy = .blocking_when_requested },
    .{ .id = .dispatcher_kernel, .domain = .compiler_kernel, .phase = .sir_backend_binding, .blocking_policy = .blocking_when_requested },
    .{ .id = .source_accounting_kernel, .domain = .compiler_kernel, .phase = .source_accounting, .blocking_policy = .always_blocking },
};

pub const GateStatus = enum {
    not_requested,
    not_run,
    accepted,
    rejected,
    skipped,
};

pub const GateStatuses = struct {
    z3_userland: GateStatus = .not_requested,
    lean_userland: GateStatus = .not_requested,
    dispatcher_kernel: GateStatus = .not_requested,
    source_accounting_kernel: GateStatus = .not_run,

    pub fn get(self: GateStatuses, id: GateId) GateStatus {
        return switch (id) {
            .z3_userland => self.z3_userland,
            .lean_userland => self.lean_userland,
            .dispatcher_kernel => self.dispatcher_kernel,
            .source_accounting_kernel => self.source_accounting_kernel,
        };
    }
};

pub fn validateSourceAccountingArming(statuses: GateStatuses, authorization: FinalAuthorization) !void {
    if (authorization == .allowed and statuses.source_accounting_kernel != .accepted) {
        return error.SourceAccountingGateNotAccepted;
    }
}

pub const FinalAuthorization = enum {
    allowed,
    blocked,
};

pub const IndexOptions = struct {
    source: []const u8,
    stem: []const u8,
    /// Directory containing artifacts while the index is written (often a
    /// compiler-owned staging directory).
    scan_dir: []const u8,
    /// Final public artifact root used as the base for recorded relative paths.
    artifact_root: []const u8,
    /// Final public directory that will contain cataloged formal artifacts.
    artifact_dir: []const u8,
    gate_statuses: GateStatuses,
    final_authorization: FinalAuthorization,
};

pub fn writeIndex(allocator: std.mem.Allocator, options: IndexOptions) !void {
    const json = try renderIndex(allocator, options);
    defer allocator.free(json);
    const index_name = try filename(allocator, options.stem, .formal_artifact_index);
    defer allocator.free(index_name);
    try std.Io.Dir.cwd().createDirPath(std.Io.Threaded.global_single_threaded.io(), options.scan_dir);
    const final_path = try std.fs.path.join(allocator, &.{ options.scan_dir, index_name });
    defer allocator.free(final_path);
    const temporary_name = try std.fmt.allocPrint(allocator, ".{s}.tmp", .{index_name});
    defer allocator.free(temporary_name);
    const temporary_path = try std.fs.path.join(allocator, &.{ options.scan_dir, temporary_name });
    defer allocator.free(temporary_path);
    errdefer std.Io.Dir.cwd().deleteFile(std.Io.Threaded.global_single_threaded.io(), temporary_path) catch {};
    try std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{
        .sub_path = temporary_path,
        .data = json,
    });
    try std.Io.Dir.rename(.cwd(), temporary_path, .cwd(), final_path, std.Io.Threaded.global_single_threaded.io());
}

pub fn renderIndex(allocator: std.mem.Allocator, options: IndexOptions) ![]u8 {
    try validateSourceAccountingArming(options.gate_statuses, options.final_authorization);
    return renderIndexForSchema(allocator, options, index_schema_version, &gates);
}

fn renderIndexForSchema(
    allocator: std.mem.Allocator,
    options: IndexOptions,
    schema_version: u32,
    gate_definitions: []const GateDefinition,
) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.print("{{\n  \"schema_version\": {d},\n  \"source\": ", .{schema_version});
    try writeJsonString(writer, options.source);
    try writer.writeAll(",\n  \"final_artifact_authorization\": ");
    try writeJsonString(writer, @tagName(options.final_authorization));
    try writer.writeAll(",\n  \"gates\": [\n");
    for (gate_definitions, 0..) |gate, index| {
        if (index != 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\"id\": ");
        try writeJsonString(writer, @tagName(gate.id));
        try writer.writeAll(", \"domain\": ");
        try writeJsonString(writer, @tagName(gate.domain));
        try writer.writeAll(", \"phase\": ");
        try writeJsonString(writer, @tagName(gate.phase));
        try writer.writeAll(", \"status\": ");
        try writeJsonString(writer, @tagName(options.gate_statuses.get(gate.id)));
        try writer.writeAll(", \"blocking_policy\": ");
        try writeJsonString(writer, @tagName(gate.blocking_policy));
        try writer.writeAll(", \"produced_artifacts\": [");
        var produced: usize = 0;
        for (artifacts) |artifact| {
            if (!artifact.include_in_index or !artifactOwnedByGate(artifact, gate.id)) continue;
            if (options.final_authorization == .blocked and !artifact.retain_on_failure) continue;
            if (!try artifactExists(allocator, options.scan_dir, options.stem, artifact.id)) continue;
            if (produced != 0) try writer.writeAll(", ");
            try writeJsonString(writer, @tagName(artifact.id));
            produced += 1;
        }
        try writer.writeByte(']');
        try writer.writeByte('}');
    }
    try writer.writeAll("\n  ],\n  \"artifacts\": [");

    const relative_dir = try relativeArtifactDir(allocator, options.artifact_root, options.artifact_dir);
    defer allocator.free(relative_dir);
    var emitted: usize = 0;
    for (artifacts) |artifact| {
        if (!artifact.include_in_index) continue;
        if (options.final_authorization == .blocked and !artifact.retain_on_failure) continue;
        const artifact_name = try filename(allocator, options.stem, artifact.id);
        defer allocator.free(artifact_name);
        const scan_path = try std.fs.path.join(allocator, &.{ options.scan_dir, artifact_name });
        defer allocator.free(scan_path);
        std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), scan_path, .{}) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        const relative_path = if (std.mem.eql(u8, relative_dir, "."))
            try allocator.dupe(u8, artifact_name)
        else
            try std.fs.path.join(allocator, &.{ relative_dir, artifact_name });
        defer allocator.free(relative_path);
        if (emitted == 0) try writer.writeByte('\n') else try writer.writeAll(",\n");
        try writer.writeAll("    {\"id\": ");
        try writeJsonString(writer, @tagName(artifact.id));
        try writer.writeAll(", \"kind\": ");
        try writeJsonString(writer, @tagName(artifact.kind));
        try writer.writeAll(", \"owner\": ");
        try writeJsonString(writer, @tagName(artifact.owner));
        try writer.writeAll(", \"path\": ");
        try writeJsonString(writer, relative_path);
        if (artifact.integrity_bound) {
            const digest = try sha256FileHex(allocator, scan_path);
            defer allocator.free(digest);
            try writer.writeAll(", \"sha256\": ");
            try writeJsonString(writer, digest);
        }
        try writer.writeByte('}');
        emitted += 1;
    }
    if (emitted != 0) try writer.writeByte('\n');
    try writer.writeAll("  ]\n}\n");
    return try out.toOwnedSlice();
}

fn artifactOwnedByGate(artifact: ArtifactDefinition, gate: GateId) bool {
    return switch (gate) {
        .z3_userland => artifact.owner == .z3_userland,
        .lean_userland => artifact.owner == .lean_userland,
        .dispatcher_kernel => artifact.owner == .dispatcher_kernel,
        .source_accounting_kernel => artifact.owner == .source_accounting_kernel,
    };
}

fn artifactExists(allocator: std.mem.Allocator, scan_dir: []const u8, stem: []const u8, id: ArtifactId) !bool {
    const artifact_name = try filename(allocator, stem, id);
    defer allocator.free(artifact_name);
    const scan_path = try std.fs.path.join(allocator, &.{ scan_dir, artifact_name });
    defer allocator.free(scan_path);
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), scan_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

fn relativeArtifactDir(allocator: std.mem.Allocator, root: []const u8, dir: []const u8) ![]u8 {
    const root_abs = try std.fs.path.resolve(allocator, &.{root});
    defer allocator.free(root_abs);
    const dir_abs = try std.fs.path.resolve(allocator, &.{dir});
    defer allocator.free(dir_abs);
    if (std.mem.eql(u8, root_abs, dir_abs)) return allocator.dupe(u8, ".");
    return std.fs.path.relative(allocator, ".", null, root_abs, dir_abs);
}

fn sha256FileHex(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Io.Threaded.global_single_threaded.io(),
        path,
        allocator,
        std.Io.Limit.limited(64 * 1024 * 1024),
    );
    defer allocator.free(bytes);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.allocPrint(allocator, "{x}", .{digest});
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0x00...0x08, 0x0b...0x0c, 0x0e...0x1f => try writer.print("\\u00{x:0>2}", .{byte}),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

test "artifact catalog covers every artifact id exactly once" {
    try std.testing.expectEqual(std.meta.fields(ArtifactId).len, artifacts.len);
    for (artifacts, 0..) |artifact, index| {
        try std.testing.expectEqual(artifact.id, definition(artifact.id).id);
        try std.testing.expectEqualStrings(artifact.suffix, suffixes[index]);
        for (artifacts[index + 1 ..]) |other| {
            try std.testing.expect(artifact.id != other.id);
            try std.testing.expect(!std.mem.eql(u8, artifact.suffix, other.suffix));
        }
    }
}

test "formal artifact index has deterministic gate and artifact order" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "contract.smt.report.json", .data = "{}\n" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "contract.lean.proof.json", .data = "certificate\n" });
    const scan_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(scan_dir);
    const json = try renderIndex(std.testing.allocator, .{
        .source = "contract.ora",
        .stem = "contract",
        .scan_dir = scan_dir,
        .artifact_root = scan_dir,
        .artifact_dir = scan_dir,
        .gate_statuses = .{ .z3_userland = .accepted, .source_accounting_kernel = .accepted },
        .final_authorization = .allowed,
    });
    defer std.testing.allocator.free(json);
    const z3_at = std.mem.indexOf(u8, json, "\"z3_userland\"").?;
    const lean_at = std.mem.indexOf(u8, json, "\"lean_userland\"").?;
    const dispatcher_at = std.mem.indexOf(u8, json, "\"dispatcher_kernel\"").?;
    const source_accounting_at = std.mem.indexOf(u8, json, "\"source_accounting_kernel\"").?;
    try std.testing.expect(z3_at < lean_at and lean_at < dispatcher_at and dispatcher_at < source_accounting_at);
    const report_at = std.mem.indexOf(u8, json, "\"smt_report_json\"").?;
    const certificate_at = std.mem.indexOf(u8, json, "\"lean_proof_certificate\"").?;
    try std.testing.expect(report_at < certificate_at);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"sha256\":") != null);
}

test "failure index records blocked authorization and rejected gates" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const scan_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(scan_dir);
    const json = try renderIndex(std.testing.allocator, .{
        .source = "failed.ora",
        .stem = "failed",
        .scan_dir = scan_dir,
        .artifact_root = scan_dir,
        .artifact_dir = scan_dir,
        .gate_statuses = .{ .z3_userland = .rejected, .lean_userland = .skipped, .dispatcher_kernel = .skipped },
        .final_authorization = .blocked,
    });
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"final_artifact_authorization\": \"blocked\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\": \"rejected\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\": \"skipped\"") != null);
}

test "artifact paths follow direct and build layouts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(std.testing.io, "staging");
    try tmp.dir.createDirPath(std.testing.io, "verify");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "staging/layout.smt.report.json", .data = "{}" });
    const root = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(root);
    const staging = try std.fs.path.join(std.testing.allocator, &.{ root, "staging" });
    defer std.testing.allocator.free(staging);
    const verify = try std.fs.path.join(std.testing.allocator, &.{ root, "verify" });
    defer std.testing.allocator.free(verify);
    const direct = try renderIndex(std.testing.allocator, .{
        .source = "layout.ora",
        .stem = "layout",
        .scan_dir = staging,
        .artifact_root = root,
        .artifact_dir = root,
        .gate_statuses = .{ .source_accounting_kernel = .accepted },
        .final_authorization = .allowed,
    });
    defer std.testing.allocator.free(direct);
    try std.testing.expect(std.mem.indexOf(u8, direct, "\"path\": \"layout.smt.report.json\"") != null);
    const build = try renderIndex(std.testing.allocator, .{
        .source = "layout.ora",
        .stem = "layout",
        .scan_dir = staging,
        .artifact_root = root,
        .artifact_dir = verify,
        .gate_statuses = .{ .source_accounting_kernel = .accepted },
        .final_authorization = .allowed,
    });
    defer std.testing.allocator.free(build);
    try std.testing.expect(std.mem.indexOf(u8, build, "\"path\": \"verify/layout.smt.report.json\"") != null);
}

test "source-accounting report descriptor is failure-retained and integrity-bound" {
    const descriptor = definition(.source_accounting_report);
    try std.testing.expectEqualStrings(".formal.accounting.json", descriptor.suffix);
    try std.testing.expectEqual(ArtifactOwner.source_accounting_kernel, descriptor.owner);
    try std.testing.expectEqual(@as(?u32, 1), descriptor.schema_version);
    try std.testing.expect(descriptor.retain_on_failure);
    try std.testing.expect(descriptor.integrity_bound);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "failed.formal.accounting.json", .data = "{\"decision\":\"rejected\"}\n" });
    const scan_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(scan_dir);
    const json = try renderIndex(std.testing.allocator, .{
        .source = "failed.ora",
        .stem = "failed",
        .scan_dir = scan_dir,
        .artifact_root = scan_dir,
        .artifact_dir = scan_dir,
        .gate_statuses = .{},
        .final_authorization = .blocked,
    });
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\": \"source_accounting_report\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"sha256\":") != null);
}

test "source-accounting schema migration is armed" {
    try std.testing.expectEqual(@as(u32, 2), index_schema_version);
    try std.testing.expectEqual(@as(usize, 4), gates.len);
    try std.testing.expectEqual(GateId.source_accounting_kernel, gates[3].id);
    try std.testing.expectEqual(BlockingPolicy.always_blocking, gates[3].blocking_policy);
    try std.testing.expectError(
        error.SourceAccountingGateNotAccepted,
        validateSourceAccountingArming(.{}, .allowed),
    );
    try validateSourceAccountingArming(.{ .source_accounting_kernel = .accepted }, .allowed);
    try validateSourceAccountingArming(.{ .source_accounting_kernel = .rejected }, .blocked);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const scan_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(scan_dir);
    const migrated = try renderIndex(std.testing.allocator, .{
        .source = "migration.ora",
        .stem = "migration",
        .scan_dir = scan_dir,
        .artifact_root = scan_dir,
        .artifact_dir = scan_dir,
        .gate_statuses = .{ .source_accounting_kernel = .accepted },
        .final_authorization = .allowed,
    });
    defer std.testing.allocator.free(migrated);
    try std.testing.expectEqualStrings(
        \\{
        \\  "schema_version": 2,
        \\  "source": "migration.ora",
        \\  "final_artifact_authorization": "allowed",
        \\  "gates": [
        \\    {"id": "z3_userland", "domain": "userland", "phase": "smt_verification", "status": "not_requested", "blocking_policy": "fallback_allowed", "produced_artifacts": []},
        \\    {"id": "lean_userland", "domain": "userland", "phase": "proof_artifact", "status": "not_requested", "blocking_policy": "blocking_when_requested", "produced_artifacts": []},
        \\    {"id": "dispatcher_kernel", "domain": "compiler_kernel", "phase": "sir_backend_binding", "status": "not_requested", "blocking_policy": "blocking_when_requested", "produced_artifacts": []},
        \\    {"id": "source_accounting_kernel", "domain": "compiler_kernel", "phase": "source_accounting", "status": "accepted", "blocking_policy": "always_blocking", "produced_artifacts": []}
        \\  ],
        \\  "artifacts": [  ]
        \\}
        \\
    ,
        migrated,
    );
}
