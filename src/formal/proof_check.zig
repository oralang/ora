//! Lean proof-row checker ingestion.
//!
//! Proof rows are not trusted by themselves. Each row is accepted only after
//! the compiler generates a fresh Lean checker against the current obligation
//! module and Lean type-checks the named theorem at `emittedQuery_<id>`.

const std = @import("std");
const obligation = @import("obligation.zig");
const proof_manifest = @import("proof_manifest.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const max_proof_artifact_bytes = 4 * 1024 * 1024;

pub const ApplyResult = struct {
    arena: std.heap.ArenaAllocator,
    set: obligation.ObligationSet,
    applied_count: usize,
    proof_check_source: []const u8 = &.{},
    certificate_json: []const u8 = &.{},

    pub fn deinit(self: *ApplyResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const ProofRow = proof_manifest.ProofRow;

pub fn applyProofRows(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    rows: []const ProofRow,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
) !?ApplyResult {
    if (rows.len == 0) return null;
    return try applyRows(allocator, set, rows, generated_namespace, obligations_source, process_environ, stdout, false);
}

fn applyRows(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    rows: []const ProofRow,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
    quiet: bool,
) !ApplyResult {
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var artifacts: std.ArrayList(obligation.ProofArtifact) = .empty;
    var lean_queries: std.ArrayList(obligation.VerificationQuery) = .empty;
    var checker_rows: std.ArrayList(CheckedProofRow) = .empty;
    var certificate_rows: std.ArrayList(CertificateRow) = .empty;

    const next_artifact_start = nextProofArtifactId(set);
    const next_query_start = nextQueryId(set);

    for (rows, 0..) |row, index| {
        try validateProofRowSyntax(row);
        const target_query = findQueryById(set, row.query_id) orelse return error.UnknownProofQuery;
        try validateProofTarget(target_query);
        if (!equalIds(target_query.obligation_ids, row.obligation_ids)) return error.ProofRowObligationMismatch;
        if (!equalIds(target_query.assumption_ids, row.assumption_ids)) return error.ProofRowAssumptionMismatch;

        const content_sha256 = try readAndCheckProofContentSha256(
            arena_allocator,
            allocator,
            io,
            row.path,
            row.content_sha256,
        );

        const index_id = std.math.cast(obligation.Id, index) orelse return error.TooManyProofRows;
        const artifact_id = next_artifact_start + index_id;
        const query_id = next_query_start + index_id;
        const proof_obligation_ids = try arena_allocator.dupe(obligation.Id, row.obligation_ids);
        const proof_assumption_ids = try arena_allocator.dupe(obligation.Id, row.assumption_ids);
        const module_name = try arena_allocator.dupe(u8, row.module_name);
        const theorem_name = try arena_allocator.dupe(u8, row.theorem_name);
        const proof_path = if (row.path) |path| try arena_allocator.dupe(u8, path) else null;

        try artifacts.append(arena_allocator, .{
            .id = artifact_id,
            .owner = target_query.owner,
            .source = target_query.source,
            .module_name = module_name,
            .theorem_name = theorem_name,
            .path = proof_path,
            .content_hash = null,
            .obligation_ids = proof_obligation_ids,
        });
        try lean_queries.append(arena_allocator, .{
            .id = query_id,
            .owner = target_query.owner,
            .source = target_query.source,
            .phase = .report,
            .origin = target_query.origin,
            .backend = .lean,
            .kind = target_query.kind,
            .logical_role = target_query.logical_role,
            .guard_id = target_query.guard_id,
            .obligation_ids = proof_obligation_ids,
            .assumption_ids = proof_assumption_ids,
            .proof_artifact_id = artifact_id,
            .discharges_query_id = row.query_id,
            .result = .{ .status = .proved },
        });
        try checker_rows.append(arena_allocator, .{
            .query_id = row.query_id,
            .module_name = module_name,
            .theorem_name = theorem_name,
        });
        try certificate_rows.append(arena_allocator, .{
            .target_query_id = row.query_id,
            .lean_query_id = query_id,
            .obligation_ids = proof_obligation_ids,
            .assumption_ids = proof_assumption_ids,
            .module_name = module_name,
            .theorem_name = theorem_name,
            .path = proof_path,
            .content_sha256 = content_sha256,
            .target_smtlib_hash = target_query.smtlib_hash,
            .target_constraint_count = target_query.constraint_count,
        });
    }

    if (checker_rows.items.len == 0) return error.EmptyProofRows;
    const proof_check = try runLeanChecker(allocator, generated_namespace, obligations_source, checker_rows.items, process_environ, stdout, quiet, arena_allocator);
    if (proof_check.axiom_audits.len != certificate_rows.items.len) return error.LeanAxiomAuditMissing;
    for (certificate_rows.items, proof_check.axiom_audits) |*row, audit| {
        row.axioms = audit.axioms;
    }
    const certificate_json = try buildCertificateJson(arena_allocator, generated_namespace, obligations_source, proof_check.source, certificate_rows.items, process_environ);

    var merged = set;
    merged.proof_artifacts = try concat(obligation.ProofArtifact, arena_allocator, set.proof_artifacts, artifacts.items);
    merged.queries = try concat(obligation.VerificationQuery, arena_allocator, set.queries, lean_queries.items);

    return .{
        .arena = arena,
        .set = merged,
        .applied_count = checker_rows.items.len,
        .proof_check_source = proof_check.source,
        .certificate_json = certificate_json,
    };
}

fn validateProofTarget(target_query: obligation.VerificationQuery) !void {
    if (target_query.obligation_ids.len == 0) return error.ProofRowMissingObligations;

    if (target_query.result) |target_result| {
        if (target_query.backend != .z3) return error.ProofRowTargetNotZ3;
        if (target_result.status != .unknown or target_result.degraded or target_result.vacuity_unknown) {
            return error.ProofRowTargetNotPlainUnknown;
        }
        return;
    }

    if (target_query.backend == .z3) return error.ProofRowTargetMissingResult;
}

fn validateProofRowSyntax(row: ProofRow) !void {
    try validateLeanModulePath(row.module_name);
    try validateLeanTheoremPath(row.theorem_name);
    if (row.content_sha256) |digest| try validateSha256Hex(digest);
    if (row.path == null and row.content_sha256 != null) return error.ProofRowDigestWithoutPath;
}

const CheckedProofRow = struct {
    query_id: obligation.Id,
    module_name: []const u8,
    theorem_name: []const u8,
};

const LeanCheckResult = struct {
    source: []const u8,
    axiom_audits: []const AxiomAuditRow,
};

const AxiomAuditRow = struct {
    check_name: []const u8,
    axioms: []const []const u8,
};

const CertificateRow = struct {
    target_query_id: obligation.Id,
    lean_query_id: obligation.Id,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id,
    module_name: []const u8,
    theorem_name: []const u8,
    path: ?[]const u8,
    content_sha256: ?[]const u8,
    axioms: []const []const u8 = &.{},
    target_smtlib_hash: ?u64,
    target_constraint_count: u32,
};

fn buildCertificateJson(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    proof_check_source: []const u8,
    rows: []const CertificateRow,
    process_environ: std.process.Environ,
) ![]const u8 {
    const lean_version = try queryLeanVersion(allocator, process_environ);
    defer allocator.free(lean_version);

    return try buildCertificateJsonWithLeanVersion(
        allocator,
        generated_namespace,
        obligations_source,
        proof_check_source,
        rows,
        lean_version,
    );
}

fn buildCertificateJsonWithLeanVersion(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    proof_check_source: []const u8,
    rows: []const CertificateRow,
    lean_version: []const u8,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    const obligations_sha256 = try sha256HexAlloc(allocator, obligations_source);
    defer allocator.free(obligations_sha256);
    const proof_check_sha256 = try sha256HexAlloc(allocator, proof_check_source);
    defer allocator.free(proof_check_sha256);

    try writer.writeAll("{\n  \"schema_version\": ");
    try writer.print("{d}", .{obligation.proof_certificate_schema_version});
    try writer.writeAll(",\n  \"status\": \"checked\",\n  \"checker\": \"lean\",\n  \"hash_algorithm\": \"sha256\",\n  \"lean_version\": ");
    try writeJsonString(writer, lean_version);
    try writer.writeAll(",\n  \"generated_namespace\": ");
    try writeJsonString(writer, generated_namespace);
    try writer.writeAll(",\n  \"obligation_module_sha256\": ");
    try writeJsonString(writer, obligations_sha256);
    try writer.writeAll(",\n  \"proof_check_sha256\": ");
    try writeJsonString(writer, proof_check_sha256);
    try writer.print(",\n  \"proof_count\": {d},\n  \"proofs\": [", .{rows.len});
    for (rows, 0..) |row, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.writeAll("\n    {\n      \"query_id\": ");
        try writer.print("{d}", .{row.target_query_id});
        try writer.writeAll(",\n      \"target_query_id\": ");
        try writer.print("{d}", .{row.target_query_id});
        try writer.writeAll(",\n      \"lean_query_id\": ");
        try writer.print("{d}", .{row.lean_query_id});
        try writer.writeAll(",\n      \"obligation_ids\": ");
        try writeJsonIdArray(writer, row.obligation_ids);
        try writer.writeAll(",\n      \"assumption_ids\": ");
        try writeJsonIdArray(writer, row.assumption_ids);
        try writer.writeAll(",\n      \"target_smtlib_hash\": ");
        if (row.target_smtlib_hash) |hash| {
            try writer.print("{d}", .{hash});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"target_constraint_count\": ");
        try writer.print("{d}", .{row.target_constraint_count});
        try writer.writeAll(",\n      \"module_name\": ");
        try writeJsonString(writer, row.module_name);
        try writer.writeAll(",\n      \"theorem_name\": ");
        try writeJsonString(writer, row.theorem_name);
        try writer.writeAll(",\n      \"path\": ");
        if (row.path) |path| {
            try writeJsonString(writer, path);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"content_sha256\": ");
        if (row.content_sha256) |digest| {
            try writeJsonString(writer, digest);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"axioms\": ");
        try writeJsonStringArray(writer, row.axioms);
        try writer.writeAll("\n    }");
    }
    try writer.writeAll("\n  ]\n}\n");
    return try allocator.dupe(u8, out.written());
}

fn queryLeanVersion(allocator: std.mem.Allocator, process_environ: std.process.Environ) ![]const u8 {
    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
        .environ = process_environ,
    });
    defer process_io.deinit();

    const argv = [_][]const u8{ "/usr/bin/env", "lake", "env", "lean", "--version" };
    const result = try std.process.run(allocator, process_io.io(), .{
        .argv = &argv,
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(16 * 1024),
        .stderr_limit = std.Io.Limit.limited(16 * 1024),
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) {
            const version = std.mem.trim(u8, result.stdout, " \t\r\n");
            if (version.len != 0) return try allocator.dupe(u8, version);
            const stderr = std.mem.trim(u8, result.stderr, " \t\r\n");
            if (stderr.len != 0) return try allocator.dupe(u8, stderr);
        },
        else => {},
    }
    return error.LeanVersionCheckFailed;
}

fn writeJsonIdArray(writer: anytype, ids: []const obligation.Id) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeJsonStringArray(writer: anytype, values: []const []const u8) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeJsonString(writer, value);
    }
    try writer.writeByte(']');
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => if (byte < 0x20) {
                try writer.writeAll("\\u00");
                try writer.writeByte(hex[byte >> 4]);
                try writer.writeByte(hex[byte & 0x0f]);
            } else {
                try writer.writeByte(byte);
            },
        }
    }
    try writer.writeByte('"');
}

fn runLeanChecker(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    rows: []const CheckedProofRow,
    process_environ: std.process.Environ,
    stdout: anytype,
    quiet: bool,
    result_allocator: std.mem.Allocator,
) !LeanCheckResult {
    _ = try namespaceSuffix(generated_namespace);
    for (rows) |row| {
        try validateLeanModulePath(row.module_name);
        try validateLeanTheoremPath(row.theorem_name);
    }
    const scratch_segment = try proofCheckScratchSegment(allocator, generated_namespace, obligations_source);
    defer allocator.free(scratch_segment);

    const io = std.Io.Threaded.global_single_threaded.io();
    const scratch_rel = try std.fmt.allocPrint(allocator, "formal/Ora/ProofCheckScratch/{s}", .{scratch_segment});
    defer allocator.free(scratch_rel);
    try std.Io.Dir.cwd().createDirPath(io, scratch_rel);
    defer std.Io.Dir.cwd().deleteTree(io, scratch_rel) catch {};

    const obligations_rel = try std.fmt.allocPrint(allocator, "{s}/Obligations.lean", .{scratch_rel});
    defer allocator.free(obligations_rel);
    const checker_rel = try std.fmt.allocPrint(allocator, "{s}/Checker.lean", .{scratch_rel});
    defer allocator.free(checker_rel);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = obligations_rel, .data = obligations_source });

    const obligations_module = try std.fmt.allocPrint(
        allocator,
        "Ora.ProofCheckScratch.{s}.Obligations",
        .{scratch_segment},
    );
    defer allocator.free(obligations_module);
    const checker_path = try std.fmt.allocPrint(
        allocator,
        "Ora/ProofCheckScratch/{s}/Checker.lean",
        .{scratch_segment},
    );
    defer allocator.free(checker_path);

    const checker_source = try buildLeanCheckerSource(result_allocator, generated_namespace, obligations_module, rows);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = checker_rel, .data = checker_source });

    try runLeanCommand(allocator, &.{ "lake", "build", obligations_module }, process_environ, stdout, quiet);
    for (rows) |row| {
        try runLeanCommand(allocator, &.{ "lake", "build", row.module_name }, process_environ, stdout, quiet);
    }
    const checker_output = try runLeanCommandCapture(allocator, &.{ "lake", "env", "lean", checker_path }, process_environ, stdout, quiet);
    defer allocator.free(checker_output.stdout);
    defer allocator.free(checker_output.stderr);

    return .{
        .source = checker_source,
        .axiom_audits = try parseLeanAxiomAuditOutput(result_allocator, checker_output.stdout, rows),
    };
}

fn buildLeanCheckerSource(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_module: []const u8,
    rows: []const CheckedProofRow,
) ![]const u8 {
    var checker = std.Io.Writer.Allocating.init(allocator);
    defer checker.deinit();
    const writer = &checker.writer;
    try writer.writeAll(
        \\-- Generated Lean proof check. This file is the source the compiler
        \\-- type-checked before accepting the referenced Lean proof artifacts.
        \\-- The obligation module imported below is materialized in an
        \\-- Ora.ProofCheckScratch namespace during the same compile.
        \\
        \\import Lean
        \\import Lean.Util.CollectAxioms
        \\
    );
    try writer.print("import {s}\n", .{obligations_module});
    for (rows) |row| {
        try writer.print("import {s}\n", .{row.module_name});
    }
    try writer.writeByte('\n');
    try writeLeanAxiomAuditPrelude(writer);
    try writer.writeByte('\n');
    for (rows, 0..) |row, index| {
        try writer.print(
            \\theorem check_query_{d}_{d} :
            \\    {s}.emittedQuery_{d} := by
            \\  exact {s}
            \\
            \\
        , .{ row.query_id, index, generated_namespace, row.query_id, row.theorem_name });
    }
    try writer.writeAll("#ora_audit_axioms [\n");
    for (rows, 0..) |row, index| {
        try writer.print("  check_query_{d}_{d}\n", .{ row.query_id, index });
    }
    try writer.writeAll("]\n");
    return try allocator.dupe(u8, checker.written());
}

fn writeLeanAxiomAuditPrelude(writer: anytype) !void {
    try writer.writeAll(
        \\open Lean Elab Command
        \\
        \\namespace Ora.ProofCheckAxiomAudit
        \\
        \\def normalizeNames (names : Array Name) : Array Name := Id.run do
        \\  let mut out := #[]
        \\  for name in names.qsort Name.lt do
        \\    unless out.any (fun existing => existing == name) do
        \\      out := out.push name
        \\  return out
        \\
        \\def allowedAxiom (name : Name) : Bool :=
        \\  name == `propext || name == `Quot.sound
        \\
        \\def formatAxioms (axioms : Array Name) : String :=
        \\  "[" ++ String.intercalate "," (axioms.toList.map (fun name => toString name)) ++ "]"
        \\
        \\syntax (name := oraProofCheckAxiomAudit) "#ora_audit_axioms " "[" ident* "]" : command
        \\
        \\@[command_elab oraProofCheckAxiomAudit] def elabOraProofCheckAxiomAudit : CommandElab := fun stx => do
        \\  let ids := stx[2].getArgs
        \\  let names := ids.map Syntax.getId
        \\  let env ← getEnv
        \\  for decl in names do
        \\    unless env.contains decl do
        \\      throwError m!"missing proof-check theorem during axiom audit: {decl}"
        \\
        \\    let axioms ← collectAxioms decl
        \\    let normalized := normalizeNames axioms
        \\    liftIO <| IO.println s!"ORA_AXIOM_AUDIT\t{decl}\t{formatAxioms normalized}"
        \\
        \\    let disallowed := normalized.filter (fun dep => !allowedAxiom dep)
        \\    if disallowed.size != 0 then
        \\      throwError m!"Lean proof uses disallowed axioms for {decl}: {formatAxioms disallowed}"
        \\
        \\end Ora.ProofCheckAxiomAudit
        \\
    );
}

fn runLeanCommand(allocator: std.mem.Allocator, argv: []const []const u8, process_environ: std.process.Environ, stdout: anytype, quiet: bool) !void {
    const output = try runLeanCommandCapture(allocator, argv, process_environ, stdout, quiet);
    defer allocator.free(output.stdout);
    defer allocator.free(output.stderr);
}

const LeanCommandOutput = struct {
    stdout: []u8,
    stderr: []u8,
};

fn runLeanCommandCapture(allocator: std.mem.Allocator, argv: []const []const u8, process_environ: std.process.Environ, stdout: anytype, quiet: bool) !LeanCommandOutput {
    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
        .environ = process_environ,
    });
    defer process_io.deinit();

    var env_argv: [8][]const u8 = undefined;
    if (argv.len + 1 > env_argv.len) return error.LeanCommandTooLong;
    env_argv[0] = "/usr/bin/env";
    @memcpy(env_argv[1 .. argv.len + 1], argv);

    const result = try std.process.run(allocator, process_io.io(), .{
        .argv = env_argv[0 .. argv.len + 1],
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(1024 * 1024),
    });

    switch (result.term) {
        .exited => |code| if (code == 0) {
            return .{ .stdout = result.stdout, .stderr = result.stderr };
        },
        else => {},
    }
    if (!quiet) {
        try stdout.writeAll("Lean proof row rejected.\n");
        if (result.stdout.len != 0) try stdout.print("{s}\n", .{result.stdout});
        if (result.stderr.len != 0) try stdout.print("{s}\n", .{result.stderr});
        try stdout.flush();
    }
    allocator.free(result.stdout);
    allocator.free(result.stderr);
    return error.LeanProofCheckFailed;
}

fn parseLeanAxiomAuditOutput(
    allocator: std.mem.Allocator,
    output: []const u8,
    rows: []const CheckedProofRow,
) ![]const AxiomAuditRow {
    const audits = try allocator.alloc(AxiomAuditRow, rows.len);
    const seen = try allocator.alloc(bool, rows.len);
    defer allocator.free(seen);
    @memset(seen, false);
    for (audits) |*audit| audit.* = .{ .check_name = &.{}, .axioms = &.{} };

    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, "\r");
        const prefix = "ORA_AXIOM_AUDIT\t";
        if (!std.mem.startsWith(u8, line, prefix)) continue;
        const rest = line[prefix.len..];
        const tab_index = std.mem.indexOfScalar(u8, rest, '\t') orelse return error.InvalidLeanAxiomAudit;
        const check_name = rest[0..tab_index];
        const axiom_list = rest[tab_index + 1 ..];

        const row_index = try findAxiomAuditRow(rows, check_name);
        if (seen[row_index]) return error.DuplicateLeanAxiomAudit;
        seen[row_index] = true;
        audits[row_index] = .{
            .check_name = try allocator.dupe(u8, check_name),
            .axioms = try parseAxiomList(allocator, axiom_list),
        };
    }

    for (seen) |was_seen| {
        if (!was_seen) return error.LeanAxiomAuditMissing;
    }
    return audits;
}

fn findAxiomAuditRow(rows: []const CheckedProofRow, check_name: []const u8) !usize {
    var expected_buf: [96]u8 = undefined;
    for (rows, 0..) |row, index| {
        const expected = std.fmt.bufPrint(&expected_buf, "check_query_{d}_{d}", .{ row.query_id, index }) catch return error.InvalidLeanAxiomAudit;
        if (std.mem.eql(u8, check_name, expected)) return index;
    }
    return error.UnexpectedLeanAxiomAudit;
}

fn parseAxiomList(allocator: std.mem.Allocator, raw: []const u8) ![]const []const u8 {
    const value = std.mem.trim(u8, raw, " \t\r\n");
    if (value.len < 2 or value[0] != '[' or value[value.len - 1] != ']') return error.InvalidLeanAxiomAudit;
    const inner = value[1 .. value.len - 1];
    if (inner.len == 0) return &.{};

    var items: std.ArrayList([]const u8) = .empty;
    var parts = std.mem.splitScalar(u8, inner, ',');
    while (parts.next()) |part_raw| {
        const part = std.mem.trim(u8, part_raw, " \t\r\n");
        if (part.len == 0) return error.InvalidLeanAxiomAudit;
        for (part) |byte| {
            if (byte < 0x20 or byte == '[' or byte == ']') return error.InvalidLeanAxiomAudit;
        }
        try items.append(allocator, try allocator.dupe(u8, part));
    }
    return try items.toOwnedSlice(allocator);
}

fn namespaceSuffix(namespace: []const u8) ![]const u8 {
    const marker = "Ora.Generated.Obligations.";
    if (!std.mem.startsWith(u8, namespace, marker)) return error.InvalidGeneratedNamespace;
    const suffix = namespace[marker.len..];
    if (suffix.len == 0) return error.InvalidGeneratedNamespace;
    validateLeanModulePath(namespace) catch return error.InvalidGeneratedNamespace;
    return suffix;
}

fn readAndCheckProofContentSha256(
    result_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    io: std.Io,
    path: ?[]const u8,
    expected_sha256: ?[]const u8,
) !?[]const u8 {
    if (expected_sha256) |digest| try validateSha256Hex(digest);
    const proof_path = path orelse {
        if (expected_sha256 != null) return error.ProofRowDigestWithoutPath;
        return null;
    };
    const content = try std.Io.Dir.cwd().readFileAlloc(
        io,
        proof_path,
        scratch_allocator,
        std.Io.Limit.limited(max_proof_artifact_bytes),
    );
    defer scratch_allocator.free(content);
    const actual = try sha256HexAlloc(result_allocator, content);
    if (expected_sha256) |expected| {
        if (!std.ascii.eqlIgnoreCase(actual, expected)) {
            result_allocator.free(actual);
            return error.ProofRowDigestMismatch;
        }
    }
    return actual;
}

fn proofCheckScratchSegment(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
) ![]const u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    var hasher = Sha256.init(.{});
    hasher.update(generated_namespace);
    hasher.update(&.{0});
    hasher.update(obligations_source);
    hasher.final(&digest);
    const digest_hex = std.fmt.bytesToHex(digest, .lower);

    const random_source = std.Random.IoSource{ .io = std.Io.Threaded.global_single_threaded.io() };
    const nonce = random_source.interface().int(u64);

    return try std.fmt.allocPrint(allocator, "Run_{s}_{x}", .{ digest_hex[0..16], nonce });
}

fn sha256HexAlloc(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(data, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return try allocator.dupe(u8, &hex);
}

fn validateSha256Hex(value: []const u8) !void {
    if (value.len != Sha256.digest_length * 2) return error.InvalidProofDigest;
    for (value) |byte| {
        if (!std.ascii.isHex(byte)) return error.InvalidProofDigest;
    }
}

fn validateLeanModulePath(path: []const u8) !void {
    if (!isLeanDottedIdentifier(path)) return error.InvalidLeanModuleName;
}

fn validateLeanTheoremPath(path: []const u8) !void {
    if (!isLeanDottedIdentifier(path)) return error.InvalidLeanTheoremName;
}

fn isLeanDottedIdentifier(path: []const u8) bool {
    if (path.len == 0) return false;
    var start: usize = 0;
    var index: usize = 0;
    while (index <= path.len) : (index += 1) {
        if (index == path.len or path[index] == '.') {
            if (!isLeanIdentifier(path[start..index])) return false;
            start = index + 1;
        }
    }
    return start == path.len + 1;
}

fn isLeanIdentifier(identifier: []const u8) bool {
    if (identifier.len == 0) return false;
    if (std.mem.eql(u8, identifier, "_")) return false;
    if (!isLeanIdentifierStart(identifier[0])) return false;
    for (identifier[1..]) |byte| {
        if (!isLeanIdentifierContinue(byte)) return false;
    }
    return !isLeanKeyword(identifier);
}

fn isLeanIdentifierStart(byte: u8) bool {
    return std.ascii.isAlphabetic(byte) or byte == '_';
}

fn isLeanIdentifierContinue(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_' or byte == '\'';
}

fn isLeanKeyword(identifier: []const u8) bool {
    const keywords = [_][]const u8{
        "abbrev",  "axiom",     "by",        "class",   "def",
        "do",      "else",      "end",       "example", "exists",
        "forall",  "fun",       "have",      "if",      "import",
        "in",      "inductive", "instance",  "let",     "macro",
        "match",   "namespace", "opaque",    "open",    "rec",
        "section", "show",      "structure", "then",    "theorem",
        "unsafe",  "where",     "with",
    };
    for (keywords) |keyword| {
        if (std.mem.eql(u8, identifier, keyword)) return true;
    }
    return false;
}

fn findQueryById(set: obligation.ObligationSet, id: obligation.Id) ?obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.id == id) return query;
    }
    return null;
}

fn nextQueryId(set: obligation.ObligationSet) obligation.Id {
    var max: obligation.Id = 0;
    for (set.queries) |query| max = @max(max, query.id);
    return max + 1;
}

fn nextProofArtifactId(set: obligation.ObligationSet) obligation.Id {
    var max: obligation.Id = 0;
    for (set.proof_artifacts) |artifact| max = @max(max, artifact.id);
    return max + 1;
}

fn equalIds(lhs: []const obligation.Id, rhs: []const obligation.Id) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| {
        if (a != b) return false;
    }
    return true;
}

fn concat(comptime T: type, allocator: std.mem.Allocator, lhs: []const T, rhs: []const T) ![]const T {
    const out = try allocator.alloc(T, lhs.len + rhs.len);
    @memcpy(out[0..lhs.len], lhs);
    @memcpy(out[lhs.len..], rhs);
    return out;
}

test "Lean proof row syntax accepts only strict dotted identifiers" {
    try validateProofRowSyntax(.{
        .query_id = 1,
        .obligation_ids = &.{1},
        .module_name = "Ora.Proofs.Transfer",
        .theorem_name = "Ora.Proofs.Transfer.preserves_supply",
    });

    try std.testing.expectError(error.InvalidLeanModuleName, validateProofRowSyntax(.{
        .query_id = 1,
        .obligation_ids = &.{1},
        .module_name = "Ora.Proofs.Transfer\nimport Bad",
        .theorem_name = "preserves_supply",
    }));
    try std.testing.expectError(error.InvalidLeanTheoremName, validateProofRowSyntax(.{
        .query_id = 1,
        .obligation_ids = &.{1},
        .module_name = "Ora.Proofs.Transfer",
        .theorem_name = "preserves_supply; exact False.elim",
    }));
    try std.testing.expectError(error.InvalidLeanModuleName, validateProofRowSyntax(.{
        .query_id = 1,
        .obligation_ids = &.{1},
        .module_name = "Ora.import.Transfer",
        .theorem_name = "preserves_supply",
    }));
}

test "proof content digest validation uses sha256" {
    const allocator = std.testing.allocator;
    const source = "theorem transfer_preserves_supply : True := by trivial\n";
    const expected = try sha256HexAlloc(allocator, source);
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(
        "9934213397f7424d8645b75b06e9636e774a98337a1d861b0adf0944e10762f3",
        expected,
    );

    try validateSha256Hex(expected);
    try std.testing.expectError(error.InvalidProofDigest, validateSha256Hex("1234"));
    try std.testing.expectError(error.InvalidProofDigest, validateSha256Hex(
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
    ));
}

test "proof content digest check reads path and rejects mismatch" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source = "theorem transfer_preserves_supply : True := by trivial\n";
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "Transfer.lean", .data = source });
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/Transfer.lean", .{tmp.sub_path});
    defer allocator.free(path);

    const expected = try sha256HexAlloc(allocator, source);
    defer allocator.free(expected);
    const actual = try readAndCheckProofContentSha256(
        allocator,
        allocator,
        std.testing.io,
        path,
        expected,
    ) orelse return error.ExpectedDigest;
    defer allocator.free(actual);
    try std.testing.expectEqualStrings(expected, actual);

    try std.testing.expectError(error.ProofRowDigestMismatch, readAndCheckProofContentSha256(
        allocator,
        allocator,
        std.testing.io,
        path,
        "0000000000000000000000000000000000000000000000000000000000000000",
    ));
    try std.testing.expectError(error.ProofRowDigestWithoutPath, readAndCheckProofContentSha256(
        allocator,
        allocator,
        std.testing.io,
        null,
        expected,
    ));
}

test "proof certificate JSON exposes stable schema fields" {
    const allocator = std.testing.allocator;
    const obligation_ids = [_]obligation.Id{7};
    const assumption_ids = [_]obligation.Id{3};
    const rows = [_]CertificateRow{.{
        .target_query_id = 11,
        .lean_query_id = 12,
        .obligation_ids = &obligation_ids,
        .assumption_ids = &assumption_ids,
        .module_name = "Ora.Proofs.Transfer",
        .theorem_name = "Ora.Proofs.Transfer.preserves_supply",
        .path = "proofs/Transfer.lean",
        .content_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
        .axioms = &.{ "propext", "Quot.sound" },
        .target_smtlib_hash = 99,
        .target_constraint_count = 5,
    }};

    const certificate = try buildCertificateJsonWithLeanVersion(
        allocator,
        "Ora.Generated.Obligations.Transfer",
        "def emittedQuery_11 : Prop := True\n",
        "theorem check_query_11_0 : True := by trivial\n",
        &rows,
        "Lean test version",
    );
    defer allocator.free(certificate);

    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"schema_version\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"version\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"hash_algorithm\": \"sha256\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"lean_version\": \"Lean test version\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"proof_count\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"target_query_id\": 11") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"lean_query_id\": 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"content_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"axioms\": [\"propext\", \"Quot.sound\"]") != null);
}

test "Lean checker source audits proof-row theorem axiom dependencies" {
    const allocator = std.testing.allocator;
    const rows = [_]CheckedProofRow{.{
        .query_id = 17,
        .module_name = "Ora.Proofs.Transfer",
        .theorem_name = "Ora.Proofs.Transfer.preserves_supply",
    }};

    const checker = try buildLeanCheckerSource(
        allocator,
        "Ora.Generated.Obligations.Transfer",
        "Ora.ProofCheckScratch.Run_test.Obligations",
        &rows,
    );
    defer allocator.free(checker);

    try std.testing.expect(std.mem.indexOf(u8, checker, "import Lean.Util.CollectAxioms") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "name == `propext || name == `Quot.sound") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "ORA_AXIOM_AUDIT\\t{decl}\\t{formatAxioms normalized}") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "theorem check_query_17_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "#ora_audit_axioms [\n  check_query_17_0\n]") != null);
}

test "Lean axiom audit output maps rows and rejects drift" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const rows = [_]CheckedProofRow{
        .{
            .query_id = 11,
            .module_name = "Ora.Proofs.Clean",
            .theorem_name = "Ora.Proofs.Clean.ok",
        },
        .{
            .query_id = 12,
            .module_name = "Ora.Proofs.Quot",
            .theorem_name = "Ora.Proofs.Quot.ok",
        },
    };
    const audits = try parseLeanAxiomAuditOutput(
        a,
        "noise\nORA_AXIOM_AUDIT\tcheck_query_11_0\t[]\nORA_AXIOM_AUDIT\tcheck_query_12_1\t[propext,Quot.sound]\n",
        &rows,
    );
    try std.testing.expectEqual(@as(usize, 2), audits.len);
    try std.testing.expectEqualStrings("check_query_11_0", audits[0].check_name);
    try std.testing.expectEqual(@as(usize, 0), audits[0].axioms.len);
    try std.testing.expectEqualStrings("check_query_12_1", audits[1].check_name);
    try std.testing.expectEqual(@as(usize, 2), audits[1].axioms.len);
    try std.testing.expectEqualStrings("propext", audits[1].axioms[0]);
    try std.testing.expectEqualStrings("Quot.sound", audits[1].axioms[1]);

    try std.testing.expectError(error.LeanAxiomAuditMissing, parseLeanAxiomAuditOutput(
        a,
        "ORA_AXIOM_AUDIT\tcheck_query_11_0\t[]\n",
        &rows,
    ));
    try std.testing.expectError(error.UnexpectedLeanAxiomAudit, parseLeanAxiomAuditOutput(
        a,
        "ORA_AXIOM_AUDIT\tcheck_query_11_0\t[]\nORA_AXIOM_AUDIT\tcheck_query_99_0\t[]\n",
        &rows,
    ));
    try std.testing.expectError(error.DuplicateLeanAxiomAudit, parseLeanAxiomAuditOutput(
        a,
        "ORA_AXIOM_AUDIT\tcheck_query_11_0\t[]\nORA_AXIOM_AUDIT\tcheck_query_11_0\t[]\n",
        rows[0..1],
    ));
}

test "Lean proof checker rejects sorry and user axioms" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var env_map = std.process.Environ.Map.init(allocator);
    defer env_map.deinit();
    const home = std.c.getenv("HOME") orelse return error.SkipZigTest;
    const home_slice = std.mem.span(home);
    try env_map.put("HOME", home_slice);
    const inherited_path = if (std.c.getenv("PATH")) |path| std.mem.span(path) else "/usr/bin:/bin:/opt/homebrew/bin";
    const path = try std.fmt.allocPrint(allocator, "{s}/.elan/bin:{s}", .{ home_slice, inherited_path });
    defer allocator.free(path);
    try env_map.put("PATH", path);
    const process_environ: std.process.Environ = .{ .block = try env_map.createPosixBlock(allocator, .{}) };
    defer process_environ.block.deinit(allocator);

    const suffix = "RunB2AxiomFixture";
    try std.Io.Dir.cwd().createDirPath(io, "formal/Ora/ProofCheckFixture");
    defer std.Io.Dir.cwd().deleteFile(io, "formal/Ora/ProofCheckFixture/RunB2AxiomFixture.lean") catch {};
    defer std.Io.Dir.cwd().deleteDir(io, "formal/Ora/ProofCheckFixture") catch {};

    const fixture_source =
        \\namespace Ora.ProofCheckFixture.RunB2AxiomFixture
        \\
        \\theorem clean : True := by
        \\  trivial
        \\
        \\axiom userAxiom : True
        \\theorem userAxiomBacked : True := by
        \\  exact userAxiom
        \\
        \\theorem sorryBacked : True := by
        \\  sorry
        \\
        \\end Ora.ProofCheckFixture.RunB2AxiomFixture
        \\
    ;
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = "formal/Ora/ProofCheckFixture/RunB2AxiomFixture.lean",
        .data = fixture_source,
    });

    try expectLeanFixtureProof(allocator, process_environ, suffix, "clean", true, null);
    try expectLeanFixtureProof(allocator, process_environ, suffix, "userAxiomBacked", false, "userAxiom");
    try expectLeanFixtureProof(allocator, process_environ, suffix, "sorryBacked", false, "sorryAx");
}

fn expectLeanFixtureProof(
    allocator: std.mem.Allocator,
    process_environ: std.process.Environ,
    suffix: []const u8,
    theorem: []const u8,
    should_accept: bool,
    expected_diagnostic: ?[]const u8,
) !void {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "fixture" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .origin_value = .{
                .origin = .source,
                .kind = .result,
                .index = 0,
            } },
        } },
        .required_backend = .lean,
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "fixture" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .z3,
        .kind = .obligation,
        .logical_role = .ensures,
        .obligation_ids = &obligation_ids,
        .result = .{ .status = .unknown },
    }};
    const set = obligation.ObligationSet{
        .obligations = &obligations,
        .queries = &queries,
    };
    const module_name = try std.fmt.allocPrint(allocator, "Ora.ProofCheckFixture.{s}", .{suffix});
    defer allocator.free(module_name);
    const theorem_name = try std.fmt.allocPrint(allocator, "Ora.ProofCheckFixture.{s}.{s}", .{ suffix, theorem });
    defer allocator.free(theorem_name);
    const rows = [_]ProofRow{.{
        .query_id = 2,
        .obligation_ids = &obligation_ids,
        .module_name = module_name,
        .theorem_name = theorem_name,
    }};
    const obligations_source =
        \\namespace Ora.Generated.Obligations.ProofCheckFixture
        \\
        \\def emittedQuery_2 : Prop := True
        \\
        \\end Ora.Generated.Obligations.ProofCheckFixture
        \\
    ;
    var stdout = std.Io.Writer.Allocating.init(allocator);
    defer stdout.deinit();

    if (should_accept) {
        var result = applyRows(
            allocator,
            set,
            &rows,
            "Ora.Generated.Obligations.ProofCheckFixture",
            obligations_source,
            process_environ,
            &stdout.writer,
            false,
        ) catch |err| {
            std.debug.print("proof-check fixture stdout:\n{s}\n", .{stdout.written()});
            return err;
        };
        defer result.deinit();
        try std.testing.expect(std.mem.indexOf(u8, result.certificate_json, "\"axioms\": []") != null);
        return;
    }

    try std.testing.expectError(error.LeanProofCheckFailed, applyRows(
        allocator,
        set,
        &rows,
        "Ora.Generated.Obligations.ProofCheckFixture",
        obligations_source,
        process_environ,
        &stdout.writer,
        false,
    ));
    if (expected_diagnostic) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, stdout.written(), needle) != null);
    }
}
