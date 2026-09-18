//! Userland formal-verification coordinator.
//!
//! This module consumes the collected
//! obligation set and Z3 prepared-query report, then coordinates optional Lean
//! proof rows, agreement checks, anti-vacuity status, and proof certificates.
//!
//! The result deliberately carries artifact-emission authority only. It cannot
//! authorize guard erasure or resource runtime-check removal; those decisions
//! remain tied to the original Z3 `VerificationResult` in the compiler driver.

const std = @import("std");
const artifact_catalog = @import("../shared/artifact_catalog.zig");
const ora_types = @import("ora_types");
const z3_verification = @import("ora_z3_verification");
const obligation = @import("../obligation.zig");
const obligation_from_mlir = @import("../obligation_from_mlir.zig");
const obligation_from_z3 = @import("../obligation_from_z3.zig");
const obligation_to_lean = @import("../obligation_to_lean.zig");
const proof_check = @import("../proof_check.zig");
const proof_manifest = @import("../proof_manifest.zig");

pub const ProofMode = union(enum) {
    automatic,
    manifest: []const u8,
};

pub const Options = struct {
    output_dir: ?[]const u8 = null,
    artifact_display_dir: ?[]const u8 = null,
    process_environ: std.process.Environ,
    suppress_artifact_logs: bool = false,
    module: ?@import("mlir_c_api").c.MlirModule = null,
};

/// Userland proofs may authorize artifact emission only. This type
/// intentionally has no guard-erasure or resource-runtime-check capability.
pub const Outcome = struct {
    artifact_emission_authorized: bool,
    certificate_emitted: bool,
};

pub fn coordinate(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
    mode: ProofMode,
    options: Options,
    stdout: anytype,
) !Outcome {
    if (!try runtimeLoopCertificateTargetsComplete(formal_result.set, stdout)) {
        return .{
            .artifact_emission_authorized = false,
            .certificate_emitted = false,
        };
    }
    return switch (mode) {
        .manifest => |path| coordinateManifest(
            allocator,
            file_path,
            formal_result,
            smt_report,
            path,
            options,
            stdout,
        ),
        .automatic => coordinateAutomatic(
            allocator,
            file_path,
            formal_result,
            smt_report,
            options,
            stdout,
        ),
    };
}

/// Emits a user-facing Lean proof recipe for targets that require a Lean
/// certificate. This is a diagnostic path only and returns no artifact or
/// runtime-check authority.
pub fn emitUnknownRecipe(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
    options: Options,
    stdout: anytype,
) !void {
    var context = try collectContext(allocator, file_path, formal_result, smt_report);
    defer context.deinit();

    var recipe_count: usize = 0;
    var unavailable_count: usize = 0;
    for (context.merged_result.set.queries) |query| {
        if (!isLeanProofTarget(query)) continue;
        if (query.function_summary_id != null) {
            recipe_count += 1;
            continue;
        }
        const projection = projectionForQuery(context.merged_result.set, query);
        const semantic_support = obligation_to_lean.querySemanticSupport(context.merged_result.set, query);
        if (projection.hasOpaqueFormula() or !semanticSupportAvailable(semantic_support)) {
            unavailable_count += 1;
        } else {
            recipe_count += 1;
        }
    }

    const unknown_rows = plainUnknownPreparedRows(smt_report);
    const classified = recipe_count + unavailable_count;
    const unmatched = if (unknown_rows > classified) unknown_rows - classified else 0;
    if (unavailable_count != 0 or unmatched != 0) {
        try stdout.writeAll("Lean proof recipe unavailable for some required userland obligations:\n");
        for (context.merged_result.set.queries) |query| {
            if (!isLeanProofTarget(query)) continue;
            if (query.function_summary_id != null) continue;
            const projection = projectionForQuery(context.merged_result.set, query);
            const semantic_support = obligation_to_lean.querySemanticSupport(context.merged_result.set, query);
            if (!projection.hasOpaqueFormula() and semanticSupportAvailable(semantic_support)) continue;
            try stdout.print("  - query: emittedQuery_{d} (", .{query.id});
            try printRecipeSource(stdout, query.source);
            try stdout.writeAll(")\n    ");
            try printProjection(stdout, projection);
            try stdout.writeByte('\n');
            if (projection.hasOpaqueFormula()) {
                try stdout.writeAll("    reason: formula is still an MLIR origin_value; Lean proof export only supports projected Term formulas today\n");
            } else switch (semantic_support) {
                .supported => try stdout.writeAll("    reason: this obligation kind is not in the Lean semantic proof fragment yet\n"),
                .unsupported => |reason| try printUnsupportedReason(stdout, reason),
            }
        }
        if (unmatched != 0) {
            const projection = projectionForSet(context.merged_result.set);
            try stdout.print("  - unmatched UNKNOWN prepared rows: {d}\n    ", .{unmatched});
            try printProjection(stdout, projection);
            try stdout.writeAll("\n    reason: no matching Lean-dischargeable obligation query was derived from the MLIR manifest\n");
        }
        try stdout.writeByte('\n');
    }
    if (recipe_count == 0) return;

    const source = renderObligationsSource(
        allocator,
        if (formal_result.set.function_summaries.len != 0) formal_result.set else context.merged_result.set,
        context.generated_namespace,
    ) catch |err| switch (err) {
        error.UnsupportedOriginValue, error.UnsupportedObligationKind => {
            const projection = projectionForSet(context.merged_result.set);
            try stdout.writeAll("Lean proof recipe unavailable for some Z3 UNKNOWN obligations:\n  - manifest cannot be rendered in the Lean Term fragment\n    ");
            try printProjection(stdout, projection);
            try stdout.writeByte('\n');
            return;
        },
        else => return err,
    };
    defer allocator.free(source);
    const module_path = try writeObligationsArtifact(allocator, file_path, options, source);
    defer allocator.free(module_path);

    try stdout.writeAll("Lean proof recipe for required userland obligations:\n");
    try stdout.print("  obligation module: {s}\n  generated namespace: {s}\n", .{ module_path, context.generated_namespace });
    for (context.merged_result.set.queries) |query| {
        if (!isLeanProofTarget(query)) continue;
        const projection = projectionForQuery(context.merged_result.set, query);
        if (query.function_summary_id == null and (projection.hasOpaqueFormula() or
            !semanticSupportAvailable(obligation_to_lean.querySemanticSupport(context.merged_result.set, query)))) continue;
        const adapter_loop = if (query.loop_summary_id) |id|
            obligation_to_lean.loopUsesCallAdapter(context.merged_result.set, obligation.findById(context.merged_result.set.loop_summaries, id).?)
        else
            false;
        const target_id = query.function_summary_id orelse (if (adapter_loop) query.loop_summary_id.? else query.id);
        const target_name = if (query.function_summary_id != null) "emittedCallCustody" else if (adapter_loop) "emittedCallLoop" else "emittedQuery";
        try stdout.print("  - query: {s}_{d} (", .{ target_name, target_id });
        try printRecipeSource(stdout, query.source);
        try stdout.print(")\n    theorem shape: theorem discharge_q{d} : {s}.{s}_{d} := by ...\n    proof row: query_id={d}, obligation_ids=", .{
            query.id,
            context.generated_namespace,
            target_name,
            target_id,
            query.id,
        });
        try printIds(stdout, query.obligation_ids);
        try stdout.writeAll(", assumption_ids=");
        try printIds(stdout, query.assumption_ids);
        try stdout.writeAll(", path=<user-proof.lean>, inline_source=true\n");
    }
    try stdout.writeAll("  pass the proof manifest with `--lean-proofs <proofs.json>`\n\n");
}

const ObligationContext = struct {
    allocator: std.mem.Allocator,
    merged_result: obligation_from_z3.OverlayResult,
    generated_namespace: []const u8,
    obligations_source: ?[]const u8 = null,

    fn deinit(self: *ObligationContext) void {
        if (self.obligations_source) |source| self.allocator.free(source);
        self.allocator.free(self.generated_namespace);
        self.merged_result.deinit();
        self.* = undefined;
    }
};

fn collectContext(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
) !ObligationContext {
    const query_manifest = smt_report.query_manifest orelse return error.MissingPreparedQueryManifest;
    for (query_manifest.certified_calls) |used| {
        const bound = for (formal_result.call_bindings) |binding| {
            if (binding.source_op_id == used.source_op_id and binding.function_id == used.function_id) break true;
        } else false;
        if (!bound) return error.CallInvocationBindingMissing;
        const digest = try obligation_to_lean.functionVcSha256(formal_result.set, used.function_id);
        if (!std.mem.eql(u8, &digest, &used.digest)) return error.CallInvocationDigestMismatch;
    }
    var merged_result = try obligation_from_z3.overlayPreparedQueryResults(
        allocator,
        formal_result.set,
        query_manifest.rows,
    );
    errdefer merged_result.deinit();
    const generated_namespace = try generatedNamespace(allocator, file_path);
    errdefer allocator.free(generated_namespace);
    return .{
        .allocator = allocator,
        .merged_result = merged_result,
        .generated_namespace = generated_namespace,
    };
}

fn buildContext(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
) !ObligationContext {
    var context = try collectContext(allocator, file_path, formal_result, smt_report);
    errdefer context.deinit();
    context.obligations_source = try renderObligationsSource(
        allocator,
        if (formal_result.set.function_summaries.len != 0) formal_result.set else context.merged_result.set,
        context.generated_namespace,
    );
    return context;
}

fn renderObligationsSource(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    generated_namespace: []const u8,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    if (set.function_summaries.len != 0) try obligation_to_lean.writeCallCustodyModule(&out.writer, set, generated_namespace) else try obligation_to_lean.writeModule(&out.writer, set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    return try out.toOwnedSlice();
}

fn coordinateManifest(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
    manifest_path: []const u8,
    options: Options,
    stdout: anytype,
) !Outcome {
    var context = try buildContext(allocator, file_path, formal_result, smt_report);
    defer context.deinit();

    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Io.Threaded.global_single_threaded.io(),
        manifest_path,
        allocator,
        .unlimited,
    );
    defer allocator.free(manifest_bytes);
    var parsed_manifest = try proof_manifest.parseProofManifestJson(allocator, manifest_bytes);
    defer parsed_manifest.deinit();

    var applied: ?proof_check.ApplyResult = if (context.merged_result.set.function_summaries.len != 0)
        try proof_check.applyCallProofRows(allocator, options.module orelse return error.CallCustodyHIRMissing, context.merged_result.set, parsed_manifest.rows, context.generated_namespace, options.process_environ, stdout)
    else
        try proof_check.applyProofRows(
            allocator,
            context.merged_result.set,
            parsed_manifest.rows,
            context.generated_namespace,
            context.obligations_source.?,
            options.process_environ,
            stdout,
        );
    defer if (applied) |*result| result.deinit();

    const decided_set = if (applied) |*result| result.set else context.merged_result.set;
    var certificate_emitted = false;
    if (applied) |*result| {
        try writeCertificate(allocator, file_path, options, result.certificate_json, stdout);
        certificate_emitted = true;
    }
    const artifact_emission_authorized = try reportDecision(
        stdout,
        "Lean proof gate",
        decided_set,
    );
    if (artifact_emission_authorized and applied != null) {
        try printAcceptedLoopCertificates(stdout, decided_set, parsed_manifest.rows);
    }
    return .{
        .artifact_emission_authorized = artifact_emission_authorized,
        .certificate_emitted = certificate_emitted,
    };
}

fn reportDecision(
    stdout: anytype,
    label: []const u8,
    set: obligation.ObligationSet,
) !bool {
    return switch (set.artifactDecision()) {
        .allowed => true,
        .blocked => |reason| blk: {
            if (reason == .blocking_diagnostic) try printBlockingDiagnostics(stdout, set);
            try stdout.print("{s} did not authorize artifact emission: {s}\n", .{ label, @tagName(reason) });
            break :blk false;
        },
    };
}

fn printAcceptedLoopCertificates(
    stdout: anytype,
    set: obligation.ObligationSet,
    rows: []const proof_check.ProofRow,
) !void {
    var count: usize = 0;
    for (rows) |row| {
        const query = obligation.findById(set.queries, row.query_id) orelse continue;
        if (query.kind == .loop_induction) count += 1;
    }
    if (count == 0) return;

    try stdout.print(
        "Lean loop proof gate accepted:\n  kernel-checked certificates: {d}\n  query ids: ",
        .{count},
    );
    var written: usize = 0;
    for (rows) |row| {
        const query = obligation.findById(set.queries, row.query_id) orelse continue;
        if (query.kind != .loop_induction) continue;
        if (written != 0) try stdout.writeAll(", ");
        try stdout.print("{d}", .{row.query_id});
        written += 1;
    }
    try stdout.writeByte('\n');
}

fn writeCertificate(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    options: Options,
    certificate_json: []const u8,
    stdout: anytype,
) !void {
    const stem = std.fs.path.stem(file_path);
    const filename = try artifact_catalog.filename(allocator, stem, .lean_proof_certificate);
    defer allocator.free(filename);
    var path_buf: ?[]u8 = null;
    defer if (path_buf) |path| allocator.free(path);
    const path = if (options.output_dir) |dir| blk: {
        try std.Io.Dir.cwd().createDirPath(std.Io.Threaded.global_single_threaded.io(), dir);
        path_buf = try std.fs.path.join(allocator, &.{ dir, filename });
        break :blk path_buf.?;
    } else filename;
    try std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{
        .sub_path = path,
        .data = certificate_json,
    });
    if (!options.suppress_artifact_logs) {
        try stdout.print("Lean proof certificate saved to {s}\n", .{path});
    }
}

pub fn printBlockingDiagnostics(stdout: anytype, set: obligation.ObligationSet) !void {
    var printed_header = false;
    for (set.diagnostics) |diagnostic| {
        if (!diagnostic.blocks_artifacts) continue;
        if (!printed_header) {
            try stdout.writeAll("Formal proof gate diagnostics:\n");
            printed_header = true;
        }
        try stdout.print("  - {s}: {s}", .{ @tagName(diagnostic.kind), diagnostic.message });
        try printSourceRef(stdout, diagnostic.source);
        try stdout.writeByte('\n');
    }
}

fn printSourceRef(stdout: anytype, source: obligation.SourceRef) !void {
    if (source.file) |file| {
        try stdout.print(" ({s}", .{file});
        if (source.line != 0) try stdout.print(":{d}", .{source.line});
        if (source.column != 0) try stdout.print(":{d}", .{source.column});
        try stdout.writeByte(')');
    } else if (source.line != 0) {
        try stdout.print(" (line {d}", .{source.line});
        if (source.column != 0) try stdout.print(":{d}", .{source.column});
        try stdout.writeByte(')');
    }
}

fn generatedNamespace(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    var component = std.Io.Writer.Allocating.init(allocator);
    defer component.deinit();
    try component.writer.writeAll("Source_");
    for (stem) |byte| {
        try component.writer.writeByte(if (std.ascii.isAlphanumeric(byte) or byte == '_') byte else '_');
    }
    return try std.fmt.allocPrint(
        allocator,
        "Ora.Generated.Obligations.{s}_{x}",
        .{ component.written(), std.hash.Wyhash.hash(0, file_path) },
    );
}

fn isPlainUnknownTarget(query: obligation.VerificationQuery) bool {
    if (query.artifact_policy == .diagnostic_only) return false;
    if (query.backend != .z3 or query.obligation_ids.len == 0) return false;
    // LoopVC policy requires an independent conclusive Z3 result. Sending a
    // loop-owned UNKNOWN row to Lean would create evidence that cannot and
    // must not authorize artifacts.
    if (query.loop_summary_id != null) return false;
    const result = query.result orelse return false;
    return result.status == .unknown and
        !result.degraded and
        !result.vacuous and
        !result.vacuity_unknown;
}

fn isLeanProofTarget(query: obligation.VerificationQuery) bool {
    if (query.kind == .loop_induction or query.function_summary_id != null) {
        return query.artifact_policy == .blocks_verified_artifacts and
            query.proof_requirement == .lean_certificate and
            query.backend == .unspecified and
            query.result == null and
            (query.loop_summary_id != null or query.function_summary_id != null) and
            query.obligation_ids.len == 0;
    }
    if (isPlainUnknownTarget(query)) return true;
    return false;
}

fn runtimeLoopCertificateTarget(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
) ?obligation.VerificationQuery {
    if (!summary.projectionSupported() or summary.query_ids.induction.len != 1) return null;
    const query = obligation.findById(set.queries, summary.query_ids.induction[0]) orelse return null;
    if (!isLeanProofTarget(query) or query.loop_summary_id != summary.id) return null;
    return query;
}

fn runtimeLoopCertificateTargetsComplete(
    set: obligation.ObligationSet,
    stdout: anytype,
) !bool {
    var complete = true;
    var printed_header = false;
    for (set.loop_summaries) |summary| {
        if (runtimeLoopCertificateTarget(set, summary) != null) continue;
        complete = false;
        if (!printed_header) {
            try stdout.writeAll("Lean proof gate cannot cover every runtime loop:\n");
            printed_header = true;
        }
        try stdout.writeAll("  - ");
        if (summary.source.file) |file| try stdout.writeAll(file) else try stdout.writeAll("<generated>");
        if (summary.source.line != 0) try stdout.print(":{d}", .{summary.source.line});
        if (summary.owner == .statement) {
            try stdout.print(" ({s})", .{summary.owner.statement.function_name});
        }
        try stdout.writeAll(" excluded_by=");
        if (summary.unsupported_reasons.len != 0) {
            for (summary.unsupported_reasons, 0..) |reason, index| {
                if (index != 0) try stdout.writeByte(',');
                try stdout.writeAll(@tagName(reason));
            }
        } else if (summary.validationReason()) |reason| {
            try stdout.writeAll(@tagName(reason));
        } else {
            try stdout.writeAll("loop_certificate_target_missing");
        }
        try stdout.writeByte('\n');
    }
    return complete;
}

fn coordinateAutomatic(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    formal_result: *const obligation_from_mlir.CollectResult,
    smt_report: z3_verification.SmtReportArtifacts,
    options: Options,
    stdout: anytype,
) !Outcome {
    var context = try collectContext(allocator, file_path, formal_result, smt_report);
    defer context.deinit();

    var targets: std.ArrayList(obligation.VerificationQuery) = .empty;
    defer targets.deinit(allocator);
    var unavailable_count: usize = 0;
    var automatic_fragment_unavailable_count: usize = 0;
    for (context.merged_result.set.queries) |query| {
        if (!isLeanProofTarget(query)) continue;
        if (!semanticSupportAvailable(obligation_to_lean.querySemanticSupport(context.merged_result.set, query))) {
            unavailable_count += 1;
            continue;
        }
        if (query.kind == .loop_induction) switch (automaticLoopPlan(context.merged_result.set, query)) {
            .supported => {},
            .unsupported => |reason| {
                automatic_fragment_unavailable_count += 1;
                try stdout.print(
                    "Automatic Lean loop proof unavailable: {s}\n",
                    .{automaticLoopProofReasonName(reason)},
                );
                continue;
            },
        };
        try targets.append(allocator, query);
    }

    const unknown_rows = plainUnknownPreparedRows(smt_report);
    const classified_targets = plainUnknownTargetCount(context.merged_result.set);
    const unmatched = if (unknown_rows > classified_targets) unknown_rows - classified_targets else 0;
    if (unavailable_count != 0 or automatic_fragment_unavailable_count != 0 or unmatched != 0) {
        try stdout.writeAll("Automatic Lean proof gate cannot cover all required userland obligations.\n");
        if (unavailable_count != 0) {
            try stdout.print("  unsupported Lean-fragment proof targets: {d}\n", .{unavailable_count});
        }
        if (unmatched != 0) {
            try stdout.print("  unmatched UNKNOWN prepared rows: {d}\n", .{unmatched});
        }
        if (automatic_fragment_unavailable_count != 0) {
            try stdout.print(
                "  obligations requiring an external Lean certificate: {d}\n",
                .{automatic_fragment_unavailable_count},
            );
        }
        return .{ .artifact_emission_authorized = false, .certificate_emitted = false };
    }
    if (targets.items.len == 0) {
        return .{ .artifact_emission_authorized = true, .certificate_emitted = false };
    }

    context.obligations_source = try renderObligationsSource(
        allocator,
        context.merged_result.set,
        context.generated_namespace,
    );
    const proof_source = try buildAutomaticProofSource(
        allocator,
        context.generated_namespace,
        context.merged_result.set,
        targets.items,
    );
    defer allocator.free(proof_source);

    const rows = try allocator.alloc(proof_check.ProofRow, targets.items.len);
    defer allocator.free(rows);
    var theorem_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (theorem_names.items) |name| allocator.free(name);
        theorem_names.deinit(allocator);
    }
    for (targets.items, rows) |query, *row| {
        const theorem_name = try std.fmt.allocPrint(
            allocator,
            "Ora.CompilerAutomaticProof.Query_{d}.discharge_q{d}",
            .{ query.id, query.id },
        );
        try theorem_names.append(allocator, theorem_name);
        row.* = .{
            .query_id = query.id,
            .obligation_ids = query.obligation_ids,
            .assumption_ids = query.assumption_ids,
            .module_name = "Ora.CompilerAutomaticProof",
            .theorem_name = theorem_name,
            .path = null,
            .content_sha256 = null,
        };
    }

    var applied = proof_check.applyInlineProofRows(
        allocator,
        context.merged_result.set,
        rows,
        context.generated_namespace,
        context.obligations_source.?,
        proof_source,
        options.process_environ,
        stdout,
    ) catch |err| {
        if (proof_check.leanProofFailureReason(err)) |_| {
            try printAutomaticProofFailure(stdout, err);
            return .{ .artifact_emission_authorized = false, .certificate_emitted = false };
        }
        return err;
    };
    defer if (applied) |*result| result.deinit();
    if (applied == null) {
        try printAutomaticProofFailure(stdout, error.LeanProofCheckFailed);
        return .{ .artifact_emission_authorized = false, .certificate_emitted = false };
    }

    try writeCertificate(allocator, file_path, options, applied.?.certificate_json, stdout);
    const artifact_emission_authorized = try reportDecision(
        stdout,
        "Automatic Lean proof gate",
        applied.?.set,
    );
    if (artifact_emission_authorized) {
        try printAcceptedLoopCertificates(stdout, applied.?.set, rows);
    }
    return .{
        .artifact_emission_authorized = artifact_emission_authorized,
        .certificate_emitted = true,
    };
}

fn printAutomaticProofFailure(stdout: anytype, err: anyerror) !void {
    const reason = proof_check.leanProofFailureReason(err) orelse "lean_proof_rejected";
    try stdout.print("Automatic Lean proof gate failed: reason={s}.\n", .{reason});
    try stdout.writeAll("  No Lean certificate was produced; artifact emission remains blocked.\n");
    switch (err) {
        error.LeanProofParseError => try stdout.writeAll(
            "  The compiler-generated Lean proof source did not parse; this is a compiler proof-generator defect.\n",
        ),
        error.LeanProofUnsolvedGoals, error.LeanProofCheckFailed => try stdout.writeAll(
            "  Use the emitted obligation module and theorem shape from the following recipe, then pass the proof manifest with `--lean-proofs <proofs.json>`.\n" ++
                "  A manifest path containing user-authored proof source for the transient obligation module must set `inline_source=true`.\n",
        ),
        error.LeanProofTimeout => try stdout.writeAll(
            "  Lean exhausted its deterministic proof-search budget while checking the generated proof.\n",
        ),
        error.LeanToolUnavailable => try stdout.writeAll(
            "  The configured Lean/Lake toolchain was unavailable.\n",
        ),
        error.LeanProofAbnormalTermination => try stdout.writeAll(
            "  The Lean checker terminated abnormally.\n",
        ),
        else => unreachable,
    }
}

fn semanticSupportAvailable(support: obligation_to_lean.SemanticSupport) bool {
    return switch (support) {
        .supported => true,
        .unsupported => false,
    };
}

fn plainUnknownPreparedRows(report: z3_verification.SmtReportArtifacts) usize {
    const manifest = report.query_manifest orelse return 0;
    var count: usize = 0;
    for (manifest.rows) |row| {
        const status = row.result_status orelse continue;
        if (status != .unknown) continue;
        if (row.vacuous or row.vacuity_unknown or row.verified_with_caveats) continue;
        count += 1;
    }
    return count;
}

fn plainUnknownTargetCount(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.queries) |query| if (isPlainUnknownTarget(query)) {
        count += 1;
    };
    return count;
}

fn findObligation(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Obligation {
    for (set.obligations) |item| if (item.id == id) return item;
    return null;
}

fn findAssumption(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Assumption {
    for (set.assumptions) |item| if (item.id == id) return item;
    return null;
}

fn collectLoopStorageLocationFreeVars(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    location: obligation.LoopStorageLocationExpr,
    ids: *std.ArrayList(obligation.FreeVarId),
) !void {
    for (location.projections) |projection| switch (projection) {
        .field => {},
        .fixed_index => |fixed| try set.collectFormulaFreeVars(
            allocator,
            fixed.index.value.formula,
            ids,
        ),
        .mapping_key => |typed| try set.collectFormulaFreeVars(
            allocator,
            typed.value.formula,
            ids,
        ),
    };
}

fn collectLoopSummaryFreeVars(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    ids: *std.ArrayList(obligation.FreeVarId),
    fuel: usize,
) !void {
    if (fuel == 0) return error.InvalidDependency;
    for (summary.init_formulas) |formula| try set.collectFormulaFreeVars(allocator, formula, ids);
    for (summary.invariant_formulas) |formula| try set.collectFormulaFreeVars(allocator, formula, ids);
    for (summary.edges) |edge| {
        for (edge.timeline) |event| switch (event.payload) {
            .call => return error.UnsupportedLoopCall,
            .path_assumption, .safety_checkpoint => |formulas| for (formulas.formulas) |formula| {
                try set.collectFormulaFreeVars(allocator, formula.formula, ids);
            },
            .summary_dependency => |dependency| {
                const child = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                    return error.InvalidDependency;
                try collectLoopSummaryFreeVars(allocator, set, child, ids, fuel - 1);
            },
            .storage_read => |read| {
                try collectLoopStorageLocationFreeVars(allocator, set, read.location, ids);
            },
            .indexed_region_read => |read| {
                try set.collectFormulaFreeVars(allocator, read.index.formula, ids);
            },
            .storage_write => |write| {
                try collectLoopStorageLocationFreeVars(allocator, set, write.location, ids);
                try set.collectFormulaFreeVars(allocator, write.value.formula, ids);
            },
        };
        for (edge.target_formulas) |formula| try set.collectFormulaFreeVars(allocator, formula, ids);
        for (edge.next_state) |binding| try set.collectFormulaFreeVars(allocator, binding.value, ids);
    }
    for (summary.post_formulas) |formula| try set.collectFormulaFreeVars(allocator, formula, ids);
}

fn collectQueryFreeVars(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) ![]obligation.FreeVarId {
    var ids: std.ArrayList(obligation.FreeVarId) = .empty;
    errdefer ids.deinit(allocator);
    for (query.assumption_ids) |id| {
        const assumption = findAssumption(set, id) orelse return error.InvalidDependency;
        if (assumption.formula) |formula| try set.collectFormulaFreeVars(allocator, formula, &ids);
    }
    for (query.obligation_ids) |id| {
        const item = findObligation(set, id) orelse return error.InvalidDependency;
        if (obligation.kindFormula(item.kind)) |formula| try set.collectFormulaFreeVars(allocator, formula, &ids);
    }
    if (query.kind == .loop_induction) {
        const summary_id = query.loop_summary_id orelse return error.InvalidDependency;
        const loop_summary = obligation.findById(set.loop_summaries, summary_id) orelse
            return error.InvalidDependency;
        try collectLoopSummaryFreeVars(
            allocator,
            set,
            loop_summary,
            &ids,
            set.loop_summaries.len + 1,
        );
    }
    return try ids.toOwnedSlice(allocator);
}

fn writeFreeVarId(writer: anytype, id: obligation.FreeVarId) !void {
    try writer.print("{{ file_id := {d}, pattern_id := {d} }}", .{ id.file_id, id.pattern_id });
}

fn writeWitnessEnv(writer: anytype, free_vars: []const obligation.FreeVarId) !void {
    if (free_vars.len == 0) return writer.writeAll("Env.empty");
    for (free_vars, 0..) |_, index| {
        try writer.writeByte('(');
        if (index == 0) try writer.writeAll("Env.empty");
    }
    for (free_vars, 0..) |free_var, index| {
        try writer.writeAll(".setFree ");
        try writeFreeVarId(writer, free_var);
        try writer.print(" (.u256 (BitVec.ofNat 256 {d})))", .{index});
    }
}

const FreeVarEqualityForm = enum {
    local_haves,
    attributed_theorems,
};

fn writeFreeVarEqualityFacts(
    writer: anytype,
    free_vars: []const obligation.FreeVarId,
    form: FreeVarEqualityForm,
) !void {
    for (free_vars, 0..) |lhs, lhs_index| for (free_vars, 0..) |rhs, rhs_index| {
        switch (form) {
            .local_haves => try writer.print("  have h_auto_free_{d}_{d} : ((", .{ lhs_index, rhs_index }),
            .attributed_theorems => try writer.print("private theorem auto_free_{d}_{d} : ((", .{ lhs_index, rhs_index }),
        }
        try writeFreeVarId(writer, lhs);
        try writer.writeAll(" : FreeVarId) == ");
        try writeFreeVarId(writer, rhs);
        try writer.print(") = {s} := by rfl\n", .{if (obligation.freeVarIdEql(lhs, rhs)) "true" else "false"});
    };
    if (form == .local_haves) return;
    try writer.writeAll("attribute [local ora_loop_data]");
    for (free_vars, 0..) |_, lhs| for (free_vars, 0..) |_, rhs| {
        try writer.print(" auto_free_{d}_{d}", .{ lhs, rhs });
    };
    try writer.writeByte('\n');
}

fn writeAutomaticSimpSet(writer: anytype, free_var_count: usize) !void {
    try writer.writeAll(
        \\[
        \\      assumptionsDenoteInEnv,
        \\      assumptionsDenoteInEnv?,
        \\      assumptionAnd?,
        \\      assumptionDenotesInEnv?,
        \\      obligationDenotesInEnv,
        \\      obligationDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      denoteValue?,
        \\      effectFrameGoalDenotes?,
        \\      resourceGoalDenotes?,
        \\      resourceGoalAmount?,
        \\      resourceGoalSource?,
        \\      resourceGoalDestination?,
        \\      emittedManifest,
        \\      emittedTerms,
        \\      emittedAssumptions,
        \\      emittedObligations,
        \\      Env.setFree,
        \\      Env.lookupVar,
        \\      Env.lookupFree,
        \\      lookupFreeBinding,
        \\      Env.lookupBound,
        \\      Env.lookupPlace,
        \\      Env.lookupEntryPlace,
        \\      Env.pushBound,
        \\      Value.eqProp?,
        \\      BinderRef.isU256,
        \\      BoundVarRef.isU256,
        \\      TyRef.isU256,
        \\      TyRef.isI256,
        \\      TyRef.isU256Carrier,
        \\      compilerTypeIdU256,
        \\      compilerTypeIdI256,
        \\      compilerTypeIdBool,
        \\      Ora.Spec.expectedCompilerTypeIdU256,
        \\      Ora.Spec.expectedCompilerTypeIdI256,
        \\      Ora.Spec.expectedCompilerTypeIdBool,
        \\      U256.sle,
        \\      U256.slt,
        \\      U256.sge,
        \\      U256.sgt
    );
    for (0..free_var_count) |lhs| for (0..free_var_count) |rhs| {
        try writer.print(",\n      h_auto_free_{d}_{d}", .{ lhs, rhs });
    };
    try writer.writeAll(
        \\
        \\    ]
    );
}

fn formulaTermId(formula: obligation.FormulaRef) ?obligation.TermId {
    return switch (formula) {
        .term => |id| id,
        .origin_value => null,
    };
}

fn termAt(set: obligation.ObligationSet, id: obligation.TermId) ?obligation.Term {
    if (id >= set.terms.len) return null;
    return set.terms[id];
}

fn decimalLiteralCanBeEmitted(value: []const u8) bool {
    if (value.len == 0) return false;
    for (value) |byte| if (!std.ascii.isDigit(byte)) return false;
    return true;
}

const AutomaticIntegerShape = struct {
    width: u16,
    signed: bool,

    fn eql(lhs: AutomaticIntegerShape, rhs: AutomaticIntegerShape) bool {
        return lhs.width == rhs.width and lhs.signed == rhs.signed;
    }
};

const AutomaticLoopProofReason = enum {
    automatic_query_shape_unsupported,
    automatic_assumptions_require_external_certificate,
    automatic_nested_summary_requires_external_certificate,
    automatic_call_relation_unsupported,
    automatic_storage_timeline_requires_external_certificate,
    loop_indexed_region_read_formula_unreviewed,
    automatic_control_shape_unsupported,
    automatic_formula_reference_unsupported,
    automatic_term_kind_unsupported,
    automatic_integer_type_missing,
    automatic_integer_type_unsupported,
    automatic_mixed_integer_shapes,
    automatic_foreign_variable,
    automatic_initial_state_dependency,
    automatic_nonlinear_arithmetic,
};

const AutomaticLoopPlan = struct {
    summary: obligation.LoopSummaryRow,
};

const AutomaticLoopPlanResult = union(enum) {
    supported: AutomaticLoopPlan,
    unsupported: AutomaticLoopProofReason,
};

fn automaticLoopProofReasonName(reason: AutomaticLoopProofReason) []const u8 {
    return @tagName(reason);
}

fn loopSummaryHasDependencies(summary: obligation.LoopSummaryRow) bool {
    for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => return true,
        else => {},
    };
    return false;
}

fn appendAutomaticSummaryClosure(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    rows: *std.ArrayList(obligation.LoopSummaryRow),
    summary: obligation.LoopSummaryRow,
    fuel: usize,
) !void {
    if (fuel == 0) return error.InvalidDependency;
    for (rows.items) |existing| if (existing.id == summary.id) return;
    try rows.append(allocator, summary);
    for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => |dependency| {
            const child = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                return error.InvalidDependency;
            try appendAutomaticSummaryClosure(allocator, set, rows, child, fuel - 1);
        },
        else => {},
    };
}

fn automaticSummaryClosure(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
) ![]obligation.LoopSummaryRow {
    var rows: std.ArrayList(obligation.LoopSummaryRow) = .empty;
    errdefer rows.deinit(allocator);
    try appendAutomaticSummaryClosure(
        allocator,
        set,
        &rows,
        summary,
        set.loop_summaries.len + 1,
    );
    return rows.toOwnedSlice(allocator);
}

fn automaticIntegerShape(ty: obligation.TypeRef) ?AutomaticIntegerShape {
    const info = switch (ty) {
        .compiler_type_id => |id| ora_types.builtin.integerInfoByComptimeTypeId(id),
        .spelling => |name| ora_types.builtin.integerInfoByName(name),
    } orelse return null;
    return .{ .width = info.width, .signed = info.signed };
}

fn automaticTermType(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
) ?obligation.TypeRef {
    const term = termAt(set, term_id) orelse return null;
    return switch (term) {
        .int_lit => |literal| literal.ty,
        .variable => |variable| switch (variable) {
            .free => |free| free.ty,
            .bound => |bound| bound.ty,
        },
        .binary => |binary| binary.ty,
        else => null,
    };
}

fn automaticVariableKnown(
    summary: obligation.LoopSummaryRow,
    id: obligation.FreeVarId,
) bool {
    for (summary.variables) |variable| {
        if (variable.id) |candidate| if (obligation.freeVarIdEql(candidate, id)) return true;
    }
    for (summary.context_variables) |variable| {
        if (variable.id) |candidate| if (obligation.freeVarIdEql(candidate, id)) return true;
    }
    return false;
}

fn automaticDependencyVariable(
    set: obligation.ObligationSet,
    edge: obligation.LoopEdgeRow,
    id: obligation.FreeVarId,
) ?obligation.LoopVariable {
    for (edge.timeline) |event| switch (event.payload) {
        .storage_read => |read| if (obligation.freeVarIdEql(read.result, id)) {
            return .{
                .index = 0,
                .id = read.result,
                .name = "storage_read",
                .ty = read.location.value_ty,
            };
        },
        .indexed_region_read => |read| if (obligation.freeVarIdEql(read.result, id)) {
            return .{
                .index = 0,
                .id = read.result,
                .name = "indexed_region_read",
                .ty = read.element_ty,
            };
        },
        .summary_dependency => |dependency| {
            const child = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                continue;
            for (child.variables) |variable| {
                const candidate = variable.id orelse continue;
                if (obligation.freeVarIdEql(candidate, id)) return variable;
            }
        },
        else => {},
    };
    return null;
}

fn automaticVariableKnownOnEdge(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: ?obligation.LoopEdgeRow,
    id: obligation.FreeVarId,
) bool {
    if (automaticVariableKnown(summary, id)) return true;
    return if (edge) |row| automaticDependencyVariable(set, row, id) != null else false;
}

const AutomaticValueShapeResult = union(enum) {
    supported: AutomaticIntegerShape,
    unsupported: AutomaticLoopProofReason,
};

fn automaticValueShapeInScope(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: ?obligation.LoopEdgeRow,
    term_id: obligation.TermId,
    fuel: u32,
) AutomaticValueShapeResult {
    if (fuel == 0) return .{ .unsupported = .automatic_term_kind_unsupported };
    const term = termAt(set, term_id) orelse
        return .{ .unsupported = .automatic_formula_reference_unsupported };
    switch (term) {
        .int_lit => |literal| {
            const ty = literal.ty orelse
                return .{ .unsupported = .automatic_integer_type_missing };
            const shape = automaticIntegerShape(ty) orelse
                return .{ .unsupported = .automatic_integer_type_unsupported };
            if (!decimalLiteralCanBeEmitted(literal.value)) {
                return .{ .unsupported = .automatic_term_kind_unsupported };
            }
            return .{ .supported = shape };
        },
        .variable => |variable| switch (variable) {
            .bound => return .{ .unsupported = .automatic_term_kind_unsupported },
            .free => |free| {
                if (!automaticVariableKnownOnEdge(set, summary, edge, free.id)) {
                    return .{ .unsupported = .automatic_foreign_variable };
                }
                const ty = free.ty orelse
                    return .{ .unsupported = .automatic_integer_type_missing };
                return .{ .supported = automaticIntegerShape(ty) orelse
                    return .{ .unsupported = .automatic_integer_type_unsupported } };
            },
        },
        .binary => |binary| {
            if (binary.op != .add and binary.op != .sub and
                binary.op != .bit_and and binary.op != .bit_xor and binary.op != .div)
            {
                return .{ .unsupported = switch (binary.op) {
                    .mul, .pow, .shl, .shr, .mod => .automatic_nonlinear_arithmetic,
                    else => .automatic_term_kind_unsupported,
                } };
            }
            const ty = binary.ty orelse
                return .{ .unsupported = .automatic_integer_type_missing };
            const result_shape = automaticIntegerShape(ty) orelse
                return .{ .unsupported = .automatic_integer_type_unsupported };
            if (binary.op == .div and result_shape.signed)
                return .{ .unsupported = .automatic_nonlinear_arithmetic };
            const lhs = automaticValueShapeInScope(set, summary, edge, binary.lhs, fuel - 1);
            const lhs_shape = switch (lhs) {
                .supported => |shape| shape,
                .unsupported => |reason| return .{ .unsupported = reason },
            };
            const rhs = automaticValueShapeInScope(set, summary, edge, binary.rhs, fuel - 1);
            const rhs_shape = switch (rhs) {
                .supported => |shape| shape,
                .unsupported => |reason| return .{ .unsupported = reason },
            };
            if (!result_shape.eql(lhs_shape) or !result_shape.eql(rhs_shape)) {
                return .{ .unsupported = .automatic_mixed_integer_shapes };
            }
            return .{ .supported = result_shape };
        },
        else => return .{ .unsupported = .automatic_term_kind_unsupported },
    }
}

fn automaticValueShape(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    term_id: obligation.TermId,
    fuel: u32,
) AutomaticValueShapeResult {
    return automaticValueShapeInScope(set, summary, null, term_id, fuel);
}

fn automaticFormulaReasonInScope(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: ?obligation.LoopEdgeRow,
    formula: obligation.FormulaRef,
    fuel: u32,
) ?AutomaticLoopProofReason {
    const term_id = formulaTermId(formula) orelse return .automatic_formula_reference_unsupported;
    if (fuel == 0) return .automatic_term_kind_unsupported;
    const term = termAt(set, term_id) orelse return .automatic_formula_reference_unsupported;
    return switch (term) {
        .bool_lit => null,
        .unary => |unary| if (unary.op == .not)
            automaticFormulaReasonInScope(
                set,
                summary,
                edge,
                .{ .term = unary.operand },
                fuel - 1,
            )
        else
            .automatic_term_kind_unsupported,
        .binary => |binary| switch (binary.op) {
            .eq, .ne, .lt, .le, .gt, .ge, .slt, .sle, .sgt, .sge => blk: {
                const lhs = automaticValueShapeInScope(
                    set,
                    summary,
                    edge,
                    binary.lhs,
                    fuel - 1,
                );
                const lhs_shape = switch (lhs) {
                    .supported => |shape| shape,
                    .unsupported => |reason| break :blk reason,
                };
                const rhs = automaticValueShapeInScope(
                    set,
                    summary,
                    edge,
                    binary.rhs,
                    fuel - 1,
                );
                const rhs_shape = switch (rhs) {
                    .supported => |shape| shape,
                    .unsupported => |reason| break :blk reason,
                };
                if (!lhs_shape.eql(rhs_shape)) break :blk .automatic_mixed_integer_shapes;
                break :blk null;
            },
            .and_, .or_, .implies => automaticFormulaReasonInScope(
                set,
                summary,
                edge,
                .{ .term = binary.lhs },
                fuel - 1,
            ) orelse automaticFormulaReasonInScope(
                set,
                summary,
                edge,
                .{ .term = binary.rhs },
                fuel - 1,
            ),
            else => .automatic_term_kind_unsupported,
        },
        else => .automatic_term_kind_unsupported,
    };
}

fn automaticFormulaReason(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formula: obligation.FormulaRef,
    fuel: u32,
) ?AutomaticLoopProofReason {
    return automaticFormulaReasonInScope(set, summary, null, formula, fuel);
}

fn automaticValueReason(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formula: obligation.FormulaRef,
) ?AutomaticLoopProofReason {
    const term_id = formulaTermId(formula) orelse return .automatic_formula_reference_unsupported;
    return switch (automaticValueShape(set, summary, term_id, 256)) {
        .supported => null,
        .unsupported => |reason| reason,
    };
}

fn automaticValueReasonOnEdge(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: obligation.LoopEdgeRow,
    formula: obligation.FormulaRef,
) ?AutomaticLoopProofReason {
    const term_id = formulaTermId(formula) orelse return .automatic_formula_reference_unsupported;
    return switch (automaticValueShapeInScope(set, summary, edge, term_id, 256)) {
        .supported => null,
        .unsupported => |reason| reason,
    };
}

fn automaticVariableIndex(
    variables: []const obligation.LoopVariable,
    id: obligation.FreeVarId,
) ?usize {
    for (variables, 0..) |variable, index| {
        const candidate = variable.id orelse continue;
        if (obligation.freeVarIdEql(candidate, id)) return index;
    }
    return null;
}

fn automaticValueUsesLoopVariable(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    term_id: obligation.TermId,
    fuel: u32,
) bool {
    if (fuel == 0) return true;
    const term = termAt(set, term_id) orelse return true;
    return switch (term) {
        .variable => |variable| switch (variable) {
            .bound => true,
            .free => |free| automaticVariableIndex(summary.variables, free.id) != null,
        },
        .binary => |binary| automaticValueUsesLoopVariable(
            set,
            summary,
            binary.lhs,
            fuel - 1,
        ) or automaticValueUsesLoopVariable(
            set,
            summary,
            binary.rhs,
            fuel - 1,
        ),
        .int_lit => false,
        else => true,
    };
}

fn timelineHasStorageReadBeforeSafetyCheckpoint(events: []const obligation.LoopTimelineEvent) bool {
    var later_checkpoint = false;
    var index = events.len;
    while (index > 0) {
        index -= 1;
        switch (events[index].payload) {
            .safety_checkpoint => later_checkpoint = true,
            .storage_read => if (later_checkpoint) return true,
            else => {},
        }
    }
    return false;
}

// Called only after term/ID validation. This permits an automatic attempt;
// every assumption still enters both prover queries and the complete witness.
// An inherited requires is not a same-block fact: never manufacture such rows.
fn automaticLoopAssumptionsSupported(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    assumption_ids: []const obligation.Id,
) bool {
    if (set.loopHeaderFactsMatchAssumptions(summary, assumption_ids)) return true;
    if (summary.header_facts.len != 0 or summary.owner != .statement) return false;
    for (set.loop_summaries) |parent| {
        if (parent.id >= summary.id or parent.owner != .statement or
            !std.mem.eql(u8, parent.owner.statement.function_name, summary.owner.statement.function_name) or
            !parent.projectionSupported() or parent.query_ids.induction.len != 1) continue;
        const query = obligation.findById(set.queries, parent.query_ids.induction[0]) orelse continue;
        if (!std.mem.eql(obligation.Id, query.assumption_ids, assumption_ids) or
            !set.loopHeaderFactsMatchAssumptions(parent, assumption_ids)) continue;
        var captured = true;
        for (parent.header_facts) |fact| {
            if (fact.source_kind != .function_requires) continue;
            for (fact.stable_values) |value| {
                const expected = automaticIntegerShape(value.ty) orelse return false;
                const found = for (summary.context_variables) |variable| {
                    const id = variable.id orelse continue;
                    if (!obligation.freeVarIdEql(id, value.id)) continue;
                    const actual = automaticIntegerShape(variable.ty) orelse return false;
                    if (expected.eql(actual)) break true;
                } else false;
                if (!found) captured = false;
            }
        }
        if (!captured) continue;
        for (parent.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
            .summary_dependency => |dependency| if (dependency.summary_id == summary.id) return true,
            else => {},
        };
    }
    return false;
}

fn automaticLoopPlan(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) AutomaticLoopPlanResult {
    if (query.kind != .loop_induction or query.loop_summary_id == null) {
        return .{ .unsupported = .automatic_query_shape_unsupported };
    }
    const summary = obligation.findById(set.loop_summaries, query.loop_summary_id.?) orelse
        return .{ .unsupported = .automatic_query_shape_unsupported };
    set.validateTermReferences() catch return .{ .unsupported = .automatic_query_shape_unsupported };
    set.validateIdReferences() catch return .{ .unsupported = .automatic_query_shape_unsupported };
    if (set.loopSummaryArtifactAuthorityBlockReason(summary) ==
        .loop_indexed_region_read_formula_unreviewed)
    {
        return .{ .unsupported = .loop_indexed_region_read_formula_unreviewed };
    }
    if (!automaticLoopAssumptionsSupported(set, summary, query.assumption_ids))
        return .{ .unsupported = .automatic_assumptions_require_external_certificate };
    if (!summary.projectionSupported() or summary.edges.len < 2) {
        return .{ .unsupported = .automatic_control_shape_unsupported };
    }
    if (summary.variables.len == 0 or summary.init_formulas.len != summary.variables.len) {
        return .{ .unsupported = .automatic_control_shape_unsupported };
    }
    var has_backedge = false;
    var has_exit = false;
    for (summary.edges) |edge| switch (edge.target orelse
        return .{ .unsupported = .automatic_control_shape_unsupported }) {
        .backedge => has_backedge = true,
        .successor, .function_return, .error_return => has_exit = true,
        .explicit_revert => return .{ .unsupported = .automatic_control_shape_unsupported },
    };
    if (!has_backedge or !has_exit) {
        return .{ .unsupported = .automatic_control_shape_unsupported };
    }
    for (summary.variables) |variable| {
        if (automaticIntegerShape(variable.ty) == null) {
            return .{ .unsupported = .automatic_integer_type_unsupported };
        }
    }
    for (summary.context_variables) |variable| {
        if (automaticIntegerShape(variable.ty) == null) {
            return .{ .unsupported = .automatic_integer_type_unsupported };
        }
    }
    for (summary.init_formulas) |formula| {
        if (automaticValueReason(set, summary, formula)) |reason| {
            return .{ .unsupported = reason };
        }
        const term_id = formulaTermId(formula) orelse
            return .{ .unsupported = .automatic_formula_reference_unsupported };
        if (automaticValueUsesLoopVariable(set, summary, term_id, 256)) {
            return .{ .unsupported = .automatic_initial_state_dependency };
        }
    }
    for (summary.invariant_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
        return .{ .unsupported = reason };
    };
    for (summary.edges) |edge| {
        if (timelineHasStorageReadBeforeSafetyCheckpoint(edge.timeline)) {
            return .{ .unsupported = .automatic_storage_timeline_requires_external_certificate };
        }
        var dependency_count: usize = 0;
        for (edge.timeline) |event| switch (event.payload) {
            .call => return .{ .unsupported = .automatic_call_relation_unsupported },
            .summary_dependency => |dependency| {
                dependency_count += 1;
                if (dependency_count > 1) return .{
                    .unsupported = .automatic_nested_summary_requires_external_certificate,
                };
                const child = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                    return .{ .unsupported = .automatic_nested_summary_requires_external_certificate };
                if (!child.projectionSupported()) {
                    return .{ .unsupported = .automatic_nested_summary_requires_external_certificate };
                }
                for (child.variables) |variable| {
                    if (variable.id == null or automaticIntegerShape(variable.ty) == null) {
                        return .{ .unsupported = .automatic_nested_summary_requires_external_certificate };
                    }
                }
                if (loopSummaryHasDependencies(child)) {
                    return .{ .unsupported = .automatic_nested_summary_requires_external_certificate };
                }
                var matching_targets: usize = 0;
                for (child.edges) |child_edge| {
                    if (child_edge.target == dependency.target) matching_targets += 1;
                }
                if (matching_targets != 1) {
                    return .{ .unsupported = .automatic_nested_summary_requires_external_certificate };
                }
            },
            .path_assumption => |path| for (path.formulas) |formula| {
                if (automaticFormulaReason(set, summary, formula.formula, 256)) |reason| {
                    return .{ .unsupported = reason };
                }
            },
            .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |formula| {
                if (automaticFormulaReasonInScope(set, summary, edge, formula.formula, 256)) |reason| {
                    return .{ .unsupported = reason };
                }
            },
            .storage_write => |write| {
                if (automaticValueReasonOnEdge(set, summary, edge, write.value.formula)) |reason| {
                    return .{ .unsupported = reason };
                }
            },
            .storage_read => {},
            .indexed_region_read => |read| {
                if (automaticValueReasonOnEdge(set, summary, edge, read.index.formula)) |reason| {
                    return .{ .unsupported = reason };
                }
                if (automaticIntegerShape(read.element_ty) == null) {
                    return .{ .unsupported = .automatic_integer_type_unsupported };
                }
            },
        };
        if (edge.next_state.len != summary.variables.len) {
            return .{ .unsupported = .automatic_control_shape_unsupported };
        }
        for (edge.next_state, 0..) |binding, index| {
            if (binding.variable_index != index) {
                return .{ .unsupported = .automatic_control_shape_unsupported };
            }
            const expected = summary.variables[index].id orelse
                return .{ .unsupported = .automatic_control_shape_unsupported };
            const actual = binding.target orelse
                return .{ .unsupported = .automatic_control_shape_unsupported };
            if (!obligation.freeVarIdEql(expected, actual)) {
                return .{ .unsupported = .automatic_control_shape_unsupported };
            }
        }
        for (edge.target_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
            return .{ .unsupported = reason };
        };
        for (edge.next_state) |binding| if (automaticValueReasonOnEdge(set, summary, edge, binding.value)) |reason| {
            return .{ .unsupported = reason };
        };
    }
    for (summary.post_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
        return .{ .unsupported = reason };
    };
    return .{ .supported = .{ .summary = summary } };
}

fn writeAutomaticLoopProof(
    allocator: std.mem.Allocator,
    writer: anytype,
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    plan: AutomaticLoopPlan,
    free_vars: []const obligation.FreeVarId,
) !void {
    const term_ids = try automaticLoopTermIds(allocator, set, query);
    defer allocator.free(term_ids);
    const summary = plan.summary;
    const nested = loopSummaryHasDependencies(summary);

    try writer.writeAll("section\n");
    try writer.print(
        "private abbrev autoSummary : Ora.Loop.SummaryRow := emittedLoopSummary_{d}\n" ++
            "attribute [local ora_loop_row] autoSummary",
        .{summary.id},
    );
    if (nested) {
        const summary_rows = try automaticSummaryClosure(allocator, set, summary);
        defer allocator.free(summary_rows);
        // The denotation's fuel uses the whole emitted list, not this query's
        // dependency closure. Match writeLoopSummaryDefinitions' row filter.
        var emitted_summary_count: usize = 0;
        for (set.loop_summaries) |row| {
            if (row.projectionSupported()) emitted_summary_count += 1;
        }
        for (summary_rows) |row| try writer.print(" emittedLoopSummary_{d}", .{row.id});
        try writer.writeByte('\n');
        if (set.terms.len > obligation_to_lean.lean_term_chunk_size) {
            try writer.writeAll("attribute [local irreducible] emittedManifest\n");
        }
        try writer.writeAll(
            "attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological\n",
        );
        try writeTermLookupAttributes(writer, "ora_loop_data", term_ids);
        try writeFreeVarEqualityFacts(writer, free_vars, .attributed_theorems);
        try writer.print(
            "private theorem autoNestedSummaryCount : " ++
                "emittedLoopSummaries.length = {d} := by rfl\n" ++
                "attribute [local ora_loop_nested] autoNestedSummaryCount\n",
            .{emitted_summary_count},
        );
        for (summary_rows) |row| {
            try writer.print(
                "private theorem autoNestedFind_{d} : " ++
                    "Ora.Loop.findSummaryRow? emittedLoopSummaries {d} = " ++
                    "some emittedLoopSummary_{d} := by rfl\n" ++
                    "attribute [local ora_loop_nested] autoNestedFind_{d}\n",
                .{ row.id, row.id, row.id, row.id },
            );
        }
        for (summary_rows) |row| {
            try writer.print(
                "private theorem autoNestedSupported_{d} : " ++
                    "emittedLoopSummary_{d}.supportedIn emittedManifest " ++
                    "emittedLoopSummaries = true := by\n" ++
                    "  ora_loop_support_in emittedManifest emittedLoopSummaries " ++
                    "emittedLoopSummary_{d}\n" ++
                    "attribute [local ora_loop_nested] autoNestedSupported_{d}\n",
                .{ row.id, row.id, row.id, row.id },
            );
        }
        var reverse_index = summary_rows.len;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const row = summary_rows[reverse_index];
            if (row.id == summary.id) continue;
            try writer.print(
                "private theorem autoNestedDenotes_{d} (base : Env) :\n" ++
                    "    (Ora.Loop.denoteLoopSummaryWithFuel? {d} " ++
                    "emittedManifest emittedLoopSummaries " ++
                    "base emittedLoopSummary_{d}).isSome = true := by\n" ++
                    "  unfold Ora.Loop.denoteLoopSummaryWithFuel?\n" ++
                    "  simp only [Ora.Loop.denoteLoopSummaryWithResolver?, " ++
                    "autoNestedSupported_{d}, if_true]\n" ++
                    "  simp only [ora_loop_row, ora_loop_nested, " ++
                    "Ora.Loop.denoteEdgesWithResolver?, " ++
                    "Ora.Loop.denoteEdgeWithResolver?, " ++
                    "Ora.Loop.denoteSummaryDependencies?, " ++
                    "Ora.Loop.EdgeRow.path, " ++
                    "Ora.Loop.EdgeRow.safetyCheckpoints, " ++
                    "Ora.Loop.EdgeRow.summaryDependencies, " ++
                    "Ora.Loop.timelinePathView, " ++
                    "Ora.Loop.timelineSafetyView, " ++
                    "Ora.Loop.timelineDependencyView, " ++
                    "Ora.Loop.denoteSummaryDependency?, Ora.Loop.findSummaryRow?, " ++
                    "Ora.Loop.denoteControlTarget?, Option.bind_eq_bind, " ++
                    "Option.pure_def, Option.some_bind, Option.none_bind, Option.isSome, " ++
                    "if_pos, if_false, Bool.false_eq_true]\n" ++
                    "attribute [local ora_loop_nested] autoNestedDenotes_{d}\n",
                .{ row.id, emitted_summary_count, row.id, row.id, row.id },
            );
        }
    } else {
        try writer.print(" emittedLoopSummary_{d}\n", .{summary.id});
        if (set.terms.len > obligation_to_lean.lean_term_chunk_size) {
            try writer.writeAll("attribute [local irreducible] emittedManifest\n");
        }
        try writer.writeAll(
            "attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological\n",
        );
        try writeTermLookupAttributes(writer, "ora_loop_data", term_ids);
        try writeFreeVarEqualityFacts(writer, free_vars, .attributed_theorems);
        try writer.writeAll(
            "private theorem autoShape : autoSummary.shapeSupported = true := by decide +kernel\n" ++
                "attribute [local ora_loop_shape] autoShape\n",
        );
        try writer.print(
            "private theorem autoSummarySupported : " ++
                "emittedLoopSummary_{d}.supported emittedManifest = true := by\n" ++
                "  change autoSummary.supported emittedManifest = true\n" ++
                "  ora_loop_support\n" ++
                "attribute [local ora_loop_supported] autoSummarySupported\n",
            .{summary.id},
        );
    }
    try writer.writeAll("private abbrev autoAssumptions : List AssumptionRow := [");
    for (query.assumption_ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("emittedAssumptionRow_{d}", .{id});
    }
    try writer.writeAll("]\nattribute [local ora_loop_row] autoAssumptions");
    for (query.assumption_ids) |id| try writer.print(" emittedAssumptionRow_{d}", .{id});
    try writer.writeByte('\n');
    if (!nested and obligation_to_lean.loopSummaryUsesDirectStorageCertificate(summary)) {
        try writer.writeAll(
            \\private abbrev autoEdgeCompatible (env : Env) (edge : Ora.Loop.EdgeRow) : Prop :=
            \\  Ora.Loop.DirectStorageEdgeCompatible emittedManifest env autoSummary.variables autoSummary.post edge
            \\    (fun locals => Ora.Loop.stateReady autoSummary.variables locals ∧
            \\      Ora.Loop.headerFactsHoldInEnv emittedManifest env autoSummary.headerFacts ∧
            \\      Ora.Loop.formulasHoldInState emittedManifest env autoSummary.variables locals autoSummary.invariants)
            \\
        );
        const context = "(env : Env)\n" ++
            "    (hAssumptions : assumptionsDenoteInEnv emittedManifest env autoAssumptions)\n" ++
            "    (hContext : Ora.Loop.contextReady env autoSummary.contextVariables)";
        for (summary.edges, 0..) |_, index| try writer.print(
            "private theorem autoEdgeCompatible_{d} {s} :\n" ++
                "    autoEdgeCompatible env (autoSummary.edges.get ⟨{d}, by decide⟩) := by\n" ++
                "  ora_loop_prepare_context hContext\n" ++
                "  ora_loop_direct_edge_compatible autoSummary env\n",
            .{ index, context, index },
        );
        try writer.print(
            "private theorem autoEdgesCompatible {s} :\n" ++
                "    ∀ edge, edge ∈ autoSummary.edges → autoEdgeCompatible env edge := by\n" ++
                "  intro edge hMember\n" ++
                "  simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false] at hMember\n" ++
                "  rcases hMember with ",
            .{context},
        );
        for (summary.edges, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl" else " | rfl");
        try writer.writeByte('\n');
        for (summary.edges, 0..) |_, index| try writer.print("  · exact autoEdgeCompatible_{d} env hAssumptions hContext\n", .{index});
        try writer.writeAll(
            "set_option maxHeartbeats 500000 in\n" ++
                "private theorem autoScalarAuthorization : " ++
                "Ora.Loop.loopProofAuthorizationFromAssumptions " ++
                "emittedManifest autoAssumptions autoSummary := by\n" ++
                "  set_option maxRecDepth 4096 in ora_loop_auto_plain\n",
        );
        try writer.print(
            "theorem discharge_q{d} : emittedQuery_{d} := by\n" ++
                "  ora_loop_auto_direct_storage_from autoScalarAuthorization autoEdgesCompatible\n" ++
                "end\n",
            .{ query.id, query.id },
        );
    } else {
        try writer.print(
            "theorem discharge_q{d} : emittedQuery_{d} := by ora_loop_auto\n" ++
                "end\n",
            .{ query.id, query.id },
        );
    }
}

fn markAutomaticTermClosure(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    marked: []bool,
) !void {
    if (term_id >= set.terms.len or term_id >= marked.len) return error.InvalidDependency;
    if (marked[term_id]) return;
    marked[term_id] = true;
    switch (set.terms[term_id]) {
        .old => |operand| try markAutomaticTermClosure(set, operand, marked),
        .unary => |unary| try markAutomaticTermClosure(set, unary.operand, marked),
        .binary => |binary| {
            try markAutomaticTermClosure(set, binary.lhs, marked);
            try markAutomaticTermClosure(set, binary.rhs, marked);
        },
        .refinement_predicate => |predicate| {
            try markAutomaticTermClosure(set, predicate.value, marked);
            for (predicate.args) |arg| try markAutomaticTermClosure(set, arg, marked);
        },
        .quantified => |quantified| {
            if (quantified.condition) |condition| try markAutomaticTermClosure(set, condition, marked);
            try markAutomaticTermClosure(set, quantified.body, marked);
        },
        .bool_lit,
        .int_lit,
        .variable,
        .result,
        .place_read,
        => {},
    }
}

fn markAutomaticFormulaClosure(
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    marked: []bool,
) !void {
    switch (formula) {
        .term => |term_id| try markAutomaticTermClosure(set, term_id, marked),
        .origin_value => return error.InvalidDependency,
    }
}

fn markAutomaticSummaryClosure(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    marked: []bool,
    fuel: usize,
) !void {
    if (fuel == 0) return error.InvalidDependency;
    for (summary.header_facts) |fact| try markAutomaticFormulaClosure(set, fact.formula, marked);
    for (summary.init_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
    for (summary.invariant_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
    for (summary.edges) |edge| {
        for (edge.timeline) |event| switch (event.payload) {
            .call => return error.UnsupportedLoopCall,
            .path_assumption, .safety_checkpoint => |formulas| for (formulas.formulas) |formula| {
                try markAutomaticFormulaClosure(set, formula.formula, marked);
            },
            .storage_write => |write| try markAutomaticFormulaClosure(set, write.value.formula, marked),
            .indexed_region_read => |read| try markAutomaticFormulaClosure(set, read.index.formula, marked),
            .summary_dependency => |dependency| {
                const child = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                    return error.InvalidDependency;
                try markAutomaticSummaryClosure(set, child, marked, fuel - 1);
            },
            .storage_read => {},
        };
        for (edge.target_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
        for (edge.next_state) |binding| try markAutomaticFormulaClosure(set, binding.value, marked);
    }
    for (summary.post_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
}

fn automaticLoopTermIds(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) ![]obligation.TermId {
    const summary_id = query.loop_summary_id orelse return error.InvalidDependency;
    const summary = obligation.findById(set.loop_summaries, summary_id) orelse
        return error.InvalidDependency;
    const marked = try allocator.alloc(bool, set.terms.len);
    defer allocator.free(marked);
    @memset(marked, false);

    for (query.assumption_ids) |assumption_id| {
        const assumption = obligation.findById(set.assumptions, assumption_id) orelse
            return error.InvalidDependency;
        if (assumption.formula) |formula| try markAutomaticFormulaClosure(set, formula, marked);
    }
    try markAutomaticSummaryClosure(set, summary, marked, set.loop_summaries.len + 1);

    var count: usize = 0;
    for (marked) |is_marked| if (is_marked) {
        count += 1;
    };
    const ids = try allocator.alloc(obligation.TermId, count);
    var next: usize = 0;
    for (marked, 0..) |is_marked, term_id| {
        if (!is_marked) continue;
        ids[next] = @intCast(term_id);
        next += 1;
    }
    return ids;
}

fn writeTermLookupAttributes(
    writer: anytype,
    attribute_name: []const u8,
    term_ids: []const obligation.TermId,
) !void {
    try writer.writeAll(
        "attribute [local ",
    );
    try writer.writeAll(attribute_name);
    try writer.writeAll("]");
    try writer.writeAll(" emitted_terms_length");
    for (term_ids) |term_id| {
        try writer.print(" emitted_term_{d}_lookup", .{term_id});
    }
    try writer.writeByte('\n');
}

fn buildAutomaticProofSource(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    set: obligation.ObligationSet,
    queries: []const obligation.VerificationQuery,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll("namespace Ora.CompilerAutomaticProof\n");
    try writer.writeAll("open Ora.Obligation\n");
    try writer.print("open {s}\n\n", .{generated_namespace});
    for (queries) |query| {
        try writer.print("namespace Query_{d}\n", .{query.id});
        const free_vars = try collectQueryFreeVars(allocator, set, query);
        defer allocator.free(free_vars);
        if (query.kind == .loop_induction) {
            const plan = switch (automaticLoopPlan(set, query)) {
                .supported => |value| value,
                .unsupported => return error.InvalidDependency,
            };
            try writeAutomaticLoopProof(allocator, writer, set, query, plan, free_vars);
            try writer.print("\nend Query_{d}\n\n", .{query.id});
            continue;
        }
        try writer.print("theorem discharge_q{d} : emittedQuery_{d} := by\n", .{ query.id, query.id });
        try writeFreeVarEqualityFacts(writer, free_vars, .local_haves);
        try writer.print("  unfold emittedQuery_{d} obligationFollowsFromAssumptions\n", .{query.id});
        try writer.writeAll("  constructor\n  · refine ⟨");
        try writeWitnessEnv(writer, free_vars);
        try writer.writeAll(", ?_⟩\n    simp ");
        try writeAutomaticSimpSet(writer, free_vars.len);
        try writer.writeAll(" <;> try decide\n  · intro env hAssumptions\n    simp ");
        try writeAutomaticSimpSet(writer, free_vars.len);
        try writer.writeAll(" at hAssumptions ⊢\n    repeat intro\n    first\n    | assumption\n    | exact U256.ult_implies_ule _ _ (by assumption)\n    | exact stable_place_read_self_eq_denotes env _\n    | simp_all ");
        try writeAutomaticSimpSet(writer, free_vars.len);
        try writer.writeAll("\n\n");
        try writer.print("end Query_{d}\n\n", .{query.id});
    }
    try writer.writeAll("end Ora.CompilerAutomaticProof\n");
    return try out.toOwnedSlice();
}

const ProjectionSummary = struct {
    term: u32 = 0,
    origin_value: u32 = 0,

    fn addFormula(self: *ProjectionSummary, formula: obligation.FormulaRef) void {
        switch (formula) {
            .term => self.term +|= 1,
            .origin_value => self.origin_value +|= 1,
        }
    }

    fn addKind(self: *ProjectionSummary, kind: obligation.Kind) void {
        switch (kind) {
            .logical => |logical| self.addFormula(logical.formula),
            .runtime_guard => |guard| self.addFormula(guard.formula),
            .resource => |resource| if (resource.amount) |amount| self.addFormula(amount),
            else => {},
        }
    }

    fn total(self: ProjectionSummary) u32 {
        return self.term +| self.origin_value;
    }

    fn ratioBasisPoints(self: ProjectionSummary) u32 {
        if (self.total() == 0) return 0;
        return @intCast((@as(u64, self.term) * 10_000) / self.total());
    }

    fn hasOpaqueFormula(self: ProjectionSummary) bool {
        return self.origin_value != 0;
    }
};

fn projectionForQuery(set: obligation.ObligationSet, query: obligation.VerificationQuery) ProjectionSummary {
    var summary: ProjectionSummary = .{};
    for (query.assumption_ids) |id| {
        const assumption = findAssumption(set, id) orelse continue;
        if (assumption.formula) |formula| summary.addFormula(formula);
    }
    if (query.kind == .loop_induction) {
        const summary_id = query.loop_summary_id orelse return summary;
        const loop_summary = obligation.findById(set.loop_summaries, summary_id) orelse return summary;
        for (loop_summary.init_formulas) |formula| summary.addFormula(formula);
        for (loop_summary.invariant_formulas) |formula| summary.addFormula(formula);
        for (loop_summary.edges) |edge| {
            for (edge.timeline) |event| switch (event.payload) {
                .path_assumption, .safety_checkpoint => |formulas| for (formulas.formulas) |formula| {
                    summary.addFormula(formula.formula);
                },
                .storage_write => |write| summary.addFormula(write.value.formula),
                .indexed_region_read => |read| summary.addFormula(read.index.formula),
                .call => |invocation| for (invocation.arguments) |argument| summary.addFormula(argument.value.formula),
                .storage_read, .summary_dependency => {},
            };
            for (edge.target_formulas) |formula| summary.addFormula(formula);
            for (edge.next_state) |binding| summary.addFormula(binding.value);
        }
        for (loop_summary.post_formulas) |formula| summary.addFormula(formula);
        return summary;
    }
    for (query.obligation_ids) |id| {
        const item = findObligation(set, id) orelse continue;
        summary.addKind(item.kind);
    }
    return summary;
}

fn projectionForSet(set: obligation.ObligationSet) ProjectionSummary {
    var summary: ProjectionSummary = .{};
    for (set.assumptions) |assumption| if (assumption.formula) |formula| summary.addFormula(formula);
    for (set.obligations) |item| summary.addKind(item.kind);
    return summary;
}

fn printProjection(writer: anytype, summary: ProjectionSummary) !void {
    try writer.print(
        "formula projection: term={d}, origin_value={d}, total={d}, term_ratio_basis_points={d}",
        .{ summary.term, summary.origin_value, summary.total(), summary.ratioBasisPoints() },
    );
}

fn printTypeRef(writer: anytype, ty: obligation.TypeRef) !void {
    switch (ty) {
        .spelling => |text| try writer.print("`{s}`", .{text}),
        .compiler_type_id => |id| if (ora_types.builtin.lookupBuiltinByComptimeTypeId(id)) |spec|
            try writer.print("`{s}`", .{spec.source_name})
        else
            try writer.print("compiler_type_id:{d}", .{id}),
    }
}

fn printUnsupportedReason(writer: anytype, reason: obligation_to_lean.SemanticUnsupportedReason) !void {
    switch (reason) {
        .empty_query => try writer.writeAll("    reason: query has no obligation ids, so no Lean semantic proposition can be emitted\n"),
        .invalid_dependency => try writer.writeAll("    reason: query references a missing obligation or assumption row\n"),
        .unsupported_obligation_kind => try writer.writeAll("    reason: this obligation kind is not in the Lean semantic proof fragment yet\n"),
        .unsupported_effect_frame_relation => |relation| try writer.print("    reason: effect-frame relation `{s}` is not in the Lean semantic proof fragment yet\n", .{@tagName(relation)}),
        .unsupported_origin_value => try writer.writeAll("    reason: formula is still an MLIR origin_value; Lean proof export only supports projected Term formulas today\n"),
        .unsupported_term_kind => try writer.writeAll("    reason: this term shape is not in the Lean semantic proof fragment yet\n"),
        .missing_type => try writer.writeAll("    reason: term is missing type metadata required by the Lean semantic proof fragment\n"),
        .unsupported_type => |ty| {
            try writer.writeAll("    reason: unsupported Lean semantic type ");
            try printTypeRef(writer, ty);
            try writer.writeAll("; the Lean semantic proof fragment supports bool and registered Ora integer types\n");
        },
        .unsupported_comparison_width => try writer.writeAll("    reason: comparison operands do not have the same registered integer width\n"),
        .unknown_signedness => try writer.writeAll("    reason: signed comparison operand is missing compiler type-id signedness metadata\n"),
        .mixed_signedness => try writer.writeAll("    reason: comparison predicate signedness does not match both operand types\n"),
        .unsupported_arithmetic_width => try writer.writeAll("    reason: arithmetic operands do not have the same registered integer width\n"),
        .unknown_arithmetic_signedness => try writer.writeAll("    reason: arithmetic value is missing compiler type-id signedness metadata\n"),
        .mixed_arithmetic_signedness => try writer.writeAll("    reason: arithmetic operation signedness does not match the value type\n"),
        .missing_key_disjoint_evidence => try writer.writeAll("    reason: evidence-backed storage frame has no key-disjointness evidence rows\n"),
        .unsupported_key_disjoint_evidence_formula => try writer.writeAll("    reason: key-disjointness evidence must be a direct requires free-variable disequality in the Lean fragment\n"),
        .unsupported_key_disjoint_evidence_kind => try writer.writeAll("    reason: key-disjointness evidence kind is not supported by the Lean fragment\n"),
        .key_disjoint_evidence_type_unsupported => try writer.writeAll("    reason: key-disjointness evidence requires matching registered integer free-variable types\n"),
        .key_disjoint_evidence_owner_mismatch => try writer.writeAll("    reason: key-disjointness evidence references an assumption from a different owner\n"),
        .key_disjoint_evidence_path_mismatch => try writer.writeAll("    reason: key-disjointness evidence does not match the first differing parameter keys of the read/write paths\n"),
        .loop_summary_missing => try writer.writeAll("    reason: loop proof target has no owner-scoped loop summary\n"),
        .loop_summary_query_mismatch => try writer.writeAll("    reason: loop proof query identity does not match its owner-scoped loop summary\n"),
        .unsupported_loop_summary => |loop_reason| try writer.print("    reason: loop summary is outside the Lean scalar fragment: {s}\n", .{@tagName(loop_reason)}),
    }
}

fn writeObligationsArtifact(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    options: Options,
    source: []const u8,
) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    const filename = try artifact_catalog.filename(allocator, stem, .lean_obligations_source);
    defer allocator.free(filename);
    var write_path_buf: ?[]u8 = null;
    defer if (write_path_buf) |path| allocator.free(path);
    const write_path = if (options.output_dir) |dir| blk: {
        try std.Io.Dir.cwd().createDirPath(std.Io.Threaded.global_single_threaded.io(), dir);
        write_path_buf = try std.fs.path.join(allocator, &.{ dir, filename });
        break :blk write_path_buf.?;
    } else filename;
    try std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{
        .sub_path = write_path,
        .data = source,
    });
    if (options.artifact_display_dir) |dir| return std.fs.path.join(allocator, &.{ dir, filename });
    if (options.output_dir) |dir| return std.fs.path.join(allocator, &.{ dir, filename });
    return allocator.dupe(u8, filename);
}

fn printRecipeSource(writer: anytype, source: obligation.SourceRef) !void {
    if (source.file) |file| {
        try writer.writeAll(file);
        if (source.line != 0) try writer.print(":{d}", .{source.line});
        if (source.column != 0) try writer.print(":{d}", .{source.column});
    } else if (source.line != 0) {
        try writer.print("line {d}", .{source.line});
        if (source.column != 0) try writer.print(":{d}", .{source.column});
    } else try writer.writeAll("unknown source");
}

fn printIds(writer: anytype, ids: []const obligation.Id) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

test "loop blocker batch unsigned division eligibility retains type and operator boundaries" {
    const summary: obligation.LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "division" } },
        .source = .{},
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
    };
    for ([_]obligation.BinaryOp{ .div, .mod, .mul }) |op| {
        for ([_][]const u8{ "u64", "u256", "i256" }) |name| {
            const ty: obligation.TypeRef = .{ .spelling = name };
            const terms = [_]obligation.Term{
                .{ .int_lit = .{ .value = "1", .ty = ty } },
                .{ .int_lit = .{ .value = "2", .ty = ty } },
                .{ .binary = .{ .op = op, .lhs = 0, .rhs = 1, .ty = ty } },
            };
            const shape = automaticValueShape(.{ .terms = &terms }, summary, 2, 4);
            try std.testing.expectEqual(op == .div and name[0] == 'u', shape == .supported);
            if (shape == .unsupported) try std.testing.expectEqual(
                AutomaticLoopProofReason.automatic_nonlinear_arithmetic,
                shape.unsupported,
            );
        }
    }
}

test "automatic proof failures retain stable named diagnostics" {
    const cases = [_]struct {
        err: anyerror,
        reason: []const u8,
        detail: []const u8,
    }{
        .{ .err = error.LeanProofParseError, .reason = "lean_tactic_parse_error", .detail = "proof-generator defect" },
        .{ .err = error.LeanProofUnsolvedGoals, .reason = "lean_unsolved_goals", .detail = "proof manifest" },
        .{ .err = error.LeanProofTimeout, .reason = "lean_tactic_timeout", .detail = "proof-search budget" },
        .{ .err = error.LeanToolUnavailable, .reason = "lean_tool_unavailable", .detail = "toolchain was unavailable" },
        .{ .err = error.LeanProofAbnormalTermination, .reason = "lean_abnormal_termination", .detail = "terminated abnormally" },
    };

    for (cases) |case| {
        var output = std.Io.Writer.Allocating.init(std.testing.allocator);
        defer output.deinit();
        try printAutomaticProofFailure(&output.writer, case.err);
        try std.testing.expect(std.mem.indexOf(u8, output.written(), case.reason) != null);
        try std.testing.expect(std.mem.indexOf(u8, output.written(), case.detail) != null);
        try std.testing.expect(std.mem.indexOf(u8, output.written(), "artifact emission remains blocked") != null);
    }
}

test "automatic loop proof source delegates to repository tactic deterministically" {
    const u256_ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const context_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "0", .ty = u256_ty } },
        .{ .variable = .{ .free = .{ .id = loop_id, .name = "i", .ty = u256_ty } } },
        .{ .variable = .{ .free = .{ .id = context_id, .name = "n", .ty = u256_ty } } },
        .{ .binary = .{ .op = .lt, .lhs = 1, .rhs = 2, .ty = u256_ty } },
        .{ .binary = .{ .op = .le, .lhs = 1, .rhs = 2, .ty = u256_ty } },
        .{ .int_lit = .{ .value = "1", .ty = u256_ty } },
        .{ .binary = .{ .op = .add, .lhs = 1, .rhs = 5, .ty = u256_ty } },
        .{ .binary = .{ .op = .lt, .lhs = 6, .rhs = 1, .ty = u256_ty } },
        .{ .unary = .{ .op = .not, .operand = 7 } },
        .{ .unary = .{ .op = .not, .operand = 3 } },
    };
    const variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = u256_ty,
    }};
    const context_variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = context_id,
        .name = "n",
        .ty = u256_ty,
    }};
    const backedge_bindings = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop:add-one",
        .value = .{ .term = 6 },
    }};
    const exit_bindings = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop:identity",
        .value = .{ .term = 1 },
    }};
    const backedge_path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 3 } }};
    const exit_path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 9 } }};
    const safety_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 8 } }};
    const backedge_timeline = [_]obligation.LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "loop:backedge/path",
            .payload = .{ .path_assumption = .{ .formulas = &backedge_path_exprs } },
        },
        .{
            .id = 1,
            .structural_key = "op:checked-add",
            .payload = .{ .safety_checkpoint = .{ .formulas = &safety_exprs } },
        },
    };
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "loop:exit/path",
        .payload = .{ .path_assumption = .{ .formulas = &exit_path_exprs } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "loop:backedge",
            .timeline = &backedge_timeline,
            .next_state = &backedge_bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "loop:exit",
            .timeline = &exit_timeline,
            .next_state = &exit_bindings,
            .target = .successor,
        },
    };
    const summaries = [_]obligation.LoopSummaryRow{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "countTo" } },
        .source = .{},
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .context_variables = &context_variables,
        .variables = &variables,
        .init_formulas = &.{.{ .term = 0 }},
        .invariant_formulas = &.{.{ .term = 4 }},
        .edges = &edges,
        .projection_completeness = .complete,
        .post_formulas = &.{.{ .term = 4 }},
        .query_ids = .{
            .base = &.{11},
            .induction = &.{10},
            .body_safety = &.{ 12, 13 },
            .step = &.{14},
            .post = &.{15},
        },
    }};
    const digest = @as([32]u8, @splat(0xA5));
    const owner: obligation.Owner = .{ .function = .{ .name = "countTo" } };
    const queries = [_]obligation.VerificationQuery{
        .{ .id = 10, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .proof_requirement = .lean_certificate, .kind = .loop_induction, .loop_summary_id = 1, .loop_vc_sha256 = digest },
        .{ .id = 11, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .backend = .z3, .kind = .loop_base, .loop_summary_id = 1, .loop_vc_sha256 = digest },
        .{ .id = 12, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .backend = .z3, .kind = .loop_body_safety, .loop_summary_id = 1, .loop_edge_id = 0, .loop_vc_sha256 = digest },
        .{ .id = 13, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .backend = .z3, .kind = .loop_body_safety, .loop_summary_id = 1, .loop_edge_id = 1, .loop_vc_sha256 = digest },
        .{ .id = 14, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .backend = .z3, .kind = .loop_invariant_step, .loop_summary_id = 1, .loop_edge_id = 0, .loop_vc_sha256 = digest },
        .{ .id = 15, .owner = owner, .source = .{}, .phase = .ora_mlir, .origin = .source, .backend = .z3, .kind = .loop_invariant_post, .loop_summary_id = 1, .loop_edge_id = 1, .loop_vc_sha256 = digest },
    };
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .queries = &queries,
        .loop_summaries = &summaries,
    };

    const first = try buildAutomaticProofSource(
        std.testing.allocator,
        "Ora.Generated.Test",
        set,
        &queries,
    );
    defer std.testing.allocator.free(first);
    const second = try buildAutomaticProofSource(
        std.testing.allocator,
        "Ora.Generated.Test",
        set,
        &queries,
    );
    defer std.testing.allocator.free(second);
    try std.testing.expectEqualStrings(first, second);
    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        first,
        1,
        "theorem discharge_q10 : emittedQuery_10 := by ora_loop_auto",
    ));
    try std.testing.expect(!std.mem.containsAtLeast(
        u8,
        first,
        1,
        "Ora.Loop.bound_state_ready",
    ));

    const storage_backedge_timeline = [_]obligation.LoopTimelineEvent{
        backedge_timeline[0],
        backedge_timeline[1],
        .{
            .id = 2,
            .structural_key = "op:store",
            .payload = .{ .storage_write = .{
                .location = .{ .root = "values", .value_ty = u256_ty },
                .value = .{ .formula = .{ .term = 1 } },
            } },
        },
    };
    var storage_edges = edges;
    storage_edges[0].timeline = &storage_backedge_timeline;
    var storage_summary = summaries[0];
    storage_summary.edges = &storage_edges;
    const storage_set: obligation.ObligationSet = .{
        .terms = &terms,
        .queries = &queries,
        .loop_summaries = &.{storage_summary},
    };
    const storage_source = try buildAutomaticProofSource(
        std.testing.allocator,
        "Ora.Generated.Test",
        storage_set,
        &queries,
    );
    defer std.testing.allocator.free(storage_source);
    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        storage_source,
        1,
        "private theorem autoScalarAuthorization",
    ));
    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        storage_source,
        1,
        "ora_loop_auto_direct_storage_from autoScalarAuthorization",
    ));

    // Inherited eligibility uses the actual parent dependency, not merely a
    // same-function summary elsewhere in the manifest. Both queries keep IDs.
    const inherited_terms = terms ++ [_]obligation.Term{
        .{ .binary = .{ .op = .ge, .lhs = 2, .rhs = 5, .ty = u256_ty } },
        .{ .int_lit = .{ .value = "4", .ty = u256_ty } },
        .{ .binary = .{ .op = .le, .lhs = 2, .rhs = 11, .ty = u256_ty } },
    };
    const stable = [_]obligation.LoopHeaderFactValue{.{ .id = context_id, .ty = u256_ty }};
    var facts: [2]obligation.LoopHeaderFactRow = undefined;
    var assumptions: [3]obligation.Assumption = undefined;
    for (&facts, 0..) |*fact, index| {
        const key = ([_][]const u8{ "requires:lower", "requires:upper" })[index];
        const term = ([_]obligation.TermId{ 10, 12 })[index];
        fact.* = .{
            .id = @intCast(index),
            .structural_key = key,
            .formula = .{ .term = term },
            .source_kind = .function_requires,
            .source = .{ .structural_key = key, .block_key = "entry", .position = @intCast(index) },
            .target = .{ .structural_key = "parent", .block_key = "entry", .position = 4 },
            .stable_values = &stable,
            .assumption_id = @intCast(21 + index),
        };
        assumptions[index] = .{
            .id = @intCast(21 + index),
            .owner = owner,
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .requires,
            .formula = .{ .term = term },
        };
    }
    assumptions[2] = assumptions[0];
    assumptions[2].id = 23;
    var parent = summaries[0];
    parent.id = 0;
    parent.owner = .{ .statement = .{ .function_name = "countTo", .ordinal = 0 } };
    parent.header_facts = &facts;
    parent.query_ids = .{ .base = &.{111}, .induction = &.{110}, .body_safety = &.{ 112, 113 }, .step = &.{114}, .post = &.{115} };
    const parent_timeline = backedge_timeline ++ [_]obligation.LoopTimelineEvent{.{
        .id = 2,
        .structural_key = "child",
        .payload = .{ .summary_dependency = .{
            .id = 0,
            .structural_key = "child",
            .summary_id = 1,
            .loop_vc_sha256 = digest,
            .target = .successor,
            .write_footprint_complete = true,
        } },
    }};
    var parent_edges = edges;
    parent_edges[0].timeline = &parent_timeline;
    parent.edges = &parent_edges;
    var child = summaries[0];
    child.owner = .{ .statement = .{ .function_name = "countTo", .ordinal = 1 } };
    var inherited_queries = queries ++ queries;
    for (&inherited_queries, 0..) |*query, index| {
        query.assumption_ids = &.{ 21, 22 };
        if (index >= queries.len) {
            query.id += 100;
            query.loop_summary_id = 0;
        }
    }
    const changed_context = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = context_id,
        .name = "n",
        .ty = .{ .compiler_type_id = 3 },
    }};
    const Case = enum {
        inherited,
        no_dependency,
        wrong_function,
        missing_context,
        wrong_input_type,
        wrong_formula,
        wrong_assumption_owner,
        reordered_ids,
        extra_ids,
        missing_parent_query,
        unmatched_parent,
    };
    for (std.meta.tags(Case)) |case| {
        errdefer std.debug.print("inherited assumption case: {s}\n", .{@tagName(case)});
        var rows = [_]obligation.LoopSummaryRow{ parent, child };
        var case_queries = inherited_queries;
        var case_assumptions = assumptions;
        switch (case) {
            .inherited => {},
            .no_dependency => rows[0].edges = &edges,
            .wrong_function => rows[0].owner = .{ .statement = .{ .function_name = "other", .ordinal = 0 } },
            .missing_context => rows[1].context_variables = &.{},
            .wrong_input_type => rows[1].context_variables = &changed_context,
            .wrong_formula => case_assumptions[0].formula = .{ .term = 12 },
            .wrong_assumption_owner => case_assumptions[0].owner = .{ .function = .{ .name = "other" } },
            .reordered_ids => for (case_queries[0..queries.len]) |*query| {
                query.assumption_ids = &.{ 22, 21 };
            },
            .extra_ids => for (case_queries[0..queries.len]) |*query| {
                query.assumption_ids = &.{ 21, 22, 23 };
            },
            .missing_parent_query => rows[0].query_ids.induction = &.{999},
            .unmatched_parent => rows[0].header_facts = &.{},
        }
        const inherited_set: obligation.ObligationSet = .{
            .terms = &inherited_terms,
            .assumptions = &case_assumptions,
            .queries = &case_queries,
            .loop_summaries = &rows,
        };
        if (case != .missing_parent_query) {
            try inherited_set.validateTermReferences();
            try inherited_set.validateIdReferences();
        }
        const result = automaticLoopPlan(inherited_set, case_queries[0]);
        switch (result) {
            .supported => try std.testing.expectEqual(Case.inherited, case),
            .unsupported => |reason| {
                try std.testing.expect(case != .inherited);
                try std.testing.expectEqual(
                    if (case == .missing_parent_query) AutomaticLoopProofReason.automatic_query_shape_unsupported else AutomaticLoopProofReason.automatic_assumptions_require_external_certificate,
                    reason,
                );
            },
        }
    }
}

test "automatic proof environment includes storage projection keys" {
    const allocator = std.testing.allocator;
    const u256_ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const key_id: obligation.FreeVarId = .{ .file_id = 7, .pattern_id = 9 };
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = key_id, .name = "key", .ty = u256_ty } } },
        .{ .int_lit = .{ .value = "0", .ty = u256_ty } },
    };
    const projections = [_]obligation.LoopStorageProjection{.{
        .mapping_key = .{
            .value = .{ .formula = .{ .term = 0 } },
            .ty = u256_ty,
        },
    }};
    const timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "op:store",
        .payload = .{ .storage_write = .{
            .location = .{
                .root = "values",
                .projections = &projections,
                .value_ty = u256_ty,
            },
            .value = .{ .formula = .{ .term = 1 } },
        } },
    }};
    const edges = [_]obligation.LoopEdgeRow{.{
        .id = 0,
        .structural_key = "body",
        .timeline = &timeline,
    }};
    const summaries = [_]obligation.LoopSummaryRow{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "storeAt" } },
        .source = .{},
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .edges = &edges,
    }};
    const query: obligation.VerificationQuery = .{
        .id = 2,
        .owner = .{ .function = .{ .name = "storeAt" } },
        .source = .{},
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .loop_induction,
        .loop_summary_id = 1,
    };
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .queries = &.{query},
        .loop_summaries = &summaries,
    };

    const free_vars = try collectQueryFreeVars(allocator, set, query);
    defer allocator.free(free_vars);
    try std.testing.expectEqual(@as(usize, 1), free_vars.len);
    try std.testing.expect(obligation.freeVarIdEql(key_id, free_vars[0]));
}

test "automatic dependency variables are scoped to their declaring edge" {
    const parent_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const child_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const parent_variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = parent_id,
        .ty = .{ .spelling = "u256" },
    }};
    const child_variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = child_id,
        .ty = .{ .spelling = "u256" },
    }};
    const dependency: obligation.LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "op:2/scf.while",
        .summary_id = 2,
        .loop_vc_sha256 = @as([32]u8, @splat(0)),
        .target = .successor,
    };
    const dependency_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = dependency.structural_key,
        .payload = .{ .summary_dependency = dependency },
    }};
    const edge: obligation.LoopEdgeRow = .{
        .id = 0,
        .structural_key = "body/straight-line-yield",
        .timeline = &dependency_timeline,
    };
    const summaries = [_]obligation.LoopSummaryRow{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "run" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .loop_source_op_id = 1,
            .loop_kind = .scf_while,
            .variables = &parent_variables,
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "run" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .loop_source_op_id = 2,
            .loop_kind = .scf_while,
            .variables = &child_variables,
        },
    };
    const terms = [_]obligation.Term{.{ .variable = .{ .free = .{
        .id = child_id,
        .name = "inner",
        .ty = .{ .spelling = "u256" },
    } } }};
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .loop_summaries = &summaries,
    };

    switch (automaticValueShape(set, summaries[0], 0, 16)) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try std.testing.expectEqual(
            AutomaticLoopProofReason.automatic_foreign_variable,
            reason,
        ),
    }
    switch (automaticValueShapeInScope(set, summaries[0], edge, 0, 16)) {
        .unsupported => return error.TestUnexpectedResult,
        .supported => |shape| try std.testing.expect(shape.eql(.{
            .width = 256,
            .signed = false,
        })),
    }
}

test "coordinator outcome cannot authorize runtime-check erasure" {
    try std.testing.expect(@hasField(Outcome, "artifact_emission_authorized"));
    try std.testing.expect(!@hasField(Outcome, "proven_guard_ids"));
    try std.testing.expect(!@hasField(Outcome, "guard_erasure_authorized"));
    try std.testing.expect(!@hasField(Outcome, "resource_runtime_checks_proved"));
}

test "Lean targets reject vacuous and inconclusive Z3 rows" {
    const base: obligation.VerificationQuery = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "test" } },
        .source = .{},
        .phase = .report,
        .origin = .source,
        .backend = .z3,
        .kind = .obligation,
        .obligation_ids = &.{2},
        .result = .{ .status = .unknown },
    };
    try std.testing.expect(isPlainUnknownTarget(base));
    var loop_owned = base;
    loop_owned.loop_summary_id = 7;
    try std.testing.expect(!isPlainUnknownTarget(loop_owned));
    var vacuous = base;
    vacuous.result.?.vacuous = true;
    try std.testing.expect(!isPlainUnknownTarget(vacuous));
    var inconclusive = base;
    inconclusive.result.?.vacuity_unknown = true;
    try std.testing.expect(!isPlainUnknownTarget(inconclusive));
}

test "runtime loop certificate coverage accepts no loops and one exact target" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try std.testing.expect(try runtimeLoopCertificateTargetsComplete(.{}, &output.writer));
    try std.testing.expectEqual(@as(usize, 0), output.written().len);

    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = .{ .spelling = "u256" },
    }};
    const next_state = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const backedge_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "body/straight-line-yield/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "guard/false-successor/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/straight-line-yield",
            .timeline = &backedge_timeline,
            .next_state = &next_state,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &exit_timeline,
            .next_state = &next_state,
            .target = .successor,
        },
    };
    const induction_ids = [_]obligation.Id{20};
    const summaries = [_]obligation.LoopSummaryRow{.{
        .id = 10,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .{ .file = "loop.ora", .line = 4 },
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &edges,
        .projection_completeness = .complete,
        .query_ids = .{ .induction = &induction_ids },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 20,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .{ .file = "loop.ora", .line = 4 },
        .phase = .ora_mlir,
        .origin = .source,
        .proof_requirement = .lean_certificate,
        .kind = .loop_induction,
        .loop_summary_id = 10,
    }};

    try std.testing.expect(try runtimeLoopCertificateTargetsComplete(
        .{ .queries = &queries, .loop_summaries = &summaries },
        &output.writer,
    ));
    try std.testing.expectEqual(@as(usize, 0), output.written().len);
}

test "runtime loop certificate coverage rejects named exclusions and missing targets" {
    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = .{ .spelling = "u256" },
    }};
    const next_state = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const backedge_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "body/straight-line-yield/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "guard/false-successor/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/straight-line-yield",
            .timeline = &backedge_timeline,
            .next_state = &next_state,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &exit_timeline,
            .next_state = &next_state,
            .target = .successor,
        },
    };
    const induction_ids = [_]obligation.Id{20};
    const missing_invariant = [_]obligation.LoopUnsupportedReason{.loop_invariant_missing};

    var unsupported_output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer unsupported_output.deinit();
    const unsupported = [_]obligation.LoopSummaryRow{.{
        .id = 10,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .{ .file = "unsupported.ora", .line = 4 },
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .unsupported_reasons = &missing_invariant,
    }};
    try std.testing.expect(!try runtimeLoopCertificateTargetsComplete(
        .{ .loop_summaries = &unsupported },
        &unsupported_output.writer,
    ));
    try std.testing.expect(std.mem.indexOf(
        u8,
        unsupported_output.written(),
        "unsupported.ora:4 (run) excluded_by=loop_invariant_missing",
    ) != null);

    var missing_target_output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer missing_target_output.deinit();
    const missing_target = [_]obligation.LoopSummaryRow{.{
        .id = 10,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .{ .file = "missing-target.ora", .line = 4 },
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &edges,
        .projection_completeness = .complete,
        .query_ids = .{ .induction = &induction_ids },
    }};
    try std.testing.expect(!try runtimeLoopCertificateTargetsComplete(
        .{ .loop_summaries = &missing_target },
        &missing_target_output.writer,
    ));
    try std.testing.expect(std.mem.indexOf(
        u8,
        missing_target_output.written(),
        "missing-target.ora:4 (run) excluded_by=loop_certificate_target_missing",
    ) != null);
}
