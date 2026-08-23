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
        context.merged_result.set,
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
        if (projection.hasOpaqueFormula() or
            !semanticSupportAvailable(obligation_to_lean.querySemanticSupport(context.merged_result.set, query))) continue;
        try stdout.print("  - query: emittedQuery_{d} (", .{query.id});
        try printRecipeSource(stdout, query.source);
        try stdout.print(")\n    theorem shape: theorem discharge_q{d} : {s}.emittedQuery_{d} := by ...\n    proof row: query_id={d}, obligation_ids=", .{
            query.id,
            context.generated_namespace,
            query.id,
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
        context.merged_result.set,
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
    try obligation_to_lean.writeModule(&out.writer, set, .{
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

    var applied = try proof_check.applyProofRows(
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
    return .{
        .artifact_emission_authorized = try reportDecision(
            stdout,
            "Lean proof gate",
            decided_set,
        ),
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

fn printBlockingDiagnostics(stdout: anytype, set: obligation.ObligationSet) !void {
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
    const result = query.result orelse return false;
    return result.status == .unknown and
        !result.degraded and
        !result.vacuous and
        !result.vacuity_unknown;
}

fn isLeanProofTarget(query: obligation.VerificationQuery) bool {
    if (query.kind == .loop_induction) {
        return query.artifact_policy == .blocks_verified_artifacts and
            query.proof_requirement == .lean_certificate and
            query.backend == .unspecified and
            query.result == null and
            query.loop_summary_id != null and
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
    return .{
        .artifact_emission_authorized = try reportDecision(
            stdout,
            "Automatic Lean proof gate",
            applied.?.set,
        ),
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

fn freeVarIdsEqual(lhs: obligation.FreeVarId, rhs: obligation.FreeVarId) bool {
    return lhs.file_id == rhs.file_id and lhs.pattern_id == rhs.pattern_id;
}

fn appendUniqueFreeVar(
    allocator: std.mem.Allocator,
    ids: *std.ArrayList(obligation.FreeVarId),
    id: obligation.FreeVarId,
) !void {
    for (ids.items) |existing| if (freeVarIdsEqual(existing, id)) return;
    try ids.append(allocator, id);
}

fn collectTermFreeVars(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    ids: *std.ArrayList(obligation.FreeVarId),
    fuel: u32,
) !void {
    if (fuel == 0) return error.LeanAutoProofTermCycle;
    if (term_id >= set.terms.len) return error.InvalidTermReference;
    switch (set.terms[term_id]) {
        .variable => |variable| switch (variable) {
            .free => |free| try appendUniqueFreeVar(allocator, ids, free.id),
            .bound => {},
        },
        .old => |operand| try collectTermFreeVars(allocator, set, operand, ids, fuel - 1),
        .unary => |unary| try collectTermFreeVars(allocator, set, unary.operand, ids, fuel - 1),
        .binary => |binary| {
            try collectTermFreeVars(allocator, set, binary.lhs, ids, fuel - 1);
            try collectTermFreeVars(allocator, set, binary.rhs, ids, fuel - 1);
        },
        .refinement_predicate => |predicate| {
            try collectTermFreeVars(allocator, set, predicate.value, ids, fuel - 1);
            for (predicate.args) |arg| try collectTermFreeVars(allocator, set, arg, ids, fuel - 1);
        },
        .quantified => |quantified| {
            if (quantified.condition) |condition| try collectTermFreeVars(allocator, set, condition, ids, fuel - 1);
            try collectTermFreeVars(allocator, set, quantified.body, ids, fuel - 1);
        },
        .bool_lit, .int_lit, .result, .place_read => {},
    }
}

fn collectFormulaFreeVars(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    ids: *std.ArrayList(obligation.FreeVarId),
) !void {
    switch (formula) {
        .term => |term| try collectTermFreeVars(allocator, set, term, ids, 256),
        .origin_value => {},
    }
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
        if (assumption.formula) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
    }
    for (query.obligation_ids) |id| {
        const item = findObligation(set, id) orelse return error.InvalidDependency;
        if (obligation.kindFormula(item.kind)) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
    }
    if (query.kind == .loop_induction) {
        const summary_id = query.loop_summary_id orelse return error.InvalidDependency;
        const loop_summary = obligation.findById(set.loop_summaries, summary_id) orelse
            return error.InvalidDependency;
        for (loop_summary.init_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
        for (loop_summary.invariant_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
        for (loop_summary.edges) |edge| {
            for (edge.path_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
            for (edge.safety_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
            for (edge.next_state) |binding| try collectFormulaFreeVars(allocator, set, binding.value, &ids);
        }
        for (loop_summary.post_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
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

fn writeFreeVarEqualityFacts(writer: anytype, free_vars: []const obligation.FreeVarId) !void {
    for (free_vars, 0..) |lhs, lhs_index| for (free_vars, 0..) |rhs, rhs_index| {
        try writer.print("  have h_auto_free_{d}_{d} : ((", .{ lhs_index, rhs_index });
        try writeFreeVarId(writer, lhs);
        try writer.writeAll(" : FreeVarId) == ");
        try writeFreeVarId(writer, rhs);
        try writer.print(") = {s} := by rfl\n", .{if (freeVarIdsEqual(lhs, rhs)) "true" else "false"});
    };
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
    backedge_index: usize,
    successor_index: usize,
};

const AutomaticLoopPlanResult = union(enum) {
    supported: AutomaticLoopPlan,
    unsupported: AutomaticLoopProofReason,
};

fn automaticLoopProofReasonName(reason: AutomaticLoopProofReason) []const u8 {
    return @tagName(reason);
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

fn automaticValueShape(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    term_id: obligation.TermId,
    fuel: u32,
) union(enum) { supported: AutomaticIntegerShape, unsupported: AutomaticLoopProofReason } {
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
                if (!automaticVariableKnown(summary, free.id)) {
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
                binary.op != .bit_and and binary.op != .bit_xor)
            {
                return .{ .unsupported = switch (binary.op) {
                    .mul, .pow, .shl, .shr, .div, .mod => .automatic_nonlinear_arithmetic,
                    else => .automatic_term_kind_unsupported,
                } };
            }
            const ty = binary.ty orelse
                return .{ .unsupported = .automatic_integer_type_missing };
            const result_shape = automaticIntegerShape(ty) orelse
                return .{ .unsupported = .automatic_integer_type_unsupported };
            const lhs = automaticValueShape(set, summary, binary.lhs, fuel - 1);
            const lhs_shape = switch (lhs) {
                .supported => |shape| shape,
                .unsupported => |reason| return .{ .unsupported = reason },
            };
            const rhs = automaticValueShape(set, summary, binary.rhs, fuel - 1);
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

fn automaticFormulaReason(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formula: obligation.FormulaRef,
    fuel: u32,
) ?AutomaticLoopProofReason {
    const term_id = formulaTermId(formula) orelse return .automatic_formula_reference_unsupported;
    if (fuel == 0) return .automatic_term_kind_unsupported;
    const term = termAt(set, term_id) orelse return .automatic_formula_reference_unsupported;
    return switch (term) {
        .bool_lit => null,
        .unary => |unary| if (unary.op == .not)
            automaticFormulaReason(set, summary, .{ .term = unary.operand }, fuel - 1)
        else
            .automatic_term_kind_unsupported,
        .binary => |binary| switch (binary.op) {
            .eq, .ne, .lt, .le, .gt, .ge, .slt, .sle, .sgt, .sge => blk: {
                const lhs = automaticValueShape(set, summary, binary.lhs, fuel - 1);
                const lhs_shape = switch (lhs) {
                    .supported => |shape| shape,
                    .unsupported => |reason| break :blk reason,
                };
                const rhs = automaticValueShape(set, summary, binary.rhs, fuel - 1);
                const rhs_shape = switch (rhs) {
                    .supported => |shape| shape,
                    .unsupported => |reason| break :blk reason,
                };
                if (!lhs_shape.eql(rhs_shape)) break :blk .automatic_mixed_integer_shapes;
                break :blk null;
            },
            .and_, .or_, .implies => automaticFormulaReason(
                set,
                summary,
                .{ .term = binary.lhs },
                fuel - 1,
            ) orelse automaticFormulaReason(
                set,
                summary,
                .{ .term = binary.rhs },
                fuel - 1,
            ),
            else => .automatic_term_kind_unsupported,
        },
        else => .automatic_term_kind_unsupported,
    };
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

fn automaticLoopPlan(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) AutomaticLoopPlanResult {
    if (query.kind != .loop_induction or query.loop_summary_id == null) {
        return .{ .unsupported = .automatic_query_shape_unsupported };
    }
    if (query.assumption_ids.len != 0) {
        return .{ .unsupported = .automatic_assumptions_require_external_certificate };
    }
    const summary = obligation.findById(set.loop_summaries, query.loop_summary_id.?) orelse
        return .{ .unsupported = .automatic_query_shape_unsupported };
    if (!summary.projectionSupported() or summary.edges.len != 2) {
        return .{ .unsupported = .automatic_control_shape_unsupported };
    }
    if (summary.variables.len == 0 or summary.init_formulas.len != summary.variables.len) {
        return .{ .unsupported = .automatic_control_shape_unsupported };
    }
    var backedge_index: ?usize = null;
    var successor_index: ?usize = null;
    for (summary.edges, 0..) |edge, index| switch (edge.target orelse
        return .{ .unsupported = .automatic_control_shape_unsupported }) {
        .backedge => {
            if (backedge_index != null) {
                return .{ .unsupported = .automatic_control_shape_unsupported };
            }
            backedge_index = index;
        },
        .successor => {
            if (successor_index != null) {
                return .{ .unsupported = .automatic_control_shape_unsupported };
            }
            successor_index = index;
        },
        else => return .{ .unsupported = .automatic_control_shape_unsupported },
    };
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
        for (edge.path_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
            return .{ .unsupported = reason };
        };
        for (edge.safety_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
            return .{ .unsupported = reason };
        };
        for (edge.next_state) |binding| if (automaticValueReason(set, summary, binding.value)) |reason| {
            return .{ .unsupported = reason };
        };
    }
    for (summary.post_formulas) |formula| if (automaticFormulaReason(set, summary, formula, 256)) |reason| {
        return .{ .unsupported = reason };
    };
    return .{ .supported = .{
        .summary = summary,
        .backedge_index = backedge_index orelse
            return .{ .unsupported = .automatic_control_shape_unsupported },
        .successor_index = successor_index orelse
            return .{ .unsupported = .automatic_control_shape_unsupported },
    } };
}

fn writeAutomaticLeanShape(writer: anytype, shape: AutomaticIntegerShape) !void {
    try writer.print("(.{s} .w{d})", .{ if (shape.signed) "signed" else "unsigned", shape.width });
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

const AutomaticStateView = enum {
    current,
    following,
};

fn writeAutomaticStateBits(
    writer: anytype,
    view: AutomaticStateView,
    index: usize,
) !void {
    try writer.print(
        "auto_{s}_bits_{d}",
        .{ if (view == .current) "state" else "following", index },
    );
}

fn writeAutomaticValueExpression(
    writer: anytype,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    term_id: obligation.TermId,
    fuel: u32,
    state_view: AutomaticStateView,
) !void {
    if (fuel == 0) return error.LeanAutoProofTermCycle;
    const term = termAt(set, term_id) orelse return error.InvalidTermReference;
    switch (term) {
        .int_lit => |literal| {
            const ty = literal.ty orelse return error.InvalidDependency;
            const shape = automaticIntegerShape(ty) orelse return error.InvalidDependency;
            try writer.print("(BitVec.ofNat {d} {s})", .{ shape.width, literal.value });
        },
        .variable => |variable| switch (variable) {
            .bound => return error.InvalidDependency,
            .free => |free| {
                if (automaticVariableIndex(summary.variables, free.id)) |index| {
                    try writeAutomaticStateBits(writer, state_view, index);
                } else if (automaticVariableIndex(summary.context_variables, free.id)) |index| {
                    try writer.print("auto_context_bits_{d}", .{index});
                } else return error.InvalidDependency;
            },
        },
        .binary => |binary| {
            const ty = binary.ty orelse return error.InvalidDependency;
            const shape = automaticIntegerShape(ty) orelse return error.InvalidDependency;
            switch (binary.op) {
                .add, .sub => {
                    try writer.writeAll("(Ora.Integer.wrappingResult ");
                    try writeAutomaticLeanShape(writer, shape);
                    try writer.print(" .{s} ", .{@tagName(binary.op)});
                },
                .bit_and, .bit_xor => {
                    try writer.print(
                        "(Ora.Integer.{s}Result ",
                        .{if (binary.op == .bit_and) "bitAnd" else "bitXor"},
                    );
                    try writeAutomaticLeanShape(writer, shape);
                    try writer.writeByte(' ');
                },
                else => return error.InvalidDependency,
            }
            try writeAutomaticValueExpression(writer, set, summary, binary.lhs, fuel - 1, state_view);
            try writer.writeByte(' ');
            try writeAutomaticValueExpression(writer, set, summary, binary.rhs, fuel - 1, state_view);
            try writer.writeByte(')');
        },
        else => return error.InvalidDependency,
    }
}

fn writeAutomaticFormulaExpression(
    writer: anytype,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formula: obligation.FormulaRef,
    fuel: u32,
    state_view: AutomaticStateView,
) !void {
    if (fuel == 0) return error.LeanAutoProofTermCycle;
    const term_id = formulaTermId(formula) orelse return error.InvalidDependency;
    const term = termAt(set, term_id) orelse return error.InvalidTermReference;
    switch (term) {
        .bool_lit => |value| try writer.writeAll(if (value) "True" else "False"),
        .unary => |unary| {
            if (unary.op != .not) return error.InvalidDependency;
            try writer.writeAll("(¬ ");
            try writeAutomaticFormulaExpression(writer, set, summary, .{ .term = unary.operand }, fuel - 1, state_view);
            try writer.writeByte(')');
        },
        .binary => |binary| switch (binary.op) {
            .and_, .or_, .implies => {
                try writer.writeByte('(');
                try writeAutomaticFormulaExpression(writer, set, summary, .{ .term = binary.lhs }, fuel - 1, state_view);
                try writer.writeAll(switch (binary.op) {
                    .and_ => " ∧ ",
                    .or_ => " ∨ ",
                    .implies => " → ",
                    else => unreachable,
                });
                try writeAutomaticFormulaExpression(writer, set, summary, .{ .term = binary.rhs }, fuel - 1, state_view);
                try writer.writeByte(')');
            },
            .eq, .ne => {
                try writer.writeByte('(');
                if (binary.op == .ne) try writer.writeAll("¬ ");
                try writeAutomaticValueExpression(writer, set, summary, binary.lhs, fuel - 1, state_view);
                try writer.writeAll(" = ");
                try writeAutomaticValueExpression(writer, set, summary, binary.rhs, fuel - 1, state_view);
                try writer.writeByte(')');
            },
            .lt, .le, .gt, .ge, .slt, .sle, .sgt, .sge => {
                const ty = automaticTermType(set, binary.lhs) orelse return error.InvalidDependency;
                const shape = automaticIntegerShape(ty) orelse return error.InvalidDependency;
                const relation = switch (binary.op) {
                    .lt, .gt => "unsignedLt",
                    .le, .ge => "unsignedLe",
                    .slt, .sgt => "signedLt",
                    .sle, .sge => "signedLe",
                    else => unreachable,
                };
                try writer.print("(Ora.Integer.{s} (shape := ", .{relation});
                try writeAutomaticLeanShape(writer, shape);
                try writer.writeAll(") ");
                const reverse = binary.op == .gt or binary.op == .ge or
                    binary.op == .sgt or binary.op == .sge;
                try writeAutomaticValueExpression(
                    writer,
                    set,
                    summary,
                    if (reverse) binary.rhs else binary.lhs,
                    fuel - 1,
                    state_view,
                );
                try writer.writeByte(' ');
                try writeAutomaticValueExpression(
                    writer,
                    set,
                    summary,
                    if (reverse) binary.lhs else binary.rhs,
                    fuel - 1,
                    state_view,
                );
                try writer.writeByte(')');
            },
            else => return error.InvalidDependency,
        },
        else => return error.InvalidDependency,
    }
}

fn writeAutomaticIndent(writer: anytype, count: usize) !void {
    try writer.splatByteAll(' ', count);
}

fn writeAutomaticTermLookupItems(
    writer: anytype,
    term_ids: []const obligation.TermId,
) !void {
    try writer.writeAll("emitted_terms_length");
    for (term_ids) |term_id| try writer.print(", emitted_term_{d}_lookup", .{term_id});
}

fn writeAutomaticFreeEqualityItems(writer: anytype, free_var_count: usize) !void {
    for (0..free_var_count) |lhs| for (0..free_var_count) |rhs| {
        try writer.print(", h_auto_free_{d}_{d}", .{ lhs, rhs });
    };
}

fn writeAutomaticContextLookupItems(writer: anytype, context_count: usize) !void {
    for (0..context_count) |index| {
        try writer.print(", h_auto_context_lookup_{d}", .{index});
    }
}

fn writeAutomaticIntegerShapeItems(writer: anytype) !void {
    try writer.writeAll(
        "TyRef.integerShape?, " ++
            "integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, " ++
            "integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, " ++
            "integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, " ++
            "integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, " ++
            "integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, " ++
            "integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, " ++
            "integerShapeForCompilerTypeId_i256",
    );
}

fn writeAutomaticFormulaReduction(
    writer: anytype,
    set: obligation.ObligationSet,
    plan: AutomaticLoopPlan,
    formula: obligation.FormulaRef,
    term_ids: []const obligation.TermId,
    free_var_count: usize,
    state_view: AutomaticStateView,
    hypothesis: ?[]const u8,
    indent: usize,
) !void {
    if (hypothesis) |name| {
        try writeAutomaticIndent(writer, indent);
        try writer.print("have {s}_denoted : ", .{name});
        try writeAutomaticFormulaExpression(writer, set, plan.summary, formula, 256, state_view);
        try writer.writeAll(" := by\n");
        try writeAutomaticIndent(writer, indent + 2);
        try writer.writeAll("simpa only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, ");
        try writeAutomaticIntegerShapeItems(writer);
        try writeAutomaticFreeEqualityItems(writer, free_var_count);
        try writer.writeAll(
            ", formulaDenotes?, denoteFormula?, denoteValue?, " ++
                "IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, " ++
                "Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, " ++
                "Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, " ++
                "Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, " ++
                "Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, " ++
                "Env.lookupFree, lookupFreeBinding, ",
        );
        try writeAutomaticIntegerShapeItems(writer);
        try writer.writeAll(", ");
        try writeAutomaticTermLookupItems(writer, term_ids);
        try writeAutomaticFreeEqualityItems(writer, free_var_count);
        try writeAutomaticContextLookupItems(writer, plan.summary.context_variables.len);
        try writer.print(", if_pos, if_false, Bool.false_eq_true] using {s}\n", .{name});
        return;
    }

    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, ");
    try writeAutomaticIntegerShapeItems(writer);
    try writeAutomaticFreeEqualityItems(writer, free_var_count);
    try writer.writeAll(", if_pos]");
    try writer.writeByte('\n');

    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "simp only [formulaDenotes?, denoteFormula?, denoteValue?, " ++
            "IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, " ++
            "Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, " ++
            "Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, " ++
            "Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, " ++
            "Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, " ++
            "Env.lookupFree, lookupFreeBinding, ",
    );
    try writeAutomaticIntegerShapeItems(writer);
    try writer.writeAll(", ");
    try writeAutomaticTermLookupItems(writer, term_ids);
    try writeAutomaticFreeEqualityItems(writer, free_var_count);
    try writeAutomaticContextLookupItems(writer, plan.summary.context_variables.len);
    try writer.writeAll(", if_pos, if_false, Bool.false_eq_true]");
    try writer.writeByte('\n');

    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("change ");
    try writeAutomaticFormulaExpression(writer, set, plan.summary, formula, 256, state_view);
    try writer.writeByte('\n');
}

fn writeAutomaticValueList(
    writer: anytype,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formulas: []const obligation.FormulaRef,
) !void {
    try writer.writeByte('[');
    for (formulas, 0..) |formula, index| {
        if (index != 0) try writer.writeAll(", ");
        const term_id = formulaTermId(formula) orelse return error.InvalidDependency;
        const ty = automaticTermType(set, term_id) orelse return error.InvalidDependency;
        const shape = automaticIntegerShape(ty) orelse return error.InvalidDependency;
        try writer.writeAll("Ora.Integer.Value.mk ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.writeByte(' ');
        try writeAutomaticValueExpression(writer, set, summary, term_id, 256, .current);
    }
    try writer.writeByte(']');
}

fn writeAutomaticInitialStateAliases(
    writer: anytype,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    indent: usize,
) !void {
    for (summary.init_formulas, 0..) |formula, index| {
        const term_id = formulaTermId(formula) orelse return error.InvalidDependency;
        try writeAutomaticIndent(writer, indent);
        try writer.print("let auto_following_bits_{d} := ", .{index});
        try writeAutomaticValueExpression(writer, set, summary, term_id, 256, .current);
        try writer.writeByte('\n');
    }
}

fn writeAutomaticInitializationReduction(
    writer: anytype,
    set: obligation.ObligationSet,
    plan: AutomaticLoopPlan,
    term_ids: []const obligation.TermId,
    free_var_count: usize,
    indent: usize,
) !void {
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "simp only [Ora.Loop.valuesInitializeState, Ora.Loop.initializeState?, " ++
            "Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?, " ++
            "Ora.Loop.valuesMatchVariables] at hInitial\n",
    );
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?, " ++
            "Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?, " ++
            "Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult, " ++
            "Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, " ++
            "Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, ",
    );
    try writeAutomaticIntegerShapeItems(writer);
    try writer.writeAll(", ");
    try writeAutomaticTermLookupItems(writer, term_ids);
    try writeAutomaticFreeEqualityItems(writer, free_var_count);
    try writeAutomaticContextLookupItems(writer, plan.summary.context_variables.len);
    try writer.writeAll(", if_pos, if_false, Bool.false_eq_true] at hInitial\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("change some ");
    try writeAutomaticValueList(writer, set, plan.summary, plan.summary.init_formulas);
    try writer.writeAll(" = some state at hInitial\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("have hState := (Option.some.inj hInitial).symm\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("subst state\n");
}

fn writeAutomaticBindingValueList(
    writer: anytype,
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    bindings: []const obligation.LoopStepAssignment,
) !void {
    try writer.writeByte('[');
    for (bindings, 0..) |binding, index| {
        if (index != 0) try writer.writeAll(", ");
        const term_id = formulaTermId(binding.value) orelse return error.InvalidDependency;
        const ty = automaticTermType(set, term_id) orelse return error.InvalidDependency;
        const shape = automaticIntegerShape(ty) orelse return error.InvalidDependency;
        try writer.writeAll("Ora.Integer.Value.mk ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.writeByte(' ');
        try writeAutomaticValueExpression(writer, set, summary, term_id, 256, .current);
    }
    try writer.writeByte(']');
}

fn writeAutomaticNextStateReduction(
    writer: anytype,
    set: obligation.ObligationSet,
    plan: AutomaticLoopPlan,
    edge: obligation.LoopEdgeRow,
    term_ids: []const obligation.TermId,
    free_var_count: usize,
    indent: usize,
) !void {
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "simp only [Ora.Loop.stateBindingsProduceState, Ora.Loop.applyStateBindings?, " ++
            "Ora.Loop.bindingsMatchVariables, Ora.Loop.bindState?, " ++
            "Ora.Loop.bindingFormulas, Ora.Loop.evaluateIntegerFormulas?, " ++
            "Ora.Loop.integerFormulaValue?, Ora.Loop.valuesMatchVariables] at hNext\n",
    );
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?, " ++
            "Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?, " ++
            "Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult, " ++
            "Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, " ++
            "Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, ",
    );
    try writeAutomaticIntegerShapeItems(writer);
    try writer.writeAll(", ");
    try writeAutomaticTermLookupItems(writer, term_ids);
    try writeAutomaticFreeEqualityItems(writer, free_var_count);
    try writeAutomaticContextLookupItems(writer, plan.summary.context_variables.len);
    try writer.writeAll(", if_pos, if_false, Bool.false_eq_true] at hNext\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("change some ");
    try writeAutomaticBindingValueList(writer, set, plan.summary, edge.next_state);
    try writer.writeAll(" = some following at hNext\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("have hFollowing := (Option.some.inj hNext).symm\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("subst following\n");
    for (edge.next_state, 0..) |binding, index| {
        const term_id = formulaTermId(binding.value) orelse return error.InvalidDependency;
        try writeAutomaticIndent(writer, indent);
        try writer.print("let auto_following_bits_{d} := ", .{index});
        try writeAutomaticValueExpression(
            writer,
            set,
            plan.summary,
            term_id,
            256,
            .current,
        );
        try writer.writeByte('\n');
    }
}

const AutomaticStateCase = struct {
    success_indent: usize,
    final_branch_indent: usize,
};

fn writeAutomaticStateCasesStart(
    writer: anytype,
    state_name: []const u8,
    ready_name: []const u8,
    variable_count: usize,
    indent: usize,
) !AutomaticStateCase {
    var level = indent;
    for (0..variable_count) |index| {
        try writeAutomaticIndent(writer, level);
        if (index == 0) {
            try writer.print(
                "rcases {s} with _ | ⟨auto_state_value_{d}, auto_state_tail_{d}⟩\n",
                .{ state_name, index, index },
            );
        } else {
            try writer.print(
                "rcases auto_state_tail_{d} with _ | ⟨auto_state_value_{d}, auto_state_tail_{d}⟩\n",
                .{ index - 1, index, index },
            );
        }
        try writeAutomaticIndent(writer, level);
        try writer.print(
            "· simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at {s}\n",
            .{ready_name},
        );
        try writeAutomaticIndent(writer, level);
        try writer.writeAll("·\n");
        level += 2;
    }
    try writeAutomaticIndent(writer, level);
    if (variable_count == 0) {
        try writer.print(
            "rcases {s} with _ | ⟨auto_state_extra, auto_state_rest⟩\n",
            .{state_name},
        );
    } else {
        try writer.print(
            "rcases auto_state_tail_{d} with _ | ⟨auto_state_extra, auto_state_rest⟩\n",
            .{variable_count - 1},
        );
    }
    try writeAutomaticIndent(writer, level);
    try writer.writeAll("·\n");
    return .{ .success_indent = level + 2, .final_branch_indent = level };
}

fn writeAutomaticStateCasesEnd(
    writer: anytype,
    ready_name: []const u8,
    state_case: AutomaticStateCase,
) !void {
    try writeAutomaticIndent(writer, state_case.final_branch_indent);
    try writer.print(
        "· simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at {s}\n",
        .{ready_name},
    );
}

fn writeAutomaticStateShapeExtraction(
    writer: anytype,
    summary: obligation.LoopSummaryRow,
    ready_name: []const u8,
    indent: usize,
) !void {
    for (summary.variables, 0..) |_, index| {
        try writeAutomaticIndent(writer, indent);
        try writer.print(
            "rcases auto_state_value_{d} with ⟨auto_state_shape_{d}, auto_state_bits_{d}⟩\n",
            .{ index, index, index },
        );
    }
    try writeAutomaticIndent(writer, indent);
    try writer.print(
        "simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at {s}\n",
        .{ready_name},
    );
    if (summary.variables.len > 1) {
        try writeAutomaticIndent(writer, indent);
        try writer.print("rcases {s} with ⟨", .{ready_name});
        for (summary.variables, 0..) |_, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("h_auto_state_shape_{d}", .{index});
        }
        try writer.writeAll("⟩\n");
    }
    for (summary.variables, 0..) |_, index| {
        try writeAutomaticIndent(writer, indent);
        try writer.print("subst auto_state_shape_{d}\n", .{index});
    }
}

fn writeAutomaticWitnessEnv(
    writer: anytype,
    summary: obligation.LoopSummaryRow,
) !void {
    if (summary.context_variables.len == 0) return writer.writeAll("Env.empty");
    for (summary.context_variables) |_| try writer.writeByte('(');
    try writer.writeAll("Env.empty");
    for (summary.context_variables, 0..) |variable, index| {
        const id = variable.id orelse return error.InvalidDependency;
        const shape = automaticIntegerShape(variable.ty) orelse return error.InvalidDependency;
        try writer.writeAll(".setFree ");
        try writeFreeVarId(writer, id);
        try writer.writeAll(" (.integer (Ora.Integer.Value.mk ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.print(" (BitVec.ofNat {d} 0))))", .{shape.width});
        _ = index;
    }
}

fn writeAutomaticContextReadyWitness(
    writer: anytype,
    summary: obligation.LoopSummaryRow,
    indent: usize,
) !void {
    if (summary.context_variables.len == 0) {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("simp [Ora.Loop.contextReady]\n");
        return;
    }
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("intro loopVar hLoopVar\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("simp at hLoopVar\n");
    if (summary.context_variables.len > 1) {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("rcases hLoopVar with hLoopVar");
        for (1..summary.context_variables.len) |_| try writer.writeAll(" | hLoopVar");
        try writer.writeByte('\n');
    }
    for (summary.context_variables, 0..) |variable, index| {
        if (summary.context_variables.len > 1) {
            try writeAutomaticIndent(writer, indent);
            try writer.writeAll("· ");
        } else {
            try writeAutomaticIndent(writer, indent);
        }
        try writer.writeAll("subst loopVar\n");
        const shape = automaticIntegerShape(variable.ty) orelse return error.InvalidDependency;
        const branch_indent = if (summary.context_variables.len > 1) indent + 2 else indent;
        try writeAutomaticIndent(writer, branch_indent);
        try writer.writeAll("exact ⟨Ora.Integer.Value.mk ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.print(" (BitVec.ofNat {d} 0), rfl, rfl⟩\n", .{shape.width});
        _ = index;
    }
}

fn writeAutomaticContextExtraction(
    writer: anytype,
    summary: obligation.LoopSummaryRow,
    indent: usize,
) !void {
    for (summary.context_variables, 0..) |variable, index| {
        const id = variable.id orelse return error.InvalidDependency;
        const shape = automaticIntegerShape(variable.ty) orelse return error.InvalidDependency;
        try writeAutomaticIndent(writer, indent);
        try writer.print("rcases hContext (autoSummary).contextVariables[{d}] ", .{index});
        try writer.print("(by simp) with ⟨auto_context_value_{d}, ", .{index});
        try writer.print("h_auto_context_shape_{d}, h_auto_context_lookup_{d}⟩\n", .{ index, index });
        try writeAutomaticIndent(writer, indent);
        try writer.print(
            "rcases auto_context_value_{d} with ⟨auto_context_shape_{d}, auto_context_bits_{d}⟩\n",
            .{ index, index, index },
        );
        try writeAutomaticIndent(writer, indent);
        try writer.print("have h_auto_context_shape_value_{d} : auto_context_shape_{d} = ", .{ index, index });
        try writeAutomaticLeanShape(writer, shape);
        try writer.writeAll(" := by\n");
        try writeAutomaticIndent(writer, indent + 2);
        try writer.writeAll("have hSome : some ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.print(" = some auto_context_shape_{d} := by simpa using h_auto_context_shape_{d}\n", .{ index, index });
        try writeAutomaticIndent(writer, indent + 2);
        try writer.writeAll("exact (Option.some.inj hSome).symm\n");
        try writeAutomaticIndent(writer, indent);
        try writer.print("subst auto_context_shape_{d}\n", .{index});
        try writeAutomaticIndent(writer, indent);
        try writer.print("change lookupFreeBinding env.freeBindings ", .{});
        try writeFreeVarId(writer, id);
        try writer.writeAll(" = some (.integer (Ora.Integer.Value.mk ");
        try writeAutomaticLeanShape(writer, shape);
        try writer.print(" auto_context_bits_{d})) at h_auto_context_lookup_{d}\n", .{ index, index });
    }
}

fn writeAutomaticArithmeticClose(
    writer: anytype,
    summary: obligation.LoopSummaryRow,
    state_view: AutomaticStateView,
    indent: usize,
) !void {
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("first\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("| bv_decide\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("| (");
    if (state_view == .following) {
        try writer.writeAll("simp only [");
        for (summary.variables, 0..) |_, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("auto_following_bits_{d}", .{index});
        }
        try writer.writeAll("]; ");
    }
    try writer.writeAll("apply Ora.Integer.signed_add_one_le_of_lt; assumption)\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("| (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("| (");
    if (state_view == .following) {
        try writer.writeAll("simp only [");
        for (summary.variables, 0..) |_, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("auto_following_bits_{d}", .{index});
        }
        try writer.writeAll("]; ");
    }
    try writer.writeAll("apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll(
        "| (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; " ++
            "simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, " ++
            "Ora.Spec.uintBits] <;> omega)\n",
    );
    if (state_view == .following) {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("| (simp_all only [");
        for (summary.variables, 0..) |_, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("auto_following_bits_{d}", .{index});
        }
        try writer.writeAll("] <;> done)\n");
    }
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("| (simp [");
    if (state_view == .following) {
        for (summary.variables, 0..) |_, index| {
            try writer.print("auto_following_bits_{d}, ", .{index});
        }
    }
    try writer.writeAll(
        "Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, " ++
            "Ora.Integer.signedLt, Ora.Integer.signedLe, " ++
            "Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)\n",
    );
}

fn writeAutomaticFormulaHypotheses(
    writer: anytype,
    set: obligation.ObligationSet,
    plan: AutomaticLoopPlan,
    formulas: []const obligation.FormulaRef,
    source_name: []const u8,
    prefix: []const u8,
    term_ids: []const obligation.TermId,
    free_var_count: usize,
    indent: usize,
) !void {
    for (formulas, 0..) |formula, index| {
        const term_id = formulaTermId(formula) orelse return error.InvalidDependency;
        try writeAutomaticIndent(writer, indent);
        try writer.print(
            "have {s}_{d} := {s} (.term {d}) (by simp)\n",
            .{ prefix, index, source_name, term_id },
        );
        const name = try std.fmt.allocPrint(std.heap.page_allocator, "{s}_{d}", .{ prefix, index });
        defer std.heap.page_allocator.free(name);
        try writeAutomaticFormulaReduction(
            writer,
            set,
            plan,
            formula,
            term_ids,
            free_var_count,
            .current,
            name,
            indent,
        );
    }
}

fn writeAutomaticFormulaGoals(
    writer: anytype,
    set: obligation.ObligationSet,
    plan: AutomaticLoopPlan,
    formulas: []const obligation.FormulaRef,
    term_ids: []const obligation.TermId,
    free_var_count: usize,
    state_view: AutomaticStateView,
    indent: usize,
) !void {
    if (formulas.len == 0) {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("simp [Ora.Loop.formulasHoldInState]\n");
        return;
    }
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("intro formula hFormula\n");
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("simp at hFormula\n");
    if (formulas.len == 1) {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("subst formula\n");
        try writeAutomaticFormulaReduction(
            writer,
            set,
            plan,
            formulas[0],
            term_ids,
            free_var_count,
            state_view,
            null,
            indent,
        );
        try writeAutomaticArithmeticClose(writer, plan.summary, state_view, indent);
        return;
    }
    try writeAutomaticIndent(writer, indent);
    try writer.writeAll("rcases hFormula with hFormula");
    for (1..formulas.len) |_| try writer.writeAll(" | hFormula");
    try writer.writeAll(" <;> subst formula\n");
    for (formulas) |formula| {
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("·\n");
        try writeAutomaticFormulaReduction(
            writer,
            set,
            plan,
            formula,
            term_ids,
            free_var_count,
            state_view,
            null,
            indent + 2,
        );
        try writeAutomaticArithmeticClose(writer, plan.summary, state_view, indent + 2);
    }
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

    try writer.writeAll("section\n");
    try writer.print(
        "private abbrev autoSummary_q{d} : Ora.Loop.SummaryRow := emittedLoopSummary_{d}\n" ++
            "local notation \"autoSummary\" => autoSummary_q{d}\n" ++
            "attribute [local simp] autoSummary_q{d} emittedLoopSummary_{d}\n",
        .{ query.id, summary.id, query.id, query.id, summary.id },
    );
    if (set.terms.len > obligation_to_lean.lean_term_chunk_size) {
        try writer.writeAll("attribute [local irreducible] emittedManifest\n");
    }
    try writeTermLookupSimpAttributes(writer, term_ids);
    try writer.print("theorem discharge_q{d} : emittedQuery_{d} := by\n", .{ query.id, query.id });
    try writer.print(
        "  unfold emittedQuery_{d} Ora.Loop.loopProofAuthorizationFromAssumptions\n",
        .{query.id},
    );
    try writeFreeVarEqualityFacts(writer, free_vars);
    try writer.print(
        "  have hSupported : emittedLoopSummary_{d}.supported emittedManifest = true := by\n",
        .{summary.id},
    );
    try writer.print(
        "    have hShape : emittedLoopSummary_{d}.shapeSupported = true := by decide\n",
        .{summary.id},
    );
    try writer.print(
        "    have hFormulas : emittedLoopSummary_{d}.formulasDenotable emittedManifest = true := by\n",
        .{summary.id},
    );
    try writer.writeAll(
        "      simp only [Ora.Loop.SummaryRow.formulasDenotable,\n" ++
            "        Ora.Loop.SummaryRow.supportEnv, Ora.Loop.bindSupportVariable,\n" ++
            "        Ora.Loop.valueForTy, Ora.Loop.edgeFormulasDenotable,\n" ++
            "        Ora.Loop.formulaDenotable, Ora.Loop.valueDenotable, List.all,\n",
    );
    try writer.print("        emittedLoopSummary_{d}, ", .{summary.id});
    try writeAutomaticIntegerShapeItems(writer);
    try writeAutomaticFreeEqualityItems(writer, free_vars.len);
    try writer.writeAll(
        ", Env.setFree, Env.lookupFree, lookupFreeBinding, if_pos, if_false,\n" ++
            "        formulaDenotes?, formulaValue?, denoteFormula?, denoteValue?,\n" ++
            "        IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,\n" ++
            "        Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,\n" ++
            "        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,\n" ++
            "        Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,\n" ++
            "        Ora.Spec.IntegerShape.isSigned, Env.lookupVar, ",
    );
    try writeAutomaticTermLookupItems(writer, term_ids);
    try writer.writeAll(
        ", Bool.false_eq_true]\n" ++
            "      decide\n" ++
            "    simp only [Ora.Loop.SummaryRow.supported, emitted_manifest_wf,\n" ++
            "      emitted_terms_topological, hShape, hFormulas, Bool.true_and]\n",
    );
    try writer.writeAll("  constructor\n");
    try writer.writeAll("  · let autoWitnessEnv : Env := ");
    try writeAutomaticWitnessEnv(writer, summary);
    try writer.writeByte('\n');
    try writer.writeAll(
        "    have hInitialSome :\n" ++
            "        (Ora.Loop.initializeState? emittedManifest autoWitnessEnv\n" ++
            "          (autoSummary).variables (autoSummary).init).isSome = true := by\n" ++
            "      simp only [Ora.Loop.initializeState?,\n" ++
            "        Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,\n" ++
            "        Ora.Loop.valuesMatchVariables, autoSummary_q",
    );
    try writer.print("{d}, emittedLoopSummary_{d}]\n", .{ query.id, summary.id });
    try writer.writeAll(
        "      simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?,\n" ++
            "        Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?,\n" ++
            "        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,\n" ++
            "        Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, ",
    );
    try writeAutomaticIntegerShapeItems(writer);
    try writer.writeAll(", ");
    try writeAutomaticTermLookupItems(writer, term_ids);
    try writeAutomaticFreeEqualityItems(writer, free_vars.len);
    try writer.writeAll(
        ", if_pos, if_false, Bool.false_eq_true]\n" ++
            "      decide\n" ++
            "    cases hInitial : Ora.Loop.initializeState? emittedManifest autoWitnessEnv\n" ++
            "        (autoSummary).variables (autoSummary).init with\n" ++
            "    | none =>\n" ++
            "        rw [hInitial] at hInitialSome\n" ++
            "        simp at hInitialSome\n" ++
            "    | some initial =>\n" ++
            "        refine ⟨autoWitnessEnv, initial, ?_⟩\n" ++
            "        constructor\n" ++
            "        · simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?]\n" ++
            "        constructor\n" ++
            "        ·\n",
    );
    try writeAutomaticContextReadyWitness(writer, summary, 10);
    try writer.writeAll(
        "        · simp only [Ora.Loop.loopInitialPremises,\n" ++
            "            Ora.Loop.denoteLoopSummary?, hSupported, Ora.Loop.denoteEdges?,\n" ++
            "            Ora.Loop.denoteEdge?, Ora.Loop.denoteControlTarget?]\n" ++
            "          exact hInitial\n" ++
            "  · intro env hAssumptions hContext\n" ++
            "    simp only [Ora.Loop.loopInductionObligationsAtEnv,\n" ++
            "      Ora.Loop.denoteLoopSummary?, hSupported, Ora.Loop.denoteEdges?,\n" ++
            "      Ora.Loop.denoteEdge?, Ora.Loop.denoteControlTarget?]\n",
    );
    try writeAutomaticContextExtraction(writer, summary, 4);
    try writer.writeAll("    constructor\n");

    // Base establishment.
    try writer.writeAll(
        "    · intro state hInitial\n" ++
            "      dsimp at hInitial ⊢\n" ++
            "      have hReady := Ora.Loop.initialized_state_ready emittedManifest env\n" ++
            "        (autoSummary).variables (autoSummary).init state hInitial\n",
    );
    try writeAutomaticInitializationReduction(
        writer,
        set,
        plan,
        term_ids,
        free_vars.len,
        6,
    );
    try writeAutomaticInitialStateAliases(writer, set, summary, 6);
    try writer.writeAll("      refine ⟨hReady, ?_⟩\n");
    try writeAutomaticFormulaGoals(
        writer,
        set,
        plan,
        summary.invariant_formulas,
        term_ids,
        free_vars.len,
        .following,
        6,
    );

    // Backedge preservation.
    try writer.writeAll(
        "    · intro edge current following hMember hTarget hInvariant hPath hSafe hNext\n" ++
            "      simp at hMember\n" ++
            "      rcases hMember with hMember | hMember\n",
    );
    for (summary.edges, 0..) |edge, edge_index| {
        try writer.writeAll("      · subst edge\n");
        if (edge_index != plan.backedge_index) {
            try writer.writeAll("        dsimp at hTarget\n        simp at hTarget\n");
            continue;
        }
        try writer.writeAll(
            "        dsimp at hTarget hInvariant hPath hSafe hNext ⊢\n" ++
                "        rcases hInvariant with ⟨hReady, hInvariant⟩\n" ++
                "        have hFollowingReady := Ora.Loop.bound_state_ready emittedManifest env\n" ++
                "          (autoSummary).variables current following\n",
        );
        try writer.print("          (autoSummary).edges[{d}].nextState hNext\n", .{edge_index});
        const state_case = try writeAutomaticStateCasesStart(
            writer,
            "current",
            "hReady",
            summary.variables.len,
            8,
        );
        const indent = state_case.success_indent;
        try writeAutomaticStateShapeExtraction(writer, summary, "hReady", indent);
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            summary.invariant_formulas,
            "hInvariant",
            "h_auto_invariant",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            edge.path_formulas,
            "hPath",
            "h_auto_path",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            edge.safety_formulas,
            "hSafe",
            "h_auto_safe",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticNextStateReduction(
            writer,
            set,
            plan,
            edge,
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticIndent(writer, indent);
        try writer.writeAll("refine ⟨hFollowingReady, ?_⟩\n");
        try writeAutomaticFormulaGoals(
            writer,
            set,
            plan,
            summary.invariant_formulas,
            term_ids,
            free_vars.len,
            .following,
            indent,
        );
        try writeAutomaticStateCasesEnd(writer, "hReady", state_case);
    }

    // Per-edge safety.
    try writer.writeAll(
        "    · intro edge state hMember hInvariant hPath\n" ++
            "      simp at hMember\n" ++
            "      rcases hMember with hMember | hMember\n",
    );
    for (summary.edges) |edge| {
        try writer.writeAll("      · subst edge\n        dsimp at hInvariant hPath ⊢\n");
        if (edge.safety_formulas.len == 0) {
            try writer.writeAll("        simp [Ora.Loop.formulasHoldInState]\n");
            continue;
        }
        try writer.writeAll("        rcases hInvariant with ⟨hReady, hInvariant⟩\n");
        const state_case = try writeAutomaticStateCasesStart(
            writer,
            "state",
            "hReady",
            summary.variables.len,
            8,
        );
        const indent = state_case.success_indent;
        try writeAutomaticStateShapeExtraction(writer, summary, "hReady", indent);
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            summary.invariant_formulas,
            "hInvariant",
            "h_auto_invariant",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            edge.path_formulas,
            "hPath",
            "h_auto_path",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaGoals(
            writer,
            set,
            plan,
            edge.safety_formulas,
            term_ids,
            free_vars.len,
            .current,
            indent,
        );
        try writeAutomaticStateCasesEnd(writer, "hReady", state_case);
    }

    // Exit target assertions.
    try writer.writeAll(
        "    · intro edge current following hMember hTarget hInvariant hPath hSafe hNext\n" ++
            "      simp at hMember\n" ++
            "      rcases hMember with hMember | hMember\n",
    );
    for (summary.edges, 0..) |edge, edge_index| {
        try writer.writeAll("      · subst edge\n");
        if (edge_index != plan.successor_index) {
            try writer.writeAll("        dsimp at hTarget\n        simp at hTarget\n");
            continue;
        }
        try writer.writeAll(
            "        dsimp at hTarget hInvariant hPath hSafe hNext ⊢\n" ++
                "        rcases hInvariant with ⟨hReady, hInvariant⟩\n",
        );
        const state_case = try writeAutomaticStateCasesStart(
            writer,
            "current",
            "hReady",
            summary.variables.len,
            8,
        );
        const indent = state_case.success_indent;
        try writeAutomaticStateShapeExtraction(writer, summary, "hReady", indent);
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            summary.invariant_formulas,
            "hInvariant",
            "h_auto_invariant",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            edge.path_formulas,
            "hPath",
            "h_auto_path",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaHypotheses(
            writer,
            set,
            plan,
            edge.safety_formulas,
            "hSafe",
            "h_auto_safe",
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticNextStateReduction(
            writer,
            set,
            plan,
            edge,
            term_ids,
            free_vars.len,
            indent,
        );
        try writeAutomaticFormulaGoals(
            writer,
            set,
            plan,
            summary.post_formulas,
            term_ids,
            free_vars.len,
            .following,
            indent,
        );
        try writeAutomaticStateCasesEnd(writer, "hReady", state_case);
    }
    try writer.writeAll("end\n");
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
    for (summary.init_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
    for (summary.invariant_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
    for (summary.edges) |edge| {
        for (edge.path_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
        for (edge.safety_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);
        for (edge.next_state) |binding| try markAutomaticFormulaClosure(set, binding.value, marked);
    }
    for (summary.post_formulas) |formula| try markAutomaticFormulaClosure(set, formula, marked);

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

fn writeTermLookupSimpAttributes(
    writer: anytype,
    term_ids: []const obligation.TermId,
) !void {
    try writer.writeAll(
        "attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length",
    );
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
        try writeFreeVarEqualityFacts(writer, free_vars);
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
            for (edge.path_formulas) |formula| summary.addFormula(formula);
            for (edge.safety_formulas) |formula| summary.addFormula(formula);
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
        .value = .{ .term = 0 },
    }};
    const path = [_]obligation.FormulaRef{.{ .term = 0 }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .path_formulas = &path,
            .next_state = &next_state,
            .target = .backedge,
        },
        .{
            .id = 1,
            .path_formulas = &path,
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
        .value = .{ .term = 0 },
    }};
    const path = [_]obligation.FormulaRef{.{ .term = 0 }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .path_formulas = &path,
            .next_state = &next_state,
            .target = .backedge,
        },
        .{
            .id = 1,
            .path_formulas = &path,
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
