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
        try stdout.writeByte('\n');
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
    for (context.merged_result.set.queries) |query| {
        if (!isLeanProofTarget(query)) continue;
        if (!semanticSupportAvailable(obligation_to_lean.querySemanticSupport(context.merged_result.set, query))) {
            unavailable_count += 1;
            continue;
        }
        try targets.append(allocator, query);
    }

    const unknown_rows = plainUnknownPreparedRows(smt_report);
    const classified_targets = plainUnknownTargetCount(context.merged_result.set);
    const unmatched = if (unknown_rows > classified_targets) unknown_rows - classified_targets else 0;
    if (unavailable_count != 0 or unmatched != 0) {
        try stdout.writeAll("Automatic Lean proof gate cannot cover all required userland obligations.\n");
        if (unavailable_count != 0) {
            try stdout.print("  unsupported Lean-fragment proof targets: {d}\n", .{unavailable_count});
        }
        if (unmatched != 0) {
            try stdout.print("  unmatched UNKNOWN prepared rows: {d}\n", .{unmatched});
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
    const scratch_segment = try automaticScratchSegment(allocator, file_path);
    defer allocator.free(scratch_segment);
    const scratch_dir = try std.fmt.allocPrint(allocator, "formal/Ora/AutoProofScratch/{s}", .{scratch_segment});
    defer allocator.free(scratch_dir);
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().createDirPath(io, scratch_dir);
    defer std.Io.Dir.cwd().deleteTree(io, scratch_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(io, "formal/Ora/AutoProofScratch") catch {};

    const module_name = try std.fmt.allocPrint(allocator, "Ora.AutoProofScratch.{s}.Proofs", .{scratch_segment});
    defer allocator.free(module_name);
    const proof_path = try std.fmt.allocPrint(allocator, "{s}/Proofs.lean", .{scratch_dir});
    defer allocator.free(proof_path);
    const proof_source = try buildAutomaticProofSource(
        allocator,
        context.obligations_source.?,
        module_name,
        context.merged_result.set,
        targets.items,
    );
    defer allocator.free(proof_source);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = proof_path, .data = proof_source });

    const rows = try allocator.alloc(proof_check.ProofRow, targets.items.len);
    defer allocator.free(rows);
    var theorem_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (theorem_names.items) |name| allocator.free(name);
        theorem_names.deinit(allocator);
    }
    for (targets.items, rows) |query, *row| {
        const theorem_name = try std.fmt.allocPrint(allocator, "{s}.discharge_q{d}", .{ module_name, query.id });
        try theorem_names.append(allocator, theorem_name);
        row.* = .{
            .query_id = query.id,
            .obligation_ids = query.obligation_ids,
            .assumption_ids = query.assumption_ids,
            .module_name = module_name,
            .theorem_name = theorem_name,
            .path = proof_path,
            .content_sha256 = null,
        };
    }

    var applied = proof_check.applyProofRows(
        allocator,
        context.merged_result.set,
        rows,
        context.generated_namespace,
        context.obligations_source.?,
        options.process_environ,
        stdout,
    ) catch |err| switch (err) {
        error.LeanProofCheckFailed => {
            try printAutomaticProofFailure(stdout);
            return .{ .artifact_emission_authorized = false, .certificate_emitted = false };
        },
        else => return err,
    };
    defer if (applied) |*result| result.deinit();
    if (applied == null) {
        try printAutomaticProofFailure(stdout);
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

fn printAutomaticProofFailure(stdout: anytype) !void {
    try stdout.writeAll(
        "Automatic Lean proof gate failed: generated tactics did not discharge every certificate-required obligation.\n" ++
            "  Use the emitted obligation module and theorem shape from the following recipe, then pass the proof manifest with `--lean-proofs <proofs.json>`.\n",
    );
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

fn automaticScratchSegment(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    var component = std.Io.Writer.Allocating.init(allocator);
    defer component.deinit();
    try component.writer.writeAll("AutoProof_");
    for (stem) |byte| {
        try component.writer.writeByte(if (std.ascii.isAlphanumeric(byte) or byte == '_') byte else '_');
    }
    return try std.fmt.allocPrint(
        allocator,
        "{s}_{x}_{d}",
        .{ component.written(), std.hash.Wyhash.hash(0, file_path), std.posix.system.getpid() },
    );
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
        if (loop_summary.guard_formula) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
        for (loop_summary.invariant_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
        for (loop_summary.step_assignments) |assignment| try collectFormulaFreeVars(allocator, set, assignment.value, &ids);
        for (loop_summary.body_safety_formulas) |formula| try collectFormulaFreeVars(allocator, set, formula, &ids);
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

const AutomaticCounterBound = union(enum) {
    literal: []const u8,
    context: obligation.FreeVarId,
};

const AutomaticCounterLoop = struct {
    summary_id: obligation.Id,
    loop_variable: obligation.FreeVarId,
    guard_term: obligation.TermId,
    invariant_term: obligation.TermId,
    post_term: ?obligation.TermId,
    has_body_safety: bool,
    bound: AutomaticCounterBound,
};

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

fn termIsFreeVariable(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    expected: obligation.FreeVarId,
) bool {
    const term = termAt(set, id) orelse return false;
    return switch (term) {
        .variable => |variable| switch (variable) {
            .free => |free| obligation.freeVarIdEql(free.id, expected),
            .bound => false,
        },
        else => false,
    };
}

fn termIsIntegerLiteral(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    expected: []const u8,
) bool {
    const term = termAt(set, id) orelse return false;
    return switch (term) {
        .int_lit => |literal| std.mem.eql(u8, literal.value, expected),
        else => false,
    };
}

fn scalarTermsEquivalent(
    set: obligation.ObligationSet,
    lhs_id: obligation.TermId,
    rhs_id: obligation.TermId,
    fuel: u32,
) bool {
    if (fuel == 0) return false;
    const lhs = termAt(set, lhs_id) orelse return false;
    const rhs = termAt(set, rhs_id) orelse return false;
    if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) return false;
    return switch (lhs) {
        .bool_lit => |value| value == rhs.bool_lit,
        .int_lit => |literal| std.mem.eql(u8, literal.value, rhs.int_lit.value),
        .variable => |variable| switch (variable) {
            .free => |free| switch (rhs.variable) {
                .free => |other| obligation.freeVarIdEql(free.id, other.id),
                .bound => false,
            },
            .bound => |bound| switch (rhs.variable) {
                .free => false,
                .bound => |other| bound.index == other.index,
            },
        },
        .unary => |unary| unary.op == rhs.unary.op and
            scalarTermsEquivalent(set, unary.operand, rhs.unary.operand, fuel - 1),
        .binary => |binary| binary.op == rhs.binary.op and
            scalarTermsEquivalent(set, binary.lhs, rhs.binary.lhs, fuel - 1) and
            scalarTermsEquivalent(set, binary.rhs, rhs.binary.rhs, fuel - 1),
        else => false,
    };
}

fn decimalLiteralCanBeEmitted(value: []const u8) bool {
    if (value.len == 0) return false;
    for (value) |byte| if (!std.ascii.isDigit(byte)) return false;
    return true;
}

fn classifyAutomaticCounterLoop(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) ?AutomaticCounterLoop {
    if (query.kind != .loop_induction or query.assumption_ids.len != 0) return null;
    const summary_id = query.loop_summary_id orelse return null;
    const summary = obligation.findById(set.loop_summaries, summary_id) orelse return null;
    if (!summary.projectionSupported() or
        summary.variables.len != 1 or
        summary.init_formulas.len != 1 or
        summary.invariant_formulas.len != 1 or
        summary.step_assignments.len != 1 or
        summary.body_safety_formulas.len > 1 or
        summary.post_formulas.len > 1) return null;

    const loop_variable = summary.variables[0].id orelse return null;
    const init_id = formulaTermId(summary.init_formulas[0]) orelse return null;
    if (!termIsIntegerLiteral(set, init_id, "0")) return null;

    const guard_term = formulaTermId(summary.guard_formula orelse return null) orelse return null;
    const guard = switch (termAt(set, guard_term) orelse return null) {
        .binary => |binary| binary,
        else => return null,
    };
    if (guard.op != .lt or !termIsFreeVariable(set, guard.lhs, loop_variable)) return null;

    const bound: AutomaticCounterBound = switch (termAt(set, guard.rhs) orelse return null) {
        .int_lit => |literal| blk: {
            if (summary.context_variables.len != 0 or !decimalLiteralCanBeEmitted(literal.value)) return null;
            break :blk .{ .literal = literal.value };
        },
        .variable => |variable| blk: {
            const free = switch (variable) {
                .free => |value| value,
                .bound => return null,
            };
            if (summary.context_variables.len != 1) return null;
            const context_id = summary.context_variables[0].id orelse return null;
            if (!obligation.freeVarIdEql(context_id, free.id)) return null;
            break :blk .{ .context = context_id };
        },
        else => return null,
    };

    const invariant_term = formulaTermId(summary.invariant_formulas[0]) orelse return null;
    const invariant = switch (termAt(set, invariant_term) orelse return null) {
        .binary => |binary| binary,
        else => return null,
    };
    if (invariant.op != .le or
        !termIsFreeVariable(set, invariant.lhs, loop_variable) or
        !scalarTermsEquivalent(set, invariant.rhs, guard.rhs, 64)) return null;

    const assignment = summary.step_assignments[0];
    if (assignment.variable_index != 0 or
        assignment.target == null or
        !obligation.freeVarIdEql(assignment.target.?, loop_variable)) return null;
    const step_term = formulaTermId(assignment.value) orelse return null;
    const step = switch (termAt(set, step_term) orelse return null) {
        .binary => |binary| binary,
        else => return null,
    };
    if (step.op != .add or
        !termIsFreeVariable(set, step.lhs, loop_variable) or
        !termIsIntegerLiteral(set, step.rhs, "1")) return null;

    if (summary.body_safety_formulas.len == 1) {
        const safety_term = formulaTermId(summary.body_safety_formulas[0]) orelse return null;
        const safety_not = switch (termAt(set, safety_term) orelse return null) {
            .unary => |unary| unary,
            else => return null,
        };
        if (safety_not.op != .not) return null;
        const safety_lt = switch (termAt(set, safety_not.operand) orelse return null) {
            .binary => |binary| binary,
            else => return null,
        };
        if (safety_lt.op != .lt or
            !scalarTermsEquivalent(set, safety_lt.lhs, step_term, 64) or
            !termIsFreeVariable(set, safety_lt.rhs, loop_variable)) return null;
    }

    const post_term = if (summary.post_formulas.len == 1)
        formulaTermId(summary.post_formulas[0]) orelse return null
    else
        null;
    if (post_term) |id| if (!scalarTermsEquivalent(set, id, invariant_term, 64)) return null;

    return .{
        .summary_id = summary_id,
        .loop_variable = loop_variable,
        .guard_term = guard_term,
        .invariant_term = invariant_term,
        .post_term = post_term,
        .has_body_safety = summary.body_safety_formulas.len == 1,
        .bound = bound,
    };
}

fn writeAutomaticCounterNotations(
    writer: anytype,
    query: obligation.VerificationQuery,
    counter: AutomaticCounterLoop,
) !void {
    try writer.writeAll("section\n");
    try writer.print(
        "private abbrev autoSummary_q{d} : Ora.Loop.SummaryRow := emittedLoopSummary_{d}\n" ++
            "local notation \"autoSummary\" => autoSummary_q{d}\n" ++
            "attribute [local simp] autoSummary_q{d} emittedLoopSummary_{d}\n",
        .{ query.id, counter.summary_id, query.id, query.id, counter.summary_id },
    );
    try writer.print(
        "private abbrev autoLoopId_q{d} : FreeVarId := {{ file_id := {d}, pattern_id := {d} }}\n" ++
            "local notation \"autoLoopId\" => autoLoopId_q{d}\n" ++
            "attribute [local simp] autoLoopId_q{d}\n",
        .{ query.id, counter.loop_variable.file_id, counter.loop_variable.pattern_id, query.id, query.id },
    );
    try writer.print(
        "private abbrev autoGuard_q{d} : FormulaRef := .term {d}\n" ++
            "local notation \"autoGuard\" => autoGuard_q{d}\n" ++
            "attribute [local simp] autoGuard_q{d}\n",
        .{ query.id, counter.guard_term, query.id, query.id },
    );
    try writer.print(
        "private abbrev autoInvariant_q{d} : FormulaRef := .term {d}\n" ++
            "local notation \"autoInvariant\" => autoInvariant_q{d}\n" ++
            "attribute [local simp] autoInvariant_q{d}\n",
        .{ query.id, counter.invariant_term, query.id, query.id },
    );
    if (counter.post_term) |post_term| {
        try writer.print(
            "private abbrev autoPost_q{d} : FormulaRef := .term {d}\n" ++
                "local notation \"autoPost\" => autoPost_q{d}\n" ++
                "attribute [local simp] autoPost_q{d}\n",
            .{ query.id, post_term, query.id, query.id },
        );
    }
    switch (counter.bound) {
        .literal => |literal| try writer.print(
            "private abbrev autoBound_q{d} : U256 := BitVec.ofNat 256 {s}\n" ++
                "local notation \"autoBound\" => autoBound_q{d}\n" ++
                "attribute [local simp] autoBound_q{d}\n",
            .{ query.id, literal, query.id, query.id },
        ),
        .context => |context| try writer.print(
            "private abbrev autoContextId_q{d} : FreeVarId := {{ file_id := {d}, pattern_id := {d} }}\n" ++
                "local notation \"autoContextId\" => autoContextId_q{d}\n" ++
                "attribute [local simp] autoContextId_q{d}\n",
            .{ query.id, context.file_id, context.pattern_id, query.id, query.id },
        ),
    }
    try writer.print("theorem discharge_q{d} : emittedQuery_{d} := by\n", .{ query.id, query.id });
    try writer.print(
        "  unfold emittedQuery_{d} Ora.Loop.loopInductionProofFromAssumptions\n",
        .{query.id},
    );
    try writer.writeAll(
        \\  have hSupported : (autoSummary).supported emittedManifest = true := by decide
        \\  have hGuard : (autoSummary).guard = some autoGuard := rfl
        \\
    );
}

fn writeAutomaticLoopFormulaEquality(writer: anytype) !void {
    try writer.writeAll(
        \\  have hFormulaEquality : ∀ env,
        \\      formulaDenotes? emittedManifest env autoInvariant =
        \\        formulaDenotes? emittedManifest env autoPost := by
        \\    intro env
        \\    simp [formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\      emittedTerms]
        \\
    );
}

fn writeAutomaticCounterBase(
    writer: anytype,
    witness_env: []const u8,
) !void {
    try writer.writeAll("  constructor\n  · refine ⟨");
    try writer.writeAll(witness_env);
    try writer.writeAll(
        \\, [BitVec.ofNat 256 0], ?_⟩
        \\    constructor
        \\    · simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?]
        \\    constructor
        \\
    );
}

fn writeAutomaticCounterInitialProof(
    writer: anytype,
    witness_env: []const u8,
    has_context: bool,
) !void {
    try writer.writeAll(
        \\    ·
        \\
    );
    if (has_context) {
        try writer.writeAll(
            \\      intro loopVar hLoopVar
            \\      simp at hLoopVar
            \\      subst loopVar
            \\      exact ⟨BitVec.ofNat 256 0, rfl⟩
            \\
        );
    } else {
        try writer.writeAll("      simp [Ora.Loop.contextReady]\n");
    }
    try writer.writeAll(
        \\    · simp only [Ora.Loop.loopInitialPremises,
        \\        Ora.Loop.denoteLoopSummary?, hSupported, hGuard]
        \\      change Ora.Loop.valuesInitializeState emittedManifest
        \\
    );
    try writer.writeAll("        ");
    try writer.writeAll(witness_env);
    try writer.writeAll(
        \\ (autoSummary).init [BitVec.ofNat 256 0]
        \\      simp [Ora.Loop.valuesInitializeState,
        \\        formulaValue?, denoteValue?, IntegerLiteralTerm.asU256?,
        \\        TyRef.isU256Carrier, TyRef.isU256, TyRef.isI256,
        \\        compilerTypeIdU256, compilerTypeIdI256,
        \\        Ora.Spec.expectedCompilerTypeIdU256,
        \\        Ora.Spec.expectedCompilerTypeIdI256, emittedManifest, emittedTerms]
        \\
    );
}

fn writeAutomaticCounterInductionStart(writer: anytype) !void {
    try writer.writeAll(
        \\    constructor
        \\    · intro state hInitial
        \\      change Ora.Loop.valuesInitializeState emittedManifest env
        \\        (autoSummary).init state at hInitial
        \\      have hState : state = [BitVec.ofNat 256 0] := by
        \\        rcases state with _ | ⟨value, state⟩
        \\        · simp [Ora.Loop.valuesInitializeState] at hInitial
        \\        · rcases state with _ | ⟨extra, state⟩
        \\          · have hValue : BitVec.ofNat 256 0 = value := by
        \\              simpa [Ora.Loop.valuesInitializeState, formulaValue?, denoteValue?,
        \\                IntegerLiteralTerm.asU256?, TyRef.isU256Carrier, TyRef.isU256,
        \\                TyRef.isI256, compilerTypeIdU256, compilerTypeIdI256,
        \\                Ora.Spec.expectedCompilerTypeIdU256,
        \\                Ora.Spec.expectedCompilerTypeIdI256,
        \\                emittedManifest, emittedTerms] using hInitial
        \\            subst value
        \\            rfl
        \\          · simp [Ora.Loop.valuesInitializeState] at hInitial
        \\      subst state
        \\      intro formula hFormula
        \\      simp [] at hFormula
        \\      subst formula
        \\      simp [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState,
        \\        formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\        emittedTerms, hAutoBound, U256.ule, Env.setFree, Env.lookupVar,
        \\        Env.lookupFree, lookupFreeBinding, hAutoIdentityA, hAutoIdentityB,
        \\        IntegerLiteralTerm.asU256?, TyRef.isU256Carrier,
        \\        TyRef.isU256, TyRef.isI256, compilerTypeIdU256,
        \\        compilerTypeIdI256, Ora.Spec.expectedCompilerTypeIdU256,
        \\        Ora.Spec.expectedCompilerTypeIdI256]
        \\    · intro current following hInvariant hLoopGuard hStep
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables current (autoSummary).invariants at hInvariant
        \\      change Ora.Loop.formulaHoldsInState emittedManifest env
        \\        (autoSummary).variables current autoGuard at hLoopGuard
        \\      change Ora.Loop.assignmentsProduceState emittedManifest env
        \\        (autoSummary).variables current (autoSummary).step following at hStep
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables following (autoSummary).invariants
        \\      have hCurrentShape : ∃ i, current = [i] := by
        \\        rcases current with _ | ⟨i, current⟩
        \\        · simp [Ora.Loop.formulasHoldInState, Ora.Loop.formulaHoldsInState,
        \\            Ora.Loop.bindState] at hInvariant
        \\        · rcases current with _ | ⟨extra, current⟩
        \\          · exact ⟨i, rfl⟩
        \\          · simp [Ora.Loop.formulasHoldInState, Ora.Loop.formulaHoldsInState,
        \\              Ora.Loop.bindState] at hInvariant
        \\      rcases hCurrentShape with ⟨i, rfl⟩
        \\      have hFollowingShape : ∃ next, following = [next] := by
        \\        rcases following with _ | ⟨next, following⟩
        \\        · simp [Ora.Loop.assignmentsProduceState, Ora.Loop.bindState,
        \\            ] at hStep
        \\        · rcases following with _ | ⟨extra, following⟩
        \\          · exact ⟨next, rfl⟩
        \\          · simp [Ora.Loop.assignmentsProduceState, Ora.Loop.bindState,
        \\              ] at hStep
        \\      rcases hFollowingShape with ⟨next, rfl⟩
        \\      simp [Ora.Loop.assignmentsProduceState, Ora.Loop.bindState,
        \\        formulaValue?, denoteValue?, emittedManifest, emittedTerms,
        \\        Env.setFree, Env.lookupVar, Env.lookupFree,
        \\        lookupFreeBinding, hAutoIdentityA, hAutoIdentityB,
        \\        IntegerLiteralTerm.asU256?, Value.binaryU256?, TyRef.isU256Carrier,
        \\        TyRef.isU256, TyRef.isI256, compilerTypeIdU256,
        \\        compilerTypeIdI256, Ora.Spec.expectedCompilerTypeIdU256,
        \\        Ora.Spec.expectedCompilerTypeIdI256] at hStep
        \\      subst next
        \\      intro formula hFormula
        \\      simp [] at hFormula
        \\      subst formula
        \\      have hGuardDecoded : U256.ult i n := by
        \\        simpa [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState,
        \\          formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\          emittedTerms, hAutoBound, Env.setFree, Env.lookupVar, Env.lookupFree,
        \\          lookupFreeBinding, TyRef.isU256, TyRef.isI256,
        \\          TyRef.isU256Carrier, compilerTypeIdU256, compilerTypeIdI256,
        \\          Ora.Spec.expectedCompilerTypeIdU256,
        \\          Ora.Spec.expectedCompilerTypeIdI256, hAutoIdentityA,
        \\          hAutoIdentityB, IntegerLiteralTerm.asU256?,
        \\          Value.binaryU256?] using hLoopGuard
        \\      have hNextInvariant := U256.lt_add_one_ule_bound i n hGuardDecoded
        \\      simp [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState,
        \\        formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\        emittedTerms, hAutoBound, Env.setFree, Env.lookupVar, Env.lookupFree,
        \\        lookupFreeBinding, TyRef.isU256, TyRef.isI256,
        \\        TyRef.isU256Carrier, compilerTypeIdU256, compilerTypeIdI256,
        \\        Ora.Spec.expectedCompilerTypeIdU256,
        \\        Ora.Spec.expectedCompilerTypeIdI256, hAutoIdentityA, hAutoIdentityB,
        \\        IntegerLiteralTerm.asU256?, Value.binaryU256?,
        \\        hNextInvariant]
        \\
    );
}

fn writeAutomaticCounterSafety(writer: anytype, has_body_safety: bool) !void {
    try writer.writeAll(
        \\    · intro state hInvariant hLoopGuard
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables state (autoSummary).bodySafety
        \\
    );
    if (!has_body_safety) {
        try writer.writeAll("      simp []\n");
        return;
    }
    try writer.writeAll(
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables state (autoSummary).invariants at hInvariant
        \\      change Ora.Loop.formulaHoldsInState emittedManifest env
        \\        (autoSummary).variables state autoGuard at hLoopGuard
        \\      have hStateShape : ∃ i, state = [i] := by
        \\        rcases state with _ | ⟨i, state⟩
        \\        · simp [Ora.Loop.formulasHoldInState, Ora.Loop.formulaHoldsInState,
        \\            Ora.Loop.bindState] at hInvariant
        \\        · rcases state with _ | ⟨extra, state⟩
        \\          · exact ⟨i, rfl⟩
        \\          · simp [Ora.Loop.formulasHoldInState, Ora.Loop.formulaHoldsInState,
        \\              Ora.Loop.bindState] at hInvariant
        \\      rcases hStateShape with ⟨i, rfl⟩
        \\      intro formula hFormula
        \\      simp [] at hFormula
        \\      subst formula
        \\      have hGuardDecoded : U256.ult i n := by
        \\        simpa [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState,
        \\          formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\          emittedTerms, hAutoBound, Env.setFree, Env.lookupVar, Env.lookupFree,
        \\          lookupFreeBinding, TyRef.isU256, TyRef.isI256,
        \\          TyRef.isU256Carrier, compilerTypeIdU256, compilerTypeIdI256,
        \\          Ora.Spec.expectedCompilerTypeIdU256,
        \\          Ora.Spec.expectedCompilerTypeIdI256, hAutoIdentityA,
        \\          hAutoIdentityB, IntegerLiteralTerm.asU256?,
        \\          Value.binaryU256?] using hLoopGuard
        \\      have hSafe := U256.lt_bound_add_one_not_lt_self i n hGuardDecoded
        \\      simp [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState,
        \\        formulaDenotes?, denoteFormula?, denoteValue?, emittedManifest,
        \\        emittedTerms, hAutoBound, Env.setFree, Env.lookupVar, Env.lookupFree,
        \\        lookupFreeBinding, TyRef.isU256, TyRef.isI256,
        \\        TyRef.isU256Carrier, compilerTypeIdU256, compilerTypeIdI256,
        \\        Ora.Spec.expectedCompilerTypeIdU256,
        \\        Ora.Spec.expectedCompilerTypeIdI256, hAutoIdentityA, hAutoIdentityB,
        \\        IntegerLiteralTerm.asU256?, Value.binaryU256?, hSafe]
        \\
    );
}

fn writeAutomaticCounterExit(
    writer: anytype,
    query_id: obligation.Id,
    has_post: bool,
) !void {
    try writer.writeAll("    · intro state hInvariant _\n");
    if (!has_post) {
        try writer.writeAll(
            \\      change Ora.Loop.formulasHoldInState emittedManifest env
            \\        (autoSummary).variables state (autoSummary).post
            \\      simp [Ora.Loop.formulasHoldInState]
            \\
        );
        return;
    }
    try writer.writeAll(
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables state (autoSummary).invariants at hInvariant
        \\      change Ora.Loop.formulasHoldInState emittedManifest env
        \\        (autoSummary).variables state (autoSummary).post
        \\      intro formula hFormula
        \\      simp [] at hFormula
        \\      subst formula
        \\      have hInvariantFormula := hInvariant autoInvariant (by simp [])
        \\      unfold Ora.Loop.formulaHoldsInState at hInvariantFormula ⊢
        \\      cases hBind : Ora.Loop.bindState env (autoSummary).variables state with
        \\      | none =>
        \\
    );
    try writer.print(
        "          dsimp [autoSummary_q{d}] at hBind hInvariantFormula ⊢\n",
        .{query_id},
    );
    try writer.writeAll(
        \\          simp [hBind] at hInvariantFormula
        \\      | some bound =>
        \\
    );
    try writer.print(
        "          dsimp [autoSummary_q{d}] at hBind hInvariantFormula ⊢\n",
        .{query_id},
    );
    try writer.writeAll(
        \\          simp only [hBind] at hInvariantFormula ⊢
        \\          rw [← hFormulaEquality]
        \\          exact hInvariantFormula
        \\
    );
}

fn writeAutomaticLiteralCounterProof(
    writer: anytype,
    query: obligation.VerificationQuery,
    counter: AutomaticCounterLoop,
    literal: []const u8,
) !void {
    try writeAutomaticCounterNotations(writer, query, counter);
    try writer.print(
        "  have hBoundLiteral : parseDecimalNat? \"{s}\" = some {s} := by rfl\n",
        .{ literal, literal },
    );
    if (counter.post_term != null) try writeAutomaticLoopFormulaEquality(writer);
    try writeAutomaticCounterBase(writer, "Env.empty");
    try writeAutomaticCounterInitialProof(writer, "Env.empty", false);
    try writer.writeAll(
        \\  · intro env _ _
        \\    simp only [Ora.Loop.loopInductionObligationsAtEnv,
        \\      Ora.Loop.denoteLoopSummary?, hSupported, hGuard]
        \\    let n : U256 := autoBound
        \\    have hAutoBound := hBoundLiteral
        \\    have hAutoIdentityA : (autoLoopId == autoLoopId) = true := by rfl
        \\    have hAutoIdentityB := hAutoIdentityA
        \\
    );
    try writeAutomaticCounterInductionStart(writer);
    try writeAutomaticCounterSafety(writer, counter.has_body_safety);
    try writeAutomaticCounterExit(writer, query.id, counter.post_term != null);
    try writer.writeAll("end\n");
}

fn writeAutomaticContextCounterProof(
    writer: anytype,
    query: obligation.VerificationQuery,
    counter: AutomaticCounterLoop,
    context: obligation.FreeVarId,
) !void {
    _ = context;
    try writeAutomaticCounterNotations(writer, query, counter);
    if (counter.post_term != null) try writeAutomaticLoopFormulaEquality(writer);
    try writeAutomaticCounterBase(
        writer,
        "Env.empty.setFree autoContextId (.u256 (BitVec.ofNat 256 0))",
    );
    try writeAutomaticCounterInitialProof(
        writer,
        "(Env.empty.setFree autoContextId (.u256 (BitVec.ofNat 256 0)))",
        true,
    );
    try writer.writeAll(
        \\  · intro env _ hContext
        \\    simp only [Ora.Loop.loopInductionObligationsAtEnv,
        \\      Ora.Loop.denoteLoopSummary?, hSupported, hGuard]
        \\    rcases hContext (autoSummary).contextVariables[0]
        \\        (by simp []) with ⟨n, hn⟩
        \\    change lookupFreeBinding env.freeBindings autoContextId = some (.u256 n) at hn
        \\
    );
    try writer.print("    dsimp [autoContextId_q{d}] at hn\n", .{query.id});
    try writer.writeAll(
        \\    have hAutoBound := hn
        \\    have hAutoIdentityA : (autoLoopId == autoLoopId) = true := by rfl
        \\    have hAutoIdentityB : (autoLoopId == autoContextId) = false := by rfl
        \\
    );
    try writeAutomaticCounterInductionStart(writer);
    try writeAutomaticCounterSafety(writer, counter.has_body_safety);
    try writeAutomaticCounterExit(writer, query.id, counter.post_term != null);
    try writer.writeAll("end\n");
}

fn writeAutomaticCounterLoopProof(
    writer: anytype,
    query: obligation.VerificationQuery,
    counter: AutomaticCounterLoop,
) !void {
    return switch (counter.bound) {
        .literal => |literal| writeAutomaticLiteralCounterProof(writer, query, counter, literal),
        .context => |context| writeAutomaticContextCounterProof(writer, query, counter, context),
    };
}

fn buildAutomaticProofSource(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    set: obligation.ObligationSet,
    queries: []const obligation.VerificationQuery,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.LeanObligationsMissingNamespace;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.LeanObligationsMissingNamespace) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.LeanObligationsMissingNamespace;
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.print("namespace {s}\n", .{module_namespace});
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');
    for (queries) |query| {
        const free_vars = try collectQueryFreeVars(allocator, set, query);
        defer allocator.free(free_vars);
        if (query.kind == .loop_induction) {
            const counter = classifyAutomaticCounterLoop(set, query) orelse {
                try writer.print("theorem discharge_q{d} : emittedQuery_{d} := by\n", .{ query.id, query.id });
                try writer.writeAll(
                    "  fail \"automatic scalar-loop synthesis does not support this induction shape; use the emitted proof manifest workflow\"\n\n",
                );
                continue;
            };
            try writeAutomaticCounterLoopProof(writer, query, counter);
            try writer.writeByte('\n');
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
    }
    try writer.print("end {s}\n", .{module_namespace});
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
        if (loop_summary.guard_formula) |formula| summary.addFormula(formula);
        for (loop_summary.invariant_formulas) |formula| summary.addFormula(formula);
        for (loop_summary.step_assignments) |assignment| summary.addFormula(assignment.value);
        for (loop_summary.body_safety_formulas) |formula| summary.addFormula(formula);
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
            try writer.writeAll("; the current Lean semantic proof fragment supports bool formulas and u256 values only\n");
        },
        .unsupported_comparison_width => try writer.writeAll("    reason: signed comparison width is outside the current Lean semantic proof fragment; only 256-bit signed comparisons are supported today\n"),
        .unknown_signedness => try writer.writeAll("    reason: signed comparison operand is missing compiler type-id signedness metadata\n"),
        .mixed_signedness => try writer.writeAll("    reason: comparison predicate signedness does not match both operand types\n"),
        .unsupported_arithmetic_width => try writer.writeAll("    reason: arithmetic value width is outside the current Lean semantic proof fragment; only 256-bit arithmetic values are supported today\n"),
        .unknown_arithmetic_signedness => try writer.writeAll("    reason: arithmetic value is missing compiler type-id signedness metadata\n"),
        .mixed_arithmetic_signedness => try writer.writeAll("    reason: arithmetic operation signedness does not match the value type\n"),
        .missing_key_disjoint_evidence => try writer.writeAll("    reason: evidence-backed storage frame has no key-disjointness evidence rows\n"),
        .unsupported_key_disjoint_evidence_formula => try writer.writeAll("    reason: key-disjointness evidence must be a direct requires free-variable disequality in the Lean fragment\n"),
        .unsupported_key_disjoint_evidence_kind => try writer.writeAll("    reason: key-disjointness evidence kind is not supported by the Lean fragment\n"),
        .key_disjoint_evidence_type_unsupported => try writer.writeAll("    reason: key-disjointness evidence currently supports only 256-bit carrier free variables\n"),
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
