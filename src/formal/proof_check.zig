//! Lean proof-row checker ingestion.
//!
//! Proof rows are not trusted by themselves. Each row is accepted only after
//! the compiler generates a fresh Lean checker against the current obligation
//! module and Lean type-checks the named theorem at `emittedQuery_<id>`.

const std = @import("std");
const obligation = @import("obligation.zig");
const obligation_to_lean = @import("obligation_to_lean.zig");
const proof_manifest = @import("proof_manifest.zig");
const lifecycle = @import("storage_lifecycle.zig");
const lifecycle_hir = @import("storage_lifecycle_from_mlir.zig");
const lifecycle_lean = @import("storage_lifecycle_to_lean.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const proof_check_scratch_root = "formal/Ora/ProofCheckScratch";
const max_proof_artifact_bytes = 4 * 1024 * 1024;
const canonical_obligations_module = "Ora.ProofCheckCanonical.Obligations";
const canonical_agreement_module = "Ora.ProofCheckCanonical.Agreement";

pub const LeanProofFailure = error{
    LeanProofParseError,
    LeanProofUnsolvedGoals,
    LeanProofTimeout,
    LeanToolUnavailable,
    LeanProofAbnormalTermination,
    LeanProofCheckFailed,
};

pub fn leanProofFailureReason(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.LeanProofParseError => "lean_tactic_parse_error",
        error.LeanProofUnsolvedGoals => "lean_unsolved_goals",
        error.LeanProofTimeout => "lean_tactic_timeout",
        error.LeanToolUnavailable => "lean_tool_unavailable",
        error.LeanProofAbnormalTermination => "lean_abnormal_termination",
        error.LeanProofCheckFailed => "lean_proof_rejected",
        else => null,
    };
}

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

pub const CallCustodyInput = struct {
    function_id: obligation.Id,
    function_vc_sha256: [32]u8,
    manifest_sha256: [32]u8,
    theorem_name: []const u8,
    proof_source: []const u8,
};

pub const CallCustodyManifest = struct {
    collected: @import("obligation_from_mlir.zig").CollectResult,
    source: []const u8,
    sha256: [32]u8,

    pub fn deinit(self: *CallCustodyManifest) void {
        self.collected.deinit();
        self.* = undefined;
    }
};

/// Preparation and checking both start from canonical HIR. Neither accepts a
/// caller-built ObligationSet or treats projection-completeness bits as evidence.
pub fn prepareCallCustody(allocator: std.mem.Allocator, module: @import("mlir_c_api").c.MlirModule, namespace: []const u8) !CallCustodyManifest {
    _ = try namespaceSuffix(namespace);
    var collected = try @import("obligation_from_mlir.zig").collect(allocator, module, .{ .project_calls = true });
    errdefer collected.deinit();
    var source = std.Io.Writer.Allocating.init(collected.arena.allocator());
    try obligation_to_lean.writeCallCustodyModule(&source.writer, collected.set, namespace);
    var digest: [32]u8 = undefined;
    Sha256.hash(source.written(), &digest, .{});
    return .{ .collected = collected, .source = source.written(), .sha256 = digest };
}

/// A custody receipt is deliberately not a successful VerificationQuery. It
/// records a kernel-checked installation, without claiming Z3, nonvacuity, or
/// caller consumption. No artifactDecision path consumes this receipt in G9a.
pub const CallCustodyReceipt = struct {
    arena: std.heap.ArenaAllocator,
    function_id: obligation.Id,
    function_vc_sha256: [32]u8,
    manifest_sha256: [32]u8,
    proof_sha256: [32]u8,
    checker_sha256: [32]u8,
    checker_source: []const u8,
    canonical_checker_source: []const u8,
    agreement_source: []const u8,
    axioms: []const []const u8,

    pub fn deinit(self: *CallCustodyReceipt) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn checkCallCustody(
    allocator: std.mem.Allocator,
    module: @import("mlir_c_api").c.MlirModule,
    namespace: []const u8,
    input: CallCustodyInput,
    process_environ: std.process.Environ,
    stdout: anytype,
) !CallCustodyReceipt {
    var manifest = try prepareCallCustody(allocator, module, namespace);
    defer manifest.deinit();
    const set = manifest.collected.set;
    _ = obligation.findById(set.function_summaries, input.function_id) orelse return error.CallCustodyFunctionMissing;
    const digest = try obligation_to_lean.functionVcSha256(set, input.function_id);
    if (!std.mem.eql(u8, &digest, &input.function_vc_sha256)) return error.CallCustodyFunctionDigestMismatch;
    if (!std.mem.eql(u8, &manifest.sha256, &input.manifest_sha256)) return error.CallCustodyManifestMismatch;
    if (input.proof_source.len == 0) return error.CallCustodyProofMissing;
    try validateLeanTheoremPath(input.theorem_name);
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const checked = try runLeanChecker(allocator, namespace, manifest.source, &.{.{
        .query_id = input.function_id,
        .module_name = "Ora.CallCustodyProof",
        .theorem_name = input.theorem_name,
        .call_function_id = input.function_id,
        .import_module = false,
    }}, input.proof_source, process_environ, stdout, false, arena.allocator());
    if (checked.axiom_audits.len != 1) return error.LeanAxiomAuditMissing;
    var proof_digest: [32]u8 = undefined;
    var checker_digest: [32]u8 = undefined;
    Sha256.hash(input.proof_source, &proof_digest, .{});
    Sha256.hash(checked.canonical_source, &checker_digest, .{});
    return .{
        .arena = arena,
        .function_id = input.function_id,
        .function_vc_sha256 = digest,
        .manifest_sha256 = manifest.sha256,
        .proof_sha256 = proof_digest,
        .checker_sha256 = checker_digest,
        .checker_source = checked.checked_source,
        .canonical_checker_source = checked.canonical_source,
        .agreement_source = checked.canonical_agreement_source,
        .axioms = checked.axiom_audits[0].axioms,
    };
}

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
    return try applyRows(
        allocator,
        set,
        rows,
        generated_namespace,
        obligations_source,
        process_environ,
        stdout,
        .{},
    );
}

pub const LifecycleCustodyInput = struct {
    inputs_sha256: lifecycle.Digest,
    targets_sha256: lifecycle.Digest,
    proof_source: []const u8,
    theorem_name: []const u8,
};

/// No VerificationQuery or artifact grant is installed from this receipt.
pub const LifecycleCustodyReceipt = struct {
    arena: std.heap.ArenaAllocator,
    inputs_sha256: lifecycle.Digest,
    targets_sha256: lifecycle.Digest,
    proof_sha256: lifecycle.Digest,
    checker_sha256: lifecycle.Digest,
    checker_source: []const u8,
    axioms: []const []const u8,

    pub fn deinit(self: *LifecycleCustodyReceipt) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn checkLifecycleCustody(allocator: std.mem.Allocator, module: @import("mlir_c_api").c.MlirModule, pass: *@import("ora_z3_verification").VerificationPass, input: LifecycleCustodyInput, process_environ: std.process.Environ, stdout: anytype) !LifecycleCustodyReceipt {
    var snapshot = try lifecycle_hir.collect(allocator, module, pass);
    defer snapshot.deinit();
    var projection = try lifecycle_hir.project(allocator, module, pass, snapshot.inventory);
    defer projection.deinit();
    const inputs = try lifecycle_hir.proofInputs(&projection);
    const digest = try inputs.sha256(allocator);
    if (!std.mem.eql(u8, &digest, &input.inputs_sha256)) return error.LifecycleInputsMismatch;
    var targets = std.Io.Writer.Allocating.init(allocator);
    defer targets.deinit();
    try lifecycle_lean.writeProducerTargets(&targets.writer, allocator, inputs, projection.canonical.set);
    const target_digest = lifecycle.hash(targets.written());
    if (!std.mem.eql(u8, &target_digest, &input.targets_sha256)) return error.LifecycleTargetsMismatch;
    if (input.proof_source.len == 0) return error.LifecycleProofMissing;
    if (input.proof_source.len > max_proof_artifact_bytes) return error.LifecycleProofTooLarge;
    try validateLeanTheoremPath(input.theorem_name);
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const checked = try runLeanChecker(allocator, "Ora.Generated.Obligations.Lifecycle", targets.written(), &.{.{
        .query_id = 0,
        .module_name = "Ora.LifecycleProof",
        .theorem_name = input.theorem_name,
        .lifecycle_target = true,
        .import_module = false,
    }}, input.proof_source, process_environ, stdout, true, arena.allocator());
    if (checked.axiom_audits.len != 1) return error.LeanAxiomAuditMissing;
    try lifecycle_hir.checkFresh(allocator, module, pass, inputs.inventory);
    return .{ .arena = arena, .inputs_sha256 = digest, .targets_sha256 = target_digest, .proof_sha256 = lifecycle.hash(input.proof_source), .checker_sha256 = lifecycle.hash(checked.canonical_source), .checker_source = checked.canonical_source, .axioms = checked.axiom_audits[0].axioms };
}

/// Kernel-checks compiler-constructed proof terms in the generated checker
/// itself. Unlike manifest proof rows, these rows never name or import a proof
/// module and never point at a separately materialized proof artifact.
pub fn applyInlineProofRows(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    rows: []const ProofRow,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    inline_proof_source: []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
) !?ApplyResult {
    if (rows.len == 0) return null;
    if (inline_proof_source.len == 0) return error.EmptyInlineProofSource;
    return try applyRows(
        allocator,
        set,
        rows,
        generated_namespace,
        obligations_source,
        process_environ,
        stdout,
        .{ .inline_proof_source = inline_proof_source },
    );
}

const ApplyRowsOptions = struct {
    quiet: bool = false,
    inline_proof_source: ?[]const u8 = null,
    call_custody: bool = false,
};

/// External rows only select targets. Expected bodies and digests come from a
/// fresh pair of HIR inventories, exactly as in checkCallCustody.
pub fn applyCallProofRows(allocator: std.mem.Allocator, module: @import("mlir_c_api").c.MlirModule, set: obligation.ObligationSet, rows: []const ProofRow, namespace: []const u8, process_environ: std.process.Environ, stdout: anytype) !ApplyResult {
    try set.validateTermReferences();
    try set.validateIdReferences();
    var fresh = try prepareCallCustody(allocator, module, namespace);
    defer fresh.deinit();
    if (fresh.collected.set.function_summaries.len != set.function_summaries.len or fresh.collected.set.loop_summaries.len != set.loop_summaries.len) return error.CallCustodyManifestMismatch;
    for (set.function_summaries) |row| {
        const expected = try obligation_to_lean.functionVcSha256(fresh.collected.set, row.id);
        if (!std.mem.eql(u8, &expected, &try obligation_to_lean.functionVcSha256(set, row.id))) return error.CallCustodyFunctionDigestMismatch;
        for (set.queries) |query| if (query.function_summary_id == row.id) {
            if (!std.meta.eql(query.function_vc_sha256, @as(?[32]u8, expected))) return error.CallCustodyFunctionDigestMismatch;
        };
    }
    for (set.loop_summaries) |row| {
        const expected = try obligation_to_lean.loopVcSha256(fresh.collected.set, row.id);
        if (!std.mem.eql(u8, &expected, &try obligation_to_lean.loopVcSha256(set, row.id))) return error.CallCustodyManifestMismatch;
        for (set.queries) |query| if (query.loop_summary_id == row.id) {
            if (!std.meta.eql(query.loop_vc_sha256, @as(?[32]u8, expected))) return error.CallCustodyLoopDigestMismatch;
            const target = obligation.findById(fresh.collected.set.queries, query.discharges_query_id orelse query.id) orelse return error.CallCustodyLoopTargetMismatch;
            if (!obligation.proofTargetIdentityEqual(query, target)) return error.CallCustodyLoopTargetMismatch;
        };
    }
    return applyRows(allocator, set, rows, namespace, fresh.source, process_environ, stdout, .{ .call_custody = true });
}

fn applyRows(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    rows: []const ProofRow,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
    options: ApplyRowsOptions,
) !ApplyResult {
    if (set.function_summaries.len != 0 and !options.call_custody) return error.CallCustodyManifestMismatch;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var artifacts: std.ArrayList(obligation.ProofArtifact) = .empty;
    var lean_queries: std.ArrayList(obligation.VerificationQuery) = .empty;
    var checker_rows: std.ArrayList(CheckedProofRow) = .empty;
    var certificate_rows: std.ArrayList(CertificateRow) = .empty;
    var external_sources: std.ArrayList(ExternalProofSource) = .empty;

    const next_artifact_start = nextProofArtifactId(set);
    const next_query_start = nextQueryId(set);

    for (rows, 0..) |row, index| {
        try validateProofRowSyntax(row);
        if (options.inline_proof_source == null) try validateExternalProofNamespace(row);
        if (options.inline_proof_source != null and
            (row.path != null or row.content_sha256 != null))
        {
            return error.InlineProofRowHasExternalArtifact;
        }
        const target_query = obligation.findById(set.queries, row.query_id) orelse return error.UnknownProofQuery;
        try validateProofTarget(target_query);
        if (!obligation.equalIdSlices(target_query.obligation_ids, row.obligation_ids)) return error.ProofRowObligationMismatch;
        if (!obligation.equalIdSlices(target_query.assumption_ids, row.assumption_ids)) return error.ProofRowAssumptionMismatch;

        const proof_content: ProofContent = if (options.inline_proof_source) |source|
            .{ .sha256 = try sha256HexAlloc(arena_allocator, source) }
        else
            try readAndCheckProofContent(
                arena_allocator,
                allocator,
                io,
                row.path,
                row.content_sha256,
            );
        if (row.inline_source) {
            if (proof_content.source) |source| {
                const path = row.path orelse return error.ProofRowSourceWithoutPath;
                var already_present = false;
                for (external_sources.items) |existing| {
                    if (std.mem.eql(u8, existing.path, path)) {
                        already_present = true;
                        break;
                    }
                }
                if (!already_present) try external_sources.append(arena_allocator, .{
                    .path = try arena_allocator.dupe(u8, path),
                    .source = source,
                });
            }
        }

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
            .loop_summary_id = target_query.loop_summary_id,
            .loop_edge_id = target_query.loop_edge_id,
            .loop_vc_sha256 = target_query.loop_vc_sha256,
            .function_summary_id = target_query.function_summary_id,
            .function_vc_sha256 = target_query.function_vc_sha256,
            .result = .{ .status = .proved },
        });
        try checker_rows.append(arena_allocator, .{
            .query_id = row.query_id,
            .module_name = module_name,
            .theorem_name = theorem_name,
            .obligation_ids = proof_obligation_ids,
            .assumption_ids = proof_assumption_ids,
            .requires_agreement = proofTargetRequiresAgreement(target_query),
            .z3_row_matched = target_query.smtlib_hash != null,
            .z3_proof_compatible = proofTargetIsCompatible(target_query),
            .zig_semantic_supported = switch (obligation_to_lean.querySemanticSupport(set, target_query)) {
                .supported => true,
                .unsupported => false,
            },
            .loop_summary_id = target_query.loop_summary_id,
            .call_function_id = target_query.function_summary_id,
            .call_loop_id = if (target_query.loop_summary_id) |id| if (obligation_to_lean.loopUsesCallAdapter(set, obligation.findById(set.loop_summaries, id).?)) id else null else null,
            .target_smtlib_hash = target_query.smtlib_hash,
            .target_constraint_count = target_query.constraint_count,
            // A path names user-authored source that is checked in the same
            // transient module as the freshly emitted obligations. Installed
            // user modules remain importable when no path is supplied.
            .import_module = options.inline_proof_source == null and !row.inline_source,
        });
        try certificate_rows.append(arena_allocator, .{
            .target_query_id = row.query_id,
            .lean_query_id = query_id,
            .obligation_ids = proof_obligation_ids,
            .assumption_ids = proof_assumption_ids,
            .module_name = module_name,
            .theorem_name = theorem_name,
            .path = proof_path,
            .content_sha256 = proof_content.sha256,
            .inline_source = row.inline_source,
            .target_smtlib_hash = target_query.smtlib_hash,
            .target_constraint_count = target_query.constraint_count,
            .loop_summary_id = target_query.loop_summary_id,
            .loop_vc_sha256 = target_query.loop_vc_sha256,
            .function_summary_id = target_query.function_summary_id,
            .function_vc_sha256 = target_query.function_vc_sha256,
        });
    }

    if (checker_rows.items.len == 0) return error.EmptyProofRows;
    const checker_proof_source = if (options.inline_proof_source) |source|
        source
    else if (external_sources.items.len != 0)
        try joinExternalProofSources(arena_allocator, external_sources.items)
    else
        null;
    const proof_check = try runLeanChecker(
        allocator,
        generated_namespace,
        obligations_source,
        checker_rows.items,
        checker_proof_source,
        process_environ,
        stdout,
        options.quiet,
        arena_allocator,
    );
    if (proof_check.axiom_audits.len != certificate_rows.items.len) return error.LeanAxiomAuditMissing;
    for (certificate_rows.items, proof_check.axiom_audits) |*row, audit| {
        row.axioms = audit.axioms;
    }
    const certificate_json = try buildCertificateJson(
        arena_allocator,
        generated_namespace,
        obligations_source,
        proof_check.canonical_source,
        proof_check.canonical_agreement_source,
        certificate_rows.items,
        process_environ,
    );

    var merged = set;
    merged.proof_artifacts = try concat(obligation.ProofArtifact, arena_allocator, set.proof_artifacts, artifacts.items);
    merged.queries = try concat(obligation.VerificationQuery, arena_allocator, set.queries, lean_queries.items);

    return .{
        .arena = arena,
        .set = merged,
        .applied_count = checker_rows.items.len,
        .proof_check_source = proof_check.checked_source,
        .certificate_json = certificate_json,
    };
}

fn validateProofTarget(target_query: obligation.VerificationQuery) !void {
    if (target_query.artifact_policy == .diagnostic_only) return error.ProofRowTargetNotEligible;
    if (target_query.kind == .loop_induction or target_query.function_summary_id != null) {
        if (target_query.loop_summary_id == null and target_query.function_summary_id == null) return error.ProofRowLoopSummaryMissing;
        if (target_query.obligation_ids.len != 0 or
            target_query.backend != .unspecified or
            target_query.result != null or
            target_query.proof_requirement != .lean_certificate)
        {
            return error.ProofRowTargetNotEligible;
        }
        return;
    }
    if (target_query.obligation_ids.len == 0) return error.ProofRowMissingObligations;
    switch (target_query.kind) {
        .loop_invariant_step, .loop_body_safety, .loop_invariant_post => return error.ProofRowTargetNotEligible,
        else => {},
    }

    if (target_query.result) |target_result| {
        if (target_query.backend != .z3) return error.ProofRowTargetNotZ3;
        if (target_result.degraded or target_result.vacuous or target_result.vacuity_unknown) {
            return error.ProofRowTargetNotPlainUnknown;
        }
        if (target_result.status != .unknown) return error.ProofRowTargetNotPlainUnknown;
        return;
    }

    if (target_query.backend == .z3) return error.ProofRowTargetMissingResult;
}

fn proofTargetIsPlainUnknown(target_query: obligation.VerificationQuery) bool {
    if (target_query.backend != .z3) return false;
    const result = target_query.result orelse return false;
    return result.status == .unknown and
        !result.degraded and
        !result.vacuous and
        !result.vacuity_unknown;
}

fn proofTargetRequiresAgreement(target_query: obligation.VerificationQuery) bool {
    return proofTargetIsCompatible(target_query);
}

fn proofTargetIsCompatible(target_query: obligation.VerificationQuery) bool {
    return proofTargetIsPlainUnknown(target_query);
}

fn validateProofRowSyntax(row: ProofRow) !void {
    try validateLeanModulePath(row.module_name);
    try validateLeanTheoremPath(row.theorem_name);
    if (row.content_sha256) |digest| try validateSha256Hex(digest);
    if (row.path == null and row.content_sha256 != null) return error.ProofRowDigestWithoutPath;
    if (row.inline_source and row.path == null) return error.InlineProofSourceWithoutPath;
}

fn validateExternalProofNamespace(row: ProofRow) !void {
    const compiler_owned_namespaces = [_][]const u8{
        "Ora.AutoProofScratch",
        "Ora.ProofCheckScratch",
        "Ora.CompilerAutomaticProof",
        "Ora.Generated",
    };
    for (compiler_owned_namespaces) |namespace| {
        if (isNamespaceOrMember(row.module_name, namespace) or
            isNamespaceOrMember(row.theorem_name, namespace))
        {
            return error.CompilerGeneratedProofNamespaceForbidden;
        }
    }
}

fn isNamespaceOrMember(path: []const u8, namespace: []const u8) bool {
    if (std.mem.eql(u8, path, namespace)) return true;
    return path.len > namespace.len and
        std.mem.startsWith(u8, path, namespace) and
        path[namespace.len] == '.';
}

const CheckedProofRow = struct {
    query_id: obligation.Id,
    module_name: []const u8,
    theorem_name: []const u8,
    obligation_ids: []const obligation.Id = &.{},
    assumption_ids: []const obligation.Id = &.{},
    requires_agreement: bool = false,
    z3_row_matched: bool = false,
    z3_proof_compatible: bool = false,
    zig_semantic_supported: bool = false,
    loop_summary_id: ?obligation.Id = null,
    call_function_id: ?obligation.Id = null,
    call_loop_id: ?obligation.Id = null,
    lifecycle_target: bool = false,
    target_smtlib_hash: ?u64 = null,
    target_constraint_count: u32 = 0,
    import_module: bool = true,
};

const LeanCheckResult = struct {
    checked_source: []const u8,
    canonical_source: []const u8,
    canonical_agreement_source: []const u8,
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
    inline_source: bool = false,
    axioms: []const []const u8 = &.{},
    target_smtlib_hash: ?u64,
    target_constraint_count: u32,
    loop_summary_id: ?obligation.Id = null,
    loop_vc_sha256: ?[32]u8 = null,
    function_summary_id: ?obligation.Id = null,
    function_vc_sha256: ?[32]u8 = null,
};

fn buildCertificateJson(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    proof_check_source: []const u8,
    agreement_source: []const u8,
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
        agreement_source,
        rows,
        lean_version,
    );
}

fn buildCertificateJsonWithLeanVersion(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    proof_check_source: []const u8,
    agreement_source: []const u8,
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
    const agreement_sha256 = try sha256HexAlloc(allocator, agreement_source);
    defer allocator.free(agreement_sha256);

    try writer.writeAll("{\n  \"schema_version\": ");
    try writer.print("{d}", .{obligation.proof_certificate_schema_version});
    try writer.writeAll(",\n  \"status\": \"checked\",\n  \"checker\": \"lean\",\n  \"hash_algorithm\": \"sha256\",\n  \"checker_source_identity\": \"canonical_scratch_modules\",\n  \"lean_version\": ");
    try writeJsonString(writer, lean_version);
    try writer.writeAll(",\n  \"generated_namespace\": ");
    try writeJsonString(writer, generated_namespace);
    try writer.writeAll(",\n  \"obligation_module_sha256\": ");
    try writeJsonString(writer, obligations_sha256);
    try writer.writeAll(",\n  \"proof_check_sha256\": ");
    try writeJsonString(writer, proof_check_sha256);
    try writer.writeAll(",\n  \"agreement_check_sha256\": ");
    try writeJsonString(writer, agreement_sha256);
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
        try obligation.writeNatList(writer, row.obligation_ids);
        try writer.writeAll(",\n      \"assumption_ids\": ");
        try obligation.writeNatList(writer, row.assumption_ids);
        try writer.writeAll(",\n      \"target_smtlib_hash\": ");
        if (row.target_smtlib_hash) |hash| {
            try writer.print("{d}", .{hash});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"target_constraint_count\": ");
        try writer.print("{d}", .{row.target_constraint_count});
        try writer.writeAll(",\n      \"loop_summary_id\": ");
        if (row.loop_summary_id) |summary_id| {
            try writer.print("{d}", .{summary_id});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"loop_vc_sha256\": ");
        if (row.loop_vc_sha256) |digest| {
            const digest_hex = std.fmt.bytesToHex(digest, .lower);
            try writeJsonString(writer, &digest_hex);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n      \"function_summary_id\": ");
        if (row.function_summary_id) |id| try writer.print("{d}", .{id}) else try writer.writeAll("null");
        try writer.writeAll(",\n      \"function_vc_sha256\": ");
        if (row.function_vc_sha256) |digest| try writeJsonString(writer, &std.fmt.bytesToHex(digest, .lower)) else try writer.writeAll("null");
        try writer.writeAll(",\n      \"authorization_claim\": ");
        try writeJsonString(
            writer,
            if (row.function_summary_id != null)
                "certified_call_installation"
            else if (row.loop_summary_id != null)
                "loop_induction_and_nonvacuity"
            else
                "query_proof",
        );
        try writer.writeAll(",\n      \"lean_status\": \"checked\"");
        try writer.writeAll(",\n      \"anti_vacuity_status\": ");
        try writeJsonString(
            writer,
            if (row.loop_summary_id != null) "proved" else "not_applicable",
        );
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
        try writer.writeAll(",\n      \"inline_source\": ");
        try writer.writeAll(if (row.inline_source) "true" else "false");
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

const CertificateBindingSources = struct {
    proof_check: []const u8,
    agreement: []const u8,
};

/// Scratch module names are randomized only to isolate concurrent Lean runs.
/// Certificate identity rebuilds the same generated sources with fixed module
/// aliases; the imported obligation, agreement, and proof contents are hashed
/// independently in the certificate.
fn buildCertificateBindingSources(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    rows: []const CheckedProofRow,
    inline_proof_source: ?[]const u8,
) !CertificateBindingSources {
    const agreement = try buildLeanAgreementSource(
        allocator,
        generated_namespace,
        canonical_obligations_module,
        rows,
        .{},
    );
    errdefer allocator.free(agreement);
    const proof_check = try buildLeanCheckerSourceWithInlineProofs(
        allocator,
        generated_namespace,
        canonical_obligations_module,
        canonical_agreement_module,
        rows,
        inline_proof_source,
    );
    return .{ .proof_check = proof_check, .agreement = agreement };
}

fn runLeanChecker(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_source: []const u8,
    rows: []const CheckedProofRow,
    inline_proof_source: ?[]const u8,
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
    try sweepProofCheckScratchRoot(allocator, std.Io.Dir.cwd(), io, proof_check_scratch_root);
    defer std.Io.Dir.cwd().deleteDir(io, proof_check_scratch_root) catch {};

    const scratch_rel = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ proof_check_scratch_root, scratch_segment });
    defer allocator.free(scratch_rel);
    try std.Io.Dir.cwd().createDirPath(io, scratch_rel);
    defer std.Io.Dir.cwd().deleteTree(io, scratch_rel) catch {};

    const lease_rel = try std.fmt.allocPrint(allocator, "{s}/.lease", .{scratch_rel});
    defer allocator.free(lease_rel);
    var lease = try std.Io.Dir.cwd().createFile(io, lease_rel, .{ .truncate = false });
    defer lease.close(io);
    try lease.lock(io, .exclusive);
    defer lease.unlock(io);

    const obligations_rel = try std.fmt.allocPrint(allocator, "{s}/Obligations.lean", .{scratch_rel});
    defer allocator.free(obligations_rel);
    const agreement_rel = try std.fmt.allocPrint(allocator, "{s}/Agreement.lean", .{scratch_rel});
    defer allocator.free(agreement_rel);
    const checker_rel = try std.fmt.allocPrint(allocator, "{s}/Checker.lean", .{scratch_rel});
    defer allocator.free(checker_rel);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = obligations_rel, .data = obligations_source });

    const obligations_module = try std.fmt.allocPrint(
        allocator,
        "Ora.ProofCheckScratch.{s}.Obligations",
        .{scratch_segment},
    );
    defer allocator.free(obligations_module);
    const agreement_module = try std.fmt.allocPrint(
        allocator,
        "Ora.ProofCheckScratch.{s}.Agreement",
        .{scratch_segment},
    );
    defer allocator.free(agreement_module);
    const checker_path = try std.fmt.allocPrint(
        allocator,
        "Ora/ProofCheckScratch/{s}/Checker.lean",
        .{scratch_segment},
    );
    defer allocator.free(checker_path);

    const agreement_source = try buildLeanAgreementSource(result_allocator, generated_namespace, obligations_module, rows, .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = agreement_rel, .data = agreement_source });

    const checker_source = try buildLeanCheckerSourceWithInlineProofs(
        result_allocator,
        generated_namespace,
        obligations_module,
        agreement_module,
        rows,
        inline_proof_source,
    );
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = checker_rel, .data = checker_source });
    const certificate_sources = try buildCertificateBindingSources(
        result_allocator,
        generated_namespace,
        rows,
        inline_proof_source,
    );

    try runLeanCommand(allocator, &.{ "lake", "build", obligations_module }, process_environ, stdout, quiet);
    try runLeanCommand(allocator, &.{ "lake", "build", agreement_module }, process_environ, stdout, quiet);
    for (rows) |row| {
        if (!row.import_module) continue;
        try runLeanCommand(allocator, &.{ "lake", "build", row.module_name }, process_environ, stdout, quiet);
    }
    const checker_output = try runLeanCommandCapture(allocator, &.{ "lake", "env", "lean", checker_path }, process_environ, stdout, quiet);
    defer allocator.free(checker_output.stdout);
    defer allocator.free(checker_output.stderr);

    return .{
        .checked_source = checker_source,
        .canonical_source = certificate_sources.proof_check,
        .canonical_agreement_source = certificate_sources.agreement,
        .axiom_audits = try parseLeanAxiomAuditOutput(result_allocator, checker_output.stdout, rows),
    };
}

fn sweepProofCheckScratchRoot(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    io: std.Io,
    root: []const u8,
) !void {
    try dir.createDirPath(io, root);
    var scratch = try dir.openDir(io, root, .{ .iterate = true });
    defer scratch.close(io);

    var entries = scratch.iterate();
    while (try entries.next(io)) |entry| {
        if (entry.kind != .directory or !std.mem.startsWith(u8, entry.name, "Run_")) continue;

        const lease_path = try std.fmt.allocPrint(allocator, "{s}/.lease", .{entry.name});
        defer allocator.free(lease_path);
        var lease = scratch.openFile(io, lease_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                try scratch.deleteTree(io, entry.name);
                continue;
            },
            else => return err,
        };
        if (!try lease.tryLock(io, .exclusive)) {
            lease.close(io);
            continue;
        }
        lease.unlock(io);
        lease.close(io);
        try scratch.deleteTree(io, entry.name);
    }
}

const AgreementSourceOptions = struct {
    omit_agreement_rows_for_testing: bool = false,
    duplicate_agreement_rows_for_testing: bool = false,
};

fn buildLeanAgreementSource(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_module: []const u8,
    rows: []const CheckedProofRow,
    options: AgreementSourceOptions,
) ![]const u8 {
    var agreement = std.Io.Writer.Allocating.init(allocator);
    defer agreement.deinit();
    const writer = &agreement.writer;

    if (!hasAgreementTargets(rows)) {
        try writer.writeAll(
            \\-- Generated Lean proof-target agreement check. No accepted proof
            \\-- row targets a Z3 plain-UNKNOWN query in this checker run.
            \\
            \\namespace Ora.ProofCheckAgreement
            \\
            \\def acceptedProofTargetIds : List Nat := []
            \\def agreementRowsMatch : Bool := true
            \\def proofTargetsCovered : Bool := true
            \\
            \\theorem agreement_rows_match : agreementRowsMatch = true := by decide
            \\theorem proof_targets_covered : proofTargetsCovered = true := by decide
            \\
            \\end Ora.ProofCheckAgreement
            \\
        );
        return try allocator.dupe(u8, agreement.written());
    }

    try writer.writeAll(
        \\-- Generated Lean proof-target agreement check. This file is emitted
        \\-- and kernel-checked during proof acceptance; it is not user input.
        \\
    );
    try writer.writeAll("import Ora.Obligation.Agreement\n");
    try writer.print("import {s}\n\n", .{obligations_module});
    try writer.writeAll(
        \\namespace Ora.ProofCheckAgreement
        \\
    );

    try writer.writeAll("def acceptedProofTargetIds : List Nat := ");
    try writeLeanAcceptedProofTargetIds(writer, rows);
    try writer.writeAll("\n\n");

    try writer.writeAll("def agreementRows : List Ora.Obligation.Agreement.Row := ");
    try writeAgreementRows(writer, generated_namespace, rows, options);
    try writer.writeAll("\n\n");

    try writer.print(
        \\def proofTargetsCovered : Bool :=
        \\  Ora.Obligation.Agreement.targetsCovered acceptedProofTargetIds agreementRows
        \\
        \\def agreementRowsMatch : Bool :=
        \\  Ora.Obligation.Agreement.rowsMatch {s}.emittedManifest agreementRows
        \\
        \\theorem agreement_rows_match : agreementRowsMatch = true := by decide
        \\theorem proof_targets_covered : proofTargetsCovered = true := by decide
        \\
        \\end Ora.ProofCheckAgreement
        \\
    , .{generated_namespace});

    return try allocator.dupe(u8, agreement.written());
}

fn hasAgreementTargets(rows: []const CheckedProofRow) bool {
    for (rows) |row| {
        if (row.requires_agreement) return true;
    }
    return false;
}

fn writeLeanAcceptedProofTargetIds(writer: anytype, rows: []const CheckedProofRow) !void {
    try writer.writeByte('[');
    var emitted_any = false;
    for (rows) |row| {
        if (!row.requires_agreement) continue;
        if (emitted_any) try writer.writeAll(", ");
        emitted_any = true;
        try writer.print("{d}", .{row.query_id});
    }
    try writer.writeByte(']');
}

fn writeAgreementRows(
    writer: anytype,
    generated_namespace: []const u8,
    rows: []const CheckedProofRow,
    options: AgreementSourceOptions,
) !void {
    if (options.omit_agreement_rows_for_testing) return writer.writeAll("[]");
    try writer.writeByte('[');
    var emitted_any = false;
    for (0..if (options.duplicate_agreement_rows_for_testing) 2 else 1) |_| {
        for (rows) |row| {
            if (!row.requires_agreement) continue;
            if (emitted_any) try writer.writeAll(", ");
            emitted_any = true;
            try writeAgreementRow(writer, generated_namespace, row);
        }
    }
    try writer.writeByte(']');
}

fn writeAgreementRow(writer: anytype, generated_namespace: []const u8, row: CheckedProofRow) !void {
    try writer.writeAll("{ queryId := ");
    try writer.print("{d}", .{row.query_id});
    try writer.writeAll(", assumptionIds := ");
    try obligation.writeNatList(writer, row.assumption_ids);
    try writer.writeAll(", obligationIds := ");
    try obligation.writeNatList(writer, row.obligation_ids);
    try writer.writeAll(", z3RowMatched := ");
    try writer.writeAll(if (row.z3_row_matched) "true" else "false");
    try writer.writeAll(", z3ProofCompatible := ");
    try writer.writeAll(if (row.z3_proof_compatible) "true" else "false");
    try writer.writeAll(", constraintCount := ");
    try writer.print("{d}", .{row.target_constraint_count});
    try writer.writeAll(", smtlibHash := ");
    if (row.target_smtlib_hash) |hash| {
        var hash_buf: [32]u8 = undefined;
        const hash_text = try std.fmt.bufPrint(&hash_buf, "0x{x}", .{hash});
        try writeJsonString(writer, hash_text);
    } else {
        try writeJsonString(writer, "");
    }
    try writer.writeAll(", zigSemanticSupported := ");
    try writer.writeAll(if (row.zig_semantic_supported) "true" else "false");
    try writer.writeAll(", usesLoopSummary := ");
    try writer.writeAll(if (row.loop_summary_id != null) "true" else "false");
    try writer.writeAll(", loopSummarySupported := ");
    if (row.loop_summary_id) |summary_id| {
        try writer.print(
            "{s}.emittedLoopSummary_{d}.supported {s}.emittedManifest",
            .{ generated_namespace, summary_id, generated_namespace },
        );
    } else {
        try writer.writeAll("false");
    }
    try writer.writeAll(" }");
}

fn buildLeanCheckerSource(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_module: []const u8,
    agreement_module: []const u8,
    rows: []const CheckedProofRow,
) ![]const u8 {
    return buildLeanCheckerSourceWithInlineProofs(
        allocator,
        generated_namespace,
        obligations_module,
        agreement_module,
        rows,
        null,
    );
}

fn buildLeanCheckerSourceWithInlineProofs(
    allocator: std.mem.Allocator,
    generated_namespace: []const u8,
    obligations_module: []const u8,
    agreement_module: []const u8,
    rows: []const CheckedProofRow,
    inline_proof_source: ?[]const u8,
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
        \\import Std.Tactic.BVDecide
        \\import Ora.Loop.Auto
        \\
    );
    try writer.print("import {s}\n", .{obligations_module});
    try writer.print("import {s}\n", .{agreement_module});
    for (rows) |row| {
        if (!row.import_module) continue;
        try writer.print("import {s}\n", .{row.module_name});
    }
    try writer.writeByte('\n');
    if (inline_proof_source) |source| {
        try writer.writeAll(source);
        if (source[source.len - 1] != '\n') try writer.writeByte('\n');
        try writer.writeByte('\n');
    }
    try writeLeanAxiomAuditPrelude(writer);
    try writer.writeByte('\n');
    try writer.writeAll("section\n");
    var has_loop_row = false;
    for (rows) |row| {
        if (row.loop_summary_id != null) {
            has_loop_row = true;
            break;
        }
    }
    if (has_loop_row) {
        try writer.print(
            "attribute [local irreducible] {s}.emittedManifest\n\n",
            .{generated_namespace},
        );
    }
    for (rows, 0..) |row, index| {
        if (row.lifecycle_target) {
            try writer.print("theorem check_query_{d}_{d} : {s}.emittedLifecycle := by\n  exact {s}\n\n", .{ row.query_id, index, generated_namespace, row.theorem_name });
        } else if (row.call_function_id) |function_id| {
            try writer.print(
                "theorem check_query_{d}_{d} : {s}.emittedCallCustody_{d} := by\n  exact {s}\n\n",
                .{ row.query_id, index, generated_namespace, function_id, row.theorem_name },
            );
        } else if (row.call_loop_id) |loop_id| {
            try writer.print("theorem check_query_{d}_{d} : {s}.emittedCallLoop_{d} := by\n  exact {s}\n\n", .{ row.query_id, index, generated_namespace, loop_id, row.theorem_name });
        } else if (row.loop_summary_id != null) {
            try writer.print(
                \\theorem check_query_{d}_{d} :
                \\    {s}.emittedVerifiedQuery_{d} := by
                \\  exact {s}.emittedQuery_{d}_sound {s}
                \\
                \\
            , .{ row.query_id, index, generated_namespace, row.query_id, generated_namespace, row.query_id, row.theorem_name });
        } else {
            try writer.print(
                \\theorem check_query_{d}_{d} :
                \\    {s}.emittedQuery_{d} := by
                \\  exact {s}
                \\
                \\
            , .{ row.query_id, index, generated_namespace, row.query_id, row.theorem_name });
        }
    }
    try writer.writeAll("#ora_audit_axioms [\n");
    for (rows, 0..) |row, index| {
        try writer.print("  check_query_{d}_{d}\n", .{ row.query_id, index });
    }
    try writer.writeAll("]\n\nend\n");
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
    const failure = classifyLeanCommandFailure(result.term, result.stdout, result.stderr);
    if (!quiet) {
        try stdout.print("Lean proof row rejected: {s}.\n", .{leanProofFailureReason(failure).?});
        if (result.stdout.len != 0) try stdout.print("{s}\n", .{result.stdout});
        if (result.stderr.len != 0) try stdout.print("{s}\n", .{result.stderr});
        try stdout.flush();
    }
    allocator.free(result.stdout);
    allocator.free(result.stderr);
    return failure;
}

fn classifyLeanCommandFailure(
    term: std.process.Child.Term,
    command_stdout: []const u8,
    command_stderr: []const u8,
) LeanProofFailure {
    const exit_code = switch (term) {
        .exited => |code| code,
        .signal, .stopped, .unknown => return error.LeanProofAbnormalTermination,
    };
    if (exit_code == 126 or exit_code == 127) return error.LeanToolUnavailable;

    if (containsLeanFailureText(command_stdout, command_stderr, "maximum heartbeats exceeded") or
        containsLeanFailureText(command_stdout, command_stderr, "deterministic timeout"))
    {
        return error.LeanProofTimeout;
    }
    if (containsLeanFailureText(command_stdout, command_stderr, "unsolved goals")) {
        return error.LeanProofUnsolvedGoals;
    }
    if (containsLeanFailureText(command_stdout, command_stderr, "unexpected token") or
        containsLeanFailureText(command_stdout, command_stderr, "unexpected end of input") or
        containsLeanFailureText(command_stdout, command_stderr, "invalid 'end'") or
        containsLeanFailureText(command_stdout, command_stderr, "parser error"))
    {
        return error.LeanProofParseError;
    }
    return error.LeanProofCheckFailed;
}

fn containsLeanFailureText(
    command_stdout: []const u8,
    command_stderr: []const u8,
    needle: []const u8,
) bool {
    return std.mem.indexOf(u8, command_stdout, needle) != null or
        std.mem.indexOf(u8, command_stderr, needle) != null;
}

test "Lean command failures have stable named classifications" {
    const testing = std.testing;

    try testing.expectEqual(
        error.LeanProofParseError,
        classifyLeanCommandFailure(.{ .exited = 1 }, "", "error: unexpected token '('"),
    );
    try testing.expectEqual(
        error.LeanProofUnsolvedGoals,
        classifyLeanCommandFailure(.{ .exited = 1 }, "error: unsolved goals", ""),
    );
    try testing.expectEqual(
        error.LeanProofTimeout,
        classifyLeanCommandFailure(.{ .exited = 1 }, "", "maximum heartbeats exceeded"),
    );
    try testing.expectEqual(
        error.LeanToolUnavailable,
        classifyLeanCommandFailure(.{ .exited = 127 }, "", "env: lake: No such file or directory"),
    );
    try testing.expectEqual(
        error.LeanProofAbnormalTermination,
        classifyLeanCommandFailure(.{ .unknown = 9 }, "", ""),
    );
    try testing.expectEqual(
        error.LeanProofCheckFailed,
        classifyLeanCommandFailure(.{ .exited = 1 }, "", "proof rejected for another reason"),
    );

    try testing.expectEqualStrings(
        "lean_tactic_parse_error",
        leanProofFailureReason(error.LeanProofParseError).?,
    );
    try testing.expect(leanProofFailureReason(error.OutOfMemory) == null);
}

test "Lean checker reports parse unsolved and unavailable failures by name" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    const scratch_dir = "formal/Ora/ProofCheckScratch/NamedFailureFixture";
    try std.Io.Dir.cwd().createDirPath(io, scratch_dir);
    defer std.Io.Dir.cwd().deleteTree(io, scratch_dir) catch {};

    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = scratch_dir ++ "/Parse.lean",
        .data =
        \\namespace Ora.ProofCheckScratch.NamedFailureFixture
        \\
        \\theorem malformed : True := by
        \\  exact (
        \\
        \\end Ora.ProofCheckScratch.NamedFailureFixture
        \\
        ,
    });
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = scratch_dir ++ "/Unsolved.lean",
        .data =
        \\namespace Ora.ProofCheckScratch.NamedFailureFixture
        \\
        \\theorem unfinished : True ∧ True := by
        \\  constructor
        \\  · trivial
        \\
        \\end Ora.ProofCheckScratch.NamedFailureFixture
        \\
        ,
    });

    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    try std.testing.expectError(error.LeanProofParseError, runLeanCommand(
        allocator,
        &.{ "lake", "env", "lean", "Ora/ProofCheckScratch/NamedFailureFixture/Parse.lean" },
        process_environ,
        &output.writer,
        true,
    ));
    try std.testing.expectError(error.LeanProofUnsolvedGoals, runLeanCommand(
        allocator,
        &.{ "lake", "env", "lean", "Ora/ProofCheckScratch/NamedFailureFixture/Unsolved.lean" },
        process_environ,
        &output.writer,
        true,
    ));
    try std.testing.expectError(error.LeanToolUnavailable, runLeanCommand(
        allocator,
        &.{"ora-intentionally-missing-lean-tool"},
        process_environ,
        &output.writer,
        true,
    ));
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

const ProofContent = struct {
    sha256: ?[]const u8 = null,
    source: ?[]const u8 = null,
};

const ExternalProofSource = struct {
    path: []const u8,
    source: []const u8,
};

fn readAndCheckProofContent(
    result_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    io: std.Io,
    path: ?[]const u8,
    expected_sha256: ?[]const u8,
) !ProofContent {
    if (expected_sha256) |digest| try validateSha256Hex(digest);
    const proof_path = path orelse {
        if (expected_sha256 != null) return error.ProofRowDigestWithoutPath;
        return .{};
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
    return .{
        .sha256 = actual,
        .source = try result_allocator.dupe(u8, content),
    };
}

fn joinExternalProofSources(
    allocator: std.mem.Allocator,
    sources: []const ExternalProofSource,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    for (sources, 0..) |source, index| {
        if (index != 0) try out.writer.writeByte('\n');
        try out.writer.writeAll(source.source);
        if (source.source.len == 0 or source.source[source.source.len - 1] != '\n') {
            try out.writer.writeByte('\n');
        }
    }
    return try out.toOwnedSlice();
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

test "proof checker startup removes abandoned runs and preserves active leases" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = "ProofCheckScratch";
    try tmp.dir.createDirPath(std.testing.io, root ++ "/Run_abandoned");
    try tmp.dir.createDirPath(std.testing.io, root ++ "/Run_active");
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = root ++ "/Run_abandoned/Checker.lean",
        .data = "stale",
    });
    var active_lease = try tmp.dir.createFile(
        std.testing.io,
        root ++ "/Run_active/.lease",
        .{ .truncate = false },
    );
    try active_lease.lock(std.testing.io, .exclusive);

    try sweepProofCheckScratchRoot(std.testing.allocator, tmp.dir, std.testing.io, root);

    try tmp.dir.access(std.testing.io, root, .{});
    try tmp.dir.access(std.testing.io, root ++ "/Run_active/.lease", .{});
    try std.testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, root ++ "/Run_abandoned/Checker.lean", .{}),
    );

    active_lease.unlock(std.testing.io);
    active_lease.close(std.testing.io);
    try sweepProofCheckScratchRoot(std.testing.allocator, tmp.dir, std.testing.io, root);
    try std.testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, root ++ "/Run_active/.lease", .{}),
    );
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
    try std.testing.expectError(error.InlineProofSourceWithoutPath, validateProofRowSyntax(.{
        .query_id = 1,
        .obligation_ids = &.{1},
        .module_name = "Ora.Proofs.Transfer",
        .theorem_name = "Ora.Proofs.Transfer.preserves_supply",
        .inline_source = true,
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

test "external proof rows cannot name compiler-generated proof namespaces" {
    const forbidden = [_][]const u8{
        "Ora.AutoProofScratch.Generated",
        "Ora.ProofCheckScratch.Run_1",
        "Ora.CompilerAutomaticProof.Query_7",
        "Ora.Generated.Obligations.Source_fixture",
    };
    for (forbidden) |namespace| {
        try std.testing.expectError(
            error.CompilerGeneratedProofNamespaceForbidden,
            validateExternalProofNamespace(.{
                .query_id = 7,
                .obligation_ids = &.{},
                .module_name = namespace,
                .theorem_name = "Ora.UserProofs.Fixture.discharge",
            }),
        );
        try std.testing.expectError(
            error.CompilerGeneratedProofNamespaceForbidden,
            validateExternalProofNamespace(.{
                .query_id = 7,
                .obligation_ids = &.{},
                .module_name = "Ora.UserProofs.Fixture",
                .theorem_name = namespace,
            }),
        );
    }

    try validateExternalProofNamespace(.{
        .query_id = 7,
        .obligation_ids = &.{},
        .module_name = "Ora.UserProofs.GeneratedFixture",
        .theorem_name = "Ora.UserProofs.GeneratedFixture.discharge",
    });
}

test "mandatory loop induction target is independent of Z3 agreement rows" {
    const base: obligation.VerificationQuery = .{
        .id = 2,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .proof_requirement = .lean_certificate,
        .kind = .loop_induction,
        .loop_summary_id = 7,
    };

    try validateProofTarget(base);
    try std.testing.expect(!proofTargetIsCompatible(base));
    try std.testing.expect(!proofTargetRequiresAgreement(base));

    var spoofed_z3_target = base;
    spoofed_z3_target.backend = .z3;
    spoofed_z3_target.result = .{ .status = .unsat };
    try std.testing.expectError(error.ProofRowTargetNotEligible, validateProofTarget(spoofed_z3_target));

    var missing_summary = base;
    missing_summary.loop_summary_id = null;
    try std.testing.expectError(error.ProofRowLoopSummaryMissing, validateProofTarget(missing_summary));
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
    const content = try readAndCheckProofContent(
        allocator,
        allocator,
        std.testing.io,
        path,
        expected,
    );
    defer allocator.free(content.sha256.?);
    defer allocator.free(content.source.?);
    try std.testing.expectEqualStrings(expected, content.sha256.?);
    try std.testing.expectEqualStrings(source, content.source.?);

    try std.testing.expectError(error.ProofRowDigestMismatch, readAndCheckProofContent(
        allocator,
        allocator,
        std.testing.io,
        path,
        "0000000000000000000000000000000000000000000000000000000000000000",
    ));
    try std.testing.expectError(error.ProofRowDigestWithoutPath, readAndCheckProofContent(
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
        .loop_summary_id = 4,
    }};

    const certificate = try buildCertificateJsonWithLeanVersion(
        allocator,
        "Ora.Generated.Obligations.Transfer",
        "def emittedQuery_11 : Prop := True\n",
        "theorem check_query_11_0 : True := by trivial\n",
        "theorem agreement_rows_match : True := by trivial\n",
        &rows,
        "Lean test version",
    );
    defer allocator.free(certificate);

    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"schema_version\": 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"version\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"hash_algorithm\": \"sha256\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"checker_source_identity\": \"canonical_scratch_modules\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"lean_version\": \"Lean test version\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"agreement_check_sha256\": ") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"proof_count\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"target_query_id\": 11") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"lean_query_id\": 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"loop_summary_id\": 4") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"authorization_claim\": \"loop_induction_and_nonvacuity\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"lean_status\": \"checked\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"anti_vacuity_status\": \"proved\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"content_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, certificate, "\"inline_source\": false") != null);
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
        "Ora.ProofCheckScratch.Run_test.Agreement",
        &rows,
    );
    defer allocator.free(checker);

    try std.testing.expect(std.mem.indexOf(u8, checker, "import Lean.Util.CollectAxioms") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "name == `propext || name == `Quot.sound") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "ORA_AXIOM_AUDIT\\t{decl}\\t{formatAxioms normalized}") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "theorem check_query_17_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, checker, "#ora_audit_axioms [\n  check_query_17_0\n]") != null);
}

test "automatic proof terms are checked inline without importing a generated proof artifact" {
    const allocator = std.testing.allocator;
    const rows = [_]CheckedProofRow{.{
        .query_id = 17,
        .module_name = "Ora.CompilerAutomaticProof",
        .theorem_name = "Ora.CompilerAutomaticProof.Query_17.discharge_q17",
        .import_module = false,
    }};
    const inline_source =
        \\namespace Ora.CompilerAutomaticProof.Query_17
        \\
        \\theorem discharge_q17 : True := by trivial
        \\
        \\end Ora.CompilerAutomaticProof.Query_17
        \\
    ;

    const checker = try buildLeanCheckerSourceWithInlineProofs(
        allocator,
        "Ora.Generated.Obligations.Transfer",
        "Ora.ProofCheckScratch.Run_test.Obligations",
        "Ora.ProofCheckScratch.Run_test.Agreement",
        &rows,
        inline_source,
    );
    defer allocator.free(checker);

    try std.testing.expect(std.mem.indexOf(
        u8,
        checker,
        "import Ora.CompilerAutomaticProof",
    ) == null);
    try std.testing.expect(std.mem.indexOf(u8, checker, inline_source) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        checker,
        "exact Ora.CompilerAutomaticProof.Query_17.discharge_q17",
    ) != null);
}

test "manifest path source is checked beside transient obligations without a module import" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const proof_source =
        \\namespace Ora.UserProofs.PathFixture
        \\
        \\theorem discharge :
        \\    Ora.Generated.Obligations.PathFixture.emittedQuery_2 := by
        \\  trivial
        \\
        \\end Ora.UserProofs.PathFixture
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "PathFixture.lean", .data = proof_source });
    const proof_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/PathFixture.lean",
        .{tmp.sub_path},
    );
    defer allocator.free(proof_path);
    const proof_sha256 = try sha256HexAlloc(allocator, proof_source);
    defer allocator.free(proof_sha256);

    const terms = [_]obligation.Term{.{ .bool_lit = true }};
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "pathFixture" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 0 } } },
        .required_backend = .lean,
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "pathFixture" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .lean,
        .kind = .obligation,
        .logical_role = .ensures,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .obligations = &obligations,
        .queries = &queries,
    };
    const rows = [_]ProofRow{.{
        .query_id = 2,
        .obligation_ids = &obligation_ids,
        .module_name = "Ora.UserProofs.PathFixture",
        .theorem_name = "Ora.UserProofs.PathFixture.discharge",
        .path = proof_path,
        .content_sha256 = proof_sha256,
        .inline_source = true,
    }};
    const obligations_source =
        \\namespace Ora.Generated.Obligations.PathFixture
        \\
        \\def emittedQuery_2 : Prop := True
        \\
        \\end Ora.Generated.Obligations.PathFixture
        \\
    ;
    var stdout = std.Io.Writer.Allocating.init(allocator);
    defer stdout.deinit();
    var applied = (try applyProofRows(
        allocator,
        set,
        &rows,
        "Ora.Generated.Obligations.PathFixture",
        obligations_source,
        process_environ,
        &stdout.writer,
    )) orelse return error.ExpectedProofResult;
    defer applied.deinit();

    try std.testing.expect(std.mem.indexOf(u8, applied.proof_check_source, proof_source) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        applied.proof_check_source,
        "import Ora.UserProofs.PathFixture",
    ) == null);
    try std.testing.expect(std.mem.indexOf(u8, applied.certificate_json, proof_sha256) != null);
    try std.testing.expect(std.mem.indexOf(u8, applied.certificate_json, "\"axioms\": []") != null);
}

test "loop checker audits the kernel-derived verified theorem" {
    const allocator = std.testing.allocator;
    const rows = [_]CheckedProofRow{.{
        .query_id = 17,
        .module_name = "Ora.Proofs.Count",
        .theorem_name = "Ora.Proofs.Count.induction",
        .loop_summary_id = 4,
    }};
    const checker = try buildLeanCheckerSource(
        allocator,
        "Ora.Generated.Obligations.Count",
        "Ora.ProofCheckScratch.Run_loop.Obligations",
        "Ora.ProofCheckScratch.Run_loop.Agreement",
        &rows,
    );
    defer allocator.free(checker);

    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        checker,
        1,
        "Ora.Generated.Obligations.Count.emittedVerifiedQuery_17",
    ));
    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        checker,
        1,
        "Ora.Generated.Obligations.Count.emittedQuery_17_sound Ora.Proofs.Count.induction",
    ));
}

test "certificate checker identity excludes randomized scratch module names" {
    const allocator = std.testing.allocator;
    const rows = [_]CheckedProofRow{.{
        .query_id = 17,
        .module_name = "Ora.Proofs.Count",
        .theorem_name = "Ora.Proofs.Count.induction",
        .loop_summary_id = 4,
    }};
    const first_checked = try buildLeanCheckerSource(
        allocator,
        "Ora.Generated.Obligations.Count",
        "Ora.ProofCheckScratch.Run_first.Obligations",
        "Ora.ProofCheckScratch.Run_first.Agreement",
        &rows,
    );
    defer allocator.free(first_checked);
    const second_checked = try buildLeanCheckerSource(
        allocator,
        "Ora.Generated.Obligations.Count",
        "Ora.ProofCheckScratch.Run_second.Obligations",
        "Ora.ProofCheckScratch.Run_second.Agreement",
        &rows,
    );
    defer allocator.free(second_checked);
    try std.testing.expect(!std.mem.eql(u8, first_checked, second_checked));

    const first = try buildCertificateBindingSources(
        allocator,
        "Ora.Generated.Obligations.Count",
        &rows,
        null,
    );
    defer allocator.free(first.proof_check);
    defer allocator.free(first.agreement);
    const second = try buildCertificateBindingSources(
        allocator,
        "Ora.Generated.Obligations.Count",
        &rows,
        null,
    );
    defer allocator.free(second.proof_check);
    defer allocator.free(second.agreement);
    try std.testing.expectEqualStrings(first.proof_check, second.proof_check);
    try std.testing.expectEqualStrings(first.agreement, second.agreement);
    try std.testing.expect(std.mem.indexOf(
        u8,
        first.proof_check,
        canonical_obligations_module,
    ) != null);
    try std.testing.expect(std.mem.indexOf(u8, first.proof_check, "Run_first") == null);

    var changed_rows = rows;
    changed_rows[0].theorem_name = "Ora.Proofs.Count.changed_induction";
    const changed = try buildCertificateBindingSources(
        allocator,
        "Ora.Generated.Obligations.Count",
        &changed_rows,
        null,
    );
    defer allocator.free(changed.proof_check);
    defer allocator.free(changed.agreement);
    try std.testing.expect(!std.mem.eql(u8, first.proof_check, changed.proof_check));
}

test "agreement checker rejects empty rows for accepted proof targets" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    try expectBoolAgreementFixture(
        allocator,
        process_environ,
        "RunAgreementEmptyRowsFixture",
        .{},
        .{ .omit_agreement_rows_for_testing = true },
        .fails,
    );
}

test "agreement checker rejects duplicate rows for accepted proof targets" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    try expectBoolAgreementFixture(
        allocator,
        process_environ,
        "RunAgreementDuplicateRowsFixture",
        .{},
        .{ .duplicate_agreement_rows_for_testing = true },
        .fails,
    );
}

test "agreement checker accepts matching supported proof target row" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    try expectBoolAgreementFixture(
        allocator,
        process_environ,
        "RunAgreementPositiveFixture",
        .{},
        .{},
        .succeeds,
    );
}

test "agreement checker rejects mutated Z3 row attestation" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    try expectBoolAgreementFixture(
        allocator,
        process_environ,
        "RunAgreementMutatedRowFixture",
        .{ .z3_row_matched = false },
        .{},
        .fails,
    );
}

test "agreement checker rejects a proof-incompatible Z3 target" {
    const allocator = std.testing.allocator;
    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    try expectBoolAgreementFixture(
        allocator,
        process_environ,
        "RunAgreementIncompatibleFixture",
        .{ .z3_proof_compatible = false },
        .{},
        .fails,
    );
}

const AgreementRowFlags = struct {
    z3_row_matched: bool = true,
    z3_proof_compatible: bool = true,
    zig_semantic_supported: bool = true,
};

fn expectBoolAgreementFixture(
    allocator: std.mem.Allocator,
    process_environ: std.process.Environ,
    scratch_name: []const u8,
    flags: AgreementRowFlags,
    options: AgreementSourceOptions,
    expectation: AgreementFixtureExpectation,
) !void {
    const terms = [_]obligation.Term{.{ .bool_lit = true }};
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "fixture" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .term = 0 },
        } },
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
        .smtlib_hash = 99,
        .constraint_count = 3,
    }};
    const set = obligation.ObligationSet{
        .terms = &terms,
        .obligations = &obligations,
        .queries = &queries,
    };

    try std.testing.expect(switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => true,
        .unsupported => false,
    });

    const rows = [_]CheckedProofRow{.{
        .query_id = 2,
        .module_name = "Ora.Proofs.Fixture",
        .theorem_name = "Ora.Proofs.Fixture.ok",
        .obligation_ids = &obligation_ids,
        .requires_agreement = true,
        .z3_row_matched = flags.z3_row_matched,
        .z3_proof_compatible = flags.z3_proof_compatible,
        .zig_semantic_supported = flags.zig_semantic_supported,
        .target_smtlib_hash = 99,
        .target_constraint_count = 3,
    }};
    try expectAgreementFixtureCheck(
        allocator,
        process_environ,
        scratch_name,
        set,
        &rows,
        options,
        expectation,
    );
}

test "agreement checker rejects unsupported semantic proof target row" {
    const allocator = std.testing.allocator;

    const process_environ = try leanTestProcessEnviron(allocator);
    defer process_environ.block.deinit(allocator);

    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "fixture" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .origin_value = .{
                .origin = .source,
                .kind = .result,
                .index = 0,
            } },
        } },
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
        .smtlib_hash = 99,
        .constraint_count = 3,
    }};
    const set = obligation.ObligationSet{
        .obligations = &obligations,
        .queries = &queries,
    };

    try std.testing.expect(switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => false,
        .unsupported => true,
    });

    const rows = [_]CheckedProofRow{.{
        .query_id = 2,
        .module_name = "Ora.Proofs.Fixture",
        .theorem_name = "Ora.Proofs.Fixture.ok",
        .obligation_ids = &obligation_ids,
        .requires_agreement = true,
        .z3_row_matched = true,
        .z3_proof_compatible = true,
        .zig_semantic_supported = false,
        .target_smtlib_hash = 99,
        .target_constraint_count = 3,
    }};
    try expectAgreementFixtureCheck(
        allocator,
        process_environ,
        "RunAgreementUnsupportedFixture",
        set,
        &rows,
        .{},
        .fails,
    );
}

const AgreementFixtureExpectation = enum { succeeds, fails };

fn leanTestProcessEnviron(allocator: std.mem.Allocator) !std.process.Environ {
    var env_map = std.process.Environ.Map.init(allocator);
    defer env_map.deinit();
    const home = std.c.getenv("HOME") orelse return error.SkipZigTest;
    const home_slice = std.mem.span(home);
    try env_map.put("HOME", home_slice);
    const inherited_path = if (std.c.getenv("PATH")) |path| std.mem.span(path) else "/usr/bin:/bin:/opt/homebrew/bin";
    const path = try std.fmt.allocPrint(allocator, "{s}/.elan/bin:{s}", .{ home_slice, inherited_path });
    defer allocator.free(path);
    try env_map.put("PATH", path);
    return .{ .block = try env_map.createPosixBlock(allocator, .{}) };
}

fn expectAgreementFixtureCheck(
    allocator: std.mem.Allocator,
    process_environ: std.process.Environ,
    scratch_name: []const u8,
    set: obligation.ObligationSet,
    rows: []const CheckedProofRow,
    options: AgreementSourceOptions,
    expectation: AgreementFixtureExpectation,
) !void {
    const io = std.testing.io;

    const dir = try std.fmt.allocPrint(allocator, "formal/Ora/ProofCheckScratch/{s}", .{scratch_name});
    defer allocator.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const generated_namespace = try std.fmt.allocPrint(allocator, "Ora.Generated.Obligations.{s}", .{scratch_name});
    defer allocator.free(generated_namespace);
    const obligations_module = try std.fmt.allocPrint(allocator, "Ora.ProofCheckScratch.{s}.Obligations", .{scratch_name});
    defer allocator.free(obligations_module);
    const agreement_module = try std.fmt.allocPrint(allocator, "Ora.ProofCheckScratch.{s}.Agreement", .{scratch_name});
    defer allocator.free(agreement_module);

    var obligations_source = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source.deinit();
    try obligation_to_lean.writeModule(&obligations_source.writer, set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });

    const obligations_path = try std.fmt.allocPrint(allocator, "{s}/Obligations.lean", .{dir});
    defer allocator.free(obligations_path);
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = obligations_path,
        .data = obligations_source.written(),
    });

    const agreement_source = try buildLeanAgreementSource(
        allocator,
        generated_namespace,
        obligations_module,
        rows,
        options,
    );
    defer allocator.free(agreement_source);

    const agreement_path = try std.fmt.allocPrint(allocator, "{s}/Agreement.lean", .{dir});
    defer allocator.free(agreement_path);
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = agreement_path,
        .data = agreement_source,
    });

    var stdout = std.Io.Writer.Allocating.init(allocator);
    defer stdout.deinit();

    try runLeanCommand(
        allocator,
        &.{ "lake", "build", obligations_module },
        process_environ,
        &stdout.writer,
        true,
    );

    switch (expectation) {
        .succeeds => try runLeanCommand(
            allocator,
            &.{ "lake", "build", agreement_module },
            process_environ,
            &stdout.writer,
            true,
        ),
        .fails => try std.testing.expectError(error.LeanProofCheckFailed, runLeanCommand(
            allocator,
            &.{ "lake", "build", agreement_module },
            process_environ,
            &stdout.writer,
            true,
        )),
    }
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

    const process_environ = try leanTestProcessEnviron(allocator);
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
        .backend = .lean,
        .kind = .obligation,
        .logical_role = .ensures,
        .obligation_ids = &obligation_ids,
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
            .{},
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
        .{},
    ));
    if (expected_diagnostic) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, stdout.written(), needle) != null);
    }
}
