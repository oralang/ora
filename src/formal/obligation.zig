//! Backend-neutral verification obligation manifest.
//!
//! This is not a replacement IR for Ora MLIR, HIR, or Z3. It is the stable
//! manifest layer that records which compiler facts must be checked, where they
//! came from, and which backend is responsible for checking them.
//!
//! Keep this file data-oriented. Builders, MLIR walkers, Z3 encoding, Lean
//! emission, and optimization/simplification belong in separate adapter files.

const std = @import("std");
const timeline_validation = @import("loop_timeline_validation.zig").Validator(@This());

pub const Id = u32;
pub const TermId = u32;

pub const obligation_dump_schema_version: u32 = 8;
pub const proof_certificate_schema_version: u32 = 2;

pub const ObligationSet = struct {
    obligations: []const Obligation = &.{},
    assumptions: []const Assumption = &.{},
    queries: []const VerificationQuery = &.{},
    /// Backend-neutral loop summaries. Every supported, owner-bound summary
    /// has one `loop_induction` query requiring a kernel-checked Lean
    /// certificate before verified artifacts may be emitted. Function
    /// postconditions enrich the induction theorem but do not arm the gate.
    loop_summaries: []const LoopSummaryRow = &.{},
    proof_artifacts: []const ProofArtifact = &.{},
    diagnostics: []const ObligationDiagnostic = &.{},
    terms: []const Term = &.{},

    pub fn hasBlockingDiagnostic(self: ObligationSet) bool {
        for (self.diagnostics) |diagnostic| {
            if (diagnostic.blocks_artifacts) return true;
        }
        return false;
    }

    pub fn artifactDecision(self: ObligationSet) ArtifactDecision {
        for (self.diagnostics) |diagnostic| {
            if (diagnostic.blocks_artifacts) return .{ .blocked = .blocking_diagnostic };
        }
        self.validateTermReferences() catch return .{ .blocked = .invalid_dependency };
        self.validateIdReferences() catch return .{ .blocked = .invalid_dependency };
        if (self.storageArtifactAuthorityBlockReason()) |reason| {
            return .{ .blocked = reason };
        }

        for (self.queries) |query| {
            if (query.artifact_policy == .diagnostic_only) continue;
            if (query.result) |result| {
                if (result.degraded) return .{ .blocked = .degraded_query };
                if (result.vacuous) return .{ .blocked = .vacuous_query };
                if (result.vacuity_unknown) return .{ .blocked = .unknown_query };
                if (result.status == .unknown) {
                    if (self.queryUnknownDischargedByLean(query)) continue;
                    return .{ .blocked = if (query.proof_requirement == .lean_certificate)
                        .lean_required_failure
                    else
                        .unknown_query };
                }
                if (!querySucceeded(query, result)) {
                    return .{ .blocked = if (query.backend == .lean) .lean_required_failure else .failed_query };
                }
                if (query.proof_requirement == .lean_certificate and
                    !self.hasSuccessfulLeanProofForQuery(query))
                {
                    return .{ .blocked = .lean_required_failure };
                }
            } else {
                const is_loop_owned_z3 = query.backend == .z3 and query.loop_summary_id != null;
                if (!is_loop_owned_z3 and self.hasSuccessfulLeanProofForQuery(query)) continue;
                return .{ .blocked = if (query.backend == .lean or query.proof_requirement == .lean_certificate)
                    .lean_required_failure
                else
                    .missing_proof };
            }
        }

        for (self.obligations) |item| {
            if (!item.blocksArtifacts()) continue;
            if (!self.hasSuccessfulProofFor(item)) {
                return .{ .blocked = if (item.required_backend == .lean) .lean_required_failure else .missing_proof };
            }
        }

        return .allowed;
    }

    fn storageArtifactAuthorityBlockReason(self: ObligationSet) ?ArtifactBlockReason {
        for (self.loop_summaries) |summary| {
            if (!self.loopSummaryRequiresLeanCertificate(summary.id)) continue;
            if (self.timelineStorageAuthorityBlockReason(summary, summary.preheader_timeline)) |reason| return reason;
            for (summary.edges) |edge| if (self.timelineStorageAuthorityBlockReason(summary, edge.timeline)) |reason| return reason;
        }
        return null;
    }

    fn timelineStorageAuthorityBlockReason(
        self: ObligationSet,
        summary: LoopSummaryRow,
        timeline: []const LoopTimelineEvent,
    ) ?ArtifactBlockReason {
        for (timeline) |event| switch (event.payload) {
            .storage_read => |read| if (read.location.projections.len != 0 and
                !self.reviewedMappingProjection(summary, read.location)) return .storage_projection_authority_unreviewed,
            .storage_write => |write| if (write.location.projections.len != 0 and
                !self.reviewedMappingProjection(summary, write.location)) return .storage_projection_authority_unreviewed,
            .summary_dependency => |dependency| if (self.loopSummaryClosureHasStorageEvents(
                dependency.summary_id,
                self.loop_summaries.len + 1,
            )) return .storage_child_frame_authority_unreviewed,
            .path_assumption, .safety_checkpoint => {},
        };
        return null;
    }

    fn reviewedMappingProjection(
        self: ObligationSet,
        summary: LoopSummaryRow,
        location: LoopStorageLocationExpr,
    ) bool {
        if (!summary.preheader_projection_complete or location.projections.len != 1) return false;
        const key = switch (location.projections[0]) {
            .mapping_key => |key| key,
            .field, .fixed_index => return false,
        };
        if (key.value.read_results.len != 0) return false;
        const term_id = switch (key.value.formula) {
            .term => |id| id,
            .origin_value => return false,
        };
        if (term_id >= self.terms.len) return false;
        const id = switch (self.terms[term_id]) {
            .variable => |variable| switch (variable) {
                .free => |free| free.id,
                .bound => return false,
            },
            else => return false,
        };
        for (summary.context_variables) |variable| {
            if (variable.id) |context_id| {
                if (freeVarIdEql(context_id, id)) return true;
            }
        }
        return false;
    }

    fn loopSummaryRequiresLeanCertificate(self: ObligationSet, summary_id: Id) bool {
        for (self.queries) |query| {
            if (query.loop_summary_id == summary_id and
                query.kind == .loop_induction and
                query.artifact_policy == .blocks_verified_artifacts and
                query.proof_requirement == .lean_certificate) return true;
        }
        return false;
    }

    fn loopSummaryClosureHasStorageEvents(
        self: ObligationSet,
        summary_id: Id,
        remaining: usize,
    ) bool {
        if (remaining == 0) return true;
        const summary = findById(self.loop_summaries, summary_id) orelse return true;
        for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
            .storage_read, .storage_write => return true,
            .summary_dependency => |dependency| if (self.loopSummaryClosureHasStorageEvents(
                dependency.summary_id,
                remaining - 1,
            )) return true,
            .path_assumption, .safety_checkpoint => {},
        };
        return false;
    }

    pub fn hasLeanCertificateRequirements(self: ObligationSet) bool {
        for (self.queries) |query| {
            if (query.artifact_policy == .blocks_verified_artifacts and
                query.proof_requirement == .lean_certificate) return true;
        }
        return false;
    }

    pub fn validateTermReferences(self: ObligationSet) !void {
        for (self.assumptions) |assumption| {
            if (assumption.formula) |formula| try self.validateFormulaRef(formula);
        }
        for (self.obligations) |obligation| {
            try self.validateKindTermRefs(obligation.kind);
        }
        for (self.terms) |term| {
            try self.validateTerm(term);
        }
        for (self.loop_summaries) |summary| {
            for (summary.init_formulas) |formula| try self.validateFormulaRef(formula);
            for (summary.invariant_formulas) |formula| try self.validateFormulaRef(formula);
            try self.validateLoopTimelineTermReferences(summary.preheader_timeline);
            for (summary.edges) |edge| {
                for (edge.target_formulas) |formula| try self.validateFormulaRef(formula);
                for (edge.next_state) |binding| try self.validateFormulaRef(binding.value);
                try self.validateLoopTimelineTermReferences(edge.timeline);
                for (edge.next_state) |binding| {
                    try self.validateLoopFormulaExpr(edge.timeline, edge.timeline.len, .{
                        .formula = binding.value,
                        .read_results = binding.read_results,
                    });
                }
            }
            for (summary.post_formulas) |formula| try self.validateFormulaRef(formula);
        }
    }

    fn validateLoopTimelineTermReferences(
        self: ObligationSet,
        events: []const LoopTimelineEvent,
    ) !void {
        for (events, 0..) |event, event_index| switch (event.payload) {
            .path_assumption => |path| for (path.formulas) |formula| {
                try self.validateLoopFormulaExpr(events, event_index, formula);
            },
            .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |formula| {
                try self.validateLoopFormulaExpr(events, event_index, formula);
            },
            .storage_read => |read| try self.validateLoopStorageLocation(events, event_index, read.location),
            .storage_write => |write| {
                try self.validateLoopStorageLocation(events, event_index, write.location);
                try self.validateLoopFormulaExpr(events, event_index, write.value);
            },
            .summary_dependency => {},
        };
    }

    fn validateKindTermRefs(self: ObligationSet, kind: Kind) !void {
        switch (kind) {
            .logical => |logical| try self.validateFormulaRef(logical.formula),
            .runtime_guard => |guard| try self.validateFormulaRef(guard.formula),
            .resource => |resource| if (resource.amount) |amount| try self.validateFormulaRef(amount),
            .quantifier,
            .type_wf,
            .type_relation,
            .region_relation,
            .effect_frame,
            .filtered_input,
            .backend_fact,
            => {},
        }
    }

    fn validateFormulaRef(self: ObligationSet, formula: FormulaRef) !void {
        switch (formula) {
            .origin_value => {},
            .term => |id| try self.validateTermId(id),
        }
    }

    fn validateLoopStorageLocation(
        self: ObligationSet,
        events: []const LoopTimelineEvent,
        event_index: usize,
        location: LoopStorageLocationExpr,
    ) !void {
        for (location.projections) |projection| switch (projection) {
            .field => {},
            .fixed_index => |fixed| try self.validateLoopFormulaExpr(
                events,
                event_index,
                fixed.index.value,
            ),
            .mapping_key => |key| try self.validateLoopFormulaExpr(events, event_index, key.value),
        };
    }

    pub fn fixedIndexBoundsReason(
        self: ObligationSet,
        events: []const LoopTimelineEvent,
    ) ?LoopUnsupportedReason {
        return timeline_validation.fixedIndexBoundsReason(self.terms, events);
    }

    fn validateLoopFormulaExpr(
        self: ObligationSet,
        events: []const LoopTimelineEvent,
        event_index: usize,
        expression: LoopFormulaExpr,
    ) !void {
        try self.validateFormulaRef(expression.formula);
        for (expression.read_results) |read_id| {
            if (!self.formulaReferencesFreeVar(expression.formula, read_id)) {
                return error.InvalidTimelineReadReference;
            }
        }
        for (events) |event| switch (event.payload) {
            .storage_read => |read| {
                if (!self.formulaReferencesFreeVar(expression.formula, read.result)) continue;
                if (!freeVarInList(read.result, expression.read_results)) {
                    return error.InvalidTimelineReadReference;
                }
                if (event.id >= event_index) return error.InvalidTimelineReadReference;
            },
            else => {},
        };
    }

    pub fn formulaReferencesFreeVar(self: ObligationSet, formula: FormulaRef, target: FreeVarId) bool {
        return switch (formula) {
            .origin_value => false,
            .term => |id| self.termReferencesFreeVar(id, target, self.terms.len + 1),
        };
    }

    fn termReferencesFreeVar(self: ObligationSet, id: TermId, target: FreeVarId, fuel: usize) bool {
        if (fuel == 0 or id >= self.terms.len) return false;
        return switch (self.terms[id]) {
            .variable => |variable| switch (variable) {
                .free => |free| freeVarIdEql(free.id, target),
                .bound => false,
            },
            .old => |operand| self.termReferencesFreeVar(operand, target, fuel - 1),
            .unary => |unary| self.termReferencesFreeVar(unary.operand, target, fuel - 1),
            .binary => |binary| self.termReferencesFreeVar(binary.lhs, target, fuel - 1) or
                self.termReferencesFreeVar(binary.rhs, target, fuel - 1),
            .refinement_predicate => |predicate| blk: {
                if (self.termReferencesFreeVar(predicate.value, target, fuel - 1)) break :blk true;
                for (predicate.args) |arg| {
                    if (self.termReferencesFreeVar(arg, target, fuel - 1)) break :blk true;
                }
                break :blk false;
            },
            .quantified => |quantified| blk: {
                if (quantified.condition) |condition| {
                    if (self.termReferencesFreeVar(condition, target, fuel - 1)) break :blk true;
                }
                break :blk self.termReferencesFreeVar(quantified.body, target, fuel - 1);
            },
            .bool_lit, .int_lit, .result, .place_read => false,
        };
    }

    fn validateTerm(self: ObligationSet, term: Term) !void {
        switch (term) {
            .bool_lit,
            .int_lit,
            .variable,
            .result,
            .place_read,
            => {},
            .old => |id| try self.validateTermId(id),
            .unary => |unary| try self.validateTermId(unary.operand),
            .binary => |binary| {
                try self.validateTermId(binary.lhs);
                try self.validateTermId(binary.rhs);
            },
            .refinement_predicate => |predicate| {
                try self.validateTermId(predicate.value);
                for (predicate.args) |arg| try self.validateTermId(arg);
            },
            .quantified => |quantified| {
                if (quantified.condition) |condition| try self.validateTermId(condition);
                try self.validateTermId(quantified.body);
            },
        }
    }

    fn validateTermId(self: ObligationSet, id: TermId) !void {
        if (id >= self.terms.len) return error.InvalidTermReference;
    }

    pub fn validateIdReferences(self: ObligationSet) !void {
        for (self.loop_summaries, 0..) |summary, summary_index| {
            for (self.loop_summaries[0..summary_index]) |previous| {
                if (previous.id == summary.id) return error.InvalidDependency;
            }
            try self.validateLoopSummaryDependencies(summary);
            try self.validateLoopQueryIds(summary);
        }
        for (self.obligations) |item| {
            for (item.dependencies) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            for (item.derived_from) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            switch (item.kind) {
                .effect_frame => |effect| {
                    for (effect.evidence) |evidence| {
                        if (!self.assumptionIdExists(evidence.assumption_id)) return error.InvalidDependency;
                    }
                },
                .filtered_input => |filtered| for (filtered.accepted_by) |id| {
                    if (!self.obligationIdExists(id) and !self.assumptionIdExists(id)) return error.InvalidDependency;
                },
                else => {},
            }
        }
        for (self.proof_artifacts) |artifact| {
            for (artifact.obligation_ids) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
        }
        for (self.queries, 0..) |query, query_index| {
            for (self.queries[0..query_index]) |candidate| {
                if (candidate.id == query.id) return error.InvalidDependency;
            }
            for (query.obligation_ids) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            for (query.assumption_ids) |id| {
                if (!self.assumptionIdExists(id)) return error.InvalidDependency;
            }
            if (query.proof_artifact_id) |artifact_id| {
                const artifact = self.proofArtifactById(artifact_id) orelse return error.InvalidDependency;
                if (query.backend != .lean) return error.InvalidDependency;
                for (query.obligation_ids) |id| {
                    if (!containsId(artifact.obligation_ids, id)) return error.InvalidDependency;
                }
            }
            if (query.loop_summary_id) |loop_summary_id| {
                if (query.kind != .loop_induction and
                    query.kind != .loop_base and
                    query.kind != .loop_invariant_step and
                    query.kind != .loop_body_safety and
                    query.kind != .loop_invariant_post)
                {
                    return error.InvalidDependency;
                }
                if (!self.loopSummaryIdExists(loop_summary_id)) return error.InvalidDependency;
                if (query.loop_vc_sha256 != null) {
                    if (query.kind == .loop_base and query.loop_edge_id != null) return error.InvalidDependency;
                    if (query.kind != .loop_base and query.kind != .loop_induction and query.loop_edge_id == null) {
                        return error.InvalidDependency;
                    }
                }
            }
        }
    }

    fn validateLoopSummaryDependencies(self: ObligationSet, summary: LoopSummaryRow) !void {
        for (summary.edges) |edge| {
            for (edge.timeline) |event| switch (event.payload) {
                .summary_dependency => |dependency| {
                    // Loop summaries are collected outer-before-inner. Requiring a
                    // strict forward reference both pins that ownership order and
                    // makes dependency cycles unrepresentable.
                    if (dependency.summary_id <= summary.id) return error.InvalidDependency;
                    if (!dependency.targetSupported()) return error.InvalidDependency;
                    const inner = findById(self.loop_summaries, dependency.summary_id) orelse
                        return error.InvalidDependency;
                    if (!inner.projectionSupported()) return error.InvalidDependency;

                    if (inner.query_ids.induction.len != 1) return error.InvalidDependency;
                    const induction = findById(self.queries, inner.query_ids.induction[0]) orelse
                        return error.InvalidDependency;
                    if (induction.kind != .loop_induction or induction.loop_summary_id != inner.id) {
                        return error.InvalidDependency;
                    }
                    const digest = induction.loop_vc_sha256 orelse
                        return error.InvalidDependency;
                    if (!std.mem.eql(u8, &digest, &dependency.loop_vc_sha256)) {
                        return error.InvalidDependency;
                    }

                    var target_exists = false;
                    for (inner.edges) |inner_edge| {
                        if (inner_edge.target == dependency.target) {
                            target_exists = true;
                            break;
                        }
                    }
                    if (!target_exists) return error.InvalidDependency;
                },
                else => {},
            };
        }
    }

    fn validateLoopQueryIds(self: ObligationSet, summary: LoopSummaryRow) !void {
        // Unsupported summaries carry diagnostic-only census rows and no proof
        // authority. Exact query inventory is mandatory once the VC is a
        // production certificate target.
        if (!summary.projectionSupported()) return;
        if (summary.query_ids.base.len != 1 or summary.query_ids.induction.len != 1) {
            return error.InvalidDependency;
        }
        try self.validateLoopQueryIdList(summary, summary.query_ids.base, .loop_base, null);
        try self.validateLoopQueryIdList(summary, summary.query_ids.induction, .loop_induction, null);

        var expected_step: usize = 0;
        var expected_post: usize = 0;
        if (summary.query_ids.body_safety.len != summary.edges.len) return error.InvalidDependency;
        for (summary.edges) |edge| switch (edge.target orelse return error.InvalidDependency) {
            .backedge => expected_step += 1,
            else => expected_post += 1,
        };
        if (summary.query_ids.step.len != expected_step or summary.query_ids.post.len != expected_post) {
            return error.InvalidDependency;
        }

        for (summary.edges, summary.query_ids.body_safety) |edge, id| {
            try self.validateLoopQueryIdList(summary, &.{id}, .loop_body_safety, edge.id);
        }
        var step_index: usize = 0;
        var post_index: usize = 0;
        for (summary.edges) |edge| switch (edge.target orelse return error.InvalidDependency) {
            .backedge => {
                try self.validateLoopQueryIdList(
                    summary,
                    &.{summary.query_ids.step[step_index]},
                    .loop_invariant_step,
                    edge.id,
                );
                step_index += 1;
            },
            else => {
                try self.validateLoopQueryIdList(
                    summary,
                    &.{summary.query_ids.post[post_index]},
                    .loop_invariant_post,
                    edge.id,
                );
                post_index += 1;
            },
        };

        if (summary.projectionSupported()) {
            const induction = findById(self.queries, summary.query_ids.induction[0]) orelse
                return error.InvalidDependency;
            const digest = induction.loop_vc_sha256 orelse return error.InvalidDependency;
            for (self.queries) |query| {
                if (query.loop_summary_id != summary.id) continue;
                if (!equalIdSlices(query.assumption_ids, induction.assumption_ids)) {
                    return error.InvalidDependency;
                }
                if (query.loop_vc_sha256 == null or !std.mem.eql(u8, &digest, &query.loop_vc_sha256.?)) {
                    return error.InvalidDependency;
                }
            }
        }
    }

    fn validateLoopQueryIdList(
        self: ObligationSet,
        summary: LoopSummaryRow,
        ids: []const Id,
        kind: VerificationQueryKind,
        edge_id: ?u32,
    ) !void {
        var previous: ?Id = null;
        for (ids) |id| {
            if (previous) |value| if (id <= value) return error.InvalidDependency;
            previous = id;
            const query = findById(self.queries, id) orelse return error.InvalidDependency;
            if (query.kind != kind or query.loop_summary_id != summary.id or query.loop_edge_id != edge_id) {
                return error.InvalidDependency;
            }
            if (query.loop_vc_sha256 != null) {
                if (kind == .loop_induction) {
                    if (query.backend != .unspecified or
                        query.artifact_policy != .blocks_verified_artifacts or
                        query.proof_requirement != .lean_certificate or
                        query.result != null)
                    {
                        return error.InvalidDependency;
                    }
                } else if (query.backend != .z3 or
                    query.artifact_policy != .blocks_verified_artifacts or
                    query.proof_requirement != .backend_result)
                {
                    return error.InvalidDependency;
                }
            }
        }
    }

    fn obligationIdExists(self: ObligationSet, id: Id) bool {
        for (self.obligations) |item| {
            if (item.id == id) return true;
        }
        return false;
    }

    fn assumptionIdExists(self: ObligationSet, id: Id) bool {
        for (self.assumptions) |item| {
            if (item.id == id) return true;
        }
        return false;
    }

    fn proofArtifactById(self: ObligationSet, id: Id) ?ProofArtifact {
        for (self.proof_artifacts) |item| {
            if (item.id == id) return item;
        }
        return null;
    }

    fn loopSummaryIdExists(self: ObligationSet, id: Id) bool {
        for (self.loop_summaries) |item| {
            if (item.id == id) return true;
        }
        return false;
    }

    fn hasSuccessfulProofFor(self: ObligationSet, item: Obligation) bool {
        for (self.queries) |query| {
            if (item.required_backend) |backend| {
                if (query.backend != backend) continue;
            }
            if (!containsId(query.obligation_ids, item.id)) continue;
            const result = query.result orelse continue;
            if (query.backend == .lean and !self.queryHasValidProofArtifact(query)) continue;
            if (querySucceeded(query, result)) return true;
        }
        return false;
    }

    fn queryUnknownDischargedByLean(self: ObligationSet, query: VerificationQuery) bool {
        if (query.backend != .z3) return false;
        // LoopVC authorization requires independent success from both proof
        // backends. A Lean certificate may never replace an inconclusive Z3
        // result for a loop-owned query.
        if (query.loop_summary_id != null) return false;
        if (query.obligation_ids.len == 0) return false;
        return self.hasSuccessfulLeanProofForQuery(query);
    }

    fn hasSuccessfulLeanProofForQuery(self: ObligationSet, target: VerificationQuery) bool {
        for (self.queries) |candidate| {
            if (candidate.backend != .lean) continue;
            if (candidate.discharges_query_id != target.id) continue;
            if (candidate.loop_summary_id != target.loop_summary_id) continue;
            if (!equalIdSlices(candidate.obligation_ids, target.obligation_ids)) continue;
            if (!equalIdSlices(candidate.assumption_ids, target.assumption_ids)) continue;
            if (!self.queryHasValidProofArtifact(candidate)) continue;
            const result = candidate.result orelse continue;
            if (querySucceeded(candidate, result)) return true;
        }
        return false;
    }

    fn queryHasValidProofArtifact(self: ObligationSet, query: VerificationQuery) bool {
        if (query.backend != .lean) return false;
        const artifact_id = query.proof_artifact_id orelse return false;
        const artifact = self.proofArtifactById(artifact_id) orelse return false;
        return equalIdSlices(artifact.obligation_ids, query.obligation_ids);
    }
};

pub const Obligation = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    kind: Kind,
    artifact_policy: ArtifactPolicy = .blocks_verified_artifacts,
    required_backend: ?VerificationBackend = null,
    dependencies: []const Id = &.{},
    derived_from: []const Id = &.{},

    pub fn blocksArtifacts(self: Obligation) bool {
        return self.artifact_policy == .blocks_verified_artifacts;
    }
};

pub const VerificationQuery = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    /// Diagnostic-only queries may carry verifier/source-accounting identity,
    /// but can neither block nor authorize artifact emission.
    artifact_policy: ArtifactPolicy = .blocks_verified_artifacts,
    proof_requirement: ProofRequirement = .backend_result,
    backend: VerificationBackend = .unspecified,
    kind: VerificationQueryKind,
    logical_role: ?LogicalRole = null,
    guard_id: ?[]const u8 = null,
    obligation_ids: []const Id = &.{},
    assumption_ids: []const Id = &.{},
    fragment: VerificationQueryFragment = .unknown,
    solver_logic: VerificationSolverLogic = .all,
    constraint_count: u32 = 0,
    smtlib_hash: ?u64 = null,
    canonical_smt_crosscheck_required: bool = false,
    canonical_smt_annotation_pure: bool = false,
    proof_artifact_id: ?Id = null,
    discharges_query_id: ?Id = null,
    loop_summary_id: ?Id = null,
    loop_edge_id: ?u32 = null,
    /// SHA-256 of the canonical LoopVC data, its term table, and the exact
    /// assumptions used by the loop-induction target.
    loop_vc_sha256: ?[32]u8 = null,
    result: ?VerificationQueryResult = null,
};

pub const FormalQueryBinding = struct {
    source_op_id: usize,
    loop_source_op_id: ?usize = null,
    kind: VerificationQueryKind,
    logical_role: ?LogicalRole = null,
    guard_id: ?[]const u8 = null,
    query_id: Id,
    assumption_ids: []const Id = &.{},
    obligation_ids: []const Id = &.{},
};

pub const VerificationBackend = enum(u8) {
    unspecified,
    z3,
    lean,
};

pub const VerificationQueryKind = enum(u8) {
    base,
    obligation,
    /// Canonical LoopVC base obligation. Unlike the verifier-wide `base`
    /// satisfiability query, success is UNSAT for its counterexample formula.
    loop_base,
    /// Compiler-owned proof target for the whole Base/Step/Safety/Exit
    /// induction bundle. It is independent of any one Z3 query row.
    loop_induction,
    loop_invariant_step,
    loop_body_safety,
    loop_invariant_post,
    guard_satisfy,
    guard_violate,
};

pub const VerificationQueryFragment = enum(u8) {
    unknown,
    qf_bv,
    qf_bv_array,
    aufbv,
    aufbv_quantifiers,
    other,
};

pub const VerificationSolverLogic = enum(u8) {
    all,
    qf_aufbv,
};

pub const VerificationQueryResult = struct {
    status: VerificationQueryStatus,
    vacuous: bool = false,
    vacuity_unknown: bool = false,
    degraded: bool = false,
};

pub const VerificationQueryStatus = enum(u8) {
    sat,
    unsat,
    unknown,
    proved,
    failed,
};

pub const VerificationQuerySummary = struct {
    base: u32 = 0,
    obligation: u32 = 0,
    loop_base: u32 = 0,
    loop_induction: u32 = 0,
    loop_invariant_step: u32 = 0,
    loop_body_safety: u32 = 0,
    loop_invariant_post: u32 = 0,
    guard_satisfy: u32 = 0,
    guard_violate: u32 = 0,

    pub fn add(self: *VerificationQuerySummary, kind: VerificationQueryKind) void {
        switch (kind) {
            .base => self.base += 1,
            .obligation => self.obligation += 1,
            .loop_base => self.loop_base += 1,
            .loop_induction => self.loop_induction += 1,
            .loop_invariant_step => self.loop_invariant_step += 1,
            .loop_body_safety => self.loop_body_safety += 1,
            .loop_invariant_post => self.loop_invariant_post += 1,
            .guard_satisfy => self.guard_satisfy += 1,
            .guard_violate => self.guard_violate += 1,
        }
    }

    pub fn fromQueries(queries: []const VerificationQuery) VerificationQuerySummary {
        var summary: VerificationQuerySummary = .{};
        for (queries) |query| summary.add(query.kind);
        return summary;
    }
};

pub const Assumption = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    kind: AssumptionKind,
    formula: ?FormulaRef = null,
};

pub const ProofArtifact = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    kind: ProofArtifactKind = .userland_lean,
    module_name: []const u8,
    theorem_name: []const u8,
    path: ?[]const u8 = null,
    content_hash: ?u64 = null,
    obligation_ids: []const Id = &.{},
};

pub const ProofArtifactKind = enum(u8) {
    userland_lean,
};

pub const ObligationDiagnostic = struct {
    kind: DiagnosticKind,
    source: SourceRef,
    message: []const u8,
    blocks_artifacts: bool = true,
};

pub const DiagnosticKind = enum(u8) {
    unsupported,
    missing_type,
    comparison_signedness_mismatch,
    missing_region,
    missing_effect_path,
    missing_formula,
    invalid_dependency,
    incomplete_resource_goals,
    unmatched_report_row,
    canonical_z3_unavailable,
    canonical_z3_mismatch,
};

pub const Phase = enum(u8) {
    sema,
    hir,
    ora_mlir,
    sir_mlir,
    sinora,
    report,
};

pub const ArtifactPolicy = enum(u8) {
    blocks_verified_artifacts,
    diagnostic_only,
};

pub const ProofRequirement = enum(u8) {
    backend_result,
    lean_certificate,
};

pub const ArtifactDecision = union(enum) {
    allowed,
    blocked: ArtifactBlockReason,

    pub fn isAllowed(self: ArtifactDecision) bool {
        return switch (self) {
            .allowed => true,
            .blocked => false,
        };
    }
};

pub const ArtifactBlockReason = enum(u8) {
    blocking_diagnostic,
    invalid_dependency,
    unknown_query,
    vacuous_query,
    degraded_query,
    failed_query,
    missing_proof,
    lean_required_failure,
    storage_projection_authority_unreviewed,
    storage_child_frame_authority_unreviewed,
    source_accounting_failure,
};

fn containsId(ids: []const Id, needle: Id) bool {
    for (ids) |id| {
        if (id == needle) return true;
    }
    return false;
}

fn SliceElement(comptime Slice: type) type {
    const info = @typeInfo(Slice);
    if (info != .pointer or info.pointer.size != .slice) {
        @compileError("findById expects a slice");
    }
    return info.pointer.child;
}

pub fn findById(rows: anytype, id: Id) ?SliceElement(@TypeOf(rows)) {
    for (rows) |row| {
        if (row.id == id) return row;
    }
    return null;
}

pub fn equalIdSlices(lhs: []const Id, rhs: []const Id) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (left != right) return false;
    }
    return true;
}

pub fn writeNatList(writer: anytype, values: anytype) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

pub fn kindFormula(kind: Kind) ?FormulaRef {
    return switch (kind) {
        .logical => |logical| logical.formula,
        .runtime_guard => |guard| guard.formula,
        .resource,
        .quantifier,
        .type_wf,
        .type_relation,
        .region_relation,
        .effect_frame,
        .filtered_input,
        .backend_fact,
        => null,
    };
}

fn querySucceeded(query: VerificationQuery, result: VerificationQueryResult) bool {
    return switch (result.status) {
        .proved => true,
        .failed, .unknown => false,
        .sat => switch (query.backend) {
            .lean => false,
            else => switch (query.kind) {
                .base, .guard_satisfy => true,
                .obligation,
                .loop_base,
                .loop_induction,
                .loop_invariant_step,
                .loop_body_safety,
                .loop_invariant_post,
                .guard_violate,
                => false,
            },
        },
        .unsat => switch (query.backend) {
            .lean => false,
            else => switch (query.kind) {
                .obligation,
                .loop_base,
                .loop_invariant_step,
                .loop_body_safety,
                .loop_invariant_post,
                .guard_violate,
                => true,
                .base, .loop_induction, .guard_satisfy => false,
            },
        },
    };
}

fn expectArtifactBlocked(expected: ArtifactBlockReason, decision: ArtifactDecision) !void {
    switch (decision) {
        .allowed => return error.TestUnexpectedResult,
        .blocked => |actual| try std.testing.expectEqual(expected, actual),
    }
}

pub const OwnerTag = enum(u8) {
    module,
    contract,
    function,
    trait_method,
    statement,
    backend,
};

pub const Owner = union(OwnerTag) {
    module: []const u8,
    contract: []const u8,
    function: FunctionOwner,
    trait_method: TraitMethodOwner,
    statement: StatementOwner,
    backend: BackendOwner,
};

pub const FunctionOwner = struct {
    module: ?[]const u8 = null,
    contract: ?[]const u8 = null,
    name: []const u8,
};

pub const TraitMethodOwner = struct {
    trait_name: []const u8,
    method_name: []const u8,
    impl_name: ?[]const u8 = null,
};

pub const StatementOwner = struct {
    function_name: []const u8,
    ordinal: u32,
};

pub const BackendOwner = struct {
    component: BackendComponent,
    name: []const u8,
};

pub const BackendComponent = enum(u8) {
    dispatcher,
    oratosir,
    sinora,
    artifact_policy,
};

/// Backend-neutral data required to denote a loop proof. Unresolved identities
/// and unsupported shapes remain explicit in `unsupported_reasons`; they can
/// never become proof targets.
pub const LoopSummaryRow = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    loop_source_op_id: ?u32,
    loop_kind: LoopKind,
    context_variables: []const LoopVariable = &.{},
    variables: []const LoopVariable = &.{},
    init_formulas: []const FormulaRef = &.{},
    invariant_formulas: []const FormulaRef = &.{},
    /// Storage actions from function entry to this loop's preheader.  The
    /// first storage-authority boundary accepts only a complete empty trace.
    preheader_timeline: []const LoopTimelineEvent = &.{},
    preheader_projection_complete: bool = false,
    /// Canonical semantic edges. These are the proof authority for paths,
    /// safety, state transition, and control outcome. Ordering is part of the
    /// certificate identity and must already be canonical.
    edges: []const LoopEdgeRow = &.{},
    /// Result of comparing the independently collected structured-control
    /// inventory with the semantic edge projection.  A proof target cannot be
    /// armed unless the two inventories agree exactly.
    projection_completeness: LoopProjectionCompleteness = .unavailable,
    post_formulas: []const FormulaRef = &.{},
    query_ids: LoopQueryIds = .{},
    unsupported_reasons: []const LoopUnsupportedReason = &.{},

    pub fn projectionSupported(self: LoopSummaryRow) bool {
        return self.unsupported_reasons.len == 0 and self.validationReason() == null;
    }

    pub fn preheaderTimelineValidationReason(self: LoopSummaryRow) ?LoopUnsupportedReason {
        return timeline_validation.timelineReason(self, self.preheader_timeline, false, false);
    }

    pub fn validationReason(self: LoopSummaryRow) ?LoopUnsupportedReason {
        switch (self.projection_completeness) {
            .complete => {},
            .unavailable => return .loop_projection_inventory_missing,
            .control_manifest_missing => return .loop_control_manifest_missing,
            .control_marker_missing => return .loop_control_marker_missing,
            .control_marker_malformed => return .loop_control_marker_malformed,
            .control_marker_duplicate => return .loop_control_marker_duplicate,
            .control_marker_target_mismatch => return .loop_control_marker_target_mismatch,
            .path_mismatch => return .loop_projection_path_mismatch,
            .effect_mismatch => return .loop_projection_effect_mismatch,
            .binding_mismatch => return .loop_projection_binding_mismatch,
            .target_mismatch => return .loop_projection_target_mismatch,
            .ambiguous_identity => return .loop_projection_identity_ambiguous,
            .unsupported_control_flow => return .loop_projection_control_flow_unsupported,
            .unsupported_effect => return .loop_projection_effect_unsupported,
            .path_limit_exceeded => return .loop_projection_path_limit_exceeded,
        }
        if (self.variables.len == 0) return .loop_state_empty;
        if (self.edges.len == 0) return .loop_edge_missing;

        for (self.variables, 0..) |variable, index| {
            if (variable.index != index or variable.id == null) return .loop_identity_missing;
            for (self.variables[0..index]) |previous| {
                if (previous.id != null and freeVarIdEql(previous.id.?, variable.id.?)) {
                    return .loop_duplicate_state_identity;
                }
            }
        }

        if (self.preheaderTimelineValidationReason()) |reason| return reason;

        var has_backedge = false;
        var has_exit = false;
        for (self.edges, 0..) |edge, edge_index| {
            for (self.edges[0..edge_index]) |previous| {
                if (previous.id == edge.id) return .loop_duplicate_edge_identity;
            }
            if (edge.id != edge_index) return .loop_edge_order_invalid;
            if (edge.structural_key.len == 0) return .loop_projection_identity_missing;
            for (self.edges[0..edge_index]) |previous| {
                if (std.mem.eql(u8, previous.structural_key, edge.structural_key)) {
                    return .loop_projection_identity_ambiguous;
                }
            }
            if (edge.timeline.len == 0) return .loop_timeline_missing;
            const target = edge.target orelse return .loop_edge_target_unresolved;
            switch (target) {
                .backedge => {
                    has_backedge = true;
                    if (edge.target_formulas.len != 0) return .loop_formula_unsupported;
                },
                .successor => {
                    has_exit = true;
                    if (edge.target_formulas.len != 0) return .loop_formula_unsupported;
                },
                .function_return, .error_return => has_exit = true,
                .explicit_revert => {},
            }
            if (edge.next_state.len < self.variables.len) return .loop_state_binding_missing;
            if (edge.next_state.len > self.variables.len) return .loop_state_binding_extra;
            for (edge.next_state, self.variables, 0..) |binding, variable, binding_index| {
                if (binding.variable_index != binding_index or binding.target == null) {
                    return .loop_state_binding_invalid;
                }
                if (binding.source_key.len == 0) return .loop_state_binding_source_missing;
                for (edge.next_state[0..binding_index]) |previous| {
                    if (previous.target != null and freeVarIdEql(previous.target.?, binding.target.?)) {
                        return .loop_state_binding_duplicate;
                    }
                }
                if (!freeVarIdEql(binding.target.?, variable.id.?)) {
                    return .loop_state_binding_foreign;
                }
            }
            if (timeline_validation.edgeReason(self, edge)) |reason| return reason;
        }
        if (!has_backedge) return .loop_backedge_missing;
        if (!has_exit) return .loop_exit_edge_missing;
        return null;
    }
};

/// Loop proof claims are made about the Ora MLIR module supplied to the
/// verifier before Ora-to-SIR conversion. The frontend, projection from this
/// module into formal rows, and later backend lowering remain trusted.
pub const loop_verification_boundary = "verifier_input_ora_mlir";

pub const LoopKind = enum(u8) {
    scf_while,
    scf_for,
    other,
};

pub const LoopVariable = struct {
    index: u32,
    id: ?FreeVarId = null,
    name: ?[]const u8 = null,
    ty: TypeRef,
};

pub const LoopStepAssignment = struct {
    variable_index: u32,
    target: ?FreeVarId = null,
    /// Deterministic identity of the verifier-input MLIR value supplying this
    /// post-state component on the containing structural path.
    source_key: []const u8 = "",
    value: FormulaRef,
    read_results: []const FreeVarId = &.{},
};

pub const LoopControlTarget = enum(u8) {
    backedge,
    successor,
    function_return,
    error_return,
    explicit_revert,
};

pub const LoopProjectionCompleteness = enum(u8) {
    unavailable,
    complete,
    control_manifest_missing,
    control_marker_missing,
    control_marker_malformed,
    control_marker_duplicate,
    control_marker_target_mismatch,
    path_mismatch,
    effect_mismatch,
    binding_mismatch,
    target_mismatch,
    ambiguous_identity,
    unsupported_control_flow,
    unsupported_effect,
    path_limit_exceeded,
};

pub const LoopStorageProjectionTag = enum(u8) {
    field,
    fixed_index,
    mapping_key,
};

pub const LoopFormulaExpr = struct {
    formula: FormulaRef,
    /// Read-result identities referenced by this expression. The manifest
    /// validator checks these declarations against the term DAG and rejects
    /// foreign or forward identities.
    read_results: []const FreeVarId = &.{},
};

pub const LoopTypedFormula = struct {
    value: LoopFormulaExpr,
    ty: TypeRef,
};

pub const LoopFixedIndexProjection = struct {
    index: LoopTypedFormula,
    /// Exact positive extent of the resolved fixed-array dimension.
    extent: u64,
};

/// One ordered projection from a persistent storage root. Keeping field and
/// key projections in one sequence distinguishes `array[i].field` from
/// `field.array[i]` and keeps fixed-array and mapping domains distinct.
pub const LoopStorageProjection = union(LoopStorageProjectionTag) {
    field: []const u8,
    fixed_index: LoopFixedIndexProjection,
    mapping_key: LoopTypedFormula,
};

pub const LoopStorageLocationExpr = struct {
    root: []const u8,
    projections: []const LoopStorageProjection = &.{},
    value_ty: TypeRef,
};

pub const LoopTimelineFormulaEvent = struct {
    formulas: []const LoopFormulaExpr,
};

pub const LoopStorageReadEvent = struct {
    result: FreeVarId,
    location: LoopStorageLocationExpr,
};

pub const LoopStorageWriteEvent = struct {
    location: LoopStorageLocationExpr,
    value: LoopFormulaExpr,
};

pub const LoopTimelineEventTag = enum(u8) {
    path_assumption,
    safety_checkpoint,
    storage_read,
    storage_write,
    summary_dependency,
};

pub const LoopTimelineEventPayload = union(LoopTimelineEventTag) {
    path_assumption: LoopTimelineFormulaEvent,
    safety_checkpoint: LoopTimelineFormulaEvent,
    storage_read: LoopStorageReadEvent,
    storage_write: LoopStorageWriteEvent,
    summary_dependency: LoopSummaryDependency,
};

pub const LoopTimelineEvent = struct {
    id: u32,
    structural_key: []const u8,
    payload: LoopTimelineEventPayload,
};

pub const LoopEdgeRow = struct {
    id: u32,
    /// Deterministic identity of the verifier-input MLIR path ending at this
    /// edge.  It is proof-relevant certificate data, not a display label.
    structural_key: []const u8 = "",
    /// Single proof-relevant execution order for assumptions, safety checks,
    /// storage actions, and certified child summaries.
    timeline: []const LoopTimelineEvent = &.{},
    next_state: []const LoopStepAssignment = &.{},
    target: ?LoopControlTarget = null,
    /// Edge-specific assertion for function and error returns. Ordinary loop
    /// successors use `LoopSummaryRow.post_formulas`; backedges carry none.
    target_formulas: []const FormulaRef = &.{},

    pub fn safetyFormulaCount(self: LoopEdgeRow) usize {
        var count: usize = 0;
        for (self.timeline) |event| switch (event.payload) {
            .safety_checkpoint => |checkpoint| count += checkpoint.formulas.len,
            else => {},
        };
        return count;
    }
};

fn freeVarInList(id: FreeVarId, values: []const FreeVarId) bool {
    for (values) |candidate| if (freeVarIdEql(id, candidate)) return true;
    return false;
}

pub const LoopSafetyCheckpoint = struct {
    id: u32,
    /// Deterministic identity of the verifier-input safety operation.
    structural_key: []const u8,
    after_dependency_count: u32,
    formulas: []const FormulaRef,
};

pub const LoopSummaryDependency = struct {
    id: u32,
    /// Deterministic verifier-input MLIR identity of the nested loop operation
    /// on the containing outer path.
    structural_key: []const u8,
    summary_id: Id,
    loop_vc_sha256: [32]u8,
    /// Certified over-approximation of persistent locations the child may
    /// mutate. The child's certificate binds this list to its LoopVC digest.
    write_footprint: []const PlaceRef = &.{},
    write_footprint_complete: bool = false,
    /// Exact terminating control relation consumed from the inner summary.
    target: LoopControlTarget,

    pub fn targetSupported(self: LoopSummaryDependency) bool {
        return switch (self.target) {
            .successor, .function_return, .error_return => true,
            .backedge, .explicit_revert => false,
        };
    }
};

test "loop projection completeness and structural identities fail closed" {
    const loop_id: FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const variables = [_]LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .ty = .{ .spelling = "u256" },
    }};
    const bindings = [_]LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const path_exprs = [_]LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const backedge_timeline = [_]LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "body/straight-line-yield/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const exit_timeline = [_]LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "guard/false-successor/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const canonical_edges = [_]LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/straight-line-yield",
            .timeline = &backedge_timeline,
            .next_state = &bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &exit_timeline,
            .next_state = &bindings,
            .target = .successor,
        },
    };
    var row: LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &canonical_edges,
    };

    try std.testing.expectEqual(LoopUnsupportedReason.loop_projection_inventory_missing, row.validationReason().?);
    row.projection_completeness = .complete;
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), row.validationReason());

    var missing_key_edges = canonical_edges;
    missing_key_edges[0].structural_key = "";
    row.edges = &missing_key_edges;
    try std.testing.expectEqual(LoopUnsupportedReason.loop_projection_identity_missing, row.validationReason().?);

    var duplicate_key_edges = canonical_edges;
    duplicate_key_edges[1].structural_key = duplicate_key_edges[0].structural_key;
    row.edges = &duplicate_key_edges;
    try std.testing.expectEqual(LoopUnsupportedReason.loop_projection_identity_ambiguous, row.validationReason().?);

    row.edges = &canonical_edges;
    row.projection_completeness = .effect_mismatch;
    try std.testing.expectEqual(LoopUnsupportedReason.loop_projection_effect_mismatch, row.validationReason().?);
    row.projection_completeness = .unsupported_effect;
    try std.testing.expectEqual(LoopUnsupportedReason.loop_projection_effect_unsupported, row.validationReason().?);
}

test "loop summary dependencies are ordered bound and fail closed before denotation" {
    const loop_id: FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const variables = [_]LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .ty = .{ .spelling = "u256" },
    }};
    const bindings = [_]LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const path_exprs = [_]LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const digest = [_]u8{7} ** 32;
    const dependency: LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "body/op:4/scf.while",
        .summary_id = 2,
        .loop_vc_sha256 = digest,
        .write_footprint_complete = true,
        .target = .successor,
    };
    const outer_timeline = [_]LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "body/yield/path",
            .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
        },
        .{
            .id = 1,
            .structural_key = dependency.structural_key,
            .payload = .{ .summary_dependency = dependency },
        },
    };
    const outer_exit_timeline = [_]LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "guard/false-successor/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    var outer_edges = [_]LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/yield",
            .timeline = &outer_timeline,
            .next_state = &bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &outer_exit_timeline,
            .next_state = &bindings,
            .target = .successor,
        },
    };
    var outer: LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .statement = .{ .function_name = "nested", .ordinal = 1 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &outer_edges,
        .projection_completeness = .complete,
    };
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), outer.validationReason());

    const safety_exprs = [_]LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const outer_timeline_with_checkpoint = [_]LoopTimelineEvent{
        outer_timeline[0],
        outer_timeline[1],
        .{
            .id = 2,
            .structural_key = "body/op:6/ora.assert",
            .payload = .{ .safety_checkpoint = .{ .formulas = &safety_exprs } },
        },
    };
    outer_edges[0].timeline = &outer_timeline_with_checkpoint;
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), outer.validationReason());

    var malformed = dependency;
    malformed.id = 1;
    var malformed_timeline = outer_timeline;
    malformed_timeline[1].payload.summary_dependency = malformed;
    outer_edges[0].timeline = &malformed_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_summary_dependency_order_invalid,
        outer.validationReason().?,
    );
    malformed = dependency;
    malformed.target = .backedge;
    malformed_timeline = outer_timeline;
    malformed_timeline[1].payload.summary_dependency = malformed;
    outer_edges[0].timeline = &malformed_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_summary_dependency_target_unsupported,
        outer.validationReason().?,
    );

    outer_edges[0].timeline = &outer_timeline;
    const inner_edges = [_]LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "inner/body/yield",
            .timeline = &.{.{
                .id = 0,
                .structural_key = "inner/body/yield/path",
                .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
            }},
            .next_state = &bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "inner/guard/false-successor",
            .timeline = &.{.{
                .id = 0,
                .structural_key = "inner/guard/false-successor/path",
                .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
            }},
            .next_state = &bindings,
            .target = .successor,
        },
    };
    const inner: LoopSummaryRow = .{
        .id = 2,
        .owner = .{ .statement = .{ .function_name = "nested", .ordinal = 2 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 2,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &inner_edges,
        .projection_completeness = .complete,
        .query_ids = .{ .induction = &.{9} },
    };
    const induction: VerificationQuery = .{
        .id = 9,
        .owner = inner.owner,
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .proof_requirement = .lean_certificate,
        .kind = .loop_induction,
        .loop_summary_id = inner.id,
        .loop_vc_sha256 = digest,
    };
    const summaries = [_]LoopSummaryRow{ outer, inner };
    const set: ObligationSet = .{ .loop_summaries = &summaries, .queries = &.{induction} };
    try set.validateLoopSummaryDependencies(outer);

    var stale = dependency;
    stale.loop_vc_sha256 = [_]u8{8} ** 32;
    var stale_timeline = outer_timeline;
    stale_timeline[1].payload.summary_dependency = stale;
    outer_edges[0].timeline = &stale_timeline;
    var stale_outer = outer;
    stale_outer.edges = &outer_edges;
    try std.testing.expectError(error.InvalidDependency, set.validateLoopSummaryDependencies(stale_outer));

    var cyclic = dependency;
    cyclic.summary_id = outer.id;
    var cyclic_timeline = outer_timeline;
    cyclic_timeline[1].payload.summary_dependency = cyclic;
    outer_edges[0].timeline = &cyclic_timeline;
    var cyclic_outer = outer;
    cyclic_outer.edges = &outer_edges;
    try std.testing.expectError(error.InvalidDependency, set.validateLoopSummaryDependencies(cyclic_outer));
}

test "ordered loop timeline schema rejects identity order and read-reference mutations" {
    const loop_id: FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const read_id: FreeVarId = .{ .file_id = 1, .pattern_id = 90 };
    const foreign_id: FreeVarId = .{ .file_id = 1, .pattern_id = 91 };
    const ty: TypeRef = .{ .compiler_type_id = 6 };
    const terms = [_]Term{
        .{ .bool_lit = true },
        .{ .variable = .{ .free = .{ .id = read_id, .name = "read", .ty = ty } } },
        .{ .int_lit = .{ .value = "0", .ty = ty } },
    };
    const path_exprs = [_]LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const read_refs = [_]FreeVarId{read_id};
    const location: LoopStorageLocationExpr = .{ .root = "counter", .value_ty = ty };
    const timeline = [_]LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "body/path",
            .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
        },
        .{
            .id = 1,
            .structural_key = "body/read",
            .payload = .{ .storage_read = .{ .result = read_id, .location = location } },
        },
        .{
            .id = 2,
            .structural_key = "body/write",
            .payload = .{ .storage_write = .{
                .location = location,
                .value = .{ .formula = .{ .term = 1 }, .read_results = &read_refs },
            } },
        },
    };
    const variables = [_]LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .ty = ty,
    }};
    const bindings = [_]LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 1 },
        .read_results = &read_refs,
    }};
    const exit_bindings = [_]LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 2 },
    }};
    var edges = [_]LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/yield",
            .timeline = &timeline,
            .next_state = &bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &.{.{
                .id = 0,
                .structural_key = "guard/false-successor/path",
                .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
            }},
            .next_state = &exit_bindings,
            .target = .successor,
        },
    };
    var summary: LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &edges,
        .projection_completeness = .complete,
    };
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), summary.validationReason());
    const set: ObligationSet = .{ .terms = &terms, .loop_summaries = &.{summary} };
    try set.validateTermReferences();

    const path_event = timeline[0];
    const read_event = timeline[1];
    const write_first_event: LoopTimelineEvent = .{
        .id = 1,
        .structural_key = "body/write",
        .payload = .{ .storage_write = .{
            .location = location,
            .value = .{ .formula = .{ .term = 2 } },
        } },
    };
    const read_second_event: LoopTimelineEvent = .{
        .id = 2,
        .structural_key = "body/read",
        .payload = read_event.payload,
    };
    const second_write_event: LoopTimelineEvent = .{
        .id = 2,
        .structural_key = "body/write:1",
        .payload = write_first_event.payload,
    };
    const checkpoint_event: LoopTimelineEvent = .{
        .id = 1,
        .structural_key = "body/check",
        .payload = .{ .safety_checkpoint = .{ .formulas = &path_exprs } },
    };
    const dependency: LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "body/child",
        .summary_id = 2,
        .loop_vc_sha256 = [_]u8{7} ** 32,
        .write_footprint_complete = true,
        .target = .successor,
    };
    const dependency_event: LoopTimelineEvent = .{
        .id = 1,
        .structural_key = dependency.structural_key,
        .payload = .{ .summary_dependency = dependency },
    };
    const accepted_timelines = [_][]const LoopTimelineEvent{
        &.{path_event},
        &.{ path_event, read_event },
        &.{ path_event, write_first_event },
        &timeline,
        &.{ path_event, write_first_event, read_second_event },
        &.{ path_event, write_first_event, second_write_event },
    };
    edges[0].next_state = &exit_bindings;
    for (accepted_timelines) |accepted| {
        edges[0].timeline = accepted;
        try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), summary.validationReason());
    }
    edges[0].timeline = &.{ path_event, checkpoint_event };
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), summary.validationReason());
    edges[0].timeline = &.{ path_event, dependency_event };
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), summary.validationReason());
    edges[0].next_state = &bindings;

    var malformed_timeline = timeline;
    malformed_timeline[1].id = 7;
    edges[0].timeline = &malformed_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_timeline_event_order_invalid,
        summary.validationReason().?,
    );

    malformed_timeline = timeline;
    malformed_timeline[1].structural_key = timeline[0].structural_key;
    edges[0].timeline = &malformed_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_timeline_event_identity_duplicate,
        summary.validationReason().?,
    );

    const forward_timeline = [_]LoopTimelineEvent{
        timeline[0],
        .{
            .id = 1,
            .structural_key = "body/write",
            .payload = timeline[2].payload,
        },
        .{
            .id = 2,
            .structural_key = "body/read",
            .payload = timeline[1].payload,
        },
    };
    edges[0].timeline = &forward_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_storage_read_identity_forward,
        summary.validationReason().?,
    );

    var foreign_refs = [_]FreeVarId{foreign_id};
    var foreign_write = timeline[2];
    foreign_write.payload.storage_write.value.read_results = &foreign_refs;
    const foreign_timeline = [_]LoopTimelineEvent{ timeline[0], timeline[1], foreign_write };
    edges[0].timeline = &foreign_timeline;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_storage_read_identity_foreign,
        summary.validationReason().?,
    );

    edges[0].timeline = &timeline;
    var undeclared_binding = bindings;
    undeclared_binding[0].read_results = &.{};
    edges[0].next_state = &undeclared_binding;
    var invalid_summary = summary;
    invalid_summary.edges = &edges;
    const invalid_set: ObligationSet = .{ .terms = &terms, .loop_summaries = &.{invalid_summary} };
    try std.testing.expectError(error.InvalidTimelineReadReference, invalid_set.validateTermReferences());

    var preheader_read = read_event;
    preheader_read.id = 0;
    preheader_read.structural_key = "preheader/read";
    var preheader_write = timeline[2];
    preheader_write.id = 1;
    preheader_write.structural_key = "preheader/write";
    const preheader = [_]LoopTimelineEvent{ preheader_read, preheader_write };
    summary.preheader_timeline = &preheader;
    edges[0].timeline = &timeline;
    edges[0].next_state = &bindings;
    try std.testing.expectEqual(@as(?LoopUnsupportedReason, null), summary.validationReason());
    try (ObligationSet{ .terms = &terms, .loop_summaries = &.{summary} }).validateTermReferences();

    const forward_preheader = [_]LoopTimelineEvent{
        .{ .id = 0, .structural_key = preheader_write.structural_key, .payload = preheader_write.payload },
        .{ .id = 1, .structural_key = preheader_read.structural_key, .payload = preheader_read.payload },
    };
    summary.preheader_timeline = &forward_preheader;
    try std.testing.expectEqual(
        LoopUnsupportedReason.loop_storage_read_identity_forward,
        summary.preheaderTimelineValidationReason().?,
    );

    preheader_write.payload.storage_write.value.read_results = &.{};
    const undeclared_preheader = [_]LoopTimelineEvent{ preheader_read, preheader_write };
    summary.preheader_timeline = &undeclared_preheader;
    try std.testing.expectError(
        error.InvalidTimelineReadReference,
        (ObligationSet{ .terms = &terms, .loop_summaries = &.{summary} }).validateTermReferences(),
    );
}

pub const LoopQueryIds = struct {
    induction: []const Id = &.{},
    base: []const Id = &.{},
    step: []const Id = &.{},
    body_safety: []const Id = &.{},
    post: []const Id = &.{},
};

pub const LoopUnsupportedReason = enum(u8) {
    loop_has_storage_write,
    loop_has_storage_read,
    loop_has_call,
    loop_has_external_call,
    loop_has_resource_operation,
    loop_has_break_or_continue,
    loop_has_error_control_flow,
    loop_has_nested_loop,
    loop_has_branching_body,
    loop_variable_not_registered_integer,
    loop_update_not_scalar_assignment,
    loop_formula_unsupported,
    loop_identity_missing,
    loop_query_not_owner_scoped,
    loop_guard_missing,
    loop_invariant_missing,
    loop_kind_unsupported,
    loop_duplicate_update,
    loop_update_target_not_loop_variable,
    loop_body_safety_missing,
    loop_state_empty,
    loop_duplicate_state_identity,
    loop_edge_missing,
    loop_duplicate_edge_identity,
    loop_edge_order_invalid,
    loop_edge_path_missing,
    loop_edge_target_unresolved,
    loop_control_target_unsupported,
    loop_backedge_missing,
    loop_exit_edge_missing,
    loop_state_binding_missing,
    loop_state_binding_extra,
    loop_state_binding_duplicate,
    loop_state_binding_invalid,
    loop_state_binding_source_missing,
    loop_state_binding_foreign,
    loop_projection_inventory_missing,
    loop_control_manifest_missing,
    loop_control_marker_missing,
    loop_control_marker_malformed,
    loop_control_marker_duplicate,
    loop_control_marker_target_mismatch,
    loop_projection_path_mismatch,
    loop_projection_effect_mismatch,
    loop_projection_binding_mismatch,
    loop_projection_target_mismatch,
    loop_projection_identity_missing,
    loop_projection_identity_ambiguous,
    loop_projection_control_flow_unsupported,
    loop_projection_effect_unsupported,
    loop_projection_path_limit_exceeded,
    loop_preheader_storage_relation_unsupported,
    loop_preheader_control_unsupported,
    loop_preheader_effect_unsupported,
    loop_preheader_value_unsupported,
    loop_preheader_location_unsupported,
    loop_preheader_inventory_mismatch,
    loop_summary_dependency_order_invalid,
    loop_summary_dependency_target_unsupported,
    loop_summary_dependency_value_out_of_scope,
    loop_timeline_missing,
    loop_timeline_event_order_invalid,
    loop_timeline_event_identity_missing,
    loop_timeline_event_identity_duplicate,
    loop_timeline_path_missing,
    loop_storage_root_missing,
    loop_storage_projection_invalid,
    loop_fixed_index_extent_invalid,
    loop_fixed_index_bounds_checkpoint_missing,
    loop_fixed_index_bounds_formula_shape_mismatch,
    loop_fixed_index_bounds_index_identity_mismatch,
    loop_fixed_index_bounds_type_mismatch,
    loop_fixed_index_bounds_extent_literal_mismatch,
    loop_storage_type_invalid,
    loop_storage_read_identity_duplicate,
    loop_storage_read_identity_foreign,
    loop_storage_read_identity_forward,
    loop_storage_child_footprint_missing,
    loop_value_type_unresolved,
    loop_value_width_mismatch,
    loop_value_signedness_mismatch,
};

pub const SourceRef = struct {
    file: ?[]const u8 = null,
    line: u32 = 0,
    column: u32 = 0,
    byte_start: u32 = 0,
    byte_end: u32 = 0,

    pub fn generated() SourceRef {
        return .{};
    }
};

pub const OriginTag = enum(u8) {
    source,
    sema_fact,
    mlir_op,
    effect_slot,
    resource_op,
    backend_fact,
};

pub const Origin = union(OriginTag) {
    source,
    sema_fact: SemaFactRef,
    mlir_op: MlirStableRef,
    effect_slot: EffectRef,
    resource_op: ResourceOpRef,
    backend_fact: BackendFactRef,
};

pub const SemaFactRef = struct {
    kind: []const u8,
    ordinal: u32,
};

pub const MlirStableRef = struct {
    op_name: []const u8,
    symbol: ?[]const u8 = null,
    ordinal: u32 = 0,
};

pub const EffectRef = struct {
    slot: PlaceRef,
    access: EffectAccess,
};

pub const EffectAccess = enum(u8) {
    read,
    write,
    read_write,
    lock,
    unlock,
};

pub const ResourceOpRef = struct {
    op: ResourceOperation,
    domain: []const u8,
    ordinal: u32 = 0,
};

pub const BackendFactRef = struct {
    component: BackendComponent,
    fact: []const u8,
    ordinal: u32 = 0,
};

pub const KindTag = enum(u8) {
    logical,
    runtime_guard,
    type_wf,
    type_relation,
    region_relation,
    effect_frame,
    resource,
    quantifier,
    filtered_input,
    backend_fact,
};

pub const Kind = union(KindTag) {
    logical: LogicalGoal,
    runtime_guard: RuntimeGuardGoal,
    type_wf: TypeWellFormedGoal,
    type_relation: TypeRelationGoal,
    region_relation: RegionRelationGoal,
    effect_frame: EffectFrameGoal,
    resource: ResourceGoal,
    quantifier: QuantifierGoal,
    filtered_input: FilteredInputGoal,
    backend_fact: BackendFactGoal,
};

pub const LogicalGoal = struct {
    role: LogicalRole,
    formula: FormulaRef,
    arithmetic_safety: ?ArithmeticSafetyKind = null,
};

pub const LogicalRole = enum(u8) {
    invariant,
    requires,
    callee_precondition,
    ensures,
    ensures_ok,
    ensures_err,
    assert,
    guard,
    loop_invariant,
    contract_invariant,
    arithmetic_safety,
    refinement,
    imported_callee_obligation,
    imported_callee_ensures,
};

pub const ArithmeticSafetyKind = enum(u8) {
    addition_overflow,
    subtraction_overflow,
    multiplication_overflow,
    power_overflow,
    negation_overflow,
    division_by_zero,
    signed_division_overflow,
    shift_amount_bounds,
    fixed_array_bounds,
};

pub const RuntimeGuardGoal = struct {
    guard_id: []const u8,
    formula: FormulaRef,
    erasure: GuardErasurePolicy = .may_elide_if_proven,
};

pub const GuardErasurePolicy = enum(u8) {
    may_elide_if_proven,
    always_runtime,
};

pub const TypeWellFormedGoal = struct {
    ty: TypeRef,
};

pub const TypeRelationGoal = struct {
    relation: TypeRelation,
    lhs: TypeRef,
    rhs: TypeRef,
};

pub const TypeRelation = enum(u8) {
    eql,
    assignable,
};

pub const RegionRelationGoal = struct {
    relation: RegionRelation,
    from: RegionRef,
    to: RegionRef,
};

pub const RegionRelation = enum(u8) {
    assignable,
    eql,
};

pub const EffectFrameGoal = struct {
    relation: EffectFrameRelation,
    declared: []const PlaceRef = &.{},
    actual: []const PlaceRef = &.{},
    evidence: []const KeyDisjointEvidence = &.{},
};

pub const EffectFrameRelation = enum(u8) {
    write_covered_by_modifies,
    read_preserved_by_frame,
    read_preserved_by_key_evidence,
    lock_covers_write,
    external_call_frame,
};

pub const KeyDisjointEvidenceKind = enum(u8) {
    free_var_disequality,
};

pub const KeyDisjointEvidence = struct {
    kind: KeyDisjointEvidenceKind,
    assumption_id: Id,
    lhs: FreeVarId,
    rhs: FreeVarId,
    read: PlaceRef,
    write: PlaceRef,
    key_index: u32,
};

pub const ResourceGoal = struct {
    op: ResourceOperation,
    domain: []const u8,
    source: ?PlaceRef = null,
    destination: ?PlaceRef = null,
    amount: ?FormulaRef = null,
    property: ResourceProperty,
};

pub const ResourceOperation = enum(u8) {
    move,
    create,
    destroy,
};

pub const ResourceProperty = enum(u8) {
    amount_non_negative,
    source_sufficient,
    destination_no_overflow,
    same_place_identity,
    conservation,
};

pub const QuantifierGoal = struct {
    quantifier: Quantifier,
    variable: []const u8,
    binder_type: TypeRef,
    binder_sort: QuantifierBinderSort,
    fragment: VerificationQueryFragment,
    pattern_status: QuantifierPatternStatus,
    degradation: ?QuantifierDegradation = null,
};

pub const QuantifierBinderSort = enum(u8) {
    bool,
    bit_vector,
    byte_sequence,
    opaque_unknown,
};

pub const QuantifierPatternStatus = enum(u8) {
    explicit,
    synthesized,
    absent,
};

pub const QuantifierDegradation = enum(u8) {
    unsupported_binder_type,
    malformed_binder_width,
};

pub const FilteredInputGoal = struct {
    value: VarRef,
    sink: PlaceRef,
    accepted_by: []const Id = &.{},
};

pub const BackendFactGoal = struct {
    component: BackendComponent,
    property: BackendProperty,
};

pub const BackendProperty = enum(u8) {
    complete,
    disjoint,
    ordered,
    preserves_selector_behavior,
    no_unknown_strategy,
    dependency_valid,
};

pub const AssumptionKind = enum(u8) {
    requires,
    assume,
    path_assume,
    env_assume,
    binding,
    two_state_linkage,
    frame,
    loop_invariant,
    callee_obligation,
    callee_ensures,
    ghost_axiom,
    goal,
};

pub const FormulaRefTag = enum(u8) {
    origin_value,
    term,
};

pub const FormulaRef = union(FormulaRefTag) {
    /// Formula is still owned by a compiler IR operation/value. Adapters may use
    /// this to encode directly from MLIR without duplicating that IR here.
    origin_value: ValueRef,
    /// Formula was projected into the small proof-term fragment below for
    /// serialization or Lean/Z3 export. This is not an optimization target.
    term: TermId,
};

pub const ValueRef = struct {
    origin: Origin,
    kind: ValueRefKind = .result,
    index: u32 = 0,
};

pub const ValueRefKind = enum(u8) {
    result,
    operand,
    derived,
};

pub const TermTag = enum(u8) {
    bool_lit,
    int_lit,
    variable,
    old,
    result,
    place_read,
    unary,
    binary,
    refinement_predicate,
    quantified,
};

pub const Term = union(TermTag) {
    bool_lit: bool,
    int_lit: IntegerLiteralTerm,
    variable: VarRef,
    old: TermId,
    result,
    place_read: PlaceRef,
    unary: UnaryTerm,
    binary: BinaryTerm,
    refinement_predicate: RefinementPredicateTerm,
    quantified: QuantifiedTerm,
};

pub const IntegerLiteralTerm = struct {
    value: []const u8,
    ty: ?TypeRef = null,
};

pub const FreeVarId = struct {
    /// Compiler binding identity for a free source variable.
    ///
    /// HIR locals use `ast.PatternId`; after MLIR lowering the formal collector
    /// keeps that id scoped by `source.FileId` so variable names remain
    /// alpha-convertible.
    file_id: u32,
    pattern_id: u32,
};

pub fn freeVarIdEql(lhs: FreeVarId, rhs: FreeVarId) bool {
    return lhs.file_id == rhs.file_id and lhs.pattern_id == rhs.pattern_id;
}

pub const VarRefTag = enum(u8) {
    free,
    bound,
};

pub const VarRef = union(VarRefTag) {
    free: FreeVarRef,
    bound: BoundVarRef,
};

pub const FreeVarRef = struct {
    id: FreeVarId,
    name: []const u8,
    ty: ?TypeRef = null,
    region: ?RegionRef = null,
};

pub const BoundVarRef = struct {
    /// De Bruijn index. `0` refers to the nearest enclosing binder.
    index: u32,
    name: []const u8 = "",
    ty: ?TypeRef = null,
    region: ?RegionRef = null,
};

pub const BinderRef = struct {
    name: []const u8,
    ty: ?TypeRef = null,
    region: ?RegionRef = null,
    origin: BinderOrigin = .user,
};

pub const BinderOrigin = enum(u8) {
    user,
    function_param,
};

pub const UnaryTerm = struct {
    op: UnaryOp,
    operand: TermId,
};

pub const UnaryOp = enum(u8) {
    not,
    neg,
};

pub const BinaryTerm = struct {
    op: BinaryOp,
    lhs: TermId,
    rhs: TermId,
    ty: ?TypeRef = null,
};

pub const BinaryOp = enum(u8) {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    slt,
    sle,
    sgt,
    sge,
    add,
    sub,
    mul,
    pow,
    shl,
    shr,
    div,
    mod,
    bit_and,
    bit_xor,
    and_,
    or_,
    implies,
};

pub const RefinementPredicateTerm = struct {
    name: []const u8,
    value: TermId,
    args: []const TermId = &.{},
};

pub const QuantifiedTerm = struct {
    quantifier: Quantifier,
    binder: BinderRef,
    condition: ?TermId = null,
    body: TermId,
};

pub const Quantifier = enum(u8) {
    forall,
    exists,
};

pub const PlaceRef = struct {
    root: []const u8,
    region: RegionRef,
    fields: []const []const u8 = &.{},
    keys: []const PlaceKey = &.{},
};

pub const PlaceKeyTag = enum(u8) {
    parameter,
    comptime_parameter,
    comptime_range_parameter,
    constant,
    msg_sender,
    tx_origin,
    unknown,
};

pub const PlaceKey = union(PlaceKeyTag) {
    parameter: FreeVarId,
    comptime_parameter: u32,
    comptime_range_parameter: u32,
    constant: []const u8,
    msg_sender,
    tx_origin,
    unknown,
};

pub fn placeRefEql(lhs: PlaceRef, rhs: PlaceRef) bool {
    if (!std.mem.eql(u8, lhs.root, rhs.root)) return false;
    if (lhs.region != rhs.region) return false;
    if (lhs.fields.len != rhs.fields.len) return false;
    if (lhs.keys.len != rhs.keys.len) return false;
    for (lhs.fields, rhs.fields) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    for (lhs.keys, rhs.keys) |left, right| {
        if (!placeKeyEql(left, right)) return false;
    }
    return true;
}

pub fn placeKeyEql(lhs: PlaceKey, rhs: PlaceKey) bool {
    if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) return false;
    return switch (lhs) {
        .parameter => |value| freeVarIdEql(value, rhs.parameter),
        .comptime_parameter => |value| value == rhs.comptime_parameter,
        .comptime_range_parameter => |value| value == rhs.comptime_range_parameter,
        .constant => |value| std.mem.eql(u8, value, rhs.constant),
        .msg_sender, .tx_origin, .unknown => true,
    };
}

pub fn placeDefinitelyDisjoint(lhs: PlaceRef, rhs: PlaceRef) bool {
    if (!regionIsConcrete(lhs.region) or !regionIsConcrete(rhs.region)) return false;
    if (isComputedStorageRoot(lhs.root) or isComputedStorageRoot(rhs.root)) return false;
    if (lhs.region != rhs.region) return true;
    if (!std.mem.eql(u8, lhs.root, rhs.root)) return true;
    if (!stringSlicesEql(lhs.fields, rhs.fields)) return false;
    return placeKeysDefinitelyDisjoint(lhs.keys, rhs.keys);
}

pub fn placeKeysDefinitelyDistinct(lhs: PlaceKey, rhs: PlaceKey) bool {
    if (lhs == .constant and rhs == .constant) {
        const lhs_value = parseDecimalU256(lhs.constant) orelse return false;
        const rhs_value = parseDecimalU256(rhs.constant) orelse return false;
        return lhs_value != rhs_value;
    }
    return false;
}

fn placeKeysDefinitelyDisjoint(lhs: []const PlaceKey, rhs: []const PlaceKey) bool {
    const limit = @min(lhs.len, rhs.len);
    var index: usize = 0;
    while (index < limit) : (index += 1) {
        if (placeKeyEql(lhs[index], rhs[index])) continue;
        return placeKeysDefinitelyDistinct(lhs[index], rhs[index]);
    }
    return false;
}

fn regionIsConcrete(region: RegionRef) bool {
    return region != .none;
}

fn isComputedStorageRoot(root: []const u8) bool {
    return std.mem.eql(u8, root, "$computed_storage");
}

fn stringSlicesEql(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    return true;
}

fn parseDecimalU256(value: []const u8) ?u256 {
    if (value.len == 0) return null;
    for (value) |byte| {
        if (byte < '0' or byte > '9') return null;
    }
    return std.fmt.parseInt(u256, value, 10) catch null;
}

pub const StorageDisjointnessFixture = struct {
    label: []const u8,
    lhs: PlaceRef,
    rhs: PlaceRef,
    expected: bool,
};

const fixture_key_const_1 = [_]PlaceKey{.{ .constant = "1" }};
const fixture_key_const_01 = [_]PlaceKey{.{ .constant = "01" }};
const fixture_key_const_2 = [_]PlaceKey{.{ .constant = "2" }};
const fixture_key_const_1001 = [_]PlaceKey{.{ .constant = "1001" }};
const fixture_key_const_1_000 = [_]PlaceKey{.{ .constant = "1_000" }};
const fixture_key_const_bad = [_]PlaceKey{.{ .constant = "0..2" }};
const fixture_param_0: FreeVarId = .{ .file_id = 0, .pattern_id = 0 };
const fixture_param_1: FreeVarId = .{ .file_id = 0, .pattern_id = 1 };
const fixture_key_param_0 = [_]PlaceKey{.{ .parameter = fixture_param_0 }};
const fixture_key_param_1 = [_]PlaceKey{.{ .parameter = fixture_param_1 }};
const fixture_key_msg_sender = [_]PlaceKey{.{ .msg_sender = {} }};
const fixture_key_tx_origin = [_]PlaceKey{.{ .tx_origin = {} }};
const fixture_key_unknown = [_]PlaceKey{.{ .unknown = {} }};
const fixture_keys_prefix = [_]PlaceKey{ .{ .parameter = fixture_param_0 }, .{ .constant = "1" } };
const fixture_field_owner = [_][]const u8{"owner"};
const fixture_field_admin = [_][]const u8{"admin"};

pub const storage_disjointness_fixtures = [_]StorageDisjointnessFixture{
    .{
        .label = "different_roots",
        .lhs = .{ .root = "balances", .region = .storage },
        .rhs = .{ .root = "allowances", .region = .storage },
        .expected = true,
    },
    .{
        .label = "same_root_exact_path",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1 },
        .expected = false,
    },
    .{
        .label = "whole_root_write_vs_keyed_read",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1 },
        .rhs = .{ .root = "balances", .region = .storage },
        .expected = false,
    },
    .{
        .label = "normalized_unequal_constants",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_2 },
        .expected = true,
    },
    .{
        .label = "raw_noncanonical_equal_constants",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_01 },
        .expected = false,
    },
    .{
        .label = "unparseable_constant_blocks",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_bad },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_2 },
        .expected = false,
    },
    .{
        .label = "underscore_constant_blocks",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1_000 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_1001 },
        .expected = false,
    },
    .{
        .label = "parameter_vs_parameter",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_param_0 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_param_1 },
        .expected = false,
    },
    .{
        .label = "parameter_vs_constant",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_param_0 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_2 },
        .expected = false,
    },
    .{
        .label = "msg_sender_vs_tx_origin",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_msg_sender },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_tx_origin },
        .expected = false,
    },
    .{
        .label = "unknown_key_blocks",
        .lhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_unknown },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_2 },
        .expected = false,
    },
    .{
        .label = "same_prefix_blocks",
        .lhs = .{ .root = "allowances", .region = .storage, .keys = &fixture_key_param_0 },
        .rhs = .{ .root = "allowances", .region = .storage, .keys = &fixture_keys_prefix },
        .expected = false,
    },
    .{
        .label = "different_concrete_regions",
        .lhs = .{ .root = "scratch", .region = .transient },
        .rhs = .{ .root = "scratch", .region = .storage },
        .expected = true,
    },
    .{
        .label = "none_region_blocks",
        .lhs = .{ .root = "scratch", .region = .none },
        .rhs = .{ .root = "scratch", .region = .storage },
        .expected = false,
    },
    .{
        .label = "computed_storage_blocks",
        .lhs = .{ .root = "$computed_storage", .region = .storage, .keys = &fixture_key_const_1 },
        .rhs = .{ .root = "balances", .region = .storage, .keys = &fixture_key_const_2 },
        .expected = false,
    },
    .{
        .label = "different_fields_deferred",
        .lhs = .{ .root = "config", .region = .storage, .fields = &fixture_field_owner },
        .rhs = .{ .root = "config", .region = .storage, .fields = &fixture_field_admin },
        .expected = false,
    },
};

pub const RegionRef = enum(u8) {
    none,
    storage,
    memory,
    transient,
    calldata,
};

pub const TypeRefTag = enum(u8) {
    spelling,
    compiler_type_id,
};

pub const TypeRef = union(TypeRefTag) {
    spelling: []const u8,
    compiler_type_id: u32,
};

test "manifest enum tags stay byte-sized" {
    inline for (.{
        DiagnosticKind,
        Phase,
        ArtifactPolicy,
        ProofRequirement,
        ArtifactBlockReason,
        OwnerTag,
        BackendComponent,
        LoopKind,
        LoopStorageProjectionTag,
        LoopTimelineEventTag,
        LoopUnsupportedReason,
        OriginTag,
        EffectAccess,
        KindTag,
        VerificationBackend,
        VerificationQueryKind,
        VerificationQueryFragment,
        VerificationSolverLogic,
        VerificationQueryStatus,
        ProofArtifactKind,
        LogicalRole,
        ArithmeticSafetyKind,
        GuardErasurePolicy,
        ValueRefKind,
        TypeRelation,
        RegionRelation,
        EffectFrameRelation,
        ResourceOperation,
        ResourceProperty,
        QuantifierBinderSort,
        QuantifierPatternStatus,
        QuantifierDegradation,
        BackendProperty,
        AssumptionKind,
        FormulaRefTag,
        TermTag,
        VarRefTag,
        UnaryOp,
        BinaryOp,
        Quantifier,
        PlaceKeyTag,
        RegionRef,
        TypeRefTag,
    }) |T| {
        try std.testing.expectEqual(@as(usize, 1), @sizeOf(T));
    }
}

test "storage place disjointness fixtures match Zig relation" {
    for (storage_disjointness_fixtures) |fixture| {
        try std.testing.expectEqual(
            fixture.expected,
            placeDefinitelyDisjoint(fixture.lhs, fixture.rhs),
        );
        try std.testing.expectEqual(
            fixture.expected,
            placeDefinitelyDisjoint(fixture.rhs, fixture.lhs),
        );
    }
}

test "verification query summary counts solver-facing categories" {
    const queries = [_]VerificationQuery{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "func.func", .symbol = "transfer" } },
            .kind = .base,
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
            .kind = .obligation,
            .logical_role = .ensures,
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "ora.refinement_guard", .symbol = "transfer" } },
            .kind = .guard_violate,
            .guard_id = "guard:transfer:amount",
        },
    };

    const summary = VerificationQuerySummary.fromQueries(&queries);
    try std.testing.expectEqual(@as(u32, 1), summary.base);
    try std.testing.expectEqual(@as(u32, 1), summary.obligation);
    try std.testing.expectEqual(@as(u32, 0), summary.guard_satisfy);
    try std.testing.expectEqual(@as(u32, 1), summary.guard_violate);
}

test "blocking diagnostics gate obligation set" {
    const diagnostics = [_]ObligationDiagnostic{
        .{
            .kind = .unsupported,
            .source = .generated(),
            .message = "unsupported quantified binder",
        },
    };
    const set: ObligationSet = .{ .diagnostics = &diagnostics };
    try std.testing.expect(set.hasBlockingDiagnostic());
}

test "term references validate canonical formulas" {
    const args = [_]TermId{1};
    const terms = [_]Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .refinement_predicate = .{ .name = "MinValue", .value = 0, .args = &args } },
    };
    const obligation: Obligation = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "deposit" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "refinement_guard", .ordinal = 0 } },
        .kind = .{ .logical = .{
            .role = .refinement,
            .formula = .{ .term = 2 },
        } },
    };
    const set: ObligationSet = .{ .obligations = &.{obligation}, .terms = &terms };
    try set.validateTermReferences();
}

test "term reference validation rejects dangling ids" {
    const terms = [_]Term{
        .{ .unary = .{ .op = .not, .operand = 1 } },
    };
    const set: ObligationSet = .{ .terms = &terms };
    try std.testing.expectError(error.InvalidTermReference, set.validateTermReferences());
}

test "diagnostic-only entries do not block artifacts" {
    const diagnostics = [_]ObligationDiagnostic{
        .{
            .kind = .unsupported,
            .source = .generated(),
            .message = "informational",
            .blocks_artifacts = false,
        },
    };
    const set: ObligationSet = .{ .diagnostics = &diagnostics };
    try std.testing.expect(!set.hasBlockingDiagnostic());
}

test "artifact policy allows blocking obligation only after successful proof query" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unsat },
        },
    };
    const set: ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &.{.{ .bool_lit = true }},
    };

    try std.testing.expect(set.artifactDecision().isAllowed());
}

test "artifact policy blocks unknown and degraded query results" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const unknown_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unknown },
        },
    };
    const degraded_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unsat, .degraded = true },
        },
    };

    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .queries = &unknown_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.degraded_query, (ObligationSet{
        .obligations = &obligations,
        .queries = &degraded_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "artifact policy blocks vacuous query results even when proved" {
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .result = .{ .status = .proved, .vacuous = true },
        },
    };

    try expectArtifactBlocked(.vacuous_query, (ObligationSet{
        .queries = &queries,
    }).artifactDecision());
}

test "artifact policy blocks invalid dependencies and missing proofs" {
    const invalid_dependencies = [_]Id{42};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const invalid_obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .dependencies = &invalid_dependencies,
        },
    };

    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &invalid_obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .obligations = &obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "artifact policy blocks a canonical loop Z3 row with no result" {
    const digest = [_]u8{0} ** 32;
    const summaries = [_]LoopSummaryRow{.{
        .id = 7,
        .owner = .{ .statement = .{ .function_name = "count", .ordinal = 0 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        // Deliberately unsupported so this unit isolates the artifact policy
        // from the complete canonical-query inventory validator.
        .unsupported_reasons = &.{.loop_state_empty},
    }};
    const queries = [_]VerificationQuery{.{
        .id = 8,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .z3,
        .kind = .loop_base,
        .loop_summary_id = 7,
        .loop_vc_sha256 = digest,
    }};

    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .loop_summaries = &summaries,
        .queries = &queries,
    }).artifactDecision());
}

test "artifact policy treats Lean-required proof gaps as hard failures" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .obligation_ids = &obligation_ids,
        },
    };
    const failed_lean_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .failed },
        },
    };
    const proved_lean_queries_without_artifact = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .proved },
        },
    };
    const proved_lean_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
    };

    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .obligations = &obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .obligations = &obligations,
        .queries = &failed_lean_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .obligations = &obligations,
        .queries = &proved_lean_queries_without_artifact,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .queries = &proved_lean_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision().isAllowed());
}

test "loop induction certificate is independent and cannot override a failed Z3 query" {
    const obligation_ids = [_]Id{1};
    const induction_query_ids = [_]Id{2};
    const post_query_ids = [_]Id{3};
    const obligations = [_]Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 0 } } },
    }};
    const summaries = [_]LoopSummaryRow{.{
        .id = 7,
        .owner = .{ .statement = .{ .function_name = "count", .ordinal = 0 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .query_ids = .{ .induction = &induction_query_ids, .post = &post_query_ids },
    }};
    const induction_target: VerificationQuery = .{
        .id = 2,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .proof_requirement = .lean_certificate,
        .kind = .loop_induction,
        .loop_summary_id = 7,
    };
    const z3_query: VerificationQuery = .{
        .id = 3,
        .owner = induction_target.owner,
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .z3,
        .kind = .loop_invariant_post,
        .obligation_ids = &obligation_ids,
        .loop_summary_id = 7,
        .result = .{ .status = .unsat },
    };

    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .queries = &.{ induction_target, z3_query },
    }).artifactDecision());

    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = z3_query.owner,
            .source = .generated(),
            .module_name = "Ora.Proofs.Count",
            .theorem_name = "count_induction",
        },
        .{
            .id = 11,
            .owner = z3_query.owner,
            .source = .generated(),
            .module_name = "Ora.Proofs.Count",
            .theorem_name = "forged_post_override",
            .obligation_ids = &obligation_ids,
        },
    };
    const lean_query: VerificationQuery = .{
        .id = 4,
        .owner = z3_query.owner,
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .lean,
        .kind = .loop_induction,
        .proof_artifact_id = 10,
        .discharges_query_id = 2,
        .loop_summary_id = 7,
        .result = .{ .status = .proved },
    };
    try std.testing.expect((ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .proof_artifacts = &artifacts,
        .queries = &.{ induction_target, z3_query, lean_query },
    }).artifactDecision().isAllowed());

    var failed_z3_query = z3_query;
    failed_z3_query.result = .{ .status = .sat };
    try expectArtifactBlocked(.failed_query, (ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .proof_artifacts = &artifacts,
        .queries = &.{ induction_target, failed_z3_query, lean_query },
    }).artifactDecision());

    const forged_lean_override: VerificationQuery = .{
        .id = 5,
        .owner = z3_query.owner,
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .lean,
        .kind = .loop_invariant_post,
        .obligation_ids = &obligation_ids,
        .proof_artifact_id = 11,
        .discharges_query_id = 3,
        .loop_summary_id = 7,
        .result = .{ .status = .proved },
    };
    var unknown_z3_query = z3_query;
    unknown_z3_query.result = .{ .status = .unknown };
    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .proof_artifacts = &artifacts,
        .queries = &.{ induction_target, unknown_z3_query, lean_query, forged_lean_override },
    }).artifactDecision());

    var missing_z3_query = z3_query;
    missing_z3_query.result = null;
    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .proof_artifacts = &artifacts,
        .queries = &.{ induction_target, missing_z3_query, lean_query, forged_lean_override },
    }).artifactDecision());

    var degraded_z3_query = unknown_z3_query;
    degraded_z3_query.result.?.degraded = true;
    try expectArtifactBlocked(.degraded_query, (ObligationSet{
        .terms = &.{.{ .bool_lit = true }},
        .obligations = &obligations,
        .loop_summaries = &summaries,
        .proof_artifacts = &artifacts,
        .queries = &.{ induction_target, degraded_z3_query, lean_query, forged_lean_override },
    }).artifactDecision());
}

test "userland Lean proof artifact attaches to a required obligation" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{ .file = "proofs/ERC20/Transfer.lean", .line = 1 },
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .path = "proofs/ERC20/Transfer.lean",
            .content_hash = 0x1234,
            .obligation_ids = &obligation_ids,
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
    };
    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .queries = &queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision().isAllowed());
}

test "userland Lean proof attachment fails closed when missing or mismatched" {
    const obligation_ids = [_]Id{1};
    const other_obligation_ids = [_]Id{2};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "mint" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 1 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "mint" } },
            .source = .generated(),
            .module_name = "ERC20.Mint",
            .theorem_name = "mint_supply",
            .obligation_ids = &other_obligation_ids,
        },
    };
    const missing_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 99,
            .result = .{ .status = .proved },
        },
    };
    const mismatched_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
    };
    const z3_with_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .unsat },
        },
    };

    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &missing_artifact_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &mismatched_artifact_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &z3_with_artifact_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "artifact policy accepts Lean proof for missing structural query only with valid artifact" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .{ .effect_frame = .{
                .relation = .write_covered_by_modifies,
            } },
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .module_name = "Ora.Obligation.Theorems",
            .theorem_name = "effect_frame_write_covered_by_modifies_shape_follows_from_row",
            .obligation_ids = &obligation_ids,
        },
    };
    const structural_query = VerificationQuery{
        .id = 2,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    };
    const proved_query = VerificationQuery{
        .id = 3,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .lean,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .proof_artifact_id = 10,
        .discharges_query_id = 2,
        .result = .{ .status = .proved },
    };
    const missing_artifact_query = VerificationQuery{
        .id = 3,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .backend = .lean,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .discharges_query_id = 2,
        .result = .{ .status = .proved },
    };

    const blocked_queries = [_]VerificationQuery{structural_query};
    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .obligations = &obligations,
        .queries = &blocked_queries,
    }).artifactDecision());

    const invalid_queries = [_]VerificationQuery{ structural_query, missing_artifact_query };
    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .obligations = &obligations,
        .queries = &invalid_queries,
        .proof_artifacts = &artifacts,
    }).artifactDecision());

    const proved_queries = [_]VerificationQuery{ structural_query, proved_query };
    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .queries = &proved_queries,
        .proof_artifacts = &artifacts,
    }).artifactDecision().isAllowed());
}

test "userland Lean proof artifact discharges only plain Z3 unknown" {
    const obligation_ids = [_]Id{1};
    const assumption_ids = [_]Id{1};
    const other_assumption_ids = [_]Id{2};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const assumptions = [_]Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 0 },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 1 } },
            .kind = .requires,
            .formula = .{ .term = 0 },
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .obligation_ids = &obligation_ids,
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown },
        },
    };
    const mismatched_assumption_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &other_assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown },
        },
    };
    const missing_artifact_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown },
        },
    };
    const wrong_target_query = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 99,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown },
        },
    };
    const sat_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .sat },
        },
    };
    const vacuity_unknown_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown, .vacuity_unknown = true },
        },
    };
    const degraded_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .proof_artifact_id = 10,
            .discharges_query_id = 3,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
            .result = .{ .status = .unknown, .degraded = true },
        },
    };

    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision().isAllowed());
    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &mismatched_assumption_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &missing_artifact_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &wrong_target_query,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.failed_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &sat_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &vacuity_unknown_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.degraded_query, (ObligationSet{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &degraded_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "obligation can point at MLIR origin without proof term" {
    const obligation: Obligation = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .{ .file = "erc20.ora", .line = 10, .column = 5 },
        .phase = .ora_mlir,
        .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .origin_value = .{
                .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
            } },
        } },
    };

    try std.testing.expect(obligation.blocksArtifacts());
}

test "storage artifact authority admits only reviewed context-key mappings without storage child frames" {
    const ty: TypeRef = .{ .spelling = "u256" };
    const direct_timeline = [_]LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "store/counter",
        .payload = .{ .storage_write = .{
            .location = .{ .root = "counter", .value_ty = ty },
            .value = .{ .formula = .{ .term = 0 } },
        } },
    }};
    const direct_edges = [_]LoopEdgeRow{.{
        .id = 0,
        .structural_key = "edge/direct",
        .timeline = &direct_timeline,
    }};
    const direct_summary: LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "direct" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .edges = &direct_edges,
    };
    const certificate_query: VerificationQuery = .{
        .id = 2,
        .owner = direct_summary.owner,
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .proof_requirement = .lean_certificate,
        .kind = .loop_induction,
        .loop_summary_id = direct_summary.id,
    };
    const direct_set: ObligationSet = .{
        .queries = &.{certificate_query},
        .loop_summaries = &.{direct_summary},
    };
    try std.testing.expectEqual(
        @as(?ArtifactBlockReason, null),
        direct_set.storageArtifactAuthorityBlockReason(),
    );

    const index_projection = [_]LoopStorageProjection{.{ .fixed_index = .{
        .index = .{
            .value = .{ .formula = .{ .term = 0 } },
            .ty = ty,
        },
        .extent = 4,
    } }};
    var projected_timeline = direct_timeline;
    projected_timeline[0].payload.storage_write.location.projections = &index_projection;
    var projected_edges = direct_edges;
    projected_edges[0].timeline = &projected_timeline;
    var projected_summary = direct_summary;
    projected_summary.edges = &projected_edges;
    try std.testing.expectEqual(
        ArtifactBlockReason.storage_projection_authority_unreviewed,
        (ObligationSet{
            .queries = &.{certificate_query},
            .loop_summaries = &.{projected_summary},
        }).storageArtifactAuthorityBlockReason().?,
    );

    const context_id: FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const context_terms = [_]Term{.{ .variable = .{ .free = .{
        .id = context_id,
        .name = "key",
        .ty = ty,
    } } }};
    const context_variables = [_]LoopVariable{.{
        .index = 0,
        .id = context_id,
        .name = "key",
        .ty = ty,
    }};
    const mapping_projection = [_]LoopStorageProjection{.{ .mapping_key = .{
        .value = .{ .formula = .{ .term = 0 } },
        .ty = ty,
    } }};
    var mapping_timeline = direct_timeline;
    mapping_timeline[0].payload.storage_write.location.projections = &mapping_projection;
    var mapping_edges = direct_edges;
    mapping_edges[0].timeline = &mapping_timeline;
    var mapping_summary = direct_summary;
    mapping_summary.context_variables = &context_variables;
    mapping_summary.preheader_projection_complete = true;
    mapping_summary.preheader_timeline = &mapping_timeline;
    mapping_summary.edges = &mapping_edges;
    try std.testing.expectEqual(
        @as(?ArtifactBlockReason, null),
        (ObligationSet{
            .queries = &.{certificate_query},
            .terms = &context_terms,
            .loop_summaries = &.{mapping_summary},
        }).storageArtifactAuthorityBlockReason(),
    );

    var fixed_preheader_summary = direct_summary;
    fixed_preheader_summary.preheader_projection_complete = true;
    fixed_preheader_summary.preheader_timeline = &projected_timeline;
    try std.testing.expectEqual(
        ArtifactBlockReason.storage_projection_authority_unreviewed,
        (ObligationSet{
            .queries = &.{certificate_query},
            .loop_summaries = &.{fixed_preheader_summary},
        }).storageArtifactAuthorityBlockReason().?,
    );

    const dependency_timeline = [_]LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "child/direct",
        .payload = .{ .summary_dependency = .{
            .id = 0,
            .structural_key = "child/direct",
            .summary_id = 2,
            .loop_vc_sha256 = [_]u8{0} ** 32,
            .write_footprint_complete = true,
            .target = .successor,
        } },
    }};
    const dependency_edges = [_]LoopEdgeRow{.{
        .id = 0,
        .structural_key = "edge/dependency",
        .timeline = &dependency_timeline,
    }};
    var parent_summary = direct_summary;
    parent_summary.edges = &dependency_edges;
    var child_summary = direct_summary;
    child_summary.id = 2;
    try std.testing.expectEqual(
        ArtifactBlockReason.storage_child_frame_authority_unreviewed,
        (ObligationSet{
            .queries = &.{certificate_query},
            .loop_summaries = &.{ parent_summary, child_summary },
        }).storageArtifactAuthorityBlockReason().?,
    );
}

test "diagnostic-only query carries accounting identity without artifact authority" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "loop" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .logical = .{
            .role = .loop_invariant,
            .formula = .{ .term = 0 },
        } },
        .artifact_policy = .diagnostic_only,
    }};
    const queries = [_]VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "loop" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .artifact_policy = .diagnostic_only,
        .backend = .z3,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .result = .{ .status = .unknown },
    }};
    const set: ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &.{.{ .bool_lit = true }},
    };

    try std.testing.expect(set.artifactDecision().isAllowed());
}

test "derived obligations keep provenance instead of rewriting originals" {
    const derived = [_]Id{ 1, 2 };
    const obligation: Obligation = .{
        .id = 3,
        .owner = .{ .backend = .{ .component = .dispatcher, .name = "erc20" } },
        .source = .generated(),
        .phase = .sinora,
        .origin = .{ .backend_fact = .{ .component = .dispatcher, .fact = "selector_table_complete" } },
        .kind = .{ .backend_fact = .{
            .component = .dispatcher,
            .property = .complete,
        } },
        .derived_from = &derived,
    };

    try std.testing.expectEqualSlices(Id, &derived, obligation.derived_from);
}
