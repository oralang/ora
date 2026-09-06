//! Ora MLIR to obligation manifest adapter.
//!
//! This is a boundary collector, not a verifier and not a proof encoder. It
//! walks canonical Ora MLIR, records the obligation surface already present
//! there, and leaves formulas owned by MLIR values. Z3 and Lean exporters must
//! consume the same manifest instead of rediscovering different obligations.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const obligation = @import("obligation.zig");
const obligation_to_lean = @import("obligation_to_lean.zig");
const type_builtin = @import("ora_types").builtin;

pub const CollectOptions = struct {
    /// Borrowed owner used when the walker has not entered a symbol-bearing op.
    owner: obligation.Owner = .{ .module = "ora_mlir" },
};

pub const CollectResult = struct {
    arena: std.heap.ArenaAllocator,
    set: obligation.ObligationSet,
    query_bindings: []const obligation.FormalQueryBinding = &.{},
    source_fact_bindings: []const SourceFactOpBinding = &.{},
    runtime_owner_bindings: []const RuntimeOwnerBinding = &.{},
    runtime_check_producers: []const RuntimeCheckProducer = &.{},
    state_effect_producers: []const StateEffectProducer = &.{},

    pub fn deinit(self: *CollectResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const RuntimeOwnerBinding = struct {
    source_op_id: usize,
    op_ordinal: u32,
    source: obligation.SourceRef,
    symbol: ?[]const u8 = null,
    module_path: []const u8,
    owner_key: []const u8,
    template_activation: []const u8,
    specialization_bindings: []const []const u8 = &.{},
    trait_implementation: ?[]const u8 = null,
    trait_method: ?[]const u8 = null,
};

/// One concrete runtime-enforcement operation present in canonical HIR. The
/// producer id is deterministic within the collected module and enters the
/// dedicated runtime-check evidence namespace; it is not an obligation id.
pub const RuntimeCheckProducer = struct {
    id: u32,
    source_op_id: usize,
    op_ordinal: u32,
    source: obligation.SourceRef,
};

/// One concrete state mutation directive present in canonical HIR. Havoc is
/// not a proof obligation, but its presence is authoritative evidence that
/// the symbolic state was invalidated at the source-directed point.
pub const StateEffectProducer = struct {
    id: u32,
    source_op_id: usize,
    op_ordinal: u32,
    source: obligation.SourceRef,
};

/// Structural join between one source-formal MLIR operation and the manifest
/// rows produced while visiting it. `source_op_id` is process-local and used
/// only to join the live Z3 query builder; every reportable identity is carried
/// separately in deterministic fields.
pub const SourceFactOpBinding = struct {
    source_op_id: usize,
    op_ordinal: u32,
    source: obligation.SourceRef,
    source_fact_id: u32,
    kind: []const u8,
    roles: []const []const u8,
    module_path: ?[]const u8 = null,
    owner_key: ?[]const u8 = null,
    template_activation: ?[]const u8 = null,
    runtime_symbol: ?[]const u8 = null,
    specialization_bindings: []const []const u8 = &.{},
    trait_implementation: ?[]const u8 = null,
    trait_method: ?[]const u8 = null,
    obligation_ids: []const obligation.Id = &.{},
    assumption_ids: []const obligation.Id = &.{},
    query_ids: []const obligation.Id = &.{},
    runtime_check_ids: []const u32 = &.{},
    runtime_check_present: bool = false,
    state_effect_ids: []const u32 = &.{},
    state_effect_present: bool = false,
};

const OpKind = enum(u8) {
    requires,
    ensures,
    invariant,
    assert,
    cf_assert,
    assume,
    refinement_guard,
    resource_move,
    resource_create,
    resource_destroy,
    quantifier,
};

const op_kind_map = std.StaticStringMap(OpKind).initComptime(.{
    .{ "ora.requires", .requires },
    .{ "ora.ensures", .ensures },
    .{ "ora.invariant", .invariant },
    .{ "ora.assert", .assert },
    .{ "cf.assert", .cf_assert },
    .{ "ora.assume", .assume },
    .{ "ora.refinement_guard", .refinement_guard },
    .{ "ora.move", .resource_move },
    .{ "ora.create", .resource_create },
    .{ "ora.destroy", .resource_destroy },
    .{ "ora.quantified", .quantifier },
});

const arithmetic_op_map = std.StaticStringMap(obligation.ArithmeticSafetyKind).initComptime(.{
    .{ "arith.divui", .division_by_zero },
    .{ "arith.divsi", .division_by_zero },
    .{ "arith.remui", .division_by_zero },
    .{ "arith.remsi", .division_by_zero },
});

const assert_safety_map = std.StaticStringMap(obligation.ArithmeticSafetyKind).initComptime(.{
    .{ "checked addition overflow", .addition_overflow },
    .{ "checked subtraction overflow", .subtraction_overflow },
    .{ "checked multiplication overflow", .multiplication_overflow },
    .{ "checked power overflow", .power_overflow },
    .{ "checked negation overflow", .negation_overflow },
    .{ "checked division by zero", .division_by_zero },
    .{ "checked remainder by zero", .division_by_zero },
    .{ "checked division overflow", .signed_division_overflow },
    .{ "checked shift amount out of range", .shift_amount_bounds },
    // A for-range may imply this condition, but only the emitted assertion
    // owns the checkpoint; loop bounds are premises, never a synthesized check.
    .{ "fixed-array index out of bounds", .fixed_array_bounds },
});

const transparent_value_ops = std.StaticStringMap(void).initComptime(.{
    .{ "ora.refinement_to_base", {} },
    .{ "ora.base_to_refinement", {} },
    .{ "arith.bitcast", {} },
    .{ "arith.extui", {} },
    .{ "arith.extsi", {} },
    .{ "arith.index_cast", {} },
    .{ "arith.index_castui", {} },
    .{ "arith.index_castsi", {} },
    .{ "builtin.unrealized_conversion_cast", {} },
    .{ "tensor.cast", {} },
});

const ArithmeticSignednessSource = enum {
    compiler_attribute,
    unsigned,
    signed,
};

const ArithmeticSafetySource = enum {
    required,
    wrapping,
    compiler_attribute,
};

const ArithmeticValueOp = struct {
    op: obligation.BinaryOp,
    signedness: ArithmeticSignednessSource,
    safety: ArithmeticSafetySource,
};

const arithmetic_value_op_map = std.StaticStringMap(ArithmeticValueOp).initComptime(.{
    .{ "arith.addi", ArithmeticValueOp{ .op = .add, .signedness = .compiler_attribute, .safety = .required } },
    .{ "arith.subi", ArithmeticValueOp{ .op = .sub, .signedness = .compiler_attribute, .safety = .required } },
    .{ "arith.muli", ArithmeticValueOp{ .op = .mul, .signedness = .compiler_attribute, .safety = .required } },
    .{ "ora.add_wrapping", ArithmeticValueOp{ .op = .add, .signedness = .compiler_attribute, .safety = .wrapping } },
    .{ "ora.sub_wrapping", ArithmeticValueOp{ .op = .sub, .signedness = .compiler_attribute, .safety = .wrapping } },
    .{ "ora.mul_wrapping", ArithmeticValueOp{ .op = .mul, .signedness = .compiler_attribute, .safety = .wrapping } },
    .{ "ora.power", ArithmeticValueOp{ .op = .pow, .signedness = .compiler_attribute, .safety = .compiler_attribute } },
    .{ "arith.shli", ArithmeticValueOp{ .op = .shl, .signedness = .compiler_attribute, .safety = .required } },
    .{ "ora.shl_wrapping", ArithmeticValueOp{ .op = .shl, .signedness = .compiler_attribute, .safety = .wrapping } },
    .{ "arith.shrui", ArithmeticValueOp{ .op = .shr, .signedness = .unsigned, .safety = .required } },
    .{ "arith.shrsi", ArithmeticValueOp{ .op = .shr, .signedness = .signed, .safety = .required } },
    .{ "ora.shr_wrapping", ArithmeticValueOp{ .op = .shr, .signedness = .compiler_attribute, .safety = .wrapping } },
});

const ArithmeticDivRemValueOp = struct {
    op: obligation.BinaryOp,
    signed: bool,
};

const arithmetic_div_rem_value_op_map = std.StaticStringMap(ArithmeticDivRemValueOp).initComptime(.{
    .{ "arith.divui", ArithmeticDivRemValueOp{ .op = .div, .signed = false } },
    .{ "arith.remui", ArithmeticDivRemValueOp{ .op = .mod, .signed = false } },
    .{ "arith.divsi", ArithmeticDivRemValueOp{ .op = .div, .signed = true } },
    .{ "arith.remsi", ArithmeticDivRemValueOp{ .op = .mod, .signed = true } },
});

const ResourceRootOp = enum(u8) {
    storage_load,
    transient_load,
    struct_field_extract,
    integer_constant,
};

const resource_root_op_map = std.StaticStringMap(ResourceRootOp).initComptime(.{
    .{ "ora.sload", .storage_load },
    .{ "ora.tload", .transient_load },
    .{ "ora.struct_field_extract", .struct_field_extract },
    .{ "arith.constant", .integer_constant },
});

const quantifier_map = std.StaticStringMap(obligation.Quantifier).initComptime(.{
    .{ "forall", .forall },
    .{ "exists", .exists },
});

const resource_move_properties = [_]obligation.ResourceProperty{
    .amount_non_negative,
    .source_sufficient,
    .destination_no_overflow,
    .same_place_identity,
    .conservation,
};

const resource_create_properties = [_]obligation.ResourceProperty{
    .amount_non_negative,
    .destination_no_overflow,
};

const resource_destroy_properties = [_]obligation.ResourceProperty{
    .amount_non_negative,
    .source_sufficient,
};

pub fn collect(
    allocator: std.mem.Allocator,
    module: mlir.MlirModule,
    options: CollectOptions,
) !CollectResult {
    if (mlir.oraModuleIsNull(module)) return error.InvalidModule;

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();

    var collector = Collector{
        .allocator = arena.allocator(),
        .options = options,
    };

    try collector.walkOperation(mlir.oraModuleGetOperation(module), null);
    try collector.addBaseQueriesForFunctionOwners();
    try collector.verifyResourceGoalCompleteness();
    try bindLoopSummaryDependencyDigests(&collector);
    try armLeanLoopCertificateRequirements(&collector);

    const set: obligation.ObligationSet = .{
        .terms = try collector.terms.toOwnedSlice(collector.allocator),
        .obligations = try collector.obligations.toOwnedSlice(collector.allocator),
        .assumptions = try collector.assumptions.toOwnedSlice(collector.allocator),
        .queries = try collector.queries.toOwnedSlice(collector.allocator),
        .loop_summaries = try collector.loop_summaries.toOwnedSlice(collector.allocator),
        .diagnostics = try collector.diagnostics.toOwnedSlice(collector.allocator),
    };

    return .{
        .arena = arena,
        .set = set,
        .query_bindings = try collector.query_bindings.toOwnedSlice(collector.allocator),
        .source_fact_bindings = try collector.source_fact_bindings.toOwnedSlice(collector.allocator),
        .runtime_owner_bindings = try collector.runtime_owner_bindings.toOwnedSlice(collector.allocator),
        .runtime_check_producers = try collector.runtime_check_producers.toOwnedSlice(collector.allocator),
        .state_effect_producers = try collector.state_effect_producers.toOwnedSlice(collector.allocator),
    };
}

fn armLeanLoopCertificateRequirements(collector: *Collector) !void {
    const view: obligation.ObligationSet = .{
        .terms = collector.terms.items,
        .obligations = collector.obligations.items,
        .assumptions = collector.assumptions.items,
        .queries = collector.queries.items,
        .loop_summaries = collector.loop_summaries.items,
        .diagnostics = collector.diagnostics.items,
    };
    for (collector.queries.items) |*query| {
        if (query.kind != .loop_induction) continue;
        switch (obligation_to_lean.querySemanticSupport(view, query.*)) {
            .supported => {
                const summary_id = query.loop_summary_id orelse return error.InvalidLoopSummary;
                const digest = try obligation_to_lean.loopVcSha256(view, summary_id);
                for (collector.queries.items) |*loop_query| {
                    if (loop_query.loop_summary_id != summary_id) continue;
                    loop_query.loop_vc_sha256 = digest;
                    loop_query.artifact_policy = .blocks_verified_artifacts;
                    if (loop_query.kind == .loop_induction) {
                        loop_query.proof_requirement = .lean_certificate;
                    }
                }
                try collector.armCanonicalLoopPostBindings(summary_id);
            },
            .unsupported => |reason| {
                query.artifact_policy = .diagnostic_only;
                query.proof_requirement = .backend_result;
                const summary_id = query.loop_summary_id orelse continue;
                const summary = loopSummaryByIdMut(collector.loop_summaries.items, summary_id) orelse continue;
                if (summary.unsupported_reasons.len == 0) {
                    try collector.appendLoopSummaryReason(summary, loopUnsupportedReasonForSemantic(reason));
                }
            },
        }
    }
}

fn bindLoopSummaryDependencyDigests(collector: *Collector) !void {
    // Parent summaries are collected before their nested children. Walking in
    // reverse therefore binds the deepest summaries first and makes every
    // parent digest include already-final child digests.
    var summary_index = collector.loop_summaries.items.len;
    while (summary_index > 0) {
        summary_index -= 1;
        const inner = collector.loop_summaries.items[summary_index];
        if (!inner.projectionSupported()) {
            try markLoopSummaryDependentsUnsupported(collector, summary_index);
            continue;
        }
        const view: obligation.ObligationSet = .{
            .terms = collector.terms.items,
            .obligations = collector.obligations.items,
            .assumptions = collector.assumptions.items,
            .queries = collector.queries.items,
            .loop_summaries = collector.loop_summaries.items,
            .diagnostics = collector.diagnostics.items,
        };
        if (inner.query_ids.induction.len != 1) return error.InvalidLoopSummary;
        const induction_query = obligation.findById(
            collector.queries.items,
            inner.query_ids.induction[0],
        ) orelse return error.InvalidLoopSummary;
        if (induction_query.kind != .loop_induction or induction_query.loop_summary_id != inner.id) {
            return error.InvalidLoopSummary;
        }
        switch (obligation_to_lean.querySemanticSupport(view, induction_query)) {
            .supported => {},
            .unsupported => |reason| {
                try collector.appendLoopSummaryReason(
                    &collector.loop_summaries.items[summary_index],
                    loopUnsupportedReasonForSemantic(reason),
                );
                try markLoopSummaryDependentsUnsupported(collector, summary_index);
                continue;
            },
        }
        const digest = try obligation_to_lean.loopVcSha256(view, inner.id);
        for (collector.loop_summaries.items[0..summary_index]) |*outer| {
            if (!loopSummaryBindsSummary(outer.*, inner.id)) continue;

            const rebound_edges = try collector.allocator.dupe(obligation.LoopEdgeRow, outer.edges);
            for (outer.edges, rebound_edges) |edge, *rebound_edge| {
                if (!timelineBindsSummary(edge.timeline, inner.id)) continue;
                const rebound_timeline = try collector.allocator.dupe(
                    obligation.LoopTimelineEvent,
                    edge.timeline,
                );
                for (rebound_timeline) |*event| switch (event.payload) {
                    .summary_dependency => |*dependency| {
                        if (dependency.summary_id == inner.id) dependency.loop_vc_sha256 = digest;
                    },
                    else => {},
                };
                rebound_edge.timeline = rebound_timeline;
            }
            outer.edges = rebound_edges;
        }
    }
}

fn timelineBindsSummary(
    timeline: []const obligation.LoopTimelineEvent,
    summary_id: obligation.Id,
) bool {
    for (timeline) |event| switch (event.payload) {
        .summary_dependency => |dependency| if (dependency.summary_id == summary_id) return true,
        else => {},
    };
    return false;
}

fn loopSummaryBindsSummary(summary: obligation.LoopSummaryRow, summary_id: obligation.Id) bool {
    for (summary.edges) |edge| if (timelineBindsSummary(edge.timeline, summary_id)) return true;
    return false;
}

fn markLoopSummaryDependentsUnsupported(
    collector: *Collector,
    summary_index: usize,
) !void {
    const inner_id = collector.loop_summaries.items[summary_index].id;
    for (collector.loop_summaries.items[0..summary_index]) |*outer| {
        if (loopSummaryBindsSummary(outer.*, inner_id)) {
            try collector.appendLoopSummaryReason(outer, .loop_formula_unsupported);
        }
    }
}

fn loopSummaryByIdMut(
    rows: []obligation.LoopSummaryRow,
    id: obligation.Id,
) ?*obligation.LoopSummaryRow {
    for (rows) |*row| if (row.id == id) return row;
    return null;
}

fn verificationQueryByIdMut(
    rows: []obligation.VerificationQuery,
    id: obligation.Id,
) ?*obligation.VerificationQuery {
    for (rows) |*row| if (row.id == id) return row;
    return null;
}

fn loopUniqueEdgeIdForTarget(
    summary: obligation.LoopSummaryRow,
    target: obligation.LoopControlTarget,
) ?u32 {
    var match: ?u32 = null;
    for (summary.edges) |edge| {
        if (edge.target != target) continue;
        if (match != null) return null;
        match = edge.id;
    }
    return match;
}

fn loopPostQueryForEdge(
    queries: []const obligation.VerificationQuery,
    summary: obligation.LoopSummaryRow,
    edge_id: u32,
) ?obligation.Id {
    var post_index: usize = 0;
    for (summary.edges) |edge| {
        if (edge.target == .backedge) continue;
        if (post_index >= summary.query_ids.post.len) return null;
        const query_id = summary.query_ids.post[post_index];
        post_index += 1;
        if (edge.id != edge_id) continue;
        const query = obligation.findById(queries, query_id) orelse return null;
        if (query.kind != .loop_invariant_post or query.loop_edge_id != edge.id) return null;
        return query_id;
    }
    return null;
}

fn loopUnsupportedReasonForSemantic(
    reason: obligation_to_lean.SemanticUnsupportedReason,
) obligation.LoopUnsupportedReason {
    return switch (reason) {
        .loop_summary_missing => .loop_identity_missing,
        .loop_summary_query_mismatch => .loop_query_not_owner_scoped,
        .unsupported_loop_summary => |loop_reason| loop_reason,
        else => .loop_formula_unsupported,
    };
}

const LoopSummaryDraft = struct {
    loop_op: mlir.MlirOperation = std.mem.zeroes(mlir.MlirOperation),
    body: mlir.MlirBlock = std.mem.zeroes(mlir.MlirBlock),
    loop_kind: obligation.LoopKind = .other,
    symbol: ?[]const u8 = null,
    ordinal: u32 = 0,
    guard_formula: ?obligation.FormulaRef = null,
    safety_sites: []const LoopSafetySite = &.{},
    target_sites: []const LoopTargetSite = &.{},
};

const LoopSafetySite = struct {
    op_id: usize,
};

const LoopTargetSite = struct {
    op_id: usize,
    structural_key: []const u8,
    obligation_id: obligation.Id,
};

const CanonicalLoopPostBindingDraft = struct {
    summary_id: obligation.Id,
    edge_id: u32,
    query_id: ?obligation.Id = null,
    source_op_id: usize,
    obligation_id: obligation.Id,
};

const max_loop_projection_paths: usize = 256;

const LoopValueSource = struct {
    value_id: usize,
    key: []const u8,
};

const SemanticLoopPath = struct {
    key: []const u8,
    path_formulas: []const obligation.FormulaRef = &.{},
    formula_overrides: []const Collector.LoopFormulaOverride = &.{},
    source_overrides: []const LoopValueSource = &.{},
    state_source_keys: []const []const u8 = &.{},
    safety_checkpoints: []const obligation.LoopSafetyCheckpoint = &.{},
    target_formulas: []const obligation.FormulaRef = &.{},
    target_keys: []const []const u8 = &.{},
    summary_dependencies: []const obligation.LoopSummaryDependency = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8 = &.{},
    timeline: []const obligation.LoopTimelineEvent = &.{},
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
};

const InventoryLoopPath = struct {
    key: []const u8,
    source_overrides: []const LoopValueSource = &.{},
    state_source_keys: []const []const u8 = &.{},
    safety_checkpoints: []const LoopSafetyCheckpointKey = &.{},
    target_keys: []const []const u8 = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8 = &.{},
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
};

const LoopTimelineInventoryEvent = struct {
    structural_key: []const u8,
    tag: obligation.LoopTimelineEventTag,
    formula_source: ?[]const u8 = null,
    storage_root: ?[]const u8 = null,
    storage_projections: []const LoopStorageProjectionInventory = &.{},
    storage_result: ?obligation.FreeVarId = null,
    storage_value_source: ?[]const u8 = null,
    storage_ty: ?obligation.TypeRef = null,
    summary_dependency: ?obligation.LoopSummaryDependency = null,
};

const ProjectedLoopPreheader = struct {
    operation_keys: []const []const u8 = &.{},
    timeline: []const obligation.LoopTimelineEvent = &.{},
    event_inventory: []const LoopTimelineInventoryEvent = &.{},
    formula_overrides: []const Collector.LoopFormulaOverride = &.{},
};

const InventoryLoopPreheader = struct {
    operation_keys: []const []const u8 = &.{},
    event_inventory: []const LoopTimelineInventoryEvent = &.{},
};

const CollectedLoopPreheader = struct {
    timeline: []const obligation.LoopTimelineEvent = &.{},
    formula_overrides: []const Collector.LoopFormulaOverride = &.{},
    complete: bool = false,
    reason: ?obligation.LoopUnsupportedReason = null,
};

const LoopStorageProjectionInventoryTag = enum {
    field,
    fixed_index,
    mapping_key,
};

const LoopStorageProjectionInventory = struct {
    tag: LoopStorageProjectionInventoryTag,
    source_key: ?[]const u8 = null,
    field: ?[]const u8 = null,
    ty: ?obligation.TypeRef = null,
    extent: ?u64 = null,
};

const ProjectedLoopStorageLocation = struct {
    location: obligation.LoopStorageLocationExpr,
    inventory: []const LoopStorageProjectionInventory,
};

const InventoryLoopStorageLocation = struct {
    root: []const u8,
    projections: []const LoopStorageProjectionInventory,
    value_ty: obligation.TypeRef,
};

const LoopControlKind = enum { break_loop, continue_loop, function_return, error_return };

const LoopControlMarker = struct {
    kind: LoopControlKind,
    source_statement_id: u32,
    target: obligation.LoopControlTarget,
    marker_key: []const u8,
};

const SemanticLoopTerminal = struct {
    path: SemanticLoopPath,
    terminator_key: []const u8,
    yielded: []const mlir.MlirValue,
    control: ?LoopControlMarker = null,
};

const InventoryLoopTerminal = struct {
    path: InventoryLoopPath,
    terminator_key: []const u8,
    yielded: []const mlir.MlirValue,
    control: ?LoopControlMarker = null,
};

const ProjectedLoopEdge = struct {
    edge: obligation.LoopEdgeRow,
    terminator_key: []const u8,
    safety_checkpoints: []const LoopSafetyCheckpointKey,
    target_keys: []const []const u8 = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8,
    state_source_keys: []const []const u8,
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
};

const InventoryLoopEdge = struct {
    structural_key: []const u8,
    target: obligation.LoopControlTarget,
    terminator_key: []const u8,
    binding_source_keys: []const []const u8,
    safety_checkpoints: []const LoopSafetyCheckpointKey,
    target_keys: []const []const u8 = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8,
    state_source_keys: []const []const u8,
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
};

const LoopSafetyCheckpointKey = struct {
    structural_key: []const u8,
    after_dependency_count: u32,
};

const Collector = struct {
    allocator: std.mem.Allocator,
    options: CollectOptions,
    terms: std.ArrayList(obligation.Term) = .empty,
    obligations: std.ArrayList(obligation.Obligation) = .empty,
    assumptions: std.ArrayList(obligation.Assumption) = .empty,
    queries: std.ArrayList(obligation.VerificationQuery) = .empty,
    loop_summaries: std.ArrayList(obligation.LoopSummaryRow) = .empty,
    loop_summary_drafts: std.ArrayList(LoopSummaryDraft) = .empty,
    active_loop_summary_indices: std.ArrayList(usize) = .empty,
    query_bindings: std.ArrayList(obligation.FormalQueryBinding) = .empty,
    source_fact_bindings: std.ArrayList(SourceFactOpBinding) = .empty,
    canonical_loop_post_binding_drafts: std.ArrayList(CanonicalLoopPostBindingDraft) = .empty,
    runtime_owner_bindings: std.ArrayList(RuntimeOwnerBinding) = .empty,
    runtime_check_producers: std.ArrayList(RuntimeCheckProducer) = .empty,
    state_effect_producers: std.ArrayList(StateEffectProducer) = .empty,
    diagnostics: std.ArrayList(obligation.ObligationDiagnostic) = .empty,
    resource_sites: std.ArrayList(ResourceCompletenessSite) = .empty,
    active_binders: std.ArrayList(BinderFrame) = .empty,
    loop_value_bindings: std.ArrayList(LoopValueBinding) = .empty,
    operation_ordinals: std.AutoHashMapUnmanaged(usize, u32) = .empty,
    storage_global_types: std.StringHashMapUnmanaged(obligation.TypeRef) = .empty,
    active_loop_formula_overrides: []const LoopFormulaOverride = &.{},
    arithmetic_safety_terms: std.AutoHashMapUnmanaged(obligation.TermId, void) = .empty,
    next_id: obligation.Id = 1,
    next_loop_summary_id: obligation.Id = 1,
    next_loop_variable_pattern_id: u32 = 0,
    use_loop_value_bindings: bool = false,
    active_loop_depth: u32 = 0,
    next_ordinal: u32 = 0,
    function_param_names: []const []const u8 = &.{},
    function_param_binding_ids: []const obligation.FreeVarId = &.{},
    function_param_types: []const obligation.TypeRef = &.{},
    function_entry_block: mlir.MlirBlock = std.mem.zeroes(mlir.MlirBlock),
    function_write_slots: ?[]const obligation.PlaceRef = null,
    function_write_slots_complete: bool = false,
    function_has_external_call: bool = false,
    function_source_module_path: ?[]const u8 = null,
    function_source_owner_key: ?[]const u8 = null,
    function_source_activation: ?[]const u8 = null,
    function_runtime_symbol: ?[]const u8 = null,
    function_source_specialization_bindings: []const []const u8 = &.{},
    function_source_trait_implementation: ?[]const u8 = null,
    function_source_trait_method: ?[]const u8 = null,
    contract_source_owner_key: ?[]const u8 = null,

    const synthetic_file_id = std.math.maxInt(u32);
    const storage_read_file_id = synthetic_file_id - 2;

    const BinderFrame = struct {
        name: []const u8,
        ty: obligation.TypeRef,
        region: ?obligation.RegionRef = null,
    };

    const LoopValueBinding = struct {
        value_id: usize,
        variable: obligation.LoopVariable,
    };

    const LoopFormulaOverride = struct {
        value_id: usize,
        formula: obligation.FormulaRef,
    };

    const ResourcePlaces = struct {
        source: ?obligation.PlaceRef = null,
        destination: ?obligation.PlaceRef = null,
    };

    const MoveOperandSegments = struct {
        source_len: usize,
        destination_len: usize,
    };

    const ResourceCompletenessSite = struct {
        op: obligation.ResourceOperation,
        domain: []const u8,
        ordinal: u32,
        expected_mask: u32,
        emitted_mask: u32 = 0,
    };

    fn walkOperation(self: *Collector, op: mlir.MlirOperation, inherited_symbol: ?[]const u8) !void {
        if (mlir.oraOperationIsNull(op)) return;

        const ordinal = self.nextOrdinal();
        try self.operation_ordinals.put(self.allocator, @intFromPtr(op.ptr), ordinal);
        const op_name = operationName(op);
        const symbol = try self.symbolForOperation(op, op_name, inherited_symbol);
        if (std.mem.eql(u8, op_name, "ora.global")) {
            const root = (try self.stringAttr(op, "sym_name")) orelse "";
            const type_id = (try self.u32Attr(op, "ora.compiler_type_id")) orelse 0;
            if (root.len != 0) try self.storage_global_types.put(
                self.allocator,
                root,
                .{ .compiler_type_id = type_id },
            );
        }
        const previous_param_names = self.function_param_names;
        const previous_param_binding_ids = self.function_param_binding_ids;
        const previous_param_types = self.function_param_types;
        const previous_entry_block = self.function_entry_block;
        const previous_write_slots = self.function_write_slots;
        const previous_write_slots_complete = self.function_write_slots_complete;
        const previous_has_external_call = self.function_has_external_call;
        const previous_source_module_path = self.function_source_module_path;
        const previous_source_owner_key = self.function_source_owner_key;
        const previous_source_activation = self.function_source_activation;
        const previous_runtime_symbol = self.function_runtime_symbol;
        const previous_source_specialization_bindings = self.function_source_specialization_bindings;
        const previous_source_trait_implementation = self.function_source_trait_implementation;
        const previous_source_trait_method = self.function_source_trait_method;
        const previous_contract_source_owner_key = self.contract_source_owner_key;
        const previous_binder_len = self.active_binders.items.len;
        const previous_loop_value_binding_len = self.loop_value_bindings.items.len;
        const previous_active_loop_summary_len = self.active_loop_summary_indices.items.len;
        const is_function = std.mem.eql(u8, op_name, "func.func");
        const is_contract = std.mem.eql(u8, op_name, "ora.contract");
        if (is_contract) {
            self.contract_source_owner_key = if (try self.stringAttr(op, "sym_name")) |name|
                try std.fmt.allocPrint(self.allocator, "module/contract:{s}", .{name})
            else
                null;
        }
        if (is_function) {
            self.function_param_names = (try self.stringArrayAttr(op, "ora.param_names")) orelse &.{};
            self.function_param_binding_ids = (try self.freeVarIdArrayAttr(op, "ora.param_binding_ids")) orelse &.{};
            self.function_entry_block = functionEntryBlock(op);
            self.function_param_types = try self.functionParamTypesFromFunctionOp(op, self.function_param_names.len);
            self.function_write_slots = try self.placeArrayAttr(op, "ora.write_slots");
            self.function_write_slots_complete = (try self.boolAttr(op, "ora.write_slots_complete")) orelse false;
            self.function_has_external_call = operationBodyHasExternalCall(op);
            self.function_source_module_path = try self.stringAttr(op, "ora.source_module_path");
            self.function_source_owner_key = try self.stringAttr(op, "ora.source_owner_key");
            self.function_source_activation = try self.stringAttr(op, "ora.source_template_activation");
            self.function_runtime_symbol = if (symbol) |name| try self.allocator.dupe(u8, name) else null;
            self.function_source_specialization_bindings = (try self.stringArrayAttr(op, "ora.source_specialization_bindings")) orelse &.{};
            self.function_source_trait_implementation = try self.stringAttr(op, "ora.source_trait_implementation");
            self.function_source_trait_method = try self.stringAttr(op, "ora.source_trait_method");
            if (self.function_source_owner_key) |owner_key| {
                if (self.function_source_module_path) |module_path| {
                    if (self.function_source_activation) |activation| {
                        if (sourceOpId(op)) |source_op_id| try self.runtime_owner_bindings.append(self.allocator, .{
                            .source_op_id = source_op_id,
                            .op_ordinal = ordinal,
                            .source = try self.sourceForOperation(op),
                            .symbol = if (symbol) |name| try self.allocator.dupe(u8, name) else null,
                            .module_path = module_path,
                            .owner_key = owner_key,
                            .template_activation = activation,
                            .specialization_bindings = self.function_source_specialization_bindings,
                            .trait_implementation = self.function_source_trait_implementation,
                            .trait_method = self.function_source_trait_method,
                        });
                    } else {
                        try self.addBlockingDiagnostic(.missing_type, "source-owned func.func is missing ora.source_template_activation");
                    }
                } else {
                    try self.addBlockingDiagnostic(.missing_type, "source-owned func.func is missing ora.source_module_path");
                }
            }
            if (self.function_param_names.len != 0 and self.function_param_binding_ids.len == 0) {
                try self.addBlockingDiagnostic(
                    .missing_type,
                    "func.func has ora.param_names but is missing ora.param_binding_ids",
                );
            } else if (self.function_param_binding_ids.len != 0 and
                self.function_param_names.len != 0 and
                self.function_param_binding_ids.len != self.function_param_names.len)
            {
                try self.addBlockingDiagnosticFmt(
                    .missing_type,
                    "ora.param_binding_ids has {d} entries but ora.param_names has {d}",
                    .{ self.function_param_binding_ids.len, self.function_param_names.len },
                );
            }
            self.active_binders.shrinkRetainingCapacity(previous_binder_len);
            self.loop_value_bindings.shrinkRetainingCapacity(previous_loop_value_binding_len);
            self.active_loop_summary_indices.shrinkRetainingCapacity(previous_active_loop_summary_len);
        }
        defer {
            if (is_function) {
                self.active_binders.shrinkRetainingCapacity(previous_binder_len);
                self.loop_value_bindings.shrinkRetainingCapacity(previous_loop_value_binding_len);
                self.active_loop_summary_indices.shrinkRetainingCapacity(previous_active_loop_summary_len);
                self.function_param_types = previous_param_types;
                self.function_param_binding_ids = previous_param_binding_ids;
                self.function_param_names = previous_param_names;
                self.function_entry_block = previous_entry_block;
                self.function_has_external_call = previous_has_external_call;
                self.function_write_slots_complete = previous_write_slots_complete;
                self.function_write_slots = previous_write_slots;
                self.function_source_module_path = previous_source_module_path;
                self.function_source_owner_key = previous_source_owner_key;
                self.function_source_activation = previous_source_activation;
                self.function_runtime_symbol = previous_runtime_symbol;
                self.function_source_specialization_bindings = previous_source_specialization_bindings;
                self.function_source_trait_implementation = previous_source_trait_implementation;
                self.function_source_trait_method = previous_source_trait_method;
            }
            if (is_contract) self.contract_source_owner_key = previous_contract_source_owner_key;
        }

        const obligation_start = self.obligations.items.len;
        const assumption_start = self.assumptions.items.len;
        const query_start = self.queries.items.len;
        try self.collectEffectFrameAttrs(op, op_name, symbol, ordinal);
        const is_loop_operation = std.mem.eql(u8, op_name, "scf.while") or std.mem.eql(u8, op_name, "scf.for");
        if (is_loop_operation) {
            try self.collectLoopSummary(op, op_name, symbol, ordinal);
        }
        if (op_kind_map.get(op_name)) |kind| {
            try self.collectOperation(op, kind, op_name, symbol, ordinal);
        } else if (arithmetic_op_map.get(op_name)) |safety| {
            if (!self.operationFeedsOnlyFormalFormula(op) and
                !hasAttr(op, "ora.arithmetic_safety_asserted"))
            {
                try self.addArithmeticSafetyOp(op, op_name, symbol, ordinal, safety, null, try self.sourceForOperation(op));
            }
        }
        try self.recordSourceFactBinding(
            op,
            op_name,
            ordinal,
            obligation_start,
            assumption_start,
            query_start,
        );

        const previous_loop_projection = self.use_loop_value_bindings;
        if (is_loop_operation) {
            self.active_loop_depth += 1;
            self.use_loop_value_bindings = true;
            try self.active_loop_summary_indices.append(self.allocator, self.loop_summaries.items.len - 1);
        }
        defer {
            if (is_loop_operation) {
                self.active_loop_depth -= 1;
                self.active_loop_summary_indices.shrinkRetainingCapacity(previous_active_loop_summary_len);
            }
            self.use_loop_value_bindings = previous_loop_projection;
        }

        const num_regions = mlir.oraOperationGetNumRegions(op);
        var region_index: usize = 0;
        while (region_index < num_regions) : (region_index += 1) {
            const region = mlir.oraOperationGetRegion(op, region_index);
            if (mlir.oraRegionIsNull(region)) continue;

            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
                var child = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                    try self.walkOperation(child, symbol);
                }
            }
        }

        if (is_loop_operation) try self.finalizeActiveLoopSummary();

        if (is_function) {
            try self.collectEvidenceBackedEffectFrameAttrs(op, op_name, symbol, ordinal);
            try self.recordFunctionModifiesBindings(op, op_name, symbol, ordinal, obligation_start);
        }
    }

    fn collectLoopSummary(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const previous_loop_projection = self.use_loop_value_bindings;
        self.use_loop_value_bindings = true;
        defer self.use_loop_value_bindings = previous_loop_projection;

        const loop_kind: obligation.LoopKind = if (std.mem.eql(u8, op_name, "scf.while"))
            .scf_while
        else if (std.mem.eql(u8, op_name, "scf.for"))
            .scf_for
        else
            .other;
        const loop_source_op_id = sourceStatementIndex(op);
        const statement_ordinal = loop_source_op_id orelse ordinal;
        const owner: obligation.Owner = .{ .statement = .{
            .function_name = symbol orelse "<module>",
            .ordinal = statement_ordinal,
        } };

        var reasons: std.ArrayList(obligation.LoopUnsupportedReason) = .empty;
        defer reasons.deinit(self.allocator);

        const variables = try self.collectLoopVariables(op, loop_kind);
        try self.bindLoopValues(op, loop_kind, variables);
        const context_variables = try self.collectLoopContextVariables();
        const preheader = try self.collectLoopPreheader(op, symbol, ordinal);
        const previous_formula_overrides = self.active_loop_formula_overrides;
        self.active_loop_formula_overrides = preheader.formula_overrides;
        defer self.active_loop_formula_overrides = previous_formula_overrides;
        const init_formulas = try self.collectLoopInitFormulas(op, loop_kind, variables, symbol, ordinal);
        const guard_formula = try self.collectLoopGuardFormula(op, loop_kind, variables, symbol, ordinal);
        const invariant_formulas = try self.collectLoopInvariantFormulas(op, loop_kind, symbol, ordinal);
        const body = loopBodyBlock(op, loop_kind);
        const body_facts = collectLoopBodyFacts(body);

        if (loop_source_op_id == null) try appendLoopReason(&reasons, self.allocator, .loop_identity_missing);
        for (variables) |variable| {
            if (variable.id == null) try appendLoopReason(&reasons, self.allocator, .loop_identity_missing);
            const value = loopVariableValue(op, loop_kind, variable.index);
            const semantic_integer = typeRefIntegerInfo(variable.ty) != null;
            if (!semantic_integer) try appendLoopReason(&reasons, self.allocator, .loop_value_type_unresolved);
            const carrier_is_integer = !mlir.oraValueIsNull(value) and
                (loop_kind == .scf_for and variable.index == 0 or
                    mlirTypeIsSupportedInteger(mlir.oraValueGetType(value)));
            if (!carrier_is_integer) {
                try appendLoopReason(&reasons, self.allocator, .loop_variable_not_registered_integer);
            }
        }
        if (guard_formula == null) try appendLoopReason(&reasons, self.allocator, .loop_guard_missing);
        if (invariant_formulas.len == 0) try appendLoopReason(&reasons, self.allocator, .loop_invariant_missing);
        if (loop_kind == .other) try appendLoopReason(&reasons, self.allocator, .loop_kind_unsupported);
        if (formulasContainOriginValue(init_formulas) or
            optionalFormulaIsOriginValue(guard_formula) or
            formulasContainOriginValue(invariant_formulas))
        {
            try appendLoopReason(&reasons, self.allocator, .loop_formula_unsupported);
        }
        if (body_facts.has_storage_write) try appendLoopReason(&reasons, self.allocator, .loop_has_storage_write);
        if (body_facts.has_storage_read) try appendLoopReason(&reasons, self.allocator, .loop_has_storage_read);
        if (body_facts.has_call) try appendLoopReason(&reasons, self.allocator, .loop_has_call);
        if (body_facts.has_external_call) try appendLoopReason(&reasons, self.allocator, .loop_has_external_call);
        if (body_facts.has_resource_operation) try appendLoopReason(&reasons, self.allocator, .loop_has_resource_operation);
        if (body_facts.has_break_or_continue) try appendLoopReason(&reasons, self.allocator, .loop_has_break_or_continue);
        if (body_facts.has_error_control_flow) try appendLoopReason(&reasons, self.allocator, .loop_has_error_control_flow);
        const preheader_is_proof_relevant = self.formulasReferenceStoragePlace(init_formulas) or
            self.formulasReferenceStoragePlace(invariant_formulas);
        if (preheader_is_proof_relevant and preheader.reason != null) {
            try appendLoopReason(&reasons, self.allocator, preheader.reason.?);
        }
        if (preheader_is_proof_relevant and !preheader.complete) {
            try appendLoopReason(&reasons, self.allocator, .loop_preheader_storage_relation_unsupported);
        }

        // Until canonical Z3 execution is wired, preserve the reviewed empty
        // prefix representation whenever the complete prefix has no
        // persistent action.  The full collector still inventories every
        // operation; only proof-relevant nonempty storage traces enter the
        // LoopVC digest in this task.
        const preheader_timeline = if (preheader_is_proof_relevant and
            loopTimelineHasPersistentStorageEvent(preheader.timeline))
            preheader.timeline
        else
            &.{};

        const id = self.next_loop_summary_id;
        self.next_loop_summary_id += 1;
        try self.loop_summaries.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = try self.sourceForOperation(op),
            .phase = .report,
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .loop_source_op_id = loop_source_op_id,
            .loop_kind = loop_kind,
            .context_variables = context_variables,
            .variables = variables,
            .init_formulas = init_formulas,
            .invariant_formulas = invariant_formulas,
            .preheader_timeline = preheader_timeline,
            .preheader_projection_complete = preheader.complete,
            .projection_completeness = .unavailable,
            .query_ids = .{},
            .unsupported_reasons = try reasons.toOwnedSlice(self.allocator),
        });
        try self.loop_summary_drafts.append(self.allocator, .{
            .loop_op = op,
            .body = body,
            .loop_kind = loop_kind,
            .symbol = symbol,
            .ordinal = ordinal,
            .guard_formula = guard_formula,
        });
    }

    fn collectLoopPreheader(
        self: *Collector,
        loop_op: mlir.MlirOperation,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !CollectedLoopPreheader {
        const terms_start = self.terms.items.len;
        var retain_projected_terms = false;
        defer if (!retain_projected_terms) self.discardTermsFrom(terms_start);

        const projected = self.projectLoopPreheader(loop_op, symbol, ordinal) catch |err| {
            return .{ .reason = preheaderProjectionFailure(err) orelse return err };
        };
        const inventory = self.inventoryLoopPreheader(loop_op) catch |err| {
            return .{ .reason = preheaderProjectionFailure(err) orelse return err };
        };
        if (!preheaderProjectionMatchesInventory(projected, inventory)) return .{
            .reason = .loop_preheader_inventory_mismatch,
        };
        retain_projected_terms = loopTimelineHasPersistentStorageEvent(projected.timeline);
        return .{
            .timeline = projected.timeline,
            .formula_overrides = projected.formula_overrides,
            .complete = true,
        };
    }

    fn discardTermsFrom(self: *Collector, start: usize) void {
        for (start..self.terms.items.len) |index| {
            _ = self.arithmetic_safety_terms.remove(@intCast(index));
        }
        self.terms.shrinkRetainingCapacity(start);
    }

    fn projectLoopPreheader(
        self: *Collector,
        loop_op: mlir.MlirOperation,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !ProjectedLoopPreheader {
        if (mlir.oraBlockIsNull(self.function_entry_block)) return error.LoopPreheaderControlUnsupported;
        const draft: LoopSummaryDraft = .{ .symbol = symbol, .ordinal = ordinal };
        var result: ProjectedLoopPreheader = .{};
        var op = mlir.oraBlockGetFirstOperation(self.function_entry_block);
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            const key = self.operationStructuralKey(op, name) catch return error.LoopPreheaderControlUnsupported;
            result.operation_keys = try appendString(self.allocator, result.operation_keys, key);
            if (preheaderControlOperationUnsupported(op, name)) return error.LoopPreheaderControlUnsupported;

            const effect = mlir.oraOperationGetMemoryEffectKind(op);
            if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_RECURSIVE or
                effect == mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN)
            {
                return error.LoopPreheaderEffectUnsupported;
            }
            if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_NONE and
                !preheaderPureValueOperationSupported(name))
            {
                return error.LoopPreheaderValueUnsupported;
            }

            if (std.mem.eql(u8, name, "ora.sload")) {
                if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1) return error.LoopPreheaderEffectUnsupported;
                if (isAggregateStorageCarrierLoad(op)) continue;
                const value = mlir.oraOperationGetResult(op, 0);
                const location = self.directStorageLocation(op, value) catch
                    return error.LoopPreheaderLocationUnsupported;
                const read_id = self.storageReadId(op) catch return error.LoopPreheaderValueUnsupported;
                const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                    .id = read_id,
                    .name = location.root,
                    .ty = location.value_ty,
                } } });
                result.formula_overrides = try appendFormulaOverride(self.allocator, result.formula_overrides, .{
                    .value_id = @intFromPtr(value.ptr),
                    .formula = .{ .term = read_term },
                });
                result.timeline = try appendLoopTimelineEvent(self.allocator, result.timeline, .{
                    .storage_read = .{ .result = read_id, .location = location },
                }, key);
                result.event_inventory = try appendLoopTimelineInventoryEvent(self.allocator, result.event_inventory, .{
                    .structural_key = key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_result = read_id,
                    .storage_ty = location.value_ty,
                });
                continue;
            }
            if (std.mem.eql(u8, name, "ora.sstore")) {
                if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 1) return error.LoopPreheaderEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const location = self.directStorageLocation(op, value) catch
                    return error.LoopPreheaderLocationUnsupported;
                const formula = self.loopFormulaFromValueWithOverrides(
                    value,
                    result.formula_overrides,
                    name,
                    symbol,
                    ordinal,
                    0,
                ) catch return error.LoopPreheaderValueUnsupported;
                if (formula == .origin_value) return error.LoopPreheaderValueUnsupported;
                const source = self.sourceKeyForValue(value, &.{}) catch
                    return error.LoopPreheaderValueUnsupported;
                result.timeline = try appendLoopTimelineEvent(self.allocator, result.timeline, .{
                    .storage_write = .{
                        .location = location,
                        .value = .{
                            .formula = formula,
                            .read_results = try self.readResultsForFormula(formula, result.timeline),
                        },
                    },
                }, key);
                result.event_inventory = try appendLoopTimelineInventoryEvent(self.allocator, result.event_inventory, .{
                    .structural_key = key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_value_source = source,
                    .storage_ty = location.value_ty,
                });
                continue;
            }
            if (std.mem.eql(u8, name, "ora.map_get") or std.mem.eql(u8, name, "ora.map_store")) {
                try self.projectPreheaderMappingEvent(op, name, effect, draft, key, &result);
                continue;
            }
            if ((std.mem.eql(u8, name, "memref.load") and inventoryMemrefLoadTargetsPersistentRoot(op)) or
                (std.mem.eql(u8, name, "memref.store") and inventoryMemrefStoreTargetsPersistentRoot(op)) or
                std.mem.startsWith(u8, name, "ora.storage."))
            {
                return error.LoopPreheaderLocationUnsupported;
            }
            if (preheaderAssumptionOperation(name) or isLoopSafetyOperation(op, name)) {
                if (mlir.oraOperationGetNumOperands(op) == 0) return error.LoopPreheaderValueUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const formula = self.loopFormulaFromValueWithOverrides(
                    value,
                    result.formula_overrides,
                    name,
                    symbol,
                    ordinal,
                    0,
                ) catch return error.LoopPreheaderValueUnsupported;
                if (formula == .origin_value) return error.LoopPreheaderValueUnsupported;
                const source = self.sourceKeyForValue(value, &.{}) catch
                    return error.LoopPreheaderValueUnsupported;
                const payload: obligation.LoopTimelineEventPayload = if (preheaderAssumptionOperation(name))
                    .{ .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(&.{formula}, result.timeline) } }
                else
                    .{ .safety_checkpoint = .{ .formulas = try self.loopFormulaExprsWithReads(&.{formula}, result.timeline) } };
                result.timeline = try appendLoopTimelineEvent(self.allocator, result.timeline, payload, key);
                result.event_inventory = try appendLoopTimelineInventoryEvent(self.allocator, result.event_inventory, .{
                    .structural_key = key,
                    .tag = std.meta.activeTag(payload),
                    .formula_source = source,
                });
                continue;
            }
            if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_NONE) return error.LoopPreheaderEffectUnsupported;
        }
        if (mlir.oraOperationIsNull(op)) return error.LoopPreheaderControlUnsupported;
        return result;
    }

    fn projectPreheaderMappingEvent(
        self: *Collector,
        op: mlir.MlirOperation,
        name: []const u8,
        effect: mlir.OraOperationMemoryEffectKind,
        draft: LoopSummaryDraft,
        key: []const u8,
        result: *ProjectedLoopPreheader,
    ) !void {
        const is_read = std.mem.eql(u8, name, "ora.map_get");
        const expected_effect: mlir.OraOperationMemoryEffectKind = if (is_read)
            mlir.ORA_OPERATION_MEMORY_EFFECT_READ
        else
            mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE;
        const expected_operands: usize = if (is_read) 2 else 3;
        if (effect != expected_effect) {
            return error.LoopPreheaderEffectUnsupported;
        }
        if (mlir.oraOperationGetNumOperands(op) != expected_operands or
            is_read and mlir.oraOperationGetNumResults(op) != 1)
        {
            return error.LoopPreheaderEffectUnsupported;
        }
        const value = if (is_read) mlir.oraOperationGetResult(op, 0) else mlir.oraOperationGetOperand(op, 2);
        if (is_read and isAggregateStorageValue(value)) return;
        const projected = self.projectedMapStorageLocation(
            op,
            mlir.oraOperationGetOperand(op, 0),
            mlir.oraOperationGetOperand(op, 1),
            value,
            result.formula_overrides,
            &.{},
            result.timeline,
            draft,
        ) catch return error.LoopPreheaderLocationUnsupported;
        if (projected.location.projections.len != 1 or
            !self.preheaderMappingProjectionReviewed(projected.location.projections[0]))
        {
            return error.LoopPreheaderLocationUnsupported;
        }
        if (is_read) {
            const read_id = self.storageReadId(op) catch return error.LoopPreheaderValueUnsupported;
            const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                .id = read_id,
                .name = projected.location.root,
                .ty = projected.location.value_ty,
            } } });
            result.formula_overrides = try appendFormulaOverride(self.allocator, result.formula_overrides, .{
                .value_id = @intFromPtr(value.ptr),
                .formula = .{ .term = read_term },
            });
            result.timeline = try appendLoopTimelineEvent(self.allocator, result.timeline, .{
                .storage_read = .{ .result = read_id, .location = projected.location },
            }, key);
            result.event_inventory = try appendLoopTimelineInventoryEvent(self.allocator, result.event_inventory, .{
                .structural_key = key,
                .tag = .storage_read,
                .storage_root = projected.location.root,
                .storage_projections = projected.inventory,
                .storage_result = read_id,
                .storage_ty = projected.location.value_ty,
            });
            return;
        }
        const formula = self.loopFormulaFromValueWithOverrides(
            value,
            result.formula_overrides,
            name,
            draft.symbol,
            draft.ordinal,
            2,
        ) catch return error.LoopPreheaderValueUnsupported;
        if (formula == .origin_value) return error.LoopPreheaderValueUnsupported;
        const source = self.sourceKeyForValue(value, &.{}) catch return error.LoopPreheaderValueUnsupported;
        result.timeline = try appendLoopTimelineEvent(self.allocator, result.timeline, .{
            .storage_write = .{
                .location = projected.location,
                .value = .{
                    .formula = formula,
                    .read_results = try self.readResultsForFormula(formula, result.timeline),
                },
            },
        }, key);
        result.event_inventory = try appendLoopTimelineInventoryEvent(self.allocator, result.event_inventory, .{
            .structural_key = key,
            .tag = .storage_write,
            .storage_root = projected.location.root,
            .storage_projections = projected.inventory,
            .storage_value_source = source,
            .storage_ty = projected.location.value_ty,
        });
    }

    fn preheaderMappingProjectionReviewed(
        self: *const Collector,
        projection: obligation.LoopStorageProjection,
    ) bool {
        const key = switch (projection) {
            .mapping_key => |key| key,
            .field, .fixed_index => return false,
        };
        if (key.value.read_results.len != 0) return false;
        const term_id = switch (key.value.formula) {
            .term => |id| id,
            .origin_value => return false,
        };
        if (term_id >= self.terms.items.len) return false;
        const id = switch (self.terms.items[term_id]) {
            .variable => |variable| switch (variable) {
                .free => |free| free.id,
                .bound => return false,
            },
            else => return false,
        };
        for (self.function_param_binding_ids) |parameter_id| {
            if (obligation.freeVarIdEql(parameter_id, id)) return true;
        }
        return false;
    }

    fn inventoryLoopPreheader(
        self: *Collector,
        loop_op: mlir.MlirOperation,
    ) !InventoryLoopPreheader {
        if (mlir.oraBlockIsNull(self.function_entry_block)) return error.LoopPreheaderControlUnsupported;
        var result: InventoryLoopPreheader = .{};
        var op = mlir.oraBlockGetFirstOperation(self.function_entry_block);
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            const key = self.operationStructuralKey(op, name) catch return error.LoopPreheaderControlUnsupported;
            result.operation_keys = try appendString(self.allocator, result.operation_keys, key);
            if (preheaderControlOperationUnsupported(op, name)) return error.LoopPreheaderControlUnsupported;

            const effect = mlir.oraOperationGetMemoryEffectKind(op);
            if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_RECURSIVE or
                effect == mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN)
            {
                return error.LoopPreheaderEffectUnsupported;
            }
            if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_NONE and
                !preheaderPureValueOperationSupported(name))
            {
                return error.LoopPreheaderValueUnsupported;
            }

            var event: ?LoopTimelineInventoryEvent = null;
            if (std.mem.eql(u8, name, "ora.sload")) {
                if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1) return error.LoopPreheaderEffectUnsupported;
                if (isAggregateStorageCarrierLoad(op)) continue;
                const value = mlir.oraOperationGetResult(op, 0);
                const location = self.directStorageLocation(op, value) catch
                    return error.LoopPreheaderLocationUnsupported;
                event = .{
                    .structural_key = key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_result = self.storageReadId(op) catch return error.LoopPreheaderValueUnsupported,
                    .storage_ty = location.value_ty,
                };
            } else if (std.mem.eql(u8, name, "ora.sstore")) {
                if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 1) return error.LoopPreheaderEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const location = self.directStorageLocation(op, value) catch
                    return error.LoopPreheaderLocationUnsupported;
                event = .{
                    .structural_key = key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_value_source = self.sourceKeyForValue(value, &.{}) catch
                        return error.LoopPreheaderValueUnsupported,
                    .storage_ty = location.value_ty,
                };
            } else if (std.mem.eql(u8, name, "ora.map_get") or std.mem.eql(u8, name, "ora.map_store")) {
                const is_read = std.mem.eql(u8, name, "ora.map_get");
                const expected_effect: mlir.OraOperationMemoryEffectKind = if (is_read)
                    mlir.ORA_OPERATION_MEMORY_EFFECT_READ
                else
                    mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE;
                const expected_operands: usize = if (is_read) 2 else 3;
                if (effect != expected_effect or
                    mlir.oraOperationGetNumOperands(op) != expected_operands or
                    is_read and mlir.oraOperationGetNumResults(op) != 1)
                {
                    return error.LoopPreheaderEffectUnsupported;
                }
                const value = if (is_read) mlir.oraOperationGetResult(op, 0) else mlir.oraOperationGetOperand(op, 2);
                if (is_read and isAggregateStorageValue(value)) continue;
                const location = self.inventoryMapStorageLocation(
                    mlir.oraOperationGetOperand(op, 0),
                    mlir.oraOperationGetOperand(op, 1),
                    value,
                    &.{},
                ) catch return error.LoopPreheaderLocationUnsupported;
                if (location.projections.len != 1 or location.projections[0].tag != .mapping_key) {
                    return error.LoopPreheaderLocationUnsupported;
                }
                event = .{
                    .structural_key = key,
                    .tag = if (is_read) .storage_read else .storage_write,
                    .storage_root = location.root,
                    .storage_projections = location.projections,
                    .storage_result = if (is_read)
                        self.storageReadId(op) catch return error.LoopPreheaderValueUnsupported
                    else
                        null,
                    .storage_value_source = if (!is_read)
                        self.sourceKeyForValue(value, &.{}) catch return error.LoopPreheaderValueUnsupported
                    else
                        null,
                    .storage_ty = location.value_ty,
                };
            } else if (preheaderAssumptionOperation(name) or isLoopSafetyOperation(op, name)) {
                if (mlir.oraOperationGetNumOperands(op) == 0) return error.LoopPreheaderValueUnsupported;
                event = .{
                    .structural_key = key,
                    .tag = if (preheaderAssumptionOperation(name)) .path_assumption else .safety_checkpoint,
                    .formula_source = self.sourceKeyForValue(mlir.oraOperationGetOperand(op, 0), &.{}) catch
                        return error.LoopPreheaderValueUnsupported,
                };
            } else if ((std.mem.eql(u8, name, "memref.load") and inventoryMemrefLoadTargetsPersistentRoot(op)) or
                (std.mem.eql(u8, name, "memref.store") and inventoryMemrefStoreTargetsPersistentRoot(op)) or
                std.mem.startsWith(u8, name, "ora.storage."))
            {
                return error.LoopPreheaderLocationUnsupported;
            } else if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_NONE) {
                return error.LoopPreheaderEffectUnsupported;
            }
            if (event) |row| result.event_inventory = try appendLoopTimelineInventoryEvent(
                self.allocator,
                result.event_inventory,
                row,
            );
        }
        if (mlir.oraOperationIsNull(op)) return error.LoopPreheaderControlUnsupported;
        return result;
    }

    fn formulasReferenceStoragePlace(
        self: *const Collector,
        formulas: []const obligation.FormulaRef,
    ) bool {
        for (formulas) |formula| switch (formula) {
            .origin_value => {},
            .term => |id| if (self.termReferencesStoragePlace(id, self.terms.items.len + 1)) return true,
        };
        return false;
    }

    fn termReferencesStoragePlace(self: *const Collector, id: obligation.TermId, fuel: usize) bool {
        if (fuel == 0 or id >= self.terms.items.len) return false;
        return switch (self.terms.items[id]) {
            .place_read => |place| place.region == .storage,
            .old => |operand| self.termReferencesStoragePlace(operand, fuel - 1),
            .unary => |unary| self.termReferencesStoragePlace(unary.operand, fuel - 1),
            .binary => |binary| self.termReferencesStoragePlace(binary.lhs, fuel - 1) or
                self.termReferencesStoragePlace(binary.rhs, fuel - 1),
            .refinement_predicate => |predicate| blk: {
                if (self.termReferencesStoragePlace(predicate.value, fuel - 1)) break :blk true;
                for (predicate.args) |arg| {
                    if (self.termReferencesStoragePlace(arg, fuel - 1)) break :blk true;
                }
                break :blk false;
            },
            .quantified => |quantified| blk: {
                if (quantified.condition) |condition| {
                    if (self.termReferencesStoragePlace(condition, fuel - 1)) break :blk true;
                }
                break :blk self.termReferencesStoragePlace(quantified.body, fuel - 1);
            },
            .bool_lit, .int_lit, .variable, .result => false,
        };
    }

    fn recordActiveLoopBodySafetySite(
        self: *Collector,
        op: mlir.MlirOperation,
        formula: obligation.FormulaRef,
    ) !void {
        if (self.active_loop_summary_indices.items.len == 0) return;
        const summary_index = self.active_loop_summary_indices.items[self.active_loop_summary_indices.items.len - 1];
        const summary = &self.loop_summaries.items[summary_index];
        const draft = &self.loop_summary_drafts.items[summary_index];
        const sites = try self.allocator.alloc(LoopSafetySite, draft.safety_sites.len + 1);
        @memcpy(sites[0..draft.safety_sites.len], draft.safety_sites);
        sites[sites.len - 1] = .{ .op_id = @intFromPtr(op.ptr) };
        draft.safety_sites = sites;
        if (formula == .origin_value) try self.appendLoopSummaryReason(summary, .loop_formula_unsupported);
    }

    fn recordActiveLoopTargetSite(
        self: *Collector,
        op: mlir.MlirOperation,
        formula: obligation.FormulaRef,
        obligation_id: obligation.Id,
    ) !void {
        if (self.active_loop_summary_indices.items.len == 0) return;
        const summary_index = self.active_loop_summary_indices.items[self.active_loop_summary_indices.items.len - 1];
        const summary = &self.loop_summaries.items[summary_index];
        const draft = &self.loop_summary_drafts.items[summary_index];
        const sites = try self.allocator.alloc(LoopTargetSite, draft.target_sites.len + 1);
        @memcpy(sites[0..draft.target_sites.len], draft.target_sites);
        sites[sites.len - 1] = .{
            .op_id = @intFromPtr(op.ptr),
            .structural_key = try self.operationStructuralKey(op, operationName(op)),
            .obligation_id = obligation_id,
        };
        draft.target_sites = sites;
        if (formula == .origin_value) try self.appendLoopSummaryReason(summary, .loop_formula_unsupported);
    }

    fn appendLoopSummaryReason(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
        reason: obligation.LoopUnsupportedReason,
    ) !void {
        for (summary.unsupported_reasons) |existing| if (existing == reason) return;
        const reasons = try self.allocator.alloc(obligation.LoopUnsupportedReason, summary.unsupported_reasons.len + 1);
        @memcpy(reasons[0..summary.unsupported_reasons.len], summary.unsupported_reasons);
        reasons[reasons.len - 1] = reason;
        summary.unsupported_reasons = reasons;
    }

    fn removeLoopSummaryReason(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
        reason: obligation.LoopUnsupportedReason,
    ) !void {
        var count: usize = 0;
        for (summary.unsupported_reasons) |existing| count += @intFromBool(existing != reason);
        if (count == summary.unsupported_reasons.len) return;
        const reasons = try self.allocator.alloc(obligation.LoopUnsupportedReason, count);
        var index: usize = 0;
        for (summary.unsupported_reasons) |existing| {
            if (existing == reason) continue;
            reasons[index] = existing;
            index += 1;
        }
        summary.unsupported_reasons = reasons;
    }

    fn armCanonicalLoopPostBindings(
        self: *Collector,
        summary_id: obligation.Id,
    ) !void {
        const summary = loopSummaryByIdMut(self.loop_summaries.items, summary_id) orelse
            return error.InvalidLoopSummary;
        const loop_source_op_id: usize = summary.loop_source_op_id orelse
            return error.InvalidLoopSummary;

        for (self.canonical_loop_post_binding_drafts.items) |draft| {
            if (draft.summary_id != summary_id) continue;
            const query_id = draft.query_id orelse loopPostQueryForEdge(
                self.queries.items,
                summary.*,
                draft.edge_id,
            ) orelse return error.MissingLoopPostQuery;
            const query = verificationQueryByIdMut(self.queries.items, query_id) orelse
                return error.MissingLoopPostQuery;
            if (query.kind != .loop_invariant_post or query.loop_summary_id != summary_id) {
                return error.InvalidLoopPostQuery;
            }
            query.obligation_ids = try appendIdUnique(
                self.allocator,
                query.obligation_ids,
                draft.obligation_id,
            );

            const binding_obligation_ids = try self.allocator.alloc(obligation.Id, 1);
            binding_obligation_ids[0] = draft.obligation_id;
            try self.query_bindings.append(self.allocator, .{
                .source_op_id = draft.source_op_id,
                .loop_source_op_id = loop_source_op_id,
                .kind = .loop_invariant_post,
                .logical_role = .ensures,
                .query_id = query_id,
                .assumption_ids = query.assumption_ids,
                .obligation_ids = binding_obligation_ids,
            });

            for (self.query_bindings.items) |binding| {
                if (binding.source_op_id != draft.source_op_id or
                    binding.kind != .obligation or
                    binding.logical_role != .ensures) continue;
                const generic_query = verificationQueryByIdMut(self.queries.items, binding.query_id) orelse
                    return error.InvalidLoopPostQuery;
                generic_query.artifact_policy = .diagnostic_only;
            }

            var source_binding_found = false;
            for (self.source_fact_bindings.items) |*source_binding| {
                if (source_binding.source_op_id != draft.source_op_id) continue;
                if (source_binding_found) return error.DuplicateLoopPostSourceBinding;
                source_binding_found = true;
                source_binding.query_ids = try replaceGenericEnsuresQueryWithCanonical(
                    self.allocator,
                    self.queries.items,
                    source_binding.query_ids,
                    query_id,
                );
            }
            if (!source_binding_found) return error.MissingLoopPostSourceBinding;
        }
    }

    fn finalizeActiveLoopSummary(self: *Collector) !void {
        if (self.active_loop_summary_indices.items.len == 0) return error.MissingActiveLoopSummary;
        const summary_index = self.active_loop_summary_indices.items[self.active_loop_summary_indices.items.len - 1];
        const summary = &self.loop_summaries.items[summary_index];
        const draft = self.loop_summary_drafts.items[summary_index];
        try self.buildCanonicalLoopEdges(summary, draft);
        if (summary.validationReason()) |reason| try self.appendLoopSummaryReason(summary, reason);
        try self.addLoopInductionQuery(summary);
    }

    fn buildCanonicalLoopEdges(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
        draft: LoopSummaryDraft,
    ) !void {
        const guard = draft.guard_formula orelse return;
        if (summary.variables.len == 0 or mlir.oraBlockIsNull(draft.body)) return;
        for (summary.variables) |variable| {
            if (variable.id == null or typeRefIntegerInfo(variable.ty) == null) {
                summary.projection_completeness = .binding_mismatch;
                return;
            }
        }

        const control_manifest = self.loopControlManifest(draft.loop_op) catch |err| {
            summary.projection_completeness = loopControlProjectionFailure(err) orelse return err;
            return;
        };

        var semantic_terminals: std.ArrayList(SemanticLoopTerminal) = .empty;
        defer semantic_terminals.deinit(self.allocator);
        const initial_path_key = "body/path";
        const semantic_path: SemanticLoopPath = .{
            .key = "body",
            .path_formulas = try appendFormula(self.allocator, &.{}, guard),
            .timeline = try appendLoopTimelineEvent(
                self.allocator,
                &.{},
                .{ .path_assumption = .{
                    .formulas = try loopFormulaExprs(self.allocator, &.{guard}),
                } },
                initial_path_key,
            ),
        };
        self.projectSemanticLoopSequence(
            mlir.oraBlockGetFirstOperation(draft.body),
            semantic_path,
            draft,
            summary,
            &semantic_terminals,
        ) catch |err| switch (err) {
            error.LoopControlMarkerMalformed,
            error.LoopControlMarkerTargetMismatch,
            => {
                summary.projection_completeness = loopControlProjectionFailure(err).?;
                return;
            },
            error.UnsupportedLoopControlFlow => {
                summary.projection_completeness = .unsupported_control_flow;
                return;
            },
            error.LoopProjectionPathLimitExceeded => {
                summary.projection_completeness = .path_limit_exceeded;
                return;
            },
            error.LoopProjectionEffectUnsupported => {
                summary.projection_completeness = .unsupported_effect;
                return;
            },
            error.LoopTargetFormulaTargetMismatch => {
                summary.projection_completeness = .target_mismatch;
                return;
            },
            else => return err,
        };
        self.validateSemanticLoopControlManifest(control_manifest, semantic_terminals.items) catch |err| {
            summary.projection_completeness = loopControlProjectionFailure(err) orelse return err;
            return;
        };

        var inventory_terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
        defer inventory_terminals.deinit(self.allocator);
        self.inventoryLoopSequence(
            mlir.oraBlockGetFirstOperation(draft.body),
            .{
                .key = "body",
                .timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, &.{}, .{
                    .structural_key = initial_path_key,
                    .tag = .path_assumption,
                }),
            },
            draft,
            summary,
            &inventory_terminals,
        ) catch |err| switch (err) {
            error.LoopControlMarkerMalformed,
            error.LoopControlMarkerTargetMismatch,
            => {
                summary.projection_completeness = loopControlProjectionFailure(err).?;
                return;
            },
            error.UnsupportedLoopControlFlow => {
                summary.projection_completeness = .unsupported_control_flow;
                return;
            },
            error.LoopProjectionPathLimitExceeded => {
                summary.projection_completeness = .path_limit_exceeded;
                return;
            },
            error.LoopProjectionEffectUnsupported => {
                summary.projection_completeness = .unsupported_effect;
                return;
            },
            error.LoopTargetFormulaTargetMismatch => {
                summary.projection_completeness = .target_mismatch;
                return;
            },
            else => return err,
        };
        self.validateInventoryLoopControlManifest(control_manifest, inventory_terminals.items) catch |err| {
            summary.projection_completeness = loopControlProjectionFailure(err) orelse return err;
            return;
        };

        var projected: std.ArrayList(ProjectedLoopEdge) = .empty;
        defer projected.deinit(self.allocator);
        for (semantic_terminals.items) |terminal| {
            const edge = self.projectedTerminalEdge(summary, draft, terminal) catch |err| switch (err) {
                error.LoopTargetFormulaTargetMismatch => {
                    summary.projection_completeness = .target_mismatch;
                    return;
                },
                else => return err,
            };
            try projected.append(self.allocator, edge);
        }

        var inventory: std.ArrayList(InventoryLoopEdge) = .empty;
        defer inventory.deinit(self.allocator);
        for (inventory_terminals.items) |terminal| {
            const edge = self.inventoryTerminalEdge(summary, draft, terminal) catch |err| switch (err) {
                error.LoopTargetFormulaTargetMismatch => {
                    summary.projection_completeness = .target_mismatch;
                    return;
                },
                else => return err,
            };
            try inventory.append(self.allocator, edge);
        }

        const exit_bindings = try self.allocator.alloc(obligation.LoopStepAssignment, summary.variables.len);
        for (summary.variables, exit_bindings) |variable, *binding| {
            const id = variable.id orelse return;
            const value = try self.addTerm(.{ .variable = .{ .free = .{
                .id = id,
                .name = variable.name orelse "",
                .ty = variable.ty,
            } } });
            binding.* = .{
                .variable_index = variable.index,
                .target = id,
                .source_key = try std.fmt.allocPrint(self.allocator, "loop-state:{d}", .{variable.index}),
                .value = .{ .term = value },
            };
        }

        const negated_guard: obligation.FormulaRef = switch (guard) {
            .term => |term| .{ .term = try self.addTerm(.{ .unary = .{
                .op = .not,
                .operand = term,
            } }) },
            .origin_value => guard,
        };
        const exit_path = try self.allocator.alloc(obligation.FormulaRef, 1);
        exit_path[0] = negated_guard;

        try projected.append(self.allocator, .{
            .edge = .{
                .id = @intCast(projected.items.len),
                .structural_key = "guard/false-successor",
                .timeline = try appendLoopTimelineEvent(self.allocator, &.{}, .{
                    .path_assumption = .{ .formulas = try loopFormulaExprs(self.allocator, exit_path) },
                }, "guard/false-successor/path"),
                .next_state = exit_bindings,
                .target = .successor,
            },
            .terminator_key = "guard/false-successor",
            .safety_checkpoints = &.{},
            .target_keys = &.{},
            .effect_keys = &.{},
            .state_source_keys = &.{},
        });
        const exit_source_keys = try self.allocator.alloc([]const u8, exit_bindings.len);
        for (exit_bindings, exit_source_keys) |binding, *key| key.* = binding.source_key;
        try inventory.append(self.allocator, .{
            .structural_key = "guard/false-successor",
            .target = .successor,
            .terminator_key = "guard/false-successor",
            .binding_source_keys = exit_source_keys,
            .safety_checkpoints = &.{},
            .target_keys = &.{},
            .effect_keys = &.{},
            .state_source_keys = &.{},
            .timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, &.{}, .{
                .structural_key = "guard/false-successor/path",
                .tag = .path_assumption,
            }),
        });

        summary.projection_completeness = compareLoopProjection(projected.items, inventory.items);
        if (summary.projection_completeness != .complete) return;
        var has_storage_event = false;
        for (projected.items) |item| has_storage_event = has_storage_event or
            timelineInventoryHasStorageEvent(item.timeline_inventory);
        if (has_storage_event and self.function_write_slots_complete) {
            try self.removeLoopSummaryReason(summary, .loop_has_storage_read);
            try self.removeLoopSummaryReason(summary, .loop_has_storage_write);
        }

        const edges = try self.allocator.alloc(obligation.LoopEdgeRow, projected.items.len);
        for (projected.items, edges, 0..) |item, *edge, index| {
            edge.* = item.edge;
            edge.id = @intCast(index);
            if (edge.target == .function_return or edge.target == .error_return) {
                if (edge.target_formulas.len != item.target_keys.len) {
                    summary.projection_completeness = .target_mismatch;
                    return;
                }
                for (item.target_keys) |target_key| {
                    const site = loopTargetSiteForStructuralKey(draft, target_key) orelse {
                        summary.projection_completeness = .target_mismatch;
                        return;
                    };
                    try self.canonical_loop_post_binding_drafts.append(self.allocator, .{
                        .summary_id = summary.id,
                        .edge_id = edge.id,
                        .source_op_id = site.op_id,
                        .obligation_id = site.obligation_id,
                    });
                }
            }
            if (loopTimelineContainsOriginValue(edge.timeline) or
                formulasContainOriginValue(edge.target_formulas) or
                stepAssignmentsContainOriginValue(edge.next_state))
            {
                try self.appendLoopSummaryReason(summary, .loop_formula_unsupported);
            }
            if (edge.target == .backedge and !loopStepAssignmentsAreScalar(summary.variables, edge.next_state)) {
                try self.appendLoopSummaryReason(summary, .loop_update_not_scalar_assignment);
            }
            if (edge.target == .backedge and
                self.stepAssignmentsRequireSafety(edge.next_state) and
                edge.safetyFormulaCount() == 0)
            {
                try self.appendLoopSummaryReason(summary, .loop_body_safety_missing);
            }
        }
        summary.edges = edges;
        if (!formulasContainOriginValue(summary.init_formulas) and
            !formulasContainOriginValue(summary.invariant_formulas) and
            !formulasContainOriginValue(summary.post_formulas))
        {
            var edges_supported = true;
            for (edges) |edge| {
                if (loopTimelineContainsOriginValue(edge.timeline) or
                    formulasContainOriginValue(edge.target_formulas) or
                    stepAssignmentsContainOriginValue(edge.next_state)) edges_supported = false;
            }
            if (edges_supported) try self.removeLoopSummaryReason(summary, .loop_formula_unsupported);
        }
    }

    fn projectSemanticLoopSequence(
        self: *Collector,
        first: mlir.MlirOperation,
        initial_path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) anyerror!void {
        var path = initial_path;
        var op = first;
        while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            if (std.mem.eql(u8, name, "ora.loop_control")) {
                try self.appendSemanticLoopControlTerminal(op, path, draft, summary, terminals);
                return;
            }
            if (isRegionYield(name)) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, .{
                    .path = path,
                    .terminator_key = try self.operationStructuralKey(op, name),
                    .yielded = try self.operationOperands(op),
                });
                return;
            }
            if (std.mem.eql(u8, name, "scf.if")) {
                if (try isLoopControlRuntimeGuard(op)) {
                    try self.projectSemanticLoopControlRuntimeGuard(op, path, draft, summary, terminals);
                    return;
                }
                try self.projectSemanticIf(op, path, draft, summary, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                try self.projectSemanticSwitch(op, path, draft, summary, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
                path = try self.bindSemanticNestedLoop(op, path);
                continue;
            }
            if (mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0) {
                return error.UnsupportedLoopControlFlow;
            }

            const effect_key = try self.operationStructuralKey(op, name);
            const effect_kind = mlir.oraOperationGetMemoryEffectKind(op);
            switch (effect_kind) {
                mlir.ORA_OPERATION_MEMORY_EFFECT_NONE => {},
                mlir.ORA_OPERATION_MEMORY_EFFECT_READ,
                mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE,
                => path.effect_keys = try appendString(self.allocator, path.effect_keys, effect_key),
                mlir.ORA_OPERATION_MEMORY_EFFECT_RECURSIVE,
                mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN,
                => return error.LoopProjectionEffectUnsupported,
                else => return error.LoopProjectionEffectUnsupported,
            }
            if (!directRootLoopEffectSupported(op, name, effect_kind)) {
                return error.LoopProjectionEffectUnsupported;
            }
            if (std.mem.eql(u8, name, "ora.sload")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1) return error.LoopProjectionEffectUnsupported;
                if (isAggregateStorageCarrierLoad(op)) continue;
                const result = mlir.oraOperationGetResult(op, 0);
                const location = try self.directStorageLocation(op, result);
                const read_id = try self.storageReadId(op);
                const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                    .id = read_id,
                    .name = location.root,
                    .ty = location.value_ty,
                } } });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .formula = .{ .term = read_term },
                });
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_read = .{ .result = read_id, .location = location },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_result = read_id,
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.sstore")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 1) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const location = try self.directStorageLocation(op, value);
                const formula = try self.loopFormulaFromValueWithOverrides(
                    value,
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    0,
                );
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_write = .{
                        .location = location,
                        .value = .{
                            .formula = formula,
                            .read_results = try self.readResultsForFormula(formula, path.timeline),
                        },
                    },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "memref.load") and
                inventoryMemrefLoadTargetsPersistentRoot(op))
            {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1 or
                    mlir.oraOperationGetNumOperands(op) < 2) return error.LoopProjectionEffectUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                const projected = try self.projectedMemrefStorageLocation(
                    op,
                    0,
                    1,
                    result,
                    path.formula_overrides,
                    path.source_overrides,
                    path.timeline,
                    draft,
                );
                const read_id = try self.storageReadId(op);
                const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                    .id = read_id,
                    .name = projected.location.root,
                    .ty = projected.location.value_ty,
                } } });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .formula = .{ .term = read_term },
                });
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_read = .{ .result = read_id, .location = projected.location },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = projected.location.root,
                    .storage_projections = projected.inventory,
                    .storage_result = read_id,
                    .storage_ty = projected.location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "memref.store") and
                inventoryMemrefStoreTargetsPersistentRoot(op))
            {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) < 3) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const projected = try self.projectedMemrefStorageLocation(
                    op,
                    1,
                    2,
                    value,
                    path.formula_overrides,
                    path.source_overrides,
                    path.timeline,
                    draft,
                );
                const formula = try self.loopFormulaFromValueWithOverrides(
                    value,
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    0,
                );
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_write = .{
                        .location = projected.location,
                        .value = .{
                            .formula = formula,
                            .read_results = try self.readResultsForFormula(formula, path.timeline),
                        },
                    },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = projected.location.root,
                    .storage_projections = projected.inventory,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = projected.location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.map_get")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1 or
                    mlir.oraOperationGetNumOperands(op) != 2) return error.LoopProjectionEffectUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                if (isAggregateStorageValue(result)) continue;
                const projected = try self.projectedMapStorageLocation(
                    op,
                    mlir.oraOperationGetOperand(op, 0),
                    mlir.oraOperationGetOperand(op, 1),
                    result,
                    path.formula_overrides,
                    path.source_overrides,
                    path.timeline,
                    draft,
                );
                const read_id = try self.storageReadId(op);
                const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                    .id = read_id,
                    .name = projected.location.root,
                    .ty = projected.location.value_ty,
                } } });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .formula = .{ .term = read_term },
                });
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_read = .{ .result = read_id, .location = projected.location },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = projected.location.root,
                    .storage_projections = projected.inventory,
                    .storage_result = read_id,
                    .storage_ty = projected.location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.map_store")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 3) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 2);
                const projected = try self.projectedMapStorageLocation(
                    op,
                    mlir.oraOperationGetOperand(op, 0),
                    mlir.oraOperationGetOperand(op, 1),
                    value,
                    path.formula_overrides,
                    path.source_overrides,
                    path.timeline,
                    draft,
                );
                const formula = try self.loopFormulaFromValueWithOverrides(
                    value,
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    2,
                );
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .storage_write = .{
                        .location = projected.location,
                        .value = .{
                            .formula = formula,
                            .read_results = try self.readResultsForFormula(formula, path.timeline),
                        },
                    },
                }, effect_key);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = projected.location.root,
                    .storage_projections = projected.inventory,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = projected.location.value_ty,
                });
            }
            if (draftHasSafetySite(draft, op)) {
                if (mlir.oraOperationGetNumOperands(op) == 0) return error.LoopProjectionEffectUnsupported;
                const formula = try self.loopFormulaFromValueWithOverrides(
                    mlir.oraOperationGetOperand(op, 0),
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    0,
                );
                if (arithmeticSafetyFromAssertOp(op) == .fixed_array_bounds) {
                    // This seam only reuses the source-owned comparison's lhs
                    // identity. fixedIndexBoundsReason separately requires the
                    // exact `lt(index, canonical extent)` term before support.
                    const predicate = mlir.oraOperationGetOperand(op, 0);
                    if (!mlir.oraValueIsAOpResult(predicate)) return error.LoopProjectionEffectUnsupported;
                    const compare = mlir.oraOpResultGetOwner(predicate);
                    if (mlir.oraOperationIsNull(compare) or mlir.oraOperationGetNumOperands(compare) < 2) {
                        return error.LoopProjectionEffectUnsupported;
                    }
                    const formula_id = switch (formula) {
                        .term => |id| id,
                        .origin_value => return error.LoopProjectionEffectUnsupported,
                    };
                    if (formula_id >= self.terms.items.len) return error.LoopProjectionEffectUnsupported;
                    const index_term = switch (self.terms.items[formula_id]) {
                        .binary => |binary| binary.lhs,
                        else => return error.LoopProjectionEffectUnsupported,
                    };
                    path.formula_overrides = try appendFormulaOverride(
                        self.allocator,
                        path.formula_overrides,
                        .{
                            .value_id = @intFromPtr(fixedIndexSourceValue(
                                mlir.oraOperationGetOperand(compare, 0),
                            ).ptr),
                            .formula = .{ .term = index_term },
                        },
                    );
                }
                path.safety_checkpoints = try appendLoopSafetyCheckpoint(
                    self.allocator,
                    path.safety_checkpoints,
                    .{
                        .id = @intCast(path.safety_checkpoints.len),
                        .structural_key = effect_key,
                        .after_dependency_count = @intCast(path.summary_dependencies.len),
                        .formulas = try self.allocator.dupe(obligation.FormulaRef, &.{formula}),
                    },
                );
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                    .safety_checkpoint = .{ .formulas = try self.loopFormulaExprsWithReads(
                        &.{formula},
                        path.timeline,
                    ) },
                }, effect_key);
            }
            if (draftHasTargetSite(draft, op)) {
                if (mlir.oraOperationGetNumOperands(op) == 0) return error.LoopProjectionEffectUnsupported;
                const formula = try self.loopFormulaFromValueWithOverrides(
                    mlir.oraOperationGetOperand(op, 0),
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    0,
                );
                path.target_formulas = try appendFormula(self.allocator, path.target_formulas, formula);
                path.target_keys = try appendString(self.allocator, path.target_keys, effect_key);
            }
        }
        return error.UnsupportedLoopControlFlow;
    }

    fn inventoryLoopSequence(
        self: *Collector,
        first: mlir.MlirOperation,
        initial_path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) anyerror!void {
        var path = initial_path;
        var op = first;
        while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            if (std.mem.eql(u8, name, "ora.loop_control")) {
                try self.appendInventoryLoopControlTerminal(op, path, draft, summary, terminals);
                return;
            }
            if (isRegionYield(name)) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, .{
                    .path = path,
                    .terminator_key = try self.operationStructuralKey(op, name),
                    .yielded = try self.operationOperands(op),
                });
                return;
            }
            if (std.mem.eql(u8, name, "scf.if")) {
                if (try isLoopControlRuntimeGuard(op)) {
                    try self.inventoryLoopControlRuntimeGuard(op, path, draft, summary, terminals);
                    return;
                }
                try self.inventoryIf(op, path, draft, summary, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                try self.inventorySwitch(op, path, draft, summary, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
                path = try self.bindInventoryNestedLoop(op, path);
                continue;
            }
            if (mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0) {
                return error.UnsupportedLoopControlFlow;
            }

            const effect_key = try self.operationStructuralKey(op, name);
            const effect_kind = mlir.oraOperationGetMemoryEffectKind(op);
            switch (effect_kind) {
                mlir.ORA_OPERATION_MEMORY_EFFECT_NONE => {},
                mlir.ORA_OPERATION_MEMORY_EFFECT_READ,
                mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE,
                => path.effect_keys = try appendString(self.allocator, path.effect_keys, effect_key),
                mlir.ORA_OPERATION_MEMORY_EFFECT_RECURSIVE,
                mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN,
                => return error.LoopProjectionEffectUnsupported,
                else => return error.LoopProjectionEffectUnsupported,
            }
            if (!directRootLoopEffectSupported(op, name, effect_kind)) {
                return error.LoopProjectionEffectUnsupported;
            }
            if (std.mem.eql(u8, name, "ora.sload")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1) return error.LoopProjectionEffectUnsupported;
                if (isAggregateStorageCarrierLoad(op)) continue;
                const result = mlir.oraOperationGetResult(op, 0);
                const location = try self.directStorageLocation(op, result);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_result = try self.storageReadId(op),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.sstore")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 1) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const location = try self.directStorageLocation(op, value);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "memref.load") and
                inventoryMemrefLoadTargetsPersistentRoot(op))
            {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1 or
                    mlir.oraOperationGetNumOperands(op) < 2) return error.LoopProjectionEffectUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                const location = try self.inventoryMemrefStorageLocation(
                    op,
                    0,
                    1,
                    result,
                    path.source_overrides,
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_projections = location.projections,
                    .storage_result = try self.storageReadId(op),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "memref.store") and
                inventoryMemrefStoreTargetsPersistentRoot(op))
            {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) < 3) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 0);
                const location = try self.inventoryMemrefStorageLocation(
                    op,
                    1,
                    2,
                    value,
                    path.source_overrides,
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_projections = location.projections,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.map_get")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1 or
                    mlir.oraOperationGetNumOperands(op) != 2) return error.LoopProjectionEffectUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                if (isAggregateStorageValue(result)) continue;
                const location = try self.inventoryMapStorageLocation(
                    mlir.oraOperationGetOperand(op, 0),
                    mlir.oraOperationGetOperand(op, 1),
                    result,
                    path.source_overrides,
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_read,
                    .storage_root = location.root,
                    .storage_projections = location.projections,
                    .storage_result = try self.storageReadId(op),
                    .storage_ty = location.value_ty,
                });
            } else if (std.mem.eql(u8, name, "ora.map_store")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or
                    mlir.oraOperationGetNumOperands(op) != 3) return error.LoopProjectionEffectUnsupported;
                const value = mlir.oraOperationGetOperand(op, 2);
                const location = try self.inventoryMapStorageLocation(
                    mlir.oraOperationGetOperand(op, 0),
                    mlir.oraOperationGetOperand(op, 1),
                    value,
                    path.source_overrides,
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .storage_write,
                    .storage_root = location.root,
                    .storage_projections = location.projections,
                    .storage_value_source = try self.sourceKeyForValue(value, path.source_overrides),
                    .storage_ty = location.value_ty,
                });
            }
            if (isLoopSafetyOperation(op, name)) {
                path.safety_checkpoints = try appendLoopSafetyCheckpointKey(
                    self.allocator,
                    path.safety_checkpoints,
                    .{
                        .structural_key = effect_key,
                        .after_dependency_count = @intCast(path.summary_dependency_keys.len),
                    },
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .safety_checkpoint,
                });
            }
            if (isLoopTargetOperation(op, name)) {
                path.target_keys = try appendString(self.allocator, path.target_keys, effect_key);
            }
        }
        return error.UnsupportedLoopControlFlow;
    }

    fn loopControlManifest(
        self: *Collector,
        loop_op: mlir.MlirOperation,
    ) ![]const LoopControlMarker {
        const values = (try self.stringArrayAttr(loop_op, "ora.loop_control_manifest")) orelse
            return error.LoopControlManifestMissing;
        const manifest = try self.allocator.alloc(LoopControlMarker, values.len);
        for (values, manifest, 0..) |value, *entry, index| {
            var parts = std.mem.splitScalar(u8, value, ':');
            const kind_text = parts.next() orelse return error.LoopControlMarkerMalformed;
            const source_text = parts.next() orelse return error.LoopControlMarkerMalformed;
            if (parts.next() != null or source_text.len == 0) return error.LoopControlMarkerMalformed;
            const kind: LoopControlKind = loopControlKindFromText(kind_text) orelse
                return error.LoopControlMarkerMalformed;
            const source_id = std.fmt.parseInt(u32, source_text, 10) catch
                return error.LoopControlMarkerMalformed;
            const canonical = try std.fmt.allocPrint(self.allocator, "{d}", .{source_id});
            if (!std.mem.eql(u8, canonical, source_text)) return error.LoopControlMarkerMalformed;
            for (manifest[0..index]) |previous| {
                if (previous.source_statement_id == source_id) return error.LoopControlMarkerDuplicate;
            }
            entry.* = .{
                .kind = kind,
                .source_statement_id = source_id,
                .target = loopControlTarget(kind),
                .marker_key = "",
            };
        }
        return manifest;
    }

    fn appendSemanticLoopControlTerminal(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) !void {
        if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
        const kind_text = (try self.stringAttr(op, "kind")) orelse return error.LoopControlMarkerMalformed;
        const kind: LoopControlKind = loopControlKindFromText(kind_text) orelse
            return error.LoopControlMarkerMalformed;
        const source_id = self.u32Attr(op, "source_control_id") catch
            return error.LoopControlMarkerMalformed;
        const target_id = self.u32Attr(op, "target_loop_source_id") catch
            return error.LoopControlMarkerMalformed;
        const source_statement_id = source_id orelse return error.LoopControlMarkerMalformed;
        const target_statement_id = target_id orelse return error.LoopControlMarkerMalformed;
        if (sourceStatementIndex(op) != source_statement_id) return error.LoopControlMarkerMalformed;
        if (sourceStatementIndex(draft.loop_op) != target_statement_id) {
            const current_manifest = try self.loopControlManifest(draft.loop_op);
            if (loopControlManifestContains(current_manifest, kind, source_statement_id)) {
                return error.LoopControlMarkerTargetMismatch;
            }
            return error.UnsupportedLoopControlFlow;
        }
        const operands = try self.operationOperands(op);
        try validateLoopControlStateTypes(summary.variables, operands);
        const region_yield = mlir.oraOperationGetNextInBlock(op);
        if (mlir.oraOperationIsNull(region_yield) or !isRegionYield(operationName(region_yield))) {
            return error.LoopControlMarkerMalformed;
        }
        const marker_key = try self.operationStructuralKey(op, "ora.loop_control");
        const yield_key = try self.operationStructuralKey(region_yield, operationName(region_yield));
        try terminals.append(self.allocator, .{
            .path = path,
            .terminator_key = try std.fmt.allocPrint(self.allocator, "{s}->{s}", .{ marker_key, yield_key }),
            .yielded = operands,
            .control = .{
                .kind = kind,
                .source_statement_id = source_statement_id,
                .target = loopControlTarget(kind),
                .marker_key = marker_key,
            },
        });
    }

    fn appendInventoryLoopControlTerminal(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) !void {
        if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
        const kind_view = stringAttrView(op, "kind") orelse return error.LoopControlMarkerMalformed;
        const kind: LoopControlKind = loopControlKindFromText(kind_view.slice()) orelse
            return error.LoopControlMarkerMalformed;
        const source_attr = mlir.oraOperationGetAttributeByName(op, strRef("source_control_id"));
        const target_attr = mlir.oraOperationGetAttributeByName(op, strRef("target_loop_source_id"));
        if (mlir.oraAttributeIsNull(source_attr) or mlir.oraAttributeIsNull(target_attr)) {
            return error.LoopControlMarkerMalformed;
        }
        const source_text = mlir.oraIntegerAttrGetValueString(source_attr);
        defer if (source_text.data != null) mlir.oraStringRefFree(source_text);
        const target_text = mlir.oraIntegerAttrGetValueString(target_attr);
        defer if (target_text.data != null) mlir.oraStringRefFree(target_text);
        if (source_text.data == null or target_text.data == null) return error.LoopControlMarkerMalformed;
        const source_statement_id = std.fmt.parseInt(u32, source_text.data[0..source_text.length], 10) catch
            return error.LoopControlMarkerMalformed;
        const target_statement_id = std.fmt.parseInt(u32, target_text.data[0..target_text.length], 10) catch
            return error.LoopControlMarkerMalformed;
        if (sourceStatementIndex(op) != source_statement_id) return error.LoopControlMarkerMalformed;
        if (sourceStatementIndex(draft.loop_op) != target_statement_id) {
            const current_manifest = try self.loopControlManifest(draft.loop_op);
            if (loopControlManifestContains(current_manifest, kind, source_statement_id)) {
                return error.LoopControlMarkerTargetMismatch;
            }
            return error.UnsupportedLoopControlFlow;
        }
        const operands = try self.operationOperands(op);
        try validateLoopControlStateTypes(summary.variables, operands);
        const region_yield = mlir.oraOperationGetNextInBlock(op);
        if (mlir.oraOperationIsNull(region_yield) or !isRegionYield(operationName(region_yield))) {
            return error.LoopControlMarkerMalformed;
        }
        const marker_key = try self.operationStructuralKey(op, "ora.loop_control");
        const yield_key = try self.operationStructuralKey(region_yield, operationName(region_yield));
        try terminals.append(self.allocator, .{
            .path = path,
            .terminator_key = try std.fmt.allocPrint(self.allocator, "{s}->{s}", .{ marker_key, yield_key }),
            .yielded = operands,
            .control = .{
                .kind = kind,
                .source_statement_id = source_statement_id,
                .target = loopControlTarget(kind),
                .marker_key = marker_key,
            },
        });
    }

    fn validateSemanticLoopControlManifest(
        _: *Collector,
        manifest: []const LoopControlMarker,
        terminals: []const SemanticLoopTerminal,
    ) !void {
        try validateLoopControlManifestAgainstTerminals(SemanticLoopTerminal, manifest, terminals);
    }

    fn validateInventoryLoopControlManifest(
        _: *Collector,
        manifest: []const LoopControlMarker,
        terminals: []const InventoryLoopTerminal,
    ) !void {
        try validateLoopControlManifestAgainstTerminals(InventoryLoopTerminal, manifest, terminals);
    }

    fn projectSemanticLoopControlRuntimeGuard(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) !void {
        if (mlir.oraOperationGetNumRegions(op) != 2) return error.LoopControlMarkerMalformed;
        try validateRuntimeGuardElseRegion(op);
        const then_block = singleRegionBlock(op, 0) orelse return error.LoopControlMarkerMalformed;
        var guarded_terminals: std.ArrayList(SemanticLoopTerminal) = .empty;
        defer guarded_terminals.deinit(self.allocator);
        try self.projectSemanticLoopSequence(
            mlir.oraBlockGetFirstOperation(then_block),
            path,
            draft,
            summary,
            &guarded_terminals,
        );
        for (guarded_terminals.items) |terminal| {
            if (terminal.control != null) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, terminal);
                continue;
            }
            const continued = try self.bindSemanticRegionResults(op, terminal);
            try self.projectSemanticLoopSequence(
                mlir.oraOperationGetNextInBlock(op),
                continued,
                draft,
                summary,
                terminals,
            );
        }
    }

    fn inventoryLoopControlRuntimeGuard(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) !void {
        if (mlir.oraOperationGetNumRegions(op) != 2) return error.LoopControlMarkerMalformed;
        try validateRuntimeGuardElseRegion(op);
        const then_block = singleRegionBlock(op, 0) orelse return error.LoopControlMarkerMalformed;
        var guarded_terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
        defer guarded_terminals.deinit(self.allocator);
        try self.inventoryLoopSequence(
            mlir.oraBlockGetFirstOperation(then_block),
            path,
            draft,
            summary,
            &guarded_terminals,
        );
        for (guarded_terminals.items) |terminal| {
            if (terminal.control != null) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, terminal);
                continue;
            }
            const continued = try self.bindInventoryRegionResults(op, terminal);
            try self.inventoryLoopSequence(
                mlir.oraOperationGetNextInBlock(op),
                continued,
                draft,
                summary,
                terminals,
            );
        }
    }

    fn projectSemanticIf(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) anyerror!void {
        if (mlir.oraOperationGetNumOperands(op) == 0) return error.UnsupportedLoopControlFlow;
        const condition = try self.loopFormulaFromValueWithOverrides(
            mlir.oraOperationGetOperand(op, 0),
            path.formula_overrides,
            "scf.if",
            draft.symbol,
            draft.ordinal,
            0,
        );
        const branch_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        if (branch_count == 0 or branch_count > 2) return error.UnsupportedLoopControlFlow;
        const op_key = try self.operationStructuralKey(op, "scf.if");

        for (0..branch_count) |branch_index| {
            var branch_path = path;
            branch_path.key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/{s}",
                .{ path.key, op_key, if (branch_index == 0) "then" else "else" },
            );
            branch_path.path_formulas = try appendFormula(
                self.allocator,
                branch_path.path_formulas,
                if (branch_index == 0) condition else try self.negateFormula(condition),
            );
            const branch_formula = branch_path.path_formulas[branch_path.path_formulas.len - 1];
            const branch_path_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch_path.key});
            branch_path.timeline = try appendLoopTimelineEvent(self.allocator, branch_path.timeline, .{
                .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(
                    &.{branch_formula},
                    branch_path.timeline,
                ) },
            }, branch_path_key);
            const block = singleRegionBlock(op, branch_index) orelse return error.UnsupportedLoopControlFlow;
            var branch_terminals: std.ArrayList(SemanticLoopTerminal) = .empty;
            defer branch_terminals.deinit(self.allocator);
            try self.projectSemanticLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, summary, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindSemanticRegionResults(op, branch_terminal);
                try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, summary, terminals);
            }
        }

        if (branch_count == 1) {
            if (mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
            var false_path = path;
            false_path.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/else-implicit", .{ path.key, op_key });
            false_path.path_formulas = try appendFormula(
                self.allocator,
                false_path.path_formulas,
                try self.negateFormula(condition),
            );
            const false_formula = false_path.path_formulas[false_path.path_formulas.len - 1];
            const false_path_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{false_path.key});
            false_path.timeline = try appendLoopTimelineEvent(self.allocator, false_path.timeline, .{
                .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(
                    &.{false_formula},
                    false_path.timeline,
                ) },
            }, false_path_key);
            try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), false_path, draft, summary, terminals);
        }
    }

    fn inventoryIf(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) anyerror!void {
        const branch_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        if (mlir.oraOperationGetNumOperands(op) == 0 or branch_count == 0 or branch_count > 2) {
            return error.UnsupportedLoopControlFlow;
        }
        const op_key = try self.operationStructuralKey(op, "scf.if");
        for (0..branch_count) |branch_index| {
            var branch_path = path;
            branch_path.key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/{s}",
                .{ path.key, op_key, if (branch_index == 0) "then" else "else" },
            );
            const branch_path_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch_path.key});
            branch_path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, branch_path.timeline_inventory, .{
                .structural_key = branch_path_key,
                .tag = .path_assumption,
            });
            const block = singleRegionBlock(op, branch_index) orelse return error.UnsupportedLoopControlFlow;
            var branch_terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
            defer branch_terminals.deinit(self.allocator);
            try self.inventoryLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, summary, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindInventoryRegionResults(op, branch_terminal);
                try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, summary, terminals);
            }
        }
        if (branch_count == 1) {
            if (mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
            var false_path = path;
            false_path.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/else-implicit", .{ path.key, op_key });
            false_path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, false_path.timeline_inventory, .{
                .structural_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{false_path.key}),
                .tag = .path_assumption,
            });
            try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), false_path, draft, summary, terminals);
        }
    }

    fn projectSemanticSwitch(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) anyerror!void {
        if (mlir.oraOperationGetNumOperands(op) == 0) return error.UnsupportedLoopControlFlow;
        const region_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        const case_kinds = denseI64ArrayAttr(op, "case_kinds") orelse return error.UnsupportedLoopControlFlow;
        if (mlir.oraDenseI64ArrayAttrGetNumElements(case_kinds) != region_count) {
            return error.UnsupportedLoopControlFlow;
        }
        const default_index = switchDefaultIndex(op, region_count) catch return error.UnsupportedLoopControlFlow;
        const op_key = try self.operationStructuralKey(op, operationName(op));
        const raw_by_region = try self.allocator.alloc(?obligation.FormulaRef, region_count);
        for (0..region_count) |region_index| {
            const kind = mlir.oraDenseI64ArrayAttrGetElement(case_kinds, region_index);
            raw_by_region[region_index] = if (kind == 2)
                null
            else if (kind == 0 or kind == 1)
                try self.switchCaseFormula(op, region_index, path.formula_overrides)
            else
                return error.UnsupportedLoopControlFlow;
        }

        for (0..region_count) |region_index| {
            const kind = mlir.oraDenseI64ArrayAttrGetElement(case_kinds, region_index);
            const is_default = kind == 2;
            if (is_default != (default_index != null and default_index.? == region_index)) {
                return error.UnsupportedLoopControlFlow;
            }
            var branch_path = path;
            branch_path.key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/case:{d}",
                .{ path.key, op_key, region_index },
            );
            const path_formula_start = branch_path.path_formulas.len;
            for (raw_by_region, 0..) |maybe_predicate, predicate_index| {
                const predicate = maybe_predicate orelse continue;
                if (!is_default and predicate_index >= region_index) continue;
                branch_path.path_formulas = try appendFormula(
                    self.allocator,
                    branch_path.path_formulas,
                    try self.negateFormula(predicate),
                );
            }
            if (!is_default) branch_path.path_formulas = try appendFormula(
                self.allocator,
                branch_path.path_formulas,
                raw_by_region[region_index].?,
            );
            const branch_path_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch_path.key});
            branch_path.timeline = try appendLoopTimelineEvent(self.allocator, branch_path.timeline, .{
                .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(
                    branch_path.path_formulas[path_formula_start..],
                    branch_path.timeline,
                ) },
            }, branch_path_key);
            const block = singleRegionBlock(op, region_index) orelse return error.UnsupportedLoopControlFlow;
            var branch_terminals: std.ArrayList(SemanticLoopTerminal) = .empty;
            defer branch_terminals.deinit(self.allocator);
            try self.projectSemanticLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, summary, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindSemanticRegionResults(op, branch_terminal);
                try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, summary, terminals);
            }
        }

        if (default_index == null) {
            if (mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
            var no_match = path;
            no_match.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/no-match", .{ path.key, op_key });
            const path_formula_start = no_match.path_formulas.len;
            for (raw_by_region) |maybe_predicate| {
                const predicate = maybe_predicate orelse continue;
                no_match.path_formulas = try appendFormula(
                    self.allocator,
                    no_match.path_formulas,
                    try self.negateFormula(predicate),
                );
            }
            const no_match_path_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{no_match.key});
            no_match.timeline = try appendLoopTimelineEvent(self.allocator, no_match.timeline, .{
                .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(
                    no_match.path_formulas[path_formula_start..],
                    no_match.timeline,
                ) },
            }, no_match_path_key);
            try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), no_match, draft, summary, terminals);
        }
    }

    fn inventorySwitch(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        summary: *const obligation.LoopSummaryRow,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) anyerror!void {
        if (mlir.oraOperationGetNumOperands(op) == 0) return error.UnsupportedLoopControlFlow;
        const region_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        const case_kinds = denseI64ArrayAttr(op, "case_kinds") orelse return error.UnsupportedLoopControlFlow;
        if (mlir.oraDenseI64ArrayAttrGetNumElements(case_kinds) != region_count) {
            return error.UnsupportedLoopControlFlow;
        }
        const default_index = switchDefaultIndex(op, region_count) catch return error.UnsupportedLoopControlFlow;
        const op_key = try self.operationStructuralKey(op, operationName(op));
        for (0..region_count) |region_index| {
            const kind = mlir.oraDenseI64ArrayAttrGetElement(case_kinds, region_index);
            const is_default = kind == 2;
            if (kind < 0 or kind > 2 or is_default != (default_index != null and default_index.? == region_index)) {
                return error.UnsupportedLoopControlFlow;
            }
            var branch_path = path;
            branch_path.key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/case:{d}",
                .{ path.key, op_key, region_index },
            );
            branch_path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, branch_path.timeline_inventory, .{
                .structural_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch_path.key}),
                .tag = .path_assumption,
            });
            const block = singleRegionBlock(op, region_index) orelse return error.UnsupportedLoopControlFlow;
            var branch_terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
            defer branch_terminals.deinit(self.allocator);
            try self.inventoryLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, summary, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindInventoryRegionResults(op, branch_terminal);
                try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, summary, terminals);
            }
        }
        if (default_index == null) {
            if (mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
            var no_match = path;
            no_match.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/no-match", .{ path.key, op_key });
            no_match.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, no_match.timeline_inventory, .{
                .structural_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{no_match.key}),
                .tag = .path_assumption,
            });
            try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), no_match, draft, summary, terminals);
        }
    }

    fn bindSemanticRegionResults(
        self: *Collector,
        parent: mlir.MlirOperation,
        terminal: SemanticLoopTerminal,
    ) !SemanticLoopPath {
        const result_count: usize = @intCast(mlir.oraOperationGetNumResults(parent));
        if (terminal.yielded.len != result_count) return error.UnsupportedLoopControlFlow;
        var path = terminal.path;
        for (0..result_count) |index| {
            const result = mlir.oraOperationGetResult(parent, index);
            const source = terminal.yielded[index];
            const result_key = try self.directOperationResultKey(result);
            const source_key = try self.sourceKeyForValue(source, path.source_overrides);
            path.state_source_keys = try appendString(
                self.allocator,
                path.state_source_keys,
                try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ result_key, source_key }),
            );
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .key = source_key,
            });
            path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .formula = try self.loopFormulaFromValueWithOverrides(
                    source,
                    path.formula_overrides,
                    operationName(parent),
                    null,
                    self.operationOrdinal(parent) orelse 0,
                    @intCast(index),
                ),
            });
        }
        return path;
    }

    fn bindInventoryRegionResults(
        self: *Collector,
        parent: mlir.MlirOperation,
        terminal: InventoryLoopTerminal,
    ) !InventoryLoopPath {
        const result_count: usize = @intCast(mlir.oraOperationGetNumResults(parent));
        if (terminal.yielded.len != result_count) return error.UnsupportedLoopControlFlow;
        var path = terminal.path;
        for (0..result_count) |index| {
            const result = mlir.oraOperationGetResult(parent, index);
            const result_key = try self.directOperationResultKey(result);
            const source_key = try self.sourceKeyForValue(terminal.yielded[index], path.source_overrides);
            path.state_source_keys = try appendString(
                self.allocator,
                path.state_source_keys,
                try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ result_key, source_key }),
            );
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .key = source_key,
            });
        }
        return path;
    }

    fn nestedLoopSummaryForOperation(
        self: *Collector,
        op: mlir.MlirOperation,
    ) ?obligation.LoopSummaryRow {
        for (self.loop_summary_drafts.items, self.loop_summaries.items) |candidate, summary| {
            if (candidate.loop_op.ptr == op.ptr) return summary;
        }
        return null;
    }

    fn nestedLoopHasOnlySuccessorExit(summary: obligation.LoopSummaryRow) bool {
        var successor_count: usize = 0;
        for (summary.edges) |edge| switch (edge.target orelse return false) {
            .backedge => {},
            .successor => successor_count += 1,
            .function_return, .error_return, .explicit_revert => return false,
        };
        return successor_count == 1;
    }

    fn bindSemanticNestedLoop(
        self: *Collector,
        op: mlir.MlirOperation,
        initial: SemanticLoopPath,
    ) !SemanticLoopPath {
        const inner = self.nestedLoopSummaryForOperation(op) orelse
            return error.UnsupportedLoopControlFlow;
        if (!inner.projectionSupported() or !nestedLoopHasOnlySuccessorExit(inner)) {
            return error.UnsupportedLoopControlFlow;
        }
        const result_count: usize = @intCast(mlir.oraOperationGetNumResults(op));
        const variable_start: usize = @intFromBool(inner.loop_kind == .scf_for);
        if (result_count + variable_start != inner.variables.len) {
            return error.UnsupportedLoopControlFlow;
        }

        var path = initial;
        const dependency_key = try self.operationStructuralKey(op, operationName(op));
        path.summary_dependencies = try appendLoopSummaryDependency(
            self.allocator,
            path.summary_dependencies,
            .{
                .id = @intCast(path.summary_dependencies.len),
                .structural_key = dependency_key,
                .summary_id = inner.id,
                .loop_vc_sha256 = [_]u8{0} ** 32,
                .write_footprint_complete = true,
                .target = .successor,
            },
        );
        path.summary_dependency_keys = try appendString(
            self.allocator,
            path.summary_dependency_keys,
            dependency_key,
        );
        const dependency = path.summary_dependencies[path.summary_dependencies.len - 1];
        path.timeline = try appendLoopTimelineEvent(
            self.allocator,
            path.timeline,
            .{ .summary_dependency = dependency },
            dependency_key,
        );
        for (0..result_count) |result_index| {
            const result = mlir.oraOperationGetResult(op, result_index);
            const variable = inner.variables[variable_start + result_index];
            const variable_id = variable.id orelse return error.UnsupportedLoopControlFlow;
            const result_key = try self.directOperationResultKey(result);
            const source_key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/final-variable:{d}",
                .{ dependency_key, variable.index },
            );
            path.state_source_keys = try appendString(
                self.allocator,
                path.state_source_keys,
                try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ result_key, source_key }),
            );
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .key = source_key,
            });
            const variable_term = try self.addTerm(.{ .variable = .{ .free = .{
                .id = variable_id,
                .name = variable.name orelse "nested_loop_result",
                .ty = variable.ty,
            } } });
            path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .formula = .{ .term = variable_term },
            });
        }
        return path;
    }

    fn bindInventoryNestedLoop(
        self: *Collector,
        op: mlir.MlirOperation,
        initial: InventoryLoopPath,
    ) !InventoryLoopPath {
        const inner = self.nestedLoopSummaryForOperation(op) orelse
            return error.UnsupportedLoopControlFlow;
        if (!inner.projectionSupported() or !nestedLoopHasOnlySuccessorExit(inner)) {
            return error.UnsupportedLoopControlFlow;
        }
        const result_count: usize = @intCast(mlir.oraOperationGetNumResults(op));
        const variable_start: usize = @intFromBool(inner.loop_kind == .scf_for);
        if (result_count + variable_start != inner.variables.len) {
            return error.UnsupportedLoopControlFlow;
        }

        var path = initial;
        const dependency_key = try self.operationStructuralKey(op, operationName(op));
        const dependency: obligation.LoopSummaryDependency = .{
            .id = @intCast(path.summary_dependency_keys.len),
            .structural_key = dependency_key,
            .summary_id = inner.id,
            .loop_vc_sha256 = [_]u8{0} ** 32,
            .write_footprint_complete = true,
            .target = .successor,
        };
        path.summary_dependency_keys = try appendString(
            self.allocator,
            path.summary_dependency_keys,
            dependency_key,
        );
        path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
            .structural_key = dependency_key,
            .tag = .summary_dependency,
            .summary_dependency = dependency,
        });
        for (0..result_count) |result_index| {
            const result = mlir.oraOperationGetResult(op, result_index);
            const variable = inner.variables[variable_start + result_index];
            const result_key = try self.directOperationResultKey(result);
            const source_key = try std.fmt.allocPrint(
                self.allocator,
                "{s}/final-variable:{d}",
                .{ dependency_key, variable.index },
            );
            path.state_source_keys = try appendString(
                self.allocator,
                path.state_source_keys,
                try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ result_key, source_key }),
            );
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                .value_id = @intFromPtr(result.ptr),
                .key = source_key,
            });
        }
        return path;
    }

    fn projectedTerminalEdge(
        self: *Collector,
        summary: *const obligation.LoopSummaryRow,
        draft: LoopSummaryDraft,
        terminal: SemanticLoopTerminal,
    ) !ProjectedLoopEdge {
        const target = if (terminal.control) |control| control.target else obligation.LoopControlTarget.backedge;
        const is_return_target = target == .function_return or target == .error_return;
        if (!is_return_target and terminal.path.target_keys.len != 0) {
            return error.LoopTargetFormulaTargetMismatch;
        }
        const complete_control_state = terminal.control != null;
        const implicit_induction_count: usize = @intFromBool(
            draft.loop_kind == .scf_for and !complete_control_state,
        );
        if (terminal.yielded.len + implicit_induction_count != summary.variables.len) {
            return error.UnsupportedLoopControlFlow;
        }
        const assignments = try self.allocator.alloc(obligation.LoopStepAssignment, summary.variables.len);
        const advances_for_induction = draft.loop_kind == .scf_for and target == .backedge;
        if (advances_for_induction) {
            if (mlir.oraOperationGetNumOperands(draft.loop_op) < 3) return error.UnsupportedLoopControlFlow;
            const induction = if (complete_control_state)
                terminal.yielded[0]
            else
                loopVariableValue(draft.loop_op, draft.loop_kind, 0);
            const step = mlir.oraOperationGetOperand(draft.loop_op, 2);
            const induction_ty = summary.variables[0].ty;
            const signed_override = if (typeRefIntegerInfo(induction_ty)) |info| info.signed else null;
            const induction_term = (try self.termFromValueWithOverrides(
                induction,
                signed_override,
                terminal.path.formula_overrides,
            )) orelse return error.UnsupportedLoopControlFlow;
            const step_term = (try self.forInductionStepTerm(
                step,
                induction_ty,
                signed_override,
                terminal.path.formula_overrides,
            )) orelse return error.UnsupportedLoopControlFlow;
            const induction_formula: obligation.FormulaRef = .{ .term = try self.addTerm(.{ .binary = .{
                .op = .add,
                .lhs = induction_term,
                .rhs = step_term,
                .ty = induction_ty,
            } }) };
            assignments[0] = .{
                .variable_index = 0,
                .target = summary.variables[0].id,
                .source_key = if (complete_control_state)
                    try std.fmt.allocPrint(
                        self.allocator,
                        "for/control-induction-step:{s}+{s}",
                        .{
                            try self.sourceKeyForValue(induction, terminal.path.source_overrides),
                            try self.sourceKeyForValue(step, terminal.path.source_overrides),
                        },
                    )
                else
                    try std.fmt.allocPrint(
                        self.allocator,
                        "for/induction-step:{s}",
                        .{try self.sourceKeyForValue(step, terminal.path.source_overrides)},
                    ),
                .value = induction_formula,
                .read_results = try self.readResultsForFormula(induction_formula, terminal.path.timeline),
            };
        }
        const yielded_start: usize = @intFromBool(advances_for_induction and complete_control_state);
        const variable_start: usize = @intFromBool(advances_for_induction);
        for (terminal.yielded[yielded_start..], variable_start..) |value, variable_index| {
            const variable = summary.variables[variable_index];
            const signed_override = if (typeRefIntegerInfo(variable.ty)) |info| info.signed else null;
            const value_formula = try self.loopFormulaFromValueWithIntegerSignednessAndOverrides(
                value,
                signed_override,
                terminal.path.formula_overrides,
                operationName(mlir.oraBlockGetTerminator(draft.body)),
                draft.symbol,
                draft.ordinal,
                @intCast(variable_index),
            );
            assignments[variable_index] = .{
                .variable_index = @intCast(variable_index),
                .target = variable.id,
                .source_key = try self.sourceKeyForValue(value, terminal.path.source_overrides),
                .value = value_formula,
                .read_results = try self.readResultsForFormula(value_formula, terminal.path.timeline),
            };
        }
        const structural_key = if (terminal.control) |control|
            try std.fmt.allocPrint(
                self.allocator,
                "{s}/control:{s}:{d}",
                .{
                    terminal.path.key,
                    loopControlKindText(control.kind),
                    control.source_statement_id,
                },
            )
        else if (std.mem.eql(u8, terminal.path.key, "body"))
            "body/straight-line-yield"
        else
            try std.fmt.allocPrint(self.allocator, "{s}/yield", .{terminal.path.key});
        return .{
            .edge = .{
                .id = 0,
                .structural_key = structural_key,
                .timeline = terminal.path.timeline,
                .next_state = assignments,
                .target = target,
                .target_formulas = if (is_return_target) terminal.path.target_formulas else &.{},
            },
            .terminator_key = terminal.terminator_key,
            .safety_checkpoints = try safetyCheckpointKeysFromRows(
                self.allocator,
                terminal.path.safety_checkpoints,
            ),
            .target_keys = terminal.path.target_keys,
            .summary_dependency_keys = terminal.path.summary_dependency_keys,
            .effect_keys = terminal.path.effect_keys,
            .state_source_keys = terminal.path.state_source_keys,
            .timeline_inventory = terminal.path.timeline_inventory,
        };
    }

    fn forInductionStepTerm(
        self: *Collector,
        step: mlir.MlirValue,
        induction_ty: obligation.TypeRef,
        signed_override: ?bool,
        overrides: []const LoopFormulaOverride,
    ) !?obligation.TermId {
        if (integerTypeWidth(mlir.oraValueGetType(step)) != null) {
            return self.termFromValueWithOverrides(step, signed_override, overrides);
        }
        if (!mlir.oraValueIsAOpResult(step)) return null;
        const owner = mlir.oraOpResultGetOwner(step);
        if (mlir.oraOperationIsNull(owner) or
            !std.mem.eql(u8, operationName(owner), "arith.constant")) return null;
        const value_attr = mlir.oraOperationGetAttributeByName(owner, strRef("value"));
        if (mlir.oraAttributeIsNull(value_attr)) return null;
        const text = mlir.oraIntegerAttrGetValueString(value_attr);
        defer if (text.data != null) mlir.oraStringRefFree(text);
        if (text.data == null) return null;
        const info = typeRefIntegerInfo(induction_ty) orelse return null;
        return try self.addTerm(.{ .int_lit = .{
            .value = try normalizeIntegerLiteralText(
                self.allocator,
                text.data[0..text.length],
                info.width,
            ),
            .ty = induction_ty,
        } });
    }

    fn inventoryTerminalEdge(
        self: *Collector,
        summary: *const obligation.LoopSummaryRow,
        draft: LoopSummaryDraft,
        terminal: InventoryLoopTerminal,
    ) !InventoryLoopEdge {
        const target = if (terminal.control) |control| control.target else obligation.LoopControlTarget.backedge;
        const is_return_target = target == .function_return or target == .error_return;
        if (!is_return_target and terminal.path.target_keys.len != 0) {
            return error.LoopTargetFormulaTargetMismatch;
        }
        const complete_control_state = terminal.control != null;
        const implicit_induction_count: usize = @intFromBool(
            draft.loop_kind == .scf_for and !complete_control_state,
        );
        if (terminal.yielded.len + implicit_induction_count != summary.variables.len) {
            return error.UnsupportedLoopControlFlow;
        }
        const source_keys = try self.allocator.alloc([]const u8, summary.variables.len);
        const advances_for_induction = draft.loop_kind == .scf_for and target == .backedge;
        if (advances_for_induction) {
            if (mlir.oraOperationGetNumOperands(draft.loop_op) < 3) return error.UnsupportedLoopControlFlow;
            const step = mlir.oraOperationGetOperand(draft.loop_op, 2);
            source_keys[0] = if (complete_control_state)
                try std.fmt.allocPrint(
                    self.allocator,
                    "for/control-induction-step:{s}+{s}",
                    .{
                        try self.sourceKeyForValue(terminal.yielded[0], terminal.path.source_overrides),
                        try self.sourceKeyForValue(step, terminal.path.source_overrides),
                    },
                )
            else
                try std.fmt.allocPrint(
                    self.allocator,
                    "for/induction-step:{s}",
                    .{try self.sourceKeyForValue(step, terminal.path.source_overrides)},
                );
        }
        const yielded_start: usize = @intFromBool(advances_for_induction and complete_control_state);
        const variable_start: usize = @intFromBool(advances_for_induction);
        for (terminal.yielded[yielded_start..], variable_start..) |value, variable_index| {
            source_keys[variable_index] = try self.sourceKeyForValue(
                value,
                terminal.path.source_overrides,
            );
        }
        const structural_key = if (terminal.control) |control|
            try std.fmt.allocPrint(
                self.allocator,
                "{s}/control:{s}:{d}",
                .{
                    terminal.path.key,
                    loopControlKindText(control.kind),
                    control.source_statement_id,
                },
            )
        else if (std.mem.eql(u8, terminal.path.key, "body"))
            "body/straight-line-yield"
        else
            try std.fmt.allocPrint(self.allocator, "{s}/yield", .{terminal.path.key});
        return .{
            .structural_key = structural_key,
            .target = target,
            .terminator_key = terminal.terminator_key,
            .binding_source_keys = source_keys,
            .safety_checkpoints = terminal.path.safety_checkpoints,
            .target_keys = terminal.path.target_keys,
            .summary_dependency_keys = terminal.path.summary_dependency_keys,
            .effect_keys = terminal.path.effect_keys,
            .state_source_keys = terminal.path.state_source_keys,
            .timeline_inventory = terminal.path.timeline_inventory,
        };
    }

    fn sourceKeyForValue(
        self: *Collector,
        value: mlir.MlirValue,
        overrides: []const LoopValueSource,
    ) ![]const u8 {
        if (mlir.oraValueIsNull(value)) return error.UnsupportedLoopControlFlow;
        const value_id = @intFromPtr(value.ptr);
        var override_index = overrides.len;
        while (override_index > 0) {
            override_index -= 1;
            if (overrides[override_index].value_id == value_id) {
                return try std.fmt.allocPrint(
                    self.allocator,
                    "{s}<-{s}",
                    .{ try self.directOperationResultKey(value), overrides[override_index].key },
                );
            }
        }
        if (self.loopVariableForValue(value)) |variable| {
            return try std.fmt.allocPrint(self.allocator, "loop-state:{d}", .{variable.index});
        }
        if (mlir.mlirValueIsABlockArgument(value)) {
            const index = self.functionEntryBlockArgumentNumber(value) orelse return error.UnsupportedLoopControlFlow;
            return try std.fmt.allocPrint(self.allocator, "function-param:{d}", .{index});
        }
        if (!mlir.oraValueIsAOpResult(value)) return error.UnsupportedLoopControlFlow;
        const owner = mlir.oraOpResultGetOwner(value);
        const ordinal = self.operationOrdinal(owner) orelse return error.UnsupportedLoopControlFlow;
        const result_index = operationResultIndex(owner, value) orelse return error.UnsupportedLoopControlFlow;
        const direct = try std.fmt.allocPrint(self.allocator, "module-op:{d}/result:{d}", .{ ordinal, result_index });
        return (try self.sourceDependencyKey(value, overrides, self.operation_ordinals.count() + 1)) orelse direct;
    }

    fn sourceDependencyKey(
        self: *Collector,
        value: mlir.MlirValue,
        overrides: []const LoopValueSource,
        fuel: usize,
    ) !?[]const u8 {
        if (fuel == 0 or mlir.oraValueIsNull(value)) return error.UnsupportedLoopControlFlow;
        const value_id = @intFromPtr(value.ptr);
        var override_index = overrides.len;
        while (override_index > 0) {
            override_index -= 1;
            if (overrides[override_index].value_id != value_id) continue;
            return try std.fmt.allocPrint(
                self.allocator,
                "{s}<-{s}",
                .{ try self.directOperationResultKey(value), overrides[override_index].key },
            );
        }
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        var dependencies: ?[]const u8 = null;
        const operand_count: usize = @intCast(mlir.oraOperationGetNumOperands(owner));
        for (0..operand_count) |index| {
            const dependency = (try self.sourceDependencyKey(
                mlir.oraOperationGetOperand(owner, index),
                overrides,
                fuel - 1,
            )) orelse continue;
            dependencies = if (dependencies) |existing|
                try std.fmt.allocPrint(self.allocator, "{s},{s}", .{ existing, dependency })
            else
                dependency;
        }
        const joined = dependencies orelse return null;
        return try std.fmt.allocPrint(
            self.allocator,
            "{s}({s})",
            .{ try self.directOperationResultKey(value), joined },
        );
    }

    fn directOperationResultKey(self: *Collector, value: mlir.MlirValue) ![]const u8 {
        if (!mlir.oraValueIsAOpResult(value)) return error.UnsupportedLoopControlFlow;
        const owner = mlir.oraOpResultGetOwner(value);
        const ordinal = self.operationOrdinal(owner) orelse return error.UnsupportedLoopControlFlow;
        const result_index = operationResultIndex(owner, value) orelse return error.UnsupportedLoopControlFlow;
        return try std.fmt.allocPrint(self.allocator, "module-op:{d}/result:{d}", .{ ordinal, result_index });
    }

    fn operationStructuralKey(
        self: *const Collector,
        op: mlir.MlirOperation,
        name: []const u8,
    ) ![]const u8 {
        const ordinal = self.operationOrdinal(op) orelse return error.UnsupportedLoopControlFlow;
        return try std.fmt.allocPrint(self.allocator, "op:{d}/{s}", .{ ordinal, name });
    }

    fn storageReadId(self: *const Collector, op: mlir.MlirOperation) !obligation.FreeVarId {
        return .{
            .file_id = storage_read_file_id,
            .pattern_id = self.operationOrdinal(op) orelse return error.UnsupportedLoopControlFlow,
        };
    }

    fn directStorageLocation(
        self: *Collector,
        op: mlir.MlirOperation,
        value: mlir.MlirValue,
    ) !obligation.LoopStorageLocationExpr {
        const root = (try self.stringAttr(op, "global")) orelse
            return error.LoopProjectionEffectUnsupported;
        const ty = self.storage_global_types.get(root) orelse obligation.TypeRef{ .compiler_type_id = 0 };
        const width = integerTypeWidth(mlir.oraValueGetType(value)) orelse
            return error.LoopProjectionEffectUnsupported;
        if (typeRefIntegerInfo(ty)) |info| {
            if (info.width != width) return error.LoopProjectionEffectUnsupported;
        }
        return .{ .root = root, .value_ty = ty };
    }

    fn projectedMemrefStorageLocation(
        self: *Collector,
        op: mlir.MlirOperation,
        memref_operand_index: usize,
        first_index_operand: usize,
        value: mlir.MlirValue,
        formula_overrides: []const LoopFormulaOverride,
        source_overrides: []const LoopValueSource,
        timeline: []const obligation.LoopTimelineEvent,
        draft: LoopSummaryDraft,
    ) !ProjectedLoopStorageLocation {
        if (mlir.oraOperationGetNumOperands(op) <= first_index_operand) {
            return error.LoopProjectionEffectUnsupported;
        }
        const root = try persistentStorageRootForCarrier(
            mlir.oraOperationGetOperand(op, memref_operand_index),
        );
        const value_ty = try self.exactLoopIntegerTypeForValue(value);
        const projection_count: usize = @intCast(mlir.oraOperationGetNumOperands(op) - first_index_operand);
        const projections = try self.allocator.alloc(obligation.LoopStorageProjection, projection_count);
        const inventory = try self.allocator.alloc(LoopStorageProjectionInventory, projection_count);
        for (0..projection_count) |index| {
            const raw_index = mlir.oraOperationGetOperand(op, first_index_operand + index);
            const source = fixedIndexSourceValue(raw_index);
            const extent = try fixedIndexExtent(op, memref_operand_index, projection_count, index);
            const formula = try self.loopFormulaFromValueWithOverrides(
                source,
                formula_overrides,
                operationName(op),
                draft.symbol,
                draft.ordinal,
                @intCast(first_index_operand + index),
            );
            const ty = try self.exactLoopIntegerTypeForValue(source);
            projections[index] = .{ .fixed_index = .{
                .index = .{
                    .value = .{
                        .formula = formula,
                        .read_results = try self.readResultsForFormula(formula, timeline),
                    },
                    .ty = ty,
                },
                .extent = extent,
            } };
            inventory[index] = .{
                .tag = .fixed_index,
                .source_key = try self.sourceKeyForValue(source, source_overrides),
                .ty = ty,
                .extent = extent,
            };
        }
        return .{
            .location = .{ .root = root, .projections = projections, .value_ty = value_ty },
            .inventory = inventory,
        };
    }

    fn projectedMapStorageLocation(
        self: *Collector,
        op: mlir.MlirOperation,
        map_value: mlir.MlirValue,
        final_key: mlir.MlirValue,
        value: mlir.MlirValue,
        formula_overrides: []const LoopFormulaOverride,
        source_overrides: []const LoopValueSource,
        timeline: []const obligation.LoopTimelineEvent,
        draft: LoopSummaryDraft,
    ) !ProjectedLoopStorageLocation {
        const inputs = try self.mappingProjectionInputs(map_value, final_key);
        const projections = try self.allocator.alloc(obligation.LoopStorageProjection, inputs.keys.len);
        const inventory = try self.allocator.alloc(LoopStorageProjectionInventory, inputs.keys.len);
        for (inputs.keys, 0..) |key, index| {
            const formula = try self.loopFormulaFromValueWithOverrides(
                key,
                formula_overrides,
                operationName(op),
                draft.symbol,
                draft.ordinal,
                @intCast(index + 1),
            );
            const ty = try self.exactLoopIntegerTypeForValue(key);
            projections[index] = .{ .mapping_key = .{
                .value = .{
                    .formula = formula,
                    .read_results = try self.readResultsForFormula(formula, timeline),
                },
                .ty = ty,
            } };
            inventory[index] = .{
                .tag = .mapping_key,
                .source_key = try self.sourceKeyForValue(key, source_overrides),
                .ty = ty,
            };
        }
        return .{
            .location = .{
                .root = inputs.root,
                .projections = projections,
                .value_ty = try self.exactLoopIntegerTypeForValue(value),
            },
            .inventory = inventory,
        };
    }

    fn inventoryMemrefStorageLocation(
        self: *Collector,
        op: mlir.MlirOperation,
        memref_operand_index: usize,
        first_index_operand: usize,
        value: mlir.MlirValue,
        source_overrides: []const LoopValueSource,
    ) !InventoryLoopStorageLocation {
        if (mlir.oraOperationGetNumOperands(op) <= first_index_operand) {
            return error.LoopProjectionEffectUnsupported;
        }
        const root = try persistentStorageRootForCarrier(
            mlir.oraOperationGetOperand(op, memref_operand_index),
        );
        const projection_count: usize = @intCast(mlir.oraOperationGetNumOperands(op) - first_index_operand);
        const projections = try self.allocator.alloc(LoopStorageProjectionInventory, projection_count);
        for (0..projection_count) |index| {
            const source = fixedIndexSourceValue(
                mlir.oraOperationGetOperand(op, first_index_operand + index),
            );
            projections[index] = .{
                .tag = .fixed_index,
                .source_key = try self.sourceKeyForValue(source, source_overrides),
                .ty = try self.exactLoopIntegerTypeForValue(source),
                .extent = try fixedIndexExtent(op, memref_operand_index, projection_count, index),
            };
        }
        return .{
            .root = root,
            .projections = projections,
            .value_ty = try self.exactLoopIntegerTypeForValue(value),
        };
    }

    fn inventoryMapStorageLocation(
        self: *Collector,
        map_value: mlir.MlirValue,
        final_key: mlir.MlirValue,
        value: mlir.MlirValue,
        source_overrides: []const LoopValueSource,
    ) !InventoryLoopStorageLocation {
        const inputs = try self.mappingProjectionInputs(map_value, final_key);
        const projections = try self.allocator.alloc(LoopStorageProjectionInventory, inputs.keys.len);
        for (inputs.keys, 0..) |key, index| projections[index] = .{
            .tag = .mapping_key,
            .source_key = try self.sourceKeyForValue(key, source_overrides),
            .ty = try self.exactLoopIntegerTypeForValue(key),
        };
        return .{
            .root = inputs.root,
            .projections = projections,
            .value_ty = try self.exactLoopIntegerTypeForValue(value),
        };
    }

    const MappingProjectionInputs = struct {
        root: []const u8,
        keys: []const mlir.MlirValue,
    };

    fn mappingProjectionInputs(
        self: *Collector,
        map_value: mlir.MlirValue,
        final_key: mlir.MlirValue,
    ) !MappingProjectionInputs {
        var reversed: std.ArrayList(mlir.MlirValue) = .empty;
        defer reversed.deinit(self.allocator);
        try reversed.append(self.allocator, final_key);

        var current = map_value;
        var depth: u32 = 0;
        while (depth <= 32) : (depth += 1) {
            const owner = storageMapSourceOwner(current) orelse
                return error.LoopProjectionEffectUnsupported;
            const name = operationName(owner);
            if (std.mem.eql(u8, name, "ora.sload")) {
                const root = (try self.stringAttr(owner, "global")) orelse
                    return error.LoopProjectionEffectUnsupported;
                const keys = try self.allocator.alloc(mlir.MlirValue, reversed.items.len);
                for (reversed.items, 0..) |key, index| keys[keys.len - 1 - index] = key;
                return .{ .root = root, .keys = keys };
            }
            if (!std.mem.eql(u8, name, "ora.map_get") or
                mlir.oraOperationGetNumOperands(owner) != 2)
            {
                return error.LoopProjectionEffectUnsupported;
            }
            try reversed.append(self.allocator, mlir.oraOperationGetOperand(owner, 1));
            current = mlir.oraOperationGetOperand(owner, 0);
        }
        return error.LoopProjectionEffectUnsupported;
    }

    fn exactLoopIntegerTypeForValue(
        self: *Collector,
        value: mlir.MlirValue,
    ) !obligation.TypeRef {
        var signed_override: ?bool = null;
        if (mlir.oraValueIsAOpResult(value)) {
            const owner = mlir.oraOpResultGetOwner(value);
            if (!mlir.oraOperationIsNull(owner)) {
                signed_override = try self.boolAttr(owner, "ora.integer_signed");
            }
        }
        const ty = try self.canonicalTypeRefForValue(value, signed_override);
        const info = typeRefIntegerInfo(ty) orelse return error.LoopProjectionEffectUnsupported;
        if (integerTypeWidth(mlir.oraValueGetType(value)) != info.width) {
            return error.LoopProjectionEffectUnsupported;
        }
        return ty;
    }

    fn readResultsForFormula(
        self: *Collector,
        formula: obligation.FormulaRef,
        timeline: []const obligation.LoopTimelineEvent,
    ) ![]const obligation.FreeVarId {
        var results: std.ArrayList(obligation.FreeVarId) = .empty;
        defer results.deinit(self.allocator);
        const set: obligation.ObligationSet = .{ .terms = self.terms.items };
        for (timeline) |event| switch (event.payload) {
            .storage_read => |read| if (set.formulaReferencesFreeVar(formula, read.result)) {
                try results.append(self.allocator, read.result);
            },
            else => {},
        };
        return results.toOwnedSlice(self.allocator);
    }

    fn loopFormulaExprsWithReads(
        self: *Collector,
        formulas: []const obligation.FormulaRef,
        timeline: []const obligation.LoopTimelineEvent,
    ) ![]const obligation.LoopFormulaExpr {
        const expressions = try self.allocator.alloc(obligation.LoopFormulaExpr, formulas.len);
        for (formulas, expressions) |formula, *expression| expression.* = .{
            .formula = formula,
            .read_results = try self.readResultsForFormula(formula, timeline),
        };
        return expressions;
    }

    fn operationOrdinal(self: *const Collector, op: mlir.MlirOperation) ?u32 {
        if (mlir.oraOperationIsNull(op)) return null;
        return self.operation_ordinals.get(@intFromPtr(op.ptr));
    }

    fn operationOperands(self: *Collector, op: mlir.MlirOperation) ![]const mlir.MlirValue {
        const count: usize = @intCast(mlir.oraOperationGetNumOperands(op));
        const operands = try self.allocator.alloc(mlir.MlirValue, count);
        for (operands, 0..) |*operand, index| operand.* = mlir.oraOperationGetOperand(op, index);
        return operands;
    }

    fn loopFormulaFromValueWithOverrides(
        self: *Collector,
        value: mlir.MlirValue,
        overrides: []const LoopFormulaOverride,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        index: u32,
    ) !obligation.FormulaRef {
        return self.loopFormulaFromValueWithIntegerSignednessAndOverrides(
            value,
            null,
            overrides,
            op_name,
            symbol,
            ordinal,
            index,
        );
    }

    fn loopFormulaFromValueWithIntegerSignednessAndOverrides(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
        overrides: []const LoopFormulaOverride,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        index: u32,
    ) !obligation.FormulaRef {
        const previous = self.active_loop_formula_overrides;
        self.active_loop_formula_overrides = overrides;
        defer self.active_loop_formula_overrides = previous;
        return self.loopFormulaFromValueWithIntegerSignedness(
            value,
            signed_override,
            op_name,
            symbol,
            ordinal,
            index,
        );
    }

    fn termFromValueWithOverrides(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
        overrides: []const LoopFormulaOverride,
    ) !?obligation.TermId {
        const previous = self.active_loop_formula_overrides;
        self.active_loop_formula_overrides = overrides;
        defer self.active_loop_formula_overrides = previous;
        return self.termFromValueWithIntegerSignedness(value, signed_override);
    }

    fn negateFormula(self: *Collector, formula: obligation.FormulaRef) !obligation.FormulaRef {
        return switch (formula) {
            .term => |term| .{ .term = try self.addTerm(.{ .unary = .{ .op = .not, .operand = term } }) },
            .origin_value => formula,
        };
    }

    fn switchCaseFormula(
        self: *Collector,
        op: mlir.MlirOperation,
        case_index: usize,
        overrides: []const LoopFormulaOverride,
    ) !obligation.FormulaRef {
        const kinds = denseI64ArrayAttr(op, "case_kinds") orelse return error.UnsupportedLoopControlFlow;
        const kind = mlir.oraDenseI64ArrayAttrGetElement(kinds, case_index);
        if (kind == 2) return error.UnsupportedLoopControlFlow;
        const scrutinee_value = mlir.oraOperationGetOperand(op, 0);
        const ty = try self.typeRefFromValue(scrutinee_value);
        const signed_override = if (typeRefIntegerInfo(ty)) |info| info.signed else null;
        const scrutinee = (try self.termFromValueWithOverrides(scrutinee_value, signed_override, overrides)) orelse
            return error.UnsupportedLoopControlFlow;
        if (kind == 0) {
            const values = denseI64ArrayAttr(op, "case_values") orelse return error.UnsupportedLoopControlFlow;
            if (mlir.oraDenseI64ArrayAttrGetNumElements(values) <= case_index) return error.UnsupportedLoopControlFlow;
            const literal = try self.switchIntegerLiteral(mlir.oraDenseI64ArrayAttrGetElement(values, case_index), ty);
            return .{ .term = try self.addTerm(.{ .binary = .{
                .op = .eq,
                .lhs = scrutinee,
                .rhs = literal,
                .ty = ty,
            } }) };
        }
        if (kind == 1) {
            const starts = denseI64ArrayAttr(op, "range_starts") orelse return error.UnsupportedLoopControlFlow;
            const ends = denseI64ArrayAttr(op, "range_ends") orelse return error.UnsupportedLoopControlFlow;
            if (mlir.oraDenseI64ArrayAttrGetNumElements(starts) <= case_index or
                mlir.oraDenseI64ArrayAttrGetNumElements(ends) <= case_index) return error.UnsupportedLoopControlFlow;
            const start = try self.switchIntegerLiteral(mlir.oraDenseI64ArrayAttrGetElement(starts, case_index), ty);
            const end = try self.switchIntegerLiteral(mlir.oraDenseI64ArrayAttrGetElement(ends, case_index), ty);
            const info = typeRefIntegerInfo(ty) orelse return error.UnsupportedLoopControlFlow;
            const ge = try self.addTerm(.{ .binary = .{
                .op = if (info.signed) .sge else .ge,
                .lhs = scrutinee,
                .rhs = start,
                .ty = ty,
            } });
            const le = try self.addTerm(.{ .binary = .{
                .op = if (info.signed) .sle else .le,
                .lhs = scrutinee,
                .rhs = end,
                .ty = ty,
            } });
            return .{ .term = try self.addTerm(.{ .binary = .{ .op = .and_, .lhs = ge, .rhs = le, .ty = .{ .spelling = "bool" } } }) };
        }
        return error.UnsupportedLoopControlFlow;
    }

    fn switchIntegerLiteral(
        self: *Collector,
        value: i64,
        ty: obligation.TypeRef,
    ) !obligation.TermId {
        const info = typeRefIntegerInfo(ty) orelse return error.UnsupportedLoopControlFlow;
        const text = try std.fmt.allocPrint(self.allocator, "{d}", .{value});
        return try self.addTerm(.{ .int_lit = .{
            .value = try normalizeIntegerLiteralText(self.allocator, text, info.width),
            .ty = ty,
        } });
    }

    fn addLoopInductionQuery(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
    ) !void {
        if (summary.owner != .statement) return error.InvalidLoopSummaryOwner;
        const owner: obligation.Owner = .{ .function = .{
            .name = summary.owner.statement.function_name,
        } };
        const assumption_ids = try self.assumptionIdsForLoopOwner(owner);

        const base_query_id = self.nextId();
        try self.queries.append(self.allocator, .{
            .id = base_query_id,
            .owner = owner,
            .source = summary.source,
            .phase = .report,
            .origin = summary.origin,
            .artifact_policy = .diagnostic_only,
            .backend = .z3,
            .kind = .loop_base,
            .logical_role = .invariant,
            .assumption_ids = assumption_ids,
            .loop_summary_id = summary.id,
        });
        summary.query_ids.base = try appendId(self.allocator, summary.query_ids.base, base_query_id);

        for (summary.edges) |edge| {
            const safety_query_id = self.nextId();
            try self.queries.append(self.allocator, .{
                .id = safety_query_id,
                .owner = owner,
                .source = summary.source,
                .phase = .report,
                .origin = summary.origin,
                .artifact_policy = .diagnostic_only,
                .backend = .z3,
                .kind = .loop_body_safety,
                .logical_role = .invariant,
                .assumption_ids = assumption_ids,
                .loop_summary_id = summary.id,
                .loop_edge_id = edge.id,
            });
            summary.query_ids.body_safety = try appendId(
                self.allocator,
                summary.query_ids.body_safety,
                safety_query_id,
            );

            const target_kind: obligation.VerificationQueryKind = switch (edge.target orelse
                return error.InvalidLoopEdgeTarget) {
                .backedge => .loop_invariant_step,
                else => .loop_invariant_post,
            };
            const target_query_id = self.nextId();
            try self.queries.append(self.allocator, .{
                .id = target_query_id,
                .owner = owner,
                .source = summary.source,
                .phase = .report,
                .origin = summary.origin,
                .artifact_policy = .diagnostic_only,
                .backend = .z3,
                .kind = target_kind,
                .logical_role = if (target_kind == .loop_invariant_step) .invariant else .ensures,
                .assumption_ids = assumption_ids,
                .loop_summary_id = summary.id,
                .loop_edge_id = edge.id,
            });
            if (target_kind == .loop_invariant_step) {
                summary.query_ids.step = try appendId(self.allocator, summary.query_ids.step, target_query_id);
            } else {
                summary.query_ids.post = try appendId(self.allocator, summary.query_ids.post, target_query_id);
            }
        }

        const query_id = self.nextId();
        try self.queries.append(self.allocator, .{
            .id = query_id,
            .owner = owner,
            .source = summary.source,
            .phase = .report,
            .origin = summary.origin,
            // Arming occurs only after the whole function has been collected,
            // including any postconditions. Until semantic classification
            // succeeds this row carries no artifact authority.
            .artifact_policy = .diagnostic_only,
            .kind = .loop_induction,
            .logical_role = .invariant,
            .assumption_ids = assumption_ids,
            .loop_summary_id = summary.id,
        });
        summary.query_ids.induction = try appendId(
            self.allocator,
            summary.query_ids.induction,
            query_id,
        );
    }

    fn stepAssignmentsRequireSafety(
        self: *const Collector,
        assignments: []const obligation.LoopStepAssignment,
    ) bool {
        for (assignments) |assignment| {
            if (self.termRequiresArithmeticSafety(assignment.value, self.terms.items.len + 1)) return true;
        }
        return false;
    }

    fn termRequiresArithmeticSafety(
        self: *const Collector,
        formula: obligation.FormulaRef,
        fuel: usize,
    ) bool {
        if (formula != .term or formula.term >= self.terms.items.len or fuel == 0) return false;
        return switch (self.terms.items[formula.term]) {
            .binary => |binary| self.arithmeticSafetyTermRecorded(formula.term) or
                self.termRequiresArithmeticSafety(.{ .term = binary.lhs }, fuel - 1) or
                self.termRequiresArithmeticSafety(.{ .term = binary.rhs }, fuel - 1),
            .unary => |unary| self.termRequiresArithmeticSafety(.{ .term = unary.operand }, fuel - 1),
            .old => |operand| self.termRequiresArithmeticSafety(.{ .term = operand }, fuel - 1),
            .refinement_predicate => |predicate| blk: {
                if (self.termRequiresArithmeticSafety(.{ .term = predicate.value }, fuel - 1)) break :blk true;
                for (predicate.args) |arg| {
                    if (self.termRequiresArithmeticSafety(.{ .term = arg }, fuel - 1)) break :blk true;
                }
                break :blk false;
            },
            .quantified => |quantified| self.termRequiresArithmeticSafety(.{ .term = quantified.body }, fuel - 1),
            else => false,
        };
    }

    fn arithmeticSafetyTermRecorded(self: *const Collector, term_id: obligation.TermId) bool {
        return self.arithmetic_safety_terms.contains(term_id);
    }

    fn recordArithmeticSafetyTerm(self: *Collector, term_id: obligation.TermId) !void {
        try self.arithmetic_safety_terms.put(self.allocator, term_id, {});
    }

    fn collectLoopVariables(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
    ) ![]const obligation.LoopVariable {
        const count = loopVariableCount(op, loop_kind);
        if (count == 0) return &.{};
        const variables = try self.allocator.alloc(obligation.LoopVariable, count);
        for (variables, 0..) |*variable, index| {
            const value = loopVariableValue(op, loop_kind, @intCast(index));
            const pattern_id = self.next_loop_variable_pattern_id;
            self.next_loop_variable_pattern_id = std.math.add(u32, pattern_id, 1) catch
                return error.TooManyLoopVariables;
            variable.* = .{
                .index = @intCast(index),
                // A reserved file-id namespace makes loop-carried identities
                // globally disjoint from source bindings and from the existing
                // synthetic function-parameter fallback namespace.
                .id = .{
                    .file_id = synthetic_file_id - 1,
                    .pattern_id = pattern_id,
                },
                .name = try self.loopVariableName(op, loop_kind, index),
                .ty = try self.loopVariableTypeRef(op, loop_kind, index, value),
            };
        }
        return variables;
    }

    fn loopVariableTypeRef(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        variable_index: usize,
        value: mlir.MlirValue,
    ) !obligation.TypeRef {
        if (mlir.oraValueIsNull(value)) return self.unknownTypeRef();
        const result_index = switch (loop_kind) {
            .scf_while => variable_index,
            .scf_for => if (variable_index == 0) {
                const type_id = self.u32Attr(op, "ora.induction_compiler_type_id") catch
                    return self.unknownTypeRef();
                const spec = type_builtin.lookupBuiltinByComptimeTypeId(type_id orelse
                    return self.unknownTypeRef()) orelse return self.unknownTypeRef();
                if (spec.id != .u256) return self.unknownTypeRef();
                return .{ .compiler_type_id = spec.comptime_type_id };
            } else variable_index - 1,
            .other => return self.typeRefFromValue(value),
        };
        const attr_name = try std.fmt.allocPrint(
            self.allocator,
            "ora.result_integer_signed_{d}",
            .{result_index},
        );
        const signed = (try self.boolAttr(op, attr_name)) orelse return self.unknownTypeRef();
        return self.typeRefFromValueWithIntegerSignedness(value, signed);
    }

    fn collectLoopContextVariables(self: *Collector) ![]const obligation.LoopVariable {
        if (self.function_param_names.len == 0) return &.{};
        const variables = try self.allocator.alloc(obligation.LoopVariable, self.function_param_names.len);
        for (variables, 0..) |*variable, index| {
            variable.* = .{
                .index = @intCast(index),
                .id = self.freeVarIdForFunctionParam(index),
                .name = self.function_param_names[index],
                .ty = if (index < self.function_param_types.len)
                    self.function_param_types[index]
                else
                    try self.unknownTypeRef(),
            };
        }
        return variables;
    }

    fn bindLoopValues(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        variables: []const obligation.LoopVariable,
    ) !void {
        for (variables) |variable| {
            const body_value = loopVariableValue(op, loop_kind, variable.index);
            try self.appendLoopValueBinding(body_value, variable);
            if (loop_kind == .scf_while) {
                const before = mlir.oraScfWhileOpGetBeforeBlock(op);
                if (!mlir.oraBlockIsNull(before) and variable.index < mlir.oraBlockGetNumArguments(before)) {
                    try self.appendLoopValueBinding(mlir.oraBlockGetArgument(before, variable.index), variable);
                }
            }
        }
        const result_count: usize = @intCast(mlir.oraOperationGetNumResults(op));
        for (0..result_count) |result_index| {
            const variable_index = result_index + @intFromBool(loop_kind == .scf_for);
            if (variable_index >= variables.len) continue;
            try self.appendLoopValueBinding(mlir.oraOperationGetResult(op, result_index), variables[variable_index]);
        }
    }

    fn appendLoopValueBinding(
        self: *Collector,
        value: mlir.MlirValue,
        variable: obligation.LoopVariable,
    ) !void {
        if (mlir.oraValueIsNull(value)) return;
        const value_id = @intFromPtr(value.ptr);
        for (self.loop_value_bindings.items) |existing| {
            if (existing.value_id == value_id) return;
        }
        try self.loop_value_bindings.append(self.allocator, .{
            .value_id = value_id,
            .variable = variable,
        });
    }

    fn loopVariableName(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        index: usize,
    ) !?[]const u8 {
        if (loop_kind == .scf_for and index == 0) return try self.allocator.dupe(u8, "induction");
        const result_index = if (loop_kind == .scf_for) index -| 1 else index;
        const attr_name = try std.fmt.allocPrint(self.allocator, "ora.result_name_{d}", .{result_index});
        return try self.stringAttr(op, attr_name);
    }

    fn collectLoopInitFormulas(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        variables: []const obligation.LoopVariable,
        symbol: ?[]const u8,
        ordinal: u32,
    ) ![]const obligation.FormulaRef {
        const operand_count: usize = @intCast(mlir.oraOperationGetNumOperands(op));
        const count = switch (loop_kind) {
            .scf_while => operand_count,
            .scf_for => if (operand_count >= 3) operand_count - 2 else operand_count,
            .other => 0,
        };
        if (count == 0) return &.{};
        const formulas = try self.allocator.alloc(obligation.FormulaRef, count);
        for (formulas, 0..) |*formula, index| {
            const operand_index = if (loop_kind == .scf_for and index > 0) index + 2 else index;
            const value = mlir.oraOperationGetOperand(op, operand_index);
            const signed_override = if (index < variables.len)
                if (typeRefIntegerInfo(variables[index].ty)) |info| info.signed else null
            else
                null;
            formula.* = try self.loopFormulaFromValueWithIntegerSignedness(
                value,
                signed_override,
                operationName(op),
                symbol,
                ordinal,
                @intCast(operand_index),
            );
        }
        return formulas;
    }

    fn collectLoopGuardFormula(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        variables: []const obligation.LoopVariable,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !?obligation.FormulaRef {
        if (loop_kind == .scf_for) {
            if (mlir.oraOperationGetNumOperands(op) < 2 or variables.len == 0) return null;
            const induction = loopVariableValue(op, loop_kind, 0);
            const induction_ty = variables[0].ty;
            const signed_override = if (typeRefIntegerInfo(induction_ty)) |info| info.signed else null;
            const induction_term = (try self.termFromValueWithIntegerSignedness(induction, signed_override)) orelse return null;
            const upper_term = (try self.termFromValueWithIntegerSignedness(
                mlir.oraOperationGetOperand(op, 1),
                signed_override,
            )) orelse return null;
            return .{ .term = try self.addTerm(.{ .binary = .{
                .op = .lt,
                .lhs = induction_term,
                .rhs = upper_term,
                .ty = induction_ty,
            } }) };
        }
        if (loop_kind != .scf_while) return null;
        const before = mlir.oraScfWhileOpGetBeforeBlock(op);
        if (mlir.oraBlockIsNull(before)) return null;
        const condition = findOperationInBlock(before, "scf.condition") orelse return null;
        if (mlir.oraOperationGetNumOperands(condition) == 0) return null;
        const source_guard = self.sourceLoopGuardValue(mlir.oraOperationGetOperand(condition, 0)) orelse
            return null;
        return try self.loopFormulaFromValue(
            source_guard,
            "scf.condition",
            symbol,
            ordinal,
            0,
        );
    }

    fn sourceLoopGuardValue(self: *Collector, initial: mlir.MlirValue) ?mlir.MlirValue {
        var value = initial;
        var remaining = self.operation_ordinals.count() + 1;
        while (remaining > 0 and mlir.oraValueIsAOpResult(value)) : (remaining -= 1) {
            const owner = mlir.oraOpResultGetOwner(value);
            const attr = mlir.oraOperationGetAttributeByName(owner, strRef("ora.loop_source_guard_operand"));
            if (mlir.oraAttributeIsNull(attr)) return value;
            if (!std.mem.eql(u8, operationName(owner), "arith.andi")) return null;
            const text = mlir.oraIntegerAttrGetValueString(attr);
            defer if (text.data != null) mlir.oraStringRefFree(text);
            if (text.data == null) return null;
            const operand_index = std.fmt.parseInt(usize, text.data[0..text.length], 10) catch return null;
            if (operand_index >= mlir.oraOperationGetNumOperands(owner)) return null;
            value = mlir.oraOperationGetOperand(owner, operand_index);
        }
        return if (remaining == 0) null else value;
    }

    fn collectLoopInvariantFormulas(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        symbol: ?[]const u8,
        ordinal: u32,
    ) ![]const obligation.FormulaRef {
        const body = loopBodyBlock(op, loop_kind);
        if (mlir.oraBlockIsNull(body)) return &.{};
        var invariant_ops: std.ArrayList(mlir.MlirOperation) = .empty;
        defer invariant_ops.deinit(self.allocator);
        try collectNamedOperationsInBlock(self.allocator, &invariant_ops, body, "ora.invariant");
        if (invariant_ops.items.len == 0) return &.{};

        const formulas = try self.allocator.alloc(obligation.FormulaRef, invariant_ops.items.len);
        for (invariant_ops.items, formulas) |invariant_op, *formula| {
            if (mlir.oraOperationGetNumOperands(invariant_op) == 0) {
                formula.* = .{ .origin_value = .{
                    .origin = mlirOrigin("ora.invariant", symbol, ordinal),
                    .kind = .operand,
                    .index = 0,
                } };
                continue;
            }
            formula.* = try self.loopFormulaFromValue(
                mlir.oraOperationGetOperand(invariant_op, 0),
                "ora.invariant",
                symbol,
                ordinal,
                0,
            );
        }
        return formulas;
    }

    fn loopFormulaFromValue(
        self: *Collector,
        value: mlir.MlirValue,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        index: u32,
    ) !obligation.FormulaRef {
        return self.loopFormulaFromValueWithIntegerSignedness(value, null, op_name, symbol, ordinal, index);
    }

    fn loopFormulaFromValueWithIntegerSignedness(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        index: u32,
    ) !obligation.FormulaRef {
        if (try self.termFromValueWithIntegerSignedness(value, signed_override)) |term| return .{ .term = term };
        return .{ .origin_value = .{
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = .operand,
            .index = index,
        } };
    }

    fn collectEffectFrameAttrs(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const write_slots = (try self.placeArrayAttr(op, "ora.write_slots")) orelse &.{};
        const read_slots = (try self.placeArrayAttr(op, "ora.read_slots")) orelse &.{};
        const modifies_slots_attr = try self.placeArrayAttr(op, "ora.modifies_slots");
        const modifies_slots = modifies_slots_attr orelse &.{};
        const write_slots_complete = (try self.boolAttr(op, "ora.write_slots_complete")) orelse false;
        const source = try self.sourceForOperation(op);

        if (modifies_slots_attr != null) {
            try self.addEffectFrameGoal(.write_covered_by_modifies, modifies_slots, write_slots, op_name, symbol, ordinal, source);
        }
        if (read_slots.len != 0 and write_slots.len != 0 and write_slots_complete) {
            const preserved_reads = try self.readsDisjointFromWrites(read_slots, write_slots);
            if (preserved_reads.len != 0) {
                try self.addEffectFrameGoal(.read_preserved_by_frame, write_slots, preserved_reads, op_name, symbol, ordinal, source);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.lock")) {
            if (try self.placeAttr(op, "key")) |locked| {
                const declared = try self.allocator.alloc(obligation.PlaceRef, 1);
                declared[0] = locked;
                try self.addEffectFrameGoal(.lock_covers_write, declared, &.{}, op_name, symbol, ordinal, source);
            } else {
                try self.addBlockingDiagnostic(.missing_effect_path, "ora.lock missing key attribute");
            }
        }

        if (try self.stringAttr(op, "ora.trusted_extern_frame")) |frame| {
            const declared = try self.allocator.alloc(obligation.PlaceRef, 1);
            declared[0] = .{
                .root = frame,
                .region = .none,
            };
            try self.addEffectFrameGoal(.external_call_frame, declared, &.{}, op_name, symbol, ordinal, source);
        }
    }

    fn addEffectFrameGoal(
        self: *Collector,
        relation: obligation.EffectFrameRelation,
        declared: []const obligation.PlaceRef,
        actual: []const obligation.PlaceRef,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        source: obligation.SourceRef,
    ) !void {
        try self.addEffectFrameGoalWithEvidence(relation, declared, actual, &.{}, false, op_name, symbol, ordinal, source);
    }

    fn addEffectFrameGoalWithEvidence(
        self: *Collector,
        relation: obligation.EffectFrameRelation,
        declared: []const obligation.PlaceRef,
        actual: []const obligation.PlaceRef,
        evidence: []const obligation.KeyDisjointEvidence,
        with_assumptions: bool,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        source: obligation.SourceRef,
    ) !void {
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .effect_frame = .{
                .relation = relation,
                .declared = declared,
                .actual = actual,
                .evidence = evidence,
            } },
        });
        if (with_assumptions) {
            try self.addQuery(.obligation, null, null, id, owner, origin, source);
        } else {
            try self.addQueryNoAssumptions(.obligation, null, null, id, owner, origin, source);
        }
    }

    fn collectEvidenceBackedEffectFrameAttrs(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const write_slots_complete = (try self.boolAttr(op, "ora.write_slots_complete")) orelse false;
        if (!write_slots_complete) return;

        const write_slots = (try self.placeArrayAttr(op, "ora.write_slots")) orelse &.{};
        const read_slots = (try self.placeArrayAttr(op, "ora.read_slots")) orelse &.{};
        if (read_slots.len == 0 or write_slots.len == 0) return;

        var actual_reads: std.ArrayList(obligation.PlaceRef) = .empty;
        var evidence_rows: std.ArrayList(obligation.KeyDisjointEvidence) = .empty;
        const owner = self.ownerFor(symbol);

        for (read_slots) |read| {
            if (placeIsDefinitelyDisjointFromAll(read, write_slots)) continue;
            const start = evidence_rows.items.len;
            if (try self.appendEvidenceForRead(owner, read, write_slots, &evidence_rows)) {
                try actual_reads.append(self.allocator, read);
            } else {
                evidence_rows.shrinkRetainingCapacity(start);
            }
        }

        if (actual_reads.items.len == 0) return;
        const source = try self.sourceForOperation(op);
        try self.addEffectFrameGoalWithEvidence(
            .read_preserved_by_key_evidence,
            write_slots,
            try actual_reads.toOwnedSlice(self.allocator),
            try evidence_rows.toOwnedSlice(self.allocator),
            true,
            op_name,
            symbol,
            ordinal,
            source,
        );
    }

    fn appendEvidenceForRead(
        self: *Collector,
        owner: obligation.Owner,
        read: obligation.PlaceRef,
        writes: []const obligation.PlaceRef,
        evidence_rows: *std.ArrayList(obligation.KeyDisjointEvidence),
    ) !bool {
        for (writes) |write| {
            if (obligation.placeDefinitelyDisjoint(read, write)) continue;
            const evidence = (try self.findKeyDisjointEvidence(owner, read, write)) orelse return false;
            try evidence_rows.append(self.allocator, evidence);
        }
        return true;
    }

    fn findKeyDisjointEvidence(
        self: *Collector,
        owner: obligation.Owner,
        read: obligation.PlaceRef,
        write: obligation.PlaceRef,
    ) !?obligation.KeyDisjointEvidence {
        const key_pair = firstDifferingParameterKeyPair(read, write) orelse return null;
        for (self.assumptions.items) |assumption| {
            if (!ownerEqual(assumption.owner, owner)) continue;
            if (assumption.kind != .requires) continue;
            const formula = assumption.formula orelse continue;
            if (formula != .term) continue;
            const disequality = self.freeVarDisequality(formula.term) orelse continue;
            if (!freeVarPairMatches(disequality.lhs.id, disequality.rhs.id, key_pair.read, key_pair.write)) continue;
            if (!typeRefsAreSameRegisteredInteger(disequality.lhs.ty, disequality.rhs.ty)) continue;
            return .{
                .kind = .free_var_disequality,
                .assumption_id = assumption.id,
                .lhs = disequality.lhs.id,
                .rhs = disequality.rhs.id,
                .read = read,
                .write = write,
                .key_index = key_pair.index,
            };
        }
        return null;
    }

    const ParameterKeyPair = struct {
        index: u32,
        read: obligation.FreeVarId,
        write: obligation.FreeVarId,
    };

    fn firstDifferingParameterKeyPair(read: obligation.PlaceRef, write: obligation.PlaceRef) ?ParameterKeyPair {
        if (read.region == .none or write.region == .none) return null;
        if (std.mem.eql(u8, read.root, "$computed_storage") or std.mem.eql(u8, write.root, "$computed_storage")) return null;
        if (read.region != write.region) return null;
        if (!std.mem.eql(u8, read.root, write.root)) return null;
        if (!stringSlicesEql(read.fields, write.fields)) return null;
        if (read.keys.len != write.keys.len) return null;
        for (read.keys, write.keys, 0..) |read_key, write_key, index| {
            if (obligation.placeKeyEql(read_key, write_key)) continue;
            if (read_key != .parameter or write_key != .parameter) return null;
            if (obligation.freeVarIdEql(read_key.parameter, write_key.parameter)) return null;
            return .{
                .index = @intCast(index),
                .read = read_key.parameter,
                .write = write_key.parameter,
            };
        }
        return null;
    }

    const FreeVarDisequality = struct {
        lhs: obligation.FreeVarRef,
        rhs: obligation.FreeVarRef,
    };

    fn freeVarDisequality(self: *Collector, id: obligation.TermId) ?FreeVarDisequality {
        if (id >= self.terms.items.len) return null;
        const term = self.terms.items[id];
        if (term != .binary or term.binary.op != .ne) return null;
        const lhs = self.freeVarRefFromTerm(term.binary.lhs) orelse return null;
        const rhs = self.freeVarRefFromTerm(term.binary.rhs) orelse return null;
        return .{ .lhs = lhs, .rhs = rhs };
    }

    fn freeVarRefFromTerm(self: *Collector, id: obligation.TermId) ?obligation.FreeVarRef {
        if (id >= self.terms.items.len) return null;
        const term = self.terms.items[id];
        if (term != .variable or term.variable != .free) return null;
        return term.variable.free;
    }

    fn readsDisjointFromWrites(
        self: *Collector,
        reads: []const obligation.PlaceRef,
        writes: []const obligation.PlaceRef,
    ) ![]const obligation.PlaceRef {
        if (reads.len == 0) return &.{};

        var preserved: std.ArrayList(obligation.PlaceRef) = .empty;
        for (reads) |read| {
            if (placeIsDefinitelyDisjointFromAll(read, writes)) try preserved.append(self.allocator, read);
        }
        return try preserved.toOwnedSlice(self.allocator);
    }

    fn collectOperation(
        self: *Collector,
        op: mlir.MlirOperation,
        kind: OpKind,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        switch (kind) {
            .requires => try self.addAssumptionOp(op, op_name, symbol, ordinal, .requires),
            .ensures => try self.addLogicalOp(op, op_name, symbol, ordinal, .ensures),
            .invariant => try self.addLogicalOp(op, op_name, symbol, ordinal, .invariant),
            .assert => try self.addAssertOp(op, op_name, symbol, ordinal, .assert),
            .cf_assert => try self.addAssertOp(op, op_name, symbol, ordinal, .contract_invariant),
            .assume => try self.addOraAssumeOp(op, op_name, symbol, ordinal),
            .refinement_guard => try self.addRefinementGuardOp(op, op_name, symbol, ordinal),
            .resource_move => try self.addResourceOp(op, .move, op_name, symbol, ordinal),
            .resource_create => try self.addResourceOp(op, .create, op_name, symbol, ordinal),
            .resource_destroy => try self.addResourceOp(op, .destroy, op_name, symbol, ordinal),
            .quantifier => try self.addQuantifierOp(op, op_name, symbol, ordinal),
        }
    }

    fn addQuantifierOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        var quantifier_view = stringAttrView(op, "quantifier") orelse {
            try self.addBlockingDiagnostic(.missing_formula, "ora.quantified missing quantifier attribute");
            return;
        };
        const quantifier_text = quantifier_view.slice();
        const variable = try self.stringAttr(op, "variable") orelse {
            try self.addBlockingDiagnostic(.missing_formula, "ora.quantified missing variable attribute");
            return;
        };
        const binder_type = try self.stringAttr(op, "variable_type") orelse {
            try self.addBlockingDiagnostic(.missing_type, "ora.quantified missing variable_type attribute");
            return;
        };

        const quantifier = quantifier_map.get(quantifier_text) orelse {
            try self.addBlockingDiagnosticFmt(.unsupported, "ora.quantified uses unsupported quantifier '{s}'", .{quantifier_text});
            return;
        };
        const classification = classifyQuantifierBinder(binder_type);
        if (classification.degradation) |degradation| {
            try self.addBlockingDiagnosticFmt(.unsupported, "ora.quantified binder '{s}' is {s}", .{
                binder_type,
                @tagName(degradation),
            });
        }

        try self.obligations.append(self.allocator, .{
            .id = self.nextId(),
            .owner = self.ownerFor(symbol),
            .source = try self.sourceForOperation(op),
            .phase = .ora_mlir,
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = .{ .quantifier = .{
                .quantifier = quantifier,
                .variable = variable,
                .binder_type = .{ .spelling = binder_type },
                .binder_sort = classification.sort,
                .fragment = classification.fragment,
                .pattern_status = .absent,
                .degradation = classification.degradation,
            } },
            .artifact_policy = .diagnostic_only,
        });
    }

    fn addAssertOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        default_role: obligation.LogicalRole,
    ) !void {
        var handled_tag = false;

        if (hasAttr(op, "ora.requires")) {
            // `ora.requires` is emitted as the semantic precondition; the
            // tagged assert is the runtime enforcement mirror and should not
            // duplicate the formal assumption.
            handled_tag = true;
        }
        if (hasAttr(op, "ora.ensures")) {
            try self.addLogicalOp(op, op_name, symbol, ordinal, .ensures);
            handled_tag = true;
        }

        if (stringAttrView(op, "ora.verification_type")) |view_value| {
            var verification_type = view_value;
            if (std.mem.eql(u8, verification_type.slice(), "guard")) {
                try self.addGuardAssertOp(op, op_name, symbol, ordinal);
                handled_tag = true;
            }
        }

        if (!handled_tag) {
            if (arithmeticSafetyFromAssertOp(op)) |safety| {
                const formula = (try self.formulaOperand(op, op_name, symbol, ordinal, 0)) orelse {
                    try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                    return;
                };
                if (self.active_loop_summary_indices.items.len != 0) {
                    const body_formula = (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse {
                        try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                        return;
                    };
                    try self.recordActiveLoopBodySafetySite(op, body_formula);
                }
                try self.addArithmeticSafetyOp(op, op_name, symbol, ordinal, safety, formula, try self.sourceForOperation(op));
                return;
            }
            try self.addLogicalOp(op, op_name, symbol, ordinal, default_role);
        }
    }

    fn arithmeticSafetyFromAssertOp(op: mlir.MlirOperation) ?obligation.ArithmeticSafetyKind {
        var message = stringAttrView(op, "message") orelse return null;
        return assert_safety_map.get(message.slice());
    }

    fn addGuardAssertOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const formula = (try self.formulaOperand(op, op_name, symbol, ordinal, 0)) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const source = try self.sourceForOperation(op);
        if (self.active_loop_summary_indices.items.len != 0) {
            const body_formula = (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse {
                try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                return;
            };
            try self.recordActiveLoopBodySafetySite(op, body_formula);
        }
        if (try self.stringAttr(op, "ora.guard_id")) |guard_id| {
            const id = self.nextId();
            try self.obligations.append(self.allocator, .{
                .id = id,
                .owner = owner,
                .source = source,
                .phase = .ora_mlir,
                .origin = origin,
                .kind = .{ .runtime_guard = .{
                    .guard_id = guard_id,
                    .formula = formula,
                    .erasure = .may_elide_if_proven,
                } },
            });
            try self.addQueryForOperation(.guard_satisfy, null, guard_id, id, owner, origin, source, op);
            try self.addQueryForOperation(.guard_violate, null, guard_id, id, owner, origin, source, op);
        } else {
            const id = self.nextId();
            try self.obligations.append(self.allocator, .{
                .id = id,
                .owner = owner,
                .source = source,
                .phase = .ora_mlir,
                .origin = origin,
                .kind = .{ .logical = .{
                    .role = .guard,
                    .formula = formula,
                } },
            });
            try self.addQueryForOperation(.obligation, .guard, null, id, owner, origin, source, op);
        }
    }

    fn addOraAssumeOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const kind: obligation.AssumptionKind = blk: {
            if (stringAttrView(op, "ora.assume_origin")) |view_value| {
                var origin = view_value;
                if (std.mem.eql(u8, origin.slice(), "path")) break :blk .path_assume;
            }
            if (stringAttrView(op, "ora.verification_context")) |view_value| {
                var context = view_value;
                if (std.mem.eql(u8, context.slice(), "path_assumption")) break :blk .path_assume;
            }
            break :blk .assume;
        };
        try self.addAssumptionOp(op, op_name, symbol, ordinal, kind);
    }

    fn addLogicalOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        role: obligation.LogicalRole,
    ) !void {
        const formula = (try self.formulaOperand(op, op_name, symbol, ordinal, 0)) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const is_loop_return_target = role == .ensures and self.active_loop_summary_indices.items.len != 0;
        const loop_post_formula = if (role == .ensures and !is_loop_return_target and self.functionHasLoopSummary(symbol)) blk: {
            const previous_loop_projection = self.use_loop_value_bindings;
            self.use_loop_value_bindings = true;
            defer self.use_loop_value_bindings = previous_loop_projection;
            break :blk (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse
                return error.MissingLoopPostFormula;
        } else null;
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        const source = try self.sourceForOperation(op);

        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .logical = .{
                .role = role,
                .formula = formula,
            } },
            // Loop invariant and body-safety authorization is owned by the
            // verifier's first-class loop queries. A generic source query for
            // the nested annotation has different premises and must not claim
            // one-to-one proof authority.
            .artifact_policy = if (self.active_loop_depth != 0 and role == .invariant)
                .diagnostic_only
            else
                .blocks_verified_artifacts,
        });
        if (role != .invariant and self.active_loop_summary_indices.items.len != 0) {
            const body_formula = (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse {
                try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                return;
            };
            if (role == .ensures) {
                try self.recordActiveLoopTargetSite(op, body_formula, id);
            } else {
                try self.recordActiveLoopBodySafetySite(op, body_formula);
            }
        }
        try self.addQueryForOperation(.obligation, role, null, id, owner, origin, source, op);
        if (loop_post_formula) |post_formula| {
            try self.attachLoopPostFormulas(
                owner,
                post_formula,
                sourceOpId(op) orelse return error.MissingLoopPostSourceBinding,
                id,
            );
        }
    }

    fn functionHasLoopSummary(self: *const Collector, symbol: ?[]const u8) bool {
        const function_name = symbol orelse return false;
        for (self.loop_summaries.items) |summary| {
            if (summary.owner == .statement and
                std.mem.eql(u8, summary.owner.statement.function_name, function_name)) return true;
        }
        return false;
    }

    fn attachLoopPostFormulas(
        self: *Collector,
        owner: obligation.Owner,
        post_formula: obligation.FormulaRef,
        source_op_id: usize,
        obligation_id: obligation.Id,
    ) !void {
        if (owner != .function) return;
        for (self.loop_summaries.items) |*summary| {
            if (summary.owner != .statement or
                !std.mem.eql(u8, summary.owner.statement.function_name, owner.function.name)) continue;
            summary.post_formulas = try appendFormula(self.allocator, summary.post_formulas, post_formula);
            if (post_formula == .origin_value) try self.appendLoopSummaryReason(summary, .loop_formula_unsupported);
            // Canonical post-query ownership is armed only for a complete,
            // supported projection. Unsupported loops retain their generic
            // postcondition query and their precise projection failure.
            if (!summary.projectionSupported()) continue;
            const successor_edge_id = loopUniqueEdgeIdForTarget(summary.*, .successor) orelse
                return error.MissingLoopPostQuery;
            const successor_query_id = loopPostQueryForEdge(
                self.queries.items,
                summary.*,
                successor_edge_id,
            ) orelse return error.MissingLoopPostQuery;
            try self.canonical_loop_post_binding_drafts.append(self.allocator, .{
                .summary_id = summary.id,
                .edge_id = successor_edge_id,
                .query_id = successor_query_id,
                .source_op_id = source_op_id,
                .obligation_id = obligation_id,
            });
        }
    }

    fn addAssumptionOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        kind: obligation.AssumptionKind,
    ) !void {
        const formula = (try self.formulaOperand(op, op_name, symbol, ordinal, 0)) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };

        try self.assumptions.append(self.allocator, .{
            .id = self.nextId(),
            .owner = self.ownerFor(symbol),
            .source = try self.sourceForOperation(op),
            .phase = .ora_mlir,
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = kind,
            .formula = formula,
        });
    }

    fn addRefinementGuardOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const formula = (try self.formulaOperand(op, op_name, symbol, ordinal, 0)) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const guard_id = try self.stringAttr(op, "ora.guard_id") orelse {
            try self.addBlockingDiagnostic(.missing_formula, "ora.refinement_guard missing ora.guard_id");
            return;
        };
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        const source = try self.sourceForOperation(op);
        if (self.active_loop_summary_indices.items.len != 0) {
            const body_formula = (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse {
                try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                return;
            };
            try self.recordActiveLoopBodySafetySite(op, body_formula);
        }

        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .runtime_guard = .{
                .guard_id = guard_id,
                .formula = formula,
                .erasure = .may_elide_if_proven,
            } },
        });
        try self.addQueryForOperation(.guard_satisfy, null, guard_id, id, owner, origin, source, op);
        try self.addQueryForOperation(.guard_violate, null, guard_id, id, owner, origin, source, op);
    }

    fn addResourceOp(
        self: *Collector,
        op: mlir.MlirOperation,
        resource_op: obligation.ResourceOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        const domain = try self.stringAttr(op, "domain") orelse {
            try self.addBlockingDiagnostic(.missing_type, "resource operation missing domain attribute");
            return;
        };
        const operand_count = mlir.oraOperationGetNumOperands(op);
        if (operand_count == 0) {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        }
        const amount_index = operand_count - 1;
        const amount = (try self.valueOperand(op, op_name, symbol, ordinal, @intCast(amount_index))) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const places = try self.resourcePlaces(op, resource_op, operand_count, op_name) orelse return;
        const source = try self.sourceForOperation(op);
        const site_index = try self.recordResourceSite(resource_op, domain, ordinal);

        for (resourceProperties(resource_op)) |property| {
            try self.addResourceGoal(resource_op, domain, places, amount, symbol, ordinal, site_index, property, source);
        }
    }

    fn resourcePlaces(
        self: *Collector,
        op: mlir.MlirOperation,
        resource_op: obligation.ResourceOperation,
        operand_count: usize,
        op_name: []const u8,
    ) !?ResourcePlaces {
        switch (resource_op) {
            .move => {
                const segments = try self.moveOperandSegments(op, operand_count, op_name) orelse return null;
                return .{
                    .source = (try self.resourcePlaceFromOperands(op, 0, segments.source_len, op_name, "source")) orelse return null,
                    .destination = (try self.resourcePlaceFromOperands(op, segments.source_len, segments.destination_len, op_name, "destination")) orelse return null,
                };
            },
            .create => {
                return .{
                    .destination = (try self.resourcePlaceFromOperands(op, 0, operand_count - 1, op_name, "destination")) orelse return null,
                };
            },
            .destroy => {
                return .{
                    .source = (try self.resourcePlaceFromOperands(op, 0, operand_count - 1, op_name, "source")) orelse return null,
                };
            },
        }
    }

    fn moveOperandSegments(
        self: *Collector,
        op: mlir.MlirOperation,
        operand_count: usize,
        op_name: []const u8,
    ) !?MoveOperandSegments {
        const attr = denseI32ArrayAttr(op, "operand_segment_sizes") orelse
            denseI32ArrayAttr(op, "operandSegmentSizes") orelse {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} missing operand segment sizes", .{op_name});
            return null;
        };

        if (mlir.oraDenseI32ArrayAttrGetNumElements(attr) != 3) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} malformed operand segment sizes", .{op_name});
            return null;
        }
        const source_len_raw = mlir.oraDenseI32ArrayAttrGetElement(attr, 0);
        const destination_len_raw = mlir.oraDenseI32ArrayAttrGetElement(attr, 1);
        const amount_len_raw = mlir.oraDenseI32ArrayAttrGetElement(attr, 2);
        if (source_len_raw <= 0 or destination_len_raw <= 0 or amount_len_raw != 1) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} invalid resource operand segments", .{op_name});
            return null;
        }
        const source_len: usize = @intCast(source_len_raw);
        const destination_len: usize = @intCast(destination_len_raw);
        if (source_len + destination_len + 1 != operand_count) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} resource operand segments do not match operands", .{op_name});
            return null;
        }
        return .{
            .source_len = source_len,
            .destination_len = destination_len,
        };
    }

    fn resourcePlaceFromOperands(
        self: *Collector,
        op: mlir.MlirOperation,
        start: usize,
        count: usize,
        op_name: []const u8,
        label: []const u8,
    ) !?obligation.PlaceRef {
        if (count == 0) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} missing {s} resource place", .{ op_name, label });
            return null;
        }

        var place = try self.placeFromResourceRootValue(mlir.oraOperationGetOperand(op, start), op_name, label) orelse return null;
        if (count == 1) return place;

        const existing_len = place.keys.len;
        const keys = try self.allocator.alloc(obligation.PlaceKey, existing_len + count - 1);
        @memcpy(keys[0..existing_len], place.keys);
        for (keys[existing_len..], 1..) |*key, index| {
            key.* = try self.placeKeyFromValue(mlir.oraOperationGetOperand(op, start + index));
        }
        place.keys = keys;
        return place;
    }

    fn placeFromResourceRootValue(
        self: *Collector,
        value: mlir.MlirValue,
        op_name: []const u8,
        label: []const u8,
    ) !?obligation.PlaceRef {
        if (mlir.oraValueIsNull(value)) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} has null {s} resource place", .{ op_name, label });
            return null;
        }

        if (mlir.mlirValueIsABlockArgument(value)) {
            const arg_number = mlir.mlirBlockArgumentGetArgNumber(value);
            return .{
                .root = try std.fmt.allocPrint(self.allocator, "arg#{d}", .{arg_number}),
                .region = .storage,
            };
        }

        if (!mlir.oraValueIsAOpResult(value)) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} has unsupported {s} resource place value", .{ op_name, label });
            return null;
        }

        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) {
            try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} has ownerless {s} resource place result", .{ op_name, label });
            return null;
        }

        const owner_name = operationName(owner);
        if (resource_root_op_map.get(owner_name)) |root_op| switch (root_op) {
            .storage_load => {
                const root = try self.stringAttr(owner, "global") orelse {
                    try self.addBlockingDiagnostic(.missing_effect_path, "ora.sload resource place missing global attribute");
                    return null;
                };
                return .{ .root = root, .region = .storage };
            },
            .transient_load => {
                const root = try self.stringAttr(owner, "key") orelse {
                    try self.addBlockingDiagnostic(.missing_effect_path, "ora.tload resource place missing key attribute");
                    return null;
                };
                return .{ .root = root, .region = .transient };
            },
            .struct_field_extract => {
                if (mlir.oraOperationGetNumOperands(owner) < 1) {
                    try self.addBlockingDiagnostic(.missing_effect_path, "ora.struct_field_extract resource place missing source operand");
                    return null;
                }
                var base = try self.placeFromResourceRootValue(mlir.oraOperationGetOperand(owner, 0), op_name, label) orelse return null;
                const field_name = try self.stringAttr(owner, "field_name") orelse {
                    try self.addBlockingDiagnostic(.missing_effect_path, "ora.struct_field_extract resource place missing field_name attribute");
                    return null;
                };
                base.fields = try appendField(self.allocator, base.fields, field_name);
                return base;
            },
            .integer_constant => {},
        };
        if (isTransparentValueOp(owner_name)) {
            if (mlir.oraOperationGetNumOperands(owner) < 1) {
                try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} has operandless {s} resource place cast", .{ op_name, label });
                return null;
            }
            return try self.placeFromResourceRootValue(mlir.oraOperationGetOperand(owner, 0), op_name, label);
        }

        try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} has unsupported {s} resource place root '{s}'", .{ op_name, label, owner_name });
        return null;
    }

    fn placeKeyFromValue(self: *Collector, value: mlir.MlirValue) !obligation.PlaceKey {
        return try self.placeKeyFromValueDepth(value, 0);
    }

    fn placeKeyFromValueDepth(self: *Collector, value: mlir.MlirValue, depth: u32) !obligation.PlaceKey {
        if (depth > 32) return .{ .unknown = {} };
        if (mlir.oraValueIsNull(value)) return .{ .unknown = {} };
        if (mlir.mlirValueIsABlockArgument(value)) {
            const arg_number = self.functionEntryBlockArgumentNumber(value) orelse return .{ .unknown = {} };
            const id = (try self.functionParamBindingIdForPlaceKey(arg_number, "MLIR block argument")) orelse return .{ .unknown = {} };
            return .{ .parameter = id };
        }
        if (mlir.oraValueIsAOpResult(value)) {
            const owner = mlir.oraOpResultGetOwner(value);
            if (!mlir.oraOperationIsNull(owner)) {
                const owner_name = operationName(owner);
                if (resource_root_op_map.get(owner_name) == .integer_constant) {
                    const attr = mlir.oraOperationGetAttributeByName(owner, strRef("value"));
                    if (!mlir.oraAttributeIsNull(attr)) {
                        const text = mlir.oraIntegerAttrGetValueString(attr);
                        defer if (text.data != null) mlir.oraStringRefFree(text);
                        if (text.data != null) return .{ .constant = try self.allocator.dupe(u8, text.data[0..text.length]) };
                    }
                }
                if (mlir.oraTypeIsAddressType(mlir.oraValueGetType(value))) {
                    if (std.mem.eql(u8, owner_name, "ora.evm.caller")) return .{ .msg_sender = {} };
                    if (std.mem.eql(u8, owner_name, "ora.evm.origin")) return .{ .tx_origin = {} };
                }
                if (isTransparentValueOp(owner_name) and mlir.oraOperationGetNumOperands(owner) >= 1) {
                    return try self.placeKeyFromValueDepth(mlir.oraOperationGetOperand(owner, 0), depth + 1);
                }
            }
        }
        return .{ .unknown = {} };
    }

    fn addResourceGoal(
        self: *Collector,
        resource_op: obligation.ResourceOperation,
        domain: []const u8,
        places: ResourcePlaces,
        amount: obligation.FormulaRef,
        symbol: ?[]const u8,
        ordinal: u32,
        site_index: usize,
        property: obligation.ResourceProperty,
        source: obligation.SourceRef,
    ) !void {
        const origin: obligation.Origin = .{ .resource_op = .{
            .op = resource_op,
            .domain = domain,
            .ordinal = ordinal,
        } };
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .resource = .{
                .op = resource_op,
                .domain = domain,
                .source = places.source,
                .destination = places.destination,
                .amount = amount,
                .property = property,
            } },
        });
        try self.addQuery(.obligation, null, null, id, owner, origin, source);
        try self.markResourcePropertyEmitted(site_index, property);
    }

    fn recordResourceSite(
        self: *Collector,
        resource_op: obligation.ResourceOperation,
        domain: []const u8,
        ordinal: u32,
    ) !usize {
        const index = self.resource_sites.items.len;
        try self.resource_sites.append(self.allocator, .{
            .op = resource_op,
            .domain = domain,
            .ordinal = ordinal,
            .expected_mask = expectedResourcePropertyMask(resource_op),
        });
        return index;
    }

    fn markResourcePropertyEmitted(
        self: *Collector,
        site_index: usize,
        property: obligation.ResourceProperty,
    ) !void {
        if (site_index >= self.resource_sites.items.len) {
            try self.addBlockingDiagnostic(.incomplete_resource_goals, "resource goal emitted for unknown resource operation site");
            return;
        }

        const bit = resourcePropertyBit(property);
        const site = &self.resource_sites.items[site_index];
        if ((site.expected_mask & bit) == 0) {
            try self.addBlockingDiagnosticFmt(.incomplete_resource_goals, "resource {s} emitted unexpected {s} goal", .{
                @tagName(site.op),
                @tagName(property),
            });
            return;
        }
        if ((site.emitted_mask & bit) != 0) {
            try self.addBlockingDiagnosticFmt(.incomplete_resource_goals, "resource {s} emitted duplicate {s} goal", .{
                @tagName(site.op),
                @tagName(property),
            });
            return;
        }
        site.emitted_mask |= bit;
    }

    fn verifyResourceGoalCompleteness(self: *Collector) !void {
        for (self.resource_sites.items) |site| {
            if (resourceSiteComplete(site)) continue;
            try self.addBlockingDiagnosticFmt(.incomplete_resource_goals, "resource {s} op #{d} in domain '{s}' is missing verifier goals: expected mask 0x{x}, emitted mask 0x{x}", .{
                @tagName(site.op),
                site.ordinal,
                site.domain,
                site.expected_mask,
                site.emitted_mask,
            });
        }
    }

    fn addArithmeticSafetyOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        safety: obligation.ArithmeticSafetyKind,
        formula_override: ?obligation.FormulaRef,
        source: obligation.SourceRef,
    ) !void {
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .logical = .{
                .role = .arithmetic_safety,
                .formula = formula_override orelse derivedFormula(op_name, symbol, ordinal),
                .arithmetic_safety = safety,
            } },
            .artifact_policy = if (self.active_loop_depth != 0)
                .diagnostic_only
            else
                .blocks_verified_artifacts,
        });
        try self.addQueryForOperation(.obligation, .arithmetic_safety, null, id, owner, origin, source, op);
    }

    fn addQuery(
        self: *Collector,
        kind: obligation.VerificationQueryKind,
        logical_role: ?obligation.LogicalRole,
        guard_id: ?[]const u8,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
        source: obligation.SourceRef,
    ) !void {
        const assumption_ids = try self.assumptionIdsForOwner(owner);
        try self.appendQuery(kind, logical_role, guard_id, obligation_id, owner, origin, source, assumption_ids, null);
    }

    fn addQueryForOperation(
        self: *Collector,
        kind: obligation.VerificationQueryKind,
        logical_role: ?obligation.LogicalRole,
        guard_id: ?[]const u8,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
        source: obligation.SourceRef,
        op: mlir.MlirOperation,
    ) !void {
        const assumption_ids = try self.assumptionIdsForOwner(owner);
        try self.appendQuery(
            kind,
            logical_role,
            guard_id,
            obligation_id,
            owner,
            origin,
            source,
            assumption_ids,
            sourceOpId(op),
        );
    }

    fn addQueryNoAssumptions(
        self: *Collector,
        kind: obligation.VerificationQueryKind,
        logical_role: ?obligation.LogicalRole,
        guard_id: ?[]const u8,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
        source: obligation.SourceRef,
    ) !void {
        try self.appendQuery(kind, logical_role, guard_id, obligation_id, owner, origin, source, &.{}, null);
    }

    fn appendQuery(
        self: *Collector,
        kind: obligation.VerificationQueryKind,
        logical_role: ?obligation.LogicalRole,
        guard_id: ?[]const u8,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
        source: obligation.SourceRef,
        assumption_ids: []const obligation.Id,
        source_op_id: ?usize,
    ) !void {
        const obligation_ids = try self.allocator.alloc(obligation.Id, 1);
        obligation_ids[0] = obligation_id;
        const query_id = self.nextId();
        const query: obligation.VerificationQuery = .{
            .id = query_id,
            .owner = owner,
            .source = source,
            .phase = .report,
            .origin = origin,
            .artifact_policy = if (self.active_loop_depth != 0)
                .diagnostic_only
            else
                .blocks_verified_artifacts,
            .kind = kind,
            .logical_role = logical_role,
            .guard_id = guard_id,
            .obligation_ids = obligation_ids,
            .assumption_ids = assumption_ids,
        };
        try self.queries.append(self.allocator, .{
            .id = query.id,
            .owner = query.owner,
            .source = query.source,
            .phase = query.phase,
            .origin = query.origin,
            .artifact_policy = query.artifact_policy,
            .kind = query.kind,
            .logical_role = query.logical_role,
            .guard_id = query.guard_id,
            .obligation_ids = query.obligation_ids,
            .assumption_ids = query.assumption_ids,
        });
        if (source_op_id) |id| {
            try self.query_bindings.append(self.allocator, .{
                .source_op_id = id,
                .kind = kind,
                .logical_role = logical_role,
                .guard_id = guard_id,
                .query_id = query_id,
                .assumption_ids = assumption_ids,
                .obligation_ids = obligation_ids,
            });
        }
    }

    fn assumptionIdsForOwner(self: *Collector, owner: obligation.Owner) ![]const obligation.Id {
        var count: usize = 0;
        for (self.assumptions.items) |item| {
            if (ownerEqual(item.owner, owner)) count += 1;
        }
        if (count == 0) return &.{};

        const ids = try self.allocator.alloc(obligation.Id, count);
        var index: usize = 0;
        for (self.assumptions.items) |item| {
            if (!ownerEqual(item.owner, owner)) continue;
            ids[index] = item.id;
            index += 1;
        }
        return ids;
    }

    fn assumptionIdsForLoopOwner(self: *Collector, owner: obligation.Owner) ![]const obligation.Id {
        var count: usize = 0;
        for (self.assumptions.items) |item| {
            if (item.kind != .path_assume and ownerEqual(item.owner, owner)) count += 1;
        }
        if (count == 0) return &.{};

        const ids = try self.allocator.alloc(obligation.Id, count);
        var index: usize = 0;
        for (self.assumptions.items) |item| {
            if (item.kind == .path_assume or !ownerEqual(item.owner, owner)) continue;
            ids[index] = item.id;
            index += 1;
        }
        return ids;
    }

    fn addBaseQueriesForFunctionOwners(self: *Collector) !void {
        var seen = std.StringHashMap(void).init(self.allocator);
        defer seen.deinit();

        for (self.assumptions.items) |item| {
            try self.addBaseQueryForOwner(item.owner, &seen);
        }
        for (self.obligations.items) |item| {
            try self.addBaseQueryForOwner(item.owner, &seen);
        }
    }

    fn addBaseQueryForOwner(
        self: *Collector,
        owner: obligation.Owner,
        seen: *std.StringHashMap(void),
    ) !void {
        if (owner != .function) return;
        const name = owner.function.name;
        const entry = try seen.getOrPut(name);
        if (entry.found_existing) return;
        try self.queries.append(self.allocator, .{
            .id = self.nextId(),
            .owner = owner,
            .source = .generated(),
            .phase = .report,
            .origin = .{ .source = {} },
            .kind = .base,
        });
    }

    fn formulaOperand(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        operand_index: u32,
    ) !?obligation.FormulaRef {
        if (mlir.oraOperationGetNumOperands(op) <= operand_index) return null;
        const value = mlir.oraOperationGetOperand(op, operand_index);
        if (try self.termFromValue(value)) |term| {
            var formula: obligation.FormulaRef = .{ .term = term };
            if (self.function_param_names.len != 0 and
                !std.mem.eql(u8, op_name, "ora.requires") and
                !std.mem.eql(u8, op_name, "ora.assume") and
                !hasAttr(op, "ora.requires"))
            {
                formula = .{ .term = try self.wrapFunctionParamsWithOwnerAssumptions(term, self.ownerFor(symbol)) };
            }
            return formula;
        }
        return .{ .origin_value = .{
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = .operand,
            .index = operand_index,
        } };
    }

    fn valueOperand(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        operand_index: u32,
    ) !?obligation.FormulaRef {
        if (mlir.oraOperationGetNumOperands(op) <= operand_index) return null;
        const value = mlir.oraOperationGetOperand(op, operand_index);
        if (try self.termFromValue(value)) |term| return .{ .term = term };
        return .{ .origin_value = .{
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = .operand,
            .index = operand_index,
        } };
    }

    fn termFromValue(self: *Collector, value: mlir.MlirValue) anyerror!?obligation.TermId {
        return self.termFromValueWithIntegerSignedness(value, null);
    }

    fn termFromValueWithIntegerSignedness(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        if (mlir.oraValueIsNull(value)) return null;

        const value_id = @intFromPtr(value.ptr);
        var override_index = self.active_loop_formula_overrides.len;
        while (override_index > 0) {
            override_index -= 1;
            const override = self.active_loop_formula_overrides[override_index];
            if (override.value_id != value_id) continue;
            return switch (override.formula) {
                .term => |term| term,
                .origin_value => null,
            };
        }

        if (self.use_loop_value_bindings) if (self.loopVariableForValue(value)) |variable| {
            return try self.addTerm(.{
                .variable = .{
                    .free = .{
                        .id = variable.id orelse return null,
                        .name = variable.name orelse "loop_var",
                        // Canonical MLIR integer carriers are frequently signless.  A
                        // relation such as equality therefore carries no predicate
                        // signedness from which to recover the Ora type.  Preserve
                        // the compiler-owned loop-variable type unless an operation
                        // explicitly supplies a signedness interpretation.
                        .ty = try self.canonicalTypeRefForValue(value, signed_override),
                    },
                },
            });
        };

        if (mlir.mlirValueIsABlockArgument(value)) {
            const arg_number: usize = self.functionEntryBlockArgumentNumber(value) orelse return null;
            const name = if (arg_number < self.function_param_names.len)
                self.function_param_names[arg_number]
            else
                try std.fmt.allocPrint(self.allocator, "arg#{d}", .{arg_number});
            return try self.addTerm(.{
                .variable = .{
                    .free = .{
                        .id = self.freeVarIdForFunctionParam(arg_number),
                        .name = name,
                        // Function parameter metadata owns Ora signedness.  The
                        // MLIR block argument alone cannot distinguish iN from
                        // uN when the consuming operation is signedness-neutral.
                        .ty = try self.canonicalTypeRefForValue(value, signed_override),
                    },
                },
            });
        }

        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;

        const owner_name = operationName(owner);
        if (isTransparentValueOp(owner_name)) {
            if (mlir.oraOperationGetNumOperands(owner) == 0) return null;
            return try self.termFromValueWithIntegerSignedness(mlir.oraOperationGetOperand(owner, 0), signed_override);
        }
        if (std.mem.eql(u8, owner_name, "ora.old")) return try self.oldTerm(owner);
        if (std.mem.eql(u8, owner_name, "ora.sload")) return try self.scalarStorageLoadTerm(owner);
        if (std.mem.eql(u8, owner_name, "ora.map_get")) return try self.storageMapReadTerm(owner);
        if (std.mem.eql(u8, owner_name, "memref.load")) return try self.fixedArrayReadTerm(owner);
        if (std.mem.eql(u8, owner_name, "arith.constant")) return try self.constantTerm(owner, signed_override);
        if (std.mem.eql(u8, owner_name, "arith.trunci")) {
            return try self.truncatedIntegerLiteralTerm(owner, signed_override);
        }
        if (arithmetic_value_op_map.get(owner_name)) |spec| {
            const type_source: BinaryTypeSource = switch (spec.signedness) {
                .compiler_attribute => blk: {
                    const signed = (try self.boolAttr(owner, "ora.integer_signed")) orelse {
                        try self.addBlockingDiagnosticFmt(
                            .missing_type,
                            "{s} has no compiler-owned ora.integer_signed metadata",
                            .{owner_name},
                        );
                        return null;
                    };
                    break :blk .{ .explicit_signedness = signed };
                },
                .unsigned => .{ .explicit_signedness = false },
                .signed => .{ .explicit_signedness = true },
            };
            const term_id = (try self.binaryTermFromOperands(owner, spec.op, type_source)) orelse return null;
            const requires_safety = switch (spec.safety) {
                .required => true,
                .wrapping => false,
                .compiler_attribute => (try self.boolAttr(owner, "ora.integer_checked")) orelse {
                    try self.addBlockingDiagnosticFmt(
                        .missing_type,
                        "{s} has no compiler-owned ora.integer_checked metadata",
                        .{owner_name},
                    );
                    return null;
                },
            };
            if (requires_safety) try self.recordArithmeticSafetyTerm(term_id);
            return term_id;
        }
        if (arithmetic_div_rem_value_op_map.get(owner_name)) |spec| {
            const term_id = (try self.binaryTermFromOperands(
                owner,
                spec.op,
                .{ .explicit_signedness = spec.signed },
            )) orelse return null;
            try self.recordArithmeticSafetyTerm(term_id);
            return term_id;
        }
        if (std.mem.eql(u8, owner_name, "arith.xori")) return try self.xoriTerm(owner, signed_override);
        if (std.mem.eql(u8, owner_name, "arith.andi")) return try self.andiTerm(owner, signed_override);
        if (std.mem.eql(u8, owner_name, "arith.ori")) return try self.booleanLogicalTerm(owner, .or_);
        if (std.mem.eql(u8, owner_name, "arith.cmpi")) return try self.cmpiTerm(owner);
        if (std.mem.eql(u8, owner_name, "ora.cmp")) return try self.oraCmpTerm(owner);
        if (std.mem.eql(u8, owner_name, "ora.quantified")) return try self.quantifiedTerm(owner);
        return null;
    }

    fn loopVariableForValue(
        self: *Collector,
        value: mlir.MlirValue,
    ) ?obligation.LoopVariable {
        const value_id = @intFromPtr(value.ptr);
        var index = self.loop_value_bindings.items.len;
        while (index != 0) {
            index -= 1;
            const binding = self.loop_value_bindings.items[index];
            if (binding.value_id == value_id) return binding.variable;
        }
        return null;
    }

    fn scalarStorageLoadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        const root = try self.stringAttr(op, "global") orelse return null;
        // Function queries require an independently proven stable place. A
        // LoopVC instead denotes this term against its explicit current store.
        if (!self.canProjectStablePlace(.{ .root = root, .region = .storage }) and
            !self.use_loop_value_bindings) return null;
        return try self.placeReadTerm(root);
    }

    fn storageMapReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        const place = (try self.storageMapReadPlaceFromMapGet(op)) orelse return null;
        if (!self.canProjectStablePlace(place) and !self.canProjectLoopCurrentMappingPlace(place)) return null;
        return try self.placeReadTermFromPlace(place);
    }

    fn fixedArrayReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!self.use_loop_value_bindings or
            !mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        const place = (try self.fixedArrayReadPlaceFromLoad(op)) orelse return null;
        return try self.placeReadTermFromPlace(place);
    }

    fn canProjectLoopCurrentMappingPlace(self: *const Collector, place: obligation.PlaceRef) bool {
        if (!self.use_loop_value_bindings or place.region != .storage or
            place.fields.len != 0 or place.keys.len != 1) return false;
        return place.keys[0] == .parameter;
    }

    fn canProjectStablePlace(self: *Collector, place: obligation.PlaceRef) bool {
        if (self.function_has_external_call) return false;
        if (!self.function_write_slots_complete) return false;
        const write_slots = self.function_write_slots orelse return false;
        return placeIsDefinitelyDisjointFromAll(place, write_slots);
    }

    fn oldTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) != 1) return null;
        if (!mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        if (self.function_has_external_call) return null;
        if (!self.function_write_slots_complete) return null;
        const write_slots = self.function_write_slots orelse return null;

        const operand = mlir.oraOperationGetOperand(op, 0);
        if (try self.storageMapReadPlaceFromValue(operand)) |place| {
            if (placeIsDefinitelyDisjointFromAll(place, write_slots)) return try self.placeReadTermFromPlace(place);
            if (self.canProjectLoopCurrentMappingPlace(place)) {
                return try self.addTerm(.{ .old = try self.placeReadTermFromPlace(place) });
            }
            return null;
        }

        if (try self.fixedArrayReadPlaceFromValue(operand)) |place| {
            if (!self.use_loop_value_bindings) return null;
            return try self.addTerm(.{ .old = try self.placeReadTermFromPlace(place) });
        }

        const root = (try self.scalarStorageLoadRootFromValue(operand)) orelse return null;
        const place: obligation.PlaceRef = .{ .root = root, .region = .storage };

        if (placeIsDefinitelyDisjointFromAll(place, write_slots)) return try self.placeReadTerm(root);

        const place_read = try self.placeReadTerm(root);
        return try self.addTerm(.{ .old = place_read });
    }

    fn scalarStorageLoadRootFromValue(self: *Collector, value: mlir.MlirValue) !?[]const u8 {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;
        if (!std.mem.eql(u8, operationName(owner), "ora.sload")) return null;
        if (!mlirTypeIsSupportedInteger(self.operationResultType(owner, 0))) return null;
        return try self.stringAttr(owner, "global");
    }

    fn placeReadTerm(self: *Collector, root: []const u8) !obligation.TermId {
        return try self.placeReadTermFromPlace(.{
            .root = root,
            .region = .storage,
        });
    }

    fn placeReadTermFromPlace(self: *Collector, place: obligation.PlaceRef) !obligation.TermId {
        return try self.addTerm(.{ .place_read = place });
    }

    fn storageMapReadPlaceFromValue(self: *Collector, value: mlir.MlirValue) !?obligation.PlaceRef {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;
        if (!std.mem.eql(u8, operationName(owner), "ora.map_get")) return null;
        return try self.storageMapReadPlaceFromMapGet(owner);
    }

    fn fixedArrayReadPlaceFromValue(self: *Collector, value: mlir.MlirValue) !?obligation.PlaceRef {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner) or
            !std.mem.eql(u8, operationName(owner), "memref.load")) return null;
        return try self.fixedArrayReadPlaceFromLoad(owner);
    }

    fn fixedArrayReadPlaceFromLoad(self: *Collector, op: mlir.MlirOperation) !?obligation.PlaceRef {
        if (mlir.oraOperationGetNumOperands(op) != 2 or
            mlir.oraOperationGetNumResults(op) != 1) return null;
        _ = fixedIndexExtent(op, 0, 1, 0) catch return null;
        const root = persistentStorageRootForCarrier(
            mlir.oraOperationGetOperand(op, 0),
        ) catch return null;
        const key = try self.placeKeyFromValue(fixedIndexSourceValue(
            mlir.oraOperationGetOperand(op, 1),
        ));
        if (key != .parameter) return null;
        const keys = try self.allocator.alloc(obligation.PlaceKey, 1);
        keys[0] = key;
        return .{
            .root = root,
            .region = .storage,
            .projection_domain = .fixed_index,
            .keys = keys,
        };
    }

    fn storageMapReadPlaceFromMapGet(self: *Collector, op: mlir.MlirOperation) !?obligation.PlaceRef {
        var reversed_keys: std.ArrayList(obligation.PlaceKey) = .empty;
        defer reversed_keys.deinit(self.allocator);

        var current = op;
        var depth: u32 = 0;
        while (true) : (depth += 1) {
            if (depth > 32) return null;
            if (mlir.oraOperationGetNumOperands(current) < 2) return null;

            const key = try self.placeKeyFromValue(mlir.oraOperationGetOperand(current, 1));
            if (key == .unknown) return null;
            try reversed_keys.append(self.allocator, key);

            const source = mlir.oraOperationGetOperand(current, 0);
            const source_owner = storageMapSourceOwner(source) orelse return null;
            const source_name = operationName(source_owner);
            if (std.mem.eql(u8, source_name, "ora.map_get")) {
                current = source_owner;
                continue;
            }
            if (!std.mem.eql(u8, source_name, "ora.sload")) return null;

            const root = try self.stringAttr(source_owner, "global") orelse return null;
            const keys = try self.allocator.alloc(obligation.PlaceKey, reversed_keys.items.len);
            for (reversed_keys.items, 0..) |item, index| {
                keys[keys.len - 1 - index] = item;
            }
            return .{
                .root = root,
                .region = .storage,
                .keys = keys,
            };
        }
    }

    fn constantTerm(self: *Collector, op: mlir.MlirOperation, signed_override: ?bool) anyerror!?obligation.TermId {
        if (try self.stringAttr(op, "ora.bound_variable")) |name| {
            const binder = self.boundVariableForName(name) orelse {
                try self.addBlockingDiagnosticFmt(.missing_formula, "bound variable '{s}' has no active binder", .{name});
                return null;
            };
            return try self.addTerm(.{ .variable = .{
                .bound = binder,
            } });
        }

        const value_attr = mlir.oraOperationGetAttributeByName(op, strRef("value"));
        if (mlir.oraAttributeIsNull(value_attr)) return null;

        const text = mlir.oraIntegerAttrGetValueString(value_attr);
        defer if (text.data != null) mlir.oraStringRefFree(text);
        if (text.data != null) {
            const result_ty = self.operationResultType(op, 0);
            if (operationResultIsI1(op)) {
                const literal = std.mem.trim(u8, text.data[0..text.length], " \t\n\r");
                if (std.mem.eql(u8, literal, "0")) return try self.addTerm(.{ .bool_lit = false });
                if (std.mem.eql(u8, literal, "1")) return try self.addTerm(.{ .bool_lit = true });
            }
            const ty = try self.typeRefFromMlirTypeWithIntegerSignedness(result_ty, signed_override);
            return try self.addTerm(.{ .int_lit = .{
                .value = try normalizeIntegerLiteralText(self.allocator, text.data[0..text.length], integerTypeWidth(result_ty)),
                .ty = ty,
            } });
        }

        return try self.addTerm(.{ .bool_lit = mlir.oraBoolAttrGetValue(value_attr) });
    }

    fn truncatedIntegerLiteralTerm(
        self: *Collector,
        op: mlir.MlirOperation,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumResults(op) != 1) return null;

        const operand = mlir.oraOperationGetOperand(op, 0);
        if (!mlir.oraValueIsAOpResult(operand)) return null;
        const literal_op = mlir.oraOpResultGetOwner(operand);
        if (mlir.oraOperationIsNull(literal_op) or
            !std.mem.eql(u8, operationName(literal_op), "arith.constant")) return null;

        const target_ty = self.operationResultType(op, 0);
        const target_width = integerTypeWidth(target_ty) orelse return null;
        if (target_width <= 1) return null;

        const value_attr = mlir.oraOperationGetAttributeByName(literal_op, strRef("value"));
        if (mlir.oraAttributeIsNull(value_attr)) return null;
        const text = mlir.oraIntegerAttrGetValueString(value_attr);
        defer if (text.data != null) mlir.oraStringRefFree(text);
        if (text.data == null) return null;

        return try self.addTerm(.{ .int_lit = .{
            .value = try normalizeIntegerLiteralText(
                self.allocator,
                text.data[0..text.length],
                target_width,
            ),
            .ty = try self.typeRefFromMlirTypeWithIntegerSignedness(target_ty, signed_override),
        } });
    }

    fn cmpiTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) < 2) return null;
        const predicate_attr = mlir.oraOperationGetAttributeByName(op, strRef("predicate"));
        if (mlir.oraAttributeIsNull(predicate_attr)) return null;
        const binary_op = cmpiPredicateToBinaryOp(mlir.oraIntegerAttrGetValueSInt(predicate_attr)) orelse return null;
        return try self.binaryTermFromOperands(op, binary_op, .predicate_signedness);
    }

    fn oraCmpTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) < 2) return null;
        var predicate = stringAttrView(op, "predicate") orelse return null;
        const binary_op = stringPredicateToBinaryOp(predicate.slice()) orelse return null;
        if (!try self.validateOraCmpPredicateTypes(op, predicate.slice(), binary_op)) return null;
        return try self.binaryTermFromOperands(op, binary_op, .operand_signedness);
    }

    fn xoriTerm(
        self: *Collector,
        op: mlir.MlirOperation,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) < 2) return null;
        if (!operationResultIsI1(op)) return try self.integerBitwiseTerm(op, .bit_xor, signed_override);
        const lhs = (try self.termFromValue(mlir.oraOperationGetOperand(op, 0))) orelse return null;
        const rhs = (try self.termFromValue(mlir.oraOperationGetOperand(op, 1))) orelse return null;
        if (self.termIsBoolLiteral(rhs, true)) {
            return try self.addTerm(.{ .unary = .{ .op = .not, .operand = lhs } });
        }
        if (self.termIsBoolLiteral(lhs, true)) {
            return try self.addTerm(.{ .unary = .{ .op = .not, .operand = rhs } });
        }
        return null;
    }

    fn andiTerm(
        self: *Collector,
        op: mlir.MlirOperation,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        if (operationResultIsI1(op)) return try self.booleanLogicalTerm(op, .and_);
        return try self.integerBitwiseTerm(op, .bit_and, signed_override);
    }

    fn booleanLogicalTerm(
        self: *Collector,
        op: mlir.MlirOperation,
        binary_op: obligation.BinaryOp,
    ) anyerror!?obligation.TermId {
        if (!operationResultIsI1(op)) return null;
        return try self.binaryTermFromOperands(op, binary_op, .operand_signedness);
    }

    fn integerBitwiseTerm(
        self: *Collector,
        op: mlir.MlirOperation,
        binary_op: obligation.BinaryOp,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        return try self.binaryTermFromOperands(
            op,
            binary_op,
            if (signed_override) |signed|
                .{ .explicit_signedness = signed }
            else
                .operand_signedness,
        );
    }

    fn termIsBoolLiteral(self: *Collector, id: obligation.TermId, value: bool) bool {
        if (id >= self.terms.items.len) return false;
        const term = self.terms.items[id];
        return term == .bool_lit and term.bool_lit == value;
    }

    fn operationResultType(_: *Collector, op: mlir.MlirOperation, index: usize) mlir.MlirType {
        if (mlir.oraOperationGetNumResults(op) <= index) return std.mem.zeroes(mlir.MlirType);
        const result = mlir.oraOperationGetResult(op, index);
        if (mlir.oraValueIsNull(result)) return std.mem.zeroes(mlir.MlirType);
        return mlir.oraValueGetType(result);
    }

    fn operationFeedsOnlyFormalFormula(_: *Collector, op: mlir.MlirOperation) bool {
        const result_count = mlir.oraOperationGetNumResults(op);
        if (result_count == 0) return false;
        var index: usize = 0;
        while (index < result_count) : (index += 1) {
            if (!valueFeedsOnlyFormalFormula(mlir.oraOperationGetResult(op, index), 0)) return false;
        }
        return true;
    }

    fn functionParamTypesFromFunctionOp(
        self: *Collector,
        op: mlir.MlirOperation,
        expected_count: usize,
    ) ![]const obligation.TypeRef {
        if (expected_count == 0) return &.{};

        const types = try self.allocator.alloc(obligation.TypeRef, expected_count);
        const block = functionEntryBlock(op);
        const arg_count: usize = if (!mlir.oraBlockIsNull(block)) @intCast(mlir.oraBlockGetNumArguments(block)) else 0;
        const compiler_type_ids = (try self.u32ArrayAttr(op, "ora.param_compiler_type_ids")) orelse &.{};
        if (arg_count < expected_count) {
            try self.addBlockingDiagnosticFmt(
                .missing_type,
                "func.func has {d} named parameters but only {d} MLIR block arguments",
                .{ expected_count, arg_count },
            );
        }
        if (compiler_type_ids.len != 0 and compiler_type_ids.len != expected_count) {
            try self.addBlockingDiagnosticFmt(
                .missing_type,
                "ora.param_compiler_type_ids has {d} entries but ora.param_names has {d}",
                .{ compiler_type_ids.len, expected_count },
            );
        }

        for (types, 0..) |*ty, index| {
            if (index >= arg_count) {
                ty.* = try self.unknownTypeRef();
                continue;
            }
            const value = mlir.oraBlockGetArgument(block, index);
            if (index < compiler_type_ids.len and compiler_type_ids[index] != 0) {
                const type_id = compiler_type_ids[index];
                const info = type_builtin.integerInfoByComptimeTypeId(type_id) orelse {
                    ty.* = .{ .compiler_type_id = type_id };
                    continue;
                };
                if (integerTypeWidth(mlir.oraValueGetType(value)) != info.width) {
                    try self.addBlockingDiagnosticFmt(
                        .missing_type,
                        "parameter {d} compiler type id {d} disagrees with its MLIR width",
                        .{ index, type_id },
                    );
                    ty.* = try self.unknownTypeRef();
                    continue;
                }
                ty.* = .{ .compiler_type_id = type_id };
                continue;
            }
            ty.* = try self.typeRefFromValue(value);
        }
        return types;
    }

    fn functionEntryBlockArgumentNumber(self: *Collector, value: mlir.MlirValue) ?usize {
        if (!mlir.mlirValueIsABlockArgument(value)) return null;
        if (mlir.oraBlockIsNull(self.function_entry_block)) return null;
        const owner_block = mlir.mlirBlockArgumentGetOwner(value);
        if (mlir.oraBlockIsNull(owner_block)) return null;
        if (owner_block.ptr != self.function_entry_block.ptr) return null;
        const raw_arg_number = mlir.mlirBlockArgumentGetArgNumber(value);
        if (raw_arg_number < 0) return null;
        return @intCast(raw_arg_number);
    }

    fn typeRefFromValue(self: *Collector, value: mlir.MlirValue) !obligation.TypeRef {
        if (mlir.oraValueIsNull(value)) return self.unknownTypeRef();
        return self.typeRefFromMlirType(mlir.oraValueGetType(value));
    }

    fn canonicalTypeRefForValue(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
    ) !obligation.TypeRef {
        if (self.use_loop_value_bindings) if (self.loopVariableForValue(value)) |variable| {
            if (signed_override) |signed| {
                const info = typeRefIntegerInfo(variable.ty) orelse return self.unknownTypeRef();
                if (info.signed != signed) return self.unknownTypeRef();
            }
            return variable.ty;
        };
        if (signed_override != null) {
            return self.typeRefFromValueWithIntegerSignedness(value, signed_override);
        }
        if (mlir.mlirValueIsABlockArgument(value)) {
            if (self.functionEntryBlockArgumentNumber(value)) |arg_number| {
                if (arg_number < self.function_param_types.len) {
                    return self.function_param_types[arg_number];
                }
            }
        }
        return self.typeRefFromValue(value);
    }

    fn typeRefFromValueWithIntegerSignedness(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
    ) !obligation.TypeRef {
        if (mlir.oraValueIsNull(value)) return self.unknownTypeRef();
        return self.typeRefFromMlirTypeWithIntegerSignedness(mlir.oraValueGetType(value), signed_override);
    }

    fn typeRefFromMlirTypeWithIntegerSignedness(
        self: *Collector,
        ty: mlir.MlirType,
        signed_override: ?bool,
    ) !obligation.TypeRef {
        if (signed_override) |signed| {
            if (integerTypeWidth(ty)) |width| return self.integerTypeRef(width, signed);
        }
        return self.typeRefFromMlirType(ty);
    }

    fn typeRefFromMlirType(self: *Collector, ty: mlir.MlirType) !obligation.TypeRef {
        if (mlir.oraTypeIsNull(ty)) return self.unknownTypeRef();

        const refinement_base = mlir.oraRefinementTypeGetBaseType(ty);
        if (!mlir.oraTypeIsNull(refinement_base)) return self.typeRefFromMlirType(refinement_base);

        if (mlir.oraTypeIsAddressType(ty)) return self.builtinTypeRef(.address);

        if (integerTypeWidth(ty)) |width| {
            return self.integerTypeRef(width, mlirIntegerTypeIsSigned(ty));
        }

        return self.printedTypeRef(ty);
    }

    fn integerTypeRef(self: *Collector, width: u32, signed: bool) !obligation.TypeRef {
        if (width == 1) return self.builtinTypeRef(.bool);
        if (width <= std.math.maxInt(u16)) {
            if (type_builtin.lookupIntegerBuiltin(signed, @intCast(width))) |spec| {
                return .{ .compiler_type_id = spec.comptime_type_id };
            }
        }
        return .{ .spelling = try std.fmt.allocPrint(
            self.allocator,
            "{c}{d}",
            .{ if (signed) @as(u8, 'i') else @as(u8, 'u'), width },
        ) };
    }

    fn builtinTypeRef(_: *Collector, id: type_builtin.BuiltinTypeId) obligation.TypeRef {
        return .{ .compiler_type_id = type_builtin.lookupBuiltinById(id).comptime_type_id };
    }

    fn unknownTypeRef(self: *Collector) !obligation.TypeRef {
        return .{ .spelling = try self.allocator.dupe(u8, "<unknown>") };
    }

    const MlirPrintCollector = struct {
        allocator: std.mem.Allocator,
        buffer: std.ArrayList(u8),
        failed: bool = false,
    };

    fn printMlirChunk(value: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
        const collector: *MlirPrintCollector = @ptrCast(@alignCast(user_data orelse return));
        if (value.data == null or value.length == 0) return;
        collector.buffer.appendSlice(collector.allocator, value.data[0..value.length]) catch {
            collector.failed = true;
        };
    }

    fn printedTypeRef(self: *Collector, ty: mlir.MlirType) !obligation.TypeRef {
        var collector = MlirPrintCollector{
            .allocator = self.allocator,
            .buffer = .empty,
        };
        errdefer collector.buffer.deinit(self.allocator);
        mlir.mlirTypePrint(ty, printMlirChunk, &collector);
        if (collector.failed) {
            try self.addBlockingDiagnostic(.missing_type, "failed to collect printed MLIR type");
            return self.unknownTypeRef();
        }
        return .{ .spelling = try collector.buffer.toOwnedSlice(self.allocator) };
    }

    const BinaryTypeSource = union(enum) {
        operand_signedness,
        predicate_signedness,
        explicit_signedness: bool,
    };

    fn binaryTermFromOperands(
        self: *Collector,
        op: mlir.MlirOperation,
        binary_op: obligation.BinaryOp,
        type_source: BinaryTypeSource,
    ) anyerror!?obligation.TermId {
        const signed_override = binaryOperandSignednessOverride(binary_op, type_source);
        const lhs = (try self.termFromValueWithIntegerSignedness(
            mlir.oraOperationGetOperand(op, 0),
            signed_override,
        )) orelse return null;
        const rhs = (try self.termFromValueWithIntegerSignedness(
            mlir.oraOperationGetOperand(op, 1),
            signed_override,
        )) orelse return null;
        const ty = try self.binaryTermTypeRef(op, binary_op, type_source);
        return try self.addTerm(.{ .binary = .{
            .op = binary_op,
            .lhs = lhs,
            .rhs = rhs,
            .ty = ty,
        } });
    }

    fn binaryTermTypeRef(
        self: *Collector,
        op: mlir.MlirOperation,
        binary_op: obligation.BinaryOp,
        type_source: BinaryTypeSource,
    ) !?obligation.TypeRef {
        if (mlir.oraOperationGetNumOperands(op) < 1) return null;
        const lhs = mlir.oraOperationGetOperand(op, 0);
        const lhs_type = mlir.oraValueGetType(lhs);
        const width = integerTypeWidth(lhs_type) orelse return null;
        const signed_override = switch (type_source) {
            .operand_signedness => mlirIntegerTypeIsSigned(lhs_type),
            .predicate_signedness => binaryOpSignedness(binary_op),
            .explicit_signedness => |value| value,
        };
        const preserve_compiler_identity = switch (type_source) {
            .predicate_signedness => signed_override == null,
            else => false,
        };
        if (preserve_compiler_identity) {
            return try self.canonicalTypeRefForValue(lhs, null);
        }
        return try self.integerTypeRef(width, signed_override orelse mlirIntegerTypeIsSigned(lhs_type));
    }

    fn validateOraCmpPredicateTypes(
        self: *Collector,
        op: mlir.MlirOperation,
        predicate: []const u8,
        binary_op: obligation.BinaryOp,
    ) !bool {
        const expected_signed = binaryOpSignedness(binary_op) orelse return true;
        const lhs_type = mlir.oraValueGetType(mlir.oraOperationGetOperand(op, 0));
        const rhs_type = mlir.oraValueGetType(mlir.oraOperationGetOperand(op, 1));
        const lhs_width = integerTypeWidth(lhs_type) orelse {
            try self.addBlockingDiagnosticFmt(.missing_type, "ora.cmp '{s}' has non-integer lhs type", .{predicate});
            return false;
        };
        const rhs_width = integerTypeWidth(rhs_type) orelse {
            try self.addBlockingDiagnosticFmt(.missing_type, "ora.cmp '{s}' has non-integer rhs type", .{predicate});
            return false;
        };
        if (lhs_width != rhs_width) {
            try self.addBlockingDiagnosticFmt(.missing_type, "ora.cmp '{s}' operand widths differ: lhs={d}, rhs={d}", .{
                predicate,
                lhs_width,
                rhs_width,
            });
            return false;
        }
        const lhs_signed = mlirIntegerTypeIsSigned(lhs_type);
        const rhs_signed = mlirIntegerTypeIsSigned(rhs_type);
        if (lhs_signed != expected_signed or rhs_signed != expected_signed) {
            try self.addBlockingDiagnosticFmt(
                .comparison_signedness_mismatch,
                "ora.cmp '{s}' predicate signedness does not match operand types",
                .{predicate},
            );
            return false;
        }
        return true;
    }

    fn quantifiedTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        var quantifier_view = stringAttrView(op, "quantifier") orelse return null;
        const quantifier = quantifier_map.get(quantifier_view.slice()) orelse return null;
        const variable = try self.stringAttr(op, "variable") orelse return null;
        const variable_type = try self.stringAttr(op, "variable_type") orelse return null;
        const operand_count = mlir.oraOperationGetNumOperands(op);
        if (operand_count == 0) return null;

        const binder_depth = self.active_binders.items.len;
        try self.active_binders.append(self.allocator, .{
            .name = variable,
            .ty = .{ .spelling = variable_type },
        });
        defer self.active_binders.shrinkRetainingCapacity(binder_depth);

        const has_condition = operand_count >= 2;
        const condition = if (has_condition)
            (try self.termFromValue(mlir.oraOperationGetOperand(op, 0))) orelse return null
        else
            null;
        const body_operand_index: usize = if (has_condition) 1 else 0;
        const body = (try self.termFromValue(mlir.oraOperationGetOperand(op, body_operand_index))) orelse return null;
        return try self.addTerm(.{ .quantified = .{
            .quantifier = quantifier,
            .binder = .{
                .name = variable,
                .ty = .{ .spelling = variable_type },
            },
            .condition = condition,
            .body = body,
        } });
    }

    fn wrapFunctionParamsWithOwnerAssumptions(
        self: *Collector,
        term: obligation.TermId,
        owner: obligation.Owner,
    ) anyerror!obligation.TermId {
        const condition = try self.combinedAssumptionConditionForOwner(owner);
        var current = term;
        var condition_available = condition;
        var index = self.function_param_names.len;
        while (index != 0) {
            index -= 1;
            const param_id = self.freeVarIdForFunctionParam(index);
            const param_ty = if (index < self.function_param_types.len)
                self.function_param_types[index]
            else
                try self.unknownTypeRef();
            current = try self.bindFreeVariableInTerm(current, param_id, self.function_param_names[index], param_ty);
            if (condition_available) |condition_term| {
                condition_available = try self.bindFreeVariableInTerm(condition_term, param_id, self.function_param_names[index], param_ty);
            }
            current = try self.addTerm(.{ .quantified = .{
                .quantifier = .forall,
                .binder = .{
                    .name = self.function_param_names[index],
                    .ty = param_ty,
                    .origin = .function_param,
                },
                .condition = condition_available,
                .body = current,
            } });
            condition_available = null;
        }
        return current;
    }

    fn bindFreeVariableInTerm(
        self: *Collector,
        term: obligation.TermId,
        free_id: obligation.FreeVarId,
        name: []const u8,
        ty: obligation.TypeRef,
    ) anyerror!obligation.TermId {
        return self.bindFreeVariableInTermAtDepth(term, free_id, name, ty, 0);
    }

    fn bindFreeVariableInTermAtDepth(
        self: *Collector,
        term_id: obligation.TermId,
        free_id: obligation.FreeVarId,
        name: []const u8,
        ty: obligation.TypeRef,
        binder_depth: u32,
    ) anyerror!obligation.TermId {
        if (term_id >= self.terms.items.len) return error.InvalidTermReference;
        return switch (self.terms.items[term_id]) {
            .bool_lit,
            .int_lit,
            .result,
            .place_read,
            => term_id,
            .variable => |variable| switch (variable) {
                .free => |free| {
                    if (!obligation.freeVarIdEql(free.id, free_id)) return term_id;
                    return try self.addTerm(.{ .variable = .{ .bound = .{
                        .index = binder_depth,
                        .name = name,
                        .ty = free.ty orelse ty,
                        .region = free.region,
                    } } });
                },
                .bound => term_id,
            },
            .old => |operand| {
                const rebound = try self.bindFreeVariableInTermAtDepth(operand, free_id, name, ty, binder_depth);
                if (rebound == operand) return term_id;
                return try self.addTerm(.{ .old = rebound });
            },
            .unary => |unary| {
                const operand = try self.bindFreeVariableInTermAtDepth(unary.operand, free_id, name, ty, binder_depth);
                if (operand == unary.operand) return term_id;
                return try self.addTerm(.{ .unary = .{ .op = unary.op, .operand = operand } });
            },
            .binary => |binary| {
                const lhs = try self.bindFreeVariableInTermAtDepth(binary.lhs, free_id, name, ty, binder_depth);
                const rhs = try self.bindFreeVariableInTermAtDepth(binary.rhs, free_id, name, ty, binder_depth);
                if (lhs == binary.lhs and rhs == binary.rhs) return term_id;
                return try self.addTerm(.{ .binary = .{
                    .op = binary.op,
                    .lhs = lhs,
                    .rhs = rhs,
                    .ty = binary.ty,
                } });
            },
            .refinement_predicate => |predicate| {
                const value = try self.bindFreeVariableInTermAtDepth(predicate.value, free_id, name, ty, binder_depth);
                var changed = value != predicate.value;
                if (predicate.args.len == 0) {
                    if (!changed) return term_id;
                    return try self.addTerm(.{ .refinement_predicate = .{
                        .name = predicate.name,
                        .value = value,
                    } });
                }
                const args = try self.allocator.alloc(obligation.TermId, predicate.args.len);
                for (predicate.args, 0..) |arg, arg_index| {
                    args[arg_index] = try self.bindFreeVariableInTermAtDepth(arg, free_id, name, ty, binder_depth);
                    changed = changed or args[arg_index] != arg;
                }
                if (!changed) return term_id;
                return try self.addTerm(.{ .refinement_predicate = .{
                    .name = predicate.name,
                    .value = value,
                    .args = args,
                } });
            },
            .quantified => |quantified| {
                const condition = if (quantified.condition) |condition_id|
                    try self.bindFreeVariableInTermAtDepth(condition_id, free_id, name, ty, binder_depth + 1)
                else
                    null;
                const body = try self.bindFreeVariableInTermAtDepth(quantified.body, free_id, name, ty, binder_depth + 1);
                if (condition == quantified.condition and body == quantified.body) return term_id;
                return try self.addTerm(.{ .quantified = .{
                    .quantifier = quantified.quantifier,
                    .binder = quantified.binder,
                    .condition = condition,
                    .body = body,
                } });
            },
        };
    }

    fn boundVariableForName(self: *Collector, name: []const u8) ?obligation.BoundVarRef {
        var index = self.active_binders.items.len;
        var depth: u32 = 0;
        while (index != 0) {
            index -= 1;
            const binder = self.active_binders.items[index];
            if (std.mem.eql(u8, binder.name, name)) return .{
                .index = depth,
                .name = binder.name,
                .ty = binder.ty,
                .region = binder.region,
            };
            depth += 1;
        }
        return null;
    }

    fn freeVarIdForFunctionParam(self: *Collector, index: usize) obligation.FreeVarId {
        if (index < self.function_param_binding_ids.len) return self.function_param_binding_ids[index];
        return .{
            .file_id = synthetic_file_id,
            .pattern_id = @intCast(index),
        };
    }

    fn combinedAssumptionConditionForOwner(self: *Collector, owner: obligation.Owner) !?obligation.TermId {
        var current: ?obligation.TermId = null;
        for (self.assumptions.items) |item| {
            if (!ownerEqual(item.owner, owner)) continue;
            const formula = item.formula orelse continue;
            if (formula != .term) continue;
            current = if (current) |lhs|
                try self.addTerm(.{ .binary = .{ .op = .and_, .lhs = lhs, .rhs = formula.term } })
            else
                formula.term;
        }
        return current;
    }

    fn addTerm(self: *Collector, term: obligation.Term) !obligation.TermId {
        const id: obligation.TermId = @intCast(self.terms.items.len);
        try self.terms.append(self.allocator, term);
        return id;
    }

    fn addMissingFormulaDiagnostic(
        self: *Collector,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        _ = symbol;
        _ = ordinal;
        try self.addBlockingDiagnosticFmt(.missing_formula, "{s} missing condition operand", .{op_name});
    }

    fn addBlockingDiagnostic(
        self: *Collector,
        kind: obligation.DiagnosticKind,
        message: []const u8,
    ) !void {
        try self.diagnostics.append(self.allocator, .{
            .kind = kind,
            .source = .generated(),
            .message = message,
            .blocks_artifacts = true,
        });
    }

    fn addBlockingDiagnosticFmt(
        self: *Collector,
        kind: obligation.DiagnosticKind,
        comptime fmt: []const u8,
        args: anytype,
    ) !void {
        try self.addBlockingDiagnostic(kind, try std.fmt.allocPrint(self.allocator, fmt, args));
    }

    fn recordSourceFactBinding(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        ordinal: u32,
        obligation_start: usize,
        assumption_start: usize,
        query_start: usize,
    ) !void {
        const source_fact_id = (try self.u32Attr(op, "ora.source_fact_id")) orelse return;
        const kind = try self.stringAttr(op, "ora.source_fact_kind") orelse {
            try self.addBlockingDiagnostic(.missing_type, "source-formal operation is missing ora.source_fact_kind");
            return;
        };
        const roles = try self.stringArrayAttr(op, "ora.source_fact_roles") orelse {
            try self.addBlockingDiagnostic(.missing_type, "source-formal operation is missing ora.source_fact_roles");
            return;
        };
        if (roles.len == 0) {
            try self.addBlockingDiagnostic(.missing_type, "source-formal operation has an empty ora.source_fact_roles");
            return;
        }

        const obligation_ids = try self.allocator.alloc(obligation.Id, self.obligations.items.len - obligation_start);
        for (self.obligations.items[obligation_start..], obligation_ids) |row, *id| id.* = row.id;
        const assumption_ids = try self.allocator.alloc(obligation.Id, self.assumptions.items.len - assumption_start);
        for (self.assumptions.items[assumption_start..], assumption_ids) |row, *id| id.* = row.id;
        const query_ids = try self.allocator.alloc(obligation.Id, self.queries.items.len - query_start);
        for (self.queries.items[query_start..], query_ids) |row, *id| id.* = row.id;

        var runtime_check_present = false;
        for (roles) |role| {
            if (!std.mem.eql(u8, role, "runtime_condition")) continue;
            runtime_check_present = std.mem.eql(u8, op_name, "ora.assert") or
                std.mem.eql(u8, op_name, "cf.assert") or
                std.mem.eql(u8, op_name, "ora.refinement_guard");
        }
        const runtime_check_ids = if (runtime_check_present) blk: {
            const source_op_id = sourceOpId(op) orelse {
                try self.addBlockingDiagnostic(.missing_type, "runtime source-formal operation has no live operation identity");
                break :blk &.{};
            };
            const producer_id = std.math.add(u32, ordinal, 1) catch
                return error.SourceAccountingRuntimeCheckIdOverflow;
            try self.runtime_check_producers.append(self.allocator, .{
                .id = producer_id,
                .source_op_id = source_op_id,
                .op_ordinal = ordinal,
                .source = try self.sourceForOperation(op),
            });
            const ids = try self.allocator.alloc(u32, 1);
            ids[0] = producer_id;
            break :blk ids;
        } else &.{};
        var state_effect_present = false;
        for (roles) |role| {
            if (!std.mem.eql(u8, role, "state_directive")) continue;
            state_effect_present = std.mem.eql(u8, op_name, "ora.havoc");
        }
        const state_effect_ids = if (state_effect_present) blk: {
            const source_op_id = sourceOpId(op) orelse {
                try self.addBlockingDiagnostic(.missing_type, "state-effect source-formal operation has no live operation identity");
                break :blk &.{};
            };
            const producer_id = std.math.add(u32, ordinal, 1) catch
                return error.SourceAccountingStateEffectIdOverflow;
            try self.state_effect_producers.append(self.allocator, .{
                .id = producer_id,
                .source_op_id = source_op_id,
                .op_ordinal = ordinal,
                .source = try self.sourceForOperation(op),
            });
            const ids = try self.allocator.alloc(u32, 1);
            ids[0] = producer_id;
            break :blk ids;
        } else &.{};
        const source = try self.sourceForOperation(op);
        const is_contract_invariant = std.mem.eql(u8, kind, "contract_invariant");
        const module_path = self.function_source_module_path orelse
            if (is_contract_invariant) source.file else null;
        const owner_key = self.function_source_owner_key orelse
            if (is_contract_invariant) self.contract_source_owner_key else null;
        const activation = self.function_source_activation orelse
            if (is_contract_invariant and owner_key != null) @as(?[]const u8, "runtime_body") else null;
        try self.source_fact_bindings.append(self.allocator, .{
            .source_op_id = sourceOpId(op) orelse return,
            .op_ordinal = ordinal,
            .source = source,
            .source_fact_id = source_fact_id,
            .kind = kind,
            .roles = roles,
            .module_path = module_path,
            .owner_key = owner_key,
            .template_activation = activation,
            .runtime_symbol = self.function_runtime_symbol,
            .specialization_bindings = self.function_source_specialization_bindings,
            .trait_implementation = self.function_source_trait_implementation,
            .trait_method = self.function_source_trait_method,
            .obligation_ids = obligation_ids,
            .assumption_ids = assumption_ids,
            .query_ids = query_ids,
            .runtime_check_ids = runtime_check_ids,
            .runtime_check_present = runtime_check_present,
            .state_effect_ids = state_effect_ids,
            .state_effect_present = state_effect_present,
        });
    }

    fn recordFunctionModifiesBindings(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        obligation_start: usize,
    ) !void {
        const source_fact_ids = (try self.u32ArrayAttr(op, "ora.modifies_source_fact_ids")) orelse return;
        if (source_fact_ids.len == 0) {
            try self.addBlockingDiagnostic(.missing_type, "ora.modifies_source_fact_ids is empty");
            return;
        }

        var frame_ids: std.ArrayList(obligation.Id) = .empty;
        for (self.obligations.items[obligation_start..]) |item| {
            if (item.kind != .effect_frame) continue;
            const origin = switch (item.origin) {
                .mlir_op => |origin| origin,
                else => continue,
            };
            if (!std.mem.eql(u8, origin.op_name, op_name)) continue;
            if (origin.ordinal != ordinal) continue;
            if (!optionalStringEqual(origin.symbol, symbol)) continue;
            try frame_ids.append(self.allocator, item.id);
        }
        if (frame_ids.items.len == 0) {
            try self.addBlockingDiagnostic(.missing_effect_path, "modifies source facts have no function frame result");
            return;
        }
        const owned_frame_ids = try frame_ids.toOwnedSlice(self.allocator);
        for (source_fact_ids) |source_fact_id| {
            try self.source_fact_bindings.append(self.allocator, .{
                .source_op_id = sourceOpId(op) orelse return,
                .op_ordinal = ordinal,
                .source = try self.sourceForOperation(op),
                .source_fact_id = source_fact_id,
                .kind = "modifies",
                .roles = &.{"frame_directive"},
                .module_path = self.function_source_module_path,
                .owner_key = self.function_source_owner_key,
                .template_activation = self.function_source_activation,
                .runtime_symbol = self.function_runtime_symbol,
                .specialization_bindings = self.function_source_specialization_bindings,
                .trait_implementation = self.function_source_trait_implementation,
                .trait_method = self.function_source_trait_method,
                .obligation_ids = owned_frame_ids,
            });
        }
    }

    fn u32Attr(_: *Collector, op: mlir.MlirOperation, name: []const u8) !?u32 {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;
        const text = mlir.oraIntegerAttrGetValueString(attr);
        defer if (text.data != null) mlir.oraStringRefFree(text);
        if (text.data == null) return error.InvalidSourceFactIdAttribute;
        return std.fmt.parseInt(u32, text.data[0..text.length], 10) catch
            return error.InvalidSourceFactIdAttribute;
    }

    fn u32ArrayAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const u32 {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;
        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        const values = try self.allocator.alloc(u32, count);
        for (values, 0..) |*slot, index| {
            const element = mlir.oraArrayAttrGetElement(attr, @intCast(index));
            const text = mlir.oraIntegerAttrGetValueString(element);
            defer if (text.data != null) mlir.oraStringRefFree(text);
            if (text.data == null) return error.InvalidSourceFactIdAttribute;
            slot.* = std.fmt.parseInt(u32, text.data[0..text.length], 10) catch
                return error.InvalidSourceFactIdAttribute;
        }
        return values;
    }

    fn sourceForOperation(self: *Collector, op: mlir.MlirOperation) !obligation.SourceRef {
        if (mlir.oraOperationIsNull(op)) return .generated();
        const loc = mlir.oraOperationGetLocation(op);
        if (mlir.oraLocationIsNull(loc)) return .generated();

        const printed = mlir.oraLocationPrintToString(loc);
        defer if (printed.data != null) mlir.oraStringRefFree(printed);
        if (printed.data == null or printed.length == 0) return .generated();
        return (try sourceRefFromLocationText(self.allocator, printed.data[0..printed.length])) orelse .generated();
    }

    fn stringAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const u8 {
        var value = stringAttrView(op, name) orelse return null;
        return try self.allocator.dupe(u8, value.slice());
    }

    fn boolAttr(_: *Collector, op: mlir.MlirOperation, name: []const u8) !?bool {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;
        return mlir.oraBoolAttrGetValue(attr);
    }

    fn stringArrayAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const []const u8 {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        const values = try self.allocator.alloc([]const u8, count);
        for (values, 0..) |*slot, index| {
            const element = mlir.oraArrayAttrGetElement(attr, @intCast(index));
            var value = stringAttrValueView(element) orelse {
                try self.addBlockingDiagnosticFmt(.missing_type, "{s} contains a non-string element", .{name});
                return null;
            };
            slot.* = try self.allocator.dupe(u8, value.slice());
        }
        return values;
    }

    fn freeVarIdArrayAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const obligation.FreeVarId {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        const values = try self.allocator.alloc(obligation.FreeVarId, count);
        for (values, 0..) |*slot, index| {
            const element = mlir.oraArrayAttrGetElement(attr, @intCast(index));
            var value = stringAttrValueView(element) orelse {
                try self.addBlockingDiagnosticFmt(.missing_type, "{s} contains a non-string binding id", .{name});
                return null;
            };
            const id_text = value.slice();
            slot.* = parseFreeVarId(id_text) orelse {
                try self.addBlockingDiagnosticFmt(.missing_type, "{s} contains malformed binding id '{s}'", .{ name, id_text });
                return null;
            };
        }
        return values;
    }

    fn placeArrayAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const obligation.PlaceRef {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        const places = try self.allocator.alloc(obligation.PlaceRef, count);
        for (places, 0..) |*place, index| {
            const element = mlir.oraArrayAttrGetElement(attr, @intCast(index));
            var value = stringAttrValueView(element) orelse {
                try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} contains a non-string effect slot", .{name});
                return null;
            };
            const path = value.slice();
            place.* = self.placeFromSlotPath(path) catch |err| {
                if (err == error.InvalidEffectPath) {
                    try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} contains malformed effect slot '{s}'", .{
                        name,
                        path,
                    });
                    return null;
                }
                return err;
            };
        }
        return places;
    }

    fn placeAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?obligation.PlaceRef {
        var value = stringAttrView(op, name) orelse return null;
        const path = value.slice();
        return self.placeFromSlotPath(path) catch |err| {
            if (err == error.InvalidEffectPath) {
                try self.addBlockingDiagnosticFmt(.missing_effect_path, "{s} contains malformed effect slot '{s}'", .{ name, path });
                return null;
            }
            return err;
        };
    }

    fn placeFromSlotPath(self: *Collector, raw_path: []const u8) !obligation.PlaceRef {
        var path = raw_path;
        var region: obligation.RegionRef = .storage;
        if (std.mem.startsWith(u8, path, "transient:")) {
            region = .transient;
            path = path["transient:".len..];
        }

        const root_end = effectSlotPathRootEnd(path);
        if (root_end == 0) return error.InvalidEffectPath;
        const root = try self.allocator.dupe(u8, path[0..root_end]);

        const counts = try countEffectSlotPathSegments(path, root_end);
        const fields: [][]const u8 = if (counts.fields == 0) &.{} else try self.allocator.alloc([]const u8, counts.fields);
        const keys: []obligation.PlaceKey = if (counts.keys == 0) &.{} else try self.allocator.alloc(obligation.PlaceKey, counts.keys);
        var field_index: usize = 0;
        var key_index: usize = 0;
        var cursor = root_end;
        while (cursor < path.len) {
            switch (path[cursor]) {
                '.' => {
                    cursor += 1;
                    const start = cursor;
                    while (cursor < path.len and path[cursor] != '.' and path[cursor] != '[') : (cursor += 1) {}
                    if (start == cursor) return error.InvalidEffectPath;
                    fields[field_index] = try self.allocator.dupe(u8, path[start..cursor]);
                    field_index += 1;
                },
                '[' => {
                    cursor += 1;
                    const start = cursor;
                    while (cursor < path.len and path[cursor] != ']') : (cursor += 1) {}
                    if (cursor >= path.len) return error.InvalidEffectPath;
                    keys[key_index] = try self.placeKeyFromSegment(path[start..cursor]);
                    key_index += 1;
                    cursor += 1;
                },
                else => return error.InvalidEffectPath,
            }
        }

        return .{
            .root = root,
            .region = region,
            .fields = fields,
            .keys = keys,
        };
    }

    fn placeKeyFromSegment(self: *Collector, segment: []const u8) !obligation.PlaceKey {
        if (std.mem.startsWith(u8, segment, "param#")) {
            const index = try std.fmt.parseInt(usize, segment["param#".len..], 10);
            const id = (try self.functionParamBindingIdForPlaceKey(index, segment)) orelse return .{ .unknown = {} };
            return .{ .parameter = id };
        }
        if (std.mem.startsWith(u8, segment, "comptime_param#")) {
            return .{ .comptime_parameter = try std.fmt.parseInt(u32, segment["comptime_param#".len..], 10) };
        }
        if (std.mem.startsWith(u8, segment, "comptime_range_param#")) {
            return .{ .comptime_range_parameter = try std.fmt.parseInt(u32, segment["comptime_range_param#".len..], 10) };
        }
        if (std.mem.eql(u8, segment, "msg.sender")) return .{ .msg_sender = {} };
        if (std.mem.eql(u8, segment, "tx.origin")) return .{ .tx_origin = {} };
        if (std.mem.eql(u8, segment, "?")) return .{ .unknown = {} };
        return .{ .constant = try self.allocator.dupe(u8, segment) };
    }

    fn functionParamBindingIdForPlaceKey(
        self: *Collector,
        index: usize,
        context: []const u8,
    ) !?obligation.FreeVarId {
        if (index < self.function_param_binding_ids.len) return self.function_param_binding_ids[index];
        try self.addBlockingDiagnosticFmt(
            .missing_effect_path,
            "storage place key '{s}' references parameter {d} but ora.param_binding_ids is missing or too short",
            .{ context, index },
        );
        return null;
    }

    fn symbolForOperation(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        inherited_symbol: ?[]const u8,
    ) !?[]const u8 {
        if (!std.mem.eql(u8, op_name, "func.func")) return inherited_symbol;
        return try self.stringAttr(op, "sym_name") orelse inherited_symbol;
    }

    fn ownerFor(self: *Collector, symbol: ?[]const u8) obligation.Owner {
        if (symbol) |name| return .{ .function = .{ .name = name } };
        return self.options.owner;
    }

    fn nextId(self: *Collector) obligation.Id {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    fn nextOrdinal(self: *Collector) u32 {
        const ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        return ordinal;
    }
};

fn ownerEqual(lhs: obligation.Owner, rhs: obligation.Owner) bool {
    return switch (lhs) {
        .module => |left| switch (rhs) {
            .module => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .function => |left| switch (rhs) {
            .function => |right| std.mem.eql(u8, left.name, right.name),
            else => false,
        },
        .contract => |left| switch (rhs) {
            .contract => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .trait_method => |left| switch (rhs) {
            .trait_method => |right| std.mem.eql(u8, left.trait_name, right.trait_name) and
                std.mem.eql(u8, left.method_name, right.method_name) and
                optionalStringEqual(left.impl_name, right.impl_name),
            else => false,
        },
        .statement => |left| switch (rhs) {
            .statement => |right| left.ordinal == right.ordinal and
                std.mem.eql(u8, left.function_name, right.function_name),
            else => false,
        },
        .backend => |left| switch (rhs) {
            .backend => |right| left.component == right.component and std.mem.eql(u8, left.name, right.name),
            else => false,
        },
    };
}

fn freeVarPairMatches(
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    first: obligation.FreeVarId,
    second: obligation.FreeVarId,
) bool {
    return (obligation.freeVarIdEql(lhs, first) and obligation.freeVarIdEql(rhs, second)) or
        (obligation.freeVarIdEql(lhs, second) and obligation.freeVarIdEql(rhs, first));
}

fn typeRefIntegerInfo(ty: ?obligation.TypeRef) ?type_builtin.BuiltinIntegerInfo {
    const value = ty orelse return null;
    return switch (value) {
        .compiler_type_id => |id| blk: {
            break :blk type_builtin.integerInfoByComptimeTypeId(id);
        },
        .spelling => |name| type_builtin.integerInfoByName(name),
    };
}

fn typeRefsAreSameRegisteredInteger(lhs: ?obligation.TypeRef, rhs: ?obligation.TypeRef) bool {
    const left = typeRefIntegerInfo(lhs) orelse return false;
    const right = typeRefIntegerInfo(rhs) orelse return false;
    return left.width == right.width and left.signed == right.signed;
}

fn stringSlicesEql(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    return true;
}

fn parseFreeVarId(text: []const u8) ?obligation.FreeVarId {
    var parts = std.mem.splitScalar(u8, text, ':');
    const file_tag = parts.next() orelse return null;
    if (!std.mem.eql(u8, file_tag, "file")) return null;
    const file_text = parts.next() orelse return null;
    const pattern_tag = parts.next() orelse return null;
    if (!std.mem.eql(u8, pattern_tag, "pattern")) return null;
    const pattern_text = parts.next() orelse return null;
    if (parts.next() != null) return null;

    return .{
        .file_id = std.fmt.parseUnsigned(u32, file_text, 10) catch return null,
        .pattern_id = std.fmt.parseUnsigned(u32, pattern_text, 10) catch return null,
    };
}

fn optionalStringEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

fn operationResultIsI1(op: mlir.MlirOperation) bool {
    if (mlir.oraOperationGetNumResults(op) != 1) return false;
    const result = mlir.oraOperationGetResult(op, 0);
    if (mlir.oraValueIsNull(result)) return false;
    const ty = mlir.oraValueGetType(result);
    return integerTypeWidth(ty) == 1;
}

fn integerTypeWidth(ty: mlir.MlirType) ?u32 {
    if (mlir.oraTypeIsNull(ty)) return null;
    if (mlir.oraTypeIsIntegerType(ty)) {
        const builtin = mlir.oraTypeToBuiltin(ty);
        if (mlir.oraTypeIsNull(builtin)) return null;
        return @intCast(mlir.oraIntegerTypeGetWidth(builtin));
    }
    if (mlir.oraTypeIsAInteger(ty)) {
        return @intCast(mlir.oraIntegerTypeGetWidth(ty));
    }
    return null;
}

fn mlirTypeIsSupportedInteger(ty: mlir.MlirType) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    const refinement_base = mlir.oraRefinementTypeGetBaseType(ty);
    if (!mlir.oraTypeIsNull(refinement_base)) return mlirTypeIsSupportedInteger(refinement_base);
    const width = integerTypeWidth(ty) orelse return false;
    const signed = mlirIntegerTypeIsSigned(ty);
    return type_builtin.lookupIntegerBuiltin(signed, @intCast(width)) != null;
}

fn mlirIntegerTypeIsSigned(ty: mlir.MlirType) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    if (mlir.oraTypeIsIntegerType(ty) or mlir.oraTypeIsAInteger(ty)) {
        return mlir.oraIntegerTypeIsSigned(ty);
    }
    return false;
}

fn normalizeIntegerLiteralText(allocator: std.mem.Allocator, text: []const u8, width: ?u32) ![]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (!std.mem.startsWith(u8, trimmed, "-")) return try allocator.dupe(u8, trimmed);
    const bit_width = width orelse return try allocator.dupe(u8, trimmed);
    if (bit_width == 0 or bit_width > 256) return try allocator.dupe(u8, trimmed);

    const signed = std.fmt.parseInt(i256, trimmed, 10) catch return try allocator.dupe(u8, trimmed);
    const unsigned: u256 = @bitCast(signed);
    const normalized = if (bit_width == 256)
        unsigned
    else
        unsigned & ((@as(u256, 1) << @intCast(bit_width)) - 1);
    return try std.fmt.allocPrint(allocator, "{d}", .{normalized});
}

fn effectSlotPathRootEnd(path: []const u8) usize {
    for (path, 0..) |byte, index| {
        if (byte == '.' or byte == '[') return index;
    }
    return path.len;
}

const EffectSlotPathCounts = struct {
    fields: usize = 0,
    keys: usize = 0,
};

fn countEffectSlotPathSegments(path: []const u8, start: usize) !EffectSlotPathCounts {
    var counts: EffectSlotPathCounts = .{};
    var cursor = start;
    while (cursor < path.len) {
        switch (path[cursor]) {
            '.' => {
                cursor += 1;
                const segment_start = cursor;
                while (cursor < path.len and path[cursor] != '.' and path[cursor] != '[') : (cursor += 1) {}
                if (segment_start == cursor) return error.InvalidEffectPath;
                counts.fields += 1;
            },
            '[' => {
                cursor += 1;
                while (cursor < path.len and path[cursor] != ']') : (cursor += 1) {}
                if (cursor >= path.len) return error.InvalidEffectPath;
                counts.keys += 1;
                cursor += 1;
            },
            else => return error.InvalidEffectPath,
        }
    }
    return counts;
}

fn mlirOrigin(op_name: []const u8, symbol: ?[]const u8, ordinal: u32) obligation.Origin {
    return .{ .mlir_op = .{
        .op_name = op_name,
        .symbol = symbol,
        .ordinal = ordinal,
    } };
}

fn sourceOpId(op: mlir.MlirOperation) ?usize {
    if (mlir.oraOperationIsNull(op)) return null;
    return @intFromPtr(op.ptr);
}

fn operationName(op: mlir.MlirOperation) []const u8 {
    const name = mlir.oraOperationGetName(op);
    if (name.data == null) return "";
    return name.data[0..name.length];
}

fn functionEntryBlock(op: mlir.MlirOperation) mlir.MlirBlock {
    const region = mlir.oraOperationGetRegion(op, 0);
    if (mlir.oraRegionIsNull(region)) return std.mem.zeroes(mlir.MlirBlock);
    return mlir.oraRegionGetFirstBlock(region);
}

fn sourceStatementIndex(op: mlir.MlirOperation) ?u32 {
    if (mlir.oraOperationIsNull(op)) return null;
    const location = mlir.oraOperationGetLocation(op);
    if (mlir.oraLocationIsNull(location)) return null;
    const location_ref = mlir.oraLocationPrintToString(location);
    defer if (location_ref.data != null) mlir.oraStringRefFree(location_ref);
    if (location_ref.data == null or location_ref.length == 0) return null;

    const text = location_ref.data[0..location_ref.length];
    const marker = "ora.origin_stmt.";
    const marker_start = std.mem.indexOf(u8, text, marker) orelse return null;
    const digits = text[marker_start + marker.len ..];
    const end = std.mem.indexOfNone(u8, digits, "0123456789") orelse digits.len;
    if (end == 0) return null;
    return std.fmt.parseInt(u32, digits[0..end], 10) catch null;
}

fn loopBodyBlock(op: mlir.MlirOperation, loop_kind: obligation.LoopKind) mlir.MlirBlock {
    return switch (loop_kind) {
        .scf_while => mlir.oraScfWhileOpGetAfterBlock(op),
        .scf_for => mlir.oraScfForOpGetBodyBlock(op),
        .other => std.mem.zeroes(mlir.MlirBlock),
    };
}

fn loopVariableCount(op: mlir.MlirOperation, loop_kind: obligation.LoopKind) usize {
    const body = loopBodyBlock(op, loop_kind);
    if (mlir.oraBlockIsNull(body)) return 0;
    return @intCast(mlir.oraBlockGetNumArguments(body));
}

fn loopVariableValue(
    op: mlir.MlirOperation,
    loop_kind: obligation.LoopKind,
    index: u32,
) mlir.MlirValue {
    const body = loopBodyBlock(op, loop_kind);
    if (mlir.oraBlockIsNull(body) or index >= mlir.oraBlockGetNumArguments(body)) {
        return std.mem.zeroes(mlir.MlirValue);
    }
    return mlir.oraBlockGetArgument(body, index);
}

fn findOperationInBlock(block: mlir.MlirBlock, expected_name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraBlockIsNull(block)) return null;
    var child = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
        if (std.mem.eql(u8, operationName(child), expected_name)) return child;
    }
    return null;
}

fn collectNamedOperationsInBlock(
    allocator: std.mem.Allocator,
    operations: *std.ArrayList(mlir.MlirOperation),
    block: mlir.MlirBlock,
    expected_name: []const u8,
) !void {
    var child = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
        const child_name = operationName(child);
        if (std.mem.eql(u8, child_name, expected_name)) try operations.append(allocator, child);
        if (std.mem.eql(u8, child_name, "scf.while") or std.mem.eql(u8, child_name, "scf.for")) continue;

        const region_count = mlir.oraOperationGetNumRegions(child);
        for (0..region_count) |region_index| {
            const region = mlir.oraOperationGetRegion(child, region_index);
            var nested_block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(nested_block)) : (nested_block = mlir.oraBlockGetNextInRegion(nested_block)) {
                try collectNamedOperationsInBlock(allocator, operations, nested_block, expected_name);
            }
        }
    }
}

const LoopBodyFacts = struct {
    has_storage_write: bool = false,
    has_storage_read: bool = false,
    has_call: bool = false,
    has_external_call: bool = false,
    has_resource_operation: bool = false,
    has_break_or_continue: bool = false,
    has_error_control_flow: bool = false,
    has_nested_loop: bool = false,
    has_branching_body: bool = false,
};

fn collectLoopBodyFacts(block: mlir.MlirBlock) LoopBodyFacts {
    var facts: LoopBodyFacts = .{};
    if (!mlir.oraBlockIsNull(block)) collectLoopBodyFactsInBlock(block, &facts);
    return facts;
}

fn collectLoopBodyFactsInBlock(block: mlir.MlirBlock, facts: *LoopBodyFacts) void {
    var child = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
        const name = operationName(child);
        if (inventoryPersistentWriteOperation(child, name)) facts.has_storage_write = true;
        if (std.mem.eql(u8, name, "ora.sload") or std.mem.eql(u8, name, "ora.tload")) facts.has_storage_read = true;
        if (std.mem.eql(u8, name, "func.call") or std.mem.eql(u8, name, "ora.external_call")) facts.has_call = true;
        if (std.mem.eql(u8, name, "ora.external_call") or hasAttr(child, "ora.trusted_extern_frame")) facts.has_external_call = true;
        if (std.mem.eql(u8, name, "ora.move") or std.mem.eql(u8, name, "ora.create") or std.mem.eql(u8, name, "ora.destroy")) {
            facts.has_resource_operation = true;
        }
        if (std.mem.eql(u8, name, "ora.break") or std.mem.eql(u8, name, "ora.continue")) facts.has_break_or_continue = true;
        if (std.mem.eql(u8, name, "ora.try_stmt")) facts.has_error_control_flow = true;
        if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
            facts.has_nested_loop = true;
            continue;
        }
        if (mlir.mlirOperationGetNumSuccessors(child) > 1 or
            std.mem.eql(u8, name, "scf.if") or
            std.mem.eql(u8, name, "cf.cond_br") or
            std.mem.eql(u8, name, "ora.switch") or
            std.mem.eql(u8, name, "ora.conditional_return"))
        {
            facts.has_branching_body = true;
        }

        const region_count = mlir.oraOperationGetNumRegions(child);
        for (0..region_count) |region_index| {
            const region = mlir.oraOperationGetRegion(child, region_index);
            var nested_block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(nested_block)) : (nested_block = mlir.oraBlockGetNextInRegion(nested_block)) {
                collectLoopBodyFactsInBlock(nested_block, facts);
            }
        }
    }
}

fn appendLoopReason(
    reasons: *std.ArrayList(obligation.LoopUnsupportedReason),
    allocator: std.mem.Allocator,
    reason: obligation.LoopUnsupportedReason,
) !void {
    for (reasons.items) |existing| if (existing == reason) return;
    try reasons.append(allocator, reason);
}

fn appendFormula(
    allocator: std.mem.Allocator,
    existing: []const obligation.FormulaRef,
    value: obligation.FormulaRef,
) ![]const obligation.FormulaRef {
    const result = try allocator.alloc(obligation.FormulaRef, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[existing.len] = value;
    return result;
}

fn loopFormulaExprs(
    allocator: std.mem.Allocator,
    formulas: []const obligation.FormulaRef,
) ![]const obligation.LoopFormulaExpr {
    const result = try allocator.alloc(obligation.LoopFormulaExpr, formulas.len);
    for (formulas, result) |formula, *expression| expression.* = .{ .formula = formula };
    return result;
}

fn appendString(
    allocator: std.mem.Allocator,
    existing: []const []const u8,
    value: []const u8,
) ![]const []const u8 {
    const result = try allocator.alloc([]const u8, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendValueSource(
    allocator: std.mem.Allocator,
    existing: []const LoopValueSource,
    value: LoopValueSource,
) ![]const LoopValueSource {
    const result = try allocator.alloc(LoopValueSource, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendFormulaOverride(
    allocator: std.mem.Allocator,
    existing: []const Collector.LoopFormulaOverride,
    value: Collector.LoopFormulaOverride,
) ![]const Collector.LoopFormulaOverride {
    const result = try allocator.alloc(Collector.LoopFormulaOverride, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendLoopSummaryDependency(
    allocator: std.mem.Allocator,
    existing: []const obligation.LoopSummaryDependency,
    value: obligation.LoopSummaryDependency,
) ![]const obligation.LoopSummaryDependency {
    const result = try allocator.alloc(obligation.LoopSummaryDependency, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendLoopSafetyCheckpoint(
    allocator: std.mem.Allocator,
    existing: []const obligation.LoopSafetyCheckpoint,
    value: obligation.LoopSafetyCheckpoint,
) ![]const obligation.LoopSafetyCheckpoint {
    const result = try allocator.alloc(obligation.LoopSafetyCheckpoint, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendLoopSafetyCheckpointKey(
    allocator: std.mem.Allocator,
    existing: []const LoopSafetyCheckpointKey,
    value: LoopSafetyCheckpointKey,
) ![]const LoopSafetyCheckpointKey {
    const result = try allocator.alloc(LoopSafetyCheckpointKey, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn appendLoopTimelineEvent(
    allocator: std.mem.Allocator,
    existing: []const obligation.LoopTimelineEvent,
    payload: obligation.LoopTimelineEventPayload,
    structural_key: []const u8,
) ![]const obligation.LoopTimelineEvent {
    const result = try allocator.alloc(obligation.LoopTimelineEvent, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = .{
        .id = @intCast(existing.len),
        .structural_key = structural_key,
        .payload = payload,
    };
    return result;
}

fn appendLoopTimelineInventoryEvent(
    allocator: std.mem.Allocator,
    existing: []const LoopTimelineInventoryEvent,
    value: LoopTimelineInventoryEvent,
) ![]const LoopTimelineInventoryEvent {
    const result = try allocator.alloc(LoopTimelineInventoryEvent, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = value;
    return result;
}

fn safetyCheckpointKeysFromRows(
    allocator: std.mem.Allocator,
    checkpoints: []const obligation.LoopSafetyCheckpoint,
) ![]const LoopSafetyCheckpointKey {
    const keys = try allocator.alloc(LoopSafetyCheckpointKey, checkpoints.len);
    for (checkpoints, keys) |checkpoint, *key| {
        key.* = .{
            .structural_key = checkpoint.structural_key,
            .after_dependency_count = checkpoint.after_dependency_count,
        };
    }
    return keys;
}

fn preheaderProjectionMatchesInventory(
    projected: ProjectedLoopPreheader,
    inventory: InventoryLoopPreheader,
) bool {
    if (!stringSlicesEql(projected.operation_keys, inventory.operation_keys) or
        projected.event_inventory.len != inventory.event_inventory.len or
        !timelineMatchesInventory(projected.timeline, inventory.event_inventory)) return false;
    for (projected.event_inventory, inventory.event_inventory) |left, right| {
        if (!std.mem.eql(u8, left.structural_key, right.structural_key) or left.tag != right.tag or
            !optionalStringEql(left.formula_source, right.formula_source) or
            !optionalStringEql(left.storage_root, right.storage_root) or
            !storageProjectionInventoriesEql(left.storage_projections, right.storage_projections) or
            !optionalFreeVarEql(left.storage_result, right.storage_result) or
            !optionalStringEql(left.storage_value_source, right.storage_value_source) or
            !optionalIntegerTypeEql(left.storage_ty, right.storage_ty) or
            !optionalDependencyEql(left.summary_dependency, right.summary_dependency)) return false;
    }
    return true;
}

fn compareLoopProjection(
    projected: []const ProjectedLoopEdge,
    inventory: []const InventoryLoopEdge,
) obligation.LoopProjectionCompleteness {
    for (projected, 0..) |edge, index| {
        if (edge.edge.structural_key.len == 0) return .ambiguous_identity;
        for (projected[0..index]) |previous| {
            if (std.mem.eql(u8, previous.edge.structural_key, edge.edge.structural_key)) {
                return .ambiguous_identity;
            }
        }
    }
    for (inventory, 0..) |edge, index| {
        if (edge.structural_key.len == 0) return .ambiguous_identity;
        for (inventory[0..index]) |previous| {
            if (std.mem.eql(u8, previous.structural_key, edge.structural_key)) {
                return .ambiguous_identity;
            }
        }
    }
    if (projected.len != inventory.len) return .path_mismatch;
    for (projected, inventory) |semantic, structural| {
        if (!std.mem.eql(u8, semantic.edge.structural_key, structural.structural_key) or
            semantic.edge.target != structural.target or
            !std.mem.eql(u8, semantic.terminator_key, structural.terminator_key))
        {
            return .path_mismatch;
        }
        if (semantic.edge.next_state.len != structural.binding_source_keys.len) return .binding_mismatch;
        for (semantic.edge.next_state, structural.binding_source_keys) |binding, source_key| {
            if (!std.mem.eql(u8, binding.source_key, source_key)) return .binding_mismatch;
        }
        if (!stringSlicesEql(semantic.state_source_keys, structural.state_source_keys)) {
            return .binding_mismatch;
        }
        if (!safetyCheckpointKeysEql(
            semantic.safety_checkpoints,
            structural.safety_checkpoints,
        ) or !stringSlicesEql(semantic.effect_keys, structural.effect_keys)) {
            return .effect_mismatch;
        }
        if (!stringSlicesEql(semantic.target_keys, structural.target_keys)) {
            return .target_mismatch;
        }
        if (!stringSlicesEql(semantic.summary_dependency_keys, structural.summary_dependency_keys)) {
            return .path_mismatch;
        }
        if (!storageTimelineInventoryEql(semantic.timeline_inventory, structural.timeline_inventory) or
            !timelineMatchesInventory(semantic.edge.timeline, structural.timeline_inventory))
        {
            return .effect_mismatch;
        }
    }
    return .complete;
}

fn storageTimelineInventoryEql(
    semantic: []const LoopTimelineInventoryEvent,
    structural: []const LoopTimelineInventoryEvent,
) bool {
    var semantic_index: usize = 0;
    for (structural) |right| {
        if (right.tag != .storage_read and right.tag != .storage_write) continue;
        if (semantic_index >= semantic.len) return false;
        const left = semantic[semantic_index];
        if (!std.mem.eql(u8, left.structural_key, right.structural_key) or left.tag != right.tag or
            !optionalStringEql(left.storage_root, right.storage_root) or
            !storageProjectionInventoriesEql(left.storage_projections, right.storage_projections) or
            !optionalFreeVarEql(left.storage_result, right.storage_result) or
            !optionalStringEql(left.storage_value_source, right.storage_value_source) or
            !optionalIntegerTypeEql(left.storage_ty, right.storage_ty) or
            !optionalDependencyEql(left.summary_dependency, right.summary_dependency)) return false;
        semantic_index += 1;
    }
    return semantic_index == semantic.len;
}

fn timelineInventoryHasStorageEvent(inventory: []const LoopTimelineInventoryEvent) bool {
    for (inventory) |item| switch (item.tag) {
        .storage_read, .storage_write => return true,
        else => {},
    };
    return false;
}

fn timelineMatchesInventory(
    timeline: []const obligation.LoopTimelineEvent,
    inventory: []const LoopTimelineInventoryEvent,
) bool {
    if (timeline.len != inventory.len) return false;
    for (timeline, inventory, 0..) |event, item, index| {
        if (event.id != index or !std.mem.eql(u8, event.structural_key, item.structural_key) or
            std.meta.activeTag(event.payload) != item.tag) return false;
        switch (event.payload) {
            .storage_read => |read| if (!optionalStringEql(read.location.root, item.storage_root) or
                !storageProjectionsMatchInventory(read.location.projections, item.storage_projections) or
                !optionalFreeVarEql(read.result, item.storage_result) or
                !optionalIntegerTypeEql(read.location.value_ty, item.storage_ty)) return false,
            .storage_write => |write| if (!optionalStringEql(write.location.root, item.storage_root) or
                !storageProjectionsMatchInventory(write.location.projections, item.storage_projections) or
                !optionalIntegerTypeEql(write.location.value_ty, item.storage_ty) or
                item.storage_value_source == null) return false,
            .summary_dependency => |dependency| if (!optionalDependencyEql(dependency, item.summary_dependency)) return false,
            .path_assumption, .safety_checkpoint => {},
        }
    }
    return true;
}

fn storageProjectionInventoriesEql(
    lhs: []const LoopStorageProjectionInventory,
    rhs: []const LoopStorageProjectionInventory,
) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (left.tag != right.tag or
            !optionalStringEql(left.source_key, right.source_key) or
            !optionalStringEql(left.field, right.field) or
            !optionalIntegerTypeEql(left.ty, right.ty) or
            left.extent != right.extent) return false;
    }
    return true;
}

fn storageProjectionsMatchInventory(
    projections: []const obligation.LoopStorageProjection,
    inventory: []const LoopStorageProjectionInventory,
) bool {
    if (projections.len != inventory.len) return false;
    for (projections, inventory) |projection, item| switch (projection) {
        .field => |field| if (item.tag != .field or
            !optionalStringEql(field, item.field) or item.source_key != null or
            item.ty != null or item.extent != null) return false,
        .fixed_index => |fixed| if (item.tag != .fixed_index or
            item.source_key == null or
            !optionalIntegerTypeEql(fixed.index.ty, item.ty) or
            item.extent != fixed.extent) return false,
        .mapping_key => |key| if (item.tag != .mapping_key or
            item.source_key == null or !optionalIntegerTypeEql(key.ty, item.ty) or
            item.extent != null) return false,
    };
    return true;
}

fn optionalStringEql(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

fn optionalFreeVarEql(lhs: ?obligation.FreeVarId, rhs: ?obligation.FreeVarId) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return obligation.freeVarIdEql(lhs.?, rhs.?);
}

fn optionalIntegerTypeEql(lhs: ?obligation.TypeRef, rhs: ?obligation.TypeRef) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return switch (lhs.?) {
        .compiler_type_id => |left| switch (rhs.?) {
            .compiler_type_id => |right| left == right,
            else => false,
        },
        .spelling => |left| switch (rhs.?) {
            .spelling => |right| std.mem.eql(u8, left, right),
            else => false,
        },
    };
}

fn optionalDependencyEql(
    lhs: ?obligation.LoopSummaryDependency,
    rhs: ?obligation.LoopSummaryDependency,
) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    const left = lhs.?;
    const right = rhs.?;
    if (left.id != right.id or left.summary_id != right.summary_id or left.target != right.target or
        left.write_footprint_complete != right.write_footprint_complete or
        !std.mem.eql(u8, left.structural_key, right.structural_key) or
        !std.mem.eql(u8, &left.loop_vc_sha256, &right.loop_vc_sha256) or
        left.write_footprint.len != right.write_footprint.len) return false;
    for (left.write_footprint, right.write_footprint) |left_place, right_place| {
        if (!obligation.placeRefEql(left_place, right_place)) return false;
    }
    return true;
}

fn loopControlProjectionFailure(err: anyerror) ?obligation.LoopProjectionCompleteness {
    return switch (err) {
        error.LoopControlManifestMissing => .control_manifest_missing,
        error.LoopControlMarkerMissing => .control_marker_missing,
        error.LoopControlMarkerMalformed => .control_marker_malformed,
        error.LoopControlMarkerDuplicate => .control_marker_duplicate,
        error.LoopControlMarkerTargetMismatch => .control_marker_target_mismatch,
        else => null,
    };
}

fn validateLoopControlStateTypes(
    variables: []const obligation.LoopVariable,
    values: []const mlir.MlirValue,
) !void {
    if (values.len != variables.len) return error.LoopControlMarkerMalformed;
    for (variables, values) |variable, value| {
        const expected = typeRefIntegerInfo(variable.ty) orelse return error.LoopControlMarkerMalformed;
        const actual_width = integerTypeWidth(mlir.oraValueGetType(value)) orelse
            return error.LoopControlMarkerMalformed;
        if (actual_width != expected.width) return error.LoopControlMarkerMalformed;
    }
}

fn loopControlManifestContains(
    manifest: []const LoopControlMarker,
    kind: LoopControlKind,
    source_statement_id: u32,
) bool {
    for (manifest) |entry| {
        if (entry.kind == kind and entry.source_statement_id == source_statement_id) return true;
    }
    return false;
}

fn validateLoopControlManifestAgainstTerminals(
    comptime Terminal: type,
    manifest: []const LoopControlMarker,
    terminals: []const Terminal,
) !void {
    for (terminals) |terminal| {
        const control = terminal.control orelse continue;
        var declared = false;
        for (manifest) |entry| {
            if (entry.source_statement_id == control.source_statement_id and entry.kind == control.kind) {
                declared = true;
                break;
            }
        }
        if (!declared) return error.LoopControlMarkerMalformed;

        for (terminals) |other_terminal| {
            const other = other_terminal.control orelse continue;
            if (other.source_statement_id != control.source_statement_id or
                std.mem.eql(u8, other.marker_key, control.marker_key)) continue;
            return error.LoopControlMarkerDuplicate;
        }
    }

    for (manifest) |entry| {
        var found = false;
        for (terminals) |terminal| {
            const control = terminal.control orelse continue;
            if (control.source_statement_id == entry.source_statement_id and control.kind == entry.kind) {
                found = true;
                break;
            }
        }
        if (!found) return error.LoopControlMarkerMissing;
    }
}

fn draftHasSafetySite(draft: LoopSummaryDraft, op: mlir.MlirOperation) bool {
    const op_id = @intFromPtr(op.ptr);
    for (draft.safety_sites) |site| if (site.op_id == op_id) return true;
    return false;
}

fn draftHasTargetSite(draft: LoopSummaryDraft, op: mlir.MlirOperation) bool {
    const op_id = @intFromPtr(op.ptr);
    for (draft.target_sites) |site| if (site.op_id == op_id) return true;
    return false;
}

fn loopTargetSiteForStructuralKey(
    draft: LoopSummaryDraft,
    structural_key: []const u8,
) ?LoopTargetSite {
    for (draft.target_sites) |site| {
        if (std.mem.eql(u8, site.structural_key, structural_key)) return site;
    }
    return null;
}

fn loopControlKindFromText(text: []const u8) ?LoopControlKind {
    if (std.mem.eql(u8, text, "break")) return .break_loop;
    if (std.mem.eql(u8, text, "continue")) return .continue_loop;
    if (std.mem.eql(u8, text, "function_return")) return .function_return;
    if (std.mem.eql(u8, text, "error_return")) return .error_return;
    return null;
}

fn loopControlKindText(kind: LoopControlKind) []const u8 {
    return switch (kind) {
        .break_loop => "break",
        .continue_loop => "continue",
        .function_return => "function_return",
        .error_return => "error_return",
    };
}

fn loopControlTarget(kind: LoopControlKind) obligation.LoopControlTarget {
    return switch (kind) {
        .break_loop => .successor,
        .continue_loop => .backedge,
        .function_return => .function_return,
        .error_return => .error_return,
    };
}

fn isLoopSafetyOperation(op: mlir.MlirOperation, name: []const u8) bool {
    _ = op;
    const kind = op_kind_map.get(name) orelse return false;
    return switch (kind) {
        .assert, .cf_assert, .refinement_guard => true,
        .ensures => false,
        .requires, .invariant, .assume, .resource_move, .resource_create, .resource_destroy, .quantifier => false,
    };
}

fn isLoopTargetOperation(op: mlir.MlirOperation, name: []const u8) bool {
    _ = op;
    return op_kind_map.get(name) == .ensures;
}

fn isRegionYield(name: []const u8) bool {
    return std.mem.eql(u8, name, "scf.yield") or std.mem.eql(u8, name, "ora.yield");
}

fn isLoopControlRuntimeGuard(op: mlir.MlirOperation) !bool {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef("ora.loop_control_runtime_guard"));
    if (mlir.oraAttributeIsNull(attr)) return false;
    if (!mlir.oraBoolAttrGetValue(attr)) {
        return error.LoopControlMarkerMalformed;
    }
    return true;
}

fn validateRuntimeGuardElseRegion(op: mlir.MlirOperation) !void {
    if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumRegions(op) != 2) {
        return error.LoopControlMarkerMalformed;
    }
    const else_block = singleRegionBlock(op, 1) orelse return error.LoopControlMarkerMalformed;
    const terminator = mlir.oraBlockGetFirstOperation(else_block);
    if (mlir.oraOperationIsNull(terminator) or !isRegionYield(operationName(terminator))) {
        return error.LoopControlMarkerMalformed;
    }
    if (!mlir.oraOperationIsNull(mlir.oraOperationGetNextInBlock(terminator)) or
        mlir.oraOperationGetNumOperands(terminator) != mlir.oraOperationGetNumResults(op))
    {
        return error.LoopControlMarkerMalformed;
    }
}

fn singleRegionBlock(op: mlir.MlirOperation, region_index: usize) ?mlir.MlirBlock {
    if (region_index >= mlir.oraOperationGetNumRegions(op)) return null;
    const region = mlir.oraOperationGetRegion(op, region_index);
    if (mlir.oraRegionIsNull(region)) return null;
    const block = mlir.oraRegionGetFirstBlock(region);
    if (mlir.oraBlockIsNull(block) or !mlir.oraBlockIsNull(mlir.oraBlockGetNextInRegion(block))) return null;
    return block;
}

fn operationResultIndex(op: mlir.MlirOperation, value: mlir.MlirValue) ?usize {
    const count: usize = @intCast(mlir.oraOperationGetNumResults(op));
    for (0..count) |index| {
        if (mlir.mlirValueEqual(mlir.oraOperationGetResult(op, index), value)) return index;
    }
    return null;
}

fn denseI64ArrayAttr(op: mlir.MlirOperation, name: []const u8) ?mlir.MlirAttribute {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
    if (mlir.oraAttributeIsNull(attr)) return null;
    return attr;
}

fn switchDefaultIndex(op: mlir.MlirOperation, region_count: usize) !?usize {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef("default_case_index"));
    if (mlir.oraAttributeIsNull(attr)) return null;
    const raw = mlir.oraIntegerAttrGetValueSInt(attr);
    if (raw < 0 or raw >= region_count) return error.UnsupportedLoopControlFlow;
    return @intCast(raw);
}

fn appendId(
    allocator: std.mem.Allocator,
    existing: []const obligation.Id,
    value: obligation.Id,
) ![]const obligation.Id {
    const result = try allocator.alloc(obligation.Id, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[existing.len] = value;
    return result;
}

fn appendIdUnique(
    allocator: std.mem.Allocator,
    existing: []const obligation.Id,
    value: obligation.Id,
) ![]const obligation.Id {
    for (existing) |item| if (item == value) return existing;
    return appendId(allocator, existing, value);
}

fn replaceGenericEnsuresQueryWithCanonical(
    allocator: std.mem.Allocator,
    queries: []const obligation.VerificationQuery,
    existing: []const obligation.Id,
    canonical_query_id: obligation.Id,
) ![]const obligation.Id {
    var retained_count: usize = 0;
    var canonical_present = false;
    for (existing) |query_id| {
        if (query_id == canonical_query_id) {
            canonical_present = true;
            retained_count += 1;
            continue;
        }
        const query = obligation.findById(queries, query_id) orelse return error.InvalidLoopPostQuery;
        if (query.kind == .obligation and query.logical_role == .ensures) continue;
        retained_count += 1;
    }
    const result = try allocator.alloc(
        obligation.Id,
        retained_count + @intFromBool(!canonical_present),
    );
    var index: usize = 0;
    for (existing) |query_id| {
        if (query_id != canonical_query_id) {
            const query = obligation.findById(queries, query_id) orelse return error.InvalidLoopPostQuery;
            if (query.kind == .obligation and query.logical_role == .ensures) continue;
        }
        result[index] = query_id;
        index += 1;
    }
    if (!canonical_present) result[index] = canonical_query_id;
    return result;
}

fn formulasContainOriginValue(formulas: []const obligation.FormulaRef) bool {
    for (formulas) |formula| if (formula == .origin_value) return true;
    return false;
}

fn loopTimelineHasPersistentStorageEvent(timeline: []const obligation.LoopTimelineEvent) bool {
    for (timeline) |event| switch (event.payload) {
        .storage_read, .storage_write => return true,
        .path_assumption, .safety_checkpoint, .summary_dependency => {},
    };
    return false;
}

fn loopTimelineContainsOriginValue(timeline: []const obligation.LoopTimelineEvent) bool {
    for (timeline) |event| switch (event.payload) {
        .path_assumption => |path| for (path.formulas) |formula| {
            if (formula.formula == .origin_value) return true;
        },
        .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |formula| {
            if (formula.formula == .origin_value) return true;
        },
        .storage_write => |write| if (write.value.formula == .origin_value) return true,
        .storage_read, .summary_dependency => {},
    };
    return false;
}

fn safetyCheckpointKeysEql(
    lhs: []const LoopSafetyCheckpointKey,
    rhs: []const LoopSafetyCheckpointKey,
) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (left.after_dependency_count != right.after_dependency_count or
            !std.mem.eql(u8, left.structural_key, right.structural_key))
        {
            return false;
        }
    }
    return true;
}

fn optionalFormulaIsOriginValue(formula: ?obligation.FormulaRef) bool {
    const value = formula orelse return false;
    return value == .origin_value;
}

fn stepAssignmentsContainOriginValue(assignments: []const obligation.LoopStepAssignment) bool {
    for (assignments) |assignment| if (assignment.value == .origin_value) return true;
    return false;
}

fn loopStepAssignmentsAreScalar(
    variables: []const obligation.LoopVariable,
    assignments: []const obligation.LoopStepAssignment,
) bool {
    if (variables.len == 0) return assignments.len == 0;
    if (assignments.len != variables.len) return false;
    var previous_index: ?u32 = null;
    for (assignments) |assignment| {
        if (assignment.variable_index >= variables.len) return false;
        if (previous_index != null and previous_index.? == assignment.variable_index) return false;
        previous_index = assignment.variable_index;
    }
    return true;
}

fn operationBodyHasExternalCall(op: mlir.MlirOperation) bool {
    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (operationTreeHasExternalCall(child)) return true;
            }
        }
    }
    return false;
}

fn operationTreeHasExternalCall(op: mlir.MlirOperation) bool {
    if (mlir.oraOperationIsNull(op)) return false;
    const op_name = operationName(op);
    if (std.mem.eql(u8, op_name, "ora.external_call") or hasAttr(op, "ora.trusted_extern_frame")) return true;

    return operationBodyHasExternalCall(op);
}

fn sourceRefFromLocationText(allocator: std.mem.Allocator, text: []const u8) !?obligation.SourceRef {
    var cursor: usize = 0;
    var best: ?obligation.SourceRef = null;
    while (cursor < text.len) : (cursor += 1) {
        if (text[cursor] != '"') continue;
        const file_start = cursor + 1;
        var file_end = file_start;
        while (file_end < text.len) : (file_end += 1) {
            if (text[file_end] == '"' and (file_end == file_start or text[file_end - 1] != '\\')) break;
        }
        if (file_end >= text.len) break;

        var after = file_end + 1;
        if (after >= text.len or text[after] != ':') {
            cursor = file_end;
            continue;
        }
        after += 1;
        const line = parseU32At(text, &after) orelse {
            cursor = file_end;
            continue;
        };
        if (after >= text.len or text[after] != ':') {
            cursor = file_end;
            continue;
        }
        after += 1;
        const column = parseU32At(text, &after) orelse {
            cursor = file_end;
            continue;
        };

        best = .{
            .file = try allocator.dupe(u8, text[file_start..file_end]),
            .line = line,
            .column = column,
        };
        cursor = after;
    }
    return best;
}

fn parseU32At(text: []const u8, cursor: *usize) ?u32 {
    const start = cursor.*;
    var value: u32 = 0;
    while (cursor.* < text.len) : (cursor.* += 1) {
        const byte = text[cursor.*];
        if (byte < '0' or byte > '9') break;
        const digit: u32 = byte - '0';
        value = std.math.mul(u32, value, 10) catch return null;
        value = std.math.add(u32, value, digit) catch return null;
    }
    if (cursor.* == start) return null;
    return value;
}

fn strRef(text: []const u8) mlir.MlirStringRef {
    return .{ .data = text.ptr, .length = text.len };
}

fn hasAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
    return !mlir.oraAttributeIsNull(attr);
}

fn denseI32ArrayAttr(op: mlir.MlirOperation, name: []const u8) ?mlir.MlirAttribute {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
    if (mlir.oraAttributeIsNull(attr)) return null;
    if (mlir.oraDenseI32ArrayAttrGetNumElements(attr) == 0) return null;
    return attr;
}

const StringAttrView = struct {
    value: mlir.MlirStringRef,

    fn slice(self: StringAttrView) []const u8 {
        return self.value.data[0..self.value.length];
    }
};

fn stringAttrView(op: mlir.MlirOperation, name: []const u8) ?StringAttrView {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
    if (mlir.oraAttributeIsNull(attr)) return null;
    return stringAttrValueView(attr);
}

fn stringAttrValueView(attr: mlir.MlirAttribute) ?StringAttrView {
    const value = mlir.oraStringAttrGetValue(attr);
    if (value.data == null) return null;
    return .{ .value = value };
}

fn appendField(
    allocator: std.mem.Allocator,
    fields: []const []const u8,
    field: []const u8,
) ![]const []const u8 {
    const updated = try allocator.alloc([]const u8, fields.len + 1);
    @memcpy(updated[0..fields.len], fields);
    updated[fields.len] = field;
    return updated;
}

fn placeIsDefinitelyDisjointFromAll(place: obligation.PlaceRef, writes: []const obligation.PlaceRef) bool {
    for (writes) |write| {
        if (!obligation.placeDefinitelyDisjoint(place, write)) return false;
    }
    return true;
}

fn isTransparentValueOp(op_name: []const u8) bool {
    return transparent_value_ops.has(op_name);
}

fn inventoryPersistentWriteOperation(op: mlir.MlirOperation, op_name: []const u8) bool {
    return std.mem.eql(u8, op_name, "ora.sstore") or
        std.mem.eql(u8, op_name, "ora.tstore") or
        std.mem.eql(u8, op_name, "ora.map_store") or
        std.mem.eql(u8, op_name, "ora.storage.word_store") or
        (std.mem.eql(u8, op_name, "memref.store") and inventoryMemrefStoreTargetsPersistentRoot(op));
}

fn directRootLoopEffectSupported(
    op: mlir.MlirOperation,
    op_name: []const u8,
    effect_kind: mlir.OraOperationMemoryEffectKind,
) bool {
    _ = op;
    if (effect_kind == mlir.ORA_OPERATION_MEMORY_EFFECT_NONE) return true;
    if (std.mem.eql(u8, op_name, "ora.sload") or std.mem.eql(u8, op_name, "ora.sstore") or
        op_kind_map.has(op_name) or std.mem.eql(u8, op_name, "ora.decreases") or
        std.mem.eql(u8, op_name, "ora.increases") or std.mem.eql(u8, op_name, "ora.mload") or
        std.mem.eql(u8, op_name, "ora.mload8") or std.mem.eql(u8, op_name, "ora.mstore") or
        std.mem.eql(u8, op_name, "ora.mstore8")) return true;
    if (std.mem.eql(u8, op_name, "ora.map_get") or std.mem.eql(u8, op_name, "ora.map_store")) return true;
    if (std.mem.eql(u8, op_name, "memref.load") or std.mem.eql(u8, op_name, "memref.store")) return true;
    return false;
}

fn preheaderControlOperationUnsupported(op: mlir.MlirOperation, name: []const u8) bool {
    return std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for") or
        std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr") or
        std.mem.eql(u8, name, "ora.try_stmt") or std.mem.eql(u8, name, "ora.try_catch") or
        mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0;
}

fn preheaderProjectionFailure(err: anyerror) ?obligation.LoopUnsupportedReason {
    return switch (err) {
        error.LoopPreheaderControlUnsupported => .loop_preheader_control_unsupported,
        error.LoopPreheaderEffectUnsupported => .loop_preheader_effect_unsupported,
        error.LoopPreheaderValueUnsupported => .loop_preheader_value_unsupported,
        error.LoopPreheaderLocationUnsupported => .loop_preheader_location_unsupported,
        else => null,
    };
}

fn preheaderAssumptionOperation(name: []const u8) bool {
    return std.mem.eql(u8, name, "ora.requires") or std.mem.eql(u8, name, "ora.assume");
}

fn preheaderPureValueOperationSupported(name: []const u8) bool {
    return std.mem.eql(u8, name, "arith.constant") or std.mem.eql(u8, name, "arith.trunci") or
        std.mem.eql(u8, name, "ora.old") or isTransparentValueOp(name) or
        arithmetic_value_op_map.has(name) or arithmetic_div_rem_value_op_map.has(name) or
        std.mem.eql(u8, name, "arith.xori") or std.mem.eql(u8, name, "arith.andi") or
        std.mem.eql(u8, name, "arith.ori") or std.mem.eql(u8, name, "arith.cmpi") or
        std.mem.eql(u8, name, "ora.cmp");
}

fn isAggregateStorageCarrierLoad(op: mlir.MlirOperation) bool {
    if (!std.mem.eql(u8, operationName(op), "ora.sload") or
        mlir.oraOperationGetNumResults(op) != 1) return false;
    return isAggregateStorageValue(mlir.oraOperationGetResult(op, 0));
}

fn isAggregateStorageValue(value: mlir.MlirValue) bool {
    if (mlir.oraValueIsNull(value)) return false;
    const ty = mlir.oraValueGetType(value);
    if (mlir.oraTypeIsAMemRef(ty)) return true;
    return !mlir.oraTypeIsNull(mlir.oraMapTypeGetValueType(ty));
}

fn inventoryMemrefStoreTargetsPersistentRoot(op: mlir.MlirOperation) bool {
    if (mlir.oraOperationGetNumOperands(op) < 2) return false;
    _ = persistentStorageRootForCarrier(mlir.oraOperationGetOperand(op, 1)) catch return false;
    return true;
}

fn inventoryMemrefLoadTargetsPersistentRoot(op: mlir.MlirOperation) bool {
    if (mlir.oraOperationGetNumOperands(op) < 1) return false;
    _ = persistentStorageRootForCarrier(mlir.oraOperationGetOperand(op, 0)) catch return false;
    return true;
}

fn persistentStorageRootForCarrier(value: mlir.MlirValue) ![]const u8 {
    var current = value;
    var depth: u32 = 0;
    while (depth <= 32) : (depth += 1) {
        if (!mlir.oraValueIsAOpResult(current)) return error.LoopProjectionEffectUnsupported;
        const owner = mlir.oraOpResultGetOwner(current);
        if (mlir.oraOperationIsNull(owner)) return error.LoopProjectionEffectUnsupported;
        const owner_name = operationName(owner);
        if (std.mem.eql(u8, owner_name, "ora.sload")) {
            const root = stringAttrView(owner, "global") orelse
                return error.LoopProjectionEffectUnsupported;
            return root.slice();
        }
        if (!isTransparentValueOp(owner_name)) return error.LoopProjectionEffectUnsupported;
        if (mlir.oraOperationGetNumOperands(owner) < 1) return error.LoopProjectionEffectUnsupported;
        current = mlir.oraOperationGetOperand(owner, 0);
    }
    return error.LoopProjectionEffectUnsupported;
}

fn fixedIndexSourceValue(value: mlir.MlirValue) mlir.MlirValue {
    if (!mlir.oraValueIsAOpResult(value)) return value;
    const owner = mlir.oraOpResultGetOwner(value);
    if (mlir.oraOperationIsNull(owner) or mlir.oraOperationGetNumOperands(owner) != 1) return value;
    const name = operationName(owner);
    if (!std.mem.eql(u8, name, "arith.index_cast") and
        !std.mem.eql(u8, name, "arith.index_castui") and
        !std.mem.eql(u8, name, "arith.index_castsi")) return value;
    return mlir.oraOperationGetOperand(owner, 0);
}

fn fixedIndexExtent(
    op: mlir.MlirOperation,
    memref_operand_index: usize,
    projection_count: usize,
    projection_index: usize,
) !u64 {
    const memref = mlir.oraOperationGetOperand(op, memref_operand_index);
    const ty = mlir.oraValueGetType(memref);
    if (!mlir.oraTypeIsAMemRef(ty) or !mlir.oraShapedTypeHasStaticShape(ty)) {
        return error.LoopProjectionEffectUnsupported;
    }
    const rank: usize = @intCast(mlir.oraShapedTypeGetRank(ty));
    if (rank != projection_count or projection_index >= rank) {
        return error.LoopProjectionEffectUnsupported;
    }
    const extent = mlir.oraShapedTypeGetDimSize(ty, @intCast(projection_index));
    if (extent <= 0) return error.LoopProjectionEffectUnsupported;
    return @intCast(extent);
}

fn storageMapSourceOwner(value: mlir.MlirValue) ?mlir.MlirOperation {
    var current = value;
    var depth: u32 = 0;
    while (depth <= 32) : (depth += 1) {
        if (!mlir.oraValueIsAOpResult(current)) return null;
        const owner = mlir.oraOpResultGetOwner(current);
        if (mlir.oraOperationIsNull(owner)) return null;
        if (!isTransparentValueOp(operationName(owner))) return owner;
        if (mlir.oraOperationGetNumOperands(owner) < 1) return null;
        current = mlir.oraOperationGetOperand(owner, 0);
    }
    return null;
}

fn valueFeedsOnlyFormalFormula(value: mlir.MlirValue, depth: u32) bool {
    if (depth > 32) return false;
    var use = mlir.mlirValueGetFirstUse(value);
    if (mlir.mlirOpOperandIsNull(use)) return false;

    while (!mlir.mlirOpOperandIsNull(use)) : (use = mlir.mlirOpOperandGetNextUse(use)) {
        const user = mlir.mlirOpOperandGetOwner(use);
        if (mlir.oraOperationIsNull(user)) return false;
        const user_name = operationName(user);
        if (isFormalFormulaSink(user_name)) continue;
        if (!isFormalFormulaIntermediate(user_name)) return false;
        if (!operationResultsFeedOnlyFormalFormula(user, depth + 1)) return false;
    }
    return true;
}

fn operationResultsFeedOnlyFormalFormula(op: mlir.MlirOperation, depth: u32) bool {
    const result_count = mlir.oraOperationGetNumResults(op);
    if (result_count == 0) return false;
    var index: usize = 0;
    while (index < result_count) : (index += 1) {
        if (!valueFeedsOnlyFormalFormula(mlir.oraOperationGetResult(op, index), depth)) return false;
    }
    return true;
}

fn isFormalFormulaSink(op_name: []const u8) bool {
    return std.mem.eql(u8, op_name, "ora.requires") or
        std.mem.eql(u8, op_name, "ora.ensures") or
        std.mem.eql(u8, op_name, "ora.invariant") or
        std.mem.eql(u8, op_name, "ora.assert") or
        std.mem.eql(u8, op_name, "cf.assert") or
        std.mem.eql(u8, op_name, "ora.assume") or
        std.mem.eql(u8, op_name, "ora.refinement_guard");
}

fn isFormalFormulaIntermediate(op_name: []const u8) bool {
    return isTransparentValueOp(op_name) or
        arithmetic_value_op_map.has(op_name) or
        arithmetic_div_rem_value_op_map.has(op_name) or
        std.mem.eql(u8, op_name, "arith.xori") or
        std.mem.eql(u8, op_name, "arith.andi") or
        std.mem.eql(u8, op_name, "arith.ori") or
        std.mem.eql(u8, op_name, "arith.cmpi") or
        std.mem.eql(u8, op_name, "ora.cmp") or
        std.mem.eql(u8, op_name, "ora.quantified");
}

fn derivedFormula(op_name: []const u8, symbol: ?[]const u8, ordinal: u32) obligation.FormulaRef {
    return .{ .origin_value = .{
        .origin = mlirOrigin(op_name, symbol, ordinal),
        .kind = .derived,
    } };
}

fn cmpiPredicateToBinaryOp(predicate: i64) ?obligation.BinaryOp {
    return switch (predicate) {
        0 => .eq,
        1 => .ne,
        2 => .slt,
        3 => .sle,
        4 => .sgt,
        5 => .sge,
        6 => .lt,
        7 => .le,
        8 => .gt,
        9 => .ge,
        else => null,
    };
}

fn stringPredicateToBinaryOp(predicate: []const u8) ?obligation.BinaryOp {
    const PredicateMap = std.StaticStringMap(obligation.BinaryOp);
    const map = comptime PredicateMap.initComptime(.{
        .{ "eq", .eq },
        .{ "ne", .ne },
        .{ "neq", .ne },
        .{ "lt", .lt },
        .{ "ult", .lt },
        .{ "le", .le },
        .{ "lte", .le },
        .{ "ule", .le },
        .{ "gt", .gt },
        .{ "ugt", .gt },
        .{ "ge", .ge },
        .{ "gte", .ge },
        .{ "uge", .ge },
        .{ "slt", .slt },
        .{ "sle", .sle },
        .{ "sgt", .sgt },
        .{ "sge", .sge },
    });
    return map.get(predicate);
}

fn binaryOpSignedness(op: obligation.BinaryOp) ?bool {
    return switch (op) {
        .lt, .le, .gt, .ge => false,
        .slt, .sle, .sgt, .sge => true,
        else => null,
    };
}

fn binaryOperandSignednessOverride(op: obligation.BinaryOp, source: Collector.BinaryTypeSource) ?bool {
    return switch (source) {
        .operand_signedness => null,
        .predicate_signedness => binaryOpSignedness(op),
        .explicit_signedness => |signed| signed,
    };
}

const QuantifierBinderClassification = struct {
    sort: obligation.QuantifierBinderSort,
    fragment: obligation.VerificationQueryFragment,
    degradation: ?obligation.QuantifierDegradation = null,
};

fn resourceProperties(op: obligation.ResourceOperation) []const obligation.ResourceProperty {
    return switch (op) {
        .move => &resource_move_properties,
        .create => &resource_create_properties,
        .destroy => &resource_destroy_properties,
    };
}

fn resourcePropertyBit(property: obligation.ResourceProperty) u32 {
    const shift: u5 = @intCast(@intFromEnum(property));
    return @as(u32, 1) << shift;
}

fn resourcePropertyMask(properties: []const obligation.ResourceProperty) u32 {
    var mask: u32 = 0;
    for (properties) |property| mask |= resourcePropertyBit(property);
    return mask;
}

fn expectedResourcePropertyMask(op: obligation.ResourceOperation) u32 {
    return switch (op) {
        .move => resourcePropertyBit(.amount_non_negative) |
            resourcePropertyBit(.source_sufficient) |
            resourcePropertyBit(.destination_no_overflow) |
            resourcePropertyBit(.same_place_identity) |
            resourcePropertyBit(.conservation),
        .create => resourcePropertyBit(.amount_non_negative) |
            resourcePropertyBit(.destination_no_overflow),
        .destroy => resourcePropertyBit(.amount_non_negative) |
            resourcePropertyBit(.source_sufficient),
    };
}

fn resourceSiteComplete(site: Collector.ResourceCompletenessSite) bool {
    return site.emitted_mask == site.expected_mask;
}

test "resource goal completeness masks pin every operation property set" {
    try std.testing.expectEqual(
        expectedResourcePropertyMask(.move),
        resourcePropertyMask(&resource_move_properties),
    );
    try std.testing.expectEqual(
        expectedResourcePropertyMask(.create),
        resourcePropertyMask(&resource_create_properties),
    );
    try std.testing.expectEqual(
        expectedResourcePropertyMask(.destroy),
        resourcePropertyMask(&resource_destroy_properties),
    );

    const missing_move_conservation = expectedResourcePropertyMask(.move) & ~resourcePropertyBit(.conservation);
    try std.testing.expect(!resourceSiteComplete(.{
        .op = .move,
        .domain = "TokenUnit",
        .ordinal = 0,
        .expected_mask = expectedResourcePropertyMask(.move),
        .emitted_mask = missing_move_conservation,
    }));
}

test "resource goal completeness failures use dedicated diagnostic kind" {
    {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var collector = Collector{ .allocator = arena.allocator(), .options = .{} };

        try collector.markResourcePropertyEmitted(0, .amount_non_negative);

        try std.testing.expectEqual(@as(usize, 1), collector.diagnostics.items.len);
        try std.testing.expectEqual(obligation.DiagnosticKind.incomplete_resource_goals, collector.diagnostics.items[0].kind);
    }
    {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
        const site = try collector.recordResourceSite(.create, "TokenUnit", 7);

        try collector.markResourcePropertyEmitted(site, .source_sufficient);

        try std.testing.expectEqual(@as(usize, 1), collector.diagnostics.items.len);
        try std.testing.expectEqual(obligation.DiagnosticKind.incomplete_resource_goals, collector.diagnostics.items[0].kind);
    }
    {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
        const site = try collector.recordResourceSite(.destroy, "TokenUnit", 8);

        try collector.markResourcePropertyEmitted(site, .amount_non_negative);
        try collector.markResourcePropertyEmitted(site, .amount_non_negative);

        try std.testing.expectEqual(@as(usize, 1), collector.diagnostics.items.len);
        try std.testing.expectEqual(obligation.DiagnosticKind.incomplete_resource_goals, collector.diagnostics.items[0].kind);
    }
    {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
        const site = try collector.recordResourceSite(.move, "TokenUnit", 9);

        try collector.markResourcePropertyEmitted(site, .amount_non_negative);
        try collector.markResourcePropertyEmitted(site, .source_sufficient);
        try collector.markResourcePropertyEmitted(site, .destination_no_overflow);
        try collector.markResourcePropertyEmitted(site, .same_place_identity);
        try collector.verifyResourceGoalCompleteness();

        try std.testing.expectEqual(@as(usize, 1), collector.diagnostics.items.len);
        try std.testing.expectEqual(obligation.DiagnosticKind.incomplete_resource_goals, collector.diagnostics.items[0].kind);
    }
}

test "independent loop inventory comparison fails closed by witness class" {
    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const binding = obligation.LoopStepAssignment{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "module-op:7/result:0",
        .value = .{ .term = 0 },
    };
    const safety_checkpoint = LoopSafetyCheckpointKey{
        .structural_key = "op:6/ora.assert",
        .after_dependency_count = 0,
    };
    const semantic = ProjectedLoopEdge{
        .edge = .{
            .id = 0,
            .structural_key = "body/path/yield",
            .next_state = &.{binding},
            .target = .backedge,
        },
        .terminator_key = "op:8/scf.yield",
        .safety_checkpoints = &.{safety_checkpoint},
        .effect_keys = &.{"op:5/ora.sload"},
        .state_source_keys = &.{"module-op:4/result:0<-module-op:3/result:0"},
    };
    const inventory = InventoryLoopEdge{
        .structural_key = "body/path/yield",
        .target = .backedge,
        .terminator_key = "op:8/scf.yield",
        .binding_source_keys = &.{"module-op:7/result:0"},
        .safety_checkpoints = &.{safety_checkpoint},
        .effect_keys = &.{"op:5/ora.sload"},
        .state_source_keys = &.{"module-op:4/result:0<-module-op:3/result:0"},
    };

    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.complete,
        compareLoopProjection(&.{semantic}, &.{inventory}),
    );

    var wrong_binding = inventory;
    wrong_binding.binding_source_keys = &.{"module-op:9/result:0"};
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.binding_mismatch,
        compareLoopProjection(&.{semantic}, &.{wrong_binding}),
    );

    var wrong_source_trace = inventory;
    wrong_source_trace.state_source_keys = &.{"module-op:4/result:0<-module-op:2/result:0"};
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.binding_mismatch,
        compareLoopProjection(&.{semantic}, &.{wrong_source_trace}),
    );

    var wrong_effect = inventory;
    wrong_effect.effect_keys = &.{};
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{semantic}, &.{wrong_effect}),
    );

    var wrong_safety_stage = inventory;
    wrong_safety_stage.safety_checkpoints = &.{.{
        .structural_key = "op:6/ora.assert",
        .after_dependency_count = 1,
    }};
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{semantic}, &.{wrong_safety_stage}),
    );

    var wrong_path = inventory;
    wrong_path.terminator_key = "op:9/scf.yield";
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.path_mismatch,
        compareLoopProjection(&.{semantic}, &.{wrong_path}),
    );

    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.path_mismatch,
        compareLoopProjection(&.{semantic}, &.{}),
    );
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.ambiguous_identity,
        compareLoopProjection(&.{ semantic, semantic }, &.{ inventory, inventory }),
    );
}

test "direct-root timeline inventory compares every storage event field" {
    const ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const read_id: obligation.FreeVarId = .{ .file_id = Collector.storage_read_file_id, .pattern_id = 7 };
    const footprint = [_]obligation.PlaceRef{.{ .root = "counter", .region = .storage }};
    const dependency: obligation.LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "op:10/scf.while",
        .summary_id = 99,
        .loop_vc_sha256 = [_]u8{1} ** 32,
        .write_footprint = &footprint,
        .write_footprint_complete = true,
        .target = .successor,
    };
    const path_formulas = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const timeline = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "body/path", .payload = .{
            .path_assumption = .{ .formulas = &path_formulas },
        } },
        .{ .id = 1, .structural_key = "op:7/ora.sload", .payload = .{ .storage_read = .{
            .result = read_id,
            .location = .{ .root = "counter", .value_ty = ty },
        } } },
        .{ .id = 2, .structural_key = "op:9/ora.sstore", .payload = .{ .storage_write = .{
            .location = .{ .root = "counter", .value_ty = ty },
            .value = .{ .formula = .{ .term = 1 }, .read_results = &.{read_id} },
        } } },
        .{ .id = 3, .structural_key = dependency.structural_key, .payload = .{ .summary_dependency = dependency } },
    };
    const rows = [_]LoopTimelineInventoryEvent{
        .{ .structural_key = "body/path", .tag = .path_assumption },
        .{
            .structural_key = "op:7/ora.sload",
            .tag = .storage_read,
            .storage_root = "counter",
            .storage_result = read_id,
            .storage_ty = ty,
        },
        .{
            .structural_key = "op:9/ora.sstore",
            .tag = .storage_write,
            .storage_root = "counter",
            .storage_value_source = "module-op:8/result:0(module-op:7/result:0)",
            .storage_ty = ty,
        },
        .{
            .structural_key = dependency.structural_key,
            .tag = .summary_dependency,
            .summary_dependency = dependency,
        },
    };
    const semantic: ProjectedLoopEdge = .{
        .edge = .{ .id = 0, .structural_key = "body/yield", .timeline = &timeline, .target = .backedge },
        .terminator_key = "op:10/scf.yield",
        .safety_checkpoints = &.{},
        .effect_keys = &.{ "op:7/ora.sload", "op:9/ora.sstore" },
        .state_source_keys = &.{},
        .timeline_inventory = rows[1..3],
    };
    const inventory: InventoryLoopEdge = .{
        .structural_key = "body/yield",
        .target = .backedge,
        .terminator_key = "op:10/scf.yield",
        .binding_source_keys = &.{},
        .safety_checkpoints = &.{},
        .effect_keys = &.{ "op:7/ora.sload", "op:9/ora.sstore" },
        .state_source_keys = &.{},
        .timeline_inventory = &rows,
    };
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.complete,
        compareLoopProjection(&.{semantic}, &.{inventory}),
    );

    const semantic_projection = [_]obligation.LoopStorageProjection{.{
        .fixed_index = .{
            .index = .{
                .value = .{ .formula = .{ .term = 2 } },
                .ty = ty,
            },
            .extent = 4,
        },
    }};
    const projection_inventory = [_]LoopStorageProjectionInventory{.{
        .tag = .fixed_index,
        .source_key = "module-op:6/result:0",
        .ty = ty,
        .extent = 4,
    }};
    var projected_timeline = timeline;
    projected_timeline[1].payload.storage_read.location.projections = &semantic_projection;
    var projected_rows = rows;
    projected_rows[1].storage_projections = &projection_inventory;
    var projected_semantic = semantic;
    projected_semantic.edge.timeline = &projected_timeline;
    projected_semantic.timeline_inventory = projected_rows[1..3];
    var projected_inventory = inventory;
    projected_inventory.timeline_inventory = &projected_rows;
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.complete,
        compareLoopProjection(&.{projected_semantic}, &.{projected_inventory}),
    );

    const wrong_projection_inventory = [_]LoopStorageProjectionInventory{.{
        .tag = .mapping_key,
        .source_key = "module-op:6/result:0",
        .ty = ty,
    }};
    var wrong_projection_rows = projected_rows;
    wrong_projection_rows[1].storage_projections = &wrong_projection_inventory;
    var wrong_projection = projected_inventory;
    wrong_projection.timeline_inventory = &wrong_projection_rows;
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{projected_semantic}, &.{wrong_projection}),
    );

    var wrong_extent_items = projection_inventory;
    wrong_extent_items[0].extent = 3;
    var wrong_extent_rows = projected_rows;
    wrong_extent_rows[1].storage_projections = &wrong_extent_items;
    var wrong_extent = projected_inventory;
    wrong_extent.timeline_inventory = &wrong_extent_rows;
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{projected_semantic}, &.{wrong_extent}),
    );

    var wrong_projection_source_items = projection_inventory;
    wrong_projection_source_items[0].source_key = "module-op:5/result:0";
    var wrong_projection_source_rows = projected_rows;
    wrong_projection_source_rows[1].storage_projections = &wrong_projection_source_items;
    var wrong_projection_source = projected_inventory;
    wrong_projection_source.timeline_inventory = &wrong_projection_source_rows;
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{projected_semantic}, &.{wrong_projection_source}),
    );

    var wrong_projection_type_items = projection_inventory;
    wrong_projection_type_items[0].ty = .{ .compiler_type_id = 12 };
    var wrong_projection_type_rows = projected_rows;
    wrong_projection_type_rows[1].storage_projections = &wrong_projection_type_items;
    var wrong_projection_type = projected_inventory;
    wrong_projection_type.timeline_inventory = &wrong_projection_type_rows;
    try std.testing.expectEqual(
        obligation.LoopProjectionCompleteness.effect_mismatch,
        compareLoopProjection(&.{projected_semantic}, &.{wrong_projection_type}),
    );

    var wrong_path_rows = rows;
    wrong_path_rows[0].structural_key = "wrong-path";
    var wrong_path = inventory;
    wrong_path.timeline_inventory = &wrong_path_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{wrong_path}));

    inline for (.{
        .{ .field = "structural_key", .value = "wrong" },
        .{ .field = "storage_root", .value = "other" },
        .{ .field = "storage_value_source", .value = "wrong-source" },
    }) |mutation| {
        var mutated_rows = rows;
        @field(mutated_rows[2], mutation.field) = mutation.value;
        var mutated = inventory;
        mutated.timeline_inventory = &mutated_rows;
        try std.testing.expectEqual(
            obligation.LoopProjectionCompleteness.effect_mismatch,
            compareLoopProjection(&.{semantic}, &.{mutated}),
        );
    }

    var wrong_result_rows = rows;
    wrong_result_rows[1].storage_result = .{ .file_id = Collector.storage_read_file_id, .pattern_id = 8 };
    var wrong_result = inventory;
    wrong_result.timeline_inventory = &wrong_result_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{wrong_result}));

    var wrong_type_rows = rows;
    wrong_type_rows[1].storage_ty = .{ .compiler_type_id = 12 };
    var wrong_type = inventory;
    wrong_type.timeline_inventory = &wrong_type_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{wrong_type}));

    var wrong_kind_rows = rows;
    wrong_kind_rows[1].tag = .storage_write;
    var wrong_kind = inventory;
    wrong_kind.timeline_inventory = &wrong_kind_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{wrong_kind}));

    var reordered_rows = [_]LoopTimelineInventoryEvent{ rows[0], rows[2], rows[1], rows[3] };
    var reordered = inventory;
    reordered.timeline_inventory = &reordered_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{reordered}));

    var missing = inventory;
    missing.timeline_inventory = rows[0..3];
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{missing}));
    const duplicated_rows = [_]LoopTimelineInventoryEvent{ rows[0], rows[1], rows[1], rows[2], rows[3] };
    var duplicated = inventory;
    duplicated.timeline_inventory = &duplicated_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{duplicated}));

    const wrong_footprint = [_]obligation.PlaceRef{.{ .root = "other", .region = .storage }};
    var wrong_dependency_value = dependency;
    wrong_dependency_value.write_footprint = &wrong_footprint;
    var wrong_dependency_rows = rows;
    wrong_dependency_rows[3].summary_dependency = wrong_dependency_value;
    var wrong_dependency = inventory;
    wrong_dependency.timeline_inventory = &wrong_dependency_rows;
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{wrong_dependency}));
}

test "preheader projection completeness compares the complete ordered region" {
    const ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const write = obligation.LoopTimelineEvent{ .id = 0, .structural_key = "op:1/ora.sstore", .payload = .{
        .storage_write = .{
            .location = .{ .root = "counter", .value_ty = ty },
            .value = .{ .formula = .{ .term = 0 } },
        },
    } };
    const row = LoopTimelineInventoryEvent{
        .structural_key = write.structural_key,
        .tag = .storage_write,
        .storage_root = "counter",
        .storage_value_source = "function-param:0",
        .storage_ty = ty,
    };
    const projected: ProjectedLoopPreheader = .{
        .operation_keys = &.{ "op:0/arith.constant", write.structural_key },
        .timeline = &.{write},
        .event_inventory = &.{row},
    };
    const inventory: InventoryLoopPreheader = .{
        .operation_keys = projected.operation_keys,
        .event_inventory = &.{row},
    };
    try std.testing.expect(preheaderProjectionMatchesInventory(projected, inventory));

    var missing_operation = inventory;
    missing_operation.operation_keys = inventory.operation_keys[0..1];
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, missing_operation));

    var reordered_operation = inventory;
    reordered_operation.operation_keys = &.{ write.structural_key, "op:0/arith.constant" };
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, reordered_operation));

    var missing_event = inventory;
    missing_event.event_inventory = &.{};
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, missing_event));

    var duplicated_event = inventory;
    duplicated_event.event_inventory = &.{ row, row };
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, duplicated_event));

    var fabricated_row = row;
    fabricated_row.structural_key = "op:2/ora.sstore";
    var fabricated = inventory;
    fabricated.event_inventory = &.{fabricated_row};
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, fabricated));

    var wrong_source_row = row;
    wrong_source_row.storage_value_source = "function-param:1";
    var wrong_source = inventory;
    wrong_source.event_inventory = &.{wrong_source_row};
    try std.testing.expect(!preheaderProjectionMatchesInventory(projected, wrong_source));

    try std.testing.expectEqualStrings(
        "loop_preheader_inventory_mismatch",
        @tagName(obligation.LoopUnsupportedReason.loop_preheader_inventory_mismatch),
    );
}

fn classifyQuantifierBinder(type_text: []const u8) QuantifierBinderClassification {
    if (std.mem.eql(u8, type_text, "bool")) {
        return .{ .sort = .bool, .fragment = .aufbv_quantifiers };
    }
    if (std.mem.eql(u8, type_text, "address")) {
        return .{ .sort = .bit_vector, .fragment = .aufbv_quantifiers };
    }
    if (std.mem.eql(u8, type_text, "string") or std.mem.eql(u8, type_text, "bytes")) {
        return .{ .sort = .byte_sequence, .fragment = .other };
    }
    if (isIntegerTypeName(type_text)) {
        return .{ .sort = .bit_vector, .fragment = .aufbv_quantifiers };
    }
    if (looksLikeIntegerTypeName(type_text)) {
        return .{
            .sort = .opaque_unknown,
            .fragment = .other,
            .degradation = .malformed_binder_width,
        };
    }
    return .{
        .sort = .opaque_unknown,
        .fragment = .other,
        .degradation = .unsupported_binder_type,
    };
}

fn isIntegerTypeName(type_text: []const u8) bool {
    if (!looksLikeIntegerTypeName(type_text)) return false;
    if (type_text.len == 1) return false;
    const digits = type_text[1..];
    if (digits.len == 0) return false;
    for (digits) |byte| {
        if (byte < '0' or byte > '9') return false;
    }
    const width = std.fmt.parseInt(u16, digits, 10) catch return false;
    return width != 0;
}

fn looksLikeIntegerTypeName(type_text: []const u8) bool {
    if (type_text.len == 0) return false;
    return type_text[0] == 'u' or type_text[0] == 'i';
}
