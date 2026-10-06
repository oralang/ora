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
    /// Enable with the joint call consumers; projection alone grants no authority.
    project_calls: bool = false,
    /// Data-only lifecycle collection. The caller must independently establish
    /// complete runtime/constructor coverage; a supplied subset grants nothing.
    lifecycle_roots: ?[]const mlir.MlirOperation = null,
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
    function_ops: []const usize = &.{},
    call_bindings: []const CallSiteBinding = &.{},

    pub fn deinit(self: *CollectResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const CallSiteBinding = struct { source_op_id: usize, function_id: obligation.Id };

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
    .{ "dynamic index out of bounds", .dynamic_index_bounds },
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

    try collector.indexCallDeclarations(mlir.oraModuleGetOperation(module));
    if (options.lifecycle_roots) |roots| try collector.validateLifecycleRoots(roots);
    try collector.walkOperation(mlir.oraModuleGetOperation(module), null);
    if (options.lifecycle_roots) |roots| for (roots) |root| {
        _ = try collector.requireFunctionRow(root);
    };
    if (options.project_calls) collector.projectCallFunctions() catch |err| {
        try collector.addBlockingDiagnosticFmt(.unsupported, "call_projection_{s}", .{@errorName(err)});
    };
    try collector.addBaseQueriesForFunctionOwners();
    try collector.verifyResourceGoalCompleteness();
    if (collector.function_summaries.items.len == 0) {
        try bindLoopSummaryDependencyDigests(&collector);
        try armLeanLoopCertificateRequirements(&collector);
    } else {
        obligation_to_lean.bindCollectedCallDependencies(collector.allocator, .{
            .terms = collector.terms.items,
            .obligations = collector.obligations.items,
            .assumptions = collector.assumptions.items,
            .queries = collector.queries.items,
            .loop_summaries = collector.loop_summaries.items,
            .function_summaries = collector.function_summaries.items,
        }) catch |err| try collector.addBlockingDiagnosticFmt(.invalid_dependency, "call_projection_{s}", .{@errorName(err)});
        try armLeanLoopCertificateRequirements(&collector);
    }

    const set: obligation.ObligationSet = .{
        .terms = try collector.terms.toOwnedSlice(collector.allocator),
        .obligations = try collector.obligations.toOwnedSlice(collector.allocator),
        .assumptions = try collector.assumptions.toOwnedSlice(collector.allocator),
        .queries = try collector.queries.toOwnedSlice(collector.allocator),
        .loop_summaries = try collector.loop_summaries.toOwnedSlice(collector.allocator),
        .function_summaries = try collector.function_summaries.toOwnedSlice(collector.allocator),
        .diagnostics = try collector.diagnostics.toOwnedSlice(collector.allocator),
    };

    const function_ops = try collector.allocator.alloc(usize, collector.function_ops.items.len);
    for (collector.function_ops.items, function_ops) |op, *id| id.* = @intFromPtr(op.ptr);
    var call_bindings: std.ArrayList(CallSiteBinding) = .empty;
    for (collector.call_sites.items) |site| {
        const callee = collector.callDeclaration(site) catch continue;
        for (function_ops, set.function_summaries) |op, row| if (op == @intFromPtr(callee.ptr)) {
            try call_bindings.append(collector.allocator, .{ .source_op_id = @intFromPtr(site.ptr), .function_id = row.id });
        };
    }
    return .{
        .arena = arena,
        .set = set,
        .function_ops = function_ops,
        .call_bindings = try call_bindings.toOwnedSlice(collector.allocator),
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
        .function_summaries = collector.function_summaries.items,
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
    entry_overrides: []const Collector.LoopFormulaOverride = &.{},
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
    boolean_literal: ?bool = null,
    ty: ?obligation.TypeRef = null,
};

const CallValueBinding = struct {
    value_id: usize,
    outcome: obligation.CallOutcome,
    results: []const obligation.LoopVariable,
};

const PrivateSlotValue = struct { address: usize, value: mlir.MlirValue };
const CallInventory = struct {
    callee: obligation.Id,
    outcome: obligation.CallOutcome,
    arguments: []const []const u8,
    types: []const obligation.TypeRef,
    results: []const obligation.LoopVariable,
};
const ReturnInventory = struct { outcome: obligation.CallOutcome, sources: []const []const u8, types: []const obligation.TypeRef };

const SemanticLoopPath = struct {
    calls: []const CallValueBinding = &.{},
    slots: []const PrivateSlotValue = &.{},
    key: []const u8,
    in_guard: bool = false,
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
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
};

const InventoryLoopPath = struct {
    calls: []const CallValueBinding = &.{},
    slots: []const PrivateSlotValue = &.{},
    key: []const u8,
    in_guard: bool = false,
    source_overrides: []const LoopValueSource = &.{},
    state_source_keys: []const []const u8 = &.{},
    safety_checkpoints: []const LoopSafetyCheckpointKey = &.{},
    target_keys: []const []const u8 = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8 = &.{},
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
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
    indexed_result: ?obligation.FreeVarId = null,
    indexed_region: ?obligation.LoopIndexedRegionKind = null,
    indexed_owner: ?obligation.FreeVarId = null,
    indexed_index_source: ?[]const u8 = null,
    indexed_element_ty: ?obligation.TypeRef = null,
    indexed_checkpoint_key: ?[]const u8 = null,
    summary_dependency: ?obligation.LoopSummaryDependency = null,
    invocation: ?CallInventory = null,
};

const LoopDynamicAccessInventory = struct {
    structural_key: []const u8,
    owner_identity: []const u8,
    index_identity: []const u8,
};

const DynamicAccessSite = struct {
    base: mlir.MlirValue,
    index: mlir.MlirValue,
};

const DynamicAccessCheck = struct {
    index: mlir.MlirValue,
    owner: mlir.MlirValue,
};

const ProjectedLoopPreheader = struct {
    operation_keys: []const []const u8 = &.{},
    timeline: []const obligation.LoopTimelineEvent = &.{},
    event_inventory: []const LoopTimelineInventoryEvent = &.{},
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
    formula_overrides: []const Collector.LoopFormulaOverride = &.{},
};

const InventoryLoopPreheader = struct {
    operation_keys: []const []const u8 = &.{},
    event_inventory: []const LoopTimelineInventoryEvent = &.{},
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
};

const CollectedLoopPreheader = struct {
    timeline: []const obligation.LoopTimelineEvent = &.{},
    formula_overrides: []const Collector.LoopFormulaOverride = &.{},
    complete: bool = false,
    reason: ?obligation.LoopUnsupportedReason = null,
};

const LoopHeaderFactInventory = struct {
    ordinal: u32,
    source_name: []const u8,
    structural_key: []const u8,
    source_kind: obligation.LoopHeaderFactSource,
    position: u32,
    stable_values: []const obligation.LoopHeaderFactValue,
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
    function_return: bool = false,
    loop_exit: bool = false,
    abrupt_target: ?obligation.LoopControlTarget = null,
    path: SemanticLoopPath,
    terminator_key: []const u8,
    yielded: []const mlir.MlirValue,
    control: ?LoopControlMarker = null,
};

const InventoryLoopTerminal = struct {
    function_return: bool = false,
    loop_exit: bool = false,
    abrupt_target: ?obligation.LoopControlTarget = null,
    path: InventoryLoopPath,
    terminator_key: []const u8,
    yielded: []const mlir.MlirValue,
    control: ?LoopControlMarker = null,
};

const ProjectedLoopEdge = struct {
    return_inventory: ?ReturnInventory = null,
    edge: obligation.LoopEdgeRow,
    terminator_key: []const u8,
    safety_checkpoints: []const LoopSafetyCheckpointKey,
    target_keys: []const []const u8 = &.{},
    summary_dependency_keys: []const []const u8 = &.{},
    effect_keys: []const []const u8,
    state_source_keys: []const []const u8,
    timeline_inventory: []const LoopTimelineInventoryEvent = &.{},
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
};

const InventoryLoopEdge = struct {
    return_inventory: ?ReturnInventory = null,
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
    dynamic_accesses: []const LoopDynamicAccessInventory = &.{},
};

const LoopSafetyCheckpointKey = struct {
    structural_key: []const u8,
    after_dependency_count: u32,
};

const Collector = struct {
    allocator: std.mem.Allocator,
    options: CollectOptions,
    terms: std.ArrayList(obligation.Term) = .empty,
    boolean_terms: [2]?obligation.TermId = .{ null, null },
    obligations: std.ArrayList(obligation.Obligation) = .empty,
    assumptions: std.ArrayList(obligation.Assumption) = .empty,
    queries: std.ArrayList(obligation.VerificationQuery) = .empty,
    loop_summaries: std.ArrayList(obligation.LoopSummaryRow) = .empty,
    loop_summary_drafts: std.ArrayList(LoopSummaryDraft) = .empty,
    loop_inventories: std.AutoHashMapUnmanaged(obligation.Id, []const InventoryLoopEdge) = .empty,
    call_declarations: std.ArrayList(mlir.MlirOperation) = .empty,
    call_sites: std.ArrayList(mlir.MlirOperation) = .empty,
    function_summaries: std.ArrayList(obligation.FunctionSummaryRow) = .empty,
    function_ops: std.ArrayList(mlir.MlirOperation) = .empty,
    function_contexts: std.AutoHashMapUnmanaged(usize, FunctionContext) = .empty,
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
    lifecycle_scope: ?mlir.MlirOperation = null,
    active_loop_formula_overrides: []const LoopFormulaOverride = &.{},
    arithmetic_safety_terms: std.AutoHashMapUnmanaged(obligation.TermId, void) = .empty,
    next_id: obligation.Id = 1,
    next_loop_summary_id: obligation.Id = 1,
    next_loop_variable_pattern_id: u32 = 0,
    use_loop_value_bindings: bool = false,
    project_entry_facts: bool = false,
    allow_short_circuit_predicates: bool = false,
    active_loop_depth: u32 = 0,
    next_ordinal: u32 = 0,
    function: FunctionContext = .{},
    contract_source_owner_key: ?[]const u8 = null,

    const FunctionContext = struct {
        projection: LoopSummaryDraft = .{},
        param_names: []const []const u8 = &.{},
        param_binding_ids: []const obligation.FreeVarId = &.{},
        param_types: []const obligation.TypeRef = &.{},
        entry_block: mlir.MlirBlock = std.mem.zeroes(mlir.MlirBlock),
        write_slots: ?[]const obligation.PlaceRef = null,
        write_slots_complete: bool = false,
        has_external_call: bool = false,
        source_module_path: ?[]const u8 = null,
        source_owner_key: ?[]const u8 = null,
        source_activation: ?[]const u8 = null,
        runtime_symbol: ?[]const u8 = null,
        source_specialization_bindings: []const []const u8 = &.{},
        source_trait_implementation: ?[]const u8 = null,
        source_trait_method: ?[]const u8 = null,
    };

    const synthetic_file_id = std.math.maxInt(u32);
    const storage_read_file_id = synthetic_file_id - 2;
    const dynamic_length_file_id = synthetic_file_id - 3;

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

    // This registry supplies symbol identity only. Each semantic and inventory
    // walk still reads the declaration and the complete invoked body itself.
    fn indexCallDeclarations(self: *Collector, op: mlir.MlirOperation) !void {
        const name = operationName(op);
        if (std.mem.eql(u8, name, "func.call")) try self.call_sites.append(self.allocator, op);
        if (std.mem.eql(u8, name, "func.func") or std.mem.eql(u8, name, "ora.error.decl"))
            try self.call_declarations.append(self.allocator, op);
        for (0..mlir.oraOperationGetNumRegions(op)) |region_index| {
            var block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(op, region_index));
            while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
                var child = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child))
                    try self.indexCallDeclarations(child);
            }
        }
    }

    fn scopedCallDeclaration(self: *Collector, anchor: mlir.MlirOperation, name: []const u8, kind: []const u8) ?usize {
        var scope = mlir.mlirOperationGetParentOperation(anchor);
        while (!mlir.oraOperationIsNull(scope)) : (scope = mlir.mlirOperationGetParentOperation(scope)) {
            var found: ?usize = null;
            for (self.call_declarations.items, 0..) |candidate, index| {
                if (mlir.mlirOperationGetParentOperation(candidate).ptr != scope.ptr or !std.mem.eql(u8, operationName(candidate), kind)) continue;
                const symbol = stringAttrView(candidate, "sym_name") orelse continue;
                if (!std.mem.eql(u8, symbol.slice(), name)) continue;
                if (found != null) return null;
                found = index;
            }
            if (found != null) return found;
        }
        return null;
    }

    fn callScalarType(carrier: mlir.MlirType, type_id: u32) !obligation.TypeRef {
        if (type_id == 0) {
            if (integerTypeWidth(carrier) == 1) return .{ .spelling = "bool" };
            return error.InvalidCallSignature;
        }
        const info = type_builtin.integerInfoByComptimeTypeId(type_id) orelse return error.InvalidCallSignature;
        if (!mlirIntegerCarrierHasWidth(carrier, info.width)) return error.InvalidCallSignature;
        return .{ .compiler_type_id = type_id };
    }

    fn lifecycleCallScalarType(self: *Collector, carrier: mlir.MlirType, type_id: u32) !obligation.TypeRef {
        if (self.lifecycle_scope != null and addressCarrier(carrier)) {
            // This metadata field describes integers; the actual nominal carrier owns address identity.
            if (type_id != 0) return error.InvalidCallSignature;
            return .{ .compiler_type_id = type_builtin.lookupBuiltinById(.address).comptime_type_id };
        }
        return callScalarType(carrier, type_id);
    }

    fn lifecycleCallCarrierMatches(self: *Collector, carrier: mlir.MlirType, ty: obligation.TypeRef) bool {
        if (self.lifecycle_scope != null and addressTypeRef(ty)) return addressCarrier(carrier);
        return callCarrierMatches(carrier, ty);
    }

    fn lifecycleMapKeyType(self: *Collector, value: mlir.MlirValue) !obligation.TypeRef {
        if (self.lifecycle_scope != null and addressCarrier(mlir.oraValueGetType(value)))
            return .{ .compiler_type_id = type_builtin.lookupBuiltinById(.address).comptime_type_id };
        return self.exactLoopIntegerTypeForValue(value);
    }

    fn lifecycleAddressCastOperand(self: *Collector, op: mlir.MlirOperation) !?mlir.MlirValue {
        if (self.lifecycle_scope == null) return null;
        const name = operationName(op);
        const to_integer = std.mem.eql(u8, name, "ora.addr.to.i160");
        if (!to_integer and !std.mem.eql(u8, name, "ora.i160.to.addr")) return null;
        if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumResults(op) != 1 or
            mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0 or
            mlir.oraOperationGetMemoryEffectKind(op) != mlir.ORA_OPERATION_MEMORY_EFFECT_NONE)
            return error.InvalidLifecycleAddressCast;
        const operand = mlir.oraOperationGetOperand(op, 0);
        const input = mlir.oraValueGetType(operand);
        const output = mlir.oraValueGetType(mlir.oraOperationGetResult(op, 0));
        const address = if (to_integer) input else output;
        const integer = if (to_integer) output else input;
        if (!mlir.oraTypeIsAddressType(address) or !mlir.oraTypeIsNull(mlir.oraRefinementTypeGetBaseType(address)) or
            integerTypeWidth(integer) != 160 or mlirIntegerTypeIsSigned(integer))
            return error.InvalidLifecycleAddressCast;
        return operand;
    }

    fn callTypeIds(self: *Collector, op: mlir.MlirOperation, name: []const u8) ![]const u32 {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return &.{};
        if (!mlir.mlirAttributeIsAArray(attr)) return error.InvalidCallSignature;
        return (try self.u32ArrayAttr(op, name)).?;
    }

    fn callParameters(self: *Collector, op: mlir.MlirOperation, fn_type: mlir.MlirType) ![]const obligation.LoopVariable {
        const block = functionEntryBlock(op);
        if (mlir.oraBlockIsNull(block)) return error.InvalidCallSignature;
        const count: usize = @intCast(mlir.mlirFunctionTypeGetNumInputs(fn_type));
        if (mlir.oraBlockGetNumArguments(block) != count) return error.InvalidCallSignature;
        const id_attr = mlir.oraOperationGetAttributeByName(op, strRef("ora.param_binding_ids"));
        if (!mlir.oraAttributeIsNull(id_attr) and !mlir.mlirAttributeIsAArray(id_attr)) return error.InvalidCallSignature;
        const ids = (try self.freeVarIdArrayAttr(op, "ora.param_binding_ids")) orelse &.{};
        const types = try self.callTypeIds(op, "ora.param_compiler_type_ids");
        if (ids.len != count or types.len != count) return error.InvalidCallSignature;
        const parameters = try self.allocator.alloc(obligation.LoopVariable, count);
        for (parameters, 0..) |*parameter, index| {
            for (ids[0..index]) |previous| if (std.meta.eql(previous, ids[index])) return error.InvalidCallSignature;
            const carrier = mlir.oraValueGetType(mlir.oraBlockGetArgument(block, index));
            if (!mlir.mlirTypeEqual(carrier, mlir.mlirFunctionTypeGetInput(fn_type, @intCast(index)))) return error.InvalidCallSignature;
            parameter.* = .{ .index = @intCast(index), .id = ids[index], .ty = try self.lifecycleCallScalarType(carrier, types[index]) };
        }
        return parameters;
    }

    fn callOutcomes(self: *Collector, op: mlir.MlirOperation, fn_type: mlir.MlirType) ![]const obligation.CallOutcomeShape {
        const count = mlir.mlirFunctionTypeGetNumResults(fn_type);
        const type_id = (try self.u32Attr(op, "ora.return_compiler_type_id")) orelse return error.InvalidCallSignature;
        const error_ids = try self.callTypeIds(op, "ora.return_error_ids");
        if (count == 0) {
            if (type_id != 0 or error_ids.len != 0) return error.InvalidCallSignature;
            return &.{.{ .outcome = .success }};
        }
        if (count != 1) return error.InvalidCallSignature;
        const result = mlir.mlirFunctionTypeGetResult(fn_type, 0);
        const success = mlir.oraErrorUnionTypeGetSuccessType(result);
        const is_union = !mlir.oraTypeIsNull(success);
        const success_type = if (is_union) success else result;
        const error_count = if (is_union) mlir.oraErrorUnionTypeGetNumErrorTypes(result) else 0;
        const has_error_ids = hasAttr(op, "ora.return_error_ids");
        if (has_error_ids and error_ids.len != error_count) return error.InvalidCallSignature;
        const outcomes = try self.allocator.alloc(obligation.CallOutcomeShape, error_count + 1);
        outcomes[0] = .{ .outcome = .success };
        if (mlir.oraTypeIsANone(success_type)) {
            if (type_id != 0) return error.InvalidCallSignature;
        } else {
            const types = try self.allocator.alloc(obligation.TypeRef, 1);
            types[0] = try self.lifecycleCallScalarType(success_type, type_id);
            outcomes[0].types = types;
        }
        for (outcomes[1..], 0..) |*outcome, index| {
            const error_type = mlir.oraErrorUnionTypeGetErrorType(result, index);
            const name = mlir.oraStructTypeGetName(error_type);
            if (name.data == null or name.length == 0) return error.InvalidCallSignature;
            const declaration_index = self.scopedCallDeclaration(op, name.data[0..name.length], "ora.error.decl") orelse return error.InvalidCallSignature;
            const declaration = self.call_declarations.items[declaration_index];
            const error_id = (try self.u32Attr(declaration, "ora.error_id")) orelse return error.InvalidCallSignature;
            if (has_error_ids and error_id != error_ids[index]) return error.InvalidCallSignature;
            for (outcomes[1 .. index + 1]) |previous| if (previous.outcome.error_return == error_id) return error.InvalidCallSignature;
            const type_ids = try self.callTypeIds(declaration, "ora.param_compiler_type_ids");
            const carrier_types = mlir.oraOperationGetAttributeByName(declaration, strRef("ora.param_types"));
            if (!mlir.oraAttributeIsNull(carrier_types) and !mlir.mlirAttributeIsAArray(carrier_types)) return error.InvalidCallSignature;
            const payload_count = if (mlir.oraAttributeIsNull(carrier_types)) 0 else mlir.oraArrayAttrGetNumElements(carrier_types);
            if (type_ids.len != payload_count) return error.InvalidCallSignature;
            const payload = try self.allocator.alloc(obligation.TypeRef, payload_count);
            for (payload, 0..) |*ty, field| {
                const carrier = mlir.oraTypeAttrGetValue(mlir.oraArrayAttrGetElement(carrier_types, field));
                if (mlir.oraTypeIsNull(carrier)) return error.InvalidCallSignature;
                ty.* = try self.lifecycleCallScalarType(carrier, type_ids[field]);
            }
            outcome.* = .{ .outcome = .{ .error_return = error_id }, .types = payload };
        }
        return outcomes;
    }

    fn callSignature(self: *Collector, op: mlir.MlirOperation) !struct {
        parameters: []const obligation.LoopVariable,
        outcomes: []const obligation.CallOutcomeShape,
    } {
        if (!std.mem.eql(u8, operationName(op), "func.func")) return error.InvalidCallSignature;
        const attr = mlir.oraOperationGetAttributeByName(op, strRef("function_type"));
        if (mlir.oraAttributeIsNull(attr)) return error.InvalidCallSignature;
        const ty = mlir.oraTypeAttrGetValue(attr);
        if (mlir.oraTypeIsNull(ty) or !mlir.mlirTypeIsAFunction(ty)) return error.InvalidCallSignature;
        return .{ .parameters = try self.callParameters(op, ty), .outcomes = try self.callOutcomes(op, ty) };
    }

    fn callDeclaration(self: *Collector, op: mlir.MlirOperation) !mlir.MlirOperation {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef("callee"));
        if (mlir.oraAttributeIsNull(attr) or !mlir.mlirAttributeIsAFlatSymbolRef(attr)) return error.UnresolvedCall;
        const name = mlir.mlirFlatSymbolRefAttrGetValue(attr);
        const index = self.scopedCallDeclaration(op, name.data[0..name.length], "func.func") orelse return error.UnresolvedCall;
        const callee = self.call_declarations.items[index];
        if (mlir.oraBlockIsNull(functionEntryBlock(callee))) return error.MissingCallBody;
        const ty = mlir.oraTypeAttrGetValue(mlir.oraOperationGetAttributeByName(callee, strRef("function_type")));
        if (mlir.oraTypeIsNull(ty) or !mlir.mlirTypeIsAFunction(ty)) return error.InvalidCallSignature;
        if (mlir.oraOperationGetNumOperands(op) != mlir.mlirFunctionTypeGetNumInputs(ty) or
            mlir.oraOperationGetNumResults(op) != mlir.mlirFunctionTypeGetNumResults(ty)) return error.InvalidCallSignature;
        for (0..@intCast(mlir.oraOperationGetNumOperands(op))) |i| if (!mlir.mlirTypeEqual(mlir.oraValueGetType(mlir.oraOperationGetOperand(op, i)), mlir.mlirFunctionTypeGetInput(ty, @intCast(i)))) return error.InvalidCallSignature;
        for (0..@intCast(mlir.oraOperationGetNumResults(op))) |i| if (!mlir.mlirTypeEqual(mlir.oraValueGetType(mlir.oraOperationGetResult(op, i)), mlir.mlirFunctionTypeGetResult(ty, @intCast(i)))) return error.InvalidCallSignature;
        return callee;
    }

    fn validateLifecycleRoots(self: *Collector, roots: []const mlir.MlirOperation) !void {
        if (!self.options.project_calls or roots.len == 0) return error.MissingLifecycleRoots;
        for (roots, 0..) |root, index| {
            const found = for (self.call_declarations.items) |candidate| {
                if (candidate.ptr == root.ptr and std.mem.eql(u8, operationName(candidate), "func.func")) break true;
            } else false;
            if (!found) return error.ForeignLifecycleRoot;
            for (roots[0..index]) |previous| if (previous.ptr == root.ptr) return error.DuplicateLifecycleRoot;
            const scope = mlir.mlirOperationGetParentOperation(root);
            if (!std.mem.eql(u8, operationName(scope), "ora.contract")) return error.LifecycleOwnerUnsupported;
            if (self.lifecycle_scope) |previous| {
                if (previous.ptr != scope.ptr) return error.LifecycleOwnerUnsupported;
            } else self.lifecycle_scope = scope;
        }
        for (self.call_declarations.items) |declaration| {
            if (mlir.mlirOperationGetParentOperation(declaration).ptr != self.lifecycle_scope.?.ptr)
                return error.LifecycleOwnerUnsupported;
        }
    }

    fn callConnectedFunction(self: *Collector, function: mlir.MlirOperation) bool {
        if (!self.options.project_calls) return false;
        for (self.call_sites.items) |site| {
            const callee = self.callDeclaration(site) catch continue;
            if (callee.ptr == function.ptr) return true;
        }
        for (self.call_sites.items) |site| if (containingFunction(site).ptr == function.ptr)
            return callerHasScalarParameters(function);
        return false;
    }

    // An ordinary caller's slice is not a callee signature participant. Keep
    // that caller on the existing verifier path; actual callees still pass
    // callSignature unconditionally, including calls which pass a slice.
    fn callerHasScalarParameters(function: mlir.MlirOperation) bool {
        const block = functionEntryBlock(function);
        if (mlir.oraBlockIsNull(block)) return false;
        for (0..@intCast(mlir.oraBlockGetNumArguments(block))) |index| {
            if (integerTypeWidth(mlir.oraValueGetType(mlir.oraBlockGetArgument(block, index))) == null) return false;
        }
        return true;
    }

    fn requireFunctionRow(self: *Collector, op: mlir.MlirOperation) !usize {
        for (self.function_ops.items, 0..) |previous, index| if (previous.ptr == op.ptr) return index;
        if (singleRegionBlock(op, 0) == null) return error.CallBodyControlUnsupported;
        const signature = try self.callSignature(op);
        const name = (try self.stringAttr(op, "sym_name")) orelse return error.InvalidCallSignature;
        // Canonical assumption owners cannot distinguish scoped namesakes.
        for (self.call_declarations.items) |other| {
            if (other.ptr == op.ptr or !std.mem.eql(u8, operationName(other), "func.func")) continue;
            const other_name = stringAttrView(other, "sym_name") orelse return error.InvalidCallSignature;
            if (std.mem.eql(u8, name, other_name.slice())) return error.CallOwnerAmbiguous;
        }
        const index = self.function_ops.items.len;
        try self.function_ops.append(self.allocator, op);
        try self.function_summaries.append(self.allocator, .{
            .id = @intCast(index + 1),
            .owner = .{ .function = .{ .name = name } },
            .source = try self.sourceForOperation(op),
            .origin = .source,
            .structural_key = name,
            .parameters = signature.parameters,
            .outcomes = signature.outcomes,
        });
        return index;
    }

    fn projectCallFunctions(self: *Collector) !void {
        // The loop walk seeds the connected functions. Expand callers here
        // and callees in each sequence walk; this registry grants no authority.
        const previous = self.function;
        defer self.function = previous;
        var index: usize = 0;
        while (index < self.function_ops.items.len) : (index += 1) {
            const op = self.function_ops.items[index];
            for (self.call_sites.items) |site| {
                const callee = self.callDeclaration(site) catch continue;
                const caller = containingFunction(site);
                if (callee.ptr == op.ptr and self.callConnectedFunction(caller)) _ = try self.requireFunctionRow(caller);
            }
            self.function = self.function_contexts.get(@intFromPtr(op.ptr)) orelse return error.MissingCallBody;
            if (self.function.has_external_call) return error.CallExternalUnsupported;
            var row = self.function_summaries.items[index];
            row.structural_key = try self.operationStructuralKey(op, "func.func");
            row.origin = mlirOrigin("func.func", self.function.runtime_symbol, self.function.projection.ordinal);
            row.assumption_ids = try self.assumptionIdsForLoopOwner(row.owner);
            try self.projectFunctionEntryFacts(&row, op);
            var semantic: std.ArrayList(SemanticLoopTerminal) = .empty;
            var inventory: std.ArrayList(InventoryLoopTerminal) = .empty;
            const first = mlir.oraBlockGetFirstOperation(self.function.entry_block);
            try self.projectSemanticLoopSequence(first, .{ .key = row.structural_key }, self.function.projection, row.parameters, &semantic);
            try self.inventoryLoopSequence(first, .{ .key = row.structural_key }, self.function.projection, row.parameters, &inventory);
            if (semantic.items.len == 0 or semantic.items.len != inventory.items.len) return error.CallPathMismatch;
            const paths = try self.allocator.alloc(obligation.FunctionPathRow, semantic.items.len);
            for (semantic.items, inventory.items, paths, 0..) |actual, found, *path, ordinal| {
                if (!actual.function_return or !found.function_return or
                    !std.mem.eql(u8, actual.path.key, found.path.key) or !std.mem.eql(u8, actual.terminator_key, found.terminator_key) or
                    !safetyCheckpointKeysEql(try safetyCheckpointKeysFromRows(self.allocator, actual.path.safety_checkpoints), found.path.safety_checkpoints) or
                    !stringSlicesEql(actual.path.effect_keys, found.path.effect_keys) or !stringSlicesEql(actual.path.state_source_keys, found.path.state_source_keys) or
                    !dynamicAccessInventoriesEql(actual.path.dynamic_accesses, found.path.dynamic_accesses) or
                    !stringSlicesEql(actual.path.target_keys, found.path.target_keys) or !stringSlicesEql(actual.path.summary_dependency_keys, found.path.summary_dependency_keys) or
                    !effectTimelineInventoryEql(actual.path.timeline_inventory, found.path.timeline_inventory) or
                    !timelineMatchesInventory(actual.path.timeline, found.path.timeline_inventory)) return error.CallInventoryMismatch;
                if (actual.loop_exit != found.loop_exit) return error.CallReturnMismatch;
                if (actual.loop_exit) {
                    path.* = .{ .id = @intCast(ordinal), .structural_key = actual.path.key, .timeline = actual.path.timeline, .returns = .loop_exit };
                    continue;
                }
                const returns = try self.projectFunctionReturn(actual, row);
                const expected = try self.inventoryFunctionReturn(found, row);
                if (!returnInventoriesEql(returns.inventory, expected)) return error.CallReturnMismatch;
                path.* = .{ .id = @intCast(ordinal), .structural_key = actual.path.key, .timeline = actual.path.timeline, .returns = returns.value };
            }
            row.paths = paths;
            row.projection_completeness = .complete;
            self.function_summaries.items[index] = row;
        }
    }

    fn projectFunctionEntryFacts(self: *Collector, row: *obligation.FunctionSummaryRow, function: mlir.MlirOperation) !void {
        var parent = mlir.mlirOperationGetParentOperation(function);
        while (!mlir.oraOperationIsNull(parent) and !std.mem.eql(u8, operationName(parent), "ora.contract"))
            parent = mlir.mlirOperationGetParentOperation(parent);
        if (mlir.oraOperationIsNull(parent)) return;
        row.entry_scope = try self.operationStructuralKey(parent, "ora.contract");
        var selected: std.ArrayList(obligation.Id) = .empty;
        for (self.obligations.items) |fact| if (row.ownsEntryInvariant(fact)) {
            try selected.append(self.allocator, fact.id);
        };
        // Independently inventory actual declarations, not the selected rows.
        const block = singleRegionBlock(parent, 0) orelse return error.CallEntryScopeUnsupported;
        var op = mlir.oraBlockGetFirstOperation(block);
        var inventory: std.ArrayList(obligation.Id) = .empty;
        while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
            if (!std.mem.eql(u8, operationName(op), "ora.invariant")) continue;
            const ordinal = self.operationOrdinal(op) orelse return error.CallEntryInventoryMismatch;
            const before = inventory.items.len;
            for (self.obligations.items) |fact| {
                if (fact.origin != .mlir_op or fact.origin.mlir_op.ordinal != ordinal) continue;
                try inventory.append(self.allocator, fact.id);
            }
            if (inventory.items.len != before + 1) return error.CallEntryInventoryMismatch;
        }
        if (!std.mem.eql(obligation.Id, selected.items, inventory.items)) return error.CallEntryInventoryMismatch;
        row.entry_invariant_ids = try selected.toOwnedSlice(self.allocator);
    }

    fn projectFunctionReturn(self: *Collector, terminal: SemanticLoopTerminal, row: obligation.FunctionSummaryRow) !struct { value: obligation.CallReturnExpr, inventory: ReturnInventory } {
        var values = terminal.yielded;
        var outcome: obligation.CallOutcome = .success;
        var bound: ?[]const obligation.LoopVariable = null;
        if (values.len == 1) {
            var value = values[0];
            while (mlir.oraValueIsAOpResult(value)) {
                const owner = mlir.oraOpResultGetOwner(value);
                const name = operationName(owner);
                if (std.mem.eql(u8, name, "ora.error.ok") or std.mem.eql(u8, name, "ora.error.err") or std.mem.eql(u8, name, "ora.error.get_error")) {
                    values = try self.operationOperands(owner);
                    if (values.len != 1) break;
                    value = values[0];
                    continue;
                }
                if (std.mem.eql(u8, name, "ora.error.return")) {
                    const symbol = (try self.stringAttr(owner, "sym_name")) orelse return error.InvalidCallReturn;
                    const decl = self.scopedCallDeclaration(owner, symbol, "ora.error.decl") orelse return error.InvalidCallReturn;
                    outcome = .{ .error_return = (try self.u32Attr(self.call_declarations.items[decl], "ora.error_id")) orelse return error.InvalidCallReturn };
                    values = try self.operationOperands(owner);
                } else for (terminal.path.calls) |call| {
                    if (call.value_id == @intFromPtr(value.ptr)) {
                        outcome = call.outcome;
                        bound = call.results;
                        break;
                    }
                }
                break;
            }
        }
        const types = obligation.callOutcomeTypes(row, outcome) orelse return error.InvalidCallReturn;
        if (types.len != (if (bound) |results| results.len else values.len)) return error.InvalidCallReturn;
        const formulas = try self.allocator.alloc(obligation.LoopTypedFormula, types.len);
        const sources = try self.allocator.alloc([]const u8, types.len);
        for (types, formulas, sources, 0..) |ty, *formula, *source, index| {
            const value: obligation.FormulaRef = if (bound) |results| blk: {
                source.* = try std.fmt.allocPrint(self.allocator, "call-result:{d}:{d}", .{ results[index].id.?.file_id, results[index].id.?.pattern_id });
                if (!obligation.typeRefEqual(ty, results[index].ty)) return error.InvalidCallReturn;
                break :blk .{ .term = try self.addTerm(.{ .variable = .{ .free = .{ .id = results[index].id.?, .name = "call_result", .ty = ty } } }) };
            } else blk: {
                if (!self.lifecycleCallCarrierMatches(mlir.oraValueGetType(values[index]), ty) or !obligation.typeRefEqual(try self.callValueType(values[index], terminal.path.source_overrides), ty)) return error.InvalidCallReturn;
                source.* = try self.sourceKeyForValue(values[index], terminal.path.source_overrides);
                break :blk try self.loopFormulaFromValueWithOverrides(values[index], terminal.path.formula_overrides, "ora.return", self.function.runtime_symbol, 0, @intCast(index));
            };
            if (value == .origin_value) return error.CallReturnValueUnsupported;
            formula.* = .{ .ty = ty, .value = .{ .formula = value, .read_results = try self.readResultsForFormula(value, terminal.path.timeline) } };
        }
        return .{ .value = .{ .direct = .{ .outcome = outcome, .values = formulas } }, .inventory = .{ .outcome = outcome, .sources = sources, .types = types } };
    }

    fn inventoryFunctionReturn(self: *Collector, terminal: InventoryLoopTerminal, row: obligation.FunctionSummaryRow) !ReturnInventory {
        var operands = terminal.yielded;
        var outcome: obligation.CallOutcome = .success;
        var results: ?[]const obligation.LoopVariable = null;
        if (operands.len == 1) {
            var cursor = operands[0];
            while (mlir.oraValueIsAOpResult(cursor)) {
                const producer = mlir.oraOpResultGetOwner(cursor);
                const name = operationName(producer);
                if (std.mem.eql(u8, name, "ora.error.ok") or std.mem.eql(u8, name, "ora.error.err") or std.mem.eql(u8, name, "ora.error.get_error")) {
                    operands = try self.operationOperands(producer);
                    if (operands.len != 1) break;
                    cursor = operands[0];
                    continue;
                }
                if (std.mem.eql(u8, name, "ora.error.return")) {
                    const symbol = (try self.stringAttr(producer, "sym_name")) orelse return error.InvalidCallReturn;
                    const decl = self.scopedCallDeclaration(producer, symbol, "ora.error.decl") orelse return error.InvalidCallReturn;
                    outcome = .{ .error_return = (try self.u32Attr(self.call_declarations.items[decl], "ora.error_id")) orelse return error.InvalidCallReturn };
                    operands = try self.operationOperands(producer);
                } else for (terminal.path.calls) |call| {
                    if (call.value_id == @intFromPtr(cursor.ptr)) {
                        outcome = call.outcome;
                        results = call.results;
                        break;
                    }
                }
                break;
            }
        }
        const types = obligation.callOutcomeTypes(row, outcome) orelse return error.InvalidCallReturn;
        if (types.len != (if (results) |bindings| bindings.len else operands.len)) return error.InvalidCallReturn;
        const sources = try self.allocator.alloc([]const u8, types.len);
        for (types, sources, 0..) |ty, *source, index| {
            if (results) |bindings| {
                if (!obligation.typeRefEqual(ty, bindings[index].ty)) return error.InvalidCallReturn;
                source.* = try std.fmt.allocPrint(self.allocator, "call-result:{d}:{d}", .{ bindings[index].id.?.file_id, bindings[index].id.?.pattern_id });
            } else {
                if (!self.lifecycleCallCarrierMatches(mlir.oraValueGetType(operands[index]), ty) or !obligation.typeRefEqual(try self.callValueType(operands[index], terminal.path.source_overrides), ty)) return error.InvalidCallReturn;
                source.* = try self.sourceKeyForValue(operands[index], terminal.path.source_overrides);
            }
        }
        return .{ .outcome = outcome, .sources = sources, .types = types };
    }

    fn callResultVariables(self: *Collector, op: mlir.MlirOperation, shape: obligation.CallOutcomeShape) ![]const obligation.LoopVariable {
        const results = try self.allocator.alloc(obligation.LoopVariable, shape.types.len);
        const ordinal = self.operationOrdinal(op) orelse return error.UnresolvedCall;
        for (results, shape.types, 0..) |*result, ty, index| result.* = .{
            .index = @intCast(index),
            .id = .{ .file_id = synthetic_file_id - 4 - ordinal, .pattern_id = @intCast(index) },
            .ty = ty,
        };
        return results;
    }

    fn callValueType(self: *Collector, value: mlir.MlirValue, sources: []const LoopValueSource) !obligation.TypeRef {
        const ty = try self.rawCallValueType(value, sources);
        if (self.lifecycle_scope != null and ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.bool).comptime_type_id and
            integerTypeWidth(mlir.oraValueGetType(value)) == 1) return .{ .spelling = "bool" };
        return ty;
    }

    fn rawCallValueType(self: *Collector, value: mlir.MlirValue, sources: []const LoopValueSource) !obligation.TypeRef {
        var index = sources.len;
        while (index > 0) {
            index -= 1;
            if (sources[index].value_id == @intFromPtr(value.ptr)) if (sources[index].ty) |ty| return ty;
        }
        if (self.loopVariableForValue(value)) |variable| return variable.ty;
        if (mlir.oraValueIsAOpResult(value)) {
            var owner = mlir.oraOpResultGetOwner(value);
            if (std.mem.eql(u8, operationName(owner), "ora.error.unwrap") and mlir.oraOperationGetNumOperands(owner) == 1 and mlir.oraValueIsAOpResult(mlir.oraOperationGetOperand(owner, 0))) owner = mlir.oraOpResultGetOwner(mlir.oraOperationGetOperand(owner, 0));
            if (std.mem.eql(u8, operationName(owner), "func.call")) {
                const shape = (try self.callSignature(try self.callDeclaration(owner))).outcomes[0];
                if (shape.types.len != 1) return error.InvalidCallSignature;
                return shape.types[0];
            }
            const key = try std.fmt.allocPrint(self.allocator, "ora.result_integer_signed_{d}", .{mlir.mlirOpResultGetResultNumber(value)});
            return self.canonicalTypeRefForValue(value, (try self.boolAttr(owner, key)) orelse try self.boolAttr(owner, "ora.integer_signed"));
        }
        return self.canonicalTypeRefForValue(value, null);
    }

    fn projectSemanticCall(self: *Collector, op: mlir.MlirOperation, initial: SemanticLoopPath, shape: obligation.CallOutcomeShape) !SemanticLoopPath {
        const callee = try self.callDeclaration(op);
        const signature = try self.callSignature(callee);
        const operands = try self.operationOperands(op);
        if (operands.len != signature.parameters.len) return error.InvalidCallSignature;
        const arguments = try self.allocator.alloc(obligation.LoopTypedFormula, operands.len);
        const sources = try self.allocator.alloc([]const u8, operands.len);
        const types = try self.allocator.alloc(obligation.TypeRef, operands.len);
        for (operands, signature.parameters, arguments, sources, types) |operand, parameter, *argument, *source, *ty| {
            if (!self.lifecycleCallCarrierMatches(mlir.oraValueGetType(operand), parameter.ty) or !obligation.typeRefEqual(try self.callValueType(operand, initial.source_overrides), parameter.ty)) return error.InvalidCallSignature;
            const formula = try self.loopFormulaFromValueWithOverrides(operand, initial.formula_overrides, "func.call", self.function.runtime_symbol, 0, 0);
            if (formula == .origin_value) return error.CallArgumentUnsupported;
            argument.* = .{ .ty = parameter.ty, .value = .{ .formula = formula, .read_results = try self.readResultsForFormula(formula, initial.timeline) } };
            source.* = try self.sourceKeyForValue(operand, initial.source_overrides);
            ty.* = parameter.ty;
        }
        const row_index = try self.requireFunctionRow(callee);
        const results = try self.callResultVariables(op, shape);
        const key = try self.operationStructuralKey(op, "func.call");
        var path = initial;
        path.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/outcome:{d}", .{ path.key, key, callOutcomeKey(shape.outcome) });
        path.effect_keys = try appendString(self.allocator, path.effect_keys, key);
        path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{ .call = .{
            .callee_summary_id = @intCast(row_index + 1),
            .callee_vc_sha256 = @splat(0),
            .arguments = arguments,
            .outcome = shape.outcome,
            .results = results,
        } }, key);
        path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
            .structural_key = key,
            .tag = .call,
            .invocation = .{ .callee = @intCast(row_index + 1), .outcome = shape.outcome, .arguments = sources, .types = types, .results = results },
        });
        if (mlir.oraOperationGetNumResults(op) == 1) {
            const result = mlir.oraOperationGetResult(op, 0);
            path.calls = try std.mem.concat(self.allocator, CallValueBinding, &.{ path.calls, &.{.{ .value_id = @intFromPtr(result.ptr), .outcome = shape.outcome, .results = results }} });
            if (signature.outcomes.len == 1 and results.len == 1) {
                const term = try self.addTerm(.{ .variable = .{ .free = .{ .id = results[0].id.?, .name = "call_result", .ty = results[0].ty } } });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{ .value_id = @intFromPtr(result.ptr), .formula = .{ .term = term } });
            }
        }
        return path;
    }

    fn inventoryCall(self: *Collector, op: mlir.MlirOperation, initial: InventoryLoopPath, shape: obligation.CallOutcomeShape) !InventoryLoopPath {
        const declaration = try self.callDeclaration(op);
        const signature = try self.callSignature(declaration);
        const count: usize = @intCast(mlir.oraOperationGetNumOperands(op));
        if (count != signature.parameters.len) return error.InvalidCallSignature;
        const sources = try self.allocator.alloc([]const u8, count);
        const types = try self.allocator.alloc(obligation.TypeRef, count);
        for (sources, types, 0..) |*source, *ty, index| {
            const operand = mlir.oraOperationGetOperand(op, index);
            ty.* = signature.parameters[index].ty;
            if (!self.lifecycleCallCarrierMatches(mlir.oraValueGetType(operand), ty.*) or !obligation.typeRefEqual(try self.callValueType(operand, initial.source_overrides), ty.*)) return error.InvalidCallSignature;
            source.* = try self.sourceKeyForValue(operand, initial.source_overrides);
        }
        const row_index = try self.requireFunctionRow(declaration);
        const results = try self.callResultVariables(op, shape);
        const key = try self.operationStructuralKey(op, "func.call");
        var path = initial;
        path.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/outcome:{d}", .{ path.key, key, callOutcomeKey(shape.outcome) });
        path.effect_keys = try appendString(self.allocator, path.effect_keys, key);
        path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
            .structural_key = key,
            .tag = .call,
            .invocation = .{ .callee = @intCast(row_index + 1), .outcome = shape.outcome, .arguments = sources, .types = types, .results = results },
        });
        if (mlir.oraOperationGetNumResults(op) == 1) path.calls = try std.mem.concat(self.allocator, CallValueBinding, &.{ path.calls, &.{.{
            .value_id = @intFromPtr(mlir.oraOperationGetResult(op, 0).ptr),
            .outcome = shape.outcome,
            .results = results,
        }} });
        return path;
    }

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
            var type_id = (try self.u32Attr(op, "ora.compiler_type_id")) orelse 0;
            if (self.lifecycle_scope) |scope| {
                if (mlir.mlirOperationGetParentOperation(op).ptr != scope.ptr or self.storage_global_types.contains(root))
                    return error.LifecycleStorageOwnerAmbiguous;
                const attr = mlir.oraOperationGetAttributeByName(op, strRef("type"));
                if (mlir.oraAttributeIsNull(attr) or !mlir.mlirAttributeIsAType(attr)) return error.LifecycleStorageTypeMissing;
                if (addressCarrier(mlir.oraTypeAttrGetValue(attr))) {
                    if (type_id != 0) return error.LifecycleStorageTypeMissing;
                    type_id = type_builtin.lookupBuiltinById(.address).comptime_type_id;
                } else if (type_id == type_builtin.lookupBuiltinById(.address).comptime_type_id)
                    return error.LifecycleStorageTypeMissing;
            }
            if (root.len != 0) try self.storage_global_types.put(
                self.allocator,
                root,
                .{ .compiler_type_id = type_id },
            );
        }
        const previous_function = self.function;
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
            self.function = .{ .projection = .{ .loop_op = op, .body = functionEntryBlock(op), .symbol = symbol, .ordinal = ordinal } };
            self.function.param_names = (try self.stringArrayAttr(op, "ora.param_names")) orelse &.{};
            self.function.param_binding_ids = (try self.freeVarIdArrayAttr(op, "ora.param_binding_ids")) orelse &.{};
            self.function.entry_block = functionEntryBlock(op);
            self.function.param_types = try self.functionParamTypesFromFunctionOp(op, self.function.param_names.len);
            self.function.write_slots = try self.placeArrayAttr(op, "ora.write_slots");
            self.function.write_slots_complete = (try self.boolAttr(op, "ora.write_slots_complete")) orelse false;
            self.function.has_external_call = operationBodyHasExternalCall(op);
            self.function.source_module_path = try self.stringAttr(op, "ora.source_module_path");
            self.function.source_owner_key = try self.stringAttr(op, "ora.source_owner_key");
            self.function.source_activation = try self.stringAttr(op, "ora.source_template_activation");
            self.function.runtime_symbol = if (symbol) |name| try self.allocator.dupe(u8, name) else null;
            self.function.source_specialization_bindings = (try self.stringArrayAttr(op, "ora.source_specialization_bindings")) orelse &.{};
            self.function.source_trait_implementation = try self.stringAttr(op, "ora.source_trait_implementation");
            self.function.source_trait_method = try self.stringAttr(op, "ora.source_trait_method");
            if (self.function.source_owner_key) |owner_key| {
                if (self.function.source_module_path) |module_path| {
                    if (self.function.source_activation) |activation| {
                        if (sourceOpId(op)) |source_op_id| try self.runtime_owner_bindings.append(self.allocator, .{
                            .source_op_id = source_op_id,
                            .op_ordinal = ordinal,
                            .source = try self.sourceForOperation(op),
                            .symbol = if (symbol) |name| try self.allocator.dupe(u8, name) else null,
                            .module_path = module_path,
                            .owner_key = owner_key,
                            .template_activation = activation,
                            .specialization_bindings = self.function.source_specialization_bindings,
                            .trait_implementation = self.function.source_trait_implementation,
                            .trait_method = self.function.source_trait_method,
                        });
                    } else {
                        try self.addBlockingDiagnostic(.missing_type, "source-owned func.func is missing ora.source_template_activation");
                    }
                } else {
                    try self.addBlockingDiagnostic(.missing_type, "source-owned func.func is missing ora.source_module_path");
                }
            }
            if (self.function.param_names.len != 0 and self.function.param_binding_ids.len == 0) {
                try self.addBlockingDiagnostic(
                    .missing_type,
                    "func.func has ora.param_names but is missing ora.param_binding_ids",
                );
            } else if (self.function.param_binding_ids.len != 0 and
                self.function.param_names.len != 0 and
                self.function.param_binding_ids.len != self.function.param_names.len)
            {
                try self.addBlockingDiagnosticFmt(
                    .missing_type,
                    "ora.param_binding_ids has {d} entries but ora.param_names has {d}",
                    .{ self.function.param_binding_ids.len, self.function.param_names.len },
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
                self.function = previous_function;
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
            // Internal overflow probes mask the quotient comparison when rhs is zero.
            const guard_internal = if (stringAttrView(op, "ora.guard_internal")) |value|
                std.mem.eql(u8, value.slice(), "true")
            else
                false;
            if (!self.operationFeedsOnlyFormalFormula(op) and
                !guard_internal and
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
            try self.function_contexts.put(self.allocator, @intFromPtr(op.ptr), self.function);
        }
    }

    fn collectLoopSummary(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !void {
        if (self.callConnectedFunction(containingFunction(op))) _ = try self.requireFunctionRow(containingFunction(op));
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
        const header_facts = self.collectLoopHeaderFacts(op, .{ .function = .{
            .name = owner.statement.function_name,
        } }) catch |err| blk: {
            try appendLoopReason(&reasons, self.allocator, headerFactProjectionFailure(err) orelse return err);
            break :blk &.{};
        };
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
            const semantic_integer = typeRefIntegerInfo(variable.ty);
            if (semantic_integer == null) try appendLoopReason(&reasons, self.allocator, .loop_value_type_unresolved);
            const carrier_is_integer = !mlir.oraValueIsNull(value) and
                (loop_kind == .scf_for and variable.index == 0 or
                    semantic_integer != null and
                        mlirIntegerCarrierHasWidth(
                            mlir.oraValueGetType(value),
                            semantic_integer.?.width,
                        ));
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
        if (body_facts.has_indexed_region_write) {
            try appendLoopReason(&reasons, self.allocator, .loop_indexed_region_write_unreviewed);
        }
        if (mlir.oraOperationContainsCall(op)) try appendLoopReason(&reasons, self.allocator, .loop_has_call);
        if (body_facts.has_external_call) try appendLoopReason(&reasons, self.allocator, .loop_has_external_call);
        if (body_facts.has_resource_operation) try appendLoopReason(&reasons, self.allocator, .loop_has_resource_operation);
        if (body_facts.has_break_or_continue) try appendLoopReason(&reasons, self.allocator, .loop_has_break_or_continue);
        if (body_facts.has_error_control_flow) try appendLoopReason(&reasons, self.allocator, .loop_has_error_control_flow);
        const has_prefix_call = for (preheader.timeline) |event| {
            if (event.payload == .call) break true;
        } else false;
        const preheader_is_proof_relevant = has_prefix_call or self.formulasReferenceStoragePlace(init_formulas) or
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
            .header_facts = header_facts,
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
            .entry_overrides = preheader.formula_overrides,
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
        for (&self.boolean_terms) |*cached| if (cached.*) |id| {
            if (id >= start) cached.* = null;
        };
        for (start..self.terms.items.len) |index| {
            _ = self.arithmetic_safety_terms.remove(@intCast(index));
        }
        self.terms.shrinkRetainingCapacity(start);
    }

    fn collectLoopHeaderFacts(self: *Collector, loop_op: mlir.MlirOperation, owner: obligation.Owner) ![]const obligation.LoopHeaderFactRow {
        const block = mlir.mlirOperationGetBlock(loop_op);
        if (mlir.oraBlockIsNull(block)) return error.LoopHeaderFactPositionUnsupported;
        const projected = try self.projectLoopHeaderFactAssumptions(loop_op, owner);
        const inventory = try self.inventoryLoopHeaderFacts(loop_op);
        if (projected.len != inventory.len) return error.LoopHeaderFactInventoryMismatch;
        if (projected.len == 0) return &.{};
        const block_key = try self.headerBlockKey(block);
        const target_key = try self.operationStructuralKey(loop_op, operationName(loop_op));
        const target_position = headerOperationPosition(block, loop_op) orelse
            return error.LoopHeaderFactPositionUnsupported;
        var rows: std.ArrayList(obligation.LoopHeaderFactRow) = .empty;
        defer rows.deinit(self.allocator);
        for (projected, inventory) |assumption, item| {
            const formula = assumption.formula orelse return error.LoopHeaderFactProvenanceUnsupported;
            if (formula == .origin_value) return error.LoopHeaderFactProvenanceUnsupported;
            const origin = if (assumption.origin == .mlir_op) assumption.origin.mlir_op else return error.LoopHeaderFactProvenanceUnsupported;
            if (origin.ordinal != item.ordinal or !std.mem.eql(u8, origin.op_name, item.source_name))
                return error.LoopHeaderFactInventoryMismatch;
            var formula_values: std.ArrayList(obligation.FreeVarId) = .empty;
            defer formula_values.deinit(self.allocator);
            const set: obligation.ObligationSet = .{ .terms = self.terms.items };
            try set.collectFormulaFreeVars(self.allocator, formula, &formula_values);
            if (formula_values.items.len != item.stable_values.len) return error.LoopHeaderFactInventoryMismatch;
            for (formula_values.items) |value_id|
                if (!headerFactValuesContain(item.stable_values, value_id)) return error.LoopHeaderFactInventoryMismatch;
            for (item.stable_values) |stable| if (!optionalTypeRefEql(set.freeVarType(stable.id), stable.ty))
                return error.LoopHeaderFactInventoryMismatch;
            if ((item.source_kind == .path_assume and assumption.kind != .path_assume) or
                (item.source_kind == .function_requires and assumption.kind != .requires))
            {
                return error.LoopHeaderFactInventoryMismatch;
            }
            const id: u32 = @intCast(rows.items.len);
            try rows.append(self.allocator, .{
                .id = id,
                .structural_key = try std.fmt.allocPrint(self.allocator, "header-fact:{d}/{s}", .{ id, item.structural_key }),
                .formula = formula,
                .source_kind = item.source_kind,
                .source = .{ .structural_key = item.structural_key, .block_key = block_key, .position = item.position },
                .target = .{ .structural_key = target_key, .block_key = block_key, .position = target_position },
                .stable_values = item.stable_values,
                .assumption_id = if (item.source_kind == .function_requires) assumption.id else null,
            });
        }
        const assumption_ids = try self.assumptionIdsForLoopOwner(owner);
        var assumption_index: usize = 0;
        for (rows.items) |row| if (row.assumption_id) |id| {
            if (assumption_index >= assumption_ids.len or assumption_ids[assumption_index] != id)
                return error.LoopHeaderFactAssumptionMismatch;
            assumption_index += 1;
        };
        if (assumption_index != assumption_ids.len) return error.LoopHeaderFactAssumptionMismatch;
        return rows.toOwnedSlice(self.allocator);
    }

    fn projectLoopHeaderFactAssumptions(self: *Collector, loop_op: mlir.MlirOperation, owner: obligation.Owner) ![]const obligation.Assumption {
        var rows: std.ArrayList(obligation.Assumption) = .empty;
        defer rows.deinit(self.allocator);
        var op = mlir.oraBlockGetFirstOperation(mlir.mlirOperationGetBlock(loop_op));
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const fact = self.headerFactAssumption(op, owner) orelse continue;
            if (fact.kind == .requires or fact.kind == .path_assume)
                try rows.append(self.allocator, fact);
        }
        if (mlir.oraOperationIsNull(op)) return error.LoopHeaderFactPositionUnsupported;
        return rows.toOwnedSlice(self.allocator);
    }

    fn inventoryLoopHeaderFacts(self: *Collector, loop_op: mlir.MlirOperation) ![]const LoopHeaderFactInventory {
        const block = mlir.mlirOperationGetBlock(loop_op);
        var rows: std.ArrayList(LoopHeaderFactInventory) = .empty;
        defer rows.deinit(self.allocator);
        var prefix_supported = true;
        var position: u32 = 0;
        var op = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : ({
            op = mlir.oraOperationGetNextInBlock(op);
            position += 1;
        }) {
            const name = operationName(op);
            const key = try self.operationStructuralKey(op, name);
            if (std.mem.eql(u8, name, "ora.requires") or std.mem.eql(u8, name, "ora.assume")) {
                const source_kind = try inventoryHeaderFactSource(op, name);
                var stable: std.ArrayList(obligation.LoopHeaderFactValue) = .empty;
                defer stable.deinit(self.allocator);
                if (mlir.oraOperationGetNumOperands(op) != 1 or
                    !try self.collectStableHeaderValue(mlir.oraOperationGetOperand(op, 0), &stable, 64))
                {
                    return error.LoopHeaderFactTypeUnsupported;
                }
                try rows.append(self.allocator, .{
                    .ordinal = self.operationOrdinal(op) orelse return error.LoopHeaderFactPositionUnsupported,
                    .source_name = name,
                    .structural_key = key,
                    .source_kind = source_kind,
                    .position = position,
                    .stable_values = try stable.toOwnedSlice(self.allocator),
                });
            } else if (!self.headerPrefixOperationSupported(op, name)) {
                prefix_supported = false;
            }
        }
        if (mlir.oraOperationIsNull(op)) return error.LoopHeaderFactPositionUnsupported;
        if (rows.items.len != 0 and !prefix_supported) return error.LoopHeaderFactEffectUnsupported;
        return rows.toOwnedSlice(self.allocator);
    }

    fn headerFactAssumption(self: *const Collector, op: mlir.MlirOperation, owner: obligation.Owner) ?obligation.Assumption {
        const ordinal = self.operationOrdinal(op) orelse return null;
        const name = operationName(op);
        for (self.assumptions.items) |assumption| {
            if (!ownerEqual(assumption.owner, owner) or assumption.origin != .mlir_op) continue;
            const origin = assumption.origin.mlir_op;
            if (origin.ordinal != ordinal or !std.mem.eql(u8, origin.op_name, name)) continue;
            return assumption;
        }
        return null;
    }

    fn collectStableHeaderValue(
        self: *Collector,
        value: mlir.MlirValue,
        stable: *std.ArrayList(obligation.LoopHeaderFactValue),
        fuel: usize,
    ) !bool {
        if (fuel == 0 or mlir.oraValueIsNull(value)) return false;
        if (mlir.mlirValueIsABlockArgument(value)) {
            const index = self.functionEntryBlockArgumentNumber(value) orelse return false;
            if (index >= self.function.param_types.len or !headerFactTypeSupported(self.function.param_types[index])) {
                return false;
            }
            try appendHeaderFactValue(self.allocator, stable, .{
                .id = self.freeVarIdForFunctionParam(index),
                .ty = self.function.param_types[index],
            });
            return true;
        }
        if (!mlir.oraValueIsAOpResult(value)) return false;
        const owner = mlir.oraOpResultGetOwner(value);
        if (dynamicLengthSourceValue(owner)) |source| {
            const index = self.functionEntryBlockArgumentNumber(source) orelse return false;
            try appendHeaderFactValue(self.allocator, stable, .{
                .id = dynamicLengthFreeVarId(index),
                .ty = self.builtinTypeRef(.u256),
            });
            return true;
        }
        const name = operationName(owner);
        if (std.mem.eql(u8, name, "arith.constant")) return true;
        if (!preheaderPureValueOperationSupported(name)) return false;
        const operand_count: usize = @intCast(mlir.oraOperationGetNumOperands(owner));
        if (operand_count == 0) return false;
        for (0..operand_count) |index| {
            if (!try self.collectStableHeaderValue(mlir.oraOperationGetOperand(owner, index), stable, fuel - 1)) {
                return false;
            }
        }
        return true;
    }

    fn headerBlockKey(self: *Collector, block: mlir.MlirBlock) ![]const u8 {
        if (mlir.oraBlockIsNull(block)) return error.LoopHeaderFactPositionUnsupported;
        const first = mlir.oraBlockGetFirstOperation(block);
        const ordinal = self.operationOrdinal(first) orelse return error.LoopHeaderFactPositionUnsupported;
        return try std.fmt.allocPrint(self.allocator, "block:first-op:{d}", .{ordinal});
    }

    fn headerPrefixOperationSupported(self: *Collector, op: mlir.MlirOperation, name: []const u8) bool {
        if (self.options.project_calls and std.mem.eql(u8, name, "func.call")) {
            _ = self.callDeclaration(op) catch return false;
            return true; // Header leaves remain parameters/constants/immutable lengths.
        }
        if (preheaderPureValueOperationSupported(name) or dynamicLengthSourceValue(op) != null or
            isLoopSafetyOperation(op, name)) return true;
        if (std.mem.eql(u8, name, "memref.alloca"))
            return mlir.oraOperationGetNumResults(op) == 1 and
                headerPrivateSlot(mlir.oraOperationGetResult(op, 0));
        if (std.mem.eql(u8, name, "memref.store"))
            return mlir.oraOperationGetNumOperands(op) == 2 and
                headerPrivateSlot(mlir.oraOperationGetOperand(op, 1)) or
                inventoryMemrefStoreTargetsPersistentRoot(op);
        if (std.mem.eql(u8, name, "ora.conditional_return")) {
            const regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
            for (0..regions) |region_index| {
                var block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(op, region_index));
                while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
                    if (collectLoopBodyFacts(block).has_call) return false;
                }
            }
            return true;
        }
        if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
            for (self.loop_summary_drafts.items, 0..) |draft, index| {
                if (draft.loop_op.ptr != op.ptr) continue;
                return self.loop_summaries.items[index].projection_completeness == .complete and
                    !collectLoopBodyFacts(draft.body).has_call;
            }
            return false;
        }
        const effect = mlir.oraOperationGetMemoryEffectKind(op);
        if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_RECURSIVE or
            effect == mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN) return false;
        return effect != mlir.ORA_OPERATION_MEMORY_EFFECT_NONE and
            directRootLoopEffectSupported(op, name, effect);
    }

    fn projectLoopPreheader(
        self: *Collector,
        loop_op: mlir.MlirOperation,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !ProjectedLoopPreheader {
        if (mlir.oraBlockIsNull(self.function.entry_block)) return error.LoopPreheaderControlUnsupported;
        const draft: LoopSummaryDraft = .{ .symbol = symbol, .ordinal = ordinal };
        var result: ProjectedLoopPreheader = .{};
        var op = mlir.oraBlockGetFirstOperation(self.function.entry_block);
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            const key = self.operationStructuralKey(op, name) catch return error.LoopPreheaderControlUnsupported;
            result.operation_keys = try appendString(self.allocator, result.operation_keys, key);
            if (self.options.project_calls and std.mem.eql(u8, name, "func.call")) {
                const signature = try self.callSignature(try self.callDeclaration(op));
                if (signature.outcomes.len != 1) return error.LoopPreheaderControlUnsupported;
                const path = try self.projectSemanticCall(op, .{ .key = "prefix", .timeline = result.timeline, .timeline_inventory = result.event_inventory, .formula_overrides = result.formula_overrides }, signature.outcomes[0]);
                result.timeline = path.timeline;
                result.event_inventory = path.timeline_inventory;
                result.formula_overrides = path.formula_overrides;
                continue;
            }
            if (self.options.project_calls and
                ((std.mem.eql(u8, name, "memref.alloca") and mlir.oraOperationGetNumResults(op) == 1 and headerPrivateSlot(mlir.oraOperationGetResult(op, 0))) or
                    (std.mem.eql(u8, name, "memref.store") and mlir.oraOperationGetNumOperands(op) == 2 and headerPrivateSlot(mlir.oraOperationGetOperand(op, 1))))) continue;
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
            if (std.mem.startsWith(u8, name, "ora.mload") or std.mem.startsWith(u8, name, "ora.mstore") or
                (std.mem.eql(u8, name, "memref.load") and inventoryMemrefLoadTargetsPersistentRoot(op)) or
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
                const projected_access = if (preheaderAssumptionOperation(name))
                    null
                else
                    self.projectedDynamicAccess(
                        op,
                        formula,
                        &.{},
                    ) catch return error.LoopPreheaderValueUnsupported;
                if (projected_access) |access| {
                    result.dynamic_accesses = try appendDynamicAccessInventory(
                        self.allocator,
                        result.dynamic_accesses,
                        access,
                    );
                }
                const payload: obligation.LoopTimelineEventPayload = if (preheaderAssumptionOperation(name))
                    .{ .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(&.{formula}, result.timeline) } }
                else
                    .{ .safety_checkpoint = .{
                        .formulas = try self.loopFormulaExprsWithReads(&.{formula}, result.timeline),
                        .dynamic_index_bounds = projected_access != null,
                    } };
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
        for (self.function.param_binding_ids) |parameter_id| {
            if (obligation.freeVarIdEql(parameter_id, id)) return true;
        }
        return false;
    }

    fn inventoryLoopPreheader(
        self: *Collector,
        loop_op: mlir.MlirOperation,
    ) !InventoryLoopPreheader {
        if (mlir.oraBlockIsNull(self.function.entry_block)) return error.LoopPreheaderControlUnsupported;
        var result: InventoryLoopPreheader = .{};
        var op = mlir.oraBlockGetFirstOperation(self.function.entry_block);
        var previous_op: mlir.MlirOperation = std.mem.zeroes(mlir.MlirOperation);
        while (!mlir.oraOperationIsNull(op) and op.ptr != loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            const key = self.operationStructuralKey(op, name) catch return error.LoopPreheaderControlUnsupported;
            result.operation_keys = try appendString(self.allocator, result.operation_keys, key);
            if (self.options.project_calls and std.mem.eql(u8, name, "func.call")) {
                const signature = try self.callSignature(try self.callDeclaration(op));
                if (signature.outcomes.len != 1) return error.LoopPreheaderControlUnsupported;
                const path = try self.inventoryCall(op, .{ .key = "prefix", .timeline_inventory = result.event_inventory }, signature.outcomes[0]);
                result.event_inventory = path.timeline_inventory;
                previous_op = op;
                continue;
            }
            if (self.options.project_calls and
                ((std.mem.eql(u8, name, "memref.alloca") and mlir.oraOperationGetNumResults(op) == 1 and headerPrivateSlot(mlir.oraOperationGetResult(op, 0))) or
                    (std.mem.eql(u8, name, "memref.store") and mlir.oraOperationGetNumOperands(op) == 2 and headerPrivateSlot(mlir.oraOperationGetOperand(op, 1)))))
            {
                previous_op = op;
                continue;
            }
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

            if (try self.inventoryDynamicAccess(op, previous_op, &.{})) |access| {
                result.dynamic_accesses = try appendDynamicAccessInventory(
                    self.allocator,
                    result.dynamic_accesses,
                    access,
                );
            }

            var event: ?LoopTimelineInventoryEvent = null;
            if (std.mem.eql(u8, name, "ora.sload")) {
                if (effect != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1) return error.LoopPreheaderEffectUnsupported;
                if (isAggregateStorageCarrierLoad(op)) {
                    previous_op = op;
                    continue;
                }
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
                if (is_read and isAggregateStorageValue(value)) {
                    previous_op = op;
                    continue;
                }
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
            } else if (std.mem.startsWith(u8, name, "ora.mload") or std.mem.startsWith(u8, name, "ora.mstore") or
                (std.mem.eql(u8, name, "memref.load") and inventoryMemrefLoadTargetsPersistentRoot(op)) or
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
            previous_op = op;
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
        // Call hashes bind only after both complete body scans and DAG checking.
        if (summary.validationReason()) |reason| if (reason != .loop_call_digest_missing) try self.appendLoopSummaryReason(summary, reason);
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
        // Previously unprojectable guards are executed as ordered branches,
        // not flattened; established simple-guard rows keep their digest.
        const structured_guard = draft.loop_kind == .scf_while and
            (self.callConnectedFunction(containingFunction(draft.loop_op)) or guard == .origin_value);
        const first = mlir.oraBlockGetFirstOperation(if (structured_guard)
            mlir.oraScfWhileOpGetBeforeBlock(draft.loop_op)
        else
            draft.body);
        const initial_path_key = "body/path";
        const semantic_path: SemanticLoopPath = if (structured_guard) .{ .key = "guard", .in_guard = true, .formula_overrides = draft.entry_overrides } else .{
            .key = "body",
            .formula_overrides = draft.entry_overrides,
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
            first,
            semantic_path,
            draft,
            summary.variables,
            &semantic_terminals,
        ) catch |err| {
            if (try self.recordLoopProjectionFailure(summary, err)) return;
            return err;
        };
        self.validateSemanticLoopControlManifest(control_manifest, semantic_terminals.items) catch |err| {
            summary.projection_completeness = loopControlProjectionFailure(err) orelse return err;
            return;
        };

        var inventory_terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
        defer inventory_terminals.deinit(self.allocator);
        self.inventoryLoopSequence(
            first,
            if (structured_guard) .{ .key = "guard", .in_guard = true } else .{
                .key = "body",
                .timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, &.{}, .{
                    .structural_key = initial_path_key,
                    .tag = .path_assumption,
                }),
            },
            draft,
            summary.variables,
            &inventory_terminals,
        ) catch |err| {
            if (try self.recordLoopProjectionFailure(summary, err)) return;
            return err;
        };
        self.validateInventoryLoopControlManifest(control_manifest, inventory_terminals.items) catch |err| {
            summary.projection_completeness = loopControlProjectionFailure(err) orelse return err;
            return;
        };

        var projected: std.ArrayList(ProjectedLoopEdge) = .empty;
        defer projected.deinit(self.allocator);
        for (semantic_terminals.items) |terminal| {
            var edge = self.projectedTerminalEdge(summary, draft, terminal) catch |err| switch (err) {
                error.LoopTargetFormulaTargetMismatch => {
                    summary.projection_completeness = .target_mismatch;
                    return;
                },
                else => {
                    if (try self.recordLoopProjectionFailure(summary, err)) return;
                    return err;
                },
            };
            if (terminal.control) |control| if ((control.target == .function_return or control.target == .error_return) and self.callConnectedFunction(containingFunction(draft.loop_op))) {
                const index = try self.requireFunctionRow(containingFunction(draft.loop_op));
                var returned = terminal;
                returned.yielded = try privateReturnOperands(self.allocator, terminal.path.slots);
                const payload = try self.projectFunctionReturn(returned, self.function_summaries.items[index]);
                edge.edge.return_value = payload.value;
                edge.return_inventory = payload.inventory;
            };
            try projected.append(self.allocator, edge);
        }

        var inventory: std.ArrayList(InventoryLoopEdge) = .empty;
        defer inventory.deinit(self.allocator);
        for (inventory_terminals.items) |terminal| {
            var edge = self.inventoryTerminalEdge(summary, draft, terminal) catch |err| switch (err) {
                error.LoopTargetFormulaTargetMismatch => {
                    summary.projection_completeness = .target_mismatch;
                    return;
                },
                else => {
                    if (try self.recordLoopProjectionFailure(summary, err)) return;
                    return err;
                },
            };
            if (terminal.control) |control| if ((control.target == .function_return or control.target == .error_return) and self.callConnectedFunction(containingFunction(draft.loop_op))) {
                const index = try self.requireFunctionRow(containingFunction(draft.loop_op));
                var returned = terminal;
                returned.yielded = try privateReturnOperands(self.allocator, terminal.path.slots);
                edge.return_inventory = try self.inventoryFunctionReturn(returned, self.function_summaries.items[index]);
            };
            try inventory.append(self.allocator, edge);
        }

        if (!structured_guard) {
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
        }

        try self.loop_inventories.put(self.allocator, summary.id, try self.allocator.dupe(InventoryLoopEdge, inventory.items));
        summary.projection_completeness = compareLoopProjection(projected.items, inventory.items);
        if (summary.projection_completeness != .complete) {
            if (summary.projection_completeness == .effect_mismatch and
                loopProjectionHasIndexedRegionEvent(projected.items, inventory.items))
            {
                try self.appendLoopSummaryReason(summary, .loop_indexed_region_inventory_mismatch);
            }
            return;
        }
        if (loopProjectionHasMultipleIndexedRegionOwners(projected.items)) {
            try self.appendLoopSummaryReason(summary, .loop_indexed_region_owner_count_unsupported);
        }
        var has_storage_event = false;
        for (projected.items) |item| has_storage_event = has_storage_event or
            timelineInventoryHasStorageEvent(item.timeline_inventory);
        if (has_storage_event and self.function.write_slots_complete) {
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
        if (self.function_summaries.items.len != 0) {
            try self.removeLoopSummaryReason(summary, .loop_has_call);
        }
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

    fn recordLoopProjectionFailure(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
        err: anyerror,
    ) !bool {
        summary.projection_completeness = switch (err) {
            error.LoopControlMarkerMalformed,
            error.LoopControlMarkerTargetMismatch,
            => loopControlProjectionFailure(err).?,
            error.UnsupportedLoopControlFlow => .unsupported_control_flow,
            error.LoopProjectionPathLimitExceeded => .path_limit_exceeded,
            error.LoopProjectionEffectUnsupported => .unsupported_effect,
            error.LoopTargetFormulaTargetMismatch => .target_mismatch,
            else => if (indexedRegionProjectionFailure(err)) |reason| blk: {
                try self.appendLoopSummaryReason(summary, reason);
                break :blk .unsupported_effect;
            } else return false,
        };
        return true;
    }

    fn projectSemanticLoopSequence(
        self: *Collector,
        first: mlir.MlirOperation,
        initial_path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) anyerror!void {
        var path = initial_path;
        var op = first;
        while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            if (path.in_guard and !structuredGuardOperationSupported(op, name)) return error.LoopProjectionEffectUnsupported;
            if (path.in_guard and std.mem.eql(u8, name, "scf.condition")) {
                const values = try self.operationOperands(op);
                if (values.len != variables.len + 1) return error.UnsupportedLoopControlFlow;
                for (values[1..], 0..) |value, index|
                    if (value.ptr != mlir.oraBlockGetArgument(mlir.mlirOperationGetBlock(op), index).ptr) return error.UnsupportedLoopControlFlow;
                const value = self.sourceLoopGuardValue(values[0]) orelse return error.UnsupportedLoopControlFlow;
                const formula = try self.loopFormulaFromValueWithOverrides(value, path.formula_overrides, name, draft.symbol, draft.ordinal, 0);
                for ([_]bool{ true, false }) |enters| {
                    var branch = path;
                    branch.key = try std.fmt.allocPrint(self.allocator, "{s}/condition:{s}", .{ path.key, if (enters) "true" else "false" });
                    const condition = if (enters) formula else try self.negateFormula(formula);
                    branch.path_formulas = try appendFormula(self.allocator, path.path_formulas, condition);
                    const key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch.key});
                    branch.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{
                        .path_assumption = .{ .formulas = try loopFormulaExprs(self.allocator, &.{condition}) },
                    }, key);
                    if (enters) {
                        branch.in_guard = false;
                        try self.projectSemanticLoopSequence(mlir.oraBlockGetFirstOperation(draft.body), branch, draft, variables, terminals);
                    } else {
                        if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                        try terminals.append(self.allocator, .{ .path = branch, .terminator_key = try self.operationStructuralKey(op, name), .yielded = values[1..] });
                    }
                }
                return;
            }
            if (std.mem.eql(u8, name, "func.call")) {
                if (!self.options.project_calls) return error.LoopProjectionEffectUnsupported;
                const signature = try self.callSignature(try self.callDeclaration(op));
                for (signature.outcomes) |shape| try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), try self.projectSemanticCall(op, path, shape), draft, variables, terminals);
                return;
            }
            if (self.options.project_calls and (std.mem.eql(u8, name, "ora.error.is_error") or std.mem.eql(u8, name, "ora.error.unwrap"))) {
                if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumResults(op) != 1) return error.InvalidCallResult;
                const operand = mlir.oraOperationGetOperand(op, 0);
                const call = for (path.calls) |binding| {
                    if (binding.value_id == @intFromPtr(operand.ptr)) break binding;
                } else return error.CallResultUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                const discriminator = std.mem.eql(u8, name, "ora.error.is_error");
                if (!discriminator and (call.outcome != .success or call.results.len != 1 or !self.lifecycleCallCarrierMatches(mlir.oraValueGetType(result), call.results[0].ty))) return error.InvalidCallResult;
                const term = if (discriminator) try self.addTerm(.{ .bool_lit = call.outcome != .success }) else try self.addTerm(.{ .variable = .{ .free = .{ .id = call.results[0].id.?, .name = "call_result", .ty = call.results[0].ty } } });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{ .value_id = @intFromPtr(result.ptr), .formula = .{ .term = term } });
                path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .key = try self.directOperationResultKey(result),
                    .boolean_literal = if (discriminator) call.outcome != .success else null,
                });
                continue;
            }
            if (std.mem.eql(u8, name, "ora.loop_control")) {
                try self.appendSemanticLoopControlTerminal(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, .{ .function_return = true, .path = path, .terminator_key = try self.operationStructuralKey(op, name), .yielded = try self.operationOperands(op) });
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
            if (std.mem.eql(u8, name, "scf.if") or std.mem.eql(u8, name, "ora.conditional_return")) {
                if (try isLoopControlRuntimeGuard(op)) {
                    try self.projectSemanticLoopControlRuntimeGuard(op, path, draft, variables, terminals);
                    return;
                }
                try self.projectSemanticIf(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                try self.projectSemanticSwitch(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
                if (draft.loop_kind == .other) {
                    const child = self.nestedLoopSummaryForOperation(op) orelse return error.UnsupportedLoopControlFlow;
                    const edges = child.edges;
                    for ([_]obligation.LoopControlTarget{ .successor, .function_return, .error_return }) |target| {
                        const exists = for (edges) |edge| {
                            if (edge.target == target) break true;
                        } else false;
                        if (!exists) continue;
                        if (target != .successor) for (edges) |edge| {
                            if (edge.target == target and edge.return_value == null) return error.CallLoopReturnValueMissing;
                        };
                        var branch = try self.bindSemanticNestedLoop(op, path, true, target);
                        branch.key = try std.fmt.allocPrint(self.allocator, "{s}/loop-exit:{s}", .{ branch.key, @tagName(target) });
                        if (target == .successor) {
                            try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), branch, draft, variables, terminals);
                        } else {
                            if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                            try terminals.append(self.allocator, .{ .function_return = true, .loop_exit = true, .path = branch, .terminator_key = try self.operationStructuralKey(op, name), .yielded = &.{} });
                        }
                    }
                    return;
                }
                path = try self.bindSemanticNestedLoop(op, path, draft.loop_kind == .other, .successor);
                continue;
            }
            if (mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0) {
                return error.UnsupportedLoopControlFlow;
            }

            if (self.options.project_calls and std.mem.eql(u8, name, "memref.alloca") and mlir.oraOperationGetNumResults(op) == 1 and headerPrivateSlot(mlir.oraOperationGetResult(op, 0))) continue;
            if (self.options.project_calls and (std.mem.eql(u8, name, "memref.store") or std.mem.eql(u8, name, "memref.load")) and
                mlir.oraOperationGetNumOperands(op) == @as(usize, if (std.mem.eql(u8, name, "memref.store")) 2 else 1) and
                mlir.oraOperationGetNumResults(op) == @as(usize, if (std.mem.eql(u8, name, "memref.store")) 0 else 1) and
                headerPrivateSlot(mlir.oraOperationGetOperand(op, if (std.mem.eql(u8, name, "memref.store")) 1 else 0)))
            {
                const store = std.mem.eql(u8, name, "memref.store");
                const address = mlir.oraOperationGetOperand(op, if (store) 1 else 0);
                path.effect_keys = try appendString(self.allocator, path.effect_keys, try self.operationStructuralKey(op, name));
                if (store) {
                    path.slots = try std.mem.concat(self.allocator, PrivateSlotValue, &.{ path.slots, &.{.{ .address = @intFromPtr(address.ptr), .value = mlir.oraOperationGetOperand(op, 0) }} });
                } else {
                    var stored: ?mlir.MlirValue = null;
                    var cursor = path.slots.len;
                    while (cursor > 0) {
                        cursor -= 1;
                        if (path.slots[cursor].address != @intFromPtr(address.ptr)) continue;
                        stored = path.slots[cursor].value;
                        break;
                    }
                    const value = stored orelse try projectPrivateEntryValue(op, draft, path.slots.len != 0);
                    const result = mlir.oraOperationGetResult(op, 0);
                    const formula = try self.loopFormulaFromValueWithOverrides(value, path.formula_overrides, name, draft.symbol, 0, 0);
                    path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{ .value_id = @intFromPtr(result.ptr), .formula = formula });
                    path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{ .value_id = @intFromPtr(result.ptr), .key = try self.sourceKeyForValue(value, path.source_overrides), .boolean_literal = inventoryBooleanLiteral(value, path.source_overrides), .ty = try self.callValueType(value, path.source_overrides) });
                }
                continue;
            }
            if (self.options.project_calls and
                (std.mem.startsWith(u8, name, "ora.mload") or std.mem.startsWith(u8, name, "ora.mstore") or
                    (std.mem.eql(u8, name, "memref.load") and mlir.oraOperationGetNumOperands(op) == 1) or
                    (std.mem.eql(u8, name, "memref.store") and mlir.oraOperationGetNumOperands(op) == 2))) return error.CallPrivateSlotUnsupported;
            const effect_key = try self.operationStructuralKey(op, name);
            _ = try self.lifecycleAddressCastOperand(op);
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
            } else if (dynamicAccessSite(op)) |site| {
                const expected_effect = if (std.mem.eql(u8, name, "ora.byte_at"))
                    mlir.ORA_OPERATION_MEMORY_EFFECT_NONE
                else
                    mlir.ORA_OPERATION_MEMORY_EFFECT_READ;
                if (effect_kind != expected_effect) {
                    if (effect_kind == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE)
                        return error.LoopIndexedRegionWriteUnreviewed;
                    return error.LoopProjectionEffectUnsupported;
                }
                const result = mlir.oraOperationGetResult(op, 0);
                const owner = try self.projectedIndexedRegionOwner(site.base);
                const index_source = try self.sourceKeyForValue(site.index, path.source_overrides);
                const owner_source = try self.dynamicAccessOwnerIdentity(site.base, path.source_overrides);
                const access = latestMatchingDynamicAccess(
                    path.dynamic_accesses,
                    owner_source,
                    index_source,
                ) orelse return error.LoopIndexedRegionCheckpointMissing;
                const index_formula = try self.loopFormulaFromValueWithOverrides(
                    site.index,
                    path.formula_overrides,
                    name,
                    draft.symbol,
                    draft.ordinal,
                    1,
                );
                if (index_formula == .origin_value) return error.LoopProjectionEffectUnsupported;
                const element_ty = self.exactLoopIntegerTypeForValue(result) catch
                    return error.LoopIndexedRegionTypeUnsupported;
                if (typeRefIntegerInfo(element_ty) == null)
                    return error.LoopIndexedRegionTypeUnsupported;
                const read_id = try self.storageReadId(op);
                const read_term = try self.addTerm(.{ .variable = .{ .free = .{
                    .id = read_id,
                    .name = effect_key,
                    .ty = element_ty,
                } } });
                const event: obligation.LoopIndexedRegionReadEvent = .{
                    .result = read_id,
                    .region = .calldata,
                    .owner = owner,
                    .index = .{
                        .formula = index_formula,
                        .read_results = try self.readResultsForFormula(index_formula, path.timeline),
                    },
                    .element_ty = element_ty,
                    .checkpoint_structural_key = access.structural_key,
                };
                path.timeline = try appendLoopTimelineEvent(
                    self.allocator,
                    path.timeline,
                    .{ .indexed_region_read = event },
                    effect_key,
                );
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .indexed_region_read,
                    .indexed_result = read_id,
                    .indexed_region = .calldata,
                    .indexed_owner = owner,
                    .indexed_index_source = index_source,
                    .indexed_element_ty = element_ty,
                    .indexed_checkpoint_key = access.structural_key,
                });
                path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .formula = .{ .term = read_term },
                });
                path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .key = effect_key,
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
            if (self.function_summaries.items.len != 0 and std.mem.eql(u8, name, "ora.assume")) {
                const formula = try self.loopFormulaFromValueWithOverrides(mlir.oraOperationGetOperand(op, 0), path.formula_overrides, name, draft.symbol, 0, 0);
                path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{ .path_assumption = .{ .formulas = try self.loopFormulaExprsWithReads(&.{formula}, path.timeline) } }, effect_key);
            }
            if (draftHasSafetySite(draft, op)) {
                if (mlir.oraOperationGetNumOperands(op) == 0) return error.LoopProjectionEffectUnsupported;
                if (arithmeticSafetyFromAssertOp(op) == .dynamic_index_bounds) {
                    const check = dynamicAccessCheckFromAssert(op) orelse
                        return error.LoopIndexedRegionCheckpointMissing;
                    _ = self.projectedIndexedRegionOwner(check.owner) catch |err| {
                        const next = mlir.oraOperationGetNextInBlock(op);
                        if (!mlir.oraOperationIsNull(next) and dynamicAccessSite(next) != null and
                            mlir.oraOperationGetMemoryEffectKind(next) == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE)
                        {
                            return error.LoopIndexedRegionWriteUnreviewed;
                        }
                        return err;
                    };
                }
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
                const projected_access = try self.projectedDynamicAccess(
                    op,
                    formula,
                    path.source_overrides,
                );
                if (projected_access) |access| {
                    path.dynamic_accesses = try appendDynamicAccessInventory(
                        self.allocator,
                        path.dynamic_accesses,
                        access,
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
                    .safety_checkpoint = .{
                        .formulas = try self.loopFormulaExprsWithReads(&.{formula}, path.timeline),
                        .dynamic_index_bounds = projected_access != null,
                    },
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
        variables: []const obligation.LoopVariable,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) anyerror!void {
        var path = initial_path;
        var op = first;
        var previous_op: mlir.MlirOperation = std.mem.zeroes(mlir.MlirOperation);
        while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
            const name = operationName(op);
            if (path.in_guard and !structuredGuardOperationSupported(op, name)) return error.LoopProjectionEffectUnsupported;
            if (path.in_guard and std.mem.eql(u8, name, "scf.condition")) {
                const values = try self.operationOperands(op);
                if (values.len != variables.len + 1) return error.UnsupportedLoopControlFlow;
                for (values[1..], 0..) |value, index|
                    if (value.ptr != mlir.oraBlockGetArgument(mlir.mlirOperationGetBlock(op), index).ptr) return error.UnsupportedLoopControlFlow;
                for ([_]bool{ true, false }) |enters| {
                    var branch = path;
                    branch.key = try std.fmt.allocPrint(self.allocator, "{s}/condition:{s}", .{ path.key, if (enters) "true" else "false" });
                    branch.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                        .structural_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{branch.key}),
                        .tag = .path_assumption,
                    });
                    if (enters) {
                        branch.in_guard = false;
                        try self.inventoryLoopSequence(mlir.oraBlockGetFirstOperation(draft.body), branch, draft, variables, terminals);
                    } else {
                        if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                        try terminals.append(self.allocator, .{ .path = branch, .terminator_key = try self.operationStructuralKey(op, name), .yielded = values[1..] });
                    }
                }
                return;
            }
            if (std.mem.eql(u8, name, "func.call")) {
                if (!self.options.project_calls) return error.LoopProjectionEffectUnsupported;
                const signature = try self.callSignature(try self.callDeclaration(op));
                for (signature.outcomes) |shape| try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), try self.inventoryCall(op, path, shape), draft, variables, terminals);
                return;
            }
            if (self.options.project_calls and (std.mem.eql(u8, name, "ora.error.is_error") or std.mem.eql(u8, name, "ora.error.unwrap"))) {
                if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumResults(op) != 1) return error.InvalidCallResult;
                const operand = mlir.oraOperationGetOperand(op, 0);
                const call = for (path.calls) |binding| {
                    if (binding.value_id == @intFromPtr(operand.ptr)) break binding;
                } else return error.CallResultUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                const discriminator = std.mem.eql(u8, name, "ora.error.is_error");
                if (!discriminator and (call.outcome != .success or call.results.len != 1 or !self.lifecycleCallCarrierMatches(mlir.oraValueGetType(result), call.results[0].ty))) return error.InvalidCallResult;
                path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .key = try self.directOperationResultKey(result),
                    .boolean_literal = if (discriminator) call.outcome != .success else null,
                });
                continue;
            }
            if (std.mem.eql(u8, name, "ora.loop_control")) {
                try self.appendInventoryLoopControlTerminal(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, .{ .function_return = true, .path = path, .terminator_key = try self.operationStructuralKey(op, name), .yielded = try self.operationOperands(op) });
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
            if (std.mem.eql(u8, name, "scf.if") or std.mem.eql(u8, name, "ora.conditional_return")) {
                if (try isLoopControlRuntimeGuard(op)) {
                    try self.inventoryLoopControlRuntimeGuard(op, path, draft, variables, terminals);
                    return;
                }
                try self.inventoryIf(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                try self.inventorySwitch(op, path, draft, variables, terminals);
                return;
            }
            if (std.mem.eql(u8, name, "scf.while") or std.mem.eql(u8, name, "scf.for")) {
                if (draft.loop_kind == .other) {
                    const child = self.nestedLoopSummaryForOperation(op) orelse return error.UnsupportedLoopControlFlow;
                    const edges = self.loop_inventories.get(child.id) orelse return error.CallInventoryMismatch;
                    for ([_]obligation.LoopControlTarget{ .successor, .function_return, .error_return }) |target| {
                        const exists = for (edges) |edge| {
                            if (edge.target == target) break true;
                        } else false;
                        if (!exists) continue;
                        if (target != .successor) for (edges) |edge| {
                            if (edge.target == target and edge.return_inventory == null) return error.CallLoopReturnValueMissing;
                        };
                        var branch = try self.bindInventoryNestedLoop(op, path, true, target);
                        branch.key = try std.fmt.allocPrint(self.allocator, "{s}/loop-exit:{s}", .{ branch.key, @tagName(target) });
                        if (target == .successor) {
                            try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), branch, draft, variables, terminals);
                        } else {
                            if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                            try terminals.append(self.allocator, .{ .function_return = true, .loop_exit = true, .path = branch, .terminator_key = try self.operationStructuralKey(op, name), .yielded = &.{} });
                        }
                    }
                    return;
                }
                path = try self.bindInventoryNestedLoop(op, path, draft.loop_kind == .other, .successor);
                previous_op = op;
                continue;
            }
            if (mlir.oraOperationGetNumRegions(op) != 0 or mlir.mlirOperationGetNumSuccessors(op) != 0) {
                return error.UnsupportedLoopControlFlow;
            }

            if (self.options.project_calls and std.mem.eql(u8, name, "memref.alloca") and mlir.oraOperationGetNumResults(op) == 1 and headerPrivateSlot(mlir.oraOperationGetResult(op, 0))) continue;
            if (self.options.project_calls and (std.mem.eql(u8, name, "memref.store") or std.mem.eql(u8, name, "memref.load")) and
                mlir.oraOperationGetNumOperands(op) == @as(usize, if (std.mem.eql(u8, name, "memref.store")) 2 else 1) and
                mlir.oraOperationGetNumResults(op) == @as(usize, if (std.mem.eql(u8, name, "memref.store")) 0 else 1) and
                headerPrivateSlot(mlir.oraOperationGetOperand(op, if (std.mem.eql(u8, name, "memref.store")) 1 else 0)))
            {
                const store = std.mem.eql(u8, name, "memref.store");
                const address = mlir.oraOperationGetOperand(op, if (store) 1 else 0);
                path.effect_keys = try appendString(self.allocator, path.effect_keys, try self.operationStructuralKey(op, name));
                if (store) {
                    path.slots = try std.mem.concat(self.allocator, PrivateSlotValue, &.{ path.slots, &.{.{ .address = @intFromPtr(address.ptr), .value = mlir.oraOperationGetOperand(op, 0) }} });
                } else {
                    var stored: ?mlir.MlirValue = null;
                    var cursor = path.slots.len;
                    while (cursor > 0) {
                        cursor -= 1;
                        if (path.slots[cursor].address != @intFromPtr(address.ptr)) continue;
                        stored = path.slots[cursor].value;
                        break;
                    }
                    const value = stored orelse try inventoryPrivateEntryValue(op, draft, path.slots.len != 0);
                    const result = mlir.oraOperationGetResult(op, 0);
                    path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{ .value_id = @intFromPtr(result.ptr), .key = try self.sourceKeyForValue(value, path.source_overrides), .boolean_literal = inventoryBooleanLiteral(value, path.source_overrides), .ty = try self.callValueType(value, path.source_overrides) });
                }
                continue;
            }
            if (self.options.project_calls and
                (std.mem.startsWith(u8, name, "ora.mload") or std.mem.startsWith(u8, name, "ora.mstore") or
                    (std.mem.eql(u8, name, "memref.load") and mlir.oraOperationGetNumOperands(op) == 1) or
                    (std.mem.eql(u8, name, "memref.store") and mlir.oraOperationGetNumOperands(op) == 2))) return error.CallPrivateSlotUnsupported;
            const effect_key = try self.operationStructuralKey(op, name);
            const dynamic_access = try self.inventoryDynamicAccess(op, previous_op, path.source_overrides);
            if (dynamic_access) |access| {
                path.dynamic_accesses = try appendDynamicAccessInventory(
                    self.allocator,
                    path.dynamic_accesses,
                    access,
                );
            }
            _ = try self.lifecycleAddressCastOperand(op);
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
                if (isAggregateStorageCarrierLoad(op)) {
                    previous_op = op;
                    continue;
                }
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
            } else if (dynamicAccessSite(op)) |site| {
                const expected_effect = if (std.mem.eql(u8, name, "ora.byte_at"))
                    mlir.ORA_OPERATION_MEMORY_EFFECT_NONE
                else
                    mlir.ORA_OPERATION_MEMORY_EFFECT_READ;
                if (effect_kind != expected_effect) {
                    if (effect_kind == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE)
                        return error.LoopIndexedRegionWriteUnreviewed;
                    return error.LoopProjectionEffectUnsupported;
                }
                const result = mlir.oraOperationGetResult(op, 0);
                const owner = try self.inventoryIndexedRegionOwner(site.base);
                const element_mlir_ty = if (std.mem.eql(u8, name, "memref.load"))
                    mlir.oraShapedTypeGetElementType(mlir.oraValueGetType(site.base))
                else if (std.mem.eql(u8, name, "ora.byte_at"))
                    mlir.oraValueGetType(result)
                else
                    return error.LoopIndexedRegionTypeUnsupported;
                const element_ty = try self.typeRefFromMlirType(element_mlir_ty);
                if (typeRefIntegerInfo(element_ty) == null)
                    return error.LoopIndexedRegionTypeUnsupported;
                const read_id = try self.storageReadId(op);
                path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{
                    .structural_key = effect_key,
                    .tag = .indexed_region_read,
                    .indexed_result = read_id,
                    .indexed_region = .calldata,
                    .indexed_owner = owner,
                    .indexed_index_source = try self.sourceKeyForValue(site.index, path.source_overrides),
                    .indexed_element_ty = element_ty,
                    .indexed_checkpoint_key = (dynamic_access orelse
                        return error.LoopIndexedRegionCheckpointMissing).structural_key,
                });
                path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{
                    .value_id = @intFromPtr(result.ptr),
                    .key = effect_key,
                });
            } else if (std.mem.eql(u8, name, "ora.map_get")) {
                if (effect_kind != mlir.ORA_OPERATION_MEMORY_EFFECT_READ or
                    mlir.oraOperationGetNumResults(op) != 1 or
                    mlir.oraOperationGetNumOperands(op) != 2) return error.LoopProjectionEffectUnsupported;
                const result = mlir.oraOperationGetResult(op, 0);
                if (isAggregateStorageValue(result)) {
                    previous_op = op;
                    continue;
                }
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
            if (self.function_summaries.items.len != 0 and std.mem.eql(u8, name, "ora.assume")) path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{ .structural_key = effect_key, .tag = .path_assumption });
            if (isLoopSafetyOperation(op, name) or (draft.loop_kind == .other and isLoopTargetOperation(op, name))) {
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
            if (draft.loop_kind != .other and isLoopTargetOperation(op, name)) {
                path.target_keys = try appendString(self.allocator, path.target_keys, effect_key);
            }
            previous_op = op;
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
        variables: []const obligation.LoopVariable,
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
        try validateLoopControlStateTypes(variables, operands);
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
        variables: []const obligation.LoopVariable,
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
        try validateLoopControlStateTypes(variables, operands);
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
        variables: []const obligation.LoopVariable,
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
            variables,
            &guarded_terminals,
        );
        for (guarded_terminals.items) |terminal| {
            if (terminal.control != null or terminal.function_return) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, terminal);
                continue;
            }
            const continued = try self.bindSemanticRegionResults(op, terminal);
            try self.projectSemanticLoopSequence(
                mlir.oraOperationGetNextInBlock(op),
                continued,
                draft,
                variables,
                terminals,
            );
        }
    }

    fn inventoryLoopControlRuntimeGuard(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
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
            variables,
            &guarded_terminals,
        );
        for (guarded_terminals.items) |terminal| {
            if (terminal.control != null or terminal.function_return) {
                if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                try terminals.append(self.allocator, terminal);
                continue;
            }
            const continued = try self.bindInventoryRegionResults(op, terminal);
            try self.inventoryLoopSequence(
                mlir.oraOperationGetNextInBlock(op),
                continued,
                draft,
                variables,
                terminals,
            );
        }
    }

    fn projectSemanticIf(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
        terminals: *std.ArrayList(SemanticLoopTerminal),
    ) anyerror!void {
        if (mlir.oraOperationGetNumOperands(op) == 0) return error.UnsupportedLoopControlFlow;
        const condition = try self.loopFormulaFromValueWithOverrides(
            mlir.oraOperationGetOperand(op, 0),
            path.formula_overrides,
            operationName(op),
            draft.symbol,
            draft.ordinal,
            0,
        );
        const branch_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        if (branch_count == 0 or branch_count > 2) return error.UnsupportedLoopControlFlow;
        if (branch_count == 1 and mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
        const op_key = try self.operationStructuralKey(op, operationName(op));

        for (0..branch_count) |branch_index| {
            // The prefix remains on the selected path, including every earlier
            // checkpoint. Only an already-bound literal selects a single arm.
            if (condition == .term and self.termIsBoolLiteral(condition.term, branch_index != 0)) continue;
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
            try self.projectSemanticLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, variables, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null or branch_terminal.function_return) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindSemanticRegionResults(op, branch_terminal);
                try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, variables, terminals);
            }
        }

        if (branch_count == 1 and !(condition == .term and self.termIsBoolLiteral(condition.term, true))) {
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
            try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), false_path, draft, variables, terminals);
        }
    }

    fn inventoryIf(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
        terminals: *std.ArrayList(InventoryLoopTerminal),
    ) anyerror!void {
        const branch_count: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        if (mlir.oraOperationGetNumOperands(op) == 0 or branch_count == 0 or branch_count > 2) {
            return error.UnsupportedLoopControlFlow;
        }
        if (branch_count == 1 and mlir.oraOperationGetNumResults(op) != 0) return error.UnsupportedLoopControlFlow;
        const op_key = try self.operationStructuralKey(op, operationName(op));
        const literal = inventoryBooleanLiteral(mlir.oraOperationGetOperand(op, 0), path.source_overrides);
        for (0..branch_count) |branch_index| {
            if (literal) |value| if (value != (branch_index == 0)) continue;
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
            try self.inventoryLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, variables, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null or branch_terminal.function_return) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindInventoryRegionResults(op, branch_terminal);
                try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, variables, terminals);
            }
        }
        if (branch_count == 1 and literal != true) {
            var false_path = path;
            false_path.key = try std.fmt.allocPrint(self.allocator, "{s}/{s}/else-implicit", .{ path.key, op_key });
            false_path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, false_path.timeline_inventory, .{
                .structural_key = try std.fmt.allocPrint(self.allocator, "{s}/path", .{false_path.key}),
                .tag = .path_assumption,
            });
            try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), false_path, draft, variables, terminals);
        }
    }

    fn projectSemanticSwitch(
        self: *Collector,
        op: mlir.MlirOperation,
        path: SemanticLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
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
            try self.projectSemanticLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, variables, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null or branch_terminal.function_return) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindSemanticRegionResults(op, branch_terminal);
                try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, variables, terminals);
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
            try self.projectSemanticLoopSequence(mlir.oraOperationGetNextInBlock(op), no_match, draft, variables, terminals);
        }
    }

    fn inventorySwitch(
        self: *Collector,
        op: mlir.MlirOperation,
        path: InventoryLoopPath,
        draft: LoopSummaryDraft,
        variables: []const obligation.LoopVariable,
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
            try self.inventoryLoopSequence(mlir.oraBlockGetFirstOperation(block), branch_path, draft, variables, &branch_terminals);
            for (branch_terminals.items) |branch_terminal| {
                if (branch_terminal.control != null or branch_terminal.function_return) {
                    if (terminals.items.len >= max_loop_projection_paths) return error.LoopProjectionPathLimitExceeded;
                    try terminals.append(self.allocator, branch_terminal);
                    continue;
                }
                const continued = try self.bindInventoryRegionResults(op, branch_terminal);
                try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), continued, draft, variables, terminals);
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
            try self.inventoryLoopSequence(mlir.oraOperationGetNextInBlock(op), no_match, draft, variables, terminals);
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
                .boolean_literal = inventoryBooleanLiteral(terminal.yielded[index], path.source_overrides),
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

    fn bindSemanticNestedLoop(self: *Collector, op: mlir.MlirOperation, initial: SemanticLoopPath, lexical: bool, target: obligation.LoopControlTarget) !SemanticLoopPath {
        const inner = self.nestedLoopSummaryForOperation(op) orelse return error.UnsupportedLoopControlFlow;
        if (inner.projection_completeness != .complete or (!lexical and inner.unsupported_reasons.len != 0) or
            (!lexical and !nestedLoopHasOnlySuccessorExit(inner))) return error.UnsupportedLoopControlFlow;
        const count: usize = @intCast(mlir.oraOperationGetNumResults(op));
        const start: usize = @intFromBool(inner.loop_kind == .scf_for);
        if (count + start != inner.variables.len) return error.UnsupportedLoopControlFlow;
        const exports = try self.allocator.dupe(obligation.LoopVariable, if (target == .successor) inner.variables[start..] else &.{});
        if (lexical) for (exports, 0..) |*variable, index| {
            variable.id = .{ .file_id = synthetic_file_id - 4 - (self.operationOrdinal(op) orelse return error.UnsupportedLoopControlFlow), .pattern_id = @intCast(index) };
        };
        var path = initial;
        const key = try self.operationStructuralKey(op, operationName(op));
        const dependency: obligation.LoopSummaryDependency = .{
            .id = @intCast(path.summary_dependencies.len),
            .structural_key = key,
            .summary_id = inner.id,
            .loop_vc_sha256 = @splat(0),
            .write_footprint_complete = true,
            .target = target,
            .result_bindings = if (lexical) exports else null,
        };
        path.summary_dependencies = try appendLoopSummaryDependency(self.allocator, path.summary_dependencies, dependency);
        path.summary_dependency_keys = try appendString(self.allocator, path.summary_dependency_keys, key);
        path.timeline = try appendLoopTimelineEvent(self.allocator, path.timeline, .{ .summary_dependency = dependency }, key);
        for (exports, 0..) |variable, index| {
            const result = mlir.oraOperationGetResult(op, index);
            const source = try std.fmt.allocPrint(self.allocator, "{s}/final-variable:{d}", .{ key, variable.index });
            path.state_source_keys = try appendString(self.allocator, path.state_source_keys, try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ try self.directOperationResultKey(result), source }));
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{ .value_id = @intFromPtr(result.ptr), .key = source });
            const term = try self.addTerm(.{ .variable = .{ .free = .{ .id = variable.id.?, .name = variable.name orelse "nested_loop_result", .ty = variable.ty } } });
            path.formula_overrides = try appendFormulaOverride(self.allocator, path.formula_overrides, .{ .value_id = @intFromPtr(result.ptr), .formula = .{ .term = term } });
        }
        return path;
    }

    fn bindInventoryNestedLoop(self: *Collector, op: mlir.MlirOperation, initial: InventoryLoopPath, lexical: bool, target: obligation.LoopControlTarget) !InventoryLoopPath {
        const child = self.nestedLoopSummaryForOperation(op) orelse return error.UnsupportedLoopControlFlow;
        if (child.projection_completeness != .complete or (!lexical and child.unsupported_reasons.len != 0) or
            (!lexical and !nestedLoopHasOnlySuccessorExit(child))) return error.UnsupportedLoopControlFlow;
        const count: usize = @intCast(mlir.oraOperationGetNumResults(op));
        const first: usize = @intFromBool(child.loop_kind == .scf_for);
        if (count + first != child.variables.len) return error.UnsupportedLoopControlFlow;
        const bindings = try self.allocator.dupe(obligation.LoopVariable, if (target == .successor) child.variables[first..] else &.{});
        if (lexical) for (bindings, 0..) |*variable, index| {
            variable.id = .{ .file_id = synthetic_file_id - 4 - (self.operationOrdinal(op) orelse return error.UnsupportedLoopControlFlow), .pattern_id = @intCast(index) };
        };
        var path = initial;
        const key = try self.operationStructuralKey(op, operationName(op));
        const dependency: obligation.LoopSummaryDependency = .{
            .id = @intCast(path.summary_dependency_keys.len),
            .structural_key = key,
            .summary_id = child.id,
            .loop_vc_sha256 = @splat(0),
            .write_footprint_complete = true,
            .target = target,
            .result_bindings = if (lexical) bindings else null,
        };
        path.summary_dependency_keys = try appendString(self.allocator, path.summary_dependency_keys, key);
        path.timeline_inventory = try appendLoopTimelineInventoryEvent(self.allocator, path.timeline_inventory, .{ .structural_key = key, .tag = .summary_dependency, .summary_dependency = dependency });
        for (bindings, 0..) |variable, index| {
            const result = mlir.oraOperationGetResult(op, index);
            const source = try std.fmt.allocPrint(self.allocator, "{s}/final-variable:{d}", .{ key, variable.index });
            path.state_source_keys = try appendString(self.allocator, path.state_source_keys, try std.fmt.allocPrint(self.allocator, "{s}<-{s}", .{ try self.directOperationResultKey(result), source }));
            path.source_overrides = try appendValueSource(self.allocator, path.source_overrides, .{ .value_id = @intFromPtr(result.ptr), .key = source });
        }
        return path;
    }

    fn projectedTerminalEdge(
        self: *Collector,
        summary: *const obligation.LoopSummaryRow,
        draft: LoopSummaryDraft,
        terminal: SemanticLoopTerminal,
    ) anyerror!ProjectedLoopEdge {
        if (terminal.function_return) {
            if (!self.options.project_calls) return error.UnsupportedLoopControlFlow;
            const index = try self.requireFunctionRow(containingFunction(draft.loop_op));
            const returned = try self.projectFunctionReturn(terminal, self.function_summaries.items[index]);
            var ending = terminal;
            ending.function_return = false;
            const state = try self.allocator.alloc(mlir.MlirValue, summary.variables.len);
            for (state, 0..) |*value, i| value.* = loopVariableValue(draft.loop_op, draft.loop_kind, @intCast(i));
            ending.yielded = state;
            ending.abrupt_target = if (returned.inventory.outcome == .success) .function_return else .error_return;
            var edge = try self.projectedTerminalEdge(summary, draft, ending);
            edge.edge.return_value = returned.value;
            edge.return_inventory = returned.inventory;
            return edge;
        }
        const target = terminal.abrupt_target orelse if (terminal.control) |control| control.target else if (terminal.path.in_guard) obligation.LoopControlTarget.successor else .backedge;
        const is_return_target = target == .function_return or target == .error_return;
        if (!is_return_target and self.options.project_calls) for (terminal.path.slots) |slot| {
            var owner = mlir.oraOpResultGetOwner(.{ .ptr = @ptrFromInt(slot.address) });
            while (!mlir.oraOperationIsNull(owner) and owner.ptr != draft.loop_op.ptr) owner = mlir.mlirOperationGetParentOperation(owner);
            if (mlir.oraOperationIsNull(owner)) return error.CallPrivateLoopStateUnsupported;
        };
        if (!is_return_target and terminal.path.target_keys.len != 0) {
            return error.LoopTargetFormulaTargetMismatch;
        }
        const complete_control_state = terminal.control != null or terminal.abrupt_target != null;
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
        const structural_key = if (terminal.abrupt_target != null)
            try std.fmt.allocPrint(self.allocator, "{s}/return:{s}", .{ terminal.path.key, terminal.terminator_key })
        else if (terminal.control) |control|
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
            .dynamic_accesses = terminal.path.dynamic_accesses,
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
    ) anyerror!InventoryLoopEdge {
        if (terminal.function_return) {
            if (!self.options.project_calls) return error.UnsupportedLoopControlFlow;
            const index = try self.requireFunctionRow(containingFunction(draft.loop_op));
            const returned = try self.inventoryFunctionReturn(terminal, self.function_summaries.items[index]);
            var ending = terminal;
            ending.function_return = false;
            const state = try self.allocator.alloc(mlir.MlirValue, summary.variables.len);
            for (state, 0..) |*value, i| value.* = loopVariableValue(draft.loop_op, draft.loop_kind, @intCast(i));
            ending.yielded = state;
            ending.abrupt_target = if (returned.outcome == .success) .function_return else .error_return;
            var edge = try self.inventoryTerminalEdge(summary, draft, ending);
            edge.return_inventory = returned;
            return edge;
        }
        const target = terminal.abrupt_target orelse if (terminal.control) |control| control.target else if (terminal.path.in_guard) obligation.LoopControlTarget.successor else .backedge;
        const is_return_target = target == .function_return or target == .error_return;
        if (!is_return_target and self.options.project_calls) for (terminal.path.slots) |slot| {
            var owner = mlir.oraOpResultGetOwner(.{ .ptr = @ptrFromInt(slot.address) });
            while (!mlir.oraOperationIsNull(owner) and owner.ptr != draft.loop_op.ptr) owner = mlir.mlirOperationGetParentOperation(owner);
            if (mlir.oraOperationIsNull(owner)) return error.CallPrivateLoopStateUnsupported;
        };
        if (!is_return_target and terminal.path.target_keys.len != 0) {
            return error.LoopTargetFormulaTargetMismatch;
        }
        const complete_control_state = terminal.control != null or terminal.abrupt_target != null;
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
        const structural_key = if (terminal.abrupt_target != null)
            try std.fmt.allocPrint(self.allocator, "{s}/return:{s}", .{ terminal.path.key, terminal.terminator_key })
        else if (terminal.control) |control|
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
            .dynamic_accesses = terminal.path.dynamic_accesses,
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

    fn projectedDynamicAccess(
        self: *Collector,
        assert_op: mlir.MlirOperation,
        formula: obligation.FormulaRef,
        source_overrides: []const LoopValueSource,
    ) !?LoopDynamicAccessInventory {
        if (arithmeticSafetyFromAssertOp(assert_op) != .dynamic_index_bounds) return null;
        const formula_id = switch (formula) {
            .term => |id| id,
            .origin_value => return error.LoopProjectionEffectUnsupported,
        };
        if (formula_id >= self.terms.items.len) return error.LoopProjectionEffectUnsupported;
        const binary = switch (self.terms.items[formula_id]) {
            .binary => |value| value,
            else => return error.LoopProjectionEffectUnsupported,
        };
        const ty = typeRefIntegerInfo(binary.ty) orelse return error.LoopProjectionEffectUnsupported;
        if (binary.op != .lt or ty.signed or ty.width != 256) {
            return error.LoopProjectionEffectUnsupported;
        }
        const check = dynamicAccessCheckFromAssert(assert_op) orelse
            return error.LoopProjectionEffectUnsupported;
        return .{
            .structural_key = try self.operationStructuralKey(assert_op, operationName(assert_op)),
            .owner_identity = try self.dynamicAccessOwnerIdentity(check.owner, source_overrides),
            .index_identity = try self.sourceKeyForValue(check.index, source_overrides),
        };
    }

    fn inventoryDynamicAccess(
        self: *Collector,
        access_op: mlir.MlirOperation,
        assert_op: mlir.MlirOperation,
        source_overrides: []const LoopValueSource,
    ) !?LoopDynamicAccessInventory {
        const site = dynamicAccessSite(access_op) orelse return null;
        if (arithmeticSafetyFromAssertOp(assert_op) != .dynamic_index_bounds) {
            return error.LoopProjectionEffectUnsupported;
        }
        return .{
            .structural_key = try self.operationStructuralKey(assert_op, operationName(assert_op)),
            .owner_identity = try self.dynamicAccessOwnerIdentity(site.base, source_overrides),
            .index_identity = try self.sourceKeyForValue(site.index, source_overrides),
        };
    }

    fn dynamicAccessOwnerIdentity(
        self: *Collector,
        value: mlir.MlirValue,
        overrides: []const LoopValueSource,
    ) ![]const u8 {
        return self.sourceKeyForValue(value, overrides) catch |err| switch (err) {
            error.UnsupportedLoopControlFlow => {
                if (!mlir.oraValueIsAOpResult(value)) return err;
                return self.directOperationResultKey(value);
            },
            else => return err,
        };
    }

    fn projectedIndexedRegionOwner(
        self: *Collector,
        base: mlir.MlirValue,
    ) !obligation.FreeVarId {
        if (!mlir.mlirValueIsABlockArgument(base)) return indexedRegionOwnerFailure(base);
        const parameter_index = self.functionEntryBlockArgumentNumber(base) orelse
            return error.LoopIndexedRegionOwnerUnsupported;
        if (parameter_index >= self.function.param_binding_ids.len or
            !dynamicSequenceTypeHasLogicalLength(mlir.oraValueGetType(base)))
        {
            return error.LoopIndexedRegionOwnerUnsupported;
        }
        return self.function.param_binding_ids[parameter_index];
    }

    fn inventoryIndexedRegionOwner(
        self: *Collector,
        base: mlir.MlirValue,
    ) !obligation.FreeVarId {
        if (!mlir.mlirValueIsABlockArgument(base)) return indexedRegionOwnerFailure(base);
        if (mlir.oraBlockIsNull(self.function.entry_block) or
            mlir.mlirBlockArgumentGetOwner(base).ptr != self.function.entry_block.ptr)
        {
            return error.LoopIndexedRegionOwnerUnsupported;
        }
        const raw_index = mlir.mlirBlockArgumentGetArgNumber(base);
        if (raw_index < 0) return error.LoopIndexedRegionOwnerUnsupported;
        const parameter_index: usize = @intCast(raw_index);
        if (parameter_index >= self.function.param_binding_ids.len or
            !dynamicSequenceTypeHasLogicalLength(mlir.oraValueGetType(base)))
        {
            return error.LoopIndexedRegionOwnerUnsupported;
        }
        return self.function.param_binding_ids[parameter_index];
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
        if (self.lifecycle_scope != null) {
            if (addressTypeRef(ty)) {
                if (!addressCarrier(mlir.oraValueGetType(value))) return error.LifecycleStorageWidthMismatch;
                return .{ .root = root, .value_ty = ty };
            }
            const info = typeRefIntegerInfo(ty) orelse return error.LifecycleStorageTypeMissing;
            if (!mlirIntegerCarrierHasWidth(mlir.oraValueGetType(value), info.width)) return error.LifecycleStorageWidthMismatch;
            return .{ .root = root, .value_ty = ty };
        }
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
            const ty = try self.lifecycleMapKeyType(key);
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
            .ty = try self.lifecycleMapKeyType(key),
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
            if (self.lifecycle_scope != null) {
                const declared_key = mlir.oraMapTypeGetKeyType(mlir.oraValueGetType(current));
                const actual_key = mlir.oraValueGetType(reversed.items[reversed.items.len - 1]);
                // Nominal address keys cannot be replaced with an integer carrier.
                if (addressCarrier(declared_key) != addressCarrier(actual_key)) return error.LoopProjectionEffectUnsupported;
            }
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
        for (timeline) |event| for (0..obligation.loopTimelineReadCount(event.payload)) |index| {
            const id = obligation.loopTimelineReadId(event.payload, index).?;
            if (set.formulaReferencesFreeVar(formula, id)) try results.append(self.allocator, id);
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
            .term => |term| .{ .term = try self.addTerm(if (self.terms.items[term] == .bool_lit)
                .{ .bool_lit = !self.terms.items[term].bool_lit }
            else
                .{ .unary = .{ .op = .not, .operand = term } }) },
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
        if (self.function.param_names.len == 0) return &.{};
        var variables: std.ArrayList(obligation.LoopVariable) = .empty;
        for (self.function.param_names, 0..) |name, index| {
            const ty = if (index < self.function.param_types.len)
                self.function.param_types[index]
            else
                try self.unknownTypeRef();
            // The scalar LoopVC environment contains only registered integer
            // parameters. A formula that actually references any omitted
            // parameter remains non-denotable and therefore fails closed;
            // unrelated slice or aggregate parameters must not exclude an
            // otherwise self-contained integer loop.
            if (!headerFactTypeSupported(ty)) continue;
            try variables.append(self.allocator, .{
                .index = @intCast(index),
                .id = self.freeVarIdForFunctionParam(index),
                .name = name,
                .ty = ty,
            });
        }
        if (!mlir.oraBlockIsNull(self.function.entry_block)) {
            const argument_count: usize = @intCast(mlir.oraBlockGetNumArguments(self.function.entry_block));
            for (0..@min(argument_count, self.function.param_names.len)) |index| {
                const value = mlir.oraBlockGetArgument(self.function.entry_block, index);
                if (!dynamicSequenceTypeHasLogicalLength(mlir.oraValueGetType(value))) continue;
                try variables.append(self.allocator, .{
                    .index = @intCast(self.function.param_names.len + index),
                    .id = dynamicLengthFreeVarId(index),
                    .name = try std.fmt.allocPrint(
                        self.allocator,
                        "{s}.len",
                        .{self.function.param_names[index]},
                    ),
                    .ty = self.builtinTypeRef(.u256),
                });
            }
        }
        return variables.toOwnedSlice(self.allocator);
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
        const previous = self.allow_short_circuit_predicates;
        self.allow_short_circuit_predicates = true;
        defer self.allow_short_circuit_predicates = previous;
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
        if (!mlir.oraBlockIsNull(self.function.entry_block) and
            (kind == .assert or kind == .cf_assert or kind == .refinement_guard or kind == .ensures))
        {
            const sites = &self.function.projection.safety_sites;
            sites.* = try std.mem.concat(self.allocator, LoopSafetySite, &.{ sites.*, &.{.{ .op_id = @intFromPtr(op.ptr) }} });
        }
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
        var owner = self.ownerFor(symbol);
        var logical_role = role;
        const parent = mlir.mlirOperationGetParentOperation(op);
        if (self.options.project_calls and std.mem.eql(u8, op_name, "ora.invariant") and
            !mlir.oraOperationIsNull(parent) and std.mem.eql(u8, operationName(parent), "ora.contract"))
        {
            const kind = (try self.stringAttr(op, "ora.source_fact_kind")) orelse return error.CallEntrySourceMissing;
            if (!std.mem.eql(u8, kind, "contract_invariant") or (try self.u32Attr(op, "ora.source_fact_id")) == null) return error.CallEntrySourceMissing;
            owner = .{ .contract = try self.operationStructuralKey(parent, "ora.contract") };
            logical_role = .contract_invariant;
        }
        const previous_entry_facts = self.project_entry_facts;
        self.project_entry_facts = self.options.project_calls and logical_role == .contract_invariant and owner == .contract;
        defer self.project_entry_facts = previous_entry_facts;
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
        const id = self.nextId();
        const source = try self.sourceForOperation(op);

        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = source,
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .logical = .{
                .role = logical_role,
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
        try self.addQueryForOperation(.obligation, logical_role, null, id, owner, origin, source, op);
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
        const previous_entry_facts = self.project_entry_facts;
        self.project_entry_facts = self.options.project_calls and kind == .requires;
        defer self.project_entry_facts = previous_entry_facts;
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
        if (self.active_loop_depth == 0 and std.mem.eql(u8, try self.stringAttr(op, "ora.verification_context") orelse "", "parameter_refinement")) {
            self.queries.items[self.queries.items.len - 1].artifact_policy = .runtime_checked_input;
        }
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
        const previous = self.allow_short_circuit_predicates;
        self.allow_short_circuit_predicates = true;
        defer self.allow_short_circuit_predicates = previous;
        if (mlir.oraOperationGetNumOperands(op) <= operand_index) return null;
        const value = mlir.oraOperationGetOperand(op, operand_index);
        if (try self.termFromValue(value)) |term| {
            var formula: obligation.FormulaRef = .{ .term = term };
            if (self.function.param_names.len != 0 and
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
            const name = if (arg_number < self.function.param_names.len)
                self.function.param_names[arg_number]
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
        if (try self.lifecycleAddressCastOperand(owner)) |operand|
            return self.termFromValueWithIntegerSignedness(operand, null);
        if (isTransparentValueOp(owner_name)) {
            if (mlir.oraOperationGetNumOperands(owner) == 0) return null;
            return try self.termFromValueWithIntegerSignedness(mlir.oraOperationGetOperand(owner, 0), signed_override);
        }
        if (std.mem.eql(u8, owner_name, "ora.old")) return try self.oldTerm(owner);
        if (std.mem.eql(u8, owner_name, "memref.dim") or
            std.mem.eql(u8, owner_name, "tensor.dim") or
            std.mem.eql(u8, owner_name, "ora.length"))
        {
            return try self.dynamicLengthTerm(owner);
        }
        if (std.mem.eql(u8, owner_name, "ora.sload")) return try self.scalarStorageLoadTerm(owner);
        if (std.mem.eql(u8, owner_name, "ora.map_get")) return try self.storageMapReadTerm(owner);
        if (std.mem.eql(u8, owner_name, "memref.load")) return try self.fixedArrayReadTerm(owner);
        if (std.mem.eql(u8, owner_name, "arith.constant")) return try self.constantTerm(owner, signed_override);
        if (self.allow_short_circuit_predicates and std.mem.eql(u8, owner_name, "scf.if")) return try self.shortCircuitPredicateTerm(owner);
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
        // LoopVC and scoped entry facts instead use their explicit storage state.
        if (!self.canProjectStablePlace(.{ .root = root, .region = .storage }) and
            !self.use_loop_value_bindings and !self.project_entry_facts) return null;
        return try self.placeReadTerm(root);
    }

    fn storageMapReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        const place = (try self.storageMapReadPlaceFromMapGet(op)) orelse return null;
        if (!self.canProjectStablePlace(place) and !self.canProjectLoopCurrentMappingPlace(place) and !self.project_entry_facts) return null;
        return try self.placeReadTermFromPlace(place);
    }

    fn fixedArrayReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if ((!self.use_loop_value_bindings and !self.project_entry_facts) or
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
        if (self.function.has_external_call) return false;
        if (!self.function.write_slots_complete) return false;
        const write_slots = self.function.write_slots orelse return false;
        return placeIsDefinitelyDisjointFromAll(place, write_slots);
    }

    fn oldTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) != 1) return null;
        if (!mlirTypeIsSupportedInteger(self.operationResultType(op, 0))) return null;
        if (self.project_entry_facts) return try self.addTerm(.{
            .old = (try self.termFromValue(mlir.oraOperationGetOperand(op, 0))) orelse return null,
        });
        if (self.function.has_external_call) return null;
        if (!self.function.write_slots_complete) return null;
        const write_slots = self.function.write_slots orelse return null;

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
        if (key != .parameter and (key != .constant or obligation.parseDecimalU256(key.constant) == null)) return null;
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

    fn shortCircuitPredicateTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!structuredGuardOperationSupported(op, "scf.if")) return null;
        const block = singleRegionBlock(op, 0) orelse return null;
        var child = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
            const name = operationName(child);
            if (std.mem.eql(u8, name, "scf.yield")) {
                if (mlir.oraOperationGetNumOperands(child) != 1 or !mlir.oraOperationIsNull(mlir.oraOperationGetNextInBlock(child))) return null;
                const lhs = (try self.termFromValue(mlir.oraOperationGetOperand(op, 0))) orelse return null;
                const rhs = (try self.termFromValue(mlir.oraOperationGetOperand(child, 0))) orelse return null;
                return try self.addTerm(.{ .binary = .{ .op = .and_, .lhs = lhs, .rhs = rhs } });
            }
            if (std.mem.eql(u8, name, "scf.if")) {
                if (try self.shortCircuitPredicateTerm(child) == null) return null;
            } else {
                if (mlir.oraOperationGetNumRegions(child) != 0 or mlir.mlirOperationGetNumSuccessors(child) != 0) return null;
                if (!preheaderPureValueOperationSupported(name) and !isLoopSafetyOperation(child, name) and
                    !std.mem.eql(u8, name, "ora.sload") and
                    !(std.mem.eql(u8, name, "memref.load") and inventoryMemrefLoadTargetsPersistentRoot(child))) return null;
            }
        }
        return null;
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
                if (!mlirIntegerCarrierHasWidth(mlir.oraValueGetType(value), info.width)) {
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
        if (mlir.oraBlockIsNull(self.function.entry_block)) return null;
        const owner_block = mlir.mlirBlockArgumentGetOwner(value);
        if (mlir.oraBlockIsNull(owner_block)) return null;
        if (owner_block.ptr != self.function.entry_block.ptr) return null;
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
                if (arg_number < self.function.param_types.len) {
                    return self.function.param_types[arg_number];
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
        var index = self.function.param_names.len;
        while (index != 0) {
            index -= 1;
            const param_id = self.freeVarIdForFunctionParam(index);
            const param_ty = if (index < self.function.param_types.len)
                self.function.param_types[index]
            else
                try self.unknownTypeRef();
            current = try self.bindFreeVariableInTerm(current, param_id, self.function.param_names[index], param_ty);
            if (condition_available) |condition_term| {
                condition_available = try self.bindFreeVariableInTerm(condition_term, param_id, self.function.param_names[index], param_ty);
            }
            current = try self.addTerm(.{ .quantified = .{
                .quantifier = .forall,
                .binder = .{
                    .name = self.function.param_names[index],
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
        if (index < self.function.param_binding_ids.len) return self.function.param_binding_ids[index];
        return .{
            .file_id = synthetic_file_id,
            .pattern_id = @intCast(index),
        };
    }

    fn dynamicLengthTerm(self: *Collector, op: mlir.MlirOperation) !?obligation.TermId {
        const source = dynamicLengthSourceValue(op) orelse return null;
        const parameter_index = self.functionEntryBlockArgumentNumber(source) orelse return null;
        if (!dynamicSequenceTypeHasLogicalLength(mlir.oraValueGetType(source))) return null;
        const parameter_name = if (parameter_index < self.function.param_names.len)
            self.function.param_names[parameter_index]
        else
            "dynamic-value";
        return try self.addTerm(.{ .variable = .{ .free = .{
            .id = dynamicLengthFreeVarId(parameter_index),
            .name = try std.fmt.allocPrint(self.allocator, "{s}.len", .{parameter_name}),
            .ty = self.builtinTypeRef(.u256),
        } } });
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
        if (term == .bool_lit) if (self.boolean_terms[@intFromBool(term.bool_lit)]) |id| return id;
        const id: obligation.TermId = @intCast(self.terms.items.len);
        try self.terms.append(self.allocator, term);
        if (term == .bool_lit) self.boolean_terms[@intFromBool(term.bool_lit)] = id;
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
        const module_path = self.function.source_module_path orelse
            if (is_contract_invariant) source.file else null;
        const owner_key = self.function.source_owner_key orelse
            if (is_contract_invariant) self.contract_source_owner_key else null;
        const activation = self.function.source_activation orelse
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
            .runtime_symbol = self.function.runtime_symbol,
            .specialization_bindings = self.function.source_specialization_bindings,
            .trait_implementation = self.function.source_trait_implementation,
            .trait_method = self.function.source_trait_method,
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
                .module_path = self.function.source_module_path,
                .owner_key = self.function.source_owner_key,
                .template_activation = self.function.source_activation,
                .runtime_symbol = self.function.runtime_symbol,
                .specialization_bindings = self.function.source_specialization_bindings,
                .trait_implementation = self.function.source_trait_implementation,
                .trait_method = self.function.source_trait_method,
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
        if (index < self.function.param_binding_ids.len) return self.function.param_binding_ids[index];
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

fn mlirIntegerCarrierHasWidth(ty: mlir.MlirType, expected_width: u16) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    // Refinements retain their base carrier; metadata must still match its exact width.
    const base = mlir.oraRefinementTypeGetBaseType(ty);
    if (!mlir.oraTypeIsNull(base)) return mlirIntegerCarrierHasWidth(base, expected_width);
    const width = integerTypeWidth(ty) orelse return false;
    return width == expected_width;
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
    has_indexed_region_write: bool = false,
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
        if (dynamicAccessSite(child) != null) {
            if (mlir.oraOperationGetMemoryEffectKind(child) == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE)
                facts.has_indexed_region_write = true;
        }
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

fn appendDynamicAccessInventory(
    allocator: std.mem.Allocator,
    existing: []const LoopDynamicAccessInventory,
    access: LoopDynamicAccessInventory,
) ![]const LoopDynamicAccessInventory {
    const result = try allocator.alloc(LoopDynamicAccessInventory, existing.len + 1);
    @memcpy(result[0..existing.len], existing);
    result[result.len - 1] = access;
    return result;
}

fn indexedRegionOwnerFailure(base: mlir.MlirValue) anyerror {
    if (mlir.oraValueIsAOpResult(base)) {
        const name = operationName(mlir.oraOpResultGetOwner(base));
        if (std.mem.eql(u8, name, "scf.if") or std.mem.eql(u8, name, "arith.select"))
            return error.LoopIndexedRegionAliasUnsupported;
    }
    return error.LoopIndexedRegionOwnerUnsupported;
}

fn latestMatchingDynamicAccess(
    accesses: []const LoopDynamicAccessInventory,
    owner_identity: []const u8,
    index_identity: []const u8,
) ?LoopDynamicAccessInventory {
    var index = accesses.len;
    while (index != 0) {
        index -= 1;
        const access = accesses[index];
        if (std.mem.eql(u8, access.owner_identity, owner_identity) and
            std.mem.eql(u8, access.index_identity, index_identity)) return access;
    }
    return null;
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
        !dynamicAccessInventoriesEql(projected.dynamic_accesses, inventory.dynamic_accesses) or
        projected.event_inventory.len != inventory.event_inventory.len or
        !timelineMatchesInventory(projected.timeline, inventory.event_inventory)) return false;
    return timelineInventoriesEql(projected.event_inventory, inventory.event_inventory);
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
        if ((semantic.return_inventory == null) != (structural.return_inventory == null)) return .target_mismatch;
        if (semantic.return_inventory) |value| if (!returnInventoriesEql(value, structural.return_inventory.?)) return .target_mismatch;
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
        if (!effectTimelineInventoryEql(semantic.timeline_inventory, structural.timeline_inventory) or
            !dynamicAccessInventoriesEql(semantic.dynamic_accesses, structural.dynamic_accesses) or
            !timelineMatchesInventory(semantic.edge.timeline, structural.timeline_inventory))
        {
            return .effect_mismatch;
        }
    }
    return .complete;
}

fn containingFunction(op: mlir.MlirOperation) mlir.MlirOperation {
    var parent = op;
    while (!mlir.oraOperationIsNull(parent)) : (parent = mlir.mlirOperationGetParentOperation(parent))
        if (std.mem.eql(u8, operationName(parent), "func.func")) return parent;
    return parent;
}

fn addressCarrier(ty: mlir.MlirType) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    if (mlir.oraTypeIsAddressType(ty)) return true;
    const base = mlir.oraRefinementTypeGetBaseType(ty);
    return !mlir.oraTypeIsNull(base) and mlir.oraTypeIsAddressType(base);
}

fn addressTypeRef(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .compiler_type_id => |id| id == type_builtin.lookupBuiltinById(.address).comptime_type_id,
        .spelling => |name| std.mem.eql(u8, name, "address") or std.mem.eql(u8, name, "!ora.address"),
    };
}

fn callCarrierMatches(carrier: mlir.MlirType, ty: obligation.TypeRef) bool {
    if (ty == .spelling and std.mem.eql(u8, ty.spelling, "bool")) return integerTypeWidth(carrier) == 1;
    const info = typeRefIntegerInfo(ty) orelse return false;
    return mlirIntegerCarrierHasWidth(carrier, info.width);
}

fn callOutcomeKey(outcome: obligation.CallOutcome) u64 {
    return switch (outcome) {
        .success => 0,
        .error_return => |id| @as(u64, id) + 1,
    };
}

fn returnInventoriesEql(left: ReturnInventory, right: ReturnInventory) bool {
    if (!std.meta.eql(left.outcome, right.outcome) or !stringSlicesEql(left.sources, right.sources) or left.types.len != right.types.len) return false;
    for (left.types, right.types) |a, b| if (!obligation.typeRefEqual(a, b)) return false;
    return true;
}

fn callInventoriesEql(left: ?CallInventory, right: ?CallInventory) bool {
    if (left == null or right == null) return left == null and right == null;
    const a = left.?;
    const b = right.?;
    if (a.callee != b.callee or !returnInventoriesEql(.{ .outcome = a.outcome, .sources = a.arguments, .types = a.types }, .{ .outcome = b.outcome, .sources = b.arguments, .types = b.types }) or a.results.len != b.results.len) return false;
    for (a.results, b.results) |x, y| if (x.index != y.index or !optionalFreeVarEql(x.id, y.id) or !obligation.typeRefEqual(x.ty, y.ty)) return false;
    return true;
}

fn timelineInventoriesEql(
    lhs: []const LoopTimelineInventoryEvent,
    rhs: []const LoopTimelineInventoryEvent,
) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!timelineInventoryEventEql(left, right)) return false;
    }
    return true;
}

fn timelineInventoryEventEql(left: LoopTimelineInventoryEvent, right: LoopTimelineInventoryEvent) bool {
    return std.mem.eql(u8, left.structural_key, right.structural_key) and left.tag == right.tag and
        optionalStringEql(left.formula_source, right.formula_source) and
        optionalStringEql(left.storage_root, right.storage_root) and
        storageProjectionInventoriesEql(left.storage_projections, right.storage_projections) and
        optionalFreeVarEql(left.storage_result, right.storage_result) and
        optionalStringEql(left.storage_value_source, right.storage_value_source) and
        optionalTypeRefEql(left.storage_ty, right.storage_ty) and
        optionalFreeVarEql(left.indexed_result, right.indexed_result) and
        left.indexed_region == right.indexed_region and
        optionalFreeVarEql(left.indexed_owner, right.indexed_owner) and
        optionalStringEql(left.indexed_index_source, right.indexed_index_source) and
        optionalTypeRefEql(left.indexed_element_ty, right.indexed_element_ty) and
        optionalStringEql(left.indexed_checkpoint_key, right.indexed_checkpoint_key) and
        optionalDependencyEql(left.summary_dependency, right.summary_dependency) and
        callInventoriesEql(left.invocation, right.invocation);
}

fn effectTimelineInventoryEql(
    semantic: []const LoopTimelineInventoryEvent,
    structural: []const LoopTimelineInventoryEvent,
) bool {
    var semantic_index: usize = 0;
    for (structural) |right| {
        if (right.tag != .storage_read and right.tag != .storage_write and
            right.tag != .indexed_region_read and right.tag != .call) continue;
        if (semantic_index >= semantic.len) return false;
        const left = semantic[semantic_index];
        if (!timelineInventoryEventEql(left, right)) return false;
        semantic_index += 1;
    }
    return semantic_index == semantic.len;
}

fn timelineInventoryHasStorageEvent(inventory: []const LoopTimelineInventoryEvent) bool {
    for (inventory) |item| switch (item.tag) {
        .storage_read, .storage_write, .call => return true,
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
                !optionalTypeRefEql(read.location.value_ty, item.storage_ty)) return false,
            .storage_write => |write| if (!optionalStringEql(write.location.root, item.storage_root) or
                !storageProjectionsMatchInventory(write.location.projections, item.storage_projections) or
                !optionalTypeRefEql(write.location.value_ty, item.storage_ty) or
                item.storage_value_source == null) return false,
            .indexed_region_read => |read| {
                if (!optionalFreeVarEql(read.result, item.indexed_result) or
                    item.indexed_region != read.region or
                    !optionalFreeVarEql(read.owner, item.indexed_owner) or
                    read.index.formula == .origin_value or
                    item.indexed_index_source == null or item.indexed_index_source.?.len == 0 or
                    !optionalTypeRefEql(read.element_ty, item.indexed_element_ty) or
                    !optionalStringEql(read.checkpoint_structural_key, item.indexed_checkpoint_key)) return false;
            },
            .call => |invocation| {
                const found = item.invocation orelse return false;
                if (invocation.callee_summary_id != found.callee or !std.meta.eql(invocation.outcome, found.outcome) or
                    invocation.arguments.len != found.arguments.len or found.types.len != found.arguments.len or invocation.results.len != found.results.len) return false;
                for (invocation.arguments, found.types) |argument, ty| if (argument.value.formula == .origin_value or !obligation.typeRefEqual(argument.ty, ty)) return false;
                for (invocation.results, found.results) |result, expected| if (result.index != expected.index or !optionalFreeVarEql(result.id, expected.id) or !obligation.typeRefEqual(result.ty, expected.ty)) return false;
            },
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
            !optionalTypeRefEql(left.ty, right.ty) or
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
            !optionalTypeRefEql(fixed.index.ty, item.ty) or
            item.extent != fixed.extent) return false,
        .mapping_key => |key| if (item.tag != .mapping_key or
            item.source_key == null or !optionalTypeRefEql(key.ty, item.ty) or
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

fn optionalTypeRefEql(lhs: ?obligation.TypeRef, rhs: ?obligation.TypeRef) bool {
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

fn dynamicAccessInventoriesEql(
    lhs: []const LoopDynamicAccessInventory,
    rhs: []const LoopDynamicAccessInventory,
) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left.structural_key, right.structural_key) or
            !std.mem.eql(u8, left.owner_identity, right.owner_identity) or
            !std.mem.eql(u8, left.index_identity, right.index_identity)) return false;
    }
    return true;
}

fn loopProjectionHasMultipleIndexedRegionOwners(edges: []const ProjectedLoopEdge) bool {
    var first: ?obligation.FreeVarId = null;
    for (edges) |edge| for (edge.edge.timeline) |event| switch (event.payload) {
        .indexed_region_read => |read| {
            if (first) |owner| {
                if (!obligation.freeVarIdEql(owner, read.owner)) return true;
            } else first = read.owner;
        },
        else => {},
    };
    return false;
}

fn loopProjectionHasIndexedRegionEvent(
    projected: []const ProjectedLoopEdge,
    inventory: []const InventoryLoopEdge,
) bool {
    for (projected) |edge| for (edge.edge.timeline) |event| {
        if (event.payload == .indexed_region_read) return true;
    };
    for (inventory) |edge| for (edge.timeline_inventory) |event| {
        if (event.tag == .indexed_region_read) return true;
    };
    return false;
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
    if (left.result_bindings == null or right.result_bindings == null)
        return left.result_bindings == null and right.result_bindings == null;
    if (left.result_bindings.?.len != right.result_bindings.?.len) return false;
    for (left.result_bindings.?, right.result_bindings.?) |a, b| {
        if (a.index != b.index or !std.meta.eql(a.id, b.id) or
            !optionalStringEqual(a.name, b.name) or !obligation.typeRefEqual(a.ty, b.ty)) return false;
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
        .storage_read, .storage_write, .call => return true,
        .path_assumption, .safety_checkpoint, .indexed_region_read, .summary_dependency => {},
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
        .indexed_region_read => |read| if (read.index.formula == .origin_value) return true,
        .storage_write => |write| if (write.value.formula == .origin_value) return true,
        .call => |invocation| for (invocation.arguments) |argument| {
            if (argument.value.formula == .origin_value) return true;
        },
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

// Entry values are limited to actual false initializers of deferred controls.
// Each scan reads the prefix itself; neither supplies the other's slot values.
fn projectPrivateEntryValue(load: mlir.MlirOperation, draft: LoopSummaryDraft, other_stores: bool) !mlir.MlirValue {
    const address = mlir.oraOperationGetOperand(load, 0);
    if (draft.loop_kind == .other) return privateInitializerFailure(load, load, other_stores);
    const block = mlir.mlirOperationGetBlock(draft.loop_op);
    if (mlir.mlirOperationGetBlock(mlir.oraOpResultGetOwner(address)).ptr != block.ptr) return privateInitializerFailure(load, load, other_stores);
    var initial: ?mlir.MlirValue = null;
    var other = other_stores;
    var op = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(op) and op.ptr != draft.loop_op.ptr) : (op = mlir.oraOperationGetNextInBlock(op)) {
        if (mlir.oraOperationGetNumRegions(op) != 0) return error.CallPrivateSlotConditionalInitializer;
        if (!std.mem.eql(u8, operationName(op), "memref.store") or mlir.oraOperationGetNumOperands(op) != 2) continue;
        if (mlir.oraOperationGetOperand(op, 1).ptr == address.ptr) initial = mlir.oraOperationGetOperand(op, 0) else other = true;
    }
    const value = initial orelse return privateInitializerFailure(load, draft.loop_op, other);
    if (inventoryBooleanLiteral(value, &.{}) != false or !privateControlLoad(load)) return error.CallPrivateSlotEntryUnsupported;
    return value;
}

fn inventoryPrivateEntryValue(load: mlir.MlirOperation, draft: LoopSummaryDraft, other_stores: bool) !mlir.MlirValue {
    const slot = mlir.oraOperationGetOperand(load, 0);
    if (draft.loop_kind == .other) return privateInitializerFailure(load, load, other_stores);
    const entry = mlir.mlirOperationGetBlock(draft.loop_op);
    if (mlir.mlirOperationGetBlock(mlir.oraOpResultGetOwner(slot)).ptr != entry.ptr) return privateInitializerFailure(load, load, other_stores);
    var found: ?mlir.MlirValue = null;
    var different_address = other_stores;
    var preceding = mlir.oraBlockGetFirstOperation(entry);
    while (!mlir.oraOperationIsNull(preceding) and preceding.ptr != draft.loop_op.ptr) : (preceding = mlir.oraOperationGetNextInBlock(preceding)) {
        if (mlir.oraOperationGetNumRegions(preceding) != 0) return error.CallPrivateSlotConditionalInitializer;
        if (std.mem.eql(u8, operationName(preceding), "memref.store") and mlir.oraOperationGetNumOperands(preceding) == 2) {
            if (mlir.oraOperationGetOperand(preceding, 1).ptr == slot.ptr) found = mlir.oraOperationGetOperand(preceding, 0) else different_address = true;
        }
    }
    const value = found orelse return privateInitializerFailure(load, draft.loop_op, different_address);
    if (!privateControlLoad(load) or inventoryBooleanLiteral(value, &.{}) != false) return error.CallPrivateSlotEntryUnsupported;
    return value;
}

// Diagnostic classification only: this walk can reject, never admit a load.
fn privateInitializerFailure(load: mlir.MlirOperation, anchor: mlir.MlirOperation, other_stores: bool) anyerror {
    const block = mlir.mlirOperationGetBlock(anchor);
    const position = headerOperationPosition(block, anchor).?;
    var use = mlir.mlirValueGetFirstUse(mlir.oraOperationGetOperand(load, 0));
    while (!mlir.mlirOpOperandIsNull(use)) : (use = mlir.mlirOpOperandGetNextUse(use)) {
        const store = mlir.mlirOpOperandGetOwner(use);
        if (!std.mem.eql(u8, operationName(store), "memref.store")) continue;
        var parent = store;
        while (!mlir.oraOperationIsNull(parent) and mlir.mlirOperationGetBlock(parent).ptr != block.ptr) parent = mlir.mlirOperationGetParentOperation(parent);
        if (mlir.oraOperationIsNull(parent)) continue;
        const at = headerOperationPosition(block, parent).?;
        if (at < position and parent.ptr != store.ptr) return error.CallPrivateSlotConditionalInitializer;
        if (at > position) return error.CallPrivateSlotInitializerAfterLoad;
    }
    return if (other_stores) error.CallPrivateSlotWrongInitializerSlot else error.CallPrivateSlotUninitialized;
}

fn privateControlLoad(load: mlir.MlirOperation) bool {
    if (!operationResultIsI1(load)) return false;
    var use = mlir.mlirValueGetFirstUse(mlir.oraOperationGetResult(load, 0));
    if (mlir.mlirOpOperandIsNull(use)) return false;
    while (!mlir.mlirOpOperandIsNull(use)) : (use = mlir.mlirOpOperandGetNextUse(use)) {
        const comparison = mlir.mlirOpOperandGetOwner(use);
        if (!std.mem.eql(u8, operationName(comparison), "arith.cmpi") or mlir.oraOperationGetNumOperands(comparison) != 2 or !operationResultIsI1(comparison)) return false;
        const predicate = mlir.oraOperationGetAttributeByName(comparison, strRef("predicate"));
        if (mlir.oraAttributeIsNull(predicate) or mlir.oraIntegerAttrGetValueSInt(predicate) != 0 or inventoryBooleanLiteral(mlir.oraOperationGetOperand(comparison, 1), &.{}) != false) return false;
        var control_use = mlir.mlirValueGetFirstUse(mlir.oraOperationGetResult(comparison, 0));
        if (mlir.mlirOpOperandIsNull(control_use)) return false;
        while (!mlir.mlirOpOperandIsNull(control_use)) : (control_use = mlir.mlirOpOperandGetNextUse(control_use)) {
            const control = mlir.mlirOpOperandGetOwner(control_use);
            if ((isLoopControlRuntimeGuard(control) catch false) and std.mem.eql(u8, operationName(control), "scf.if") and mlir.oraOperationGetNumOperands(control) == 1) continue;
            const source_guard = mlir.oraOperationGetAttributeByName(control, strRef("ora.loop_source_guard_operand"));
            if (std.mem.eql(u8, operationName(control), "arith.andi") and !mlir.oraAttributeIsNull(source_guard) and mlir.oraIntegerAttrGetValueSInt(source_guard) == 1 and
                mlir.oraOperationGetNumOperands(control) == 2 and mlir.oraOperationGetOperand(control, 0).ptr == mlir.oraOperationGetResult(comparison, 0).ptr) continue;
            return false;
        }
    }
    return true;
}

fn privateReturnOperands(allocator: std.mem.Allocator, slots: []const PrivateSlotValue) ![]const mlir.MlirValue {
    var found: ?PrivateSlotValue = null;
    var index = slots.len;
    while (index > 0) {
        index -= 1;
        if (found) |previous| if (previous.address == slots[index].address) continue;
        const address = mlir.MlirValue{ .ptr = @ptrFromInt(slots[index].address) };
        var use = mlir.mlirValueGetFirstUse(address);
        while (!mlir.mlirOpOperandIsNull(use)) : (use = mlir.mlirOpOperandGetNextUse(use)) {
            const load = mlir.mlirOpOperandGetOwner(use);
            if (!std.mem.eql(u8, operationName(load), "memref.load") or mlir.oraOperationGetNumResults(load) != 1) continue;
            var loaded_use = mlir.mlirValueGetFirstUse(mlir.oraOperationGetResult(load, 0));
            while (!mlir.mlirOpOperandIsNull(loaded_use)) : (loaded_use = mlir.mlirOpOperandGetNextUse(loaded_use)) {
                if (std.mem.eql(u8, operationName(mlir.mlirOpOperandGetOwner(loaded_use)), "ora.return")) {
                    if (found) |previous| if (previous.address != slots[index].address) return error.CallPrivateSlotAmbiguous;
                    found = slots[index];
                }
            }
        }
    }
    return if (found) |slot| allocator.dupe(mlir.MlirValue, &.{slot.value}) else error.CallPrivateSlotUninitialized;
}

fn headerPrivateSlot(value: mlir.MlirValue) bool {
    if (!mlir.oraValueIsAOpResult(value)) return false;
    const allocation = mlir.oraOpResultGetOwner(value);
    if (!std.mem.eql(u8, operationName(allocation), "memref.alloca") or
        mlir.oraOperationGetNumOperands(allocation) != 0) return false;
    const ty = mlir.oraValueGetType(value);
    if (!mlir.oraTypeIsAMemRef(ty) or mlir.oraShapedTypeGetRank(ty) != 0) return false;
    // Fresh rank-zero storage cannot alias immutable header inputs. Requiring
    // every use to be a direct load/store also forbids passing its address on.
    var use = mlir.mlirValueGetFirstUse(value);
    while (!mlir.mlirOpOperandIsNull(use)) : (use = mlir.mlirOpOperandGetNextUse(use)) {
        const user = mlir.mlirOpOperandGetOwner(use);
        const name = operationName(user);
        const count = mlir.oraOperationGetNumOperands(user);
        if (std.mem.eql(u8, name, "memref.load") and count == 1 and
            mlir.oraOperationGetOperand(user, 0).ptr == value.ptr) continue;
        if (std.mem.eql(u8, name, "memref.store") and count == 2 and
            mlir.oraOperationGetOperand(user, 1).ptr == value.ptr) continue;
        return false;
    }
    return true;
}

fn structuredGuardOperationSupported(op: mlir.MlirOperation, name: []const u8) bool {
    if (std.mem.eql(u8, name, "scf.if")) {
        if (mlir.oraOperationGetNumOperands(op) != 1 or mlir.oraOperationGetNumResults(op) != 1 or
            mlir.oraOperationGetNumRegions(op) != 2 or !operationResultIsI1(op)) return false;
        const block = singleRegionBlock(op, 1) orelse return false;
        const literal = mlir.oraBlockGetFirstOperation(block);
        if (mlir.oraOperationIsNull(literal) or !std.mem.eql(u8, operationName(literal), "arith.constant") or !operationResultIsI1(literal)) return false;
        const text = mlir.oraIntegerAttrGetValueString(mlir.oraOperationGetAttributeByName(literal, strRef("value")));
        defer if (text.data != null) mlir.oraStringRefFree(text);
        if (text.data == null or !std.mem.eql(u8, text.data[0..text.length], "0")) return false;
        const yield = mlir.oraOperationGetNextInBlock(literal);
        return !mlir.oraOperationIsNull(yield) and std.mem.eql(u8, operationName(yield), "scf.yield") and
            mlir.oraOperationGetNumOperands(yield) == 1 and mlir.oraOperationIsNull(mlir.oraOperationGetNextInBlock(yield)) and
            mlir.oraOperationGetOperand(yield, 0).ptr == mlir.oraOperationGetResult(literal, 0).ptr;
    }
    return preheaderPureValueOperationSupported(name) or isLoopSafetyOperation(op, name) or
        isRegionYield(name) or std.mem.eql(u8, name, "scf.condition") or dynamicLengthSourceValue(op) != null or
        (std.mem.eql(u8, name, "memref.load") and mlir.oraOperationGetNumOperands(op) == 1 and
            headerPrivateSlot(mlir.oraOperationGetOperand(op, 0)));
}

// The independent inventory follows actual yielded values and MLIR literals;
// it does not consult the semantic walk's term table or branch decisions.
fn inventoryBooleanLiteral(value: mlir.MlirValue, sources: []const LoopValueSource) ?bool {
    var index = sources.len;
    while (index != 0) {
        index -= 1;
        if (sources[index].value_id == @intFromPtr(value.ptr)) return sources[index].boolean_literal;
    }
    if (!mlir.oraValueIsAOpResult(value)) return null;
    const owner = mlir.oraOpResultGetOwner(value);
    if (!std.mem.eql(u8, operationName(owner), "arith.constant") or !operationResultIsI1(owner) or
        hasAttr(owner, "ora.bound_variable")) return null;
    const text = mlir.oraIntegerAttrGetValueString(mlir.oraOperationGetAttributeByName(owner, strRef("value")));
    defer if (text.data != null) mlir.oraStringRefFree(text);
    if (text.data == null) return null;
    const literal = std.mem.trim(u8, text.data[0..text.length], " \t\n\r");
    if (std.mem.eql(u8, literal, "0")) return false;
    if (std.mem.eql(u8, literal, "1")) return true;
    return null;
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

fn headerFactProjectionFailure(err: anyerror) ?obligation.LoopUnsupportedReason {
    return switch (err) {
        error.LoopHeaderFactInventoryMismatch => .loop_header_fact_inventory_mismatch,
        error.LoopHeaderFactPositionUnsupported => .loop_header_fact_position_unsupported,
        error.LoopHeaderFactProvenanceUnsupported => .loop_header_fact_provenance_unsupported,
        error.LoopHeaderFactTypeUnsupported => .loop_header_fact_type_unsupported,
        error.LoopHeaderFactEffectUnsupported => .loop_header_fact_effect_unsupported,
        error.LoopHeaderFactAssumptionMismatch => .loop_header_fact_assumption_mismatch,
        else => null,
    };
}

fn indexedRegionProjectionFailure(err: anyerror) ?obligation.LoopUnsupportedReason {
    return switch (err) {
        error.LoopIndexedRegionAliasUnsupported => .loop_indexed_region_alias_unsupported,
        error.LoopIndexedRegionOwnerUnsupported => .loop_indexed_region_owner_unsupported,
        error.LoopIndexedRegionWriteUnreviewed => .loop_indexed_region_write_unreviewed,
        error.LoopIndexedRegionCheckpointMissing => .loop_indexed_region_checkpoint_missing,
        error.LoopIndexedRegionTypeUnsupported => .loop_value_type_unresolved,
        else => null,
    };
}

fn inventoryHeaderFactSource(op: mlir.MlirOperation, name: []const u8) !obligation.LoopHeaderFactSource {
    if (std.mem.eql(u8, name, "ora.requires")) {
        var kind = stringAttrView(op, "ora.source_fact_kind") orelse return error.LoopHeaderFactProvenanceUnsupported;
        if (!std.mem.eql(u8, kind.slice(), "requires") or !hasAttr(op, "ora.source_fact_id"))
            return error.LoopHeaderFactProvenanceUnsupported;
        return .function_requires;
    }
    var origin = stringAttrView(op, "ora.assume_origin") orelse return error.LoopHeaderFactProvenanceUnsupported;
    return if (std.mem.eql(u8, origin.slice(), "path")) .path_assume else error.LoopHeaderFactProvenanceUnsupported;
}

fn headerFactTypeSupported(ty: obligation.TypeRef) bool {
    if (typeRefIntegerInfo(ty) != null) return true;
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "bool") or std.mem.eql(u8, name, "i1"),
        .compiler_type_id => |id| id == type_builtin.lookupBuiltinById(.bool).comptime_type_id,
    };
}

fn headerOperationPosition(block: mlir.MlirBlock, target: mlir.MlirOperation) ?u32 {
    var position: u32 = 0;
    var op = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(op)) : ({
        op = mlir.oraOperationGetNextInBlock(op);
        position += 1;
    }) if (op.ptr == target.ptr) return position;
    return null;
}

fn appendHeaderFactValue(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(obligation.LoopHeaderFactValue),
    value: obligation.LoopHeaderFactValue,
) !void {
    for (values.items) |existing| if (obligation.freeVarIdEql(existing.id, value.id)) return;
    try values.append(allocator, value);
}

fn headerFactValuesContain(values: []const obligation.LoopHeaderFactValue, id: obligation.FreeVarId) bool {
    for (values) |value| if (obligation.freeVarIdEql(value.id, id)) return true;
    return false;
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

fn dynamicLengthFreeVarId(parameter_index: usize) obligation.FreeVarId {
    return .{
        .file_id = Collector.dynamic_length_file_id,
        .pattern_id = @intCast(parameter_index),
    };
}

fn dynamicAccessSite(op: mlir.MlirOperation) ?DynamicAccessSite {
    const name = operationName(op);
    var site: DynamicAccessSite = undefined;
    if (std.mem.eql(u8, name, "memref.load") and mlir.oraOperationGetNumOperands(op) == 2) {
        site = .{
            .base = mlir.oraOperationGetOperand(op, 0),
            .index = fixedIndexSourceValue(mlir.oraOperationGetOperand(op, 1)),
        };
    } else if (std.mem.eql(u8, name, "memref.store") and mlir.oraOperationGetNumOperands(op) == 3) {
        site = .{
            .base = mlir.oraOperationGetOperand(op, 1),
            .index = fixedIndexSourceValue(mlir.oraOperationGetOperand(op, 2)),
        };
    } else if (std.mem.eql(u8, name, "ora.byte_at") and mlir.oraOperationGetNumOperands(op) == 2) {
        const base = mlir.oraOperationGetOperand(op, 0);
        site = .{
            .base = base,
            .index = mlir.oraOperationGetOperand(op, 1),
        };
    } else return null;
    return if (dynamicSequenceTypeHasLogicalLength(mlir.oraValueGetType(site.base))) site else null;
}

fn dynamicAccessCheckFromAssert(op: mlir.MlirOperation) ?DynamicAccessCheck {
    var message = stringAttrView(op, "message") orelse return null;
    if (assert_safety_map.get(message.slice()) != .dynamic_index_bounds or
        mlir.oraOperationGetNumOperands(op) != 1) return null;
    const predicate = mlir.oraOperationGetOperand(op, 0);
    if (!mlir.oraValueIsAOpResult(predicate)) return null;
    const compare = mlir.oraOpResultGetOwner(predicate);
    if (mlir.oraOperationIsNull(compare) or mlir.oraOperationGetNumOperands(compare) != 2) return null;
    const length = mlir.oraOperationGetOperand(compare, 1);
    return .{
        .index = fixedIndexSourceValue(mlir.oraOperationGetOperand(compare, 0)),
        .owner = dynamicLengthOwnerFromValue(length) orelse return null,
    };
}

fn dynamicSequenceTypeHasLogicalLength(ty: mlir.MlirType) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    if (mlir.oraTypeIsAMemRef(ty)) {
        return mlir.oraShapedTypeGetRank(ty) == 1 and !mlir.oraShapedTypeHasStaticShape(ty);
    }
    const context = mlir.mlirTypeGetContext(ty);
    return mlir.oraTypeEqual(ty, mlir.oraBytesTypeGet(context)) or
        mlir.oraTypeEqual(ty, mlir.oraStringTypeGet(context));
}

fn dynamicLengthSourceValue(op: mlir.MlirOperation) ?mlir.MlirValue {
    const name = operationName(op);
    if (std.mem.eql(u8, name, "ora.length")) {
        if (mlir.oraOperationGetNumOperands(op) != 1) return null;
        return mlir.oraOperationGetOperand(op, 0);
    }
    if (!std.mem.eql(u8, name, "memref.dim") and !std.mem.eql(u8, name, "tensor.dim")) return null;
    if (mlir.oraOperationGetNumOperands(op) != 2) return null;
    const dimension = mlir.oraOperationGetOperand(op, 1);
    if (!mlir.oraValueIsAOpResult(dimension)) return null;
    const constant = mlir.oraOpResultGetOwner(dimension);
    if (mlir.oraOperationIsNull(constant) or !std.mem.eql(u8, operationName(constant), "arith.constant")) return null;
    const value_attr = mlir.oraOperationGetAttributeByName(constant, strRef("value"));
    if (mlir.oraAttributeIsNull(value_attr)) return null;
    const text = mlir.oraIntegerAttrGetValueString(value_attr);
    defer if (text.data != null) mlir.oraStringRefFree(text);
    if (text.data == null or !std.mem.eql(u8, std.mem.trim(u8, text.data[0..text.length], " \t\n\r"), "0")) return null;
    return mlir.oraOperationGetOperand(op, 0);
}

fn dynamicLengthOwnerFromValue(value: mlir.MlirValue) ?mlir.MlirValue {
    var current = value;
    var remaining: u32 = 32;
    while (remaining != 0 and mlir.oraValueIsAOpResult(current)) : (remaining -= 1) {
        const owner = mlir.oraOpResultGetOwner(current);
        if (dynamicLengthSourceValue(owner)) |source| return source;
        if (!isTransparentValueOp(operationName(owner)) or mlir.oraOperationGetNumOperands(owner) != 1) {
            return null;
        }
        current = mlir.oraOperationGetOperand(owner, 0);
    }
    return null;
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
    const shift: u5 = @intCast(@backingInt(property));
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

test "loop blocker batch Boolean terms remain valid after preheader rollback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var collector: Collector = .{ .allocator = arena.allocator(), .options = .{} };
    const retained = try collector.addTerm(.{ .bool_lit = true });
    const temporary = try collector.addTerm(.{ .bool_lit = false });
    collector.discardTermsFrom(temporary);
    const replacement = try collector.addTerm(.{ .int_lit = .{ .value = "7" } });
    try std.testing.expectEqual(temporary, replacement);
    const false_formula = try collector.negateFormula(.{ .term = retained });
    try std.testing.expect(false_formula.term != replacement);
    try std.testing.expectEqual(false, collector.terms.items[false_formula.term].bool_lit);
    const true_formula = try collector.negateFormula(false_formula);
    try std.testing.expectEqual(retained, true_formula.term);
    collector.discardTermsFrom(0);
    const false_again = try collector.addTerm(.{ .bool_lit = false });
    try std.testing.expectEqual(false, collector.terms.items[false_again].bool_lit);
}

test "certified call signatures validate carriers metadata and scoped error payloads" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
    const signed = type_builtin.lookupBuiltinById(.i8).comptime_type_id;
    const unsigned = type_builtin.lookupBuiltinById(.u8).comptime_type_id;
    const word = type_builtin.lookupBuiltinById(.u256).comptime_type_id;
    const cases = .{
        .{ "i8", signed, signed, true },
        .{ "i8", unsigned, unsigned, true },
        .{ "i256", word, word, true },
        .{ "i1", 0, 0, true },
        .{ "i8", 0, signed, false },
        .{ "i8", word, signed, false },
        .{ "i8", signed, word, false },
        .{ "i8", signed, 9999999, false },
    };
    inline for (cases) |case| {
        const source = try std.fmt.allocPrint(arena.allocator(),
            \\module {{
            \\  func.func @identity(%arg0: {s}) -> {s} attributes {{
            \\    ora.param_binding_ids = ["file:0:pattern:1"],
            \\    ora.param_compiler_type_ids = [{d} : i256],
            \\    ora.return_compiler_type_id = {d} : i256
            \\  }} {{ func.return %arg0 : {s} }}
            \\}}
        , .{ case[0], case[0], case[1], case[2], case[0] });
        const module = mlir.oraModuleCreateParse(context, strRef(source));
        try std.testing.expect(!mlir.oraModuleIsNull(module));
        defer mlir.oraModuleDestroy(module);
        const function = mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module));
        if (case[3]) {
            const signature = try collector.callSignature(function);
            try std.testing.expectEqual(@as(usize, 1), signature.parameters.len);
            try std.testing.expectEqual(@as(usize, 1), signature.outcomes.len);
            const expected: obligation.TypeRef = if (case[1] == 0) .{ .spelling = "bool" } else .{ .compiler_type_id = case[1] };
            try std.testing.expect(obligation.typeRefEqual(expected, signature.parameters[0].ty));
            try std.testing.expect(obligation.typeRefEqual(expected, signature.outcomes[0].types[0]));
        } else try std.testing.expectError(error.InvalidCallSignature, collector.callSignature(function));
    }
    const malformed = [_][]const u8{
        "",
        "ora.param_binding_ids = [], ora.param_compiler_type_ids = [6 : i256],",
        "ora.param_binding_ids = 1 : i256, ora.param_compiler_type_ids = [6 : i256],",
        "ora.param_binding_ids = [\"not-an-id\"], ora.param_compiler_type_ids = [6 : i256],",
        "ora.param_binding_ids = [\"file:0:pattern:1\"], ora.param_compiler_type_ids = 6 : i256,",
        "ora.param_binding_ids = [\"file:0:pattern:1\"], ora.param_compiler_type_ids = [],",
    };
    for (malformed) |attrs| {
        const source = try std.fmt.allocPrint(arena.allocator(),
            \\module {{ func.func @bad(%x: i256) -> i256 attributes {{ {s} ora.return_compiler_type_id = 6 : i256 }} {{ func.return %x : i256 }} }}
        , .{attrs});
        const module = mlir.oraModuleCreateParse(context, strRef(source));
        try std.testing.expect(!mlir.oraModuleIsNull(module));
        defer mlir.oraModuleDestroy(module);
        try std.testing.expectError(error.InvalidCallSignature, collector.callSignature(mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module))));
    }
    const source = try std.fmt.allocPrint(arena.allocator(),
        \\module {{
        \\  "ora.error.decl"() {{sym_name = "Failure", ora.error_id = 10 : i256, ora.param_types = [i8], ora.param_compiler_type_ids = [{d} : i256]}} : () -> ()
        \\  module @inner {{
        \\    "ora.error.decl"() {{sym_name = "Failure", ora.error_id = 20 : i256, ora.param_types = [i8, i1], ora.param_compiler_type_ids = [{d} : i256, 0 : i256]}} : () -> ()
        \\    func.func private @inner() -> !ora.error_union<i1, !ora.struct<"Failure">> attributes {{ora.return_compiler_type_id = 0 : i256, ora.return_error_ids = [20 : i256]}}
        \\  }}
        \\  func.func private @outer() -> !ora.error_union<i8, !ora.struct<"Failure">> attributes {{ora.return_compiler_type_id = {d} : i256, ora.return_error_ids = [10 : i256]}}
        \\  func.func @empty() attributes {{ora.return_compiler_type_id = 0 : i256}} {{func.return}}
        \\}}
    , .{ signed, unsigned, signed });
    const module = mlir.oraModuleCreateParse(context, strRef(source));
    try std.testing.expect(!mlir.oraModuleIsNull(module));
    defer mlir.oraModuleDestroy(module);
    try collector.indexCallDeclarations(mlir.oraModuleGetOperation(module));
    var checked: usize = 0;
    for (collector.call_declarations.items) |function| {
        if (!std.mem.eql(u8, operationName(function), "func.func")) continue;
        const name = stringAttrView(function, "sym_name").?.slice();
        // Declarations without a body cannot become callee certificates, but
        // their typed outcome reader can independently check scoped errors.
        const fn_type = mlir.oraTypeAttrGetValue(mlir.oraOperationGetAttributeByName(function, strRef("function_type")));
        const outcomes = try collector.callOutcomes(function, fn_type);
        if (std.mem.eql(u8, name, "empty")) {
            const signature = try collector.callSignature(function);
            try std.testing.expectEqual(@as(usize, 0), signature.parameters.len);
            try std.testing.expectEqual(@as(usize, 0), outcomes[0].types.len);
        } else {
            try std.testing.expectError(error.InvalidCallSignature, collector.callSignature(function));
            const inner = std.mem.eql(u8, name, "inner");
            try std.testing.expectEqual(@as(u32, if (inner) 20 else 10), outcomes[1].outcome.error_return);
            try std.testing.expectEqual(@as(usize, if (inner) 2 else 1), outcomes[1].types.len);
            try std.testing.expectEqual(if (inner) unsigned else signed, outcomes[1].types[0].compiler_type_id);
            const error_ids = mlir.oraOperationGetAttributeByName(function, strRef("ora.return_error_ids"));
            try std.testing.expect(mlir.mlirOperationRemoveAttributeByName(function, strRef("ora.return_error_ids")));
            const without_abi = try collector.callOutcomes(function, fn_type);
            try std.testing.expectEqualDeep(outcomes, without_abi);
            mlir.oraOperationSetAttributeByName(function, strRef("ora.return_error_ids"), mlir.mlirAttributeParseGet(context, strRef("[99 : i256]")));
            try std.testing.expectError(error.InvalidCallSignature, collector.callOutcomes(function, fn_type));
            mlir.oraOperationSetAttributeByName(function, strRef("ora.return_error_ids"), mlir.oraStringAttrCreate(context, strRef("invalid")));
            try std.testing.expectError(error.InvalidCallSignature, collector.callOutcomes(function, fn_type));
            mlir.oraOperationSetAttributeByName(function, strRef("ora.return_error_ids"), error_ids);
        }
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), checked);
    const checked_source =
        \\module {
        \\  func.func @first(%condition: i1) attributes {
        \\    ora.param_names = ["condition"], ora.param_binding_ids = ["file:0:pattern:1"],
        \\    ora.param_compiler_type_ids = [0 : i256], ora.return_compiler_type_id = 0 : i256
        \\  } { cf.assert %condition, "first check"
        \\      func.return }
        \\  func.func @second() attributes {ora.return_compiler_type_id = 0 : i256} {
        \\    %true = arith.constant true
        \\    cf.assert %true, "second check"
        \\    func.return
        \\  }
        \\}
    ;
    const checked_module = mlir.oraModuleCreateParse(context, strRef(checked_source));
    try std.testing.expect(!mlir.oraModuleIsNull(checked_module));
    defer mlir.oraModuleDestroy(checked_module);
    try collector.walkOperation(mlir.oraModuleGetOperation(checked_module), null);
    const first = mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(checked_module));
    const second = mlir.oraOperationGetNextInBlock(first);
    const first_context = collector.function_contexts.get(@intFromPtr(first.ptr)).?;
    const second_context = collector.function_contexts.get(@intFromPtr(second.ptr)).?;
    try std.testing.expectEqual(@as(usize, 1), first_context.param_binding_ids.len);
    try std.testing.expectEqual(@as(usize, 0), second_context.param_binding_ids.len);
    try std.testing.expectEqual(@as(usize, 1), first_context.projection.safety_sites.len);
    try std.testing.expectEqual(@as(usize, 1), second_context.projection.safety_sites.len);
    try std.testing.expect(first_context.projection.safety_sites[0].op_id != second_context.projection.safety_sites[0].op_id);
    try std.testing.expect(first_context.projection.ordinal < second_context.projection.ordinal);
    try std.testing.expect(mlir.oraBlockIsNull(collector.function.entry_block));
}

test "certified call rows projection records every referenced result at its prefix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var collector: Collector = .{ .allocator = arena.allocator(), .options = .{} };
    const first: obligation.FreeVarId = .{ .file_id = 81, .pattern_id = 1 };
    const second: obligation.FreeVarId = .{ .file_id = 81, .pattern_id = 2 };
    const untouched: obligation.FreeVarId = .{ .file_id = 81, .pattern_id = 3 };
    const ty: obligation.TypeRef = .{ .spelling = "u256" };
    const first_term = try collector.addTerm(.{ .variable = .{ .free = .{ .id = first, .name = "first", .ty = ty } } });
    const second_term = try collector.addTerm(.{ .variable = .{ .free = .{ .id = second, .name = "second", .ty = ty } } });
    const both = try collector.addTerm(.{ .binary = .{ .op = .eq, .lhs = first_term, .rhs = second_term, .ty = ty } });
    const timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "call",
        .payload = .{ .call = .{
            .callee_summary_id = 1,
            .callee_vc_sha256 = @splat(1),
            .outcome = .success,
            .results = &.{
                .{ .index = 0, .id = first, .ty = ty },
                .{ .index = 1, .id = second, .ty = ty },
                .{ .index = 2, .id = untouched, .ty = ty },
            },
        } },
    }};
    try std.testing.expectEqualSlices(obligation.FreeVarId, &.{ first, second }, try collector.readResultsForFormula(.{ .term = both }, &timeline));
    try std.testing.expectEqualSlices(obligation.FreeVarId, &.{second}, try collector.readResultsForFormula(.{ .term = second_term }, &timeline));
    try std.testing.expectEqual(@as(usize, 0), (try collector.readResultsForFormula(.{ .term = second_term }, &.{})).len);
}

test "header prefix control slots reject nonlocal and escaping addresses" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    const text =
        \\module {
        \\  func.func @prefix(%input: memref<i1>) -> memref<i1> {
        \\    %false = arith.constant false
        \\    %local = memref.alloca() : memref<i1>
        \\    memref.store %false, %local[] : memref<i1>
        \\    memref.store %false, %input[] : memref<i1>
        \\    %array = memref.alloca() : memref<4xi1>
        \\    %escape = memref.alloca() : memref<i1>
        \\    memref.store %false, %escape[] : memref<i1>
        \\    func.return %escape : memref<i1>
        \\  }
        \\}
    ;
    const module = mlir.oraModuleCreateParse(context, strRef(text));
    try std.testing.expect(!mlir.oraModuleIsNull(module));
    defer mlir.oraModuleDestroy(module);
    const function = mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module));
    const block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(function, 0));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
    // Local scalar slots are permitted; array, parameter and escaping slots
    // must not enter through the new allowance. All baseline cases remain.
    const expected = [_]struct { label: []const u8, operation: []const u8, supported: bool }{
        .{ .label = "Boolean constant", .operation = "arith.constant", .supported = true },
        .{ .label = "fresh rank-zero allocation", .operation = "memref.alloca", .supported = true },
        .{ .label = "store to fresh local", .operation = "memref.store", .supported = true },
        .{ .label = "store to nonlocal parameter", .operation = "memref.store", .supported = false },
        .{ .label = "rank-one allocation", .operation = "memref.alloca", .supported = false },
        .{ .label = "escaping allocation", .operation = "memref.alloca", .supported = false },
        .{ .label = "store to escaping slot", .operation = "memref.store", .supported = false },
        .{ .label = "address escape", .operation = "func.return", .supported = false },
    };
    var op = mlir.oraBlockGetFirstOperation(block);
    for (expected) |pin| {
        errdefer std.debug.print("header prefix pin failed: {s}\n", .{pin.label});
        try std.testing.expect(!mlir.oraOperationIsNull(op));
        try std.testing.expectEqualStrings(pin.operation, operationName(op));
        try std.testing.expectEqual(pin.supported, collector.headerPrefixOperationSupported(op, operationName(op)));
        op = mlir.oraOperationGetNextInBlock(op);
    }
    try std.testing.expect(mlir.oraOperationIsNull(op));
}

test "G9c private preheaders retain the G2 boundary in both walks" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    const cases = [_]struct { prefix: []const u8, returned: []const u8 = "%input", supported: bool }{
        .{ .prefix = "%slot = memref.alloca() : memref<i1>", .supported = true },
        .{ .prefix = "%slot = memref.alloca() : memref<i1>\n memref.store %false, %slot[] : memref<i1>", .supported = true },
        .{ .prefix = "memref.store %false, %input[] : memref<i1>", .supported = false },
        .{ .prefix = "%slot = memref.alloca() : memref<4xi1>", .supported = false },
        .{ .prefix = "%slot = memref.alloca() : memref<i1>", .returned = "%slot", .supported = false },
        .{ .prefix = "%slot = memref.alloca() : memref<i1>\n memref.store %false, %slot[] : memref<i1>\n %read = memref.load %slot[] : memref<i1>", .supported = false },
    };
    for (cases) |case| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const source = try std.fmt.allocPrint(arena.allocator(), "module {{ func.func @prefix(%input: memref<i1>) -> memref<i1> {{ %false = arith.constant false\n {s}\n func.return {s} : memref<i1> }} }}", .{ case.prefix, case.returned });
        const module = mlir.oraModuleCreateParse(context, strRef(source));
        try std.testing.expect(!mlir.oraModuleIsNull(module));
        defer mlir.oraModuleDestroy(module);
        const function = mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module));
        const block = functionEntryBlock(function);
        const anchor = mlir.oraBlockGetTerminator(block);
        for ([_]bool{ false, true }) |project_calls| {
            var collector = Collector{ .allocator = arena.allocator(), .options = .{} };
            try collector.walkOperation(mlir.oraModuleGetOperation(module), null);
            collector.function.entry_block = block;
            collector.options.project_calls = project_calls;
            const projected = if (collector.projectLoopPreheader(anchor, "prefix", 0)) |_| true else |_| false;
            const inventoried = if (collector.inventoryLoopPreheader(anchor)) |_| true else |_| false;
            try std.testing.expectEqual(project_calls and case.supported, projected);
            try std.testing.expectEqual(project_calls and case.supported, inventoried);
        }
    }
}

test "loop integer carriers retain an exact semantic width cross-check" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    try std.testing.expect(mlir.oraDialectRegister(context));

    const unsigned_256 = mlir.oraIntegerTypeGet(context, 256, false);
    const signed_128 = mlir.oraIntegerTypeGet(context, 128, true);
    const boolean = mlir.oraBoolTypeGet(context);
    try std.testing.expect(mlirIntegerCarrierHasWidth(unsigned_256, 256));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(unsigned_256, 128));
    try std.testing.expect(mlirIntegerCarrierHasWidth(signed_128, 128));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(boolean, 256));
}

test "refinement header carriers validate the actual base width" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    try std.testing.expect(mlir.oraDialectRegister(context));

    const base = mlir.oraIntegerTypeGet(context, 256, false);
    const refined = mlir.oraInRangeTypeGet(context, base, 0, 0, 0, 1, 0, 0, 0, 4);
    const narrow = mlir.oraInRangeTypeGet(context, mlir.oraIntegerTypeGet(context, 32, false), 0, 0, 0, 0, 0, 0, 0, 4);
    try std.testing.expect(!mlir.oraTypeIsNull(refined));
    try std.testing.expect(!mlir.oraTypeIsNull(narrow));
    try std.testing.expect(mlirIntegerCarrierHasWidth(refined, 256));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(refined, 32));
    try std.testing.expect(mlirIntegerCarrierHasWidth(narrow, 32));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(narrow, 256));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(mlir.oraBoolTypeGet(context), 256));
    try std.testing.expect(!mlirIntegerCarrierHasWidth(.{ .ptr = null }, 256));
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

test "dynamic access projection requires an identical independent inventory row" {
    const timeline = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "body/path", .payload = .{ .path_assumption = .{
            .formulas = &.{.{ .formula = .{ .term = 2 } }},
        } } },
        .{ .id = 1, .structural_key = "body/check", .payload = .{ .safety_checkpoint = .{
            .formulas = &.{.{ .formula = .{ .term = 3 } }},
            .dynamic_index_bounds = true,
        } } },
    };
    const access_inventory: LoopDynamicAccessInventory = .{
        .structural_key = "body/check",
        .owner_identity = "function-param:0",
        .index_identity = "loop-state:0",
    };
    const rows = [_]LoopTimelineInventoryEvent{
        .{ .structural_key = "body/path", .tag = .path_assumption },
        .{ .structural_key = "body/check", .tag = .safety_checkpoint, .formula_source = "module-op:3/result:0" },
    };
    const checkpoint = LoopSafetyCheckpointKey{ .structural_key = "body/check", .after_dependency_count = 0 };
    const semantic: ProjectedLoopEdge = .{
        .edge = .{ .id = 0, .structural_key = "body/yield", .timeline = &timeline, .target = .backedge },
        .terminator_key = "body/yield",
        .safety_checkpoints = &.{checkpoint},
        .effect_keys = &.{},
        .state_source_keys = &.{},
        .timeline_inventory = &.{},
        .dynamic_accesses = &.{access_inventory},
    };
    const inventory: InventoryLoopEdge = .{
        .structural_key = "body/yield",
        .target = .backedge,
        .terminator_key = "body/yield",
        .safety_checkpoints = &.{checkpoint},
        .effect_keys = &.{},
        .state_source_keys = &.{},
        .binding_source_keys = &.{},
        .timeline_inventory = &rows,
        .dynamic_accesses = &.{access_inventory},
    };
    try std.testing.expectEqual(.complete, compareLoopProjection(&.{semantic}, &.{inventory}));

    var missing = inventory;
    missing.dynamic_accesses = &.{};
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{missing}));

    var changed_access = access_inventory;
    changed_access.owner_identity = "function-param:1";
    var changed = inventory;
    changed.dynamic_accesses = &.{changed_access};
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{changed}));

    changed_access = access_inventory;
    changed_access.index_identity = "loop-state:1";
    changed.dynamic_accesses = &.{changed_access};
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{changed}));

    changed_access = access_inventory;
    changed_access.structural_key = "body/other-check";
    changed.dynamic_accesses = &.{changed_access};
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{changed}));

    changed.dynamic_accesses = &.{ access_inventory, access_inventory };
    try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{changed}));
}

test "indexed region timeline comparison covers every projected access field" {
    const ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const owner: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const result: obligation.FreeVarId = .{ .file_id = Collector.storage_read_file_id, .pattern_id = 9 };
    const read: obligation.LoopIndexedRegionReadEvent = .{
        .result = result,
        .region = .calldata,
        .owner = owner,
        .index = .{ .formula = .{ .term = 4 } },
        .element_ty = ty,
        .checkpoint_structural_key = "body/check",
    };
    const timeline = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "body/path", .payload = .{ .path_assumption = .{
            .formulas = &.{.{ .formula = .{ .term = 2 } }},
        } } },
        .{ .id = 1, .structural_key = "body/check", .payload = .{ .safety_checkpoint = .{
            .formulas = &.{.{ .formula = .{ .term = 3 } }},
            .dynamic_index_bounds = true,
        } } },
        .{ .id = 2, .structural_key = "body/read", .payload = .{ .indexed_region_read = read } },
    };
    const semantic_rows = [_]LoopTimelineInventoryEvent{.{
        .structural_key = "body/read",
        .tag = .indexed_region_read,
        .indexed_result = result,
        .indexed_region = .calldata,
        .indexed_owner = owner,
        .indexed_index_source = "loop-state:0",
        .indexed_element_ty = ty,
        .indexed_checkpoint_key = "body/check",
    }};
    var rows = [_]LoopTimelineInventoryEvent{
        .{ .structural_key = "body/path", .tag = .path_assumption },
        .{ .structural_key = "body/check", .tag = .safety_checkpoint },
        semantic_rows[0],
    };
    const checkpoint: LoopSafetyCheckpointKey = .{ .structural_key = "body/check", .after_dependency_count = 0 };
    const semantic: ProjectedLoopEdge = .{
        .edge = .{ .id = 0, .structural_key = "body/yield", .timeline = &timeline, .target = .backedge },
        .terminator_key = "body/yield",
        .safety_checkpoints = &.{checkpoint},
        .effect_keys = &.{"body/read"},
        .state_source_keys = &.{},
        .timeline_inventory = &semantic_rows,
    };
    const inventory: InventoryLoopEdge = .{
        .structural_key = "body/yield",
        .target = .backedge,
        .terminator_key = "body/yield",
        .binding_source_keys = &.{},
        .safety_checkpoints = &.{checkpoint},
        .effect_keys = &.{"body/read"},
        .state_source_keys = &.{},
        .timeline_inventory = &rows,
    };
    try std.testing.expectEqual(.complete, compareLoopProjection(&.{semantic}, &.{inventory}));

    inline for (.{ "result", "region", "owner", "index", "type", "checkpoint" }) |field| {
        rows[2] = semantic_rows[0];
        if (comptime std.mem.eql(u8, field, "result")) rows[2].indexed_result.?.pattern_id += 1;
        if (comptime std.mem.eql(u8, field, "region")) rows[2].indexed_region = .memory;
        if (comptime std.mem.eql(u8, field, "owner")) rows[2].indexed_owner.?.pattern_id += 1;
        if (comptime std.mem.eql(u8, field, "index")) rows[2].indexed_index_source = "loop-state:1";
        if (comptime std.mem.eql(u8, field, "type")) rows[2].indexed_element_ty = .{ .compiler_type_id = 5 };
        if (comptime std.mem.eql(u8, field, "checkpoint")) rows[2].indexed_checkpoint_key = "body/other";
        try std.testing.expectEqual(.effect_mismatch, compareLoopProjection(&.{semantic}, &.{inventory}));
    }
}

test "direct-root timeline inventory compares every storage event field" {
    const ty: obligation.TypeRef = .{ .compiler_type_id = 6 };
    const read_id: obligation.FreeVarId = .{ .file_id = Collector.storage_read_file_id, .pattern_id = 7 };
    const footprint = [_]obligation.PlaceRef{.{ .root = "counter", .region = .storage }};
    const dependency: obligation.LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "op:10/scf.while",
        .summary_id = 99,
        .loop_vc_sha256 = @as([32]u8, @splat(1)),
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

test "G6 call projection rejects malformed bodies and call boundaries" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const signed = type_builtin.lookupBuiltinById(.i256).comptime_type_id;
    const signed_callee = try std.fmt.allocPrint(allocator,
        \\func.func @signed(%v: i256) -> i256 attributes {{ora.param_names = ["v"], ora.param_binding_ids = ["file:1:pattern:0"], ora.param_compiler_type_ids = [{d} : i64], ora.return_compiler_type_id = {d} : i64}} {{ ora.return %v : i256 }}
    , .{ signed, signed });
    const Case = struct { name: []const u8, body: []const u8, suffix: []const u8 = "", parameter_type: u32 = 6, expected: ?anyerror = null, mutate_callee: ?[]const u8 = null };
    var paths = std.Io.Writer.Allocating.init(allocator);
    try paths.writer.writeAll("%zero = arith.constant 0 : i256\n %flag = arith.cmpi eq, %x, %zero : i256\n");
    for (0..9) |_| try paths.writer.writeAll("scf.if %flag { scf.yield } else { scf.yield }\n");
    try paths.writer.writeAll("ora.return %x : i256");
    const cases = [_]Case{
        .{ .name = "initialized private slot", .body = "%slot = memref.alloca() : memref<i256>\n memref.store %x, %slot[] : memref<i256>\n %v = memref.load %slot[] : memref<i256>\n ora.return %v : i256" },
        .{ .name = "uninitialized private slot", .body = "%slot = memref.alloca() : memref<i256>\n %v = memref.load %slot[] : memref<i256>\n ora.return %v : i256", .expected = error.CallPrivateSlotUninitialized },
        .{ .name = "escaping private slot", .body = "%slot = memref.alloca() : memref<i256>\n func.call @escape(%slot) : (memref<i256>) -> ()\n ora.return %x : i256", .suffix = "func.func private @escape(memref<i256>)", .expected = error.LoopProjectionEffectUnsupported },
        .{ .name = "unresolved callee", .body = "%r = func.call @missing(%x) : (i256) -> i256\n ora.return %r : i256", .suffix = "func.func private @missing(i256) -> i256", .mutate_callee = "unknown", .expected = error.UnresolvedCall },
        .{ .name = "missing body", .body = "%r = func.call @missing(%x) : (i256) -> i256\n ora.return %r : i256", .suffix = "func.func private @missing(i256) -> i256", .expected = error.MissingCallBody },
        .{ .name = "wrong result carrier", .body = "%r = func.call @wide(%x) : (i256) -> i128\n ora.return %x : i256", .suffix = try std.mem.concat(allocator, u8, &.{ signed_callee, " func.func private @wide(i256) -> i128" }), .mutate_callee = "signed", .expected = error.InvalidCallSignature },
        .{ .name = "wrong semantic signedness", .body = "%r = func.call @signed(%x) : (i256) -> i256\n ora.return %r : i256", .suffix = signed_callee, .expected = error.InvalidCallSignature },
        .{ .name = "matching semantic signedness", .body = "%r = func.call @signed(%x) : (i256) -> i256\n ora.return %r : i256", .suffix = signed_callee, .parameter_type = signed },
        .{ .name = "signed private slot", .body = "%slot = memref.alloca() : memref<i256>\n memref.store %x, %slot[] : memref<i256>\n %v = memref.load %slot[] : memref<i256>\n ora.return %v : i256", .parameter_type = signed },
        .{ .name = "external call", .body = "%r = \"func.call\"(%x) {callee = @signed, ora.trusted_extern_frame = true} : (i256) -> i256\n ora.return %r : i256", .suffix = signed_callee, .expected = error.CallExternalUnsupported },
        .{ .name = "recursive call", .body = "%r = func.call @test(%x) : (i256) -> i256\n ora.return %r : i256", .expected = error.CyclicCallDependency },
        .{ .name = "scoped owner with foreign false requires", .body = "ora.return %x : i256", .suffix =
        \\module @shadow { func.func @test() attributes {ora.return_compiler_type_id = 0 : i64} { %no = arith.constant false
        \\ "ora.requires"(%no) : (i1) -> ()
        \\ ora.return } }
        , .expected = error.CallOwnerAmbiguous },
        .{ .name = "distinct scoped owner with false requires", .body = "ora.return %x : i256", .suffix =
        \\module @shadow { func.func @unrelated() attributes {ora.return_compiler_type_id = 0 : i64} { %no = arith.constant false
        \\ "ora.requires"(%no) : (i1) -> ()
        \\ ora.return } }
        },
        .{ .name = "unreachable extra block", .body = "ora.return %x : i256\n ^unused: %z = arith.constant 0 : i256\n ora.return %z : i256", .expected = error.CallBodyControlUnsupported },
        .{ .name = "path exhaustion", .body = paths.written(), .expected = error.LoopProjectionPathLimitExceeded },
    };
    var good = true;
    for (cases) |case| {
        const source = try std.fmt.allocPrint(allocator,
            \\module {{ func.func @test(%x: i256) -> i256 attributes {{ora.param_names = ["x"], ora.param_binding_ids = ["file:0:pattern:0"], ora.param_compiler_type_ids = [{d} : i64], ora.return_compiler_type_id = {d} : i64}} {{ {s} }} {s} }}
        , .{ case.parameter_type, case.parameter_type, case.body, case.suffix });
        const module = mlir.oraModuleCreateParse(context, strRef(source));
        try std.testing.expect(!mlir.oraModuleIsNull(module));
        defer mlir.oraModuleDestroy(module);
        if (case.mutate_callee) |name| {
            const function = mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module));
            const call = mlir.oraBlockGetFirstOperation(functionEntryBlock(function));
            mlir.oraOperationSetAttributeByName(call, strRef("callee"), mlir.mlirFlatSymbolRefAttrGet(context, strRef(name)));
        }
        var collector = Collector{ .allocator = allocator, .options = .{ .project_calls = true } };
        const Harness = struct {
            fn run(c: *Collector, m: mlir.MlirModule) !void {
                try c.indexCallDeclarations(mlir.oraModuleGetOperation(m));
                try c.walkOperation(mlir.oraModuleGetOperation(m), null);
                _ = try c.requireFunctionRow(mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(m)));
                try c.projectCallFunctions();
                try obligation_to_lean.bindCollectedCallDependencies(c.allocator, .{ .terms = c.terms.items, .assumptions = c.assumptions.items, .function_summaries = c.function_summaries.items });
            }
        };
        const actual: ?anyerror = if (Harness.run(&collector, module)) |_| null else |err| err;
        if (actual == null and std.mem.eql(u8, case.name, "distinct scoped owner with false requires"))
            try std.testing.expectEqual(@as(usize, 0), collector.function_summaries.items[0].assumption_ids.len);
        if (actual != case.expected) {
            good = false;
            std.debug.print("G6 boundary {s}: expected {any}, got {any}\n", .{ case.name, case.expected, actual });
            if (actual == null) std.debug.print("projected first-row assumption ids: {any}\n", .{collector.function_summaries.items[0].assumption_ids});
        }
    }
    try std.testing.expect(good);
}

test "G6 call projection checks private initialization independently in both scans" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const Harness = struct {
        fn run(c: *Collector, module: mlir.MlirModule, semantic: bool) !void {
            // Register original MLIR identities through the baseline collector.
            // Then invoke each new scan separately, even when the other rejects.
            try c.indexCallDeclarations(mlir.oraModuleGetOperation(module));
            try c.walkOperation(mlir.oraModuleGetOperation(module), null);
            c.options.project_calls = true;
            const draft = c.loop_summary_drafts.items[0];
            const summary = c.loop_summaries.items[0];
            const function = containingFunction(draft.loop_op);
            c.function = c.function_contexts.get(@intFromPtr(function.ptr)).?;
            _ = try c.requireFunctionRow(function);
            try c.bindLoopValues(draft.loop_op, draft.loop_kind, summary.variables);
            c.use_loop_value_bindings = true;
            const first = mlir.oraBlockGetFirstOperation(mlir.oraScfWhileOpGetBeforeBlock(draft.loop_op));
            if (semantic) {
                var terminals: std.ArrayList(SemanticLoopTerminal) = .empty;
                try c.projectSemanticLoopSequence(first, .{ .key = "guard", .in_guard = true }, draft, summary.variables, &terminals);
                for (terminals.items) |terminal| _ = try c.projectedTerminalEdge(&summary, draft, terminal);
            } else {
                var terminals: std.ArrayList(InventoryLoopTerminal) = .empty;
                try c.inventoryLoopSequence(first, .{ .key = "guard", .in_guard = true }, draft, summary.variables, &terminals);
                for (terminals.items) |terminal| _ = try c.inventoryTerminalEdge(&summary, draft, terminal);
            }
        }
        fn check(ctx: mlir.MlirContext, source: []const u8, expected: ?anyerror, label: []const u8) !void {
            const module = mlir.oraModuleCreateParse(ctx, strRef(source));
            try std.testing.expect(!mlir.oraModuleIsNull(module));
            defer mlir.oraModuleDestroy(module);
            var good = true;
            for ([_]bool{ true, false }) |semantic| {
                var memory = std.heap.ArenaAllocator.init(std.testing.allocator);
                defer memory.deinit();
                var collector = Collector{ .allocator = memory.allocator(), .options = .{} };
                const actual: ?anyerror = if (run(&collector, module, semantic)) |_| null else |err| err;
                if (actual != expected) {
                    good = false;
                    std.debug.print("G6 slot {s}, {s}: expected {any}, got {any}\n", .{ label, if (semantic) "semantic" else "inventory", expected, actual });
                }
            }
            try std.testing.expect(good);
        }
    };
    const local_template =
        \\module {{
        \\  func.func @repeat() attributes {{ora.return_compiler_type_id = 0 : i64}} {{
        \\    %zero = arith.constant 0 : i256
        \\    %one = arith.constant 1 : i256
        \\    %result = scf.while (%i = %zero) : (i256) -> i256 {{
        \\      %guard = arith.cmpi ult, %i, %one : i256
        \\      scf.condition(%guard) %i : i256
        \\    }} do {{
        \\    ^bb0(%i: i256):
        \\      %inv = arith.cmpi ule, %i, %one : i256
        \\      ora.invariant %inv : i1
        \\      %choice = arith.cmpi eq, %i, %zero : i256
        \\      %slot = memref.alloca() : memref<{s}>
        \\      %other = memref.alloca() : memref<{s}>
        \\      {s}
        \\      %read = memref.load %slot[] : memref<{s}>
        \\      {s}
        \\      %next = ora.add_wrapping %i, %one {{ora.integer_signed = false}} : i256, i256 -> i256
        \\      scf.yield %next : i256
        \\    }} attributes {{ora.loop_control_manifest = [], ora.result_integer_signed_0 = false, ora.result_name_0 = "i"}} loc("ora.stmt.3"("ora.origin_stmt.3"("g6-slot.ora":3:1)))
        \\    func.return
        \\  }}
        \\  func.func @caller() attributes {{ora.return_compiler_type_id = 0 : i64}} {{ func.call @repeat() : () -> ()
        \\    func.return }}
        \\}}
    ;
    for ([_]bool{ false, true }) |boolean| {
        const ty = if (boolean) "i1" else "i256";
        const value = if (boolean) "%inv" else "%i";
        const store = try std.fmt.allocPrint(allocator, "memref.store {s}, %slot[] : memref<{s}>", .{ value, ty });
        const conditional = try std.fmt.allocPrint(allocator, "scf.if %choice {{ {s} }} else {{}}", .{store});
        const wrong = try std.mem.replaceOwned(u8, allocator, store, "%slot", "%other");
        const Case = struct { label: []const u8, before: []const u8, after: []const u8 = "", expected: ?anyerror = null };
        const cases = [_]Case{
            .{ .label = "local initialized", .before = store },
            .{ .label = "local removed", .before = "", .expected = error.CallPrivateSlotUninitialized },
            .{ .label = "local conditional", .before = conditional, .expected = error.CallPrivateSlotConditionalInitializer },
            .{ .label = "local wrong slot", .before = wrong, .expected = error.CallPrivateSlotWrongInitializerSlot },
            .{ .label = "local post-load store", .before = "", .after = store, .expected = error.CallPrivateSlotInitializerAfterLoad },
        };
        for (cases) |case| {
            const source = try std.fmt.allocPrint(allocator, local_template, .{ ty, ty, case.before, ty, case.after });
            try Harness.check(context, source, case.expected, case.label);
        }
    }
    const control_template =
        \\module {{
        \\  func.func @repeat() attributes {{ora.return_compiler_type_id = 0 : i64}} {{
        \\    %zero = arith.constant 0 : i256
        \\    %one = arith.constant 1 : i256
        \\    %false = arith.constant false
        \\    %flag = memref.alloca() : memref<i1>
        \\    %other = memref.alloca() : memref<i1>
        \\    {s}
        \\    %result = scf.while (%i = %zero) : (i256) -> i256 {{
        \\      %guard = arith.cmpi ult, %i, %one : i256
        \\      %flag_value = memref.load %flag[] : memref<i1>
        \\      %active = arith.cmpi eq, %flag_value, %false : i1
        \\      %condition = arith.andi %active, %guard {{ora.loop_source_guard_operand = 1 : i64}} : i1
        \\      scf.condition(%condition) %i : i256
        \\    }} do {{
        \\    ^bb0(%i: i256):
        \\      %inv = arith.cmpi ule, %i, %one : i256
        \\      ora.invariant %inv : i1
        \\      {s}
        \\      %body_flag = memref.load %flag[] : memref<i1>
        \\      %active = arith.cmpi eq, %body_flag, %false : i1
        \\      %next = scf.if %active -> i256 {{
        \\        %incremented = ora.add_wrapping %i, %one {{ora.integer_signed = false}} : i256, i256 -> i256
        \\        scf.yield %incremented : i256
        \\      }} else {{ scf.yield %i : i256 }} {{ora.loop_control_runtime_guard = true}}
        \\      scf.yield %next : i256
        \\    }} attributes {{ora.loop_control_manifest = [], ora.result_integer_signed_0 = false, ora.result_name_0 = "i"}} loc("ora.stmt.3"("ora.origin_stmt.3"("g6-control.ora":3:1)))
        \\    {s}
        \\    func.return
        \\  }}
        \\  func.func @caller() attributes {{ora.return_compiler_type_id = 0 : i64}} {{ func.call @repeat() : () -> ()
        \\    func.return }}
        \\}}
    ;
    const initializer = "memref.store %false, %flag[] : memref<i1>";
    const conditional = "scf.if %false { memref.store %false, %flag[] : memref<i1> } else {}";
    const cases = [_]struct { label: []const u8, prefix: []const u8, body: []const u8 = "", after: []const u8 = "", expected: ?anyerror = null }{
        .{ .label = "control initialized", .prefix = initializer },
        .{ .label = "control removed", .prefix = "", .expected = error.CallPrivateSlotUninitialized },
        .{ .label = "control conditional", .prefix = conditional, .expected = error.CallPrivateSlotConditionalInitializer },
        .{ .label = "control wrong slot", .prefix = "memref.store %false, %other[] : memref<i1>", .expected = error.CallPrivateSlotWrongInitializerSlot },
        .{ .label = "control post-load store", .prefix = "", .after = initializer, .expected = error.CallPrivateSlotInitializerAfterLoad },
        .{ .label = "control continuing write", .prefix = initializer, .body = initializer, .expected = error.CallPrivateLoopStateUnsupported },
        .{ .label = "control conditional overwrite", .prefix = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ initializer, conditional }), .expected = error.CallPrivateSlotConditionalInitializer },
        .{ .label = "control nonfalse initializer", .prefix = "%true = arith.constant true\n memref.store %true, %flag[] : memref<i1>", .expected = error.CallPrivateSlotEntryUnsupported },
    };
    for (cases) |case| try Harness.check(context, try std.fmt.allocPrint(allocator, control_template, .{ case.prefix, case.body, case.after }), case.expected, case.label);
    const continuing = try std.fmt.allocPrint(allocator, control_template, .{ initializer, initializer, "" });
    try Harness.check(context, try std.mem.replaceOwned(u8, allocator, continuing, "func.call @repeat() : () -> ()", ""), error.CallPrivateLoopStateUnsupported, "continuing write without caller");
    const control = try std.fmt.allocPrint(allocator, control_template, .{ initializer, "", "" });
    try Harness.check(context, try std.mem.replaceOwned(u8, allocator, control, "%active = arith.cmpi eq, %flag_value", "%unused = memref.load %flag[] : memref<i1>\n %active = arith.cmpi eq, %flag_value"), error.CallPrivateSlotEntryUnsupported, "unused inherited header load");
    try Harness.check(context, try std.mem.replaceOwned(u8, allocator, control, "ora.loop_source_guard_operand = 1", "ora.loop_source_guard_operand = 0"), error.CallPrivateSlotEntryUnsupported, "incorrect source guard selector");
    const cyclic_source = try std.fmt.allocPrint(allocator, control_template, .{ initializer, "func.call @repeat() : () -> ()", "" });
    const cyclic_module = mlir.oraModuleCreateParse(context, strRef(cyclic_source));
    try std.testing.expect(!mlir.oraModuleIsNull(cyclic_module));
    defer mlir.oraModuleDestroy(cyclic_module);
    var cyclic = try collect(std.testing.allocator, cyclic_module, .{ .project_calls = true });
    defer cyclic.deinit();
    try std.testing.expect(cyclic.set.hasBlockingDiagnostic());
    var cycle_named = false;
    for (cyclic.set.diagnostics) |diagnostic| if (std.mem.eql(u8, diagnostic.message, "call_projection_CyclicCallDependency")) {
        cycle_named = true;
    };
    try std.testing.expect(cycle_named);
    try std.testing.expectEqual(obligation.ArtifactDecision{ .blocked = .blocking_diagnostic }, cyclic.set.artifactDecision());
}

test "G7 source entry inventory rejects changed or missing canonical declarations" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    const source =
        \\module {
        \\  ora.contract @Local {
        \\    %yes = arith.constant true
        \\    "ora.invariant"(%yes) {ora.source_fact_id = 1 : i64, ora.source_fact_kind = "contract_invariant"} : (i1) -> ()
        \\    "ora.invariant"(%yes) {ora.source_fact_id = 2 : i64, ora.source_fact_kind = "contract_invariant"} : (i1) -> ()
        \\    func.func @leaf() attributes {ora.return_compiler_type_id = 0 : i64} { ora.return }
        \\  }
        \\  ora.contract @Foreign {
        \\    %no = arith.constant false
        \\    "ora.invariant"(%no) {ora.source_fact_id = 3 : i64, ora.source_fact_kind = "contract_invariant"} : (i1) -> ()
        \\  }
        \\}
    ;
    const Mutation = enum { unchanged, removed, owner, role, origin, reordered, duplicate, foreign_added };
    const mutations = [_]Mutation{ .unchanged, .removed, .owner, .role, .origin, .reordered, .duplicate, .foreign_added };
    for (mutations) |mutation| {
        const module = mlir.oraModuleCreateParse(context, strRef(source));
        try std.testing.expect(!mlir.oraModuleIsNull(module));
        defer mlir.oraModuleDestroy(module);
        var memory = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer memory.deinit();
        var collector = Collector{ .allocator = memory.allocator(), .options = .{ .project_calls = true } };
        try collector.indexCallDeclarations(mlir.oraModuleGetOperation(module));
        try collector.walkOperation(mlir.oraModuleGetOperation(module), null);
        try std.testing.expectEqual(@as(usize, 3), collector.obligations.items.len);
        const function = collector.call_declarations.items[0];
        const index = try collector.requireFunctionRow(function);
        var row = collector.function_summaries.items[index];
        switch (mutation) {
            .unchanged => {},
            .removed => _ = collector.obligations.orderedRemove(0),
            .owner => collector.obligations.items[0].owner = .{ .contract = "foreign" },
            .role => collector.obligations.items[0].kind.logical.role = .assert,
            .origin => collector.obligations.items[0].origin.mlir_op.ordinal += 1,
            .reordered => std.mem.swap(obligation.Obligation, &collector.obligations.items[0], &collector.obligations.items[1]),
            .duplicate => try collector.obligations.append(memory.allocator(), collector.obligations.items[0]),
            .foreign_added => collector.obligations.items[2].owner = collector.obligations.items[0].owner,
        }
        errdefer std.debug.print("G7 entry inventory {s}\n", .{@tagName(mutation)});
        if (mutation == .unchanged) {
            try collector.projectFunctionEntryFacts(&row, function);
            try std.testing.expectEqualSlices(obligation.Id, &.{ collector.obligations.items[0].id, collector.obligations.items[1].id }, row.entry_invariant_ids);
        } else try std.testing.expectError(error.CallEntryInventoryMismatch, collector.projectFunctionEntryFacts(&row, function));
    }
    const malformed = try std.mem.replaceOwned(u8, std.testing.allocator, source, "contract_invariant", "unrelated");
    defer std.testing.allocator.free(malformed);
    const module = mlir.oraModuleCreateParse(context, strRef(malformed));
    try std.testing.expect(!mlir.oraModuleIsNull(module));
    defer mlir.oraModuleDestroy(module);
    try std.testing.expectError(error.CallEntrySourceMissing, collect(std.testing.allocator, module, .{ .project_calls = true }));
}

test "SL2B lifecycle Boolean identity rejects wrong carriers and source identities" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    const module = mlir.oraModuleCreateParse(context, strRef("module { func.func @probe(%flag: i1, %byte: i8) -> i1 { func.return %flag : i1 } }"));
    try std.testing.expect(!mlir.oraModuleIsNull(module));
    defer mlir.oraModuleDestroy(module);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = Collector{ .allocator = arena.allocator(), .options = .{} };
    try c.indexCallDeclarations(mlir.oraModuleGetOperation(module));
    const function = c.call_declarations.items[0];
    const block = functionEntryBlock(function);
    const flag = mlir.oraBlockGetArgument(block, 0);
    const byte = mlir.oraBlockGetArgument(block, 1);
    const signature = try Collector.callScalarType(mlir.oraValueGetType(flag), 0);
    const ordinary = try c.callValueType(flag, &.{});
    try std.testing.expect(!obligation.typeRefEqual(signature, ordinary));
    try std.testing.expect(obligation.typeRefEqual(ordinary, .{ .compiler_type_id = 13 }));
    c.lifecycle_scope = mlir.oraModuleGetOperation(module); // local type test, no projection or authority
    const lifecycle = try c.callValueType(flag, &.{});
    const normalized = true;
    try std.testing.expectEqual(normalized, obligation.typeRefEqual(signature, lifecycle));
    const wrong = try c.callValueType(flag, &.{.{ .value_id = @intFromPtr(flag.ptr), .key = "wrong", .ty = .{ .compiler_type_id = 6 } }});
    try std.testing.expect(!obligation.typeRefEqual(signature, wrong));
    const forged_bool = try c.callValueType(byte, &.{.{ .value_id = @intFromPtr(byte.ptr), .key = "wrong", .ty = .{ .compiler_type_id = 13 } }});
    try std.testing.expect(!obligation.typeRefEqual(signature, forged_bool));
    try std.testing.expect(!callCarrierMatches(mlir.oraValueGetType(byte), signature));
    try std.testing.expectError(error.InvalidCallSignature, Collector.callScalarType(mlir.oraValueGetType(byte), 0));
    try std.testing.expectError(error.InvalidCallSignature, Collector.callScalarType(mlir.oraValueGetType(flag), 6));
    std.debug.print("SL2B_CONTROLS ordinary unchanged; exact bool={}; wrong identities and carriers refused\n", .{normalized});
}

test "SL3A exact address carriers and casts stay lifecycle scoped" {
    const context = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(context);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(context, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(context);
    try std.testing.expect(mlir.oraDialectRegister(context));
    const module = mlir.oraModuleCreateParse(context, strRef(
        "module { func.func @probe(%a: !ora.address, %n: !ora.non_zero_address, %b: i1, %x: i128, %y: i256, %u: i160, %s: si160) { func.return } }",
    ));
    try std.testing.expect(!mlir.oraModuleIsNull(module));
    defer mlir.oraModuleDestroy(module);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = Collector{ .allocator = arena.allocator(), .options = .{} };
    try c.indexCallDeclarations(mlir.oraModuleGetOperation(module));
    const block = functionEntryBlock(c.call_declarations.items[0]);
    const address = mlir.oraBlockGetArgument(block, 0);
    const address_ty = mlir.oraValueGetType(address);
    const nominal: obligation.TypeRef = .{ .compiler_type_id = 14 };
    try std.testing.expectError(error.InvalidCallSignature, c.lifecycleCallScalarType(address_ty, 0));
    try std.testing.expect(!c.lifecycleCallCarrierMatches(address_ty, nominal));
    c.lifecycle_scope = mlir.oraModuleGetOperation(module); // isolated helper checks, no projection/authority
    for (0..2) |index| {
        const value = mlir.oraBlockGetArgument(block, index);
        const carrier = mlir.oraValueGetType(value);
        try std.testing.expect(obligation.typeRefEqual(nominal, try c.lifecycleCallScalarType(carrier, 0)));
        try std.testing.expect(c.lifecycleCallCarrierMatches(carrier, nominal));
        try std.testing.expect(obligation.typeRefEqual(nominal, try c.lifecycleMapKeyType(value)));
        try std.testing.expectError(error.LoopProjectionEffectUnsupported, c.exactLoopIntegerTypeForValue(value));
        for ([_]u32{ 6, 14, 18 }) |wrong_id|
            try std.testing.expectError(error.InvalidCallSignature, c.lifecycleCallScalarType(carrier, wrong_id));
    }
    for (2..7) |index| {
        const carrier = mlir.oraValueGetType(mlir.oraBlockGetArgument(block, index));
        try std.testing.expect(!c.lifecycleCallCarrierMatches(carrier, nominal));
        try std.testing.expectError(error.InvalidCallSignature, c.lifecycleCallScalarType(carrier, 14));
        if (index != 2) try std.testing.expectError(error.InvalidCallSignature, c.lifecycleCallScalarType(carrier, 0));
    }
    const cases = [_]struct { name: []const u8, input: usize, output: usize, arity: usize = 1, results: usize = 1, valid: bool = false }{
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 5, .valid = true },
        .{ .name = "ora.i160.to.addr", .input = 5, .output = 0, .valid = true },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 5, .arity = 0 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 5, .arity = 2 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 5, .results = 0 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 5, .results = 2 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 2 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 3 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 4 },
        .{ .name = "ora.addr.to.i160", .input = 0, .output = 6 },
        .{ .name = "ora.addr.to.i160", .input = 5, .output = 0 },
        .{ .name = "ora.addr.to.i160", .input = 1, .output = 5 },
        .{ .name = "ora.i160.to.addr", .input = 6, .output = 0 },
        .{ .name = "ora.i160.to.addr", .input = 4, .output = 0 },
        .{ .name = "ora.i160.to.addr", .input = 5, .output = 1 },
        .{ .name = "ora.i160.to.addr", .input = 0, .output = 5 },
    };
    for (cases) |case| {
        const input = mlir.oraBlockGetArgument(block, case.input);
        const operands = [_]mlir.MlirValue{ input, input };
        const ty = mlir.oraValueGetType(mlir.oraBlockGetArgument(block, case.output));
        const types = [_]mlir.MlirType{ ty, ty };
        const op = mlir.oraOperationCreate(context, mlir.mlirLocationUnknownGet(context), strRef(case.name), &operands, case.arity, &types, case.results, null, 0, 0, false);
        try std.testing.expect(!mlir.oraOperationIsNull(op));
        defer mlir.mlirOperationDestroy(op);
        if (case.valid) {
            try std.testing.expectEqual(input.ptr, (try c.lifecycleAddressCastOperand(op)).?.ptr);
            try std.testing.expect(!isTransparentValueOp(case.name));
            c.lifecycle_scope = null;
            try std.testing.expectEqual(null, try c.lifecycleAddressCastOperand(op));
            c.lifecycle_scope = mlir.oraModuleGetOperation(module);
        } else try std.testing.expectError(error.InvalidLifecycleAddressCast, c.lifecycleAddressCastOperand(op));
    }
}
