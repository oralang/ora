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
    .{ "arith.shli", .shift_amount_bounds },
    .{ "arith.shrsi", .shift_amount_bounds },
    .{ "arith.shrui", .shift_amount_bounds },
});

const assert_safety_map = std.StaticStringMap(obligation.ArithmeticSafetyKind).initComptime(.{
    .{ "checked addition overflow", .addition_overflow },
    .{ "checked subtraction overflow", .subtraction_overflow },
    .{ "checked multiplication overflow", .multiplication_overflow },
    .{ "checked power overflow", .power_overflow },
    .{ "checked negation overflow", .negation_overflow },
});

const transparent_value_ops = std.StaticStringMap(void).initComptime(.{
    .{ "ora.refinement_to_base", {} },
    .{ "ora.base_to_refinement", {} },
    .{ "arith.bitcast", {} },
    .{ "arith.extui", {} },
    .{ "arith.extsi", {} },
    .{ "arith.trunci", {} },
    .{ "arith.index_cast", {} },
    .{ "arith.index_castui", {} },
    .{ "arith.index_castsi", {} },
    .{ "builtin.unrealized_conversion_cast", {} },
    .{ "tensor.cast", {} },
});

const arithmetic_value_op_map = std.StaticStringMap(obligation.BinaryOp).initComptime(.{
    .{ "arith.addi", .add },
    .{ "arith.subi", .sub },
    .{ "arith.muli", .mul },
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
                query.artifact_policy = .blocks_verified_artifacts;
                query.proof_requirement = .lean_certificate;
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

fn loopSummaryByIdMut(
    rows: []obligation.LoopSummaryRow,
    id: obligation.Id,
) ?*obligation.LoopSummaryRow {
    for (rows) |*row| if (row.id == id) return row;
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

const Collector = struct {
    allocator: std.mem.Allocator,
    options: CollectOptions,
    terms: std.ArrayList(obligation.Term) = .empty,
    obligations: std.ArrayList(obligation.Obligation) = .empty,
    assumptions: std.ArrayList(obligation.Assumption) = .empty,
    queries: std.ArrayList(obligation.VerificationQuery) = .empty,
    loop_summaries: std.ArrayList(obligation.LoopSummaryRow) = .empty,
    loop_operation_bindings: std.ArrayList(LoopOperationBinding) = .empty,
    active_loop_summary_indices: std.ArrayList(usize) = .empty,
    query_bindings: std.ArrayList(obligation.FormalQueryBinding) = .empty,
    source_fact_bindings: std.ArrayList(SourceFactOpBinding) = .empty,
    runtime_owner_bindings: std.ArrayList(RuntimeOwnerBinding) = .empty,
    runtime_check_producers: std.ArrayList(RuntimeCheckProducer) = .empty,
    state_effect_producers: std.ArrayList(StateEffectProducer) = .empty,
    diagnostics: std.ArrayList(obligation.ObligationDiagnostic) = .empty,
    resource_sites: std.ArrayList(ResourceCompletenessSite) = .empty,
    active_binders: std.ArrayList(BinderFrame) = .empty,
    loop_value_bindings: std.ArrayList(LoopValueBinding) = .empty,
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

    const BinderFrame = struct {
        name: []const u8,
        ty: obligation.TypeRef,
        region: ?obligation.RegionRef = null,
    };

    const LoopValueBinding = struct {
        value_id: usize,
        variable: obligation.LoopVariable,
    };

    const LoopOperationBinding = struct {
        summary_id: obligation.Id,
        source_op_id: usize,
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
        const op_name = operationName(op);
        const symbol = try self.symbolForOperation(op, op_name, inherited_symbol);
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
        const previous_loop_operation_binding_len = self.loop_operation_bindings.items.len;
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
            self.loop_operation_bindings.shrinkRetainingCapacity(previous_loop_operation_binding_len);
            self.active_loop_summary_indices.shrinkRetainingCapacity(previous_active_loop_summary_len);
        }
        defer {
            if (is_function) {
                self.active_binders.shrinkRetainingCapacity(previous_binder_len);
                self.loop_value_bindings.shrinkRetainingCapacity(previous_loop_value_binding_len);
                self.loop_operation_bindings.shrinkRetainingCapacity(previous_loop_operation_binding_len);
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
            if (!self.operationFeedsOnlyFormalFormula(op)) {
                try self.addArithmeticSafetyOp(op_name, symbol, ordinal, safety, null, try self.sourceForOperation(op));
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
        const init_formulas = try self.collectLoopInitFormulas(op, loop_kind, symbol, ordinal);
        const guard_formula = try self.collectLoopGuardFormula(op, loop_kind, symbol, ordinal);
        const invariant_formulas = try self.collectLoopInvariantFormulas(op, loop_kind, symbol, ordinal);
        const step_assignments = try self.collectLoopStepAssignments(op, loop_kind, symbol, ordinal);
        const body = loopBodyBlock(op, loop_kind);
        const body_facts = collectLoopBodyFacts(body);

        if (loop_source_op_id == null) try appendLoopReason(&reasons, self.allocator, .loop_identity_missing);
        for (variables) |variable| {
            if (variable.id == null) try appendLoopReason(&reasons, self.allocator, .loop_identity_missing);
            const value = loopVariableValue(op, loop_kind, variable.index);
            if (mlir.oraValueIsNull(value) or !mlirTypeIsSupportedU256Carrier(mlir.oraValueGetType(value))) {
                try appendLoopReason(&reasons, self.allocator, .loop_variable_not_u256);
            }
        }
        if (guard_formula == null) try appendLoopReason(&reasons, self.allocator, .loop_guard_missing);
        if (invariant_formulas.len == 0) try appendLoopReason(&reasons, self.allocator, .loop_invariant_missing);
        if (loop_kind == .other) try appendLoopReason(&reasons, self.allocator, .loop_kind_unsupported);
        if (!loopStepAssignmentsAreScalar(variables, step_assignments)) {
            try appendLoopReason(&reasons, self.allocator, .loop_update_not_scalar_assignment);
        }
        if (formulasContainOriginValue(init_formulas) or
            optionalFormulaIsOriginValue(guard_formula) or
            formulasContainOriginValue(invariant_formulas) or
            stepAssignmentsContainOriginValue(step_assignments))
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
        if (body_facts.has_nested_loop) try appendLoopReason(&reasons, self.allocator, .loop_has_nested_loop);
        if (body_facts.has_branching_body) try appendLoopReason(&reasons, self.allocator, .loop_has_branching_body);

        const id = self.next_loop_summary_id;
        self.next_loop_summary_id += 1;
        if (sourceOpId(op)) |source_op_id| {
            try self.loop_operation_bindings.append(self.allocator, .{
                .summary_id = id,
                .source_op_id = source_op_id,
            });
        }
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
            .guard_formula = guard_formula,
            .invariant_formulas = invariant_formulas,
            .step_assignments = step_assignments,
            .body_safety_formulas = &.{},
            .query_ids = .{},
            .unsupported_reasons = try reasons.toOwnedSlice(self.allocator),
        });
    }

    fn appendActiveLoopBodySafetyFormula(
        self: *Collector,
        formula: obligation.FormulaRef,
    ) !void {
        if (self.active_loop_summary_indices.items.len == 0) return;
        const summary_index = self.active_loop_summary_indices.items[self.active_loop_summary_indices.items.len - 1];
        const summary = &self.loop_summaries.items[summary_index];
        summary.body_safety_formulas = try appendFormula(self.allocator, summary.body_safety_formulas, formula);
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

    fn finalizeActiveLoopSummary(self: *Collector) !void {
        if (self.active_loop_summary_indices.items.len == 0) return error.MissingActiveLoopSummary;
        const summary_index = self.active_loop_summary_indices.items[self.active_loop_summary_indices.items.len - 1];
        const summary = &self.loop_summaries.items[summary_index];
        if (self.stepAssignmentsRequireSafety(summary.step_assignments) and summary.body_safety_formulas.len == 0) {
            try self.appendLoopSummaryReason(summary, .loop_body_safety_missing);
        }
        try self.addLoopInductionQuery(summary);
    }

    fn addLoopInductionQuery(
        self: *Collector,
        summary: *obligation.LoopSummaryRow,
    ) !void {
        if (summary.owner != .statement) return error.InvalidLoopSummaryOwner;
        const owner: obligation.Owner = .{ .function = .{
            .name = summary.owner.statement.function_name,
        } };
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
            .assumption_ids = try self.assumptionIdsForOwner(owner),
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
            .binary => |binary| switch (binary.op) {
                .add, .sub, .mul, .div, .mod => true,
                else => self.termRequiresArithmeticSafety(.{ .term = binary.lhs }, fuel - 1) or
                    self.termRequiresArithmeticSafety(.{ .term = binary.rhs }, fuel - 1),
            },
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
                .ty = try self.typeRefFromValue(value),
            };
        }
        return variables;
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
            formula.* = try self.loopFormulaFromValue(value, operationName(op), symbol, ordinal, @intCast(operand_index));
        }
        return formulas;
    }

    fn collectLoopGuardFormula(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        symbol: ?[]const u8,
        ordinal: u32,
    ) !?obligation.FormulaRef {
        if (loop_kind == .scf_for) {
            if (mlir.oraOperationGetNumOperands(op) < 2) return null;
            const induction = loopVariableValue(op, loop_kind, 0);
            const induction_term = (try self.termFromValue(induction)) orelse return null;
            const upper_term = (try self.termFromValue(mlir.oraOperationGetOperand(op, 1))) orelse return null;
            return .{ .term = try self.addTerm(.{ .binary = .{
                .op = .lt,
                .lhs = induction_term,
                .rhs = upper_term,
                .ty = try self.typeRefFromValue(induction),
            } }) };
        }
        if (loop_kind != .scf_while) return null;
        const before = mlir.oraScfWhileOpGetBeforeBlock(op);
        if (mlir.oraBlockIsNull(before)) return null;
        const condition = findOperationInBlock(before, "scf.condition") orelse return null;
        if (mlir.oraOperationGetNumOperands(condition) == 0) return null;
        return try self.loopFormulaFromValue(
            mlir.oraOperationGetOperand(condition, 0),
            "scf.condition",
            symbol,
            ordinal,
            0,
        );
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

    fn collectLoopStepAssignments(
        self: *Collector,
        op: mlir.MlirOperation,
        loop_kind: obligation.LoopKind,
        symbol: ?[]const u8,
        ordinal: u32,
    ) ![]const obligation.LoopStepAssignment {
        const body = loopBodyBlock(op, loop_kind);
        if (mlir.oraBlockIsNull(body)) return &.{};
        const terminator = mlir.oraBlockGetTerminator(body);
        if (mlir.oraOperationIsNull(terminator) or !std.mem.eql(u8, operationName(terminator), "scf.yield")) return &.{};
        const yielded_count: usize = @intCast(mlir.oraOperationGetNumOperands(terminator));
        const induction_count: usize = @intFromBool(loop_kind == .scf_for);
        const assignments = try self.allocator.alloc(obligation.LoopStepAssignment, yielded_count + induction_count);
        if (loop_kind == .scf_for) {
            if (mlir.oraOperationGetNumOperands(op) < 3) return &.{};
            const induction = loopVariableValue(op, loop_kind, 0);
            const step = mlir.oraOperationGetOperand(op, 2);
            const induction_term = (try self.termFromValue(induction)) orelse return &.{};
            const step_term = (try self.termFromValue(step)) orelse return &.{};
            const value = try self.addTerm(.{ .binary = .{
                .op = .add,
                .lhs = induction_term,
                .rhs = step_term,
                .ty = try self.typeRefFromValue(induction),
            } });
            assignments[0] = .{
                .variable_index = 0,
                .target = self.loopVariableForValue(induction).?.id,
                .value = .{ .term = value },
            };
        }
        for (assignments[induction_count..], 0..) |*assignment, index| {
            const variable_index = index + induction_count;
            const target_variable = self.loopVariableForValue(loopVariableValue(op, loop_kind, @intCast(variable_index)));
            assignment.* = .{
                .variable_index = @intCast(variable_index),
                .target = if (target_variable) |variable| variable.id else null,
                .value = try self.loopFormulaFromValue(
                    mlir.oraOperationGetOperand(terminator, index),
                    "scf.yield",
                    symbol,
                    ordinal,
                    @intCast(index),
                ),
            };
        }
        return assignments;
    }

    fn loopFormulaFromValue(
        self: *Collector,
        value: mlir.MlirValue,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        index: u32,
    ) !obligation.FormulaRef {
        if (try self.termFromValue(value)) |term| return .{ .term = term };
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

        if (write_slots.len != 0 or modifies_slots_attr != null) {
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
            if (!typeRefIsU256Carrier(disequality.lhs.ty) or !typeRefIsU256Carrier(disequality.rhs.ty)) continue;
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
                    try self.appendActiveLoopBodySafetyFormula(body_formula);
                }
                try self.addArithmeticSafetyOp(op_name, symbol, ordinal, safety, formula, try self.sourceForOperation(op));
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
            try self.appendActiveLoopBodySafetyFormula(body_formula);
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
        const loop_post_formula = if (role == .ensures and self.functionHasLoopSummary(symbol)) blk: {
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
        if (role != .invariant and self.active_loop_summary_indices.items.len != 0) {
            const body_formula = (try self.valueOperand(op, op_name, symbol, ordinal, 0)) orelse {
                try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                return;
            };
            try self.appendActiveLoopBodySafetyFormula(body_formula);
        }

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
        try self.addQueryForOperation(.obligation, role, null, id, owner, origin, source, op);
        if (loop_post_formula) |post_formula| {
            try self.attachLoopPostQueries(
                op,
                id,
                owner,
                origin,
                source,
                post_formula,
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

    fn attachLoopPostQueries(
        self: *Collector,
        ensures_op: mlir.MlirOperation,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
        source: obligation.SourceRef,
        post_formula: obligation.FormulaRef,
    ) !void {
        if (owner != .function) return;
        const ensures_source_op_id = sourceOpId(ensures_op) orelse return;
        for (self.loop_summaries.items) |*summary| {
            if (summary.owner != .statement or
                !std.mem.eql(u8, summary.owner.statement.function_name, owner.function.name)) continue;
            const loop_source_op_id = self.loopOperationSourceId(summary.id) orelse continue;
            summary.post_formulas = try appendFormula(self.allocator, summary.post_formulas, post_formula);
            if (post_formula == .origin_value) try self.appendLoopSummaryReason(summary, .loop_formula_unsupported);

            const assumption_ids = try self.assumptionIdsForOwner(owner);
            const obligation_ids = try self.allocator.alloc(obligation.Id, 1);
            obligation_ids[0] = obligation_id;
            const query_id = self.nextId();
            try self.queries.append(self.allocator, .{
                .id = query_id,
                .owner = owner,
                .source = source,
                .phase = .report,
                .origin = origin,
                .kind = .loop_invariant_post,
                .logical_role = .ensures,
                .obligation_ids = obligation_ids,
                .assumption_ids = assumption_ids,
                .loop_summary_id = summary.id,
            });
            summary.query_ids.post = try appendId(self.allocator, summary.query_ids.post, query_id);
            try self.query_bindings.append(self.allocator, .{
                .source_op_id = ensures_source_op_id,
                .loop_source_op_id = loop_source_op_id,
                .kind = .loop_invariant_post,
                .logical_role = .ensures,
                .query_id = query_id,
                .assumption_ids = assumption_ids,
                .obligation_ids = obligation_ids,
            });
        }
    }

    fn loopOperationSourceId(self: *Collector, summary_id: obligation.Id) ?usize {
        for (self.loop_operation_bindings.items) |binding| {
            if (binding.summary_id == summary_id) return binding.source_op_id;
        }
        return null;
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
            try self.appendActiveLoopBodySafetyFormula(body_formula);
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
        try self.addQuery(.obligation, .arithmetic_safety, null, id, owner, origin, source);
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

        if (self.use_loop_value_bindings) if (self.loopVariableForValue(value)) |variable| {
            return try self.addTerm(.{ .variable = .{ .free = .{
                .id = variable.id orelse return null,
                .name = variable.name orelse "loop_var",
                .ty = try self.typeRefFromValueWithIntegerSignedness(value, signed_override),
            } } });
        };

        if (mlir.mlirValueIsABlockArgument(value)) {
            const arg_number: usize = self.functionEntryBlockArgumentNumber(value) orelse return null;
            const name = if (arg_number < self.function_param_names.len)
                self.function_param_names[arg_number]
            else
                try std.fmt.allocPrint(self.allocator, "arg#{d}", .{arg_number});
            return try self.addTerm(.{ .variable = .{
                .free = .{
                    .id = self.freeVarIdForFunctionParam(arg_number),
                    .name = name,
                    .ty = try self.typeRefFromValueWithIntegerSignedness(value, signed_override),
                },
            } });
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
        if (std.mem.eql(u8, owner_name, "arith.constant")) return try self.constantTerm(owner, signed_override);
        if (arithmetic_value_op_map.get(owner_name)) |binary_op| {
            if (signed_override) |signed| return try self.binaryTermFromOperands(owner, binary_op, .{ .explicit_signedness = signed });
            return try self.binaryTermFromOperands(owner, binary_op, .operand_signedness);
        }
        if (arithmetic_div_rem_value_op_map.get(owner_name)) |spec| {
            return try self.binaryTermFromOperands(owner, spec.op, .{ .explicit_signedness = spec.signed });
        }
        if (std.mem.eql(u8, owner_name, "arith.xori")) return try self.xoriTerm(owner);
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
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        const root = try self.stringAttr(op, "global") orelse return null;
        if (!self.canProjectStablePlace(.{ .root = root, .region = .storage })) return null;
        return try self.placeReadTerm(root);
    }

    fn storageMapReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        const place = (try self.storageMapReadPlaceFromMapGet(op)) orelse return null;
        if (!self.canProjectStablePlace(place)) return null;
        return try self.placeReadTermFromPlace(place);
    }

    fn canProjectStablePlace(self: *Collector, place: obligation.PlaceRef) bool {
        if (self.function_has_external_call) return false;
        if (!self.function_write_slots_complete) return false;
        const write_slots = self.function_write_slots orelse return false;
        return placeIsDefinitelyDisjointFromAll(place, write_slots);
    }

    fn oldTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) != 1) return null;
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        if (self.function_has_external_call) return null;
        if (!self.function_write_slots_complete) return null;
        const write_slots = self.function_write_slots orelse return null;

        const operand = mlir.oraOperationGetOperand(op, 0);
        if (try self.storageMapReadPlaceFromValue(operand)) |place| {
            if (placeIsDefinitelyDisjointFromAll(place, write_slots)) return try self.placeReadTermFromPlace(place);
            return null;
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
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(owner, 0))) return null;
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

    fn xoriTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) < 2) return null;
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
        if (arg_count < expected_count) {
            try self.addBlockingDiagnosticFmt(
                .missing_type,
                "func.func has {d} named parameters but only {d} MLIR block arguments",
                .{ expected_count, arg_count },
            );
        }

        for (types, 0..) |*ty, index| {
            ty.* = if (index < arg_count)
                try self.typeRefFromValue(mlir.oraBlockGetArgument(block, index))
            else
                try self.unknownTypeRef();
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
        const lhs_type = mlir.oraValueGetType(mlir.oraOperationGetOperand(op, 0));
        const width = integerTypeWidth(lhs_type) orelse return null;
        const signed = switch (type_source) {
            .operand_signedness => mlirIntegerTypeIsSigned(lhs_type),
            .predicate_signedness => binaryOpSignedness(binary_op) orelse mlirIntegerTypeIsSigned(lhs_type),
            .explicit_signedness => |value| value,
        };
        return try self.integerTypeRef(width, signed);
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

fn typeRefIsU256Carrier(ty: ?obligation.TypeRef) bool {
    const value = ty orelse return false;
    return switch (value) {
        .compiler_type_id => |id| blk: {
            const info = type_builtin.integerInfoByComptimeTypeId(id) orelse break :blk false;
            break :blk info.width == 256;
        },
        .spelling => |name| std.mem.eql(u8, name, "u256") or
            std.mem.eql(u8, name, "uint256") or
            std.mem.eql(u8, name, "i256") or
            std.mem.eql(u8, name, "int256"),
    };
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

fn mlirTypeIsSupportedU256Carrier(ty: mlir.MlirType) bool {
    if (mlir.oraTypeIsNull(ty)) return false;
    const refinement_base = mlir.oraRefinementTypeGetBaseType(ty);
    if (!mlir.oraTypeIsNull(refinement_base)) return mlirTypeIsSupportedU256Carrier(refinement_base);
    return integerTypeWidth(ty) == 256;
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
        if (std.mem.eql(u8, name, "ora.sstore") or std.mem.eql(u8, name, "ora.tstore")) facts.has_storage_write = true;
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

fn formulasContainOriginValue(formulas: []const obligation.FormulaRef) bool {
    for (formulas) |formula| if (formula == .origin_value) return true;
    return false;
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
    if (assignments.len != variables.len and assignments.len + 1 != variables.len) return false;
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
