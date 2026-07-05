//! Ora MLIR to obligation manifest adapter.
//!
//! This is a boundary collector, not a verifier and not a proof encoder. It
//! walks canonical Ora MLIR, records the obligation surface already present
//! there, and leaves formulas owned by MLIR values. Z3 and Lean exporters must
//! consume the same manifest instead of rediscovering different obligations.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const obligation = @import("obligation.zig");
const type_builtin = @import("ora_types").builtin;

pub const CollectOptions = struct {
    /// Borrowed owner used when the walker has not entered a symbol-bearing op.
    owner: obligation.Owner = .{ .module = "ora_mlir" },
};

pub const CollectResult = struct {
    arena: std.heap.ArenaAllocator,
    set: obligation.ObligationSet,

    pub fn deinit(self: *CollectResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
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
    .same_place_net_zero,
    .conservation,
    .modifies_covered,
};

const resource_create_properties = [_]obligation.ResourceProperty{
    .amount_non_negative,
    .destination_no_overflow,
    .modifies_covered,
};

const resource_destroy_properties = [_]obligation.ResourceProperty{
    .amount_non_negative,
    .source_sufficient,
    .modifies_covered,
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

    const set: obligation.ObligationSet = .{
        .terms = try collector.terms.toOwnedSlice(collector.allocator),
        .obligations = try collector.obligations.toOwnedSlice(collector.allocator),
        .assumptions = try collector.assumptions.toOwnedSlice(collector.allocator),
        .queries = try collector.queries.toOwnedSlice(collector.allocator),
        .diagnostics = try collector.diagnostics.toOwnedSlice(collector.allocator),
    };

    return .{
        .arena = arena,
        .set = set,
    };
}

const Collector = struct {
    allocator: std.mem.Allocator,
    options: CollectOptions,
    terms: std.ArrayList(obligation.Term) = .empty,
    obligations: std.ArrayList(obligation.Obligation) = .empty,
    assumptions: std.ArrayList(obligation.Assumption) = .empty,
    queries: std.ArrayList(obligation.VerificationQuery) = .empty,
    diagnostics: std.ArrayList(obligation.ObligationDiagnostic) = .empty,
    active_binders: std.ArrayList(BinderFrame) = .empty,
    next_id: obligation.Id = 1,
    next_ordinal: u32 = 0,
    function_param_names: []const []const u8 = &.{},
    function_param_binding_ids: []const obligation.FreeVarId = &.{},
    function_param_types: []const obligation.TypeRef = &.{},
    function_entry_block: mlir.MlirBlock = std.mem.zeroes(mlir.MlirBlock),
    function_write_slots: ?[]const obligation.PlaceRef = null,
    function_write_slots_complete: bool = false,
    function_has_external_call: bool = false,

    const synthetic_file_id = std.math.maxInt(u32);

    const BinderFrame = struct {
        name: []const u8,
        ty: obligation.TypeRef,
        region: ?obligation.RegionRef = null,
    };

    const ResourcePlaces = struct {
        source: ?obligation.PlaceRef = null,
        destination: ?obligation.PlaceRef = null,
    };

    const MoveOperandSegments = struct {
        source_len: usize,
        destination_len: usize,
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
        const previous_binder_len = self.active_binders.items.len;
        const is_function = std.mem.eql(u8, op_name, "func.func");
        if (is_function) {
            self.function_param_names = (try self.stringArrayAttr(op, "ora.param_names")) orelse &.{};
            self.function_param_binding_ids = (try self.freeVarIdArrayAttr(op, "ora.param_binding_ids")) orelse &.{};
            self.function_entry_block = functionEntryBlock(op);
            self.function_param_types = try self.functionParamTypesFromFunctionOp(op, self.function_param_names.len);
            self.function_write_slots = try self.placeArrayAttr(op, "ora.write_slots");
            self.function_write_slots_complete = (try self.boolAttr(op, "ora.write_slots_complete")) orelse false;
            self.function_has_external_call = operationBodyHasExternalCall(op);
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
        }
        defer {
            if (is_function) {
                self.active_binders.shrinkRetainingCapacity(previous_binder_len);
                self.function_param_types = previous_param_types;
                self.function_param_binding_ids = previous_param_binding_ids;
                self.function_param_names = previous_param_names;
                self.function_entry_block = previous_entry_block;
                self.function_has_external_call = previous_has_external_call;
                self.function_write_slots_complete = previous_write_slots_complete;
                self.function_write_slots = previous_write_slots;
            }
        }

        try self.collectEffectFrameAttrs(op, op_name, symbol, ordinal);
        if (op_kind_map.get(op_name)) |kind| {
            try self.collectOperation(op, kind, op_name, symbol, ordinal);
        } else if (arithmetic_op_map.get(op_name)) |safety| {
            if (!self.operationFeedsOnlyFormalFormula(op)) {
                try self.addArithmeticSafetyOp(op_name, symbol, ordinal, safety, null, try self.sourceForOperation(op));
            }
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
        const modifies_slots = (try self.placeArrayAttr(op, "ora.modifies_slots")) orelse &.{};
        const source = try self.sourceForOperation(op);

        if (write_slots.len != 0 or modifies_slots.len != 0) {
            try self.addEffectFrameGoal(.write_covered_by_modifies, modifies_slots, write_slots, op_name, symbol, ordinal, source);
        }
        if (read_slots.len != 0 and write_slots.len != 0) {
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
            } },
        });
        try self.addQueryNoAssumptions(.obligation, null, null, id, owner, origin, source);
    }

    fn readsDisjointFromWrites(
        self: *Collector,
        reads: []const obligation.PlaceRef,
        writes: []const obligation.PlaceRef,
    ) ![]const obligation.PlaceRef {
        if (reads.len == 0) return &.{};

        var preserved: std.ArrayList(obligation.PlaceRef) = .empty;
        for (reads) |read| {
            if (!placeListContains(writes, read)) try preserved.append(self.allocator, read);
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
            try self.addQuery(.guard_satisfy, null, guard_id, id, owner, origin, source);
            try self.addQuery(.guard_violate, null, guard_id, id, owner, origin, source);
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
            try self.addQuery(.obligation, .guard, null, id, owner, origin, source);
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
        });
        try self.addQuery(.obligation, role, null, id, owner, origin, source);
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
        try self.addQuery(.guard_satisfy, null, guard_id, id, owner, origin, source);
        try self.addQuery(.guard_violate, null, guard_id, id, owner, origin, source);
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
        const amount = (try self.formulaOperand(op, op_name, symbol, ordinal, @intCast(amount_index))) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const places = try self.resourcePlaces(op, resource_op, operand_count, op_name) orelse return;
        const source = try self.sourceForOperation(op);

        for (resourceProperties(resource_op)) |property| {
            try self.addResourceGoal(resource_op, domain, places, amount, symbol, ordinal, property, source);
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
            return .{ .parameter = @intCast(arg_number) };
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
        try self.appendQuery(kind, logical_role, guard_id, obligation_id, owner, origin, source, assumption_ids);
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
        try self.appendQuery(kind, logical_role, guard_id, obligation_id, owner, origin, source, &.{});
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
    ) !void {
        const obligation_ids = try self.allocator.alloc(obligation.Id, 1);
        obligation_ids[0] = obligation_id;
        try self.queries.append(self.allocator, .{
            .id = self.nextId(),
            .owner = owner,
            .source = source,
            .phase = .report,
            .origin = origin,
            .kind = kind,
            .logical_role = logical_role,
            .guard_id = guard_id,
            .obligation_ids = obligation_ids,
            .assumption_ids = assumption_ids,
        });
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

    fn termFromValue(self: *Collector, value: mlir.MlirValue) anyerror!?obligation.TermId {
        return self.termFromValueWithIntegerSignedness(value, null);
    }

    fn termFromValueWithIntegerSignedness(
        self: *Collector,
        value: mlir.MlirValue,
        signed_override: ?bool,
    ) anyerror!?obligation.TermId {
        if (mlir.oraValueIsNull(value)) return null;

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

    fn scalarStorageLoadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        const root = try self.stringAttr(op, "global") orelse return null;
        if (!self.canProjectStorageRoot(root)) return null;
        return try self.placeReadTerm(root);
    }

    fn storageMapReadTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        const place = (try self.storageMapReadPlaceFromMapGet(op)) orelse return null;
        if (!self.canProjectStorageRoot(place.root)) return null;
        return try self.placeReadTermFromPlace(place);
    }

    fn canProjectStorageRoot(self: *Collector, root: []const u8) bool {
        if (self.function_has_external_call) return false;
        if (!self.function_write_slots_complete) return false;
        const write_slots = self.function_write_slots orelse return false;
        return !storageWriteSlotsContain(write_slots, root);
    }

    fn oldTerm(self: *Collector, op: mlir.MlirOperation) anyerror!?obligation.TermId {
        if (mlir.oraOperationGetNumOperands(op) != 1) return null;
        if (!mlirTypeIsSupportedU256Carrier(self.operationResultType(op, 0))) return null;
        if (self.function_has_external_call) return null;
        if (!self.function_write_slots_complete) return null;
        const write_slots = self.function_write_slots orelse return null;

        const operand = mlir.oraOperationGetOperand(op, 0);
        if (try self.storageMapReadPlaceFromValue(operand)) |place| {
            if (!storageWriteSlotsContain(write_slots, place.root)) return try self.placeReadTermFromPlace(place);
            return null;
        }

        const root = (try self.scalarStorageLoadRootFromValue(operand)) orelse return null;

        if (!storageWriteSlotsContain(write_slots, root)) return try self.placeReadTerm(root);

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
                    if (!freeVarIdEql(free.id, free_id)) return term_id;
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
            return .{ .parameter = try std.fmt.parseInt(u32, segment["param#".len..], 10) };
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

fn freeVarIdEql(lhs: obligation.FreeVarId, rhs: obligation.FreeVarId) bool {
    return lhs.file_id == rhs.file_id and lhs.pattern_id == rhs.pattern_id;
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

fn placeListContains(places: []const obligation.PlaceRef, needle: obligation.PlaceRef) bool {
    for (places) |place| {
        if (placeRefEql(place, needle)) return true;
    }
    return false;
}

fn storageWriteSlotsContain(places: []const obligation.PlaceRef, root: []const u8) bool {
    for (places) |place| {
        if (place.region == .storage and std.mem.eql(u8, place.root, root)) return true;
    }
    return false;
}

fn placeRefEql(lhs: obligation.PlaceRef, rhs: obligation.PlaceRef) bool {
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

fn placeKeyEql(lhs: obligation.PlaceKey, rhs: obligation.PlaceKey) bool {
    if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) return false;
    return switch (lhs) {
        .parameter => |value| value == rhs.parameter,
        .comptime_parameter => |value| value == rhs.comptime_parameter,
        .comptime_range_parameter => |value| value == rhs.comptime_range_parameter,
        .constant => |value| std.mem.eql(u8, value, rhs.constant),
        .msg_sender, .tx_origin, .unknown => true,
    };
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
