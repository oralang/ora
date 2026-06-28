//! Ora MLIR to obligation manifest adapter.
//!
//! This is a boundary collector, not a verifier and not a proof encoder. It
//! walks canonical Ora MLIR, records the obligation surface already present
//! there, and leaves formulas owned by MLIR values. Z3 and Lean exporters must
//! consume the same manifest instead of rediscovering different obligations.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const obligation = @import("obligation.zig");

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
    obligations: std.ArrayList(obligation.Obligation) = .empty,
    assumptions: std.ArrayList(obligation.Assumption) = .empty,
    queries: std.ArrayList(obligation.VerificationQuery) = .empty,
    diagnostics: std.ArrayList(obligation.ObligationDiagnostic) = .empty,
    next_id: obligation.Id = 1,
    next_ordinal: u32 = 0,

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
        try self.collectEffectFrameAttrs(op, op_name, symbol, ordinal);
        if (op_kind_map.get(op_name)) |kind| {
            try self.collectOperation(op, kind, op_name, symbol, ordinal);
        } else if (arithmetic_op_map.get(op_name)) |safety| {
            try self.addArithmeticSafetyOp(op_name, symbol, ordinal, safety, null);
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

        if (write_slots.len != 0 or modifies_slots.len != 0) {
            try self.addEffectFrameGoal(.write_covered_by_modifies, modifies_slots, write_slots, op_name, symbol, ordinal);
        }
        if (read_slots.len != 0) {
            try self.addEffectFrameGoal(.read_preserved_by_frame, write_slots, read_slots, op_name, symbol, ordinal);
        }

        if (std.mem.eql(u8, op_name, "ora.lock")) {
            if (try self.placeAttr(op, "key")) |locked| {
                const declared = try self.allocator.alloc(obligation.PlaceRef, 1);
                declared[0] = locked;
                try self.addEffectFrameGoal(.lock_covers_write, declared, &.{}, op_name, symbol, ordinal);
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
            try self.addEffectFrameGoal(.external_call_frame, declared, &.{}, op_name, symbol, ordinal);
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
    ) !void {
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .effect_frame = .{
                .relation = relation,
                .declared = declared,
                .actual = actual,
            } },
        });
        try self.addQuery(.obligation, null, null, id, owner, origin);
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
            .source = .generated(),
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
            try self.addAssumptionOp(op, op_name, symbol, ordinal, .requires);
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
                const formula = self.formulaOperand(op, op_name, symbol, ordinal, 0) orelse {
                    try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
                    return;
                };
                try self.addArithmeticSafetyOp(op_name, symbol, ordinal, safety, formula);
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
        const formula = self.formulaOperand(op, op_name, symbol, ordinal, 0) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        if (try self.stringAttr(op, "ora.guard_id")) |guard_id| {
            const id = self.nextId();
            try self.obligations.append(self.allocator, .{
                .id = id,
                .owner = owner,
                .source = .generated(),
                .phase = .ora_mlir,
                .origin = origin,
                .kind = .{ .runtime_guard = .{
                    .guard_id = guard_id,
                    .formula = formula,
                    .erasure = .may_elide_if_proven,
                } },
            });
            try self.addQuery(.guard_satisfy, null, guard_id, id, owner, origin);
            try self.addQuery(.guard_violate, null, guard_id, id, owner, origin);
        } else {
            const id = self.nextId();
            try self.obligations.append(self.allocator, .{
                .id = id,
                .owner = owner,
                .source = .generated(),
                .phase = .ora_mlir,
                .origin = origin,
                .kind = .{ .logical = .{
                    .role = .guard,
                    .formula = formula,
                } },
            });
            try self.addQuery(.obligation, .guard, null, id, owner, origin);
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
        const formula = self.formulaOperand(op, op_name, symbol, ordinal, 0) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();

        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .logical = .{
                .role = role,
                .formula = formula,
            } },
        });
        try self.addQuery(.obligation, role, null, id, owner, origin);
    }

    fn addAssumptionOp(
        self: *Collector,
        op: mlir.MlirOperation,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        kind: obligation.AssumptionKind,
    ) !void {
        const formula = self.formulaOperand(op, op_name, symbol, ordinal, 0) orelse {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        };

        try self.assumptions.append(self.allocator, .{
            .id = self.nextId(),
            .owner = self.ownerFor(symbol),
            .source = .generated(),
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
        const formula = self.formulaOperand(op, op_name, symbol, ordinal, 0) orelse {
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

        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .runtime_guard = .{
                .guard_id = guard_id,
                .formula = formula,
                .erasure = .may_elide_if_proven,
            } },
        });
        try self.addQuery(.guard_satisfy, null, guard_id, id, owner, origin);
        try self.addQuery(.guard_violate, null, guard_id, id, owner, origin);
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
        const amount = self.formulaOperand(op, op_name, symbol, ordinal, @intCast(amount_index)).?;
        const places = try self.resourcePlaces(op, resource_op, operand_count, op_name) orelse return;

        for (resourceProperties(resource_op)) |property| {
            try self.addResourceGoal(resource_op, domain, places, amount, symbol, ordinal, property);
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
        if (mlir.oraValueIsNull(value)) return .{ .unknown = {} };
        if (mlir.mlirValueIsABlockArgument(value)) {
            return .{ .parameter = @intCast(mlir.mlirBlockArgumentGetArgNumber(value)) };
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
                if (isTransparentValueOp(owner_name) and mlir.oraOperationGetNumOperands(owner) >= 1) {
                    return try self.placeKeyFromValue(mlir.oraOperationGetOperand(owner, 0));
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
            .source = .generated(),
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
        try self.addQuery(.obligation, null, null, id, owner, origin);
    }

    fn addArithmeticSafetyOp(
        self: *Collector,
        op_name: []const u8,
        symbol: ?[]const u8,
        ordinal: u32,
        safety: obligation.ArithmeticSafetyKind,
        formula_override: ?obligation.FormulaRef,
    ) !void {
        const origin = mlirOrigin(op_name, symbol, ordinal);
        const owner = self.ownerFor(symbol);
        const id = self.nextId();
        try self.obligations.append(self.allocator, .{
            .id = id,
            .owner = owner,
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = origin,
            .kind = .{ .logical = .{
                .role = .arithmetic_safety,
                .formula = formula_override orelse derivedFormula(op_name, symbol, ordinal),
                .arithmetic_safety = safety,
            } },
        });
        try self.addQuery(.obligation, .arithmetic_safety, null, id, owner, origin);
    }

    fn addQuery(
        self: *Collector,
        kind: obligation.VerificationQueryKind,
        logical_role: ?obligation.LogicalRole,
        guard_id: ?[]const u8,
        obligation_id: obligation.Id,
        owner: obligation.Owner,
        origin: obligation.Origin,
    ) !void {
        const obligation_ids = try self.allocator.alloc(obligation.Id, 1);
        obligation_ids[0] = obligation_id;
        try self.queries.append(self.allocator, .{
            .id = self.nextId(),
            .owner = owner,
            .source = .generated(),
            .phase = .report,
            .origin = origin,
            .kind = kind,
            .logical_role = logical_role,
            .guard_id = guard_id,
            .obligation_ids = obligation_ids,
        });
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
    ) ?obligation.FormulaRef {
        _ = self;
        if (mlir.oraOperationGetNumOperands(op) <= operand_index) return null;
        return .{ .origin_value = .{
            .origin = mlirOrigin(op_name, symbol, ordinal),
            .kind = .operand,
            .index = operand_index,
        } };
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

    fn stringAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const u8 {
        var value = stringAttrView(op, name) orelse return null;
        return try self.allocator.dupe(u8, value.slice());
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

fn isTransparentValueOp(op_name: []const u8) bool {
    return transparent_value_ops.has(op_name);
}

fn derivedFormula(op_name: []const u8, symbol: ?[]const u8, ordinal: u32) obligation.FormulaRef {
    return .{ .origin_value = .{
        .origin = mlirOrigin(op_name, symbol, ordinal),
        .kind = .derived,
    } };
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
