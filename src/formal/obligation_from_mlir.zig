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
    arithmetic_safety,
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
    .{ "arith.divui", .arithmetic_safety },
    .{ "arith.divsi", .arithmetic_safety },
    .{ "arith.remui", .arithmetic_safety },
    .{ "arith.remsi", .arithmetic_safety },
    .{ "arith.shli", .arithmetic_safety },
    .{ "arith.shrsi", .arithmetic_safety },
    .{ "arith.shrui", .arithmetic_safety },
});

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

    fn walkOperation(self: *Collector, op: mlir.MlirOperation, inherited_symbol: ?[]const u8) !void {
        if (mlir.oraOperationIsNull(op)) return;

        const ordinal = self.nextOrdinal();
        const op_name = operationName(op);
        const symbol = try self.symbolForOperation(op, op_name, inherited_symbol);
        if (op_kind_map.get(op_name)) |kind| {
            try self.collectOperation(op, kind, op_name, symbol, ordinal);
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
            .arithmetic_safety => try self.addArithmeticSafetyOp(op_name, symbol, ordinal),
        }
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

        if (try self.stringAttr(op, "ora.verification_type")) |verification_type| {
            if (std.mem.eql(u8, verification_type, "guard")) {
                try self.addGuardAssertOp(op, op_name, symbol, ordinal);
                handled_tag = true;
            }
        }

        if (!handled_tag) {
            try self.addLogicalOp(op, op_name, symbol, ordinal, default_role);
        }
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
            if (try self.stringAttr(op, "ora.assume_origin")) |origin| {
                if (std.mem.eql(u8, origin, "path")) break :blk .path_assume;
            }
            if (try self.stringAttr(op, "ora.verification_context")) |context| {
                if (std.mem.eql(u8, context, "path_assumption")) break :blk .path_assume;
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
        const amount_index = mlir.oraOperationGetNumOperands(op);
        if (amount_index == 0) {
            try self.addMissingFormulaDiagnostic(op_name, symbol, ordinal);
            return;
        }
        const amount = self.formulaOperand(op, op_name, symbol, ordinal, @intCast(amount_index - 1)).?;

        switch (resource_op) {
            .move => {
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .amount_non_negative);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .source_sufficient);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .destination_no_overflow);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .same_place_net_zero);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .conservation);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .modifies_covered);
            },
            .create => {
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .amount_non_negative);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .destination_no_overflow);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .modifies_covered);
            },
            .destroy => {
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .amount_non_negative);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .source_sufficient);
                try self.addResourceGoal(resource_op, domain, amount, symbol, ordinal, .modifies_covered);
            },
        }
    }

    fn addResourceGoal(
        self: *Collector,
        resource_op: obligation.ResourceOperation,
        domain: []const u8,
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
                .formula = derivedFormula(op_name, symbol, ordinal),
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
        try self.addBlockingDiagnostic(.missing_formula, try std.fmt.allocPrint(self.allocator, "{s} missing condition operand", .{op_name}));
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

    fn stringAttr(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const u8 {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef(name));
        if (mlir.oraAttributeIsNull(attr)) return null;
        const value = mlir.oraStringAttrGetValue(attr);
        if (value.data == null) return null;
        return try self.allocator.dupe(u8, value.data[0..value.length]);
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

fn derivedFormula(op_name: []const u8, symbol: ?[]const u8, ordinal: u32) obligation.FormulaRef {
    return .{ .origin_value = .{
        .origin = mlirOrigin(op_name, symbol, ordinal),
        .kind = .derived,
    } };
}
