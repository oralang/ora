const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_c_api").c;

const obligation = @import("formal/obligation.zig");
const obligation_from_mlir = @import("formal/obligation_from_mlir.zig");

const MlirContextHandle = struct {
    ctx: mlir.MlirContext,
};

fn createContext() MlirContextHandle {
    const ctx = mlir.oraContextCreate();
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    mlir.oraContextLoadSIRDialect(ctx);
    _ = mlir.oraDialectRegister(ctx);
    return .{ .ctx = ctx };
}

fn destroyContext(handle: MlirContextHandle) void {
    mlir.oraContextDestroy(handle.ctx);
}

fn parseModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.MlirParseFailed;
    return module;
}

fn findFirstOpByName(module: mlir.MlirModule, name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findFirstOpByNameInOp(mlir.oraModuleGetOperation(module), name);
}

fn findFirstOpByNameInOp(op: mlir.MlirOperation, expected: []const u8) ?mlir.MlirOperation {
    if (mlir.oraOperationIsNull(op)) return null;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], expected)) return op;

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (findFirstOpByNameInOp(child, expected)) |found| return found;
            }
        }
    }

    return null;
}

fn strRef(text: []const u8) mlir.MlirStringRef {
    return .{ .data = text.ptr, .length = text.len };
}

fn countLogical(set: obligation.ObligationSet, role: obligation.LogicalRole) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == role) count += 1;
    }
    return count;
}

fn countRuntimeGuards(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .runtime_guard) count += 1;
    }
    return count;
}

fn countAssumption(set: obligation.ObligationSet, kind: obligation.AssumptionKind) usize {
    var count: usize = 0;
    for (set.assumptions) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

fn countResource(set: obligation.ObligationSet, op: obligation.ResourceOperation, property: obligation.ResourceProperty) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .resource and item.kind.resource.op == op and item.kind.resource.property == property) count += 1;
    }
    return count;
}

fn countQuery(set: obligation.ObligationSet, kind: obligation.VerificationQueryKind) usize {
    var count: usize = 0;
    for (set.queries) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

test "formal obligation MLIR adapter collects verification markers and assumptions" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 3), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 2), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 4), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .assume));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));
    try testing.expect(result.set.assumptions[0].owner == .function);
    try testing.expectEqualStrings("checked", result.set.assumptions[0].owner.function.name);

    const guard = result.set.obligations[2];
    try testing.expect(guard.kind == .runtime_guard);
    try testing.expectEqualStrings("guard:checked:flag", guard.kind.runtime_guard.guard_id);
    try testing.expectEqual(obligation.ValueRefKind.operand, guard.kind.runtime_guard.formula.origin_value.kind);
    try testing.expectEqual(@as(u32, 0), guard.kind.runtime_guard.formula.origin_value.index);
}

test "formal obligation MLIR adapter covers Z3 assertion tags and implicit safety ops" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @z3_surface(%flag: i1, %x: i256, %y: i256) {
        \\    "cf.assert"(%flag) <{msg = "requires"}> : (i1) -> ()
        \\    "ora.assert"(%flag) : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    %q = arith.divui %x, %y : i256
        \\    %s = arith.shli %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const cf_assert = findFirstOpByName(module, "cf.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        cf_assert,
        strRef("ora.requires"),
        mlir.oraBoolAttrCreate(h.ctx, true),
    );

    const ora_assert = findFirstOpByName(module, "ora.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.verification_type"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard")),
    );
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:z3_surface:flag")),
    );

    const assume_op = findFirstOpByName(module, "ora.assume") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        assume_op,
        strRef("ora.assume_origin"),
        mlir.oraStringAttrCreate(h.ctx, strRef("path")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 3), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 2), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 4), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .path_assume));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .arithmetic_safety));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    for (result.set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == .arithmetic_safety) {
            try testing.expectEqual(obligation.ValueRefKind.derived, item.kind.logical.formula.origin_value.kind);
        }
    }
}

test "formal obligation MLIR adapter expands resource op properties" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @transfer_resource(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 6), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 6), result.set.queries.len);
    try testing.expectEqual(@as(usize, 6), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .amount_non_negative));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .source_sufficient));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .same_place_net_zero));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .modifies_covered));

    for (result.set.obligations) |item| {
        try testing.expect(item.kind == .resource);
        try testing.expectEqualStrings("TokenUnit", item.kind.resource.domain);
        try testing.expectEqual(obligation.ValueRefKind.operand, item.kind.resource.amount.?.origin_value.kind);
        try testing.expectEqual(@as(u32, 4), item.kind.resource.amount.?.origin_value.index);
    }
}

test "formal obligation MLIR adapter fails closed on malformed guard marker" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 0), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
}
