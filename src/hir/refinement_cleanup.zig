const std = @import("std");
const mlir = @import("mlir_c_api").c;

fn strRef(value: []const u8) mlir.MlirStringRef {
    return mlir.oraStringRefCreate(value.ptr, value.len);
}

fn insertOpBefore(block: mlir.MlirBlock, op: mlir.MlirOperation, before: mlir.MlirOperation) void {
    mlir.oraBlockInsertOwnedOperationBefore(block, op, before);
}

pub fn cleanupRefinementGuards(
    ctx: mlir.MlirContext,
    module: mlir.MlirModule,
    proven_guard_ids: *const std.StringHashMap(void),
) void {
    const module_op = mlir.oraModuleGetOperation(module);
    const debug_env = std.process.getEnvVarOwned(std.heap.page_allocator, "ORA_GUARD_DEBUG") catch null;
    const debug_enabled = debug_env != null;
    if (debug_env) |value| std.heap.page_allocator.free(value);
    walkOperation(ctx, module_op, proven_guard_ids, debug_enabled);
}

fn walkOperation(
    ctx: mlir.MlirContext,
    op: mlir.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        var block = mlir.oraRegionGetFirstBlock(region);
        while (block.ptr != null) {
            walkBlock(ctx, block, proven_guard_ids, debug_enabled);
            block = mlir.oraBlockGetNextInRegion(block);
        }
    }
}

fn walkBlock(
    ctx: mlir.MlirContext,
    block: mlir.MlirBlock,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    var current = mlir.oraBlockGetFirstOperation(block);
    while (current.ptr != null) {
        const next = mlir.oraOperationGetNextInBlock(current);
        if (isRefinementGuard(current)) {
            handleRefinementGuard(ctx, block, current, proven_guard_ids, debug_enabled);
        } else if (isVerificationOp(current)) {
            handleVerificationOp(ctx, block, current, debug_enabled);
        } else {
            walkOperation(ctx, current, proven_guard_ids, debug_enabled);
        }
        current = next;
    }
}

fn isRefinementGuard(op: mlir.MlirOperation) bool {
    const name = mlir.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return false;
    return std.mem.eql(u8, name.data[0..name.length], "ora.refinement_guard");
}

fn isVerificationOp(op: mlir.MlirOperation) bool {
    const name = mlir.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return false;
    const op_name = name.data[0..name.length];
    return std.mem.eql(u8, op_name, "ora.assume") or
        std.mem.eql(u8, op_name, "ora.assert") or
        std.mem.eql(u8, op_name, "ora.invariant") or
        std.mem.eql(u8, op_name, "ora.requires") or
        std.mem.eql(u8, op_name, "ora.ensures") or
        std.mem.eql(u8, op_name, "ora.havoc") or
        std.mem.eql(u8, op_name, "ora.decreases") or
        std.mem.eql(u8, op_name, "ora.increases");
}

fn handleRefinementGuard(
    ctx: mlir.MlirContext,
    block: mlir.MlirBlock,
    op: mlir.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    const guard_id_attr = mlir.oraOperationGetAttributeByName(op, strRef("ora.guard_id"));
    const guard_id = getStringAttr(guard_id_attr);

    if (guard_id != null and proven_guard_ids.contains(guard_id.?)) {
        if (debug_enabled) std.debug.print("[guard-cleanup] removed {s}\n", .{guard_id.?});
        mlir.oraOperationErase(op);
        return;
    }

    const condition = mlir.oraOperationGetOperand(op, 0);
    const message_attr = mlir.oraOperationGetAttributeByName(op, strRef("message"));
    const message = getStringAttr(message_attr) orelse "Refinement guard failed";
    const loc = mlir.oraOperationGetLocation(op);

    const assert_op = mlir.oraCfAssertOpCreate(ctx, loc, condition, strRef(message));
    if (assert_op.ptr != null) insertOpBefore(block, assert_op, op);

    if (debug_enabled) {
        if (guard_id) |id| {
            std.debug.print("[guard-cleanup] kept {s}\n", .{id});
        } else {
            std.debug.print("[guard-cleanup] kept <unknown>\n", .{});
        }
    }

    mlir.oraOperationErase(op);
}

fn handleVerificationOp(
    ctx: mlir.MlirContext,
    block: mlir.MlirBlock,
    op: mlir.MlirOperation,
    debug_enabled: bool,
) void {
    const name = mlir.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return;
    const op_name = name.data[0..name.length];

    if (std.mem.eql(u8, op_name, "ora.assert")) {
        const context_attr = mlir.oraOperationGetAttributeByName(op, strRef("ora.verification_context"));
        const context = getStringAttr(context_attr);
        const is_ghost = context != null and std.mem.eql(u8, context.?, "ghost_assertion");
        if (!is_ghost) {
            const condition = mlir.oraOperationGetOperand(op, 0);
            const message_attr = mlir.oraOperationGetAttributeByName(op, strRef("message"));
            const message = getStringAttr(message_attr) orelse "Assertion failed";
            const loc = mlir.oraOperationGetLocation(op);
            const assert_op = mlir.oraCfAssertOpCreate(ctx, loc, condition, strRef(message));
            if (assert_op.ptr != null) insertOpBefore(block, assert_op, op);
        }
        if (debug_enabled) std.debug.print("[verification-cleanup] removed assert ({s})\n", .{context orelse "unknown"});
        mlir.oraOperationErase(op);
        return;
    }

    if (debug_enabled) std.debug.print("[verification-cleanup] removed {s}\n", .{op_name});
    mlir.oraOperationErase(op);
}

fn getStringAttr(attr: mlir.MlirAttribute) ?[]const u8 {
    if (mlir.oraAttributeIsNull(attr)) return null;
    const value = mlir.oraStringAttrGetValue(attr);
    if (value.data == null or value.length == 0) return null;
    return value.data[0..value.length];
}
