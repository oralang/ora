// ============================================================================
// Refinement Guard Cleanup
// ============================================================================
//
// Removes proven refinement guards and lowers remaining guards to cf.assert.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const h = @import("helpers.zig");

pub fn cleanupRefinementGuards(
    ctx: c.MlirContext,
    module: c.MlirModule,
    proven_guard_ids: *const std.StringHashMap(void),
) void {
    const module_op = c.oraModuleGetOperation(module);
    const debug_env = std.process.getEnvVarOwned(std.heap.page_allocator, "ORA_GUARD_DEBUG") catch null;
    const debug_enabled = debug_env != null;
    if (debug_env) |val| std.heap.page_allocator.free(val);
    walkOperation(ctx, module_op, proven_guard_ids, debug_enabled);
}

fn walkOperation(
    ctx: c.MlirContext,
    op: c.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    const num_regions = c.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = c.oraOperationGetRegion(op, region_index);
        var block = c.oraRegionGetFirstBlock(region);
        while (block.ptr != null) {
            walkBlock(ctx, block, proven_guard_ids, debug_enabled);
            block = c.oraBlockGetNextInRegion(block);
        }
    }
}

fn walkBlock(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    var current = c.oraBlockGetFirstOperation(block);
    while (current.ptr != null) {
        const next = c.oraOperationGetNextInBlock(current);
        if (isRefinementGuard(current)) {
            handleRefinementGuard(ctx, block, current, proven_guard_ids, debug_enabled);
        } else {
            walkOperation(ctx, current, proven_guard_ids, debug_enabled);
        }
        current = next;
    }
}

fn isRefinementGuard(op: c.MlirOperation) bool {
    const name = c.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return false;
    return std.mem.eql(u8, name.data[0..name.length], "ora.refinement_guard");
}

fn handleRefinementGuard(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    op: c.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
) void {
    const guard_id_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.guard_id"));
    const guard_id = getStringAttr(guard_id_attr);

    if (guard_id != null and proven_guard_ids.contains(guard_id.?)) {
        if (debug_enabled) {
            std.debug.print("[guard-cleanup] removed {s}\n", .{guard_id.?});
        }
        c.oraOperationErase(op);
        return;
    }

    const condition = c.oraOperationGetOperand(op, 0);
    const message_attr = c.oraOperationGetAttributeByName(op, h.strRef("message"));
    const message = getStringAttr(message_attr) orelse "Refinement guard failed";
    const loc = c.oraOperationGetLocation(op);

    const assert_op = c.oraCfAssertOpCreate(ctx, loc, condition, h.strRef(message));
    if (assert_op.ptr != null) {
        h.insertOpBefore(block, assert_op, op);
    }
    if (debug_enabled) {
        if (guard_id) |id| {
            std.debug.print("[guard-cleanup] kept {s}\n", .{id});
        } else {
            std.debug.print("[guard-cleanup] kept <unknown>\n", .{});
        }
    }

    c.oraOperationErase(op);
}

fn getStringAttr(attr: c.MlirAttribute) ?[]const u8 {
    if (c.oraAttributeIsNull(attr)) return null;
    const value = c.oraStringAttrGetValue(attr);
    if (value.data == null or value.length == 0) return null;
    return value.data[0..value.length];
}
