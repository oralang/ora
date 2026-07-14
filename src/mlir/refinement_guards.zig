// ============================================================================
// Refinement Guard Cleanup
// ============================================================================
//
// Removes proven refinement guards and lowers remaining guards to cf.assert.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const h = @import("mlir_helpers");

pub fn cleanupRefinementGuards(
    ctx: c.MlirContext,
    module: c.MlirModule,
    proven_guard_ids: *const std.StringHashMap(void),
) void {
    cleanupRefinementGuardsWithOptions(ctx, module, proven_guard_ids, .{});
}

pub const CleanupOptions = struct {
    keep_proved_checks: bool = false,
};

pub fn cleanupRefinementGuardsWithOptions(
    ctx: c.MlirContext,
    module: c.MlirModule,
    proven_guard_ids: *const std.StringHashMap(void),
    options: CleanupOptions,
) void {
    const module_op = c.oraModuleGetOperation(module);
    const debug_enabled = if (@import("builtin").link_libc) std.c.getenv("ORA_GUARD_DEBUG") != null else false;
    const allocator = std.heap.page_allocator;
    var seen_guard_ids = std.StringHashMap(void).init(allocator);
    defer deinitStringSet(allocator, &seen_guard_ids);
    var duplicate_guard_ids = std.StringHashMap(void).init(allocator);
    defer deinitStringSet(allocator, &duplicate_guard_ids);

    const duplicate_guard_ids_available = collectDuplicateRefinementGuardIds(
        allocator,
        module_op,
        &seen_guard_ids,
        &duplicate_guard_ids,
    ) catch false;
    const duplicate_guard_ids_ptr: ?*const std.StringHashMap(void) = if (duplicate_guard_ids_available) &duplicate_guard_ids else null;

    walkOperation(ctx, module_op, proven_guard_ids, duplicate_guard_ids_ptr, debug_enabled, options);
}

pub fn guardIdIsDuplicated(module: c.MlirModule, guard_id: []const u8) bool {
    if (c.oraModuleIsNull(module)) return true;
    var count: usize = 0;
    countRefinementGuardIdInOperation(c.oraModuleGetOperation(module), guard_id, &count);
    return count > 1;
}

fn walkOperation(
    ctx: c.MlirContext,
    op: c.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    duplicate_guard_ids: ?*const std.StringHashMap(void),
    debug_enabled: bool,
    options: CleanupOptions,
) void {
    const num_regions = c.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = c.oraOperationGetRegion(op, region_index);
        var block = c.oraRegionGetFirstBlock(region);
        while (block.ptr != null) {
            walkBlock(ctx, block, proven_guard_ids, duplicate_guard_ids, debug_enabled, options);
            block = c.oraBlockGetNextInRegion(block);
        }
    }
}

fn walkBlock(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    proven_guard_ids: *const std.StringHashMap(void),
    duplicate_guard_ids: ?*const std.StringHashMap(void),
    debug_enabled: bool,
    options: CleanupOptions,
) void {
    var current = c.oraBlockGetFirstOperation(block);
    while (current.ptr != null) {
        const next = c.oraOperationGetNextInBlock(current);
        if (isRefinementGuard(current)) {
            _ = handleRefinementGuard(ctx, block, current, proven_guard_ids, duplicate_guard_ids, debug_enabled, options);
        } else if (isVerificationOp(current)) {
            handleVerificationOp(ctx, block, current, proven_guard_ids, debug_enabled, options);
        } else {
            walkOperation(ctx, current, proven_guard_ids, duplicate_guard_ids, debug_enabled, options);
        }
        current = next;
    }
}

fn isRefinementGuard(op: c.MlirOperation) bool {
    const name = c.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return false;
    return std.mem.eql(u8, name.data[0..name.length], "ora.refinement_guard");
}

fn isVerificationOp(op: c.MlirOperation) bool {
    const name = c.oraOperationGetName(op);
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
    ctx: c.MlirContext,
    block: c.MlirBlock,
    op: c.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    duplicate_guard_ids: ?*const std.StringHashMap(void),
    debug_enabled: bool,
    options: CleanupOptions,
) bool {
    const guard_id_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.guard_id"));
    const guard_id = getStringAttr(guard_id_attr);

    if (!options.keep_proved_checks and
        guard_id != null and
        proven_guard_ids.contains(guard_id.?) and
        !guardIdExcludedByDuplicateSnapshot(duplicate_guard_ids, guard_id.?))
    {
        if (debug_enabled) {
            std.debug.print("[guard-cleanup] removed {s}\n", .{guard_id.?});
        }
        c.oraOperationErase(op);
        return false;
    }

    const condition = c.oraOperationGetOperand(op, 0);
    const message_attr = c.oraOperationGetAttributeByName(op, h.strRef("message"));
    const message = getStringAttr(message_attr) orelse "Refinement guard failed";
    const selector_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.assert_selector"));
    const selector = getStringAttr(selector_attr);
    const loc = c.oraOperationGetLocation(op);

    const assert_op = c.oraCfAssertOpCreate(ctx, loc, condition, h.strRef(message));
    if (assert_op.ptr != null) {
        c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.verification_type"), h.stringAttr(ctx, "guard"));
        c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.verification_context"), h.stringAttr(ctx, "refinement_guard"));
        if (selector) |selector_text| {
            c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.assert_selector"), h.stringAttr(ctx, selector_text));
        }
        if (guard_id) |id| {
            c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.guard_id"), h.stringAttr(ctx, id));
        }
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
    return true;
}

fn deinitStringSet(allocator: std.mem.Allocator, set: *std.StringHashMap(void)) void {
    var it = set.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    set.deinit();
}

fn collectDuplicateRefinementGuardIds(
    allocator: std.mem.Allocator,
    op: c.MlirOperation,
    seen_guard_ids: *std.StringHashMap(void),
    duplicate_guard_ids: *std.StringHashMap(void),
) !bool {
    if (op.ptr == null) return true;

    if (isRefinementGuard(op)) {
        const guard_id_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.guard_id"));
        if (getStringAttr(guard_id_attr)) |guard_id| {
            if (seen_guard_ids.contains(guard_id)) {
                if (!duplicate_guard_ids.contains(guard_id)) {
                    try duplicate_guard_ids.put(try allocator.dupe(u8, guard_id), {});
                }
            } else {
                try seen_guard_ids.put(try allocator.dupe(u8, guard_id), {});
            }
        }
    }

    const num_regions = c.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = c.oraOperationGetRegion(op, region_index);
        var block = c.oraRegionGetFirstBlock(region);
        while (block.ptr != null) : (block = c.oraBlockGetNextInRegion(block)) {
            var child = c.oraBlockGetFirstOperation(block);
            while (child.ptr != null) : (child = c.oraOperationGetNextInBlock(child)) {
                _ = try collectDuplicateRefinementGuardIds(allocator, child, seen_guard_ids, duplicate_guard_ids);
            }
        }
    }

    return true;
}

fn guardIdExcludedByDuplicateSnapshot(
    duplicate_guard_ids: ?*const std.StringHashMap(void),
    guard_id: []const u8,
) bool {
    const ids = duplicate_guard_ids orelse return true;
    return ids.contains(guard_id);
}

fn countRefinementGuardIdInOperation(op: c.MlirOperation, guard_id: []const u8, count: *usize) void {
    if (op.ptr == null or count.* > 1) return;

    if (isRefinementGuard(op)) {
        const guard_id_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.guard_id"));
        if (getStringAttr(guard_id_attr)) |current_id| {
            if (std.mem.eql(u8, current_id, guard_id)) {
                count.* += 1;
                if (count.* > 1) return;
            }
        }
    }

    const num_regions = c.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions and count.* <= 1) : (region_index += 1) {
        const region = c.oraOperationGetRegion(op, region_index);
        var block = c.oraRegionGetFirstBlock(region);
        while (block.ptr != null and count.* <= 1) : (block = c.oraBlockGetNextInRegion(block)) {
            var child = c.oraBlockGetFirstOperation(block);
            while (child.ptr != null and count.* <= 1) : (child = c.oraOperationGetNextInBlock(child)) {
                countRefinementGuardIdInOperation(child, guard_id, count);
            }
        }
    }
}

fn verificationTypeFromOpName(op_name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, op_name, "ora.requires")) return "requires";
    if (std.mem.eql(u8, op_name, "ora.ensures")) return "ensures";
    if (std.mem.eql(u8, op_name, "ora.invariant")) return "invariant";
    return null;
}

fn handleVerificationOp(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    op: c.MlirOperation,
    proven_guard_ids: *const std.StringHashMap(void),
    debug_enabled: bool,
    options: CleanupOptions,
) void {
    const name = c.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return;
    const op_name = name.data[0..name.length];

    if (std.mem.eql(u8, op_name, "ora.assert") or
        (options.keep_proved_checks and
            (std.mem.eql(u8, op_name, "ora.requires") or
                std.mem.eql(u8, op_name, "ora.ensures") or
                std.mem.eql(u8, op_name, "ora.invariant"))))
    {
        const context_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.verification_context"));
        const context = getStringAttr(context_attr);
        const verification_type_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.verification_type"));
        const verification_type = getStringAttr(verification_type_attr);
        const effective_verification_type = verification_type orelse verificationTypeFromOpName(op_name);
        _ = proven_guard_ids;
        const is_ghost = context != null and std.mem.eql(u8, context.?, "ghost_assertion");
        if (!is_ghost) {
            const condition = c.oraOperationGetOperand(op, 0);
            const message_attr = c.oraOperationGetAttributeByName(op, h.strRef("message"));
            const message = getStringAttr(message_attr) orelse "Assertion failed";
            const selector_attr = c.oraOperationGetAttributeByName(op, h.strRef("ora.assert_selector"));
            const selector = getStringAttr(selector_attr);
            const loc = c.oraOperationGetLocation(op);
            const assert_op = c.oraCfAssertOpCreate(ctx, loc, condition, h.strRef(message));
            if (assert_op.ptr != null) {
                if (context) |ctx_str| {
                    c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.verification_context"), h.stringAttr(ctx, ctx_str));
                } else if (effective_verification_type) |type_str| {
                    c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.verification_context"), h.stringAttr(ctx, type_str));
                }
                if (effective_verification_type) |type_str| {
                    c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.verification_type"), h.stringAttr(ctx, type_str));
                }
                if (selector) |selector_text| {
                    c.oraOperationSetAttributeByName(assert_op, h.strRef("ora.assert_selector"), h.stringAttr(ctx, selector_text));
                }
                h.insertOpBefore(block, assert_op, op);
            }
        }
        if (debug_enabled) {
            std.debug.print("[verification-cleanup] removed assert ({s})\n", .{context orelse "unknown"});
        }
        c.oraOperationErase(op);
        return;
    }

    if (debug_enabled) {
        std.debug.print("[verification-cleanup] removed {s}\n", .{op_name});
    }
    c.oraOperationErase(op);
}

fn getStringAttr(attr: c.MlirAttribute) ?[]const u8 {
    if (c.oraAttributeIsNull(attr)) return null;
    const value = c.oraStringAttrGetValue(attr);
    if (value.data == null or value.length == 0) return null;
    return value.data[0..value.length];
}
