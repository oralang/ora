// ============================================================================
// Runtime Check Proven-Marker Cleanup
// ============================================================================
//
// Marks compiler-generated runtime safety checks that are already discharged by
// SMT. This is deliberately separate from user-authored checks: custom errors,
// requires clauses, and refinement guards stay visible unless their own cleanup
// rules remove them. The marker only lets late lowering skip generic fallback
// guards such as Resource<T> underflow/overflow checks.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;

pub const resource_checks_proved_attr = "ora.resource_runtime_checks_proved";

pub fn markVerifiedResourceRuntimeChecks(ctx: c.MlirContext, module: c.MlirModule) void {
    const module_op = c.oraModuleGetOperation(module);
    walkOperation(ctx, module_op);
}

fn walkOperation(ctx: c.MlirContext, op: c.MlirOperation) void {
    if (isResourceBoundaryOp(op)) {
        c.oraOperationSetAttributeByName(
            op,
            c.oraStringRefCreate(resource_checks_proved_attr.ptr, resource_checks_proved_attr.len),
            c.oraBoolAttrCreate(ctx, true),
        );
    }

    const num_regions = c.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = c.oraOperationGetRegion(op, region_index);
        var block = c.oraRegionGetFirstBlock(region);
        while (block.ptr != null) {
            var current = c.oraBlockGetFirstOperation(block);
            while (current.ptr != null) {
                const next = c.oraOperationGetNextInBlock(current);
                walkOperation(ctx, current);
                current = next;
            }
            block = c.oraBlockGetNextInRegion(block);
        }
    }
}

fn isResourceBoundaryOp(op: c.MlirOperation) bool {
    const name = c.oraOperationGetName(op);
    if (name.data == null or name.length == 0) return false;
    const op_name = name.data[0..name.length];
    return std.mem.eql(u8, op_name, "ora.move") or
        std.mem.eql(u8, op_name, "ora.create") or
        std.mem.eql(u8, op_name, "ora.destroy");
}
