const std = @import("std");
const mlir = @import("mlir_c_api").c;

pub const executable_fallback_attr_name = "ora.executable_fallback";
pub const protocol_zero_attr_name = "ora.protocol_zero";

pub const ExecutableFallbackViolation = struct {
    op_name: []const u8,
    reason: []const u8,
};

pub fn findExecutableFallback(module: mlir.MlirModule) ?ExecutableFallbackViolation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findInOperation(mlir.oraModuleGetOperation(module));
}

pub fn verifyNoExecutableFallbacks(module: mlir.MlirModule) !void {
    if (findExecutableFallback(module) != null) return error.ExecutableFallbackInHIR;
}

pub fn isAllowedProtocolZeroPurpose(purpose: []const u8) bool {
    return std.mem.eql(u8, purpose, "enum_auto_ordinal") or
        std.mem.eql(u8, purpose, "abi_padding") or
        std.mem.eql(u8, purpose, "empty_revert_data") or
        std.mem.eql(u8, purpose, "storage_zero_init");
}

fn findInOperation(op: mlir.MlirOperation) ?ExecutableFallbackViolation {
    if (mlir.oraOperationIsNull(op)) return null;

    const op_name = operationName(op);
    if (operationFallbackReason(op, op_name)) |reason| {
        return .{ .op_name = op_name, .reason = reason };
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
                if (findInOperation(child)) |violation| return violation;
            }
        }
    }

    return null;
}

fn operationFallbackReason(op: mlir.MlirOperation, op_name: []const u8) ?[]const u8 {
    if (hasAttribute(op, executable_fallback_attr_name)) return "executable fallback marker";
    if (invalidProtocolZeroReason(op)) |reason| return reason;
    if (std.mem.endsWith(u8, op_name, "_placeholder")) return "placeholder operation";
    if (std.mem.eql(u8, op_name, "ora.default_value")) return "default value operation";
    if (std.mem.eql(u8, op_name, "ora.lowering_error")) return "lowering-error placeholder";
    if (std.mem.eql(u8, op_name, "ora.missing_return")) return "missing-return placeholder";
    if (std.mem.eql(u8, op_name, "ora.uninitialized_local")) return "uninitialized-local placeholder";
    if (hasAttribute(op, "ora.unsupported")) return "unsupported lowering marker";
    return null;
}

fn invalidProtocolZeroReason(op: mlir.MlirOperation) ?[]const u8 {
    const attr = mlir.oraOperationGetAttributeByName(op, strRef(protocol_zero_attr_name));
    if (mlir.oraAttributeIsNull(attr)) return null;
    const value = mlir.oraStringAttrGetValue(attr);
    if (value.data == null) return "malformed protocol-zero marker";
    const purpose = value.data[0..value.length];
    if (!isAllowedProtocolZeroPurpose(purpose)) return "unknown protocol-zero purpose";
    return null;
}

fn hasAttribute(op: mlir.MlirOperation, name: []const u8) bool {
    return !mlir.oraAttributeIsNull(mlir.oraOperationGetAttributeByName(op, strRef(name)));
}

fn operationName(op: mlir.MlirOperation) []const u8 {
    const name = mlir.oraOperationGetName(op);
    if (name.data == null) return "<unknown>";
    return name.data[0..name.length];
}

fn strRef(text: []const u8) mlir.MlirStringRef {
    return .{ .data = text.ptr, .length = text.len };
}
