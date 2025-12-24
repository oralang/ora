// ============================================================================
// Statement Lowering Helpers
// ============================================================================
// Common helper functions used across statement lowering modules

const std = @import("std");
const c = @import("mlir_c_api").c;
const c_zig = @import("mlir_c_api");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const lib = @import("ora_lib");

fn reportRefinementGuardError(self: *const StatementLowerer, span: lib.ast.SourceSpan, message: []const u8, suggestion: ?[]const u8) void {
    if (self.expr_lowerer.error_handler) |handler| {
        handler.reportError(.MlirOperationFailed, span, message, suggestion) catch {};
    } else {
        std.log.warn("{s}", .{message});
    }
}

/// Create an ora.yield operation with optional values
pub fn createYield(self: *const StatementLowerer, values: []const c.MlirValue, loc: c.MlirLocation) void {
    const op = self.ora_dialect.createYield(values, loc);
    h.appendOp(self.block, op);
}

/// Create an empty ora.yield operation
pub fn createEmptyYield(self: *const StatementLowerer, loc: c.MlirLocation) void {
    createYield(self, &[_]c.MlirValue{}, loc);
}

/// Create a boolean constant (true or false)
pub fn createBoolConstant(self: *const StatementLowerer, value: bool, loc: c.MlirLocation) c.MlirValue {
    const i1_type = h.boolType(self.ctx);
    var const_state = h.opState("arith.constant", loc);
    const attr = c.mlirIntegerAttrGet(i1_type, if (value) 1 else 0);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&i1_type));
    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Store a value to a memref
pub fn storeToMemref(self: *const StatementLowerer, value: c.MlirValue, memref: c.MlirValue, loc: c.MlirLocation) void {
    var store_state = h.opState("memref.store", loc);
    c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ value, memref }));
    const store_op = c.mlirOperationCreate(&store_state);
    h.appendOp(self.block, store_op);
}

/// Convert a value to match a target type, using TypeMapper and bitcast as fallback
pub fn convertValueToType(
    self: *const StatementLowerer,
    value: c.MlirValue,
    target_type: c.MlirType,
    span: lib.ast.SourceSpan,
    loc: c.MlirLocation,
) c.MlirValue {
    const value_type = c.mlirValueGetType(value);

    // If types already match, return as-is
    if (c.mlirTypeEqual(value_type, target_type)) {
        return value;
    }

    // Try TypeMapper conversion first
    const converted = self.type_mapper.createConversionOp(self.block, value, target_type, span);
    const converted_type = c.mlirValueGetType(converted);

    // Check if conversion succeeded
    if (c.mlirTypeEqual(converted_type, target_type)) {
        return converted;
    }

    // Fallback: try bitcast for same-width integer types
    const value_builtin = c.oraTypeToBuiltin(value_type);
    const target_builtin = c.oraTypeToBuiltin(target_type);

    if (c.mlirTypeIsAInteger(value_builtin) and c.mlirTypeIsAInteger(target_builtin)) {
        const value_width = c.mlirIntegerTypeGetWidth(value_builtin);
        const target_width = c.mlirIntegerTypeGetWidth(target_builtin);

        if (value_width == target_width) {
            var bitcast_state = h.opState("arith.bitcast", loc);
            c.mlirOperationStateAddOperands(&bitcast_state, 1, @ptrCast(&value));
            c.mlirOperationStateAddResults(&bitcast_state, 1, @ptrCast(&target_type));
            const bitcast_op = c.mlirOperationCreate(&bitcast_state);
            h.appendOp(self.block, bitcast_op);
            return h.getResult(bitcast_op, 0);
        }
    }

    // Return converted value (may fail verification, but best effort)
    return converted;
}

/// Ensure a value is materialized (convert unrealized_conversion_cast if needed)
pub fn ensureValue(_: *const StatementLowerer, val: c.MlirValue, _: c.MlirLocation) c.MlirValue {
    // For now, just return the value as-is
    // TODO: Check for unrealized_conversion_cast if mlirValueGetDefiningOp becomes available
    return val;
}

/// Insert refinement guard for a value based on its type
/// Guard optimization is determined during type resolution and stored in AST nodes
/// skip_guard: if true, skip guard generation (set during type resolution based on optimizations)
pub fn insertRefinementGuard(
    self: *const StatementLowerer,
    value: c.MlirValue,
    ora_type: lib.ast.Types.OraType,
    span: lib.ast.SourceSpan,
    skip_guard: bool,
) StatementLowerer.LoweringError!c.MlirValue {
    if (c.mlirValueIsNull(value)) {
        reportRefinementGuardError(self, span, "Refinement guard creation failed: value is null", "Ensure the value is produced before guard insertion.");
        return value;
    }

    // Optimization: skip guard if flag is set (determined during type resolution)
    if (skip_guard) {
        return value;
    }

    const loc = self.fileLoc(span);

    switch (ora_type) {
        .min_value => |mv| {
            // Generate: require(value >= min)
            const value_type = c.mlirValueGetType(value);
            // Extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const min_type = if (base_type.ptr != null) base_type else value_type;
            // For u256 values, create attribute from string to support full precision
            const min_attr = if (mv.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{mv.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format min_value", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(min_type, min_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(min_type, @intCast(mv.min));
            };
            const min_const = createConstantValue(self, min_attr, min_type, loc);

            var cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5); // uge
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
            c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
            // Extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // Convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // Operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, min_const }));
            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const cmp_op = c.mlirOperationCreate(&cmp_state);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MinValue<u256, {d}>", .{mv.min});
            defer self.allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(self.block, assert_op);
        },
        .max_value => |mv| {
            // Generate: require(value <= max)
            const value_type = c.mlirValueGetType(value);
            // Extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const max_type = if (base_type.ptr != null) base_type else value_type;
            // For u256 values, create attribute from string to support full precision
            const max_attr = if (mv.max > std.math.maxInt(i64)) blk: {
                var max_buf: [100]u8 = undefined;
                const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{mv.max}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format max_value", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const max_str_ref = h.strRef(max_str);
                break :blk c.oraIntegerAttrGetFromString(max_type, max_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(max_type, @intCast(mv.max));
            };
            const max_const = createConstantValue(self, max_attr, max_type, loc);

            var cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3); // ule
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
            c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
            // Extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // Convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // Operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, max_const }));
            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const cmp_op = c.mlirOperationCreate(&cmp_state);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{mv.max});
            defer self.allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(self.block, assert_op);
        },
        .in_range => |ir| {
            // Generate: require(min <= value <= max)
            const value_type = c.mlirValueGetType(value);
            // Extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const op_type = if (base_type.ptr != null) base_type else value_type;
            // For u256 values, create attributes from strings to support full precision
            const min_attr = if (ir.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{ir.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format in_range min", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(op_type, min_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(op_type, @intCast(ir.min));
            };
            const max_attr = if (ir.max > std.math.maxInt(i64)) blk: {
                var max_buf: [100]u8 = undefined;
                const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{ir.max}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format in_range max", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const max_str_ref = h.strRef(max_str);
                break :blk c.oraIntegerAttrGetFromString(op_type, max_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(op_type, @intCast(ir.max));
            };
            const min_const = createConstantValue(self, min_attr, op_type, loc);
            const max_const = createConstantValue(self, max_attr, op_type, loc);

            // Check: value >= min
            var min_cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
            var min_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
            c.mlirOperationStateAddAttributes(&min_cmp_state, min_attrs.len, &min_attrs);
            // Extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // Convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // Operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            c.mlirOperationStateAddOperands(&min_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, min_const }));
            c.mlirOperationStateAddResults(&min_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const min_cmp_op = c.mlirOperationCreate(&min_cmp_state);
            h.appendOp(self.block, min_cmp_op);
            const min_check = h.getResult(min_cmp_op, 0);

            // Check: value <= max
            var max_cmp_state = h.opState("arith.cmpi", loc);
            const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
            var max_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
            c.mlirOperationStateAddAttributes(&max_cmp_state, max_attrs.len, &max_attrs);
            c.mlirOperationStateAddOperands(&max_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, max_const }));
            c.mlirOperationStateAddResults(&max_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const max_cmp_op = c.mlirOperationCreate(&max_cmp_state);
            h.appendOp(self.block, max_cmp_op);
            const max_check = h.getResult(max_cmp_op, 0);

            // Combine: min_check && max_check
            var and_state = h.opState("arith.andi", loc);
            c.mlirOperationStateAddOperands(&and_state, 2, @ptrCast(&[_]c.MlirValue{ min_check, max_check }));
            c.mlirOperationStateAddResults(&and_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const and_op = c.mlirOperationCreate(&and_state);
            h.appendOp(self.block, and_op);
            const condition = h.getResult(and_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ ir.min, ir.max });
            defer self.allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(self.block, assert_op);
        },
        .exact, .scaled => {
            // Exact and Scaled guards are inserted at specific operations (division, arithmetic)
            // No initialization guard needed
        },
        else => {
            // Not a refinement type, no guard needed
        },
    }

    return value;
}

/// Create a constant value from an attribute
pub fn createConstantValue(self: *const StatementLowerer, attr: c.MlirAttribute, ty: c.MlirType, loc: c.MlirLocation) c.MlirValue {
    var const_state = h.opState("arith.constant", loc);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&ty));
    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Check if a block ends with a proper terminator (ora.yield, scf.yield, ora.return, func.return, etc.)
pub fn blockEndsWithTerminator(_: *const StatementLowerer, block: c.MlirBlock) bool {
    // Get the last operation in the block
    const last_op = c.mlirBlockGetTerminator(block);
    if (c.mlirOperationIsNull(last_op)) {
        return false;
    }

    // Check if it's a proper terminator operation
    const op_name_ref = c.oraOperationGetName(last_op);
    const name_str = op_name_ref.data;
    const name_len = op_name_ref.length;

    // TODO: Check for common terminator operations - Hacky for now, but it works.
    const terminator_names = [_][]const u8{
        "ora.yield",
        "scf.yield",
        "ora.return",
        "func.return",
        "scf.condition",
    };

    if (name_str == null or name_len == 0) {
        c_zig.freeStringRef(op_name_ref);
        return false;
    }

    for (terminator_names) |term_name| {
        if (name_len == term_name.len) {
            if (std.mem.eql(u8, name_str[0..name_len], term_name)) {
                c_zig.freeStringRef(op_name_ref);
                return true;
            }
        }
    }

    c_zig.freeStringRef(op_name_ref);
    return false;
}

/// Check if a block ends with ora.break (which is NOT a terminator)
pub fn blockEndsWithBreak(_: *const StatementLowerer, block: c.MlirBlock) bool {
    // Get the last operation in the block (not necessarily a terminator)
    // Iterate through operations to find the last one
    var op = c.mlirBlockGetFirstOperation(block);
    var last_op: c.MlirOperation = c.MlirOperation{};

    while (!c.mlirOperationIsNull(op)) {
        last_op = op;
        op = c.mlirOperationGetNextInBlock(op);
    }

    if (c.mlirOperationIsNull(last_op)) {
        return false;
    }

    const op_name_ref = c.oraOperationGetName(last_op);
    const name_str = op_name_ref.data;
    const name_len = op_name_ref.length;

    std.debug.print("[blockEndsWithBreak] Last op name length: {}, checking for 'ora.break'\n", .{name_len});

    if (name_str == null or name_len == 0) {
        c_zig.freeStringRef(op_name_ref);
        return false;
    }

    if (name_len == 9) {
        const name_slice = name_str[0..name_len];
        std.debug.print("[blockEndsWithBreak] Last op name: {s}\n", .{name_slice});
        const result = std.mem.eql(u8, name_slice, "ora.break");
        // Free the string returned from oraOperationGetName
        c_zig.freeStringRef(op_name_ref);
        return result;
    }

    // Free the string returned from oraOperationGetName
    c_zig.freeStringRef(op_name_ref);
    return false;
}

/// Check if a block ends with ora.continue (which is NOT a terminator for if regions)
pub fn blockEndsWithContinue(_: *const StatementLowerer, block: c.MlirBlock) bool {
    // Get the last operation in the block (not necessarily a terminator)
    var op = c.mlirBlockGetFirstOperation(block);
    var last_op: c.MlirOperation = c.MlirOperation{};

    while (!c.mlirOperationIsNull(op)) {
        last_op = op;
        op = c.mlirOperationGetNextInBlock(op);
    }

    if (c.mlirOperationIsNull(last_op)) {
        return false;
    }

    const op_name_ref = c.oraOperationGetName(last_op);
    const name_str = op_name_ref.data;
    const name_len = op_name_ref.length;

    std.debug.print("[blockEndsWithContinue] Last op name length: {}, checking for 'ora.continue'\n", .{name_len});

    if (name_str == null or name_len == 0) {
        c_zig.freeStringRef(op_name_ref);
        return false;
    }

    if (name_len == 12) {
        const name_slice = name_str[0..name_len];
        std.debug.print("[blockEndsWithContinue] Last op name: {s}\n", .{name_slice});
        const result = std.mem.eql(u8, name_slice, "ora.continue");
        c_zig.freeStringRef(op_name_ref);
        return result;
    }

    c_zig.freeStringRef(op_name_ref);
    return false;
}

/// Check if a block ends with scf.yield
pub fn blockEndsWithYield(_: *const StatementLowerer, block: c.MlirBlock) bool {
    const last_op = c.mlirBlockGetTerminator(block);
    if (c.mlirOperationIsNull(last_op)) {
        return false;
    }

    const op_name_ref = c.oraOperationGetName(last_op);
    const name_str = op_name_ref.data;
    const name_len = op_name_ref.length;

    if (name_str == null or name_len == 0) {
        c_zig.freeStringRef(op_name_ref);
        return false;
    }

    const result = (name_len == 9 and std.mem.eql(u8, name_str[0..9], "scf.yield")) or
        (name_len == 9 and std.mem.eql(u8, name_str[0..9], "ora.yield"));

    // Free the string returned from oraOperationGetName
    c_zig.freeStringRef(op_name_ref);
    return result;
}

/// Check if an if statement contains return statements
pub fn ifStatementHasReturns(_: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) bool {
    // Check then branch
    for (if_stmt.then_branch.statements) |stmt| {
        if (stmt == .Return) return true;
        // Check nested if statements
        if (stmt == .If) {
            const nested_if = &stmt.If;
            for (nested_if.then_branch.statements) |nested_stmt| {
                if (nested_stmt == .Return) return true;
            }
            if (nested_if.else_branch) |nested_else| {
                for (nested_else.statements) |nested_stmt| {
                    if (nested_stmt == .Return) return true;
                }
            }
        }
    }

    // Check else branch if present
    if (if_stmt.else_branch) |else_branch| {
        for (else_branch.statements) |stmt| {
            if (stmt == .Return) return true;
            // Check nested if statements
            if (stmt == .If) {
                const nested_if = &stmt.If;
                for (nested_if.then_branch.statements) |nested_stmt| {
                    if (nested_stmt == .Return) return true;
                }
                if (nested_if.else_branch) |nested_else| {
                    for (nested_else.statements) |nested_stmt| {
                        if (nested_stmt == .Return) return true;
                    }
                }
            }
        }
    }

    return false;
}

/// Check if a block has a return statement (recursively checks nested if statements)
pub fn blockHasReturn(self: *const StatementLowerer, block: lib.ast.Statements.BlockNode) bool {
    for (block.statements) |stmt| {
        if (stmt == .Return) return true;
        // Check nested if statements recursively
        if (stmt == .If) {
            if (ifStatementHasReturns(self, &stmt.If)) return true;
        }
    }
    return false;
}

/// Create a default value for a given MLIR type
pub fn createDefaultValueForType(self: *const StatementLowerer, mlir_type: c.MlirType, loc: c.MlirLocation) StatementLowerer.LoweringError!c.MlirValue {
    // Check if it's an integer type
    if (c.mlirTypeIsAInteger(mlir_type)) {
        // Create a constant 0 value
        var const_state = h.opState("arith.constant", loc);
        const value_attr = c.mlirIntegerAttrGet(mlir_type, 0);
        const value_id = h.identifier(self.ctx, "value");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
        c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&mlir_type));
        const const_op = c.mlirOperationCreate(&const_state);
        h.appendOp(self.block, const_op);
        return h.getResult(const_op, 0);
    }

    // For other types, create a zero constant (simplified)
    var const_state = h.opState("arith.constant", loc);
    const constants = @import("../lower.zig");
    const zero_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const value_attr = c.mlirIntegerAttrGet(zero_type, 0);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&zero_type));
    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Get the return type from an if statement's return statements
pub fn getReturnTypeFromIfStatement(self: *const StatementLowerer, _: *const lib.ast.Statements.IfNode) ?c.MlirType {
    // Use the function's return type instead of trying to infer from individual returns
    return self.current_function_return_type;
}
