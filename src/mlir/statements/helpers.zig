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
const log = @import("log");

fn isErrorUnionTypeInfo(ti: lib.ast.Types.TypeInfo) bool {
    if (ti.category == .ErrorUnion) return true;
    if (ti.ora_type) |ot| switch (ot) {
        .error_union => return true,
        ._union => |members| return members.len > 0 and members[0] == .error_union,
        else => {},
    };
    return false;
}

/// Lower a value expression, inserting an implicit try inside try blocks when appropriate.
pub fn lowerValueWithImplicitTry(
    self: *const StatementLowerer,
    expr: *const lib.ast.Expressions.ExprNode,
    expected_type: ?lib.ast.Types.TypeInfo,
) c.MlirValue {
    if (self.in_try_block) {
        if (expr.* == .Call) {
            const call = expr.Call;
            if (isErrorUnionTypeInfo(call.type_info)) {
                if (expected_type == null or !isErrorUnionTypeInfo(expected_type.?)) {
                    var try_expr = lib.ast.Expressions.TryExpr{ .expr = @constCast(expr), .span = call.span };
                    return self.expr_lowerer.lowerTry(&try_expr);
                }
            }
        }
    }
    return self.expr_lowerer.lowerExpression(expr);
}

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
    const op = self.ora_dialect.createArithConstantBool(value, loc);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Store a value to a memref
pub fn storeToMemref(self: *const StatementLowerer, value: c.MlirValue, memref: c.MlirValue, loc: c.MlirLocation) void {
    const store_op = c.oraMemrefStoreOpCreate(
        self.ctx,
        loc,
        value,
        memref,
        null,
        0,
    );
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
    const value_type = c.oraValueGetType(value);

    // if types already match, return as-is
    if (c.oraTypeEqual(value_type, target_type)) {
        return value;
    }

    // try TypeMapper conversion first
    const converted = self.type_mapper.createConversionOp(self.block, value, target_type, span);
    const converted_type = c.oraValueGetType(converted);

    // check if conversion succeeded
    if (c.oraTypeEqual(converted_type, target_type)) {
        return converted;
    }

    // fallback: try bitcast for same-width integer types
    const value_builtin = c.oraTypeToBuiltin(value_type);
    const target_builtin = c.oraTypeToBuiltin(target_type);

    if (c.oraTypeIsAInteger(value_builtin) and c.oraTypeIsAInteger(target_builtin)) {
        const value_width = c.oraIntegerTypeGetWidth(value_builtin);
        const target_width = c.oraIntegerTypeGetWidth(target_builtin);

        if (value_width == target_width) {
            const bitcast_op = c.oraArithBitcastOpCreate(self.ctx, loc, value, target_type);
            h.appendOp(self.block, bitcast_op);
            return h.getResult(bitcast_op, 0);
        }
    }

    // return converted value (may fail verification, but best effort)
    return converted;
}

/// Ensure a value is materialized (convert unrealized_conversion_cast if needed)
pub fn ensureValue(_: *const StatementLowerer, val: c.MlirValue, _: c.MlirLocation) c.MlirValue {
    // for now, just return the value as-is
    // todo: Check for unrealized_conversion_cast if mlirValueGetDefiningOp becomes available
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
    if (c.oraValueIsNull(value)) {
        reportRefinementGuardError(self, span, "Refinement guard creation failed: value is null", "Ensure the value is produced before guard insertion.");
        return value;
    }

    // optimization: skip guard if flag is set (determined during type resolution)
    if (skip_guard) {
        return value;
    }

    const loc = self.fileLoc(span);

    switch (ora_type) {
        .min_value => |mv| {
            // generate: require(value >= min)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const min_type = if (base_type.ptr != null) base_type else value_type;
            // for u256 values, create attribute from string to support full precision
            const min_attr = if (mv.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{mv.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format min_value", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(min_type, min_str_ref);
            } else blk: {
                break :blk c.oraIntegerAttrCreateI64FromType(min_type, @intCast(mv.min));
            };
            const min_const = createConstantValue(self, min_attr, min_type, loc);

            // extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.oraOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 5, actual_value, min_const);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MinValue<u256, {d}>", .{mv.min});
            defer self.allocator.free(msg);
            const guard_op = self.ora_dialect.createRefinementGuard(condition, loc, msg);
            h.appendOp(self.block, guard_op);

            const guard_id = try std.fmt.allocPrint(
                self.allocator,
                "guard:{s}:{d}:{d}:min_value",
                .{ self.locations.filename, span.line, span.column },
            );
            defer self.allocator.free(guard_id);
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(self.ctx, guard_id));
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(self.ctx, "min_value"));
        },
        .max_value => |mv| {
            // generate: require(value <= max)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const max_type = if (base_type.ptr != null) base_type else value_type;
            // for u256 values, create attribute from string to support full precision
            const max_attr = if (mv.max > std.math.maxInt(i64)) blk: {
                var max_buf: [100]u8 = undefined;
                const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{mv.max}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format max_value", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const max_str_ref = h.strRef(max_str);
                break :blk c.oraIntegerAttrGetFromString(max_type, max_str_ref);
            } else blk: {
                break :blk c.oraIntegerAttrCreateI64FromType(max_type, @intCast(mv.max));
            };
            const max_const = createConstantValue(self, max_attr, max_type, loc);

            // extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.oraOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 3, actual_value, max_const);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{mv.max});
            defer self.allocator.free(msg);
            const guard_op = self.ora_dialect.createRefinementGuard(condition, loc, msg);
            h.appendOp(self.block, guard_op);

            const guard_id = try std.fmt.allocPrint(
                self.allocator,
                "guard:{s}:{d}:{d}:max_value",
                .{ self.locations.filename, span.line, span.column },
            );
            defer self.allocator.free(guard_id);
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(self.ctx, guard_id));
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(self.ctx, "max_value"));
        },
        .in_range => |ir| {
            // generate: require(min <= value <= max)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const op_type = if (base_type.ptr != null) base_type else value_type;
            // for u256 values, create attributes from strings to support full precision
            const min_attr = if (ir.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{ir.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format in_range min", "Reduce the refinement bound or report a compiler bug.");
                    return value;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(op_type, min_str_ref);
            } else blk: {
                break :blk c.oraIntegerAttrCreateI64FromType(op_type, @intCast(ir.min));
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
                break :blk c.oraIntegerAttrCreateI64FromType(op_type, @intCast(ir.max));
            };
            const min_const = createConstantValue(self, min_attr, op_type, loc);
            const max_const = createConstantValue(self, max_attr, op_type, loc);

            // check: value >= min
            // extract base type from value if it's a refinement type
            const value_base_type = c.oraRefinementTypeGetBaseType(value_type);
            const actual_value = if (value_base_type.ptr != null) blk: {
                // convert refinement type to base type using ora.refinement_to_base
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.oraOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            const min_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 5, actual_value, min_const);
            h.appendOp(self.block, min_cmp_op);
            const min_check = h.getResult(min_cmp_op, 0);

            // check: value <= max
            const max_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 3, actual_value, max_const);
            h.appendOp(self.block, max_cmp_op);
            const max_check = h.getResult(max_cmp_op, 0);

            // combine: min_check && max_check
            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, min_check, max_check);
            h.appendOp(self.block, and_op);
            const condition = h.getResult(and_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ ir.min, ir.max });
            defer self.allocator.free(msg);
            const guard_op = self.ora_dialect.createRefinementGuard(condition, loc, msg);
            h.appendOp(self.block, guard_op);

            const guard_id = try std.fmt.allocPrint(
                self.allocator,
                "guard:{s}:{d}:{d}:in_range",
                .{ self.locations.filename, span.line, span.column },
            );
            defer self.allocator.free(guard_id);
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(self.ctx, guard_id));
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(self.ctx, "in_range"));
        },
        .non_zero_address => {
            const value_type = c.oraValueGetType(value);
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const addr_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
                if (c.oraOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return value;
                }
                break :blk h.getResult(convert_op, 0);
            } else value;

            const addr_to_i160 = c.oraAddrToI160OpCreate(self.ctx, loc, addr_value);
            if (c.oraOperationIsNull(addr_to_i160)) {
                reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.addr.to.i160 returned null", "Ensure the Ora dialect is registered before lowering.");
                return value;
            }
            h.appendOp(self.block, addr_to_i160);
            const addr_i160 = h.getResult(addr_to_i160, 0);

            const i160_ty = c.oraIntegerTypeCreate(self.ctx, 160);
            const zero_attr = c.oraIntegerAttrCreateI64FromType(i160_ty, 0);
            const zero_const = createConstantValue(self, zero_attr, i160_ty, loc);

            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 1, addr_i160, zero_const);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = "Refinement violation: expected NonZeroAddress";
            const guard_op = self.ora_dialect.createRefinementGuard(condition, loc, msg);
            h.appendOp(self.block, guard_op);

            const guard_id = try std.fmt.allocPrint(
                self.allocator,
                "guard:{s}:{d}:{d}:non_zero_address",
                .{ self.locations.filename, span.line, span.column },
            );
            defer self.allocator.free(guard_id);

            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(self.ctx, guard_id));
            c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(self.ctx, "non_zero_address"));
        },
        .exact, .scaled => {
            // exact and Scaled guards are inserted at specific operations (division, arithmetic)
            // no initialization guard needed
        },
        else => {
            // not a refinement type, no guard needed
        },
    }

    return value;
}

/// Create a constant value from an attribute
pub fn createConstantValue(self: *const StatementLowerer, attr: c.MlirAttribute, ty: c.MlirType, loc: c.MlirLocation) c.MlirValue {
    const const_op = c.oraArithConstantOpCreate(self.ctx, loc, ty, attr);
    if (const_op.ptr == null) {
        @panic("Failed to create constant value");
    }
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Check if a block ends with a proper terminator (ora.yield, scf.yield, ora.return, func.return, etc.)
pub fn blockEndsWithTerminator(_: *const StatementLowerer, block: c.MlirBlock) bool {
    // get the last operation in the block
    const last_op = c.oraBlockGetTerminator(block);
    if (c.oraOperationIsNull(last_op)) {
        return false;
    }

    // check if it's a proper terminator operation
    const op_name_ref = c.oraOperationGetName(last_op);
    const name_str = op_name_ref.data;
    const name_len = op_name_ref.length;

    // todo: Check for common terminator operations - Hacky for now, but it works.
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
    _ = block;
    return false;
}

/// Check if a block ends with ora.continue (which is NOT a terminator for if regions)
pub fn blockEndsWithContinue(_: *const StatementLowerer, block: c.MlirBlock) bool {
    _ = block;
    return false;
}

/// Check if a block ends with scf.yield
pub fn blockEndsWithYield(_: *const StatementLowerer, block: c.MlirBlock) bool {
    const last_op = c.oraBlockGetTerminator(block);
    if (c.oraOperationIsNull(last_op)) {
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

    // free the string returned from oraOperationGetName
    c_zig.freeStringRef(op_name_ref);
    return result;
}

/// Check if an if statement contains return statements
pub fn ifStatementHasReturns(_: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) bool {
    // check then branch
    for (if_stmt.then_branch.statements) |stmt| {
        if (stmt == .Return) return true;
        // check nested if statements
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

    // check else branch if present
    if (if_stmt.else_branch) |else_branch| {
        for (else_branch.statements) |stmt| {
            if (stmt == .Return) return true;
            // check nested if statements
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
        // check nested if statements recursively
        if (stmt == .If) {
            if (ifStatementHasReturns(self, &stmt.If)) return true;
        }
    }
    return false;
}

/// Create a default value for a given MLIR type
pub fn createDefaultValueForType(self: *const StatementLowerer, mlir_type: c.MlirType, loc: c.MlirLocation) StatementLowerer.LoweringError!c.MlirValue {
    // check if it's an integer type
    if (c.oraTypeIsAInteger(mlir_type)) {
        // create a constant 0 value
        const const_op = self.ora_dialect.createArithConstant(0, mlir_type, loc);
        h.appendOp(self.block, const_op);
        return h.getResult(const_op, 0);
    }

    // for other types, create a zero constant (simplified)
    const constants = @import("../lower.zig");
    const zero_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const const_op = self.ora_dialect.createArithConstant(0, zero_type, loc);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Get the return type from an if statement's return statements
pub fn getReturnTypeFromIfStatement(self: *const StatementLowerer, _: *const lib.ast.Statements.IfNode) ?c.MlirType {
    // use the function's return type instead of trying to infer from individual returns
    return self.current_function_return_type;
}
