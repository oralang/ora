// ============================================================================
// Statement Lowering Helpers
// ============================================================================
// Common helper functions used across statement lowering modules

const std = @import("std");
const c = @import("mlir_c_api").c;
const c_zig = @import("mlir_c_api");
const h = @import("../helpers.zig");
const error_handling = @import("../error_handling.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const lib = @import("ora_lib");
const log = @import("log");
const constants = @import("../lower.zig");
const expr_helpers = @import("../expressions/helpers.zig");
const guard_helpers = @import("../refinement_guard_helpers.zig");
const slot_key = lib.ast.slot_key;

pub fn isErrorUnionTypeInfo(ti: lib.ast.Types.TypeInfo) bool {
    if (ti.category == .ErrorUnion) return true;
    if (ti.ora_type) |ot| switch (ot) {
        .error_union => return true,
        ._union => |members| return members.len > 0 and members[0] == .error_union,
        else => {},
    };
    return false;
}

fn pathBaseIdentifier(expr: *const lib.ast.Expressions.ExprNode) ?[]const u8 {
    return switch (expr.*) {
        .Identifier => |id| id.name,
        .FieldAccess => |fa| pathBaseIdentifier(fa.target),
        .Index => |ix| pathBaseIdentifier(ix.target),
        else => null,
    };
}

fn isStoragePath(self: *const StatementLowerer, path: *const lib.ast.Expressions.ExprNode) bool {
    const base = pathBaseIdentifier(path) orelse return false;
    if (self.symbol_table) |st| {
        if (st.lookupSymbol(base)) |sym| {
            if (std.mem.eql(u8, sym.region, "storage")) return true;
        }
    }
    if (self.storage_map) |sm| {
        if (sm.hasStorageVariable(base)) return true;
    }
    return false;
}

pub fn buildStoragePathKey(
    allocator: std.mem.Allocator,
    path: *const lib.ast.Expressions.ExprNode,
) StatementLowerer.LoweringError!?[]const u8 {
    return slot_key.buildPathSlotKey(allocator, path) catch return StatementLowerer.LoweringError.OutOfMemory;
}

pub fn buildRuntimeLockKeyForPath(
    self: *const StatementLowerer,
    path: *const lib.ast.Expressions.ExprNode,
) StatementLowerer.LoweringError!?[]const u8 {
    // Locking applies only to storage slots. Runtime lock state itself is tx-scoped in TSTORE.
    if (!isStoragePath(self, path)) return null;
    if (!slot_key.runtimeLockPathSupported(path)) return null;
    return buildStoragePathKey(self.allocator, path);
}

pub fn lockRuntimeResourceExpr(path: *const lib.ast.Expressions.ExprNode) *const lib.ast.Expressions.ExprNode {
    if (path.* == .Index and path.Index.target.* == .Identifier) return path.Index.index;
    return path;
}

pub fn maybeEmitStorageGuardForPath(
    self: *const StatementLowerer,
    path: *const lib.ast.Expressions.ExprNode,
    loc: c.MlirLocation,
) StatementLowerer.LoweringError!void {
    if (!isStoragePath(self, path)) return;
    const root = pathBaseIdentifier(path) orelse return;

    const guarded = blk: {
        if (self.guarded_storage_bases) |set| break :blk set;
        if (self.symbol_table) |st| break :blk &st.contract_locked_storage_roots;
        break :blk null;
    } orelse return;

    if (!guarded.contains(root)) return;

    const key = (try buildRuntimeLockKeyForPath(self, path)) orelse return;
    defer self.allocator.free(key);

    const resource_expr = lockRuntimeResourceExpr(path);
    const resource = self.expr_lowerer.lowerExpression(resource_expr);
    const guard_op = self.ora_dialect.createTStoreGuard(resource, key, loc);
    h.appendOp(self.block, guard_op);
}

fn unwrapRefinementValueWithCache(self: *const StatementLowerer, value: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
    const value_type = c.oraValueGetType(value);
    const base_type = c.oraRefinementTypeGetBaseType(value_type);
    if (base_type.ptr == null) {
        return value;
    }

    const unwrapped = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.expr_lowerer.refinement_base_cache, value, span);
    const unwrapped_type = c.oraValueGetType(unwrapped);
    if (!c.oraTypeEqual(unwrapped_type, base_type)) {
        reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
        return value;
    }
    return unwrapped;
}

fn refinementCacheKey(value: c.MlirValue) usize {
    if (c.mlirValueIsABlockArgument(value)) {
        const owner = c.mlirBlockArgumentGetOwner(value);
        const arg_no = c.mlirBlockArgumentGetArgNumber(value);
        const block_key = @intFromPtr(owner.ptr);
        const arg_key: usize = @intCast(arg_no);
        return block_key ^ (arg_key *% 0x9e3779b97f4a7c15);
    }
    return @intFromPtr(value.ptr);
}

/// Wrap an error union payload without encoding. Encoding happens in Ora→SIR.
pub fn encodeErrorUnionValue(
    self: *const StatementLowerer,
    payload: c.MlirValue,
    is_error: bool,
    target_type: c.MlirType,
    target_block: c.MlirBlock,
    span: lib.ast.SourceSpan,
    loc: c.MlirLocation,
) c.MlirValue {
    // Check if target is an error union type - if so, extract the success type for wrapping
    const success_type = c.oraErrorUnionTypeGetSuccessType(target_type);
    const error_id_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const encode_type = if (is_error)
        error_id_type
    else if (!c.oraTypeIsNull(success_type))
        success_type
    else
        target_type;

    const target_builtin = c.oraTypeToBuiltin(encode_type);
    if (!c.oraTypeIsAInteger(target_builtin)) {
        return payload;
    }

    const payload_conv = convertValueToType(self, payload, encode_type, span, loc);

    // If the target is an error union type, wrap the payload using ora.error.ok or ora.error.err
    if (!c.oraTypeIsNull(success_type)) {
        const wrap_op = if (is_error)
            c.oraErrorErrOpCreate(self.ctx, loc, payload_conv, target_type)
        else
            c.oraErrorOkOpCreate(self.ctx, loc, payload_conv, target_type);
        h.appendOp(target_block, wrap_op);
        return h.getResult(wrap_op, 0);
    }
    return payload_conv;
}

pub const ErrorUnionPayload = struct {
    payload: c.MlirValue,
    is_error: bool,
};

pub fn getErrorUnionPayload(
    self: *const StatementLowerer,
    value_expr: *const lib.ast.Expressions.ExprNode,
    value: c.MlirValue,
    target_type: c.MlirType,
    target_block: c.MlirBlock,
    loc: c.MlirLocation,
) ErrorUnionPayload {
    _ = target_type;
    var payload = value;
    var is_error = false;
    var err_name: ?[]const u8 = null;

    switch (value_expr.*) {
        .ErrorReturn => |err_ret| {
            is_error = true;
            err_name = err_ret.error_name;
        },
        .Identifier => |ident| {
            if (ident.type_info.category == .Error) {
                is_error = true;
                err_name = ident.name;
            }
        },
        .Call => |call| {
            const call_error_name = extractErrorNameFromCallee(call.callee);
            if (self.symbol_table) |st| {
                if (call_error_name) |name| {
                    if (st.getErrorId(name) != null) {
                        is_error = true;
                        err_name = name;
                    }
                }
            }
            if (!is_error and call.type_info.category == .Error) {
                is_error = true;
                err_name = call_error_name;
            }
        },
        else => {},
    }

    if (is_error) {
        const err_id = if (self.symbol_table) |st|
            if (err_name) |name| st.getErrorId(name) else null
        else
            null;
        const err_val: i64 = @intCast(err_id orelse 1);
        const const_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        payload = createArithConstantInBlock(self, target_block, err_val, const_type, loc);
    }

    return .{ .payload = payload, .is_error = is_error };
}

fn extractErrorNameFromCallee(callee: *const lib.ast.Expressions.ExprNode) ?[]const u8 {
    return switch (callee.*) {
        .Identifier => |ident| ident.name,
        .FieldAccess => |field_access| field_access.field,
        else => null,
    };
}

fn createArithConstantInFunction(
    self: *const StatementLowerer,
    value: i64,
    target_type: c.MlirType,
    loc: c.MlirLocation,
) c.MlirValue {
    var lowerer = self.*;
    lowerer.block = self.function_block;
    const op = lowerer.ora_dialect.createArithConstant(value, target_type, loc);
    const first_op = c.oraBlockGetFirstOperation(lowerer.block);
    if (!c.oraOperationIsNull(first_op)) {
        h.insertOpBefore(lowerer.block, op, first_op);
    } else {
        h.appendOp(lowerer.block, op);
    }
    return h.getResult(op, 0);
}

fn createArithConstantInBlock(
    self: *const StatementLowerer,
    block: c.MlirBlock,
    value: i64,
    target_type: c.MlirType,
    loc: c.MlirLocation,
) c.MlirValue {
    var lowerer = self.*;
    lowerer.block = block;
    const op = lowerer.ora_dialect.createArithConstant(value, target_type, loc);
    const first_op = c.oraBlockGetFirstOperation(lowerer.block);
    if (!c.oraOperationIsNull(first_op)) {
        h.insertOpBefore(lowerer.block, op, first_op);
    } else {
        h.appendOp(lowerer.block, op);
    }
    return h.getResult(op, 0);
}

/// Lower a value expression, inserting an implicit try inside try blocks when appropriate.
pub fn lowerValueWithImplicitTry(
    self: *StatementLowerer,
    expr: *const lib.ast.Expressions.ExprNode,
    expected_type: ?lib.ast.Types.TypeInfo,
) c.MlirValue {
    if (self.in_try_block) {
        const needs_unwrap = blk: {
            switch (expr.*) {
                .Call => |call| break :blk isErrorUnionTypeInfo(call.type_info),
                .Identifier => |id| break :blk isErrorUnionTypeInfo(id.type_info),
                .FieldAccess => |fa| break :blk isErrorUnionTypeInfo(fa.type_info),
                else => break :blk false,
            }
        };
        if (needs_unwrap) {
            if (expected_type == null or !isErrorUnionTypeInfo(expected_type.?)) {
                const expr_value = self.expr_lowerer.lowerExpressionWithExpectedType(expr, expected_type);
                if (c.oraValueIsNull(expr_value)) return expr_value;

                const expr_ty = c.oraValueGetType(expr_value);
                const success_ty = c.oraErrorUnionTypeGetSuccessType(expr_ty);
                if (!c.oraTypeIsNull(success_ty)) {
                    const loc = self.fileLoc(error_handling.getSpanFromExpression(expr));
                    // Inside a try block, unwrap directly to avoid nested ora.try_stmt.
                    const unwrap_op = self.ora_dialect.createErrorUnwrap(expr_value, success_ty, loc);
                    h.appendOp(self.block, unwrap_op);
                    return h.getResult(unwrap_op, 0);
                }
            }

            var try_expr = lib.ast.Expressions.TryExpr{
                .expr = @constCast(expr),
                .span = error_handling.getSpanFromExpression(expr),
            };
            return self.expr_lowerer.lowerTry(&try_expr);
        }
    }
    return self.expr_lowerer.lowerExpressionWithExpectedType(expr, expected_type);
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
    if (!c.oraOperationIsNull(store_op)) {
        h.appendOp(self.block, store_op);
    }
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
    const converted = self.expr_lowerer.convertToType(value, target_type, span);
    if (c.oraValueIsNull(converted)) {
        return converted;
    }
    const converted_type = c.oraValueGetType(converted);

    // check if conversion succeeded
    if (c.oraTypeEqual(converted_type, target_type)) {
        return converted;
    }

    // fallback: try bitcast for same-width integer types
    // avoid bitcast when refinement types are involved; use explicit refinement ops
    const value_ref_base = c.oraRefinementTypeGetBaseType(value_type);
    const target_ref_base = c.oraRefinementTypeGetBaseType(target_type);
    if (value_ref_base.ptr != null or target_ref_base.ptr != null) {
        return converted;
    }

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

    if (self.expr_lowerer.error_handler) |handler| {
        handler.reportError(
            .TypeMismatch,
            span,
            "Failed to convert value to target MLIR type during statement lowering",
            "Add an explicit supported conversion or align source and target types.",
        ) catch {};
    } else {
        log.err("Failed to convert value to target MLIR type during statement lowering\n", .{});
    }
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
    var_name: ?[]const u8,
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

    const value_type = c.oraValueGetType(value);
    const base_type = c.oraRefinementTypeGetBaseType(value_type);
    if (base_type.ptr != null) {
        if (self.expr_lowerer.refinement_base_cache) |cache| {
            const target_mlir = self.type_mapper.toMlirType(.{ .ora_type = ora_type });
            const same_refinement = c.oraTypeEqual(value_type, target_mlir);
            if (same_refinement and cache.contains(refinementCacheKey(value))) {
                // Same refinement already unwrapped in this block; avoid re-guarding.
                return value;
            }
        }
    }

    const loc = self.fileLoc(span);

    const is_unsigned = switch (ora_type) {
        .u8, .u16, .u32, .u64, .u128, .u256 => true,
        .min_value => |mv| switch (mv.base.*) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        },
        .max_value => |mv| switch (mv.base.*) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        },
        .in_range => |ir| switch (ir.base.*) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        },
        .scaled => |s| switch (s.base.*) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        },
        .exact => |e| switch (e.*) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        },
        else => false,
    };
    const pred_ge: i64 = if (is_unsigned) 9 else 5; // uge or sge
    const pred_le: i64 = if (is_unsigned) 7 else 3; // ule or sle

    switch (ora_type) {
        .min_value => |mv| {
            // generate: require(value >= min)
            // extract base type from refinement type for operations
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

            const actual_value = unwrapRefinementValueWithCache(self, value, span);
            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ge, actual_value, min_const);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MinValue<u256, {d}>", .{mv.min});
            defer self.allocator.free(msg);
            guard_helpers.emitRefinementGuard(
                self.ctx,
                self.block,
                self.ora_dialect,
                self.locations,
                self.refinement_guard_cache,
                span,
                condition,
                msg,
                "min_value",
                var_name,
                self.allocator,
            );
        },
        .max_value => |mv| {
            // generate: require(value <= max)
            // extract base type from refinement type for operations
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

            const actual_value = unwrapRefinementValueWithCache(self, value, span);
            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, pred_le, actual_value, max_const);
            h.appendOp(self.block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{mv.max});
            defer self.allocator.free(msg);
            guard_helpers.emitRefinementGuard(
                self.ctx,
                self.block,
                self.ora_dialect,
                self.locations,
                self.refinement_guard_cache,
                span,
                condition,
                msg,
                "max_value",
                var_name,
                self.allocator,
            );
        },
        .in_range => |ir| {
            // generate: require(min <= value <= max)
            // extract base type from refinement type for operations
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
            const actual_value = unwrapRefinementValueWithCache(self, value, span);
            const min_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ge, actual_value, min_const);
            h.appendOp(self.block, min_cmp_op);
            const min_check = h.getResult(min_cmp_op, 0);

            // check: value <= max
            const max_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, pred_le, actual_value, max_const);
            h.appendOp(self.block, max_cmp_op);
            const max_check = h.getResult(max_cmp_op, 0);

            // combine: min_check && max_check
            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, min_check, max_check);
            h.appendOp(self.block, and_op);
            const condition = h.getResult(and_op, 0);

            const msg = try std.fmt.allocPrint(self.allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ ir.min, ir.max });
            defer self.allocator.free(msg);
            guard_helpers.emitRefinementGuard(
                self.ctx,
                self.block,
                self.ora_dialect,
                self.locations,
                self.refinement_guard_cache,
                span,
                condition,
                msg,
                "in_range",
                var_name,
                self.allocator,
            );
        },
        .non_zero_address => {
            const addr_value = if (base_type.ptr != null)
                unwrapRefinementValueWithCache(self, value, span)
            else
                value;

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
            guard_helpers.emitRefinementGuard(
                self.ctx,
                self.block,
                self.ora_dialect,
                self.locations,
                self.refinement_guard_cache,
                span,
                condition,
                msg,
                "non_zero_address",
                var_name,
                self.allocator,
            );
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
        "cf.br",
        "cf.cond_br",
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

/// Check if a block ends with a specific terminator op name.
pub fn blockEndsWithOpName(block: c.MlirBlock, op_name: []const u8) bool {
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

    const matches = name_len == op_name.len and std.mem.eql(u8, name_str[0..name_len], op_name);
    c_zig.freeStringRef(op_name_ref);
    return matches;
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
        if (stmt == .Switch) {
            if (switchStatementHasReturns(self, &stmt.Switch)) return true;
        }
        if (stmt == .TryBlock) {
            if (blockHasReturn(self, stmt.TryBlock.try_block)) return true;
            if (stmt.TryBlock.catch_block) |catch_block| {
                if (blockHasReturn(self, catch_block.block)) return true;
            }
        }
        if (stmt == .While) {
            if (blockHasReturn(self, stmt.While.body)) return true;
        }
        if (stmt == .ForLoop) {
            if (blockHasReturn(self, stmt.ForLoop.body)) return true;
        }
        if (stmt == .LabeledBlock) {
            if (blockHasReturn(self, stmt.LabeledBlock.block)) return true;
        }
    }
    return false;
}

/// Check if a block always returns on all control-flow paths (conservative).
/// Get the span from an expression node
pub fn getExprSpan(expr: *const lib.ast.Expressions.ExprNode) lib.ast.SourceSpan {
    return switch (expr.*) {
        .Identifier => |ident| ident.span,
        .Literal => |lit| switch (lit) {
            .Integer => |int| int.span,
            .String => |str| str.span,
            .Bool => |bool_lit| bool_lit.span,
            .Address => |addr| addr.span,
            .Hex => |hex| hex.span,
            .Binary => |bin| bin.span,
            .Character => |char| char.span,
            .Bytes => |bytes| bytes.span,
        },
        .Binary => |bin| bin.span,
        .Unary => |unary| unary.span,
        .Assignment => |assign| assign.span,
        .CompoundAssignment => |comp_assign| comp_assign.span,
        .Call => |call| call.span,
        .Index => |index| index.span,
        .FieldAccess => |field| field.span,
        .Cast => |cast| cast.span,
        .Comptime => |comptime_expr| comptime_expr.span,
        .Old => |old| old.span,
        .Tuple => |tuple| tuple.span,
        .SwitchExpression => |switch_expr| switch_expr.span,
        .Quantified => |quantified| quantified.span,
        .Try => |try_expr| try_expr.span,
        .ErrorReturn => |error_ret| error_ret.span,
        .ErrorCast => |error_cast| error_cast.span,
        .Shift => |shift| shift.span,
        .StructInstantiation => |struct_inst| struct_inst.span,
        .AnonymousStruct => |anon_struct| anon_struct.span,
        .Range => |range| range.span,
        .LabeledBlock => |labeled_block| labeled_block.span,
        .Destructuring => |destructuring| destructuring.span,
        .EnumLiteral => |enum_lit| enum_lit.span,
        .ArrayLiteral => |array_lit| array_lit.span,
    };
}

fn switchStatementHasReturns(self: *const StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) bool {
    for (switch_stmt.cases) |case| {
        switch (case.body) {
            .Block => |block| {
                if (blockHasReturn(self, block)) return true;
            },
            .LabeledBlock => |labeled| {
                if (blockHasReturn(self, labeled.block)) return true;
            },
            else => {},
        }
    }
    if (switch_stmt.default_case) |default_block| {
        if (blockHasReturn(self, default_block)) return true;
    }
    return false;
}

/// Create a default value for a given MLIR type
pub fn createDefaultValueForType(self: *const StatementLowerer, mlir_type: c.MlirType, loc: c.MlirLocation) StatementLowerer.LoweringError!c.MlirValue {
    // check if it's a refinement type and get its base type
    const refinement_base = c.oraRefinementTypeGetBaseType(mlir_type);
    if (refinement_base.ptr != null) {
        // create a constant of the base type, then convert to refinement type
        const base_const_op = self.ora_dialect.createArithConstant(0, refinement_base, loc);
        h.appendOp(self.block, base_const_op);
        const base_value = h.getResult(base_const_op, 0);

        // convert base value to refinement type
        const convert_op = c.oraBaseToRefinementOpCreate(self.ctx, loc, base_value, mlir_type, self.block);
        if (convert_op.ptr != null) {
            return h.getResult(convert_op, 0);
        }
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .MlirOperationFailed,
                null,
                "Failed to create default refinement value during statement lowering",
                "Ensure ora.base_to_refinement is available for this refinement type.",
            ) catch {};
        } else {
            log.err("Failed to create default refinement value during statement lowering\n", .{});
        }
        return StatementLowerer.LoweringError.MlirOperationFailed;
    }

    // check if it's an integer type
    if (c.oraTypeIsAInteger(mlir_type)) {
        // create a constant 0 value
        const const_op = self.ora_dialect.createArithConstant(0, mlir_type, loc);
        h.appendOp(self.block, const_op);
        return h.getResult(const_op, 0);
    }

    if (self.expr_lowerer.error_handler) |handler| {
        handler.reportError(
            .TypeMismatch,
            null,
            "Cannot synthesize a default value for non-integer statement result type",
            "Add explicit lowering for this MLIR type instead of using an integer fallback.",
        ) catch {};
    } else {
        log.err("Cannot synthesize a default value for non-integer statement result type\n", .{});
    }
    return StatementLowerer.LoweringError.TypeMismatch;
}

/// Get the return type from an if statement's return statements
pub fn getReturnTypeFromIfStatement(self: *const StatementLowerer, _: *const lib.ast.Statements.IfNode) ?c.MlirType {
    // use the function's return type instead of trying to infer from individual returns
    return self.current_function_return_type;
}
