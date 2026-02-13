// ============================================================================
// Function Call Expression Lowering
// ============================================================================
// Lowering for function calls and method calls

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const builtins = lib.semantics.builtins;
const expr_helpers = @import("helpers.zig");
const operators = @import("operators.zig");
const log = @import("log");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower function call expressions
pub fn lowerCall(
    self: *const ExpressionLowerer,
    call: *const lib.ast.Expressions.CallExpr,
) c.MlirValue {
    // Check for @-prefixed builtins first
    if (call.callee.* == .Identifier) {
        const name = call.callee.Identifier.name;
        if (name.len > 1 and name[0] == '@') {
            const builtin_name = name[1..];
            // @bitCast(TargetType, value) — no-op reinterpret for bitfield <-> integer
            if (std.mem.eql(u8, builtin_name, "bitCast")) {
                if (call.arguments.len >= 2) {
                    // Second argument is the value; first is the target type (ignored at MLIR level)
                    return self.lowerExpression(call.arguments[1]);
                } else if (call.arguments.len == 1) {
                    return self.lowerExpression(call.arguments[0]);
                }
            }
            // @truncate(value) — mask to field width, used for wrapping bitfield writes
            if (std.mem.eql(u8, builtin_name, "truncate")) {
                if (call.arguments.len >= 1) {
                    return self.lowerExpression(call.arguments[0]);
                }
            }
            if (lowerOverflowBuiltin(self, builtin_name, call)) |result| {
                return result;
            }
        }
    }

    if (self.builtin_registry) |registry| {
        if (builtins.isMemberAccessChain(call.callee)) {
            const path = builtins.getMemberAccessPath(registry.allocator, call.callee) catch {
                return processNormalCall(self, call);
            };
            defer registry.allocator.free(path);

            if (registry.lookup(path)) |builtin_info| {
                return lowerBuiltinCall(self, &builtin_info, call);
            }
        }
    }

    return processNormalCall(self, call);
}

/// Process a normal (non-builtin) function call
pub fn processNormalCall(
    self: *const ExpressionLowerer,
    call: *const lib.ast.Expressions.CallExpr,
) c.MlirValue {
    var args = std.ArrayList(c.MlirValue){};
    defer args.deinit(std.heap.page_allocator);

    // get function name and parameter types for type conversion
    var function_name: ?[]const u8 = null;
    var param_types: ?[]const c.MlirType = null;

    switch (call.callee.*) {
        .Identifier => |ident| {
            function_name = ident.name;
            // look up function signature from symbol table
            if (self.symbol_table) |sym_table| {
                if (sym_table.lookupFunction(ident.name)) |func_symbol| {
                    param_types = func_symbol.param_types;
                }
            }
        },
        .FieldAccess => {
            // method calls - parameter types lookup not yet implemented
            // for now, skip conversion for method calls
        },
        else => {},
    }

    if (function_name) |name| {
        if (rewriteIdentitySelfCall(self, name, call)) |ret_value| {
            return ret_value;
        }
    }

    // lower arguments and convert to match parameter types if needed
    for (call.arguments, 0..) |arg, i| {
        var arg_value = self.lowerExpression(arg);

        // convert argument to match parameter type if function signature is available
        if (param_types) |params| {
            if (i < params.len) {
                const expected_param_type = params[i];
                const arg_type = c.oraValueGetType(arg_value);
                const arg_ref_base = c.oraRefinementTypeGetBaseType(arg_type);
                const expected_ref_base = c.oraRefinementTypeGetBaseType(expected_param_type);

                // convert if types don't match (handles subtyping conversions like u8 -> u256)
                const needs_refinement_bridge =
                    (arg_ref_base.ptr != null and c.oraTypeEqual(arg_ref_base, expected_param_type)) or
                    (expected_ref_base.ptr != null and c.oraTypeEqual(expected_ref_base, arg_type));
                if (!c.oraTypeEqual(arg_type, expected_param_type) or needs_refinement_bridge) {
                    const error_handling = @import("../error_handling.zig");
                    const arg_span = error_handling.getSpanFromExpression(arg);
                    arg_value = self.convertToType(arg_value, expected_param_type, arg_span);
                }
            }
        }

        args.append(std.heap.page_allocator, arg_value) catch {
            log.warn("Failed to append argument to function call\n", .{});
            return self.createErrorPlaceholder(call.span, "Failed to append argument");
        };
    }

    // If type resolution already classified this as an error constructor call,
    // do not lower it as func.call (there is no func.func symbol for errors).
    if (call.type_info.category == .Error) {
        if (getCalleeName(call.callee)) |error_name| {
            return lowerErrorConstructorCall(self, error_name, call.span);
        }
    }

    switch (call.callee.*) {
        .Identifier => |ident| {
            if (isErrorConstructorCallee(self, ident.name)) {
                return lowerErrorConstructorCall(self, ident.name, call.span);
            }
            return createDirectFunctionCall(self, ident.name, args.items, call.span, call.type_info);
        },
        .FieldAccess => |field_access| {
            // Intercept bitfield utility methods: .zero() and .sanitize()
            if (self.symbol_table) |st| {
                if (lowerBitfieldMethod(self, st, field_access, args.items, call.span)) |result| {
                    return result;
                }
            }
            return createMethodCall(self, field_access, args.items, call.span);
        },
        else => {
            log.err("Unsupported callee expression type\n", .{});
            return self.createErrorPlaceholder(call.span, "Unsupported callee type");
        },
    }
}

fn getCalleeName(callee: *const lib.ast.Expressions.ExprNode) ?[]const u8 {
    return switch (callee.*) {
        .Identifier => |ident| ident.name,
        .FieldAccess => |field_access| field_access.field,
        else => null,
    };
}

fn isErrorConstructorCallee(
    self: *const ExpressionLowerer,
    function_name: []const u8,
) bool {
    const sym_table = self.symbol_table orelse return false;

    // Prefer real functions when names collide.
    if (sym_table.lookupFunction(function_name) != null) return false;

    if (sym_table.lookupSymbol(function_name)) |symbol| {
        return symbol.symbol_kind == .Error;
    }

    return sym_table.getErrorId(function_name) != null;
}

fn lowerErrorConstructorCall(
    self: *const ExpressionLowerer,
    error_name: []const u8,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const result_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const error_code: i64 = if (self.symbol_table) |sym_table|
        @intCast(sym_table.getErrorId(error_name) orelse 1)
    else
        1;
    const error_id = h.identifier(self.ctx, "ora.error");
    const error_name_attr = h.stringAttr(self.ctx, error_name);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(error_id, error_name_attr),
    };

    const op = self.ora_dialect.createArithConstantWithAttrs(error_code, result_type, &attrs, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

fn rewriteIdentitySelfCall(
    self: *const ExpressionLowerer,
    function_name: []const u8,
    call: *const lib.ast.Expressions.CallExpr,
) ?c.MlirValue {
    const return_value = self.postcondition_return_value orelse return null;
    const current_name = self.current_function_name orelse resolveCurrentFunctionName(self) orelse return null;
    if (!std.mem.eql(u8, function_name, current_name)) return null;

    const param_map = self.param_map orelse return null;
    if (call.arguments.len != param_map.names.count()) return null;

    // Only rewrite identity self-calls: f(p0, p1, ..., pn) where each
    // argument is the corresponding parameter in order.
    for (call.arguments, 0..) |arg, i| {
        switch (arg.*) {
            .Identifier => |ident| {
                const param_index = param_map.getParamIndex(ident.name) orelse return null;
                if (param_index != i) return null;
            },
            else => return null,
        }
    }

    return return_value;
}

fn resolveCurrentFunctionName(self: *const ExpressionLowerer) ?[]const u8 {
    var current_block = self.block;
    while (!c.mlirBlockIsNull(current_block)) {
        const parent_op = c.mlirBlockGetParentOperation(current_block);
        if (c.oraOperationIsNull(parent_op)) return null;

        const parent_name_ref = c.oraOperationGetName(parent_op);
        if (parent_name_ref.data != null and std.mem.eql(u8, parent_name_ref.data[0..parent_name_ref.length], "func.func")) {
            const sym_name_attr = c.oraOperationGetAttributeByName(parent_op, h.strRef("sym_name"));
            if (c.oraAttributeIsNull(sym_name_attr)) return null;
            const sym_name_ref = c.oraStringAttrGetValue(sym_name_attr);
            if (sym_name_ref.data == null or sym_name_ref.length == 0) return null;
            return sym_name_ref.data[0..sym_name_ref.length];
        }

        current_block = c.mlirOperationGetBlock(parent_op);
    }
    return null;
}

/// Lower builtin function call
pub fn lowerBuiltinCall(
    self: *const ExpressionLowerer,
    builtin_info: *const builtins.BuiltinInfo,
    call: *const lib.ast.Expressions.CallExpr,
) c.MlirValue {
    const op_name = std.fmt.allocPrint(std.heap.page_allocator, "ora.evm.{s}", .{builtin_info.evm_opcode}) catch {
        log.err("Failed to allocate opcode name for builtin call\n", .{});
        return self.createErrorPlaceholder(call.span, "Failed to create builtin call");
    };
    defer std.heap.page_allocator.free(op_name);

    const location = self.fileLoc(call.span);

    const result_type = self.type_mapper.toMlirType(.{
        .ora_type = builtin_info.return_type,
    });

    const op = c.oraEvmOpCreate(
        self.ctx,
        location,
        h.strRef(op_name),
        null,
        0,
        result_type,
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create direct function call using func.call
pub fn createDirectFunctionCall(
    self: *const ExpressionLowerer,
    function_name: []const u8,
    args: []c.MlirValue,
    span: lib.ast.SourceSpan,
    call_type_info: ?lib.ast.Types.TypeInfo,
) c.MlirValue {
    var result_types_buf: [1]c.MlirType = undefined;
    var result_types: []const c.MlirType = &[_]c.MlirType{};
    var callee_returns_void = false;

    if (self.symbol_table) |sym_table| {
        if (sym_table.lookupFunction(function_name)) |func_symbol| {
            if (!c.oraTypeIsNull(func_symbol.return_type) and !c.oraTypeIsANone(func_symbol.return_type)) {
                result_types_buf[0] = func_symbol.return_type;
                result_types = result_types_buf[0..1];
            } else {
                callee_returns_void = true;
            }
        }
    }

    if (result_types.len == 0 and !callee_returns_void) {
        if (call_type_info) |ti| {
            if (ti.category == .Void) {
                callee_returns_void = true;
            } else {
                const inferred_ty = self.type_mapper.toMlirType(ti);
                if (!c.oraTypeIsNull(inferred_ty) and !c.oraTypeIsANone(inferred_ty)) {
                    result_types_buf[0] = inferred_ty;
                    result_types = result_types_buf[0..1];
                } else {
                    callee_returns_void = true;
                }
            }
        }
    }

    if (result_types.len == 0 and !callee_returns_void) {
        const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        result_types_buf[0] = result_ty;
        result_types = result_types_buf[0..1];
    }

    const op = self.ora_dialect.createFuncCall(function_name, args, result_types, self.fileLoc(span));
    h.appendOp(self.block, op);

    if (result_types.len == 0) {
        // Void calls have no SSA result; return a placeholder if an expression value
        // is requested. This keeps lowering robust for invalid value contexts.
        return self.createErrorPlaceholder(span, "void function call has no value");
    }

    return h.getResult(op, 0);
}

/// Create method call on contract instances
pub fn createMethodCall(
    self: *const ExpressionLowerer,
    field_access: lib.ast.Expressions.FieldAccessExpr,
    args: []c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const target = self.lowerExpression(field_access.target);
    const method_name = field_access.field;

    const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var all_operands = std.ArrayList(c.MlirValue){};
    defer all_operands.deinit(std.heap.page_allocator);

    all_operands.append(std.heap.page_allocator, target) catch {
        log.warn("Failed to append target to method call\n", .{});
        return self.createErrorPlaceholder(span, "Failed to append target");
    };
    for (args) |arg| {
        all_operands.append(std.heap.page_allocator, arg) catch {
            log.warn("Failed to append argument to method call\n", .{});
            return self.createErrorPlaceholder(span, "Failed to append argument");
        };
    }

    const op = c.oraMethodCallOpCreate(
        self.ctx,
        self.fileLoc(span),
        h.strRef(method_name),
        if (all_operands.items.len == 0) null else all_operands.items.ptr,
        all_operands.items.len,
        result_ty,
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

// ============================================================================
// Overflow-reporting builtins (@addWithOverflow, etc.)
// ============================================================================

fn lowerOverflowBuiltin(
    self: *const ExpressionLowerer,
    builtin_name: []const u8,
    call: *const lib.ast.Expressions.CallExpr,
) ?c.MlirValue {
    const loc = self.fileLoc(call.span);

    // Determine arity
    const is_unary = std.mem.eql(u8, builtin_name, "negWithOverflow");
    const expected_args: usize = if (is_unary) 1 else 2;
    if (call.arguments.len < expected_args) return null;

    const lhs = self.lowerExpression(call.arguments[0]);
    const lhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, lhs, call.span);
    const lhs_type_info = operators.extractTypeInfo(call.arguments[0]);
    const is_signed = operators.isSignedIntegerTypeInfo(lhs_type_info);
    const value_ty = c.oraValueGetType(lhs_uw);

    var value: c.MlirValue = undefined;
    var overflow_flag: c.MlirValue = undefined;

    if (std.mem.eql(u8, builtin_name, "addWithOverflow")) {
        const rhs = self.lowerExpression(call.arguments[1]);
        const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, call.span);
        const add_op = c.oraArithAddIOpCreate(self.ctx, loc, lhs_uw, rhs_uw);
        h.appendOp(self.block, add_op);
        value = h.getResult(add_op, 0);
        if (!is_signed) {
            // unsigned: overflow iff result < a
            const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ult"), value, lhs_uw);
            h.appendOp(self.block, cmp);
            overflow_flag = h.getResult(cmp, 0);
        } else {
            // signed: ((result ^ a) & (result ^ b)) < 0
            overflow_flag = computeSignedAddOverflow(self, value, lhs_uw, rhs_uw, loc);
        }
    } else if (std.mem.eql(u8, builtin_name, "subWithOverflow")) {
        const rhs = self.lowerExpression(call.arguments[1]);
        const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, call.span);
        const sub_op = c.oraArithSubIOpCreate(self.ctx, loc, lhs_uw, rhs_uw);
        h.appendOp(self.block, sub_op);
        value = h.getResult(sub_op, 0);
        if (!is_signed) {
            // unsigned: overflow iff a < b
            const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ult"), lhs_uw, rhs_uw);
            h.appendOp(self.block, cmp);
            overflow_flag = h.getResult(cmp, 0);
        } else {
            // signed: ((a ^ b) & (result ^ a)) < 0
            overflow_flag = computeSignedSubOverflow(self, value, lhs_uw, rhs_uw, loc);
        }
    } else if (std.mem.eql(u8, builtin_name, "mulWithOverflow")) {
        const rhs = self.lowerExpression(call.arguments[1]);
        const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, call.span);
        const mul_op = c.oraArithMulIOpCreate(self.ctx, loc, lhs_uw, rhs_uw);
        h.appendOp(self.block, mul_op);
        value = h.getResult(mul_op, 0);
        if (!is_signed) {
            // unsigned: overflow iff (b != 0 && value / b != a)
            const zero_op = self.ora_dialect.createArithConstant(0, value_ty, loc);
            h.appendOp(self.block, zero_op);
            const zero = h.getResult(zero_op, 0);
            const b_nz = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ne"), rhs_uw, zero);
            h.appendOp(self.block, b_nz);
            const quot = c.oraArithDivUIOpCreate(self.ctx, loc, value, rhs_uw);
            h.appendOp(self.block, quot);
            const ne = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ne"), h.getResult(quot, 0), lhs_uw);
            h.appendOp(self.block, ne);
            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(ne, 0), h.getResult(b_nz, 0));
            h.appendOp(self.block, and_op);
            overflow_flag = h.getResult(and_op, 0);
        } else {
            overflow_flag = computeSignedMulOverflow(self, value, lhs_uw, rhs_uw, value_ty, loc);
        }
    } else if (std.mem.eql(u8, builtin_name, "negWithOverflow")) {
        const zero_op = self.ora_dialect.createArithConstant(0, value_ty, loc);
        h.appendOp(self.block, zero_op);
        const zero = h.getResult(zero_op, 0);
        const sub_op = c.oraArithSubIOpCreate(self.ctx, loc, zero, lhs_uw);
        h.appendOp(self.block, sub_op);
        value = h.getResult(sub_op, 0);
        // unsigned: overflows unless a == 0
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ne"), lhs_uw, zero);
        h.appendOp(self.block, cmp);
        overflow_flag = h.getResult(cmp, 0);
    } else if (std.mem.eql(u8, builtin_name, "divWithOverflow") or
        std.mem.eql(u8, builtin_name, "modWithOverflow") or
        std.mem.eql(u8, builtin_name, "shlWithOverflow") or
        std.mem.eql(u8, builtin_name, "shrWithOverflow"))
    {
        if (call.arguments.len < 2) return null;
        const rhs = self.lowerExpression(call.arguments[1]);
        const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, call.span);
        const arith_op = if (std.mem.eql(u8, builtin_name, "divWithOverflow"))
            (if (is_signed) c.oraArithDivSIOpCreate(self.ctx, loc, lhs_uw, rhs_uw) else c.oraArithDivUIOpCreate(self.ctx, loc, lhs_uw, rhs_uw))
        else if (std.mem.eql(u8, builtin_name, "modWithOverflow"))
            (if (is_signed) c.oraArithRemSIOpCreate(self.ctx, loc, lhs_uw, rhs_uw) else c.oraArithRemUIOpCreate(self.ctx, loc, lhs_uw, rhs_uw))
        else if (std.mem.eql(u8, builtin_name, "shlWithOverflow"))
            c.oraArithShlIOpCreate(self.ctx, loc, lhs_uw, rhs_uw)
        else
            c.oraArithShrSIOpCreate(self.ctx, loc, lhs_uw, rhs_uw);
        h.appendOp(self.block, arith_op);
        value = h.getResult(arith_op, 0);
        overflow_flag = makeFalse(self, loc); // TODO: proper overflow detection
    } else {
        return null;
    }

    // Pack into (value, overflow) tuple via ora.struct_init
    // Build anonymous struct type with fields "0": T, "1": bool
    const val_ora = lib.ast.type_info.OraType.u256;
    const bool_ora = lib.ast.type_info.OraType.bool;
    const val_ptr = std.heap.page_allocator.create(lib.ast.type_info.OraType) catch return value;
    val_ptr.* = val_ora;
    const bool_ptr = std.heap.page_allocator.create(lib.ast.type_info.OraType) catch return value;
    bool_ptr.* = bool_ora;
    const fields = [_]lib.ast.type_info.AnonymousStructFieldType{
        .{ .name = "0", .typ = val_ptr },
        .{ .name = "1", .typ = bool_ptr },
    };
    const struct_type = self.type_mapper.mapAnonymousStructType(&fields);
    if (struct_type.ptr != null) {
        const vals = [_]c.MlirValue{ value, overflow_flag };
        const init_op = self.ora_dialect.createStructInit(&vals, struct_type, loc);
        h.appendOp(self.block, init_op);
        return h.getResult(init_op, 0);
    }
    return value;
}

/// Handle bitfield utility methods: T.zero() and val.sanitize()
fn lowerBitfieldMethod(
    self: *const ExpressionLowerer,
    st: *const constants.SymbolTable,
    field_access: lib.ast.Expressions.FieldAccessExpr,
    args: []c.MlirValue,
    span: lib.ast.SourceSpan,
) ?c.MlirValue {
    _ = args;
    const method_name = field_access.field;
    const loc = self.fileLoc(span);
    const int_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (std.mem.eql(u8, method_name, "zero")) {
        // T.zero() — target is a type name identifier
        if (field_access.target.* == .Identifier) {
            const type_name = field_access.target.Identifier.name;
            if (st.lookupType(type_name)) |type_sym| {
                if (type_sym.type_kind == .Bitfield) {
                    const op = self.ora_dialect.createArithConstant(0, int_ty, loc);
                    h.appendOp(self.block, op);
                    return h.getResult(op, 0);
                }
            }
        }
    } else if (std.mem.eql(u8, method_name, "sanitize")) {
        // val.sanitize() — clear all bits not owned by any field
        const target_type_info = operators.extractTypeInfo(field_access.target);
        if (target_type_info.ora_type) |ora_type| {
            if (ora_type == .bitfield_type) {
                if (st.lookupType(ora_type.bitfield_type)) |type_sym| {
                    if (type_sym.type_kind == .Bitfield) {
                        const target_val = self.lowerExpression(field_access.target);
                        // Compute used_mask = OR of ((1 << width) - 1) << offset for all fields
                        var used_mask: i64 = 0;
                        if (type_sym.fields) |fields| {
                            for (fields) |field| {
                                const off: u6 = @intCast(if (field.offset) |o| o else 0);
                                const w: u6 = @intCast(field.bit_width orelse 0);
                                if (w > 0 and w < 64) {
                                    const field_mask = ((@as(i64, 1) << w) - 1) << off;
                                    used_mask |= field_mask;
                                }
                            }
                        }
                        const mask_op = self.ora_dialect.createArithConstant(used_mask, int_ty, loc);
                        h.appendOp(self.block, mask_op);
                        const and_op = c.oraArithAndIOpCreate(self.ctx, loc, target_val, h.getResult(mask_op, 0));
                        h.appendOp(self.block, and_op);
                        return h.getResult(and_op, 0);
                    }
                }
            }
        }
    }
    return null;
}

/// Signed add overflow: ((result ^ a) & (result ^ b)) < 0
fn computeSignedAddOverflow(self: *const ExpressionLowerer, result: c.MlirValue, a: c.MlirValue, b: c.MlirValue, loc: c.MlirLocation) c.MlirValue {
    const xor_ra = c.oraArithXorIOpCreate(self.ctx, loc, result, a);
    h.appendOp(self.block, xor_ra);
    const xor_rb = c.oraArithXorIOpCreate(self.ctx, loc, result, b);
    h.appendOp(self.block, xor_rb);
    const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(xor_ra, 0), h.getResult(xor_rb, 0));
    h.appendOp(self.block, and_op);
    const zero_op = self.ora_dialect.createArithConstant(0, c.oraValueGetType(result), loc);
    h.appendOp(self.block, zero_op);
    const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("slt"), h.getResult(and_op, 0), h.getResult(zero_op, 0));
    h.appendOp(self.block, cmp);
    return h.getResult(cmp, 0);
}

/// Signed sub overflow: ((a ^ b) & (result ^ a)) < 0
fn computeSignedSubOverflow(self: *const ExpressionLowerer, result: c.MlirValue, a: c.MlirValue, b: c.MlirValue, loc: c.MlirLocation) c.MlirValue {
    const xor_ab = c.oraArithXorIOpCreate(self.ctx, loc, a, b);
    h.appendOp(self.block, xor_ab);
    const xor_ra = c.oraArithXorIOpCreate(self.ctx, loc, result, a);
    h.appendOp(self.block, xor_ra);
    const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(xor_ab, 0), h.getResult(xor_ra, 0));
    h.appendOp(self.block, and_op);
    const zero_op = self.ora_dialect.createArithConstant(0, c.oraValueGetType(result), loc);
    h.appendOp(self.block, zero_op);
    const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("slt"), h.getResult(and_op, 0), h.getResult(zero_op, 0));
    h.appendOp(self.block, cmp);
    return h.getResult(cmp, 0);
}

/// Signed mul overflow: (a == MIN_INT && b == -1) || (b != 0 && sdiv(result, b) != a)
fn computeSignedMulOverflow(self: *const ExpressionLowerer, result: c.MlirValue, a: c.MlirValue, b: c.MlirValue, val_ty: c.MlirType, loc: c.MlirLocation) c.MlirValue {
    const pred_eq = expr_helpers.predicateStringToInt("eq");
    const pred_ne = expr_helpers.predicateStringToInt("ne");
    // MIN_INT = 1 << 255
    const one_op = self.ora_dialect.createArithConstant(1, val_ty, loc);
    h.appendOp(self.block, one_op);
    const s255_op = self.ora_dialect.createArithConstant(255, val_ty, loc);
    h.appendOp(self.block, s255_op);
    const min_op = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(one_op, 0), h.getResult(s255_op, 0));
    h.appendOp(self.block, min_op);
    const neg1_op = self.ora_dialect.createArithConstant(-1, val_ty, loc);
    h.appendOp(self.block, neg1_op);
    // special = (a == MIN) && (b == -1)
    const a_min = c.oraArithCmpIOpCreate(self.ctx, loc, pred_eq, a, h.getResult(min_op, 0));
    h.appendOp(self.block, a_min);
    const b_n1 = c.oraArithCmpIOpCreate(self.ctx, loc, pred_eq, b, h.getResult(neg1_op, 0));
    h.appendOp(self.block, b_n1);
    const special = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(a_min, 0), h.getResult(b_n1, 0));
    h.appendOp(self.block, special);
    // general = b != 0 && sdiv(result, b) != a
    const zero_op = self.ora_dialect.createArithConstant(0, val_ty, loc);
    h.appendOp(self.block, zero_op);
    const b_nz = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ne, b, h.getResult(zero_op, 0));
    h.appendOp(self.block, b_nz);
    const quot = c.oraArithDivSIOpCreate(self.ctx, loc, result, b);
    h.appendOp(self.block, quot);
    const mismatch = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ne, h.getResult(quot, 0), a);
    h.appendOp(self.block, mismatch);
    const general = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(mismatch, 0), h.getResult(b_nz, 0));
    h.appendOp(self.block, general);
    // overflow = special || general
    const or_op = c.oraArithOrIOpCreate(self.ctx, loc, h.getResult(special, 0), h.getResult(general, 0));
    h.appendOp(self.block, or_op);
    return h.getResult(or_op, 0);
}

fn makeFalse(self: *const ExpressionLowerer, loc: c.MlirLocation) c.MlirValue {
    const op = self.ora_dialect.createArithConstantBool(false, loc);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}
