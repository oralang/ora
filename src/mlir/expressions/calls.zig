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
const log = @import("log");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower function call expressions
pub fn lowerCall(
    self: *const ExpressionLowerer,
    call: *const lib.ast.Expressions.CallExpr,
) c.MlirValue {
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

    // lower arguments and convert to match parameter types if needed
    for (call.arguments, 0..) |arg, i| {
        var arg_value = self.lowerExpression(arg);

        // convert argument to match parameter type if function signature is available
        if (param_types) |params| {
            if (i < params.len) {
                const expected_param_type = params[i];
                const arg_type = c.oraValueGetType(arg_value);

                // convert if types don't match (handles subtyping conversions like u8 -> u256)
                if (!c.oraTypeEqual(arg_type, expected_param_type)) {
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

    switch (call.callee.*) {
        .Identifier => |ident| {
            return createDirectFunctionCall(self, ident.name, args.items, call.span);
        },
        .FieldAccess => |field_access| {
            return createMethodCall(self, field_access, args.items, call.span);
        },
        else => {
            log.err("Unsupported callee expression type\n", .{});
            return self.createErrorPlaceholder(call.span, "Unsupported callee type");
        },
    }
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
) c.MlirValue {
    var result_types_buf: [1]c.MlirType = undefined;
    var result_types: []const c.MlirType = &[_]c.MlirType{};

    if (self.symbol_table) |sym_table| {
        if (sym_table.lookupFunction(function_name)) |func_symbol| {
            if (!c.oraTypeIsNull(func_symbol.return_type)) {
                result_types_buf[0] = func_symbol.return_type;
                result_types = result_types_buf[0..1];
            }
        }
    }

    if (result_types.len == 0) {
        const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        result_types_buf[0] = result_ty;
        result_types = result_types_buf[0..1];
    }

    const op = self.ora_dialect.createFuncCall(function_name, args, result_types, self.fileLoc(span));
    h.appendOp(self.block, op);
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
