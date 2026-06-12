const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const ConstValue = @import("ora_types").ConstValue;
const abi_runtime_encoder = @import("../abi/runtime_encoder.zig");
const abi_runtime_decoder = @import("../abi/runtime_decoder.zig");
const abi_layout_context = @import("../abi/layout_context.zig");
const const_bridge = @import("../comptime/compiler_const_bridge.zig");
const source = @import("../source/mod.zig");
const type_descriptors = @import("../sema/type_descriptors.zig");
const lookup_index = @import("../sema/lookup.zig");
const hir_locals = @import("locals.zig");
const support = @import("support.zig");

const appendOp = support.appendOp;
const appendScfYieldValues = support.appendScfYieldValues;
const appendValueOp = support.appendValueOp;
const addressType = support.addressType;
const boolType = support.boolType;
const bytesType = support.bytesType;
const cmpPredicate = support.cmpPredicate;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const exprRange = support.exprRange;
const lowerTypeDescriptor = support.lowerTypeDescriptor;
const nullStringRef = support.nullStringRef;
const namedStringAttr = support.namedStringAttr;
const namedTypeAttr = support.namedTypeAttr;
const parseUnsignedIntegerLiteral = support.parseUnsignedIntegerLiteral;
const strRef = support.strRef;
const stringType = support.stringType;
const LocalEnv = hir_locals.LocalEnv;

pub fn mixin(FunctionLowerer: type, Lowerer: type) type {
    _ = Lowerer;
    return struct {
        // Methods in this returned type are mixed into the parent lowerer. When
        // one helper calls another, use @This().helper(self): `self` is the
        // parent FunctionLowerer, not an instance of the anonymous mixin type.
        fn semaIntegerTypeIsSigned(self: *FunctionLowerer, ty: sema.Type) anyerror!?bool {
            const unwrapped = support.unwrapRefinementSemaType(ty);
            if (unwrapped == .named) {
                if (self.parent.substitutedType(unwrapped.named.name)) |substituted| {
                    return try @This().semaIntegerTypeIsSigned(self, substituted);
                }
            }
            return support.resolvedIntegerSignedness(ty) catch |err| switch (err) {
                error.MlirOperationCreationFailed => null,
                else => return err,
            };
        }

        fn exprIntegerSignedness(self: *FunctionLowerer, expr_id: ast.ExprId) anyerror!?bool {
            return try @This().semaIntegerTypeIsSigned(self, self.parent.typecheck.exprType(expr_id));
        }

        fn reportIndeterminateIntegerSignedness(self: *FunctionLowerer, range: source.TextRange) anyerror!bool {
            try self.parent.emitLoweringError(
                range,
                "cannot determine signedness for integer expression; annotate the expression or one operand with a concrete integer type",
                .{},
            );
            return false;
        }

        fn binaryIntegerExprIsSigned(self: *FunctionLowerer, expr_id: ast.ExprId, lhs_expr: ast.ExprId, rhs_expr: ast.ExprId) anyerror!bool {
            if (try @This().exprIntegerSignedness(self, lhs_expr)) |signed| return signed;
            if (try @This().exprIntegerSignedness(self, rhs_expr)) |signed| return signed;
            return (try @This().exprIntegerSignedness(self, expr_id)) orelse
                try @This().reportIndeterminateIntegerSignedness(self, exprRange(self.parent.file, expr_id));
        }

        fn integerExprSignedness(self: *FunctionLowerer, expr_id: ast.ExprId) anyerror!bool {
            return (try @This().exprIntegerSignedness(self, expr_id)) orelse
                try @This().reportIndeterminateIntegerSignedness(self, exprRange(self.parent.file, expr_id));
        }

        fn layoutContext(self: *FunctionLowerer) abi_layout_context.LayoutContext {
            return .{
                .allocator = self.parent.allocator,
                .provider = sema.abiLayoutProvider(self.parent.file, self.parent.item_index, self.parent.typecheck),
            };
        }

        fn predicateForBinaryCompare(op: ast.BinaryOp, is_signed: bool) []const u8 {
            return switch (op) {
                .lt => if (is_signed) "slt" else "ult",
                .le => if (is_signed) "sle" else "ule",
                .gt => if (is_signed) "sgt" else "ugt",
                .ge => if (is_signed) "sge" else "uge",
                .eq => "eq",
                .ne => "ne",
                else => unreachable,
            };
        }

        fn tryLowerConstEvalValue(self: *FunctionLowerer, expr_id: ast.ExprId) anyerror!?mlir.MlirValue {
            const value = self.parent.const_eval.values[expr_id.index()] orelse return null;
            const expr = self.parent.file.expression(expr_id).*;
            const is_required_comptime = expr == .Comptime;
            if (!is_required_comptime) {
                if (@This().exprDependsOnRuntimeState(self, expr_id)) return null;
                if (@This().exprDependsOnMutablePatternLocals(self, expr_id)) return null;
            }
            switch (expr) {
                .Binary, .Unary, .Comptime, .Group, .Call, .Builtin, .Tuple => {},
                .Field => {},
                .Name => {
                    const binding = self.parent.resolution.expr_bindings[expr_id.index()] orelse return null;
                    switch (binding) {
                        .pattern => return null,
                        .item => |item_id| switch (self.parent.file.item(item_id).*) {
                            .Field => |field| if (field.storage_class != .none) return null,
                            .Constant => {},
                            else => return null,
                        },
                    }
                },
                else => return null,
            }

            const result_type = self.parent.lowerExprType(expr_id);
            if (mlir.oraTypeIsNull(result_type)) return null;
            const loc = self.parent.location(exprRange(self.parent.file, expr_id));
            const lowered = try @This().lowerPersistedConstValue(self, value, self.parent.typecheck.exprType(expr_id), result_type, loc);
            if (lowered == null and is_required_comptime) return error.ComptimeValueNotRuntimeLowerable;
            return lowered;
        }

        fn lowerPersistedConstValue(
            self: *FunctionLowerer,
            value: ConstValue,
            sema_type: sema.Type,
            result_type: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!?mlir.MlirValue {
            return switch (value) {
                .integer => |integer| blk: {
                    if (!mlir.oraTypeIsAInteger(result_type)) break :blk null;
                    const attr = if (integer.toInt(i64)) |small|
                        mlir.oraIntegerAttrCreateI64FromType(result_type, small)
                    else |_| blk2: {
                        const text = try integer.toString(self.parent.allocator, 10, .lower);
                        defer self.parent.allocator.free(text);
                        break :blk2 mlir.oraIntegerAttrGetFromString(result_type, strRef(text));
                    };
                    break :blk appendValueOp(self.block, mlir.oraArithConstantOpCreate(self.parent.context, loc, result_type, attr));
                },
                .fixed_bytes => |bytes| blk: {
                    if (sema_type != .fixed_bytes or sema_type.fixed_bytes.len != bytes.len) break :blk null;
                    if (!mlir.oraTypeIsAInteger(result_type)) break :blk null;
                    var value_int = try std.math.big.int.Managed.initSet(self.parent.allocator, 0);
                    for (bytes) |byte| {
                        var shifted = try std.math.big.int.Managed.init(self.parent.allocator);
                        try std.math.big.int.Managed.shiftLeft(&shifted, &value_int, 8);
                        var byte_int = try std.math.big.int.Managed.initSet(self.parent.allocator, byte);
                        var next = try std.math.big.int.Managed.init(self.parent.allocator);
                        try std.math.big.int.Managed.bitOr(&next, &shifted, &byte_int);
                        value_int = next;
                    }
                    const attr = if (value_int.toInt(i64)) |small|
                        mlir.oraIntegerAttrCreateI64FromType(result_type, small)
                    else |_| blk2: {
                        const text = try value_int.toString(self.parent.allocator, 10, .lower);
                        defer self.parent.allocator.free(text);
                        break :blk2 mlir.oraIntegerAttrGetFromString(result_type, strRef(text));
                    };
                    break :blk appendValueOp(self.block, mlir.oraArithConstantOpCreate(self.parent.context, loc, result_type, attr));
                },
                .boolean => |boolean| blk: {
                    if (!mlir.oraTypeIsAInteger(result_type)) break :blk null;
                    break :blk appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), if (boolean) 1 else 0));
                },
                .address => |address| blk: {
                    if (!mlir.oraTypeEqual(result_type, addressType(self.parent.context))) break :blk null;
                    const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
                    var decimal_buf: [80]u8 = undefined;
                    const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{address}) catch break :blk null;
                    const attr = mlir.oraIntegerAttrGetFromString(i160_type, strRef(decimal_text));
                    const bits = appendValueOp(self.block, mlir.oraArithConstantOpCreate(self.parent.context, loc, i160_type, attr));
                    const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, bits);
                    if (mlir.oraOperationIsNull(addr_op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, addr_op);
                },
                .string => |text| blk: {
                    if (!mlir.oraTypeEqual(result_type, stringType(self.parent.context))) break :blk null;
                    const op = mlir.oraStringConstantOpCreate(self.parent.context, loc, strRef(text), stringType(self.parent.context));
                    mlir.oraOperationSetAttributeByName(
                        op,
                        strRef("length"),
                        mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(self.parent.context, 32), @intCast(text.len)),
                    );
                    break :blk appendValueOp(self.block, op);
                },
                .tuple => |elements| blk: {
                    if (sema_type.kind() != .tuple) break :blk null;
                    const tuple_types = sema_type.tuple;
                    if (tuple_types.len != elements.len) break :blk null;
                    const tuple_result_type = self.parent.lowerSemaType(sema_type, source.TextRange.empty(0));
                    const values = try self.parent.allocator.alloc(mlir.MlirValue, elements.len);
                    defer self.parent.allocator.free(values);
                    for (elements, 0..) |element, index| {
                        const element_sema_type = tuple_types[index];
                        const element_type = try lowerTypeDescriptor(self.parent.context, element_sema_type, self.parent.allocator);
                        values[index] = (try @This().lowerPersistedConstValue(self, element, element_sema_type, element_type, loc)) orelse break :blk null;
                    }
                    const op = mlir.oraTupleCreateOpCreate(self.parent.context, loc, values.ptr, values.len, tuple_result_type);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
            };
        }

        pub fn exprDependsOnRuntimeState(self: *FunctionLowerer, expr_id: ast.ExprId) bool {
            return switch (self.parent.file.expression(expr_id).*) {
                .Name => blk: {
                    const binding = self.parent.resolution.expr_bindings[expr_id.index()] orelse break :blk true;
                    break :blk switch (binding) {
                        .pattern => false,
                        .item => |item_id| switch (self.parent.file.item(item_id).*) {
                            .Field => |field| field.storage_class != .none,
                            .Constant => false,
                            .Import => false,
                            else => true,
                        },
                    };
                },
                .Group => |group| @This().exprDependsOnRuntimeState(self, group.expr),
                .Binary => |binary| @This().exprDependsOnRuntimeState(self, binary.lhs) or @This().exprDependsOnRuntimeState(self, binary.rhs),
                .Unary => |unary| @This().exprDependsOnRuntimeState(self, unary.operand),
                .Comptime => |comptime_expr| blk: {
                    const body = self.parent.file.body(comptime_expr.body).*;
                    for (body.statements) |statement_id| {
                        switch (self.parent.file.statement(statement_id).*) {
                            .Expr => |expr_stmt| if (@This().exprDependsOnRuntimeState(self, expr_stmt.expr)) break :blk true,
                            .Return => |ret| if (ret.value) |value| if (@This().exprDependsOnRuntimeState(self, value)) break :blk true,
                            else => {},
                        }
                    }
                    break :blk false;
                },
                .Field => |field| blk: {
                    const path = @This().fieldExprPath(self, expr_id) catch break :blk true;
                    defer self.parent.allocator.free(path);
                    if (std.mem.startsWith(u8, path, "std.msg.") or
                        std.mem.startsWith(u8, path, "std.transaction.") or
                        std.mem.startsWith(u8, path, "std.tx.") or
                        std.mem.startsWith(u8, path, "std.block."))
                    {
                        break :blk true;
                    }
                    break :blk @This().exprDependsOnRuntimeState(self, field.base);
                },
                .Index => |index| @This().exprDependsOnRuntimeState(self, index.base) or @This().exprDependsOnRuntimeState(self, index.index),
                else => false,
            };
        }

        fn exprDependsOnMutablePatternLocals(self: *FunctionLowerer, expr_id: ast.ExprId) bool {
            return switch (self.parent.file.expression(expr_id).*) {
                .Name => blk: {
                    const binding = self.parent.resolution.expr_bindings[expr_id.index()] orelse break :blk false;
                    break :blk switch (binding) {
                        .pattern => |pattern_id| @This().patternIsMutableLocal(self, pattern_id) or @This().nameIsMutableLocal(self, self.parent.file.expression(expr_id).Name.name),
                        .item => false,
                    };
                },
                .Group => |group| @This().exprDependsOnMutablePatternLocals(self, group.expr),
                .Binary => |binary| @This().exprDependsOnMutablePatternLocals(self, binary.lhs) or @This().exprDependsOnMutablePatternLocals(self, binary.rhs),
                .Unary => |unary| @This().exprDependsOnMutablePatternLocals(self, unary.operand),
                .Field => |field| @This().exprDependsOnMutablePatternLocals(self, field.base),
                .Index => |index| @This().exprDependsOnMutablePatternLocals(self, index.base) or @This().exprDependsOnMutablePatternLocals(self, index.index),
                .Comptime => false,
                else => false,
            };
        }

        fn patternIsMutableLocal(self: *FunctionLowerer, pattern_id: ast.PatternId) bool {
            for (self.parent.file.bodies, 0..) |_, body_index| {
                const body = self.parent.file.body(ast.BodyId.fromIndex(@intCast(body_index))).*;
                for (body.statements) |statement_id| {
                    switch (self.parent.file.statement(statement_id).*) {
                        .VariableDecl => |decl| {
                            if (decl.pattern != pattern_id) continue;
                            return switch (decl.binding_kind) {
                                .let_, .var_ => true,
                                .constant, .immutable => false,
                            };
                        },
                        else => {},
                    }
                }
            }
            return false;
        }

        fn nameIsMutableLocal(self: *FunctionLowerer, name: []const u8) bool {
            for (self.parent.file.bodies, 0..) |_, body_index| {
                const body = self.parent.file.body(ast.BodyId.fromIndex(@intCast(body_index))).*;
                for (body.statements) |statement_id| {
                    switch (self.parent.file.statement(statement_id).*) {
                        .VariableDecl => |decl| {
                            if (self.parent.file.pattern(decl.pattern).* != .Name) continue;
                            if (!std.mem.eql(u8, self.parent.file.pattern(decl.pattern).Name.name, name)) continue;
                            return switch (decl.binding_kind) {
                                .let_, .var_ => true,
                                .constant, .immutable => false,
                            };
                        },
                        else => {},
                    }
                }
            }
            return false;
        }

        pub fn lowerExpr(self: *FunctionLowerer, expr_id: ast.ExprId, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const expr = self.parent.file.expression(expr_id).*;
            if (try @This().tryLowerConstEvalValue(self, expr_id)) |folded| {
                return folded;
            }
            const loc = self.parent.location(exprRange(self.parent.file, expr_id));
            return switch (expr) {
                .TypeValue => |type_value| blk: {
                    const placeholder = try self.createValuePlaceholder("ora.type_value", "type", type_value.range, self.parent.lowerExprType(expr_id));
                    break :blk appendValueOp(self.block, placeholder);
                },
                .IntegerLiteral => |literal| blk: {
                    const ty = self.parent.lowerExprType(expr_id);
                    if (mlir.oraTypeEqual(ty, addressType(self.parent.context))) {
                        break :blk try @This().lowerContextualAddressIntegerLiteral(self, literal);
                    }
                    const parsed = parseUnsignedIntegerLiteral(u256, literal.text) orelse
                        break :blk try @This().loweringValueError(self, literal.range, ty, "invalid integer literal '{s}'", .{literal.text});
                    break :blk try @This().createWideIntegerConstant(self, ty, parsed, false, literal.range);
                },
                .BoolLiteral => |literal| blk: {
                    break :blk appendValueOp(self.block, createIntegerConstant(self.parent.context, self.parent.location(literal.range), boolType(self.parent.context), if (literal.value) 1 else 0));
                },
                .StringLiteral => |literal| blk: {
                    const op = mlir.oraStringConstantOpCreate(
                        self.parent.context,
                        self.parent.location(literal.range),
                        strRef(literal.text),
                        stringType(self.parent.context),
                    );
                    mlir.oraOperationSetAttributeByName(
                        op,
                        strRef("length"),
                        mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(self.parent.context, 32), @intCast(literal.text.len)),
                    );
                    break :blk appendValueOp(self.block, op);
                },
                .AddressLiteral => |literal| blk: {
                    const trimmed = if (std.mem.startsWith(u8, literal.text, "0x")) literal.text[2..] else literal.text;
                    const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
                    const addr_attr = blk2: {
                        const null_attr = std.mem.zeroes(mlir.MlirAttribute);
                        const parsed = std.fmt.parseInt(u256, trimmed, 16) catch break :blk2 null_attr;
                        if (parsed <= std.math.maxInt(i64)) {
                            break :blk2 mlir.oraIntegerAttrCreateI64FromType(i160_type, @intCast(parsed));
                        }
                        var decimal_buf: [80]u8 = undefined;
                        const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch break :blk2 null_attr;
                        break :blk2 mlir.oraIntegerAttrGetFromString(i160_type, strRef(decimal_text));
                    };
                    const int_const = mlir.oraArithConstantOpCreate(self.parent.context, self.parent.location(literal.range), i160_type, addr_attr);
                    const i160_value = appendValueOp(self.block, int_const);
                    const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, self.parent.location(literal.range), i160_value);
                    break :blk appendValueOp(self.block, addr_op);
                },
                .BytesLiteral => |literal| blk: {
                    const expr_type = self.parent.typecheck.exprType(expr_id);
                    if (expr_type.kind() == .fixed_bytes and literal.text.len % 2 == 0 and literal.text.len / 2 == expr_type.fixed_bytes.len) {
                        const ty = self.parent.lowerExprType(expr_id);
                        if (mlir.oraTypeIsAInteger(ty)) {
                            if (std.fmt.parseInt(u256, literal.text, 16)) |parsed| {
                                const attr = if (parsed <= std.math.maxInt(i64))
                                    mlir.oraIntegerAttrCreateI64FromType(ty, @intCast(parsed))
                                else blk2: {
                                    var decimal_buf: [80]u8 = undefined;
                                    const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch break :blk2 mlir.oraNullAttrCreate();
                                    break :blk2 mlir.oraIntegerAttrGetFromString(ty, strRef(decimal_text));
                                };
                                break :blk appendValueOp(self.block, mlir.oraArithConstantOpCreate(self.parent.context, self.parent.location(literal.range), ty, attr));
                            } else |_| {}
                        }
                    }
                    const op = mlir.oraBytesConstantOpCreate(
                        self.parent.context,
                        self.parent.location(literal.range),
                        strRef(literal.text),
                        bytesType(self.parent.context),
                    );
                    mlir.oraOperationSetAttributeByName(
                        op,
                        strRef("length"),
                        mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(self.parent.context, 32), @intCast(literal.text.len / 2)),
                    );
                    break :blk appendValueOp(self.block, op);
                },
                .Name => |name| try self.lowerNameExpr(expr_id, name, locals),
                .Result => self.current_return_value orelse
                    try @This().loweringValueError(self, expr.Result.range, self.parent.lowerExprType(expr_id), "result is not available in this context", .{}),
                .Group => |group| try self.lowerExpr(group.expr, locals),
                .Old => |old| blk: {
                    const value = try self.lowerExpr(old.expr, locals);
                    const result_type = self.parent.lowerExprType(old.expr);
                    const op = mlir.oraOldOpCreate(self.parent.context, loc, value, result_type);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .Unary => |unary| try self.lowerUnary(unary, locals),
                .Binary => |binary| try self.lowerBinary(expr_id, binary, locals),
                .Call => |call| try self.lowerCall(expr_id, call, locals),
                .Builtin => |builtin| try self.lowerBuiltin(expr_id, builtin, locals),
                .Tuple => |tuple| blk: {
                    const sema_tuple_type = self.parent.typecheck.exprType(expr_id);
                    var operands: std.ArrayList(mlir.MlirValue) = .{};
                    for (tuple.elements, 0..) |element, index| {
                        const value = if (sema_tuple_type.kind() == .tuple and index < sema_tuple_type.tuple.len)
                            try @This().lowerExprForSemaFlowTarget(self, element, sema_tuple_type.tuple[index], exprRange(self.parent.file, element), locals)
                        else
                            try self.lowerExpr(element, locals);
                        try operands.append(self.parent.allocator, value);
                    }
                    const result_type = self.parent.lowerSemaType(sema_tuple_type, tuple.range);
                    const op = mlir.oraTupleCreateOpCreate(
                        self.parent.context,
                        self.parent.location(tuple.range),
                        if (operands.items.len == 0) null else operands.items.ptr,
                        operands.items.len,
                        result_type,
                    );
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .ArrayLiteral => |array| blk: {
                    break :blk try @This().lowerArrayLiteral(self, expr_id, array, locals);
                },
                .StructLiteral => |struct_literal| blk: {
                    break :blk try @This().lowerStructLiteral(self, expr_id, struct_literal, locals);
                },
                .Switch => |switch_expr| try self.lowerSwitchExpr(expr_id, switch_expr, locals),
                .ExternalProxy => |_| error.UnsupportedExternTraitLowering,
                .Comptime => {
                    return error.ComptimeValueRequiredForRuntimeLowering;
                },
                .ErrorReturn => |error_return| blk: {
                    var args: std.ArrayList(mlir.MlirValue) = .{};
                    for (error_return.args) |arg| try args.append(self.parent.allocator, try self.lowerExpr(arg, locals));
                    const result_type = self.return_type orelse self.parent.lowerExprType(expr_id);
                    const error_symbol_name = if (self.parent.item_index.lookup(error_return.name)) |item_id|
                        @This().errorDeclSymbolName(self, item_id, error_return.name)
                    else
                        error_return.name;
                    const op = mlir.oraErrorReturnOpCreate(
                        self.parent.context,
                        loc,
                        strRef(error_symbol_name),
                        if (args.items.len == 0) null else args.items.ptr,
                        args.items.len,
                        result_type,
                    );
                    if (!mlir.oraOperationIsNull(op)) {
                        if (args.items.len != 0 or (self.function != null and self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.body_types[self.function.?.body.index()]))) {
                            mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                        }
                        break :blk appendValueOp(self.block, op);
                    }
                    const placeholder = try self.createAggregatePlaceholder("ora.error.return", error_return.range, args.items, result_type);
                    break :blk appendValueOp(self.block, placeholder);
                },
                .Field => |field| blk: {
                    break :blk try @This().lowerFieldExpr(self, expr_id, field, locals);
                },
                .Index => |index| blk: {
                    const base = try self.lowerExpr(index.base, locals);
                    const base_type = self.parent.typecheck.exprType(index.base);
                    const base_mlir_type = mlir.oraValueGetType(base);
                    if (mlir.oraTypeIsAMemRef(base_mlir_type)) {
                        const key = try self.lowerExprForFlowTarget(index.index, defaultIntegerType(self.parent.context), locals);
                        const index_value = try @This().convertIndexToIndexType(self, key, index.range);
                        const result_type = self.parent.lowerExprType(expr_id);
                        const op = mlir.oraMemrefLoadOpCreate(
                            self.parent.context,
                            self.parent.location(index.range),
                            base,
                            &[_]mlir.MlirValue{index_value},
                            1,
                            result_type,
                        );
                        if (!mlir.oraOperationIsNull(op)) break :blk appendValueOp(self.block, op);
                    }
                    switch (base_type.kind()) {
                        .bytes, .string => {
                            const key = try self.lowerExprForFlowTarget(index.index, defaultIntegerType(self.parent.context), locals);
                            const raw_byte_type = defaultIntegerType(self.parent.context);
                            const op = mlir.oraByteAtOpCreate(
                                self.parent.context,
                                self.parent.location(index.range),
                                base,
                                key,
                                raw_byte_type,
                            );
                            if (!mlir.oraOperationIsNull(op)) {
                                const raw_byte = appendValueOp(self.block, op);
                                break :blk try self.convertValueForFlow(raw_byte, self.parent.lowerExprType(expr_id), index.range);
                            }
                        },
                        else => {},
                    }
                    if (base_type == .tuple) {
                        const tuple_index = @This().constTupleIndex(self, index.index) orelse {
                            const key = try self.lowerExpr(index.index, locals);
                            const op = try self.createAggregatePlaceholder("ora.index_access", index.range, &.{ base, key }, self.parent.lowerExprType(expr_id));
                            break :blk appendValueOp(self.block, op);
                        };
                        const result_type = self.parent.lowerExprType(expr_id);
                        const op = mlir.oraTupleExtractOpCreate(
                            self.parent.context,
                            self.parent.location(index.range),
                            base,
                            @intCast(tuple_index),
                            result_type,
                        );
                        if (!mlir.oraOperationIsNull(op)) break :blk appendValueOp(self.block, op);
                    }
                    if (base_type == .map) {
                        const result_type = self.parent.lowerExprType(expr_id);
                        const map_key_type = mlir.oraMapTypeGetKeyType(base_mlir_type);
                        const key = if (!mlir.oraTypeIsNull(map_key_type))
                            try self.lowerExprForFlowTarget(index.index, map_key_type, locals)
                        else
                            try self.lowerExpr(index.index, locals);
                        const converted_key = if (!mlir.oraTypeIsNull(map_key_type))
                            try self.convertValueForFlow(key, map_key_type, index.range)
                        else
                            key;
                        const op = mlir.oraMapGetOpCreate(self.parent.context, self.parent.location(index.range), base, converted_key, result_type);
                        if (!mlir.oraOperationIsNull(op)) break :blk appendValueOp(self.block, op);
                    }
                    const op = try self.createAggregatePlaceholder("ora.index_access", index.range, &.{}, self.parent.lowerExprType(expr_id));
                    break :blk appendValueOp(self.block, op);
                },
                .Quantified => |quantified| try @This().lowerQuantifiedExpr(self, quantified, loc, locals),
                .Error => try @This().loweringValueError(self, expr.Error.range, self.parent.lowerExprType(expr_id), "cannot lower syntax error expression", .{}),
            };
        }

        pub fn lowerExprForFlowTarget(self: *FunctionLowerer, expr_id: ast.ExprId, target_type: mlir.MlirType, locals: *LocalEnv) anyerror!mlir.MlirValue {
            return switch (self.parent.file.expression(expr_id).*) {
                .Group => |group| try @This().lowerExprForFlowTarget(self, group.expr, target_type, locals),
                .IntegerLiteral => |literal| blk: {
                    if (mlir.oraTypeEqual(target_type, addressType(self.parent.context))) {
                        break :blk try @This().lowerContextualAddressIntegerLiteral(self, literal);
                    }
                    if (mlir.oraTypeIsAInteger(target_type)) {
                        const parsed = parseUnsignedIntegerLiteral(u256, literal.text) orelse
                            break :blk try @This().loweringValueError(self, literal.range, target_type, "invalid integer literal '{s}'", .{literal.text});
                        break :blk try @This().createWideIntegerConstant(self, target_type, parsed, false, literal.range);
                    }
                    break :blk try self.lowerExpr(expr_id, locals);
                },
                else => try self.lowerExpr(expr_id, locals),
            };
        }

        fn lowerExprForSemaFlowTarget(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            target_sema_type: sema.Type,
            range: source.TextRange,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const target_type = self.parent.lowerSemaType(target_sema_type, range);
            const target_ref_base = mlir.oraRefinementTypeGetBaseType(target_type);
            const flow_target = if (!mlir.oraTypeIsNull(target_ref_base)) target_ref_base else target_type;
            const raw_value = try @This().lowerExprForFlowTarget(self, expr_id, flow_target, locals);
            return try self.convertValueForSemaFlow(raw_value, target_sema_type, range);
        }

        fn lowerContextualAddressIntegerLiteral(self: *FunctionLowerer, literal: ast.IntegerLiteralExpr) !mlir.MlirValue {
            const loc = self.parent.location(literal.range);
            const parsed = parseUnsignedIntegerLiteral(u160, literal.text) orelse {
                return try @This().loweringValueError(self, literal.range, addressType(self.parent.context), "address literal value is out of range", .{});
            };
            const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
            const attr = if (parsed <= std.math.maxInt(i64))
                mlir.oraIntegerAttrCreateI64FromType(i160_type, @intCast(parsed))
            else blk: {
                var decimal_buf: [80]u8 = undefined;
                const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch {
                    return try @This().loweringValueError(self, literal.range, addressType(self.parent.context), "address literal value is out of range", .{});
                };
                break :blk mlir.oraIntegerAttrGetFromString(i160_type, strRef(decimal_text));
            };
            const bits = appendValueOp(self.block, mlir.oraArithConstantOpCreate(self.parent.context, loc, i160_type, attr));
            const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, bits);
            if (mlir.oraOperationIsNull(addr_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, addr_op);
        }

        fn lowerQuantifiedExpr(
            self: *FunctionLowerer,
            quantified: ast.QuantifiedExpr,
            loc: mlir.MlirLocation,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const pattern = self.parent.file.pattern(quantified.pattern).*;
            const variable_name = switch (pattern) {
                .Name => |name| name.name,
                else => return try @This().loweringValueError(
                    self,
                    quantified.range,
                    boolType(self.parent.context),
                    "quantified expression requires a named bound variable",
                    .{},
                ),
            };

            const variable_sema_type = self.parent.typecheck.pattern_types[quantified.pattern.index()].type;
            const variable_type = self.parent.lowerSemaType(variable_sema_type, quantified.range);
            if (mlir.oraTypeIsNull(variable_type)) {
                return try @This().loweringValueError(
                    self,
                    quantified.range,
                    boolType(self.parent.context),
                    "cannot lower quantified variable type",
                    .{},
                );
            }

            const placeholder_op = createIntegerConstant(self.parent.context, loc, variable_type, 0);
            _ = try @This().expectOperation(self, placeholder_op, quantified.range, "arith.constant");
            mlir.oraOperationSetAttributeByName(
                placeholder_op,
                strRef("ora.bound_variable"),
                mlir.oraStringAttrCreate(self.parent.context, strRef(variable_name)),
            );
            const placeholder = appendValueOp(self.block, placeholder_op);

            var quantified_locals = try locals.clone();
            try self.bindPatternValue(quantified.pattern, placeholder, &quantified_locals);

            const condition_value = if (quantified.condition) |condition|
                try self.lowerExpr(condition, &quantified_locals)
            else
                null;
            const body_value = try self.lowerExpr(quantified.body, &quantified_locals);

            const variable_type_text = try @This().quantifiedVariableTypeText(self, variable_sema_type);
            const condition_operand = if (condition_value) |value| value else std.mem.zeroes(mlir.MlirValue);
            const op = mlir.oraQuantifiedOpCreate(
                self.parent.context,
                loc,
                strRef(switch (quantified.quantifier) {
                    .forall => "forall",
                    .exists => "exists",
                }),
                strRef(variable_name),
                strRef(variable_type_text),
                condition_operand,
                condition_value != null,
                body_value,
                boolType(self.parent.context),
            );
            return try @This().expectValueOp(self, op, quantified.range, "ora.quantified");
        }

        fn quantifiedVariableTypeText(self: *FunctionLowerer, ty: sema.Type) ![]const u8 {
            if (ty.name()) |name| return name;
            var buffer: std.ArrayList(u8) = .{};
            try sema.appendTypeMangleName(self.parent.allocator, &buffer, ty);
            return try buffer.toOwnedSlice(self.parent.allocator);
        }

        pub fn lowerNameExpr(self: *FunctionLowerer, expr_id: ast.ExprId, name: ast.NameExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (self.parent.resolution.expr_bindings[expr_id.index()]) |binding| {
                switch (binding) {
                    .pattern => |pattern_id| {
                        if (locals.getValue(pattern_id)) |value| return value;
                    },
                    .item => |item_id| if (try @This().lowerBoundItemExpr(self, expr_id, name, item_id, locals)) |lowered| return lowered,
                }
            }

            if (locals.lookupName(name.name)) |local_id| {
                if (locals.getValue(local_id)) |value| return value;
            }

            if (std.mem.eql(u8, name.name, "result")) {
                if (self.current_return_value) |value| return value;
            }

            if (try @This().lowerCurrentContractMemberByName(self, expr_id, name, locals)) |lowered| {
                return lowered;
            }

            if (self.parent.item_index.lookup(name.name)) |item_id| {
                if (try @This().lowerBoundItemExpr(self, expr_id, name, item_id, locals)) |lowered| return lowered;
            }

            for (self.parent.file.items, 0..) |_, idx| {
                const item_id: ast.ItemId = @enumFromInt(idx);
                switch (self.parent.file.item(item_id).*) {
                    .Field => |field| {
                        if (!std.mem.eql(u8, field.name, name.name)) continue;
                        if (try @This().lowerBoundItemExpr(self, expr_id, name, item_id, locals)) |lowered| return lowered;
                    },
                    .Constant => |constant| {
                        if (!std.mem.eql(u8, constant.name, name.name)) continue;
                        if (try @This().lowerBoundItemExpr(self, expr_id, name, item_id, locals)) |lowered| return lowered;
                    },
                    .Function => |function_item| {
                        if (!std.mem.eql(u8, function_item.name, name.name)) continue;
                        if (try @This().lowerBoundItemExpr(self, expr_id, name, item_id, locals)) |lowered| return lowered;
                    },
                    else => {},
                }
            }

            return try @This().loweringValueError(self, name.range, self.parent.lowerExprType(expr_id), "unresolved name '{s}' during HIR lowering", .{name.name});
        }

        fn lowerBoundItemExpr(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            name: ast.NameExpr,
            item_id: ast.ItemId,
            locals: *LocalEnv,
        ) anyerror!?mlir.MlirValue {
            const item = self.parent.file.item(item_id).*;
            switch (item) {
                .Field => |field| {
                    if (field.binding_kind == .immutable and field.storage_class == .none) {
                        if (field.value) |value| return try self.lowerExpr(value, locals);
                    }
                    const result_type = if (field.type_expr) |_|
                        self.parent.lowerSemaType(self.parent.typecheck.item_types[item_id.index()], field.range)
                    else
                        self.parent.lowerExprType(expr_id);
                    const op = switch (field.storage_class) {
                        .storage => mlir.oraSLoadOpCreate(
                            self.parent.context,
                            self.parent.location(field.range),
                            strRef(field.name),
                            result_type,
                        ),
                        .memory => mlir.oraMLoadOpCreate(self.parent.context, self.parent.location(field.range), strRef(field.name), result_type),
                        .tstore => mlir.oraTLoadOpCreate(self.parent.context, self.parent.location(field.range), strRef(field.name), result_type),
                        .none => std.mem.zeroes(mlir.MlirOperation),
                    };
                    if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                },
                .Constant => |constant| return try self.lowerExpr(constant.value, locals),
                .Function => |function| {
                    const result_type = self.parent.lowerExprType(expr_id);
                    const op = mlir.oraFunctionRefOpCreate(
                        self.parent.context,
                        self.parent.location(name.range),
                        strRef(function.name),
                        result_type,
                    );
                    if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                    const placeholder = try self.createAggregatePlaceholder("ora.function_ref", name.range, &.{}, result_type);
                    return appendValueOp(self.block, placeholder);
                },
                .ErrorDecl => |error_decl| {
                    if (error_decl.parameters.len == 0) {
                        const result_type = self.return_type orelse self.parent.lowerExprType(expr_id);
                        const op = mlir.oraErrorReturnOpCreate(
                            self.parent.context,
                            self.parent.location(name.range),
                            strRef(@This().errorDeclSymbolName(self, item_id, error_decl.name)),
                            null,
                            0,
                            result_type,
                        );
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        return appendValueOp(self.block, op);
                    }
                },
                else => {},
            }
            return null;
        }

        fn lowerCurrentContractMemberByName(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            name: ast.NameExpr,
            locals: *LocalEnv,
        ) anyerror!?mlir.MlirValue {
            const function = self.function orelse return null;
            const contract_id = function.parent_contract orelse return null;
            const member_id = self.parent.item_index.lookupContractMemberWithRoles(self.parent.file, contract_id, name.name, .{
                .field = true,
                .constant = true,
                .function = true,
            }) orelse return null;
            return try @This().lowerBoundItemExpr(self, expr_id, name, member_id, locals);
        }

        pub fn lowerUnary(self: *FunctionLowerer, unary: ast.UnaryExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const operand = try self.lowerExpr(unary.operand, locals);
            const loc = self.parent.location(unary.range);
            return switch (unary.op) {
                .neg => blk: {
                    const zero = appendValueOp(
                        self.block,
                        createIntegerConstant(self.parent.context, loc, mlir.oraValueGetType(operand), 0),
                    );
                    const op = mlir.oraArithSubIOpCreate(self.parent.context, loc, zero, operand);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .not_ => blk: {
                    const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                    const op = mlir.oraArithXorIOpCreate(self.parent.context, loc, operand, one);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .bit_not => blk: {
                    const operand_ty = mlir.oraValueGetType(operand);
                    const bit_width: u32 = if (mlir.oraTypeIsAInteger(operand_ty))
                        @intCast(mlir.oraIntegerTypeGetWidth(operand_ty))
                    else
                        256;
                    const all_ones: i64 = if (bit_width >= 64) -1 else (@as(i64, 1) << @intCast(bit_width)) - 1;
                    const mask = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, operand_ty, all_ones));
                    const op = mlir.oraArithXorIOpCreate(self.parent.context, loc, operand, mask);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .try_ => try @This().lowerTryUnary(self, operand, unary.range),
            };
        }

        fn lowerTryUnary(self: *FunctionLowerer, operand: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const operand_type = mlir.oraValueGetType(operand);
            const result_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
            if (mlir.oraTypeIsNull(result_type)) return operand;

            if (self.in_try_block) {
                const unwrap = mlir.oraErrorUnwrapOpCreate(self.parent.context, loc, operand, result_type);
                if (mlir.oraOperationIsNull(unwrap)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, unwrap);
            }

            if (!self.in_try_block) {
                if (self.return_type) |return_type| {
                    if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(return_type))) {
                        const is_error = mlir.oraErrorIsErrorOpCreate(self.parent.context, loc, operand);
                        if (mlir.oraOperationIsNull(is_error)) return error.MlirOperationCreationFailed;
                        const is_error_value = appendValueOp(self.block, is_error);

                        const branch = mlir.oraConditionalReturnOpCreate(self.parent.context, loc, is_error_value);
                        if (mlir.oraOperationIsNull(branch)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, branch);

                        const then_block = mlir.oraConditionalReturnOpGetThenBlock(branch);
                        const else_block = mlir.oraConditionalReturnOpGetElseBlock(branch);
                        if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                            return error.MlirOperationCreationFailed;
                        }

                        const error_type = defaultIntegerType(self.parent.context);
                        const get_error = mlir.oraErrorGetErrorOpCreate(self.parent.context, loc, operand, error_type);
                        if (mlir.oraOperationIsNull(get_error)) return error.MlirOperationCreationFailed;
                        appendOp(then_block, get_error);
                        const error_value = mlir.oraOperationGetResult(get_error, 0);

                        const error_union = mlir.oraErrorErrOpCreate(self.parent.context, loc, error_value, return_type);
                        if (mlir.oraOperationIsNull(error_union)) return error.MlirOperationCreationFailed;
                        if (self.function != null and self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.body_types[self.function.?.body.index()])) {
                            mlir.oraOperationSetAttributeByName(error_union, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                        }
                        appendOp(then_block, error_union);
                        const error_union_value = mlir.oraOperationGetResult(error_union, 0);

                        const ret = mlir.oraReturnOpCreate(self.parent.context, loc, &[_]mlir.MlirValue{error_union_value}, 1);
                        if (mlir.oraOperationIsNull(ret)) return error.MlirOperationCreationFailed;
                        appendOp(then_block, ret);

                        try support.appendEmptyYield(self.parent.context, else_block, loc);

                        const unwrap = mlir.oraErrorUnwrapOpCreate(self.parent.context, loc, operand, result_type);
                        if (mlir.oraOperationIsNull(unwrap)) return error.MlirOperationCreationFailed;
                        return appendValueOp(self.block, unwrap);
                    }
                }
            }

            const placeholder = try self.createAggregatePlaceholder("ora.try_expr", range, &[_]mlir.MlirValue{operand}, result_type);
            return appendValueOp(self.block, placeholder);
        }

        pub fn lowerBinary(self: *FunctionLowerer, expr_id: ast.ExprId, binary: ast.BinaryExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (binary.op == .and_and or binary.op == .or_or) {
                return try @This().lowerShortCircuitLogical(self, expr_id, binary, locals);
            }

            var lhs = try self.lowerExpr(binary.lhs, locals);
            var rhs = switch (binary.op) {
                .shl, .shr, .wrapping_shl, .wrapping_shr => blk: {
                    const lhs_type = mlir.oraValueGetType(lhs);
                    if (mlir.oraTypeIsAInteger(lhs_type)) {
                        break :blk try self.lowerExprForFlowTarget(binary.rhs, lhs_type, locals);
                    }
                    break :blk try self.lowerExpr(binary.rhs, locals);
                },
                else => try self.lowerExpr(binary.rhs, locals),
            };
            const loc = self.parent.location(binary.range);
            const result_type = self.parent.lowerExprType(expr_id);

            lhs = try @This().unwrapRefinementForCast(self, lhs, binary.range);
            rhs = try @This().unwrapRefinementForCast(self, rhs, binary.range);

            switch (binary.op) {
                .eq, .ne, .lt, .le, .gt, .ge => {
                    lhs = try @This().normalizeCompareOperand(self, lhs, binary.lhs);
                    rhs = try @This().normalizeCompareOperand(self, rhs, binary.rhs);
                    const lhs_type = mlir.oraValueGetType(lhs);
                    const rhs_type = mlir.oraValueGetType(rhs);
                    if (mlir.oraTypeIsAddressType(lhs_type) and !mlir.oraTypeIsAddressType(rhs_type)) {
                        rhs = try @This().convertCompareOperandToAddress(self, rhs, binary.rhs);
                    } else if (mlir.oraTypeIsAddressType(rhs_type) and !mlir.oraTypeIsAddressType(lhs_type)) {
                        lhs = try @This().convertCompareOperandToAddress(self, lhs, binary.lhs);
                    } else if (mlir.oraTypeIsAInteger(lhs_type) and mlir.oraTypeIsAInteger(rhs_type) and !mlir.oraTypeEqual(lhs_type, rhs_type)) {
                        switch (self.parent.file.expression(binary.lhs).*) {
                            .IntegerLiteral => lhs = try self.convertValueForFlow(lhs, rhs_type, binary.range),
                            else => switch (self.parent.file.expression(binary.rhs).*) {
                                .IntegerLiteral => rhs = try self.convertValueForFlow(rhs, lhs_type, binary.range),
                                else => {},
                            },
                        }
                    }
                },
                .add, .sub, .mul, .div, .mod, .bit_and, .bit_or, .bit_xor, .shl, .shr => {
                    const lhs_type = mlir.oraValueGetType(lhs);
                    const rhs_type = mlir.oraValueGetType(rhs);
                    if (mlir.oraTypeIsAInteger(lhs_type) and mlir.oraTypeIsAInteger(rhs_type) and !mlir.oraTypeEqual(lhs_type, rhs_type)) {
                        if (mlir.oraTypeIsAInteger(result_type)) {
                            lhs = try self.convertValueForFlow(lhs, result_type, binary.range);
                            rhs = try self.convertValueForFlow(rhs, result_type, binary.range);
                        } else {
                            switch (self.parent.file.expression(binary.lhs).*) {
                                .IntegerLiteral => lhs = try self.convertValueForFlow(lhs, rhs_type, binary.range),
                                else => switch (self.parent.file.expression(binary.rhs).*) {
                                    .IntegerLiteral => rhs = try self.convertValueForFlow(rhs, lhs_type, binary.range),
                                    else => {},
                                },
                            }
                        }
                    }
                },
                else => {},
            }

            if (binary.op == .pow) {
                return self.lowerCheckedPower(lhs, rhs, result_type, try @This().binaryIntegerExprIsSigned(self, expr_id, binary.lhs, binary.rhs), binary.range);
            }

            if (binary.op == .add) {
                const result_kind = self.parent.typecheck.exprType(expr_id).kind();
                if (result_kind == .string or result_kind == .bytes) {
                    try @This().emitConcatBoundsAssert(self, lhs, rhs, binary.range);
                    const op = mlir.oraConcatOpCreate(self.parent.context, loc, lhs, rhs, result_type);
                    return try @This().expectValueOp(self, op, binary.range, "ora.concat");
                }
            }

            switch (binary.op) {
                .wrapping_add, .wrapping_sub, .wrapping_mul, .wrapping_shl, .wrapping_shr, .wrapping_pow => {
                    lhs = try self.convertValueForFlow(lhs, result_type, binary.range);
                    rhs = try self.convertValueForFlow(rhs, result_type, binary.range);
                },
                .shl, .shr => {
                    const lhs_type = mlir.oraValueGetType(lhs);
                    const rhs_type = mlir.oraValueGetType(rhs);
                    if (mlir.oraTypeIsAInteger(lhs_type) and mlir.oraTypeIsAInteger(rhs_type) and !mlir.oraTypeEqual(lhs_type, rhs_type)) {
                        rhs = try self.convertValueForFlow(rhs, lhs_type, binary.range);
                    }
                },
                .eq, .ne, .lt, .le, .gt, .ge => {
                    const lhs_type = mlir.oraValueGetType(lhs);
                    const rhs_type = mlir.oraValueGetType(rhs);
                    if (mlir.oraTypeIsAInteger(lhs_type) and mlir.oraTypeIsAInteger(rhs_type) and !mlir.oraTypeEqual(lhs_type, rhs_type)) {
                        const lhs_expr = self.parent.file.expression(binary.lhs).*;
                        const rhs_expr = self.parent.file.expression(binary.rhs).*;
                        if (lhs_expr == .IntegerLiteral and rhs_expr != .IntegerLiteral) {
                            lhs = try self.convertValueForFlow(lhs, rhs_type, binary.range);
                        } else if (rhs_expr == .IntegerLiteral and lhs_expr != .IntegerLiteral) {
                            rhs = try self.convertValueForFlow(rhs, lhs_type, binary.range);
                        } else {
                            const lhs_bits = mlir.oraIntegerTypeGetWidth(lhs_type);
                            const rhs_bits = mlir.oraIntegerTypeGetWidth(rhs_type);
                            const common_type = if (lhs_bits >= rhs_bits) lhs_type else rhs_type;
                            lhs = try self.convertValueForFlow(lhs, common_type, binary.range);
                            rhs = try self.convertValueForFlow(rhs, common_type, binary.range);
                        }
                    }
                },
                else => {},
            }

            const is_signed_int_op = switch (binary.op) {
                .add, .sub, .mul, .div, .mod, .shr => try @This().binaryIntegerExprIsSigned(self, expr_id, binary.lhs, binary.rhs),
                .lt, .le, .gt, .ge => if (mlir.oraTypeIsAInteger(mlir.oraValueGetType(lhs)) and mlir.oraTypeIsAInteger(mlir.oraValueGetType(rhs)))
                    try @This().binaryIntegerExprIsSigned(self, expr_id, binary.lhs, binary.rhs)
                else
                    false,
                else => false,
            };

            const op = switch (binary.op) {
                .add => mlir.oraArithAddIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_add => mlir.oraAddWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .sub => mlir.oraArithSubIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_sub => mlir.oraSubWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .mul => mlir.oraArithMulIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_mul => mlir.oraMulWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .div => if (is_signed_int_op) mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithDivUIOpCreate(self.parent.context, loc, lhs, rhs),
                .mod => if (is_signed_int_op) mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithRemUIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_pow => mlir.oraPowerOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .bit_and, .and_and => mlir.oraArithAndIOpCreate(self.parent.context, loc, lhs, rhs),
                .bit_or, .or_or => mlir.oraArithOrIOpCreate(self.parent.context, loc, lhs, rhs),
                .bit_xor => mlir.oraArithXorIOpCreate(self.parent.context, loc, lhs, rhs),
                .shl => mlir.oraArithShlIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_shl => mlir.oraShlWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .shr => if (is_signed_int_op) mlir.oraArithShrSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithShrUIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_shr => mlir.oraShrWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .eq => self.createCompareOp(loc, "eq", lhs, rhs),
                .ne => self.createCompareOp(loc, "ne", lhs, rhs),
                .lt, .le, .gt, .ge => self.createCompareOp(loc, @This().predicateForBinaryCompare(binary.op, is_signed_int_op), lhs, rhs),
                .pow => unreachable,
            };

            if (mlir.oraOperationIsNull(op)) return try @This().loweringValueError(
                self,
                binary.range,
                result_type,
                "failed to lower binary operator '{s}'",
                .{@tagName(binary.op)},
            );
            const value = appendValueOp(self.block, op);
            try @This().maybeEmitCheckedBinaryOverflowAssert(self, binary.op, lhs, rhs, value, is_signed_int_op, binary.range);
            return value;
        }

        pub fn maybeEmitCheckedBinaryOverflowAssert(
            self: *FunctionLowerer,
            op: ast.BinaryOp,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            result: mlir.MlirValue,
            is_signed: bool,
            range: source.TextRange,
        ) anyerror!void {
            const value_ty = mlir.oraValueGetType(result);
            if (!mlir.oraTypeIsAInteger(value_ty)) return;

            const loc = self.parent.location(range);
            const overflow_flag = switch (op) {
                .add => if (is_signed)
                    try @This().computeSignedAddOverflow(self, result, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", result, lhs)),
                .sub => if (is_signed)
                    try @This().computeSignedSubOverflow(self, result, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", lhs, rhs)),
                .mul => if (is_signed)
                    try @This().computeSignedMulOverflow(self, result, lhs, rhs, value_ty, loc)
                else
                    try @This().computeUnsignedMulOverflow(self, result, lhs, rhs, value_ty, loc),
                else => return,
            };

            const message = switch (op) {
                .add => "checked addition overflow",
                .sub => "checked subtraction overflow",
                .mul => "checked multiplication overflow",
                else => unreachable,
            };
            try FunctionLowerer.emitOverflowAssert(self, overflow_flag, message, range);
        }

        fn lowerShortCircuitLogical(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            binary: ast.BinaryExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(exprRange(self.parent.file, expr_id));
            const result_type = boolType(self.parent.context);

            var lhs = try self.lowerExpr(binary.lhs, locals);
            lhs = try self.convertValueForFlow(lhs, result_type, binary.range);

            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, lhs, &[_]mlir.MlirType{result_type}, 1, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                return error.MlirOperationCreationFailed;
            }

            if (binary.op == .and_and) {
                var then_lowerer = self.*;
                then_lowerer.block = then_block;
                var rhs = try then_lowerer.lowerExpr(binary.rhs, locals);
                rhs = try then_lowerer.convertValueForFlow(rhs, result_type, binary.range);
                try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{rhs});

                const false_value = appendValueOp(else_block, createIntegerConstant(self.parent.context, loc, result_type, 0));
                try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{false_value});
            } else {
                const true_value = appendValueOp(then_block, createIntegerConstant(self.parent.context, loc, result_type, 1));
                try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{true_value});

                var else_lowerer = self.*;
                else_lowerer.block = else_block;
                var rhs = try else_lowerer.lowerExpr(binary.rhs, locals);
                rhs = try else_lowerer.convertValueForFlow(rhs, result_type, binary.range);
                try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{rhs});
            }

            return mlir.oraOperationGetResult(if_op, 0);
        }

        pub fn createCompareOp(self: *FunctionLowerer, loc: mlir.MlirLocation, predicate: []const u8, lhs: mlir.MlirValue, rhs: mlir.MlirValue) mlir.MlirOperation {
            const lhs_type = mlir.oraValueGetType(lhs);
            const rhs_type = mlir.oraValueGetType(rhs);
            const string_ty = stringType(self.parent.context);
            const bytes_ty = bytesType(self.parent.context);
            if (mlir.oraTypeIsAddressType(lhs_type) or
                mlir.oraTypeIsAddressType(rhs_type) or
                mlir.oraTypeEqual(lhs_type, string_ty) or
                mlir.oraTypeEqual(rhs_type, string_ty) or
                mlir.oraTypeEqual(lhs_type, bytes_ty) or
                mlir.oraTypeEqual(rhs_type, bytes_ty))
            {
                return mlir.oraCmpOpCreate(self.parent.context, loc, strRef(predicate), lhs, rhs, boolType(self.parent.context));
            }
            return mlir.oraArithCmpIOpCreate(self.parent.context, loc, cmpPredicate(predicate), lhs, rhs);
        }

        fn normalizeCompareOperand(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            expr_id: ast.ExprId,
        ) anyerror!mlir.MlirValue {
            const expr = self.parent.file.expression(expr_id).*;
            return switch (expr) {
                .IntegerLiteral => |literal| if (std.mem.eql(u8, std.mem.trim(u8, literal.text, " \t\n\r"), "0"))
                    appendValueOp(
                        self.block,
                        createIntegerConstant(self.parent.context, self.parent.location(literal.range), defaultIntegerType(self.parent.context), 0),
                    )
                else
                    value,
                else => value,
            };
        }

        fn convertCompareOperandToAddress(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            expr_id: ast.ExprId,
        ) anyerror!mlir.MlirValue {
            const expr = self.parent.file.expression(expr_id).*;
            switch (expr) {
                .IntegerLiteral => |literal| {
                    if (std.mem.eql(u8, std.mem.trim(u8, literal.text, " \t\n\r"), "0")) {
                        const zero_i160 = appendValueOp(self.block, createIntegerConstant(self.parent.context, self.parent.location(literal.range), mlir.oraIntegerTypeCreate(self.parent.context, 160), 0));
                        const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, self.parent.location(literal.range), zero_i160);
                        if (mlir.oraOperationIsNull(addr_op)) return error.MlirOperationCreationFailed;
                        return appendValueOp(self.block, addr_op);
                    }
                },
                else => {},
            }
            return self.convertValueForFlow(value, addressType(self.parent.context), exprRange(self.parent.file, expr_id));
        }

        pub fn lowerCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (try @This().lowerResultBuiltinCall(self, expr_id, call, locals)) |value| return value;
            if (call.args.len == 0) {
                const callee_expr = self.parent.file.expression(call.callee).*;
                if (callee_expr == .Field) {
                    const path = try @This().fieldExprPath(self, call.callee);
                    defer self.parent.allocator.free(path);
                    if (try @This().lowerBuiltinFieldCall(self, callee_expr.Field, path)) |value| {
                        return value;
                    }
                }
            }
            if (try @This().lowerExternProxyMethodCall(self, expr_id, call, locals)) |value| return value;
            if (try @This().lowerCurrentFunctionCallResult(self, call)) |value| return value;
            if (try @This().lowerCurrentMethodSelfCallResult(self, call)) |value| return value;
            if (try @This().lowerTraitBoundMethodCall(self, expr_id, call, locals)) |value| return value;
            if (try @This().lowerAssociatedImplMethodCall(self, expr_id, call, locals)) |value| return value;
            if (try @This().lowerErrorDeclCall(self, expr_id, call, locals)) |value| return value;
            if (try @This().lowerResultConstructorCall(self, expr_id, call, locals)) |value| return value;
            if (try @This().lowerAdtConstructorCall(self, expr_id, call, locals)) |value| return value;

            var args: std.ArrayList(mlir.MlirValue) = .{};
            const callee_item_id = @This().calleeFunctionItemId(self, call.callee);
            const imported_resolution = self.parent.typecheck.exprCallResolution(expr_id);
            const imported_target = if (imported_resolution) |resolved|
                try @This().importedFunctionTargetFromResolution(self, resolved)
            else
                try @This().calleeImportedFunctionTarget(self, call.callee);
            const callee_function = if (callee_item_id) |item_id| switch (self.parent.file.item(item_id).*) {
                .Function => |function| function,
                else => null,
            } else if (imported_target) |target|
                target.function
            else
                null;
            const runtime_args = if (callee_function) |function|
                self.parent.stripGenericCallArgs(function, call)
            else
                call.args;
            const runtime_parameters = if (callee_function) |function|
                if (imported_target) |target| blk: {
                    _ = target;
                    break :blk try self.parent.runtimeFunctionParameters(function);
                } else try self.parent.runtimeFunctionParameters(function)
            else
                &.{};
            for (runtime_args, 0..) |arg, index| {
                var arg_value = try self.lowerExpr(arg, locals);
                if (index < runtime_parameters.len) {
                    const parameter = runtime_parameters[index];
                    if (callee_function != null) {
                        const target_sema_type = if (imported_resolution) |resolved|
                            if (imported_target != null and index < resolved.runtime_parameter_types.len)
                                resolved.runtime_parameter_types[index]
                            else
                                try self.parent.resolvedRuntimeParameterTypeForCall(callee_function.?, parameter, call)
                        else
                            try self.parent.resolvedRuntimeParameterTypeForCall(callee_function.?, parameter, call);
                        const target_type = self.parent.lowerSemaType(target_sema_type, parameter.range);
                        arg_value = try self.convertValueForFlow(arg_value, target_type, exprRange(self.parent.file, arg));
                    }
                }
                try args.append(self.parent.allocator, arg_value);
            }

            const result_type = self.parent.lowerExprType(expr_id);
            const callee_name = if (imported_target != null and callee_function != null)
                (try self.parent.ensureImportedFunctionSymbol(
                    imported_target.?.module_id,
                    imported_target.?.item_id,
                    callee_function.?,
                    call,
                    imported_resolution,
                    @This().currentContractParentBlock(self) orelse self.parent.module_body,
                )) orelse return try @This().loweringValueError(self, call.range, result_type, "failed to resolve imported call target", .{})
            else if (callee_function != null and callee_item_id != null)
                (try self.parent.ensureMonomorphizedFunction(
                    callee_item_id.?,
                    callee_function.?,
                    call,
                    runtime_parameters,
                    @This().currentContractItemId(self),
                )) orelse return try @This().loweringValueError(self, call.range, result_type, "failed to resolve function call target", .{})
            else
                (@This().calleeName(self, call.callee) orelse return try @This().loweringValueError(self, call.range, result_type, "failed to resolve function call target", .{}));

            var result_types: [1]mlir.MlirType = .{result_type};
            const op = mlir.oraFuncCallOpCreate(
                self.parent.context,
                self.parent.location(call.range),
                strRef(callee_name),
                if (args.items.len == 0) null else args.items.ptr,
                args.items.len,
                if (self.typeIsVoid(result_type)) null else &result_types,
                if (self.typeIsVoid(result_type)) 0 else 1,
            );
            _ = try @This().expectOperation(self, op, call.range, "ora.call");
            if (imported_target != null) {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.imported_call"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try @This().voidValue(self, call.range);
            }
            return appendValueOp(self.block, op);
        }

        fn lowerErrorDeclCall(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            call: ast.CallExpr,
            locals: *LocalEnv,
        ) anyerror!?mlir.MlirValue {
            const callee_name = @This().calleeName(self, call.callee) orelse return null;
            const item_id = self.parent.item_index.lookup(callee_name) orelse return null;
            if (self.parent.file.item(item_id).* != .ErrorDecl) return null;
            const error_decl = self.parent.file.item(item_id).ErrorDecl;

            var args: std.ArrayList(mlir.MlirValue) = .{};
            for (call.args) |arg| try args.append(self.parent.allocator, try self.lowerExpr(arg, locals));

            const result_type = self.parent.lowerExprType(expr_id);
            const error_symbol_name = @This().errorDeclSymbolName(self, item_id, error_decl.name);
            const op = mlir.oraErrorReturnOpCreate(
                self.parent.context,
                self.parent.location(call.range),
                strRef(error_symbol_name),
                if (args.items.len == 0) null else args.items.ptr,
                args.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            if (args.items.len != 0 or
                (self.function != null and self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.body_types[self.function.?.body.index()])))
            {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            return appendValueOp(self.block, op);
        }

        fn lowerResultConstructorCall(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            call: ast.CallExpr,
            locals: *LocalEnv,
        ) anyerror!?mlir.MlirValue {
            const callee_name = @This().calleeName(self, call.callee) orelse return null;
            if (!std.mem.eql(u8, callee_name, "Ok") and !std.mem.eql(u8, callee_name, "Err")) return null;
            if (call.args.len != 1) return null;

            const result_sema_type = self.parent.typecheck.exprType(expr_id);
            if (result_sema_type.kind() != .error_union) return null;

            const loc = self.parent.location(call.range);
            const result_type = self.parent.lowerExprType(expr_id);

            if (std.mem.eql(u8, callee_name, "Ok")) {
                const operand = try self.lowerExpr(call.args[0], locals);
                const ok_op = mlir.oraErrorOkOpCreate(self.parent.context, loc, operand, result_type);
                if (mlir.oraOperationIsNull(ok_op)) return error.MlirOperationCreationFailed;
                if (self.parent.errorUnionRequiresWideCarrier(result_sema_type)) {
                    mlir.oraOperationSetAttributeByName(ok_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                }
                return appendValueOp(self.block, ok_op);
            }

            if (self.parent.file.expression(call.args[0]).* == .Call) {
                const inner_call = self.parent.file.expression(call.args[0]).Call;
                const inner_name = @This().calleeName(self, inner_call.callee);
                if (inner_name) |name| {
                    if (self.parent.item_index.lookup(name)) |item_id| {
                        if (self.parent.file.item(item_id).* == .ErrorDecl) {
                            const error_decl = self.parent.file.item(item_id).ErrorDecl;
                            var args: std.ArrayList(mlir.MlirValue) = .{};
                            for (inner_call.args) |arg| try args.append(self.parent.allocator, try self.lowerExpr(arg, locals));

                            const op = mlir.oraErrorReturnOpCreate(
                                self.parent.context,
                                self.parent.location(inner_call.range),
                                strRef(@This().errorDeclSymbolName(self, item_id, error_decl.name)),
                                if (args.items.len == 0) null else args.items.ptr,
                                args.items.len,
                                result_type,
                            );
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            if (args.items.len != 0 or self.parent.errorUnionRequiresWideCarrier(result_sema_type)) {
                                mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                            }
                            return appendValueOp(self.block, op);
                        }
                    }
                }
            }

            const operand = try self.lowerExpr(call.args[0], locals);
            if (mlir.oraTypeEqual(mlir.oraValueGetType(operand), result_type)) {
                return operand;
            }

            const err_op = mlir.oraErrorErrOpCreate(self.parent.context, loc, operand, result_type);
            if (mlir.oraOperationIsNull(err_op)) return error.MlirOperationCreationFailed;
            if (self.parent.errorUnionRequiresWideCarrier(result_sema_type)) {
                mlir.oraOperationSetAttributeByName(err_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            return appendValueOp(self.block, err_op);
        }

        fn lowerAdtConstructorCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!?mlir.MlirValue {
            const field = switch (self.parent.file.expression(call.callee).*) {
                .Group => |group| switch (self.parent.file.expression(group.expr).*) {
                    .Field => |field| field,
                    else => return null,
                },
                .Field => |field| field,
                else => return null,
            };
            const result_type = self.parent.lowerExprType(expr_id);
            if (!mlir.oraTypeIsAAdt(result_type)) return null;

            const payload_type = @This().adtVariantPayloadType(result_type, field.name) orelse return null;
            const payload_sema_type = try @This().adtVariantPayloadSemaType(self, self.parent.typecheck.exprType(expr_id), field.name);
            var payload_values: std.ArrayList(mlir.MlirValue) = .{};
            if (!mlir.oraTypeIsANone(payload_type)) {
                try payload_values.append(self.parent.allocator, try @This().lowerAdtPayloadValue(self, call, payload_type, payload_sema_type, locals));
            }

            const op = mlir.oraAdtConstructOpCreate(
                self.parent.context,
                self.parent.location(call.range),
                strRef(field.name),
                if (payload_values.items.len == 0) null else payload_values.items.ptr,
                payload_values.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn lowerAdtPayloadValue(
            self: *FunctionLowerer,
            call: ast.CallExpr,
            payload_type: mlir.MlirType,
            payload_sema_type: ?sema.Type,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const tuple_count = mlir.oraTupleTypeGetNumElements(payload_type);
            if (tuple_count != 0) {
                var elements: std.ArrayList(mlir.MlirValue) = .{};
                for (call.args, 0..) |arg, index| {
                    const element_type = mlir.oraTupleTypeGetElementType(payload_type, index);
                    const raw = try self.lowerExpr(arg, locals);
                    const value = if (payload_sema_type != null and payload_sema_type.?.kind() == .tuple and index < payload_sema_type.?.tuple.len)
                        try self.convertValueForSemaFlow(raw, payload_sema_type.?.tuple[index], exprRange(self.parent.file, arg))
                    else
                        try self.convertValueForFlow(raw, element_type, exprRange(self.parent.file, arg));
                    try elements.append(self.parent.allocator, value);
                }
                const op = mlir.oraTupleCreateOpCreate(
                    self.parent.context,
                    self.parent.location(call.range),
                    if (elements.items.len == 0) null else elements.items.ptr,
                    elements.items.len,
                    payload_type,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            const field_count = mlir.oraAnonymousStructTypeGetFieldCount(payload_type);
            if (field_count != 0) {
                var fields: std.ArrayList(mlir.MlirValue) = .{};
                for (call.args, 0..) |arg, index| {
                    const field_type = mlir.oraAnonymousStructTypeGetFieldType(payload_type, index);
                    const raw = try self.lowerExpr(arg, locals);
                    const value = if (payload_sema_type != null and payload_sema_type.?.kind() == .anonymous_struct and index < payload_sema_type.?.anonymous_struct.fields.len)
                        try self.convertValueForSemaFlow(raw, payload_sema_type.?.anonymous_struct.fields[index].ty, exprRange(self.parent.file, arg))
                    else
                        try self.convertValueForFlow(raw, field_type, exprRange(self.parent.file, arg));
                    try fields.append(self.parent.allocator, value);
                }
                const op = mlir.oraStructInitOpCreate(
                    self.parent.context,
                    self.parent.location(call.range),
                    if (fields.items.len == 0) null else fields.items.ptr,
                    fields.items.len,
                    payload_type,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            const raw = try self.lowerExpr(call.args[0], locals);
            if (payload_sema_type) |target| {
                return self.convertValueForSemaFlow(raw, target, exprRange(self.parent.file, call.args[0]));
            }
            return self.convertValueForFlow(raw, payload_type, exprRange(self.parent.file, call.args[0]));
        }

        fn adtVariantPayloadSemaType(
            self: *FunctionLowerer,
            adt_sema_type: sema.Type,
            variant_name: []const u8,
        ) anyerror!?sema.Type {
            if (adt_sema_type.kind() != .enum_) return null;
            const enum_name = adt_sema_type.enum_.name;
            if (self.parent.typecheck.instantiatedEnumByName(enum_name)) |instantiated| {
                const variant = instantiated.variantByName(variant_name) orelse return null;
                return variant.payload_type;
            }
            const item_id = self.parent.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |enum_item| enum_item,
                else => return null,
            };
            const variant_index = self.parent.item_index.lookupEnumVariantIndex(item_id, variant_name) orelse return null;
            return try @This().enumVariantPayloadSemaType(self, enum_item.variants[variant_index].payload);
        }

        fn enumVariantPayloadSemaType(
            self: *FunctionLowerer,
            payload: ast.EnumVariantPayload,
        ) anyerror!?sema.Type {
            return switch (payload) {
                .none => null,
                .positional => |types| blk: {
                    if (types.len == 0) break :blk null;
                    if (types.len == 1) {
                        break :blk try type_descriptors.descriptorFromTypeExpr(self.parent.allocator, self.parent.file, self.parent.item_index, types[0]);
                    }
                    const elements = try self.parent.allocator.alloc(sema.Type, types.len);
                    for (types, 0..) |type_expr, index| {
                        elements[index] = try type_descriptors.descriptorFromTypeExpr(self.parent.allocator, self.parent.file, self.parent.item_index, type_expr);
                    }
                    break :blk .{ .tuple = elements };
                },
                .named => |fields| blk: {
                    if (fields.len == 0) break :blk null;
                    const sema_fields = try self.parent.allocator.alloc(sema.AnonymousStructField, fields.len);
                    for (fields, 0..) |field, index| {
                        sema_fields[index] = .{
                            .name = field.name,
                            .ty = try type_descriptors.descriptorFromTypeExpr(self.parent.allocator, self.parent.file, self.parent.item_index, field.type_expr),
                        };
                    }
                    break :blk .{ .anonymous_struct = .{ .fields = sema_fields } };
                },
            };
        }

        fn adtVariantPayloadType(adt_type: mlir.MlirType, variant_name: []const u8) ?mlir.MlirType {
            const count = mlir.oraAdtTypeGetNumVariants(adt_type);
            for (0..count) |index| {
                const name_ref = mlir.oraAdtTypeGetVariantName(adt_type, index);
                if (std.mem.eql(u8, name_ref.data[0..name_ref.length], variant_name)) {
                    return mlir.oraAdtTypeGetVariantPayloadType(adt_type, index);
                }
            }
            return null;
        }

        fn lowerResultBuiltinCall(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            call: ast.CallExpr,
            locals: *LocalEnv,
        ) anyerror!?mlir.MlirValue {
            _ = self;
            _ = expr_id;
            _ = call;
            _ = locals;
            return null;
        }

        fn lowerResultTransformCall(
            self: *FunctionLowerer,
            call: ast.CallExpr,
            locals: *LocalEnv,
            path: []const u8,
            operand: mlir.MlirValue,
            result_type: mlir.MlirType,
            result_sema_type: sema.Type,
            callback_item_id: ast.ItemId,
            callback_function: ast.FunctionItem,
        ) anyerror!mlir.MlirValue {
            _ = locals;
            _ = path;
            const loc = self.parent.location(call.range);
            const is_error = mlir.oraErrorIsErrorOpCreate(self.parent.context, loc, operand);
            if (mlir.oraOperationIsNull(is_error)) return error.MlirOperationCreationFailed;
            const is_error_value = appendValueOp(self.block, is_error);

            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, is_error_value, &[_]mlir.MlirType{result_type}, 1, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            const error_type = self.parent.lowerSemaType(self.parent.typecheck.exprType(call.args[0]).error_union.error_types[0], exprRange(self.parent.file, call.args[0]));
            const raw_error = mlir.oraErrorGetErrorOpCreate(self.parent.context, loc, operand, error_type);
            if (mlir.oraOperationIsNull(raw_error)) return error.MlirOperationCreationFailed;
            appendOp(then_block, raw_error);
            const err_op = mlir.oraErrorErrOpCreate(self.parent.context, loc, mlir.oraOperationGetResult(raw_error, 0), result_type);
            if (mlir.oraOperationIsNull(err_op)) return error.MlirOperationCreationFailed;
            if (self.parent.errorUnionRequiresWideCarrier(result_sema_type)) {
                mlir.oraOperationSetAttributeByName(err_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            appendOp(then_block, err_op);
            try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(err_op, 0)});

            const payload_type = self.parent.lowerSemaType(self.parent.typecheck.exprType(call.args[0]).error_union.payload_type.*, exprRange(self.parent.file, call.args[0]));
            const unwrap = mlir.oraErrorUnwrapOpCreate(self.parent.context, loc, operand, payload_type);
            if (mlir.oraOperationIsNull(unwrap)) return error.MlirOperationCreationFailed;
            appendOp(else_block, unwrap);
            const callback_name = callback_function.name;
            const callback_type = self.parent.typecheck.item_types[callback_item_id.index()];
            const chained = try @This().lowerDirectFunctionItemCall(self, else_block, callback_name, callback_type, call.range, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap, 0)});
            try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{chained});
            return mlir.oraOperationGetResult(if_op, 0);
        }

        fn lowerDirectFunctionItemCall(
            self: *FunctionLowerer,
            block: mlir.MlirBlock,
            callee_name: []const u8,
            callee_type: sema.Type,
            range: source.TextRange,
            loc: mlir.MlirLocation,
            args: []const mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            const return_types = callee_type.function.return_types;
            const result_type = self.parent.lowerSemaType(return_types[0], range);
            var result_types: [1]mlir.MlirType = .{result_type};
            const op = mlir.oraFuncCallOpCreate(
                self.parent.context,
                loc,
                strRef(callee_name),
                if (args.len == 0) null else args.ptr,
                args.len,
                &result_types,
                1,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(block, op);
        }

        const ResolvedExternProxyMethodCall = struct {
            field: ast.FieldExpr,
            trait_name: []const u8,
            method: sema.TraitMethodSignature,
            trait_method_owner: ?sema.VerificationTraitMethodOwner = null,
        };

        fn externTraitMethodOwner(self: *FunctionLowerer, trait_name: []const u8, method_name: []const u8) ?sema.VerificationTraitMethodOwner {
            const trait_item_id = self.parent.item_index.lookup(trait_name) orelse return null;
            const method_index = self.parent.item_index.lookupTraitMethodIndex(trait_item_id, method_name) orelse return null;
            return .{
                .trait_item = trait_item_id,
                .method_index = method_index,
            };
        }

        fn externSummaryLocals(
            self: *FunctionLowerer,
            base_locals: *const LocalEnv,
            trait_method: ast.nodes.TraitMethod,
            args: []const mlir.MlirValue,
        ) anyerror!LocalEnv {
            var clause_locals = try self.cloneLocals(base_locals);
            const bind_count = @min(trait_method.parameters.len, args.len);
            for (0..bind_count) |index| {
                try self.bindPatternValue(trait_method.parameters[index].pattern, args[index], &clause_locals);
            }
            return clause_locals;
        }

        fn emitExternSummaryRequires(
            self: *FunctionLowerer,
            owner: ?sema.VerificationTraitMethodOwner,
            base_locals: *const LocalEnv,
            args: []const mlir.MlirValue,
        ) anyerror!void {
            const fact_owner = owner orelse return;
            const method = self.parent.traitMethodForVerificationOwner(fact_owner) orelse return error.InvalidVerificationFact;
            for (self.parent.traitMethodVerificationFactEntries(fact_owner)) |entry| {
                const fact = self.parent.traitMethodVerificationFact(entry);
                if (fact.kind != .requires) continue;
                const expr = fact.expr orelse return error.InvalidVerificationFact;
                var clause_locals = try @This().externSummaryLocals(self, base_locals, method, args);
                const condition = try self.lowerExpr(expr, &clause_locals);
                const op = mlir.oraAssertOpCreate(self.parent.context, self.parent.location(fact.range), condition, strRef("extern trait summary requires"));
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_context"), namedStringAttr(self.parent.context, "ora.verification_context", "extern_trait_summary").attribute);
                appendOp(self.block, op);
            }
        }

        fn externSummaryHasEnsures(self: *FunctionLowerer, owner: ?sema.VerificationTraitMethodOwner) bool {
            const fact_owner = owner orelse return false;
            for (self.parent.traitMethodVerificationFactEntries(fact_owner)) |entry| {
                const kind = self.parent.traitMethodVerificationFact(entry).kind;
                if (kind == .ensures or kind == .ensures_ok) return true;
            }
            return false;
        }

        fn externSummaryHasClauses(self: *FunctionLowerer, owner: ?sema.VerificationTraitMethodOwner) bool {
            const fact_owner = owner orelse return false;
            return self.parent.traitMethodVerificationFactEntries(fact_owner).len > 0;
        }

        fn externReturndataStrictSupports(ty: sema.Type) bool {
            return externReturndataStrictSupportsInner(ty, true);
        }

        fn externReturndataStrictSupportsInner(ty: sema.Type, top_level: bool) bool {
            return switch (ty) {
                .bool, .address, .integer, .fixed_bytes, .bitfield => true,
                .string, .bytes => top_level,
                .slice => |slice| top_level and externReturndataStrictSupportsSliceElement(slice.element_type.*),
                // The current ABI decode op carries one enum_variant_count attr,
                // so only a top-level enum has enough metadata for strict
                // returndata range validation in this slice.
                .enum_ => top_level,
                .refinement => |refinement| externReturndataStrictSupportsInner(refinement.base_type.*, top_level),
                .tuple => |tuple| blk: {
                    if (tuple.len == 0) break :blk false;
                    if (top_level and externReturndataStrictSupportsMixedDynamicTuple(tuple)) break :blk true;
                    for (tuple) |element| {
                        if (!externReturndataStrictSupportsInner(element, false)) break :blk false;
                    }
                    break :blk true;
                },
                else => false,
            };
        }

        fn externReturndataStrictSupportsMixedDynamicTuple(elements: []const sema.Type) bool {
            return elements.len == 2 and
                externReturndataStrictSupportsU256(elements[0]) and
                externReturndataStrictSupportsTopLevelDynamic(elements[1]);
        }

        fn externReturndataStrictSupportsTopLevelDynamic(ty: sema.Type) bool {
            return switch (ty) {
                .string, .bytes => true,
                .slice => |slice| externReturndataStrictSupportsSliceElement(slice.element_type.*),
                .refinement => |refinement| externReturndataStrictSupportsTopLevelDynamic(refinement.base_type.*),
                else => false,
            };
        }

        fn externReturndataStrictSupportsSliceElement(ty: sema.Type) bool {
            return switch (ty) {
                .bool, .address, .fixed_bytes => true,
                .integer => |integer| integer.isUnsignedBits(256),
                .refinement => |refinement| externReturndataStrictSupportsSliceElement(refinement.base_type.*),
                else => false,
            };
        }

        fn externReturndataStrictSupportsU256(ty: sema.Type) bool {
            return switch (ty) {
                .integer => |integer| integer.isUnsignedBits(256),
                .refinement => |refinement| externReturndataStrictSupportsU256(refinement.base_type.*),
                else => false,
            };
        }

        fn lowerExternSummaryEnsuresCondition(
            self: *FunctionLowerer,
            owner: ?sema.VerificationTraitMethodOwner,
            base_locals: *const LocalEnv,
            args: []const mlir.MlirValue,
            result: mlir.MlirValue,
            block: mlir.MlirBlock,
        ) anyerror!?mlir.MlirValue {
            const fact_owner = owner orelse return null;
            const method = self.parent.traitMethodForVerificationOwner(fact_owner) orelse return error.InvalidVerificationFact;
            const previous_block = self.block;
            const previous_result = self.current_return_value;
            self.block = block;
            self.current_return_value = result;
            defer {
                self.block = previous_block;
                self.current_return_value = previous_result;
            }

            var combined: ?mlir.MlirValue = null;
            for (self.parent.traitMethodVerificationFactEntries(fact_owner)) |entry| {
                const fact = self.parent.traitMethodVerificationFact(entry);
                if (fact.kind != .ensures and fact.kind != .ensures_ok) continue;
                const expr = fact.expr orelse return error.InvalidVerificationFact;
                var clause_locals = try @This().externSummaryLocals(self, base_locals, method, args);
                const condition = try self.lowerExpr(expr, &clause_locals);
                combined = if (combined) |acc| blk: {
                    const and_op = mlir.oraArithAndIOpCreate(self.parent.context, self.parent.location(fact.range), acc, condition);
                    if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(block, and_op);
                } else condition;
            }
            return combined;
        }

        fn lowerExternProxyMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!?mlir.MlirValue {
            const resolved = @This().resolveExternProxyMethodCall(self, call.callee) orelse return null;
            switch (resolved.method.return_type.kind()) {
                .bool, .address, .integer, .fixed_bytes, .enum_, .bitfield, .refinement, .string, .bytes, .tuple, .struct_, .contract, .slice => {},
                else => return error.UnsupportedExternTraitLowering,
            }

            const loc = self.parent.location(call.range);
            const payload_type = self.parent.lowerSemaType(resolved.method.return_type, exprRange(self.parent.file, resolved.field.base));
            const result_type = self.parent.lowerExprType(expr_id);
            const proxy = switch (self.parent.file.expression(resolved.field.base).*) {
                .ExternalProxy => |proxy| proxy,
                .Group => |group| switch (self.parent.file.expression(group.expr).*) {
                    .ExternalProxy => |proxy| proxy,
                    else => return error.UnsupportedExternTraitLowering,
                },
                else => return error.UnsupportedExternTraitLowering,
            };

            var encode_args: std.ArrayList(mlir.MlirValue) = .{};
            for (call.args) |arg| {
                try encode_args.append(self.parent.allocator, try self.lowerExpr(arg, locals));
            }
            try @This().emitExternSummaryRequires(self, resolved.trait_method_owner, locals, encode_args.items);

            const layout_context = @This().layoutContext(self);
            const encode_op = try abi_runtime_encoder.createAbiEncodeWithSelectorOp(
                self.parent.allocator,
                self.parent.context,
                loc,
                layout_context,
                resolved.method.name,
                resolved.method.param_types,
                encode_args.items,
                defaultIntegerType(self.parent.context),
            );
            if (mlir.oraOperationIsNull(encode_op)) return error.MlirOperationCreationFailed;
            const calldata = appendValueOp(self.block, encode_op);

            const target = try self.lowerExpr(proxy.address_expr, locals);
            const gas = try self.lowerExprForFlowTarget(proxy.gas_expr, defaultIntegerType(self.parent.context), locals);

            const external_call_op = mlir.oraExternalCallOpCreate(
                self.parent.context,
                loc,
                strRef(if (resolved.method.extern_call_kind == .staticcall) "staticcall" else "call"),
                strRef(resolved.trait_name),
                strRef(resolved.method.name),
                target,
                gas,
                calldata,
                boolType(self.parent.context),
                defaultIntegerType(self.parent.context),
            );
            if (mlir.oraOperationIsNull(external_call_op)) return error.MlirOperationCreationFailed;
            if (resolved.method.extern_call_kind == .call and @This().externSummaryHasClauses(self, resolved.trait_method_owner)) {
                mlir.oraOperationSetAttributeByName(external_call_op, strRef("ora.trusted_extern_frame"), namedStringAttr(self.parent.context, "ora.trusted_extern_frame", "caller_storage").attribute);
            }
            appendOp(self.block, external_call_op);
            const success = mlir.oraOperationGetResult(external_call_op, 0);
            const returndata = mlir.oraOperationGetResult(external_call_op, 1);

            const has_extern_summary_ensures = @This().externSummaryHasEnsures(self, resolved.trait_method_owner);
            const if_result_types = if (has_extern_summary_ensures)
                [_]mlir.MlirType{ result_type, boolType(self.parent.context) }
            else
                [_]mlir.MlirType{ result_type, undefined };
            const if_result_count: usize = if (has_extern_summary_ensures) 2 else 1;
            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, success, &if_result_types, if_result_count, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            const external_result_requires_wide = self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.exprType(expr_id));
            if (external_result_requires_wide) {
                mlir.oraOperationSetAttributeByName(if_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            const can_strict_decode_returndata = externReturndataStrictSupports(resolved.method.return_type);
            if (!can_strict_decode_returndata) {
                // N3c: summary clauses still need the decoded payload value to
                // evaluate the summary expression. Unsupported summary return
                // shapes stay on the legacy payload projection until their
                // strict decode surface exists.
                const decode_op = try abi_runtime_decoder.createAbiDecodeOp(
                    self.parent.allocator,
                    self.parent.context,
                    loc,
                    layout_context,
                    resolved.method.return_type,
                    returndata,
                    payload_type,
                );
                if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
                appendOp(then_block, decode_op);
                const decoded = mlir.oraOperationGetResult(decode_op, 0);
                const summary_condition = try @This().lowerExternSummaryEnsuresCondition(self, resolved.trait_method_owner, locals, encode_args.items, decoded, then_block);
                const ok_op = mlir.oraErrorOkOpCreate(self.parent.context, loc, decoded, result_type);
                if (mlir.oraOperationIsNull(ok_op)) return error.MlirOperationCreationFailed;
                if (external_result_requires_wide) {
                    mlir.oraOperationSetAttributeByName(ok_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                }
                appendOp(then_block, ok_op);
                if (has_extern_summary_ensures) {
                    const condition = summary_condition orelse appendValueOp(then_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                    try support.appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{ mlir.oraOperationGetResult(ok_op, 0), condition });
                } else {
                    try support.appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(ok_op, 0)});
                }
            } else {
                const decode_op = try abi_runtime_decoder.createExternalReturnAbiDecodeOp(
                    self.parent.allocator,
                    self.parent.context,
                    loc,
                    layout_context,
                    resolved.method.return_type,
                    returndata,
                    result_type,
                    "ExternalCallFailed",
                );
                if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
                if (external_result_requires_wide) {
                    mlir.oraOperationSetAttributeByName(decode_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                }
                appendOp(then_block, decode_op);
                const decoded_result = mlir.oraOperationGetResult(decode_op, 0);
                if (has_extern_summary_ensures) {
                    const is_error_op = mlir.oraErrorIsErrorOpCreate(self.parent.context, loc, decoded_result);
                    if (mlir.oraOperationIsNull(is_error_op)) return error.MlirOperationCreationFailed;
                    appendOp(then_block, is_error_op);
                    const is_error = mlir.oraOperationGetResult(is_error_op, 0);

                    const summary_if_result_types = [_]mlir.MlirType{ result_type, boolType(self.parent.context) };
                    const summary_if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, is_error, &summary_if_result_types, summary_if_result_types.len, true);
                    if (mlir.oraOperationIsNull(summary_if_op)) return error.MlirOperationCreationFailed;
                    if (external_result_requires_wide) {
                        mlir.oraOperationSetAttributeByName(summary_if_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                    }
                    appendOp(then_block, summary_if_op);

                    const summary_err_block = mlir.oraScfIfOpGetThenBlock(summary_if_op);
                    const summary_ok_block = mlir.oraScfIfOpGetElseBlock(summary_if_op);
                    if (mlir.oraBlockIsNull(summary_err_block) or mlir.oraBlockIsNull(summary_ok_block)) return error.MlirOperationCreationFailed;

                    const true_value = appendValueOp(summary_err_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                    try support.appendScfYieldValues(self.parent.context, summary_err_block, loc, &[_]mlir.MlirValue{ decoded_result, true_value });

                    const unwrap_op = mlir.oraErrorUnwrapOpCreate(self.parent.context, loc, decoded_result, payload_type);
                    if (mlir.oraOperationIsNull(unwrap_op)) return error.MlirOperationCreationFailed;
                    appendOp(summary_ok_block, unwrap_op);
                    const decoded_payload = mlir.oraOperationGetResult(unwrap_op, 0);
                    const summary_condition = try @This().lowerExternSummaryEnsuresCondition(self, resolved.trait_method_owner, locals, encode_args.items, decoded_payload, summary_ok_block);
                    const condition = summary_condition orelse appendValueOp(summary_ok_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                    try support.appendScfYieldValues(self.parent.context, summary_ok_block, loc, &[_]mlir.MlirValue{ decoded_result, condition });

                    try support.appendScfYieldValues(
                        self.parent.context,
                        then_block,
                        loc,
                        &[_]mlir.MlirValue{
                            mlir.oraOperationGetResult(summary_if_op, 0),
                            mlir.oraOperationGetResult(summary_if_op, 1),
                        },
                    );
                } else {
                    try support.appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{decoded_result});
                }
            }

            const error_result = try @This().lowerExternErrorResult(self, expr_id, resolved.method, else_block, loc, returndata);
            if (has_extern_summary_ensures) {
                const true_value = appendValueOp(else_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                try support.appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{ error_result, true_value });

                const summary_assume = mlir.oraAssumeOpCreate(self.parent.context, loc, mlir.oraOperationGetResult(if_op, 1));
                if (mlir.oraOperationIsNull(summary_assume)) return error.MlirOperationCreationFailed;
                mlir.oraOperationSetAttributeByName(summary_assume, strRef("ora.verification_context"), namedStringAttr(self.parent.context, "ora.verification_context", "extern_trait_summary").attribute);
                appendOp(self.block, summary_assume);
            } else {
                try support.appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{error_result});
            }

            return mlir.oraOperationGetResult(if_op, 0);
        }

        fn lowerExternErrorResult(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            method: sema.TraitMethodSignature,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            returndata: mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            if (method.errors.len == 0) {
                return @This().lowerNamedExternErrorResult(self, expr_id, block, loc, "ExternalCallFailed", returndata);
            }

            const selector_word = try @This().decodeExternSelectorWord(self, block, loc, returndata);
            return try @This().lowerExternErrorMatchChain(self, expr_id, method.errors, 0, selector_word, block, loc, returndata);
        }

        fn decodeExternSelectorWord(
            self: *FunctionLowerer,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            returndata: mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            const layout_context = @This().layoutContext(self);
            // TODO(N3b): error selector decode reads the first 4 raw returndata
            // bytes. The temporary u256 layout matches the legacy lowering, but
            // the layout-driven runtime decoder needs a raw selector-byte shape.
            const decode_op = try abi_runtime_decoder.createAbiDecodeOp(
                self.parent.allocator,
                self.parent.context,
                loc,
                layout_context,
                .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } },
                returndata,
                defaultIntegerType(self.parent.context),
            );
            if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
            appendOp(block, decode_op);
            return mlir.oraOperationGetResult(decode_op, 0);
        }

        fn lowerExternErrorMatchChain(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            error_names: []const []const u8,
            index: usize,
            selector_word: mlir.MlirValue,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            returndata: mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            if (index >= error_names.len) {
                return @This().lowerNamedExternErrorResult(self, expr_id, block, loc, "ExternalCallFailed", returndata);
            }

            const error_name = error_names[index];
            const error_item_id = self.parent.item_index.lookup(error_name);
            if (error_item_id == null or self.parent.file.item(error_item_id.?).* != .ErrorDecl) {
                return try @This().lowerExternErrorMatchChain(self, expr_id, error_names, index + 1, selector_word, block, loc, returndata);
            }

            const error_op = @This().findErrorDeclOp(self, error_item_id.?) orelse
                return try @This().lowerExternErrorMatchChain(self, expr_id, error_names, index + 1, selector_word, block, loc, returndata);
            const selector_attr = mlir.oraOperationGetAttributeByName(error_op, strRef("ora.error_selector"));
            if (mlir.oraAttributeIsNull(selector_attr)) {
                return try @This().lowerExternErrorMatchChain(self, expr_id, error_names, index + 1, selector_word, block, loc, returndata);
            }

            const selector_ref = mlir.oraStringAttrGetValue(selector_attr);
            const selector_text = selector_ref.data[0..selector_ref.length];
            const shifted_selector = try @This().createShiftedSelectorWord(self, block, loc, selector_text);
            const cmp_op = mlir.oraArithCmpIOpCreate(self.parent.context, loc, cmpPredicate("eq"), selector_word, shifted_selector);
            if (mlir.oraOperationIsNull(cmp_op)) return error.MlirOperationCreationFailed;
            const condition = appendValueOp(block, cmp_op);

            const result_type = self.parent.lowerExprType(expr_id);
            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, condition, &[_]mlir.MlirType{result_type}, 1, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            const matched = try @This().lowerNamedExternErrorResult(self, expr_id, then_block, loc, error_name, returndata);
            try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{matched});

            const fallback = try @This().lowerExternErrorMatchChain(self, expr_id, error_names, index + 1, selector_word, else_block, loc, returndata);
            try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{fallback});

            return mlir.oraOperationGetResult(if_op, 0);
        }

        fn lowerNamedExternErrorResult(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            error_name: []const u8,
            returndata: mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            const error_type = @This().externErrorTypeByName(self, expr_id, error_name) orelse self.parent.typecheck.exprType(expr_id).errorTypes()[0];
            if (@This().errorTypeHasPayload(self, error_name)) {
                return try @This().lowerPayloadExternErrorResult(self, expr_id, block, loc, error_type, returndata);
            }
            const error_symbol_name = if (self.parent.item_index.lookup(error_name)) |item_id|
                @This().errorDeclSymbolName(self, item_id, error_name)
            else
                error_name;
            const error_op = mlir.oraErrorReturnOpCreate(
                self.parent.context,
                loc,
                strRef(error_symbol_name),
                null,
                0,
                self.parent.lowerExprType(expr_id),
            );
            if (mlir.oraOperationIsNull(error_op)) return error.MlirOperationCreationFailed;
            if (self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.exprType(expr_id))) {
                mlir.oraOperationSetAttributeByName(error_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            appendOp(block, error_op);
            return mlir.oraOperationGetResult(error_op, 0);
        }

        fn lowerPayloadExternErrorResult(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            error_type: sema.Type,
            returndata: mlir.MlirValue,
        ) anyerror!mlir.MlirValue {
            const lowered_error_type = self.parent.lowerSemaType(error_type, exprRange(self.parent.file, expr_id));
            const payload_offset = appendValueOp(block, createIntegerConstant(self.parent.context, loc, defaultIntegerType(self.parent.context), 4));
            const payload_ptr_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, returndata, payload_offset);
            if (mlir.oraOperationIsNull(payload_ptr_op)) return error.MlirOperationCreationFailed;
            appendOp(block, payload_ptr_op);
            const payload_returndata = mlir.oraOperationGetResult(payload_ptr_op, 0);
            const layout_context = @This().layoutContext(self);
            const decode_op = try abi_runtime_decoder.createAbiDecodeOp(
                self.parent.allocator,
                self.parent.context,
                loc,
                layout_context,
                error_type,
                payload_returndata,
                lowered_error_type,
            );
            if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
            appendOp(block, decode_op);
            const decoded = mlir.oraOperationGetResult(decode_op, 0);
            const err_op = mlir.oraErrorErrOpCreate(self.parent.context, loc, decoded, self.parent.lowerExprType(expr_id));
            if (mlir.oraOperationIsNull(err_op)) return error.MlirOperationCreationFailed;
            if (self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.exprType(expr_id))) {
                mlir.oraOperationSetAttributeByName(err_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            appendOp(block, err_op);
            return mlir.oraOperationGetResult(err_op, 0);
        }

        fn errorTypeHasPayload(self: *FunctionLowerer, error_name: []const u8) bool {
            const error_item_id = self.parent.item_index.lookup(error_name) orelse return false;
            if (self.parent.file.item(error_item_id).* != .ErrorDecl) return false;
            return self.parent.file.item(error_item_id).ErrorDecl.parameters.len != 0;
        }

        fn externErrorTypeByName(self: *FunctionLowerer, expr_id: ast.ExprId, error_name: []const u8) ?sema.Type {
            const result_type = self.parent.typecheck.exprType(expr_id);
            if (result_type.kind() != .error_union) return null;
            for (result_type.errorTypes()) |error_type| {
                if (std.mem.eql(u8, error_type.name() orelse "", error_name)) return error_type;
            }
            return null;
        }

        fn createShiftedSelectorWord(
            self: *FunctionLowerer,
            block: mlir.MlirBlock,
            loc: mlir.MlirLocation,
            selector_text: []const u8,
        ) anyerror!mlir.MlirValue {
            const selector_value = try std.fmt.parseUnsigned(u32, selector_text[2..], 16);
            const selector_u32_type = mlir.oraIntegerTypeCreate(self.parent.context, 32);
            const selector_const = appendValueOp(block, createIntegerConstant(self.parent.context, loc, selector_u32_type, selector_value));
            const selector_u256_op = mlir.oraArithExtUIOpCreate(self.parent.context, loc, selector_const, defaultIntegerType(self.parent.context));
            if (mlir.oraOperationIsNull(selector_u256_op)) return error.MlirOperationCreationFailed;
            const selector_u256 = appendValueOp(block, selector_u256_op);
            const shift = appendValueOp(block, createIntegerConstant(self.parent.context, loc, defaultIntegerType(self.parent.context), 224));
            const shifted = mlir.oraArithShlIOpCreate(self.parent.context, loc, selector_u256, shift);
            if (mlir.oraOperationIsNull(shifted)) return error.MlirOperationCreationFailed;
            return appendValueOp(block, shifted);
        }

        fn findErrorDeclOp(self: *FunctionLowerer, item_id: ast.ItemId) ?mlir.MlirOperation {
            for (self.parent.items.items) |handle| {
                if (handle.item_id.index() == item_id.index() and handle.kind == .error_decl) return handle.raw_operation;
            }
            return null;
        }

        fn errorDeclSymbolName(self: *FunctionLowerer, item_id: ast.ItemId, fallback_name: []const u8) []const u8 {
            const op = @This().findErrorDeclOp(self, item_id) orelse return fallback_name;
            const attr = mlir.oraOperationGetAttributeByName(op, strRef("sym_name"));
            if (mlir.oraAttributeIsNull(attr)) return fallback_name;
            const value = mlir.oraStringAttrGetValue(attr);
            return value.data[0..value.length];
        }

        fn resolveExternProxyMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId) ?ResolvedExternProxyMethodCall {
            const field = switch (self.parent.file.expression(expr_id).*) {
                .Field => |field| field,
                .Group => |group| return @This().resolveExternProxyMethodCall(self, group.expr),
                else => return null,
            };
            const base_type = self.parent.typecheck.exprType(field.base);
            if (base_type.kind() != .external_proxy) return null;
            const trait_interface = self.parent.typecheck.traitInterfaceByName(base_type.external_proxy.trait_name) orelse return null;
            const method = trait_interface.methodByName(field.name) orelse return null;
            return .{
                .field = field,
                .trait_name = base_type.external_proxy.trait_name,
                .method = method,
                .trait_method_owner = @This().externTraitMethodOwner(self, base_type.external_proxy.trait_name, method.name),
            };
        }

        fn lowerCurrentMethodSelfCallResult(self: *FunctionLowerer, call: ast.CallExpr) anyerror!?mlir.MlirValue {
            const current_result = self.current_return_value orelse return null;
            const current_item_id = self.item_id orelse return null;
            const current_function = self.function orelse return null;
            if (!@This().functionHasRuntimeSelf(self, current_function)) return null;

            const callee_item_id = @This().calleeFunctionItemId(self, call.callee) orelse return null;
            if (callee_item_id.index() != current_item_id.index()) return null;

            const runtime_args = if (current_function.is_generic)
                self.parent.stripGenericCallArgs(current_function, call)
            else
                call.args;
            if (runtime_args.len != 1) return null;

            const self_pattern = blk: {
                for (current_function.parameters) |parameter| {
                    if (parameter.is_comptime) continue;
                    if (std.mem.eql(u8, self.parent.patternName(parameter.pattern) orelse "", "self")) {
                        break :blk parameter.pattern;
                    }
                    break :blk null;
                }
                break :blk null;
            } orelse return null;

            const self_binding = self.parent.resolution.expr_bindings[runtime_args[0].index()] orelse return null;
            switch (self_binding) {
                .pattern => |pattern_id| {
                    if (pattern_id.index() != self_pattern.index()) return null;
                    return current_result;
                },
                .item => return null,
            }
        }

        fn lowerCurrentFunctionCallResult(self: *FunctionLowerer, call: ast.CallExpr) anyerror!?mlir.MlirValue {
            const current_result = self.current_return_value orelse return null;
            const current_item_id = self.item_id orelse return null;
            const current_function = self.function orelse return null;

            const callee_item_id = @This().calleeFunctionItemId(self, call.callee) orelse return null;
            if (callee_item_id.index() != current_item_id.index()) return null;

            const runtime_args = if (current_function.is_generic)
                self.parent.stripGenericCallArgs(current_function, call)
            else
                call.args;

            var runtime_param_index: usize = 0;
            for (current_function.parameters) |parameter| {
                if (parameter.is_comptime) continue;
                if (runtime_param_index >= runtime_args.len) return null;

                const binding = self.parent.resolution.expr_bindings[runtime_args[runtime_param_index].index()] orelse return null;
                switch (binding) {
                    .pattern => |pattern_id| {
                        if (pattern_id.index() != parameter.pattern.index()) return null;
                    },
                    .item => return null,
                }

                runtime_param_index += 1;
            }

            if (runtime_param_index != runtime_args.len) return null;
            return current_result;
        }

        fn lowerTraitBoundMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!?mlir.MlirValue {
            const resolved = @This().resolveTraitBoundMethodCall(self, call.callee) orelse return null;
            const runtime_parameters = try self.parent.runtimeFunctionParameters(resolved.function);
            var generic_count: usize = 0;
            for (resolved.function.parameters) |parameter| {
                if (!parameter.is_comptime) break;
                generic_count += 1;
            }
            const method_runtime_count = if (runtime_parameters.len == 0) 0 else runtime_parameters.len - 1;
            const explicit_generics = resolved.function.is_generic and call.args.len >= generic_count + method_runtime_count;
            const runtime_args = if (explicit_generics)
                call.args[generic_count..]
            else
                call.args;

            var args: std.ArrayList(mlir.MlirValue) = .{};
            var receiver_value = try self.lowerExpr(resolved.receiver_expr, locals);
            const receiver_type = self.parent.lowerExprType(resolved.receiver_expr);
            receiver_value = try self.convertValueForFlow(receiver_value, receiver_type, exprRange(self.parent.file, resolved.receiver_expr));
            try args.append(self.parent.allocator, receiver_value);

            for (runtime_args, 0..) |arg, index| {
                var arg_value = try self.lowerExpr(arg, locals);
                const parameter = runtime_parameters[index + 1];
                const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[parameter.pattern.index()].type, parameter.range);
                arg_value = try self.convertValueForFlow(arg_value, target_type, exprRange(self.parent.file, arg));
                try args.append(self.parent.allocator, arg_value);
            }

            const callee_name = try self.parent.ensureLoweredImplMethod(
                resolved.method_item_id,
                resolved.function,
                resolved.trait_name,
                resolved.target_name,
                call,
                @This().currentContractParentBlock(self),
                @This().currentContractItemId(self),
            );

            const result_type = self.parent.lowerExprType(expr_id);
            var result_types: [1]mlir.MlirType = .{result_type};
            const op = mlir.oraFuncCallOpCreate(
                self.parent.context,
                self.parent.location(call.range),
                strRef(callee_name),
                if (args.items.len == 0) null else args.items.ptr,
                args.items.len,
                if (self.typeIsVoid(result_type)) null else &result_types,
                if (self.typeIsVoid(result_type)) 0 else 1,
            );
            _ = try @This().expectOperation(self, op, call.range, "ora.call");
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try @This().voidValue(self, call.range);
            }
            return appendValueOp(self.block, op);
        }

        const ResolvedTraitBoundMethodCall = struct {
            impl_item_id: ast.ItemId,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
            trait_name: []const u8,
            target_name: []const u8,
            receiver_expr: ast.ExprId,
        };

        fn resolveTraitBoundMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId) ?ResolvedTraitBoundMethodCall {
            const field = switch (self.parent.file.expression(expr_id).*) {
                .Field => |field| field,
                .Group => |group| return @This().resolveTraitBoundMethodCall(self, group.expr),
                else => return null,
            };
            const receiver_type = self.parent.typecheck.exprType(field.base);
            const receiver_name = receiver_type.name() orelse return null;
            if (@This().resolveConcreteValueMethodCall(self, field, receiver_name)) |resolved| return resolved;

            const function = self.function orelse return null;
            const concrete_type = self.parent.substitutedType(receiver_name) orelse return null;
            const target_name = concrete_type.name() orelse return null;

            var matched_trait: ?[]const u8 = null;
            for (function.trait_bounds) |bound| {
                if (!std.mem.eql(u8, bound.parameter_name, receiver_name)) continue;
                const trait_interface = self.parent.typecheck.traitInterfaceByName(bound.trait_name) orelse continue;
                if (trait_interface.methodByNameAndReceiver(field.name, .value_self) == null) continue;
                if (matched_trait != null) return null;
                matched_trait = bound.trait_name;
            }
            const trait_name = matched_trait orelse return null;
            const impl_item_id = self.parent.item_index.lookupImpl(trait_name, target_name) orelse return null;
            const method_item_id = self.parent.item_index.lookupImplMethodByReceiver(self.parent.file, impl_item_id, field.name, .value_self) orelse return null;
            const method_function = self.parent.file.item(method_item_id).Function;
            return .{
                .impl_item_id = impl_item_id,
                .method_item_id = method_item_id,
                .function = method_function,
                .trait_name = trait_name,
                .target_name = target_name,
                .receiver_expr = field.base,
            };
        }

        fn resolveConcreteValueMethodCall(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            target_name: []const u8,
        ) ?ResolvedTraitBoundMethodCall {
            var matched_impl_item_id: ?ast.ItemId = null;
            var matched_method_item_id: ?ast.ItemId = null;
            var matched_function: ?ast.FunctionItem = null;
            var matched_trait_name: ?[]const u8 = null;

            for (self.parent.file.items, 0..) |item, item_index| {
                if (item != .Impl) continue;
                const impl_item = item.Impl;
                if (!std.mem.eql(u8, impl_item.target_name, target_name)) continue;
                const impl_item_id = ast.ItemId.fromIndex(item_index);
                const method_count = self.parent.item_index.countImplMethodsByReceiver(self.parent.file, impl_item_id, field.name, .value_self);
                if (method_count == 0) continue;
                if (matched_impl_item_id != null or method_count > 1) return null;
                const method_item_id = self.parent.item_index.lookupImplMethodByReceiver(self.parent.file, impl_item_id, field.name, .value_self) orelse continue;
                const method_item = self.parent.file.item(method_item_id).*;
                if (method_item != .Function) continue;
                matched_impl_item_id = impl_item_id;
                matched_method_item_id = method_item_id;
                matched_function = method_item.Function;
                matched_trait_name = impl_item.trait_name;
            }

            return .{
                .impl_item_id = matched_impl_item_id orelse return null,
                .method_item_id = matched_method_item_id orelse return null,
                .function = matched_function orelse return null,
                .trait_name = matched_trait_name orelse return null,
                .target_name = target_name,
                .receiver_expr = field.base,
            };
        }

        const ResolvedAssociatedImplMethodCall = struct {
            impl_item_id: ast.ItemId,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
            trait_name: []const u8,
            target_name: []const u8,
        };

        fn lowerAssociatedImplMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!?mlir.MlirValue {
            const resolved = @This().resolveAssociatedImplMethodCall(self, call.callee) orelse return null;
            const runtime_args = if (resolved.function.is_generic)
                self.parent.stripGenericCallArgs(resolved.function, call)
            else
                call.args;
            const runtime_parameters = try self.parent.runtimeFunctionParameters(resolved.function);

            var args: std.ArrayList(mlir.MlirValue) = .{};
            for (runtime_args, 0..) |arg, index| {
                var arg_value = try self.lowerExpr(arg, locals);
                const parameter = runtime_parameters[index];
                const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[parameter.pattern.index()].type, parameter.range);
                arg_value = try self.convertValueForFlow(arg_value, target_type, exprRange(self.parent.file, arg));
                try args.append(self.parent.allocator, arg_value);
            }

            const callee_name = try self.parent.ensureLoweredImplMethod(
                resolved.method_item_id,
                resolved.function,
                resolved.trait_name,
                resolved.target_name,
                call,
                @This().currentContractParentBlock(self),
                @This().currentContractItemId(self),
            );

            const result_type = self.parent.lowerExprType(expr_id);
            var result_types: [1]mlir.MlirType = .{result_type};
            const op = mlir.oraFuncCallOpCreate(
                self.parent.context,
                self.parent.location(call.range),
                strRef(callee_name),
                if (args.items.len == 0) null else args.items.ptr,
                args.items.len,
                if (self.typeIsVoid(result_type)) null else &result_types,
                if (self.typeIsVoid(result_type)) 0 else 1,
            );
            _ = try @This().expectOperation(self, op, call.range, "ora.call");
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try @This().voidValue(self, call.range);
            }
            return appendValueOp(self.block, op);
        }

        fn resolveAssociatedImplMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId) ?ResolvedAssociatedImplMethodCall {
            const field = switch (self.parent.file.expression(expr_id).*) {
                .Field => |field| field,
                .Group => |group| return @This().resolveAssociatedImplMethodCall(self, group.expr),
                else => return null,
            };

            if (@This().resolveTraitBoundAssociatedMethodCall(self, field)) |resolved| return resolved;
            return @This().resolveConcreteAssociatedMethodCall(self, field);
        }

        fn resolveConcreteAssociatedMethodCall(self: *FunctionLowerer, field: ast.FieldExpr) ?ResolvedAssociatedImplMethodCall {
            const target_name = @This().concreteTypeNameForExpr(self, field.base) orelse return null;
            var matched_impl_item_id: ?ast.ItemId = null;
            var matched_method_item_id: ?ast.ItemId = null;
            var matched_function: ?ast.FunctionItem = null;
            var matched_trait_name: ?[]const u8 = null;

            for (self.parent.typecheck.impl_interfaces) |impl_interface| {
                if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
                const method_count = impl_interface.methodCountByNameAndReceiver(field.name, .none);
                if (method_count == 0) continue;
                if (matched_impl_item_id != null or method_count > 1) return null;
                const impl_item = self.parent.file.item(impl_interface.impl_item_id).Impl;
                const method_index = impl_interface.methodIndexByNameAndReceiver(field.name, .none) orelse continue;
                matched_impl_item_id = impl_interface.impl_item_id;
                matched_method_item_id = impl_item.methods[method_index];
                matched_function = self.parent.file.item(impl_item.methods[method_index]).Function;
                matched_trait_name = impl_interface.trait_name;
            }

            return .{
                .impl_item_id = matched_impl_item_id orelse return null,
                .method_item_id = matched_method_item_id orelse return null,
                .function = matched_function orelse return null,
                .trait_name = matched_trait_name orelse return null,
                .target_name = target_name,
            };
        }

        fn resolveTraitBoundAssociatedMethodCall(self: *FunctionLowerer, field: ast.FieldExpr) ?ResolvedAssociatedImplMethodCall {
            const function = self.function orelse return null;
            const type_param_name = @This().genericTypeParameterNameForExpr(self, field.base) orelse return null;
            var matched_trait: ?[]const u8 = null;
            for (function.trait_bounds) |bound| {
                if (!std.mem.eql(u8, bound.parameter_name, type_param_name)) continue;
                const trait_interface = self.parent.typecheck.traitInterfaceByName(bound.trait_name) orelse continue;
                if (trait_interface.methodByNameAndReceiver(field.name, .none) == null) continue;
                if (matched_trait != null) return null;
                matched_trait = bound.trait_name;
            }
            const trait_name = matched_trait orelse return null;
            const concrete_type = self.parent.substitutedType(type_param_name) orelse return null;
            const target_name = concrete_type.name() orelse return null;
            return @This().resolveAssociatedMethodFromLookup(self, trait_name, target_name, field.name);
        }

        fn resolveAssociatedMethodFromLookup(self: *FunctionLowerer, trait_name: []const u8, target_name: []const u8, method_name: []const u8) ?ResolvedAssociatedImplMethodCall {
            const impl_item_id = self.parent.item_index.lookupImpl(trait_name, target_name) orelse return null;
            const method_item_id = self.parent.item_index.lookupImplMethodByReceiver(self.parent.file, impl_item_id, method_name, .none) orelse return null;
            const function = self.parent.file.item(method_item_id).Function;
            return .{
                .impl_item_id = impl_item_id,
                .method_item_id = method_item_id,
                .function = function,
                .trait_name = trait_name,
                .target_name = target_name,
            };
        }

        fn genericTypeParameterNameForExpr(self: *FunctionLowerer, expr_id: ast.ExprId) ?[]const u8 {
            return switch (self.parent.file.expression(expr_id).*) {
                .Group => |group| @This().genericTypeParameterNameForExpr(self, group.expr),
                .Name => if (self.parent.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                    .pattern => |pattern_id| blk: {
                        const ty = self.parent.typecheck.pattern_types[pattern_id.index()].type;
                        const type_name = if (ty.name()) |name| name else break :blk null;
                        if (!std.mem.eql(u8, type_name, "type")) break :blk null;
                        const pattern = self.parent.file.pattern(pattern_id).*;
                        if (pattern != .Name) break :blk null;
                        break :blk pattern.Name.name;
                    },
                    else => null,
                } else null,
                else => null,
            };
        }

        fn concreteTypeNameForExpr(self: *FunctionLowerer, expr_id: ast.ExprId) ?[]const u8 {
            return switch (self.parent.file.expression(expr_id).*) {
                .Group => |group| @This().concreteTypeNameForExpr(self, group.expr),
                .TypeValue => |type_value| switch (self.parent.file.typeExpr(type_value.type_expr).*) {
                    .Path => |path| std.mem.trim(u8, path.name, " \t\n\r"),
                    else => null,
                },
                .Name => if (self.parent.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                    .item => |item_id| switch (self.parent.file.item(item_id).*) {
                        .Struct => |item| item.name,
                        .Enum => |item| item.name,
                        .Bitfield => |item| item.name,
                        .Contract => |item| item.name,
                        else => null,
                    },
                    else => null,
                } else null,
                else => null,
            };
        }

        fn functionHasRuntimeSelf(self: *FunctionLowerer, function: ast.FunctionItem) bool {
            return sema.functionHasRuntimeSelf(self.parent.file, function);
        }

        fn calleeName(self: *FunctionLowerer, expr_id: ast.ExprId) ?[]const u8 {
            return switch (self.parent.file.expression(expr_id).*) {
                .Name => |name| name.name,
                .Group => |group| @This().calleeName(self, group.expr),
                else => null,
            };
        }

        const ImportedFunctionTarget = struct {
            module_id: source.ModuleId,
            item_id: ast.ItemId,
            function: ast.FunctionItem,
        };

        fn importedFunctionTargetFromResolution(
            self: *FunctionLowerer,
            resolved: sema.ResolvedCall,
        ) anyerror!?ImportedFunctionTarget {
            const query = self.parent.module_query orelse return null;
            const target_file = try query.astFile(resolved.module_id);
            return switch (target_file.item(resolved.item_id).*) {
                .Function => |function| .{
                    .module_id = resolved.module_id,
                    .item_id = resolved.item_id,
                    .function = function,
                },
                else => null,
            };
        }

        fn currentContractParentBlock(self: *FunctionLowerer) ?mlir.MlirBlock {
            const function = self.function orelse return null;
            const contract_id = function.parent_contract orelse self.parent.active_impl_contract_scope orelse return null;
            const block = self.parent.contract_body_blocks[contract_id.index()];
            if (mlir.oraBlockIsNull(block)) return null;
            return block;
        }

        fn currentContractItemId(self: *FunctionLowerer) ?ast.ItemId {
            const function = self.function orelse return null;
            return function.parent_contract orelse self.parent.active_impl_contract_scope;
        }

        fn calleeFunctionItem(self: *FunctionLowerer, expr_id: ast.ExprId) ?ast.FunctionItem {
            const item_id = @This().calleeFunctionItemId(self, expr_id) orelse return null;
            return switch (self.parent.file.item(item_id).*) {
                .Function => |function| function,
                else => null,
            };
        }

        fn calleeFunctionItemId(self: *FunctionLowerer, expr_id: ast.ExprId) ?ast.ItemId {
            return switch (self.parent.file.expression(expr_id).*) {
                .Group => |group| @This().calleeFunctionItemId(self, group.expr),
                else => blk: {
                    const binding = self.parent.resolution.expr_bindings[expr_id.index()] orelse break :blk null;
                    break :blk switch (binding) {
                        .item => |item_id| switch (self.parent.file.item(item_id).*) {
                            .Function => item_id,
                            else => null,
                        },
                        else => null,
                    };
                },
            };
        }

        fn calleeImportedFunctionTarget(self: *FunctionLowerer, expr_id: ast.ExprId) anyerror!?ImportedFunctionTarget {
            const query = self.parent.module_query orelse return null;
            return switch (self.parent.file.expression(expr_id).*) {
                .Group => |group| try @This().calleeImportedFunctionTarget(self, group.expr),
                .Field => |field| blk: {
                    const target_module_id = (try @This().importedModuleForExpr(self, field.base)) orelse break :blk null;
                    const target_item_id = (try query.lookupItem(target_module_id, field.name)) orelse break :blk null;
                    const target_file = try query.astFile(target_module_id);
                    break :blk switch (target_file.item(target_item_id).*) {
                        .Function => |function| .{
                            .module_id = target_module_id,
                            .item_id = target_item_id,
                            .function = function,
                        },
                        else => null,
                    };
                },
                else => null,
            };
        }

        pub fn lowerBuiltin(self: *FunctionLowerer, expr_id: ast.ExprId, builtin: ast.BuiltinExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (std.mem.eql(u8, builtin.name, "cast") and builtin.args.len > 0) {
                return try @This().lowerCastBuiltin(self, expr_id, builtin, locals, true);
            }
            if (std.mem.eql(u8, builtin.name, "bitCast") and builtin.args.len > 0) {
                return try @This().lowerBitcastBuiltin(self, expr_id, builtin, locals);
            }
            if (std.mem.eql(u8, builtin.name, "addWithOverflow") or
                std.mem.eql(u8, builtin.name, "subWithOverflow") or
                std.mem.eql(u8, builtin.name, "mulWithOverflow") or
                std.mem.eql(u8, builtin.name, "divWithOverflow") or
                std.mem.eql(u8, builtin.name, "modWithOverflow") or
                std.mem.eql(u8, builtin.name, "negWithOverflow") or
                std.mem.eql(u8, builtin.name, "shlWithOverflow") or
                std.mem.eql(u8, builtin.name, "shrWithOverflow") or
                std.mem.eql(u8, builtin.name, "powerWithOverflow"))
            {
                return try @This().lowerOverflowBuiltin(self, expr_id, builtin, locals);
            }
            if (builtin.args.len >= 2 and (std.mem.eql(u8, builtin.name, "divTrunc") or
                std.mem.eql(u8, builtin.name, "divFloor") or
                std.mem.eql(u8, builtin.name, "divCeil") or
                std.mem.eql(u8, builtin.name, "divExact") or
                std.mem.eql(u8, builtin.name, "divmod")))
            {
                return try @This().lowerDivisionBuiltin(self, expr_id, builtin, locals);
            }
            if (builtin.args.len > 0 and std.mem.eql(u8, builtin.name, "truncate")) {
                return try @This().lowerCastBuiltin(self, expr_id, builtin, locals, false);
            }
            if (builtin.args.len == 1 and std.mem.eql(u8, builtin.name, "keccak256")) {
                const value = try self.lowerExpr(builtin.args[0], locals);
                const result_type = self.parent.lowerExprType(expr_id);
                var operands = [_]mlir.MlirValue{value};
                const op = mlir.oraEvmOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    strRef("ora.keccak256"),
                    &operands,
                    operands.len,
                    result_type,
                );
                return try @This().expectValueOp(self, op, builtin.range, "ora.keccak256");
            }
            if (std.mem.eql(u8, builtin.name, "abiEncode")) {
                return try @This().lowerAbiEncodeBuiltin(self, expr_id, builtin, locals);
            }
            if ((std.mem.eql(u8, builtin.name, "abiDecode") or std.mem.eql(u8, builtin.name, "abiDecodePermissive")) and builtin.args.len > 0) {
                return try @This().lowerAbiDecodeBuiltin(self, expr_id, builtin, locals);
            }
            if (builtin.args.len == 2 and std.mem.eql(u8, builtin.name, "concat")) {
                const lhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[0], locals), exprRange(self.parent.file, builtin.args[0]));
                const rhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[1], locals), exprRange(self.parent.file, builtin.args[1]));
                const result_type = self.parent.lowerExprType(expr_id);
                try @This().emitConcatBoundsAssert(self, lhs, rhs, builtin.range);
                const op = mlir.oraConcatOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    lhs,
                    rhs,
                    result_type,
                );
                return try @This().expectValueOp(self, op, builtin.range, "ora.concat");
            }
            if (builtin.args.len == 3 and std.mem.eql(u8, builtin.name, "slice")) {
                const value = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[0], locals), exprRange(self.parent.file, builtin.args[0]));
                var start = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[1], locals), exprRange(self.parent.file, builtin.args[1]));
                var length = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[2], locals), exprRange(self.parent.file, builtin.args[2]));
                const result_type = self.parent.lowerExprType(expr_id);
                const loc = self.parent.location(builtin.range);
                const u256_type = mlir.oraIntegerTypeCreate(self.parent.context, 256);
                start = try self.convertValueForFlow(start, u256_type, builtin.range);
                length = try self.convertValueForFlow(length, u256_type, builtin.range);
                try @This().emitSliceBoundsAssert(self, value, start, length, builtin.range);
                const op = mlir.oraSliceOpCreate(
                    self.parent.context,
                    loc,
                    value,
                    start,
                    length,
                    result_type,
                );
                return try @This().expectValueOp(self, op, builtin.range, "ora.slice");
            }
            try self.parent.emitLoweringError(
                builtin.range,
                "builtin '@{s}' reached HIR lowering without a runtime implementation or const-eval result",
                .{builtin.name},
            );
            return error.MlirOperationCreationFailed;
        }

        fn lowerAbiEncodeBuiltin(self: *FunctionLowerer, expr_id: ast.ExprId, builtin: ast.BuiltinExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (builtin.args.len != 1) return error.MlirOperationCreationFailed;
            const arg_id = builtin.args[0];
            const arg_value = try self.lowerExpr(arg_id, locals);
            const arg_type = self.parent.typecheck.exprType(arg_id);
            var param_types = [_]sema.Type{arg_type};
            var operands = [_]mlir.MlirValue{arg_value};
            const encode_op = try abi_runtime_encoder.createAbiEncodeOp(
                self.parent.allocator,
                self.parent.context,
                self.parent.location(builtin.range),
                @This().layoutContext(self),
                &param_types,
                &operands,
                self.parent.lowerExprType(expr_id),
            );
            if (mlir.oraOperationIsNull(encode_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, encode_op);
        }

        fn lowerAbiDecodeBuiltin(self: *FunctionLowerer, expr_id: ast.ExprId, builtin: ast.BuiltinExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const result_sema_type = self.parent.typecheck.exprType(expr_id);
            const target_type = switch (result_sema_type) {
                .error_union => |error_union| error_union.payload_type.*,
                // Sema guarantees @abiDecode returns Result<T, AbiDecodeError>.
                // Reaching this path means lowering and sema have diverged.
                else => return error.MlirOperationCreationFailed,
            };
            const bytes_arg = if (builtin.type_arg != null) builtin.args[0] else if (builtin.args.len > 1) builtin.args[1] else return error.MlirOperationCreationFailed;
            const bytes_value = try self.lowerExpr(bytes_arg, locals);
            const decode_op = if (std.mem.eql(u8, builtin.name, "abiDecodePermissive"))
                try abi_runtime_decoder.createMemoryResultAbiDecodeOpPermissive(
                    self.parent.allocator,
                    self.parent.context,
                    self.parent.location(builtin.range),
                    @This().layoutContext(self),
                    target_type,
                    bytes_value,
                    self.parent.lowerExprType(expr_id),
                )
            else
                try abi_runtime_decoder.createMemoryResultAbiDecodeOp(
                    self.parent.allocator,
                    self.parent.context,
                    self.parent.location(builtin.range),
                    @This().layoutContext(self),
                    target_type,
                    bytes_value,
                    self.parent.lowerExprType(expr_id),
                );
            if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, decode_op);
        }

        fn emitConcatBoundsAssert(
            self: *FunctionLowerer,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            range: source.TextRange,
        ) anyerror!void {
            const loc = self.parent.location(range);
            const u256_type = mlir.oraIntegerTypeCreate(self.parent.context, 256);
            const lhs_len_op = mlir.oraLengthOpCreate(self.parent.context, loc, lhs, u256_type);
            if (mlir.oraOperationIsNull(lhs_len_op)) return error.MlirOperationCreationFailed;
            const lhs_len = appendValueOp(self.block, lhs_len_op);
            const rhs_len_op = mlir.oraLengthOpCreate(self.parent.context, loc, rhs, u256_type);
            if (mlir.oraOperationIsNull(rhs_len_op)) return error.MlirOperationCreationFailed;
            const rhs_len = appendValueOp(self.block, rhs_len_op);

            const result_len_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, lhs_len, rhs_len);
            if (mlir.oraOperationIsNull(result_len_op)) return error.MlirOperationCreationFailed;
            const result_len = appendValueOp(self.block, result_len_op);
            const length_overflow = appendValueOp(self.block, self.createCompareOp(loc, "ult", result_len, lhs_len));
            try FunctionLowerer.emitOverflowAssert(self, length_overflow, "@concat length overflow", range);

            const header_size = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, u256_type, 32));
            const total_size_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, result_len, header_size);
            if (mlir.oraOperationIsNull(total_size_op)) return error.MlirOperationCreationFailed;
            const total_size = appendValueOp(self.block, total_size_op);
            const allocation_overflow = appendValueOp(self.block, self.createCompareOp(loc, "ult", total_size, result_len));
            try FunctionLowerer.emitOverflowAssert(self, allocation_overflow, "@concat allocation size overflow", range);
        }

        fn emitSliceBoundsAssert(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            start: mlir.MlirValue,
            length: mlir.MlirValue,
            range: source.TextRange,
        ) anyerror!void {
            const loc = self.parent.location(range);
            const u256_type = mlir.oraIntegerTypeCreate(self.parent.context, 256);
            const end_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, start, length);
            if (mlir.oraOperationIsNull(end_op)) return error.MlirOperationCreationFailed;
            const end_offset = appendValueOp(self.block, end_op);

            const overflow = appendValueOp(self.block, self.createCompareOp(loc, "ult", end_offset, start));
            try FunctionLowerer.emitOverflowAssert(self, overflow, "@slice start + length overflow", range);

            const source_len_op = mlir.oraLengthOpCreate(self.parent.context, loc, value, u256_type);
            if (mlir.oraOperationIsNull(source_len_op)) return error.MlirOperationCreationFailed;
            const source_len = appendValueOp(self.block, source_len_op);
            const in_bounds = appendValueOp(self.block, self.createCompareOp(loc, "ule", end_offset, source_len));
            const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, in_bounds, strRef("@slice requires start + length <= value.len"));
            if (mlir.oraOperationIsNull(assert_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, assert_op);
        }

        fn lowerDivisionBuiltin(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            builtin: ast.BuiltinExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(builtin.range);
            const result_type = self.parent.lowerExprType(expr_id);
            if (builtin.args.len < 2) return error.MlirOperationCreationFailed;

            var lhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[0], locals), exprRange(self.parent.file, builtin.args[0]));
            var rhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[1], locals), exprRange(self.parent.file, builtin.args[1]));
            const lhs_type = mlir.oraValueGetType(lhs);
            const rhs_type = mlir.oraValueGetType(rhs);
            if (mlir.oraTypeIsAInteger(lhs_type) and mlir.oraTypeIsAInteger(rhs_type) and !mlir.oraTypeEqual(lhs_type, rhs_type)) {
                const lhs_expr = self.parent.file.expression(builtin.args[0]).*;
                const rhs_expr = self.parent.file.expression(builtin.args[1]).*;
                if (lhs_expr == .IntegerLiteral and rhs_expr != .IntegerLiteral) {
                    lhs = try self.convertValueForFlow(lhs, rhs_type, builtin.range);
                } else {
                    rhs = try self.convertValueForFlow(rhs, lhs_type, builtin.range);
                }
            }

            const value_ty = mlir.oraValueGetType(lhs);
            const is_signed = blk: {
                if (try @This().exprIntegerSignedness(self, builtin.args[0])) |signed| break :blk signed;
                if (try @This().exprIntegerSignedness(self, builtin.args[1])) |signed| break :blk signed;
                break :blk try @This().reportIndeterminateIntegerSignedness(self, builtin.range);
            };
            const div_op = if (is_signed)
                mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs)
            else
                mlir.oraArithDivUIOpCreate(self.parent.context, loc, lhs, rhs);
            const quotient = try @This().expectValueOp(self, div_op, builtin.range, "arith.div");

            const rem_op = if (is_signed)
                mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs)
            else
                mlir.oraArithRemUIOpCreate(self.parent.context, loc, lhs, rhs);
            const remainder = try @This().expectValueOp(self, rem_op, builtin.range, "arith.rem");

            if (std.mem.eql(u8, builtin.name, "divmod")) {
                const tuple_op = mlir.oraTupleCreateOpCreate(
                    self.parent.context,
                    loc,
                    &[_]mlir.MlirValue{ quotient, remainder },
                    2,
                    result_type,
                );
                return try @This().expectValueOp(self, tuple_op, builtin.range, "ora.tuple.create");
            }

            if (std.mem.eql(u8, builtin.name, "divTrunc")) return quotient;

            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            const remainder_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", remainder, zero));

            if (std.mem.eql(u8, builtin.name, "divExact")) {
                const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, appendValueOp(self.block, self.createCompareOp(loc, "eq", remainder, zero)), strRef("exact division requires zero remainder"));
                _ = try @This().expectOperation(self, assert_op, builtin.range, "ora.assert");
                appendOp(self.block, assert_op);
                return quotient;
            }

            if (!is_signed) {
                if (std.mem.eql(u8, builtin.name, "divFloor")) return quotient;
                const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));
                const adjusted = appendValueOp(self.block, mlir.oraArithAddIOpCreate(self.parent.context, loc, quotient, one));
                return try @This().selectValue(self, remainder_non_zero, adjusted, quotient, value_ty, loc);
            }

            const lhs_negative = appendValueOp(self.block, self.createCompareOp(loc, "slt", lhs, zero));
            const rhs_negative = appendValueOp(self.block, self.createCompareOp(loc, "slt", rhs, zero));
            const signs_differ = appendValueOp(self.block, mlir.oraArithXorIOpCreate(self.parent.context, loc, lhs_negative, rhs_negative));
            const same_sign = appendValueOp(self.block, self.createCompareOp(loc, "eq", lhs_negative, rhs_negative));
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));

            if (std.mem.eql(u8, builtin.name, "divFloor")) {
                const adjust_needed = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, remainder_non_zero, signs_differ));
                const adjusted = appendValueOp(self.block, mlir.oraArithSubIOpCreate(self.parent.context, loc, quotient, one));
                return try @This().selectValue(self, adjust_needed, adjusted, quotient, value_ty, loc);
            }

            const adjust_needed = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, remainder_non_zero, same_sign));
            const adjusted = appendValueOp(self.block, mlir.oraArithAddIOpCreate(self.parent.context, loc, quotient, one));
            return try @This().selectValue(self, adjust_needed, adjusted, quotient, value_ty, loc);
        }

        fn selectValue(
            self: *FunctionLowerer,
            condition: mlir.MlirValue,
            then_value: mlir.MlirValue,
            else_value: mlir.MlirValue,
            value_ty: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!mlir.MlirValue {
            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, condition, &[_]mlir.MlirType{value_ty}, 1, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);
            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;
            try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{then_value});
            try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{else_value});
            return mlir.oraOperationGetResult(if_op, 0);
        }

        fn lowerCastBuiltin(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            builtin: ast.BuiltinExpr,
            locals: *LocalEnv,
            checked: bool,
        ) anyerror!mlir.MlirValue {
            const target_type = self.parent.lowerExprType(expr_id);
            const target_ref_base = mlir.oraRefinementTypeGetBaseType(target_type);
            const concrete_target = if (!mlir.oraTypeIsNull(target_ref_base)) target_ref_base else target_type;

            var value = try self.lowerExpr(builtin.args[0], locals);
            value = try @This().unwrapRefinementForCast(self, value, builtin.range);
            if (try @This().lowerStaticArrayToSliceCast(self, value, concrete_target, builtin.range)) |slice_value| {
                value = slice_value;
            } else {
                value = try @This().convertBuiltinCastValue(
                    self,
                    value,
                    concrete_target,
                    if (mlir.oraTypeIsAInteger(mlir.oraValueGetType(value))) try @This().integerExprSignedness(self, builtin.args[0]) else false,
                    if (mlir.oraTypeIsAInteger(concrete_target)) try @This().integerExprSignedness(self, expr_id) else false,
                    builtin.range,
                    checked,
                );
            }

            if (!mlir.oraTypeIsNull(target_ref_base) and !mlir.oraTypeEqual(mlir.oraValueGetType(value), target_type)) {
                const wrap_op = mlir.oraBaseToRefinementOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    value,
                    target_type,
                    self.block,
                );
                if (mlir.oraOperationIsNull(wrap_op)) return error.MlirOperationCreationFailed;
                return mlir.oraOperationGetResult(wrap_op, 0);
            }
            return value;
        }

        fn lowerStaticArrayToSliceCast(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            target_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!?mlir.MlirValue {
            if (!mlir.oraTypeIsAShaped(target_type) or
                mlir.oraShapedTypeGetRank(target_type) != 1 or
                mlir.oraShapedTypeHasStaticShape(target_type))
            {
                return null;
            }

            const value_type = mlir.oraValueGetType(value);
            if (!mlir.oraTypeIsAShaped(value_type) or
                mlir.oraShapedTypeGetRank(value_type) != 1 or
                !mlir.oraShapedTypeHasStaticShape(value_type))
            {
                return null;
            }

            const value_element_type = mlir.oraShapedTypeGetElementType(value_type);
            const target_element_type = mlir.oraShapedTypeGetElementType(target_type);
            const len = mlir.oraShapedTypeGetDimSize(value_type, 0);
            if (len < 0) return null;
            // Empty array literals carry no element evidence, so [] may lower
            // with the fallback i256 element type. The cast still materializes
            // a real zero-length target slice, such as memref<?x!ora.string>.
            if (len != 0 and !mlir.oraTypeEqual(value_element_type, target_element_type)) return null;

            const loc = self.parent.location(range);
            const u256_type = mlir.oraIntegerTypeCreate(self.parent.context, 256);
            const raw_length = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, u256_type, @intCast(len)));
            const length = try @This().convertIndexToIndexType(self, raw_length, range);
            const alloc = mlir.oraMemrefAllocaDynamicOpCreate(
                self.parent.context,
                loc,
                target_type,
                &[_]mlir.MlirValue{length},
                1,
            );
            if (mlir.oraOperationIsNull(alloc)) return error.MlirOperationCreationFailed;
            const slice = appendValueOp(self.block, alloc);

            for (0..@intCast(len)) |index| {
                const raw_index = appendValueOp(
                    self.block,
                    createIntegerConstant(self.parent.context, loc, u256_type, @intCast(index)),
                );
                const index_value = try @This().convertIndexToIndexType(self, raw_index, range);
                const loaded = mlir.oraMemrefLoadOpCreate(
                    self.parent.context,
                    loc,
                    value,
                    &[_]mlir.MlirValue{index_value},
                    1,
                    value_element_type,
                );
                if (mlir.oraOperationIsNull(loaded)) return error.MlirOperationCreationFailed;
                const loaded_value = appendValueOp(self.block, loaded);
                const store = mlir.oraMemrefStoreOpCreate(
                    self.parent.context,
                    loc,
                    loaded_value,
                    slice,
                    &[_]mlir.MlirValue{index_value},
                    1,
                );
                if (mlir.oraOperationIsNull(store)) return error.MlirOperationCreationFailed;
                appendOp(self.block, store);
            }

            return slice;
        }

        fn lowerBitcastBuiltin(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            builtin: ast.BuiltinExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const target_type = self.parent.lowerExprType(expr_id);
            const target_ref_base = mlir.oraRefinementTypeGetBaseType(target_type);
            const concrete_target = if (!mlir.oraTypeIsNull(target_ref_base)) target_ref_base else target_type;

            var value = try self.lowerExpr(builtin.args[0], locals);
            value = try @This().unwrapRefinementForCast(self, value, builtin.range);

            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, concrete_target)) {
                if (!mlir.oraTypeIsNull(target_ref_base)) {
                    const wrap_op = mlir.oraBaseToRefinementOpCreate(
                        self.parent.context,
                        self.parent.location(builtin.range),
                        value,
                        target_type,
                        self.block,
                    );
                    if (mlir.oraOperationIsNull(wrap_op)) return error.MlirOperationCreationFailed;
                    return mlir.oraOperationGetResult(wrap_op, 0);
                }
                return value;
            }

            const bitcast = mlir.oraArithBitcastOpCreate(
                self.parent.context,
                self.parent.location(builtin.range),
                value,
                concrete_target,
            );
            if (mlir.oraOperationIsNull(bitcast)) return error.MlirOperationCreationFailed;
            const casted = appendValueOp(self.block, bitcast);

            if (!mlir.oraTypeIsNull(target_ref_base)) {
                const wrap_op = mlir.oraBaseToRefinementOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    casted,
                    target_type,
                    self.block,
                );
                if (mlir.oraOperationIsNull(wrap_op)) return error.MlirOperationCreationFailed;
                return mlir.oraOperationGetResult(wrap_op, 0);
            }
            return casted;
        }

        fn unwrapRefinementForCast(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            const base_type = mlir.oraRefinementTypeGetBaseType(value_type);
            if (mlir.oraTypeIsNull(base_type)) return value;
            const op = mlir.oraRefinementToBaseOpCreate(self.parent.context, self.parent.location(range), value, self.block);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return mlir.oraOperationGetResult(op, 0);
        }

        fn convertBuiltinCastValue(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            target_type: mlir.MlirType,
            source_is_signed: bool,
            target_is_signed: bool,
            range: source.TextRange,
            checked: bool,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, target_type)) return value;

            const loc = self.parent.location(range);
            const value_is_int = mlir.oraTypeIsAInteger(value_type);
            const target_is_int = mlir.oraTypeIsAInteger(target_type);

            if (mlir.oraTypeIsAddressType(value_type) and target_is_int) {
                const op = mlir.oraAddrToI160OpCreate(self.parent.context, loc, value);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                const i160_value = appendValueOp(self.block, op);
                if (mlir.oraIntegerTypeGetWidth(target_type) == 160) return i160_value;
                return try @This().convertBuiltinCastValue(self, i160_value, target_type, false, target_is_signed, range, checked);
            }

            if (value_is_int and mlir.oraTypeIsAddressType(target_type)) {
                const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
                if (mlir.oraTypeIsNull(i160_type)) return error.MlirOperationCreationFailed;
                const i160_value = if (mlir.oraIntegerTypeGetWidth(value_type) == 160)
                    value
                else
                    try @This().convertBuiltinCastValue(self, value, i160_type, source_is_signed, false, range, checked);
                const op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, i160_value);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (!(value_is_int and target_is_int)) return value;

            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const target_width = mlir.oraIntegerTypeGetWidth(target_type);
            if (value_width == target_width) {
                const op = mlir.oraArithBitcastOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (value_width < target_width) {
                const op = if (source_is_signed)
                    mlir.oraArithExtSIOpCreate(self.parent.context, loc, value, target_type)
                else
                    mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (checked) {
                try @This().emitCastOverflowCheck(self, value, target_type, target_is_signed, range);
            }

            const trunc = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, target_type);
            if (mlir.oraOperationIsNull(trunc)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, trunc);
        }

        fn emitCastOverflowCheck(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            target_type: mlir.MlirType,
            target_is_signed: bool,
            range: source.TextRange,
        ) anyerror!void {
            const value_type = mlir.oraValueGetType(value);
            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const target_width = mlir.oraIntegerTypeGetWidth(target_type);
            if (target_width >= value_width) return;

            const loc = self.parent.location(range);
            if (!target_is_signed) {
                const trunc_op = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(trunc_op)) return error.MlirOperationCreationFailed;
                const truncated = appendValueOp(self.block, trunc_op);
                const ext_op = mlir.oraArithExtUIOpCreate(self.parent.context, loc, truncated, value_type);
                if (mlir.oraOperationIsNull(ext_op)) return error.MlirOperationCreationFailed;
                const widened = appendValueOp(self.block, ext_op);
                const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", value, widened));
                try @This().emitCastOverflowAssert(self, mismatch, "safe cast narrowing overflow", range);
                return;
            }

            const max_u256 = (@as(u256, 1) << @intCast(target_width - 1)) - 1;
            const min_u256_abs = @as(u256, 1) << @intCast(target_width - 1);
            const max_constant = try @This().createWideIntegerConstant(self, value_type, max_u256, false, range);
            const min_constant = try @This().createWideIntegerConstant(self, value_type, min_u256_abs, true, range);
            const above_max = appendValueOp(self.block, self.createCompareOp(loc, "sgt", value, max_constant));
            const below_min = appendValueOp(self.block, self.createCompareOp(loc, "slt", value, min_constant));
            const overflow_op = mlir.oraArithOrIOpCreate(self.parent.context, loc, above_max, below_min);
            if (mlir.oraOperationIsNull(overflow_op)) return error.MlirOperationCreationFailed;
            const overflow = appendValueOp(self.block, overflow_op);
            try @This().emitCastOverflowAssert(self, overflow, "safe cast narrowing overflow", range);
        }

        fn createWideIntegerConstant(
            self: *FunctionLowerer,
            ty: mlir.MlirType,
            value: u256,
            negative: bool,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const attr = if (!negative and value <= std.math.maxInt(i64))
                mlir.oraIntegerAttrCreateI64FromType(ty, @intCast(value))
            else blk: {
                var buf: [80]u8 = undefined;
                const text = if (negative)
                    std.fmt.bufPrint(&buf, "-{}", .{value}) catch return error.MlirOperationCreationFailed
                else
                    std.fmt.bufPrint(&buf, "{}", .{value}) catch return error.MlirOperationCreationFailed;
                break :blk mlir.oraIntegerAttrGetFromString(ty, strRef(text));
            };
            const op = mlir.oraArithConstantOpCreate(self.parent.context, self.parent.location(range), ty, attr);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn emitCastOverflowAssert(
            self: *FunctionLowerer,
            overflow_flag: mlir.MlirValue,
            message: []const u8,
            range: source.TextRange,
        ) anyerror!void {
            const loc = self.parent.location(range);
            const true_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
            const not_op = mlir.oraArithXorIOpCreate(self.parent.context, loc, overflow_flag, true_value);
            if (mlir.oraOperationIsNull(not_op)) return error.MlirOperationCreationFailed;
            const no_overflow = appendValueOp(self.block, not_op);
            const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, no_overflow, strRef(message));
            if (mlir.oraOperationIsNull(assert_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, assert_op);
        }

        fn lowerOverflowBuiltin(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            builtin: ast.BuiltinExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(builtin.range);
            const result_type = self.parent.lowerExprType(expr_id);
            const is_unary = std.mem.eql(u8, builtin.name, "negWithOverflow");
            const expected_args: usize = if (is_unary) 1 else 2;
            if (builtin.args.len < expected_args) return error.MlirOperationCreationFailed;

            const lhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[0], locals), exprRange(self.parent.file, builtin.args[0]));
            const lhs_type = mlir.oraValueGetType(lhs);
            const is_signed = try @This().integerExprSignedness(self, builtin.args[0]);

            var value: mlir.MlirValue = lhs;
            var overflow_flag: mlir.MlirValue = undefined;

            if (std.mem.eql(u8, builtin.name, "negWithOverflow")) {
                const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, lhs_type, 0));
                const sub_op = mlir.oraArithSubIOpCreate(self.parent.context, loc, zero, lhs);
                value = try @This().expectValueOp(self, sub_op, builtin.range, "arith.sub");
                if (is_signed) {
                    const bit_width: i64 = @intCast(mlir.oraIntegerTypeGetWidth(lhs_type) - 1);
                    const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, lhs_type, 1));
                    const shift = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, lhs_type, bit_width));
                    const min_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, one, shift);
                    if (mlir.oraOperationIsNull(min_op)) return error.MlirOperationCreationFailed;
                    const min_int = appendValueOp(self.block, min_op);
                    overflow_flag = appendValueOp(self.block, self.createCompareOp(loc, "eq", lhs, min_int));
                } else {
                    const zero_cmp = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, lhs_type, 0));
                    overflow_flag = appendValueOp(self.block, self.createCompareOp(loc, "ne", lhs, zero_cmp));
                }
                return @This().packOverflowResult(self, value, overflow_flag, result_type, builtin.range);
            }

            const rhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[1], locals), exprRange(self.parent.file, builtin.args[1]));
            if (std.mem.eql(u8, builtin.name, "addWithOverflow")) {
                const add_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, lhs, rhs);
                value = try @This().expectValueOp(self, add_op, builtin.range, "arith.add");
                overflow_flag = if (is_signed)
                    try @This().computeSignedAddOverflow(self, value, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", value, lhs));
            } else if (std.mem.eql(u8, builtin.name, "subWithOverflow")) {
                const sub_op = mlir.oraArithSubIOpCreate(self.parent.context, loc, lhs, rhs);
                value = try @This().expectValueOp(self, sub_op, builtin.range, "arith.sub");
                overflow_flag = if (is_signed)
                    try @This().computeSignedSubOverflow(self, value, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", lhs, rhs));
            } else if (std.mem.eql(u8, builtin.name, "mulWithOverflow")) {
                const mul_op = mlir.oraArithMulIOpCreate(self.parent.context, loc, lhs, rhs);
                value = try @This().expectValueOp(self, mul_op, builtin.range, "arith.mul");
                overflow_flag = if (is_signed)
                    try @This().computeSignedMulOverflow(self, value, lhs, rhs, lhs_type, loc)
                else
                    try @This().computeUnsignedMulOverflow(self, value, lhs, rhs, lhs_type, loc);
            } else if (std.mem.eql(u8, builtin.name, "powerWithOverflow")) {
                const power = try self.lowerPowerWithOverflow(lhs, rhs, lhs_type, is_signed, builtin.range);
                value = power.value;
                overflow_flag = power.overflow;
            } else if (std.mem.eql(u8, builtin.name, "shlWithOverflow")) {
                const shl_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, lhs, rhs);
                value = try @This().expectValueOp(self, shl_op, builtin.range, "arith.shl");
                const shr_op = if (is_signed)
                    mlir.oraArithShrSIOpCreate(self.parent.context, loc, value, rhs)
                else
                    mlir.oraArithShrUIOpCreate(self.parent.context, loc, value, rhs);
                if (mlir.oraOperationIsNull(shr_op)) return error.MlirOperationCreationFailed;
                const shifted_back = appendValueOp(self.block, shr_op);
                overflow_flag = appendValueOp(self.block, self.createCompareOp(loc, "ne", shifted_back, lhs));
            } else if (std.mem.eql(u8, builtin.name, "shrWithOverflow")) {
                const shr_op = if (is_signed)
                    mlir.oraArithShrSIOpCreate(self.parent.context, loc, lhs, rhs)
                else
                    mlir.oraArithShrUIOpCreate(self.parent.context, loc, lhs, rhs);
                value = try @This().expectValueOp(self, shr_op, builtin.range, "arith.shr");
                overflow_flag = try @This().makeFalse(self, loc);
            } else if (std.mem.eql(u8, builtin.name, "divWithOverflow") or std.mem.eql(u8, builtin.name, "modWithOverflow")) {
                const arith_op = if (std.mem.eql(u8, builtin.name, "divWithOverflow"))
                    (if (is_signed) mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithDivUIOpCreate(self.parent.context, loc, lhs, rhs))
                else
                    (if (is_signed) mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithRemUIOpCreate(self.parent.context, loc, lhs, rhs));
                value = try @This().expectValueOp(self, arith_op, builtin.range, "arith.divrem");
                overflow_flag = try @This().makeFalse(self, loc);
            } else {
                return error.MlirOperationCreationFailed;
            }

            return @This().packOverflowResult(self, value, overflow_flag, result_type, builtin.range);
        }

        fn packOverflowResult(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            overflow_flag: mlir.MlirValue,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const op = mlir.oraTupleCreateOpCreate(
                self.parent.context,
                self.parent.location(range),
                &[_]mlir.MlirValue{ value, overflow_flag },
                2,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn computeSignedAddOverflow(
            self: *FunctionLowerer,
            result: mlir.MlirValue,
            a: mlir.MlirValue,
            b: mlir.MlirValue,
            loc: mlir.MlirLocation,
        ) anyerror!mlir.MlirValue {
            const xor_ra = mlir.oraArithXorIOpCreate(self.parent.context, loc, result, a);
            if (mlir.oraOperationIsNull(xor_ra)) return error.MlirOperationCreationFailed;
            const xor_rb = mlir.oraArithXorIOpCreate(self.parent.context, loc, result, b);
            if (mlir.oraOperationIsNull(xor_rb)) return error.MlirOperationCreationFailed;
            const left = appendValueOp(self.block, xor_ra);
            const right = appendValueOp(self.block, xor_rb);
            const and_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, left, right);
            if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
            const and_value = appendValueOp(self.block, and_op);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, mlir.oraValueGetType(result), 0));
            return appendValueOp(self.block, self.createCompareOp(loc, "slt", and_value, zero));
        }

        fn computeSignedSubOverflow(
            self: *FunctionLowerer,
            result: mlir.MlirValue,
            a: mlir.MlirValue,
            b: mlir.MlirValue,
            loc: mlir.MlirLocation,
        ) anyerror!mlir.MlirValue {
            const xor_ab = mlir.oraArithXorIOpCreate(self.parent.context, loc, a, b);
            if (mlir.oraOperationIsNull(xor_ab)) return error.MlirOperationCreationFailed;
            const xor_ra = mlir.oraArithXorIOpCreate(self.parent.context, loc, result, a);
            if (mlir.oraOperationIsNull(xor_ra)) return error.MlirOperationCreationFailed;
            const left = appendValueOp(self.block, xor_ab);
            const right = appendValueOp(self.block, xor_ra);
            const and_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, left, right);
            if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
            const and_value = appendValueOp(self.block, and_op);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, mlir.oraValueGetType(result), 0));
            return appendValueOp(self.block, self.createCompareOp(loc, "slt", and_value, zero));
        }

        fn computeSignedMulOverflow(
            self: *FunctionLowerer,
            result: mlir.MlirValue,
            a: mlir.MlirValue,
            b: mlir.MlirValue,
            val_ty: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!mlir.MlirValue {
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, val_ty, 1));
            const shift = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, val_ty, @intCast(mlir.oraIntegerTypeGetWidth(val_ty) - 1)));
            const min_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, one, shift);
            if (mlir.oraOperationIsNull(min_op)) return error.MlirOperationCreationFailed;
            const min_int = appendValueOp(self.block, min_op);
            const neg_one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, val_ty, -1));
            const a_min = appendValueOp(self.block, self.createCompareOp(loc, "eq", a, min_int));
            const b_neg_one = appendValueOp(self.block, self.createCompareOp(loc, "eq", b, neg_one));
            const special_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, a_min, b_neg_one);
            if (mlir.oraOperationIsNull(special_op)) return error.MlirOperationCreationFailed;
            const special = appendValueOp(self.block, special_op);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, val_ty, 0));
            const b_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", b, zero));
            const quot_op = mlir.oraArithDivSIOpCreate(self.parent.context, loc, result, b);
            if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
            mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
            const quotient = appendValueOp(self.block, quot_op);
            const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", quotient, a));
            const general_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, mismatch, b_non_zero);
            if (mlir.oraOperationIsNull(general_op)) return error.MlirOperationCreationFailed;
            const general = appendValueOp(self.block, general_op);
            const overflow_op = mlir.oraArithOrIOpCreate(self.parent.context, loc, special, general);
            if (mlir.oraOperationIsNull(overflow_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, overflow_op);
        }

        fn computeUnsignedMulOverflow(
            self: *FunctionLowerer,
            _: mlir.MlirValue,
            a: mlir.MlirValue,
            b: mlir.MlirValue,
            value_ty: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!mlir.MlirValue {
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            const b_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", b, zero));

            const width = mlir.oraIntegerTypeGetWidth(value_ty);
            const max_value: u256 = if (width >= 256)
                std.math.maxInt(u256)
            else
                (@as(u256, 1) << @intCast(width)) - 1;
            const max_const = try @This().createWideIntegerConstant(self, value_ty, max_value, false, source.TextRange.empty(0));
            const quot_op = mlir.oraArithDivUIOpCreate(self.parent.context, loc, max_const, b);
            if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
            mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
            const max_div_b = appendValueOp(self.block, quot_op);
            const above_limit = appendValueOp(self.block, self.createCompareOp(loc, "ugt", a, max_div_b));
            const overflow_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, b_non_zero, above_limit);
            if (mlir.oraOperationIsNull(overflow_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, overflow_op);
        }

        fn makeFalse(self: *FunctionLowerer, loc: mlir.MlirLocation) anyerror!mlir.MlirValue {
            return appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
        }

        fn lowerStructLiteral(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            struct_literal: ast.StructLiteralExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const expr_type = self.parent.typecheck.exprType(expr_id);
            if (expr_type.kind() == .enum_) {
                if (@This().adtStructLiteralVariantName(struct_literal.type_name)) |_| {
                    return try @This().lowerAdtNamedPayloadStructLiteral(self, expr_id, struct_literal, locals);
                }
            }
            if (expr_type.kind() == .anonymous_struct) {
                return try @This().lowerAnonymousStructLiteral(self, expr_id, struct_literal, expr_type.anonymous_struct.fields, locals);
            }
            const concrete_name = expr_type.name() orelse struct_literal.type_name;
            const struct_item_id = self.parent.item_index.lookup(concrete_name) orelse {
                if (self.parent.typecheck.instantiatedStructByName(concrete_name)) |instantiated| {
                    return try @This().lowerInstantiatedStructLiteral(self, expr_id, struct_literal, instantiated.fields, concrete_name, locals);
                }
                return error.MlirOperationCreationFailed;
            };
            const item = self.parent.file.item(struct_item_id).*;
            if (item == .Bitfield) {
                const bitfield_type = self.parent.typecheck.exprType(expr_id);
                const word_type = self.parent.lowerExprType(expr_id);
                var packed_word = appendValueOp(
                    self.block,
                    createIntegerConstant(self.parent.context, self.parent.location(struct_literal.range), word_type, 0),
                );
                for (struct_literal.fields) |init| {
                    const field_value = try @This().lowerExprForFlowTarget(self, init.value, word_type, locals);
                    packed_word = try self.createBitfieldFieldUpdate(packed_word, bitfield_type, init.name, field_value, init.range);
                }
                return packed_word;
            }
            const struct_item = switch (item) {
                .Struct => |struct_item| blk: {
                    if (struct_item.is_generic) {
                        if (self.parent.typecheck.instantiatedStructByName(concrete_name)) |instantiated| {
                            return try @This().lowerInstantiatedStructLiteral(self, expr_id, struct_literal, instantiated.fields, concrete_name, locals);
                        }
                    }
                    break :blk struct_item;
                },
                else => {
                    if (self.parent.typecheck.instantiatedStructByName(concrete_name)) |instantiated| {
                        return try @This().lowerInstantiatedStructLiteral(self, expr_id, struct_literal, instantiated.fields, concrete_name, locals);
                    }
                    return error.MlirOperationCreationFailed;
                },
            };

            const result_type = self.parent.lowerExprType(expr_id);
            var operands: std.ArrayList(mlir.MlirValue) = .{};
            const init_lookup = try lookup_index.buildNamed(ast.StructFieldInit, self.parent.allocator, struct_literal.fields, "name");
            defer self.parent.allocator.free(init_lookup);
            for (struct_item.fields) |decl_field| {
                const init = lookup_index.findNamedItem(ast.StructFieldInit, struct_literal.fields, init_lookup, decl_field.name) orelse {
                    return error.MlirOperationCreationFailed;
                };
                const field_sema_type = try type_descriptors.descriptorFromTypeExpr(self.parent.allocator, self.parent.file, self.parent.item_index, decl_field.type_expr);
                const value = try @This().lowerExprForSemaFlowTarget(self, init.value, field_sema_type, init.range, locals);
                try operands.append(self.parent.allocator, value);
            }

            const op = mlir.oraStructInstantiateOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                strRef(concrete_name),
                if (operands.items.len == 0) null else operands.items.ptr,
                operands.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn lowerAdtNamedPayloadStructLiteral(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            struct_literal: ast.StructLiteralExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const variant_name = @This().adtStructLiteralVariantName(struct_literal.type_name) orelse return error.MlirOperationCreationFailed;
            const result_type = self.parent.lowerExprType(expr_id);
            if (!mlir.oraTypeIsAAdt(result_type)) return error.MlirOperationCreationFailed;

            const payload_type = @This().adtVariantPayloadType(result_type, variant_name) orelse return error.MlirOperationCreationFailed;
            if (mlir.oraTypeIsANone(payload_type)) {
                const op = mlir.oraAdtConstructOpCreate(
                    self.parent.context,
                    self.parent.location(struct_literal.range),
                    strRef(variant_name),
                    null,
                    0,
                    result_type,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            const field_count = mlir.oraAnonymousStructTypeGetFieldCount(payload_type);
            if (field_count == 0) return error.MlirOperationCreationFailed;

            const payload_sema_type = try @This().adtVariantPayloadSemaType(self, self.parent.typecheck.exprType(expr_id), variant_name);
            var fields: std.ArrayList(mlir.MlirValue) = .{};
            defer fields.deinit(self.parent.allocator);
            const init_lookup = try lookup_index.buildNamed(ast.StructFieldInit, self.parent.allocator, struct_literal.fields, "name");
            defer self.parent.allocator.free(init_lookup);
            for (0..field_count) |index| {
                const field_name_ref = mlir.oraAnonymousStructTypeGetFieldName(payload_type, index);
                const field_name = field_name_ref.data[0..field_name_ref.length];
                const init = lookup_index.findNamedItem(ast.StructFieldInit, struct_literal.fields, init_lookup, field_name) orelse return error.MlirOperationCreationFailed;
                const value = if (payload_sema_type != null and payload_sema_type.?.kind() == .anonymous_struct and index < payload_sema_type.?.anonymous_struct.fields.len)
                    try @This().lowerExprForSemaFlowTarget(self, init.value, payload_sema_type.?.anonymous_struct.fields[index].ty, init.range, locals)
                else blk: {
                    const field_type = mlir.oraAnonymousStructTypeGetFieldType(payload_type, index);
                    break :blk try @This().lowerExprForFlowTarget(self, init.value, field_type, locals);
                };
                try fields.append(self.parent.allocator, value);
            }

            const payload_op = mlir.oraStructInitOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                if (fields.items.len == 0) null else fields.items.ptr,
                fields.items.len,
                payload_type,
            );
            if (mlir.oraOperationIsNull(payload_op)) return error.MlirOperationCreationFailed;
            const payload = appendValueOp(self.block, payload_op);

            const operands = [_]mlir.MlirValue{payload};
            const op = mlir.oraAdtConstructOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                strRef(variant_name),
                &operands,
                operands.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn adtStructLiteralVariantName(type_name: []const u8) ?[]const u8 {
            const dot_index = std.mem.lastIndexOfScalar(u8, type_name, '.') orelse return null;
            if (dot_index + 1 >= type_name.len) return null;
            return type_name[dot_index + 1 ..];
        }

        fn lowerInstantiatedStructLiteral(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            struct_literal: ast.StructLiteralExpr,
            fields: []const sema.InstantiatedStructField,
            concrete_name: []const u8,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const result_type = self.parent.lowerExprType(expr_id);
            var operands: std.ArrayList(mlir.MlirValue) = .{};
            const init_lookup = try lookup_index.buildNamed(ast.StructFieldInit, self.parent.allocator, struct_literal.fields, "name");
            defer self.parent.allocator.free(init_lookup);
            for (fields) |decl_field| {
                const init = lookup_index.findNamedItem(ast.StructFieldInit, struct_literal.fields, init_lookup, decl_field.name) orelse {
                    return error.MlirOperationCreationFailed;
                };
                const value = try @This().lowerExprForSemaFlowTarget(self, init.value, decl_field.ty, init.range, locals);
                try operands.append(self.parent.allocator, value);
            }
            const op = mlir.oraStructInstantiateOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                strRef(concrete_name),
                if (operands.items.len == 0) null else operands.items.ptr,
                operands.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn lowerAnonymousStructLiteral(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            struct_literal: ast.StructLiteralExpr,
            fields: []const sema.AnonymousStructField,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const result_type = self.parent.lowerExprType(expr_id);
            var operands: std.ArrayList(mlir.MlirValue) = .{};
            const init_lookup = try lookup_index.buildNamed(ast.StructFieldInit, self.parent.allocator, struct_literal.fields, "name");
            defer self.parent.allocator.free(init_lookup);
            for (fields) |decl_field| {
                const init = lookup_index.findNamedItem(ast.StructFieldInit, struct_literal.fields, init_lookup, decl_field.name) orelse {
                    return error.MlirOperationCreationFailed;
                };
                const value = try @This().lowerExprForSemaFlowTarget(self, init.value, decl_field.ty, init.range, locals);
                try operands.append(self.parent.allocator, value);
            }
            const op = mlir.oraStructInitOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                if (operands.items.len == 0) null else operands.items.ptr,
                operands.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn lowerFieldExpr(self: *FunctionLowerer, expr_id: ast.ExprId, field: ast.FieldExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (try @This().lowerBuiltinFieldExpr(self, expr_id, field)) |value| {
                return value;
            }
            const result_type = self.parent.lowerExprType(expr_id);
            if (try @This().lowerImportedFieldExpr(self, field, result_type)) |value| {
                return value;
            }
            if (self.parent.resolution.expr_bindings[expr_id.index()]) |binding| {
                switch (binding) {
                    .item => |item_id| switch (self.parent.file.item(item_id).*) {
                        .Constant => |constant| return self.lowerExpr(constant.value, locals),
                        .Function => |function| {
                            const op = mlir.oraFunctionRefOpCreate(
                                self.parent.context,
                                self.parent.location(field.range),
                                strRef(function.name),
                                result_type,
                            );
                            if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                        },
                        else => {},
                    },
                    else => {},
                }
            }
            const base_type = self.parent.typecheck.exprType(field.base);
            if (try @This().lowerNamespaceFieldExpr(self, expr_id, field, locals, result_type)) |value| {
                return value;
            }
            if (self.parent.resolution.expr_bindings[field.base.index()]) |binding| {
                switch (binding) {
                    .item => |item_id| if (try @This().lowerNamespaceFieldForItem(self, field, item_id, locals, result_type)) |value| {
                        return value;
                    },
                    else => {},
                }
            }
            if (base_type == .enum_) {
                if (@This().enumFieldExists(self, field.base, field.name)) {
                    if (mlir.oraTypeIsAAdt(result_type)) {
                        const payload_type = @This().adtVariantPayloadType(result_type, field.name) orelse
                            return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, result_type));
                        if (!mlir.oraTypeIsANone(payload_type)) {
                            return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, result_type));
                        }
                        const op = mlir.oraAdtConstructOpCreate(
                            self.parent.context,
                            self.parent.location(field.range),
                            strRef(field.name),
                            null,
                            0,
                            result_type,
                        );
                        if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                        return error.MlirOperationCreationFailed;
                    }
                    const enum_name = base_type.name() orelse
                        return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, result_type));
                    const op = mlir.oraEnumConstantOpCreate(
                        self.parent.context,
                        self.parent.location(field.range),
                        strRef(enum_name),
                        strRef(field.name),
                        result_type,
                    );
                    if (!mlir.oraOperationIsNull(op)) {
                        try @This().attachEnumConstantValueAttrs(self, op, field.base, field.name, result_type);
                        return appendValueOp(self.block, op);
                    }
                    return error.MlirOperationCreationFailed;
                }
                return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, result_type));
            }

            const base = try self.lowerExpr(field.base, locals);
            if (std.mem.eql(u8, field.name, "len")) {
                switch (base_type.kind()) {
                    .string, .bytes => {
                        const op = mlir.oraLengthOpCreate(
                            self.parent.context,
                            self.parent.location(field.range),
                            base,
                            result_type,
                        );
                        if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                    },
                    else => {},
                }
            }
            const base_value_type = mlir.oraValueGetType(base);
            if (mlir.oraAnonymousStructTypeGetFieldCount(base_value_type) != 0) {
                const struct_op = mlir.oraStructFieldExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    strRef(field.name),
                    result_type,
                );
                if (mlir.oraOperationIsNull(struct_op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, struct_op);
            }
            if (@This().overflowTupleFieldIndex(base_type, field.name)) |tuple_index| {
                const extract_type = if (std.mem.eql(u8, field.name, "overflow"))
                    mlir.oraIntegerTypeCreate(self.parent.context, 1)
                else
                    result_type;
                const op = mlir.oraTupleExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    tuple_index,
                    extract_type,
                );
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
            }
            if (@This().fallbackOverflowFieldIndex(base_type, field.name)) |tuple_index| {
                const extract_type = if (std.mem.eql(u8, field.name, "overflow"))
                    mlir.oraIntegerTypeCreate(self.parent.context, 1)
                else
                    result_type;
                const op = mlir.oraTupleExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    tuple_index,
                    extract_type,
                );
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
            }
            if (base_type.kind() == .anonymous_struct) {
                const struct_op = mlir.oraStructFieldExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    strRef(field.name),
                    result_type,
                );
                if (mlir.oraOperationIsNull(struct_op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, struct_op);
            }
            if (@This().isBitfieldLikeType(self, base_type)) {
                const extracted_type = self.parent.lowerExprType(expr_id);
                return try self.createBitfieldFieldExtract(base, base_type, field.name, extracted_type, field.range);
            }
            if (base_type == .struct_ or (base_type == .named and blk: {
                const name = base_type.name() orelse break :blk false;
                if (self.parent.typecheck.instantiatedStructByName(name) != null) break :blk true;
                const item_id = self.parent.item_index.lookup(name) orelse break :blk false;
                break :blk switch (self.parent.file.item(item_id).*) {
                    .Struct, .ErrorDecl => true,
                    else => false,
                };
            })) {
                const op = mlir.oraStructFieldExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    strRef(field.name),
                    result_type,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }
            return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.field_access", field.range, &.{base}, self.parent.lowerExprType(expr_id)));
        }

        fn isBitfieldLikeType(self: *FunctionLowerer, ty: sema.Type) bool {
            if (ty.kind() == .bitfield) return true;
            const name = ty.name() orelse return false;
            if (self.parent.typecheck.instantiatedBitfieldByName(name) != null) return true;
            const item_id = self.parent.item_index.lookup(name) orelse return false;
            return self.parent.file.item(item_id).* == .Bitfield;
        }

        fn overflowTupleFieldIndex(base_type: sema.Type, field_name: []const u8) ?i64 {
            if (base_type.kind() != .tuple) return null;
            const elements = base_type.tuple;
            if (elements.len != 2) return null;
            if (elements[1].kind() != .bool) return null;
            if (std.mem.eql(u8, field_name, "value")) return 0;
            if (std.mem.eql(u8, field_name, "overflow")) return 1;
            return null;
        }

        fn fallbackOverflowFieldIndex(base_type: sema.Type, field_name: []const u8) ?i64 {
            if (base_type.kind() == .tuple) return null;
            if (base_type.kind() == .struct_ or base_type.kind() == .named) return null;
            if (std.mem.eql(u8, field_name, "value")) return 0;
            if (std.mem.eql(u8, field_name, "overflow")) return 1;
            return null;
        }

        fn enumFieldExists(self: *FunctionLowerer, base_expr: ast.ExprId, field_name: []const u8) bool {
            const base_type = self.parent.typecheck.exprType(base_expr);
            const enum_name = base_type.name() orelse return false;
            if (self.parent.typecheck.instantiatedEnumByName(enum_name)) |instantiated| {
                return instantiated.variantIndex(field_name) != null;
            }
            const item_id = self.parent.item_index.lookup(enum_name) orelse return false;
            switch (self.parent.file.item(item_id).*) {
                .Enum => {},
                else => return false,
            }
            return self.parent.item_index.lookupEnumVariantIndex(item_id, field_name) != null;
        }

        fn attachEnumConstantValueAttrs(
            self: *FunctionLowerer,
            op: mlir.MlirOperation,
            base_expr: ast.ExprId,
            field_name: []const u8,
            result_type: mlir.MlirType,
        ) anyerror!void {
            if (mlir.oraTypeEqual(result_type, stringType(self.parent.context))) {
                const value = try @This().enumFieldStringValue(self, base_expr, field_name) orelse return;
                mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_string_value"), mlir.oraStringAttrCreate(self.parent.context, strRef(value)));
                return;
            }
            if (mlir.oraTypeEqual(result_type, bytesType(self.parent.context))) {
                const value = try @This().enumFieldBytesValue(self, base_expr, field_name) orelse return;
                mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_bytes_value"), mlir.oraStringAttrCreate(self.parent.context, strRef(value)));
                return;
            }
            if (@This().enumFieldOrdinal(self, base_expr, field_name)) |ordinal| {
                mlir.oraOperationSetAttributeByName(
                    op,
                    strRef("ora.enum_ordinal"),
                    mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.parent.context), ordinal),
                );
            }
        }

        fn enumFieldOrdinal(self: *FunctionLowerer, base_expr: ast.ExprId, field_name: []const u8) ?i64 {
            const base_type = self.parent.typecheck.exprType(base_expr);
            const enum_name = base_type.name() orelse return null;
            if (self.parent.typecheck.instantiatedEnumByName(enum_name)) |instantiated| {
                const variant_index = instantiated.variantIndex(field_name) orelse return null;
                var next_value: i64 = 0;
                for (instantiated.variants[0 .. variant_index + 1], 0..) |variant, index| {
                    const resolved_value = if (variant.explicit_value) |explicit| switch (explicit) {
                        .integer => |integer| integer.toInt(i64) catch return null,
                        else => next_value,
                    } else next_value;
                    if (index == variant_index) return resolved_value;
                    next_value = resolved_value + 1;
                }
                return null;
            }
            const item_id = self.parent.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |item| item,
                else => return null,
            };
            const variant_index = self.parent.item_index.lookupEnumVariantIndex(item_id, field_name) orelse return null;
            var next_value: i64 = 0;
            for (enum_item.variants[0 .. variant_index + 1], 0..) |variant, index| {
                const resolved_value = if (variant.value) |expr_id|
                    @This().enumIntegerValue(self, expr_id) orelse return null
                else
                    next_value;
                if (index == variant_index) return resolved_value;
                next_value = resolved_value + 1;
            }
            return null;
        }

        fn enumFieldStringValue(self: *FunctionLowerer, base_expr: ast.ExprId, field_name: []const u8) anyerror!?[]const u8 {
            const base_type = self.parent.typecheck.exprType(base_expr);
            const enum_name = base_type.name() orelse return null;
            if (self.parent.typecheck.instantiatedEnumByName(enum_name)) |instantiated| {
                const variant = instantiated.variantByName(field_name) orelse return null;
                if (variant.explicit_value) |explicit| switch (explicit) {
                    .string => |literal| return literal,
                    else => return null,
                };
                return try std.fmt.allocPrint(self.parent.allocator, "{s}.{s}", .{ enum_name, field_name });
            }
            const item_id = self.parent.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |item| item,
                else => return null,
            };
            const variant_index = self.parent.item_index.lookupEnumVariantIndex(item_id, field_name) orelse return null;
            const variant = enum_item.variants[variant_index];
            if (variant.value) |expr_id| return @This().enumStringValue(self, expr_id);
            return try std.fmt.allocPrint(self.parent.allocator, "{s}.{s}", .{ enum_name, field_name });
        }

        fn enumFieldBytesValue(self: *FunctionLowerer, base_expr: ast.ExprId, field_name: []const u8) anyerror!?[]const u8 {
            const base_type = self.parent.typecheck.exprType(base_expr);
            const enum_name = base_type.name() orelse return null;
            if (self.parent.typecheck.instantiatedEnumByName(enum_name)) |instantiated| {
                const variant = instantiated.variantByName(field_name) orelse return null;
                if (variant.explicit_value) |explicit| switch (explicit) {
                    .bytes => |literal| return literal,
                    else => return null,
                };
                return null;
            }
            const item_id = self.parent.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |item| item,
                else => return null,
            };
            const variant_index = self.parent.item_index.lookupEnumVariantIndex(item_id, field_name) orelse return null;
            if (enum_item.variants[variant_index].value) |expr_id| return @This().enumBytesValue(self, expr_id);
            return null;
        }

        fn enumIntegerValue(self: *FunctionLowerer, expr_id: ast.ExprId) ?i64 {
            const value = self.parent.const_eval.values[expr_id.index()] orelse return null;
            return switch (value) {
                .integer => |integer| integer.toInt(i64) catch null,
                .boolean => |boolean| if (boolean) 1 else 0,
                else => null,
            };
        }

        fn enumStringValue(self: *FunctionLowerer, expr_id: ast.ExprId) ?[]const u8 {
            const value = self.parent.const_eval.values[expr_id.index()] orelse return null;
            return switch (value) {
                .string => |string| string,
                else => null,
            };
        }

        fn enumBytesValue(self: *FunctionLowerer, expr_id: ast.ExprId) ?[]const u8 {
            return switch (self.parent.file.expression(expr_id).*) {
                .BytesLiteral => |literal| literal.text,
                .Group => |group| @This().enumBytesValue(self, group.expr),
                else => null,
            };
        }

        fn lowerBuiltinFieldExpr(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            field: ast.FieldExpr,
        ) anyerror!?mlir.MlirValue {
            const path = try @This().fieldExprPath(self, expr_id);
            defer self.parent.allocator.free(path);
            if (try @This().lowerBuiltinFieldCall(self, field, path)) |value| {
                return value;
            }
            return null;
        }

        fn lowerBuiltinFieldCall(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            path: []const u8,
        ) anyerror!?mlir.MlirValue {
            const opcode_name = if (std.mem.eql(u8, path, "std.msg.sender") or
                std.mem.eql(u8, path, "std.transaction.sender"))
                "ora.evm.caller"
            else if (std.mem.eql(u8, path, "std.tx.origin"))
                "ora.evm.origin"
            else if (std.mem.eql(u8, path, "std.msg.value"))
                "ora.evm.callvalue"
            else if (std.mem.eql(u8, path, "std.transaction.gasprice"))
                "ora.evm.gasprice"
            else if (std.mem.eql(u8, path, "std.block.timestamp"))
                "ora.evm.timestamp"
            else if (std.mem.eql(u8, path, "std.block.number"))
                "ora.evm.number"
            else if (std.mem.eql(u8, path, "std.block.coinbase"))
                "ora.evm.coinbase"
            else
                return null;

            const builtin_result_type =
                if (std.mem.eql(u8, opcode_name, "ora.evm.caller") or
                std.mem.eql(u8, opcode_name, "ora.evm.origin") or
                std.mem.eql(u8, opcode_name, "ora.evm.coinbase"))
                    addressType(self.parent.context)
                else
                    defaultIntegerType(self.parent.context);

            const op = mlir.oraEvmOpCreate(
                self.parent.context,
                self.parent.location(field.range),
                strRef(opcode_name),
                null,
                0,
                builtin_result_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn fieldExprPath(self: *FunctionLowerer, expr_id: ast.ExprId) anyerror![]const u8 {
            const expr = self.parent.file.expression(expr_id).*;
            return switch (expr) {
                .Name => |name| self.parent.allocator.dupe(u8, name.name),
                .Group => |group| @This().fieldExprPath(self, group.expr),
                .Field => |field| blk: {
                    const base = try @This().fieldExprPath(self, field.base);
                    break :blk std.fmt.allocPrint(self.parent.allocator, "{s}.{s}", .{ base, field.name });
                },
                else => self.parent.allocator.dupe(u8, ""),
            };
        }

        fn importedModuleForExpr(self: *FunctionLowerer, expr_id: ast.ExprId) !?source.ModuleId {
            const query = self.parent.module_query orelse return null;
            return switch (self.parent.file.expression(expr_id).*) {
                .Name => |name| try query.resolveImportAlias(self.parent.module_id, name.name),
                .Field => |field| blk: {
                    const base_module_id = (try @This().importedModuleForExpr(self, field.base)) orelse break :blk null;
                    break :blk try query.resolveImportAlias(base_module_id, field.name);
                },
                .Group => |group| try @This().importedModuleForExpr(self, group.expr),
                else => null,
            };
        }

        fn lowerImportedFieldExpr(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            result_type: mlir.MlirType,
        ) anyerror!?mlir.MlirValue {
            const query = self.parent.module_query orelse return null;
            const target_module_id = (try @This().importedModuleForExpr(self, field.base)) orelse return null;
            const target_item_id = (try query.lookupItem(target_module_id, field.name)) orelse return null;
            const target_file = try query.astFile(target_module_id);
            const target_typecheck = try query.moduleTypeCheck(target_module_id);
            const target_const_eval = try query.constEval(target_module_id);

            const value_expr = switch (target_file.item(target_item_id).*) {
                .Constant => |constant| constant.value,
                .Field => |decl| decl.value orelse return null,
                else => return null,
            };
            const value = target_const_eval.values[value_expr.index()] orelse return null;
            const sema_type = target_typecheck.exprType(value_expr);
            const loc = self.parent.location(field.range);
            return try @This().lowerPersistedConstValue(self, value, sema_type, result_type, loc);
        }

        fn lowerNamespaceFieldExpr(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            field: ast.FieldExpr,
            locals: *LocalEnv,
            result_type: mlir.MlirType,
        ) anyerror!?mlir.MlirValue {
            _ = expr_id;
            const base_type = self.parent.typecheck.exprType(field.base);
            const base_name = base_type.name() orelse return null;
            const item_id = self.parent.item_index.lookup(base_name) orelse return null;
            return try @This().lowerNamespaceFieldForItem(self, field, item_id, locals, result_type);
        }

        fn lowerNamespaceFieldForItem(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            item_id: ast.ItemId,
            locals: *LocalEnv,
            result_type: mlir.MlirType,
        ) anyerror!?mlir.MlirValue {
            switch (self.parent.file.item(item_id).*) {
                .Contract => {
                    const member_id = self.parent.item_index.lookupContractMemberWithRoles(self.parent.file, item_id, field.name, .{
                        .field = true,
                        .constant = true,
                        .function = true,
                        .struct_ = true,
                        .enum_ = true,
                        .bitfield = true,
                    }) orelse return null;
                    switch (self.parent.file.item(member_id).*) {
                        .Field => |contract_field| {
                            if (contract_field.binding_kind == .immutable) {
                                if (contract_field.value) |value| {
                                    return try self.lowerExpr(value, locals);
                                }
                            }
                            const op = switch (contract_field.storage_class) {
                                .storage => mlir.oraSLoadOpCreate(
                                    self.parent.context,
                                    self.parent.location(contract_field.range),
                                    strRef(contract_field.name),
                                    result_type,
                                ),
                                .memory => mlir.oraMLoadOpCreate(
                                    self.parent.context,
                                    self.parent.location(contract_field.range),
                                    strRef(contract_field.name),
                                    result_type,
                                ),
                                .tstore => mlir.oraTLoadOpCreate(
                                    self.parent.context,
                                    self.parent.location(contract_field.range),
                                    strRef(contract_field.name),
                                    result_type,
                                ),
                                .none => std.mem.zeroes(mlir.MlirOperation),
                            };
                            if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                        },
                        .Constant => |constant| return try self.lowerExpr(constant.value, locals),
                        .Function => |function| {
                            const op = mlir.oraFunctionRefOpCreate(
                                self.parent.context,
                                self.parent.location(field.range),
                                strRef(function.name),
                                result_type,
                            );
                            if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                        },
                        .Struct, .Enum, .Bitfield => return null,
                        else => {},
                    }
                },
                else => {},
            }
            return null;
        }

        fn lowerArrayLiteral(self: *FunctionLowerer, expr_id: ast.ExprId, array: ast.ArrayLiteralExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const result_type = self.parent.lowerExprType(expr_id);
            const alloc = mlir.oraMemrefAllocaOpCreate(
                self.parent.context,
                self.parent.location(array.range),
                result_type,
            );
            if (mlir.oraOperationIsNull(alloc)) {
                var operands: std.ArrayList(mlir.MlirValue) = .{};
                for (array.elements) |element| try operands.append(self.parent.allocator, try self.lowerExpr(element, locals));
                const placeholder = try self.createAggregatePlaceholder("ora.array.create", array.range, operands.items, result_type);
                return appendValueOp(self.block, placeholder);
            }

            const memref = appendValueOp(self.block, alloc);
            const element_type = mlir.oraShapedTypeGetElementType(result_type);
            for (array.elements, 0..) |element, index| {
                const raw_element_value = try self.lowerExpr(element, locals);
                const element_value = try self.convertValueForFlow(raw_element_value, element_type, array.range);
                const raw_index = appendValueOp(
                    self.block,
                    createIntegerConstant(
                        self.parent.context,
                        self.parent.location(array.range),
                        defaultIntegerType(self.parent.context),
                        @intCast(index),
                    ),
                );
                const index_value = try @This().convertIndexToIndexType(self, raw_index, array.range);
                const store = mlir.oraMemrefStoreOpCreate(
                    self.parent.context,
                    self.parent.location(array.range),
                    element_value,
                    memref,
                    &[_]mlir.MlirValue{index_value},
                    1,
                );
                if (mlir.oraOperationIsNull(store)) {
                    var operands: std.ArrayList(mlir.MlirValue) = .{};
                    for (array.elements) |fallback_element| try operands.append(self.parent.allocator, try self.lowerExpr(fallback_element, locals));
                    const placeholder = try self.createAggregatePlaceholder("ora.array.create", array.range, operands.items, result_type);
                    return appendValueOp(self.block, placeholder);
                }
                appendOp(self.block, store);
            }
            return memref;
        }

        fn constTupleIndex(self: *FunctionLowerer, expr_id: ast.ExprId) ?usize {
            if (self.parent.const_eval.values[expr_id.index()]) |value| {
                return switch (value) {
                    .integer => |integer| const_bridge.positiveShiftAmount(integer),
                    else => null,
                };
            }
            return switch (self.parent.file.expression(expr_id).*) {
                .IntegerLiteral => |literal| parseTupleIndexLiteral(literal.text),
                .Group => |group| @This().constTupleIndex(self, group.expr),
                else => null,
            };
        }

        fn parseTupleIndexLiteral(text: []const u8) ?usize {
            const trimmed = std.mem.trim(u8, text, " \t\n\r");
            if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X")) {
                return std.fmt.parseInt(usize, trimmed[2..], 16) catch null;
            }
            if (std.mem.startsWith(u8, trimmed, "0b") or std.mem.startsWith(u8, trimmed, "0B")) {
                return std.fmt.parseInt(usize, trimmed[2..], 2) catch null;
            }
            return std.fmt.parseInt(usize, trimmed, 10) catch null;
        }

        pub fn createValuePlaceholder(self: *FunctionLowerer, op_name: []const u8, text: []const u8, range: source.TextRange, result_type: mlir.MlirType) anyerror!mlir.MlirOperation {
            self.parent.recordPlaceholder();
            var attrs: [2]mlir.MlirNamedAttribute = .{
                namedStringAttr(self.parent.context, "value", text),
                namedStringAttr(self.parent.context, "ora.executable_fallback", op_name),
            };
            return mlir.oraOperationCreate(
                self.parent.context,
                self.parent.location(range),
                strRef(op_name),
                null,
                0,
                &[_]mlir.MlirType{result_type},
                1,
                &attrs,
                attrs.len,
                0,
                false,
            );
        }

        pub fn createAggregatePlaceholder(self: *FunctionLowerer, op_name: []const u8, range: source.TextRange, operands: []const mlir.MlirValue, result_type: mlir.MlirType) anyerror!mlir.MlirOperation {
            self.parent.recordPlaceholder();
            var attrs: [2]mlir.MlirNamedAttribute = .{
                namedTypeAttr(self.parent.context, "ora.type", result_type),
                namedStringAttr(self.parent.context, "ora.executable_fallback", op_name),
            };
            return mlir.oraOperationCreate(
                self.parent.context,
                self.parent.location(range),
                strRef(op_name),
                if (operands.len == 0) null else operands.ptr,
                operands.len,
                &[_]mlir.MlirType{result_type},
                1,
                &attrs,
                attrs.len,
                0,
                false,
            );
        }

        fn expectOperation(self: *FunctionLowerer, op: mlir.MlirOperation, range: source.TextRange, op_name: []const u8) anyerror!mlir.MlirOperation {
            if (!mlir.oraOperationIsNull(op)) return op;
            try self.parent.emitLoweringError(
                range,
                "failed to create MLIR operation '{s}'",
                .{op_name},
            );
            return error.MlirOperationCreationFailed;
        }

        fn expectValueOp(self: *FunctionLowerer, op: mlir.MlirOperation, range: source.TextRange, op_name: []const u8) anyerror!mlir.MlirValue {
            return appendValueOp(self.block, try @This().expectOperation(self, op, range, op_name));
        }

        fn loweringValueError(self: *FunctionLowerer, range: source.TextRange, result_type: mlir.MlirType, comptime fmt: []const u8, args: anytype) anyerror!mlir.MlirValue {
            try self.parent.emitLoweringError(range, fmt, args);
            const op = try self.createAggregatePlaceholder("ora.lowering_error", range, &.{}, result_type);
            return appendValueOp(self.block, op);
        }

        fn voidValue(self: *FunctionLowerer, range: source.TextRange) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const seed = appendValueOp(
                self.block,
                createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0),
            );
            const none_cast = mlir.oraUnrealizedConversionCastOpCreate(
                self.parent.context,
                loc,
                seed,
                mlir.oraNoneTypeCreate(self.parent.context),
            );
            if (mlir.oraOperationIsNull(none_cast)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, none_cast);
        }

        pub fn typeIsVoid(self: *const FunctionLowerer, ty: mlir.MlirType) bool {
            const none = mlir.oraNoneTypeCreate(self.parent.context);
            return mlir.oraTypeEqual(ty, none);
        }

        fn createZeroAddressValue(self: *FunctionLowerer, range: source.TextRange) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
            const zero_i160 = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, i160_type, 0));
            const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, zero_i160);
            if (mlir.oraOperationIsNull(addr_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, addr_op);
        }

        fn convertIndexToIndexType(self: *FunctionLowerer, index: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const index_type = mlir.oraIndexTypeCreate(self.parent.context);
            if (mlir.oraTypeEqual(mlir.oraValueGetType(index), index_type)) {
                return index;
            }

            const op = mlir.oraArithIndexCastUIOpCreate(
                self.parent.context,
                self.parent.location(range),
                index,
                index_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }
    };
}
