const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const abi_support = @import("abi.zig");
const const_bridge = @import("../comptime/compiler_const_bridge.zig");
const source = @import("../source/mod.zig");
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
const parseIntLiteral = support.parseIntLiteral;
const strRef = support.strRef;
const stringType = support.stringType;
const LocalEnv = hir_locals.LocalEnv;

fn unwrapRefinementSemaType(ty: sema.Type) sema.Type {
    return if (ty.refinementBaseType()) |base| base.* else ty;
}

pub fn mixin(FunctionLowerer: type, Lowerer: type) type {
    _ = Lowerer;
    return struct {
        fn isSignedIntegerExpr(self: *FunctionLowerer, expr_id: ast.ExprId) bool {
            const ty = unwrapRefinementSemaType(self.parent.typecheck.expr_types[expr_id.index()]);
            return switch (ty) {
                .integer => |integer| integer.signed orelse false,
                else => false,
            };
        }

        fn signednessForBinaryIntegerOp(self: *FunctionLowerer, lhs_expr: ast.ExprId, rhs_expr: ast.ExprId) bool {
            const lhs_node = self.parent.file.expression(lhs_expr).*;
            const rhs_node = self.parent.file.expression(rhs_expr).*;
            if (lhs_node == .IntegerLiteral and rhs_node != .IntegerLiteral) {
                return @This().isSignedIntegerExpr(self, rhs_expr);
            }
            if (rhs_node == .IntegerLiteral and lhs_node != .IntegerLiteral) {
                return @This().isSignedIntegerExpr(self, lhs_expr);
            }
            return @This().isSignedIntegerExpr(self, lhs_expr);
        }

        fn predicateForBinaryCompare(self: *FunctionLowerer, op: ast.BinaryOp, lhs_expr: ast.ExprId, rhs_expr: ast.ExprId) []const u8 {
            const is_signed = @This().signednessForBinaryIntegerOp(self, lhs_expr, rhs_expr);
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
            if (@This().exprDependsOnRuntimeState(self, expr_id)) return null;
            if (@This().exprDependsOnMutablePatternLocals(self, expr_id)) return null;
            const expr = self.parent.file.expression(expr_id).*;
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
            return try @This().lowerPersistedConstValue(self, value, self.parent.typecheck.exprType(expr_id), result_type, loc);
        }

        fn lowerPersistedConstValue(
            self: *FunctionLowerer,
            value: sema.ConstValue,
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
                    if (mlir.oraOperationIsNull(addr_op)) break :blk null;
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
                        const element_type = lowerTypeDescriptor(self.parent.context, element_sema_type);
                        values[index] = (try @This().lowerPersistedConstValue(self, element, element_sema_type, element_type, loc)) orelse break :blk null;
                    }
                    const op = mlir.oraTupleCreateOpCreate(self.parent.context, loc, values.ptr, values.len, tuple_result_type);
                    if (mlir.oraOperationIsNull(op)) break :blk null;
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
                    break :blk appendValueOp(self.block, createIntegerConstant(self.parent.context, self.parent.location(literal.range), ty, parseIntLiteral(literal.text) orelse 0));
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
                .Result => self.current_return_value orelse try self.defaultValue(self.return_type orelse defaultIntegerType(self.parent.context), expr.Result.range),
                .Group => |group| try self.lowerExpr(group.expr, locals),
                .Old => |old| blk: {
                    const value = try self.lowerExpr(old.expr, locals);
                    const result_type = self.parent.lowerExprType(old.expr);
                    const op = mlir.oraOldOpCreate(self.parent.context, loc, value, result_type);
                    if (mlir.oraOperationIsNull(op)) break :blk value;
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
                        const raw_value = try self.lowerExpr(element, locals);
                        const value = if (sema_tuple_type.kind() == .tuple and index < sema_tuple_type.tuple.len)
                            try self.convertValueForFlow(raw_value, self.parent.lowerSemaType(sema_tuple_type.tuple[index], tuple.range), exprRange(self.parent.file, element))
                        else
                            raw_value;
                        try operands.append(self.parent.allocator, value);
                    }
                    const result_type = self.parent.lowerSemaType(sema_tuple_type, tuple.range);
                    const file_path = self.parent.sources.file(self.parent.file.file_id).path;
                    const line_col = self.parent.sources.lineColumn(.{ .file_id = self.parent.file.file_id, .range = tuple.range });
                    if (std.mem.endsWith(u8, file_path, "ora-example/tuples/tuple_basics.ora") and line_col.line == 13) {
                        std.debug.print(
                            "[tuple-debug] expr_id={d} sema_kind={s} lowered_sema={any} lowered_expr={any} line={d}:{d}\n",
                            .{
                                expr_id.index(),
                                @tagName(sema_tuple_type.kind()),
                                result_type,
                                self.parent.lowerExprType(expr_id),
                                line_col.line,
                                line_col.column,
                            },
                        );
                    }
                    const op = mlir.oraTupleCreateOpCreate(
                        self.parent.context,
                        self.parent.location(tuple.range),
                        if (operands.items.len == 0) null else operands.items.ptr,
                        operands.items.len,
                        result_type,
                    );
                    if (mlir.oraOperationIsNull(op)) {
                        const placeholder = try self.createAggregatePlaceholder("ora.tuple.create", tuple.range, operands.items, result_type);
                        break :blk appendValueOp(self.block, placeholder);
                    }
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
                .Comptime => |comptime_expr| blk: {
                    var child_locals = try self.cloneLocals(locals);
                    const body = self.parent.file.body(comptime_expr.body).*;
                    if (body.statements.len == 0) break :blk try self.defaultValue(self.parent.lowerExprType(expr_id), comptime_expr.range);

                    var index: usize = 0;
                    while (index + 1 < body.statements.len) : (index += 1) {
                        _ = try self.lowerStmt(body.statements[index], &child_locals);
                    }

                    const last_stmt = self.parent.file.statement(body.statements[body.statements.len - 1]).*;
                    break :blk switch (last_stmt) {
                        .Expr => |expr_stmt| try self.lowerExpr(expr_stmt.expr, &child_locals),
                        .Return => |ret| if (ret.value) |value|
                            try self.lowerExpr(value, &child_locals)
                        else
                            try self.defaultValue(self.parent.lowerExprType(expr_id), comptime_expr.range),
                        else => blk2: {
                            _ = try self.lowerStmt(body.statements[body.statements.len - 1], &child_locals);
                            break :blk2 try self.defaultValue(self.parent.lowerExprType(expr_id), comptime_expr.range);
                        },
                    };
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
                    const key = try self.lowerExpr(index.index, locals);
                    const base_type = self.parent.typecheck.exprType(index.base);
                    if (mlir.oraTypeIsAMemRef(mlir.oraValueGetType(base))) {
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
                        const map_key_type = mlir.oraMapTypeGetKeyType(mlir.oraValueGetType(base));
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
                .Quantified => |quantified| blk: {
                    if (quantified.condition) |condition| _ = try self.lowerExpr(condition, locals);
                    _ = try self.lowerExpr(quantified.body, locals);
                    break :blk appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                },
                .Error => try self.defaultValue(defaultIntegerType(self.parent.context), expr.Error.range),
            };
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

            return self.defaultValue(self.parent.lowerExprType(expr_id), name.range);
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
            const contract = switch (self.parent.file.item(contract_id).*) {
                .Contract => |contract| contract,
                else => return null,
            };

            for (contract.members) |member_id| {
                switch (self.parent.file.item(member_id).*) {
                    .Field => |field| {
                        if (!std.mem.eql(u8, field.name, name.name)) continue;
                        return try @This().lowerBoundItemExpr(self, expr_id, name, member_id, locals);
                    },
                    .Constant => |constant| {
                        if (!std.mem.eql(u8, constant.name, name.name)) continue;
                        return try @This().lowerBoundItemExpr(self, expr_id, name, member_id, locals);
                    },
                    .Function => |function_item| {
                        if (!std.mem.eql(u8, function_item.name, name.name)) continue;
                        return try @This().lowerBoundItemExpr(self, expr_id, name, member_id, locals);
                    },
                    else => {},
                }
            }
            return null;
        }

        pub fn lowerUnary(self: *FunctionLowerer, unary: ast.UnaryExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const operand = try self.lowerExpr(unary.operand, locals);
            const loc = self.parent.location(unary.range);
            return switch (unary.op) {
                .neg => blk: {
                    const zero = try self.defaultValue(mlir.oraValueGetType(operand), unary.range);
                    const op = mlir.oraArithSubIOpCreate(self.parent.context, loc, zero, operand);
                    if (mlir.oraOperationIsNull(op)) break :blk operand;
                    break :blk appendValueOp(self.block, op);
                },
                .not_ => blk: {
                    const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                    const op = mlir.oraArithXorIOpCreate(self.parent.context, loc, operand, one);
                    if (mlir.oraOperationIsNull(op)) break :blk operand;
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
                    if (mlir.oraOperationIsNull(op)) break :blk operand;
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
                if (mlir.oraOperationIsNull(unwrap)) return operand;
                return appendValueOp(self.block, unwrap);
            }

            if (!self.in_try_block) {
                if (self.return_type) |return_type| {
                    if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(return_type))) {
                        const is_error = mlir.oraErrorIsErrorOpCreate(self.parent.context, loc, operand);
                        if (mlir.oraOperationIsNull(is_error)) return operand;
                        const is_error_value = appendValueOp(self.block, is_error);

                        const branch = mlir.oraConditionalReturnOpCreate(self.parent.context, loc, is_error_value);
                        if (mlir.oraOperationIsNull(branch)) return operand;
                        appendOp(self.block, branch);

                        const then_block = mlir.oraConditionalReturnOpGetThenBlock(branch);
                        const else_block = mlir.oraConditionalReturnOpGetElseBlock(branch);
                        if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                            return error.MlirOperationCreationFailed;
                        }

                        const error_type = defaultIntegerType(self.parent.context);
                        const get_error = mlir.oraErrorGetErrorOpCreate(self.parent.context, loc, operand, error_type);
                        if (mlir.oraOperationIsNull(get_error)) return operand;
                        appendOp(then_block, get_error);
                        const error_value = mlir.oraOperationGetResult(get_error, 0);

                        const error_union = mlir.oraErrorErrOpCreate(self.parent.context, loc, error_value, return_type);
                        if (mlir.oraOperationIsNull(error_union)) return operand;
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
                        if (mlir.oraOperationIsNull(unwrap)) return operand;
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
            var rhs = try self.lowerExpr(binary.rhs, locals);
            const loc = self.parent.location(binary.range);
            const result_type = self.parent.lowerExprType(expr_id);
            const is_signed_int_op = @This().signednessForBinaryIntegerOp(self, binary.lhs, binary.rhs);

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
                return self.lowerCheckedPower(lhs, rhs, result_type, binary.range);
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
                .lt, .le, .gt, .ge => self.createCompareOp(loc, @This().predicateForBinaryCompare(self, binary.op, binary.lhs, binary.rhs), lhs, rhs),
                .pow => unreachable,
            };

            if (mlir.oraOperationIsNull(op)) {
                return self.defaultValue(result_type, binary.range);
            }
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
            if (mlir.oraTypeIsAddressType(lhs_type) or mlir.oraTypeIsAddressType(rhs_type)) {
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
                    if (try @This().lowerBuiltinFieldCall(self, callee_expr.Field, self.parent.lowerExprType(expr_id), path)) |value| {
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
                } else
                    try self.parent.runtimeFunctionParameters(function)
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

            const callee_name = if (imported_target != null and callee_function != null)
                (try self.parent.ensureImportedFunctionSymbol(
                    imported_target.?.module_id,
                    imported_target.?.item_id,
                    callee_function.?,
                    call,
                    imported_resolution,
                    @This().currentContractParentBlock(self) orelse self.parent.module_body,
                )) orelse return self.defaultValue(self.parent.lowerExprType(expr_id), call.range)
            else if (callee_function != null and callee_item_id != null)
                (try self.parent.ensureMonomorphizedFunction(
                    callee_item_id.?,
                    callee_function.?,
                    call,
                    runtime_parameters,
                    @This().currentContractItemId(self),
                )) orelse return self.defaultValue(self.parent.lowerExprType(expr_id), call.range)
            else
                (@This().calleeName(self, call.callee) orelse return self.defaultValue(self.parent.lowerExprType(expr_id), call.range));

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
            if (mlir.oraOperationIsNull(op)) return try self.defaultValue(result_type, call.range);
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try self.defaultValue(defaultIntegerType(self.parent.context), call.range);
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
        };

        fn lowerExternProxyMethodCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!?mlir.MlirValue {
            const resolved = @This().resolveExternProxyMethodCall(self, call.callee) orelse return null;
            switch (resolved.method.return_type.kind()) {
                .bool, .address, .integer, .string, .bytes, .tuple, .struct_, .contract, .slice => {},
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

            var arg_type_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer arg_type_attrs.deinit(self.parent.allocator);
            for (resolved.method.param_types) |param_type| {
                const abi_type = try abi_support.canonicalAbiType(self.parent.allocator, param_type);
                defer self.parent.allocator.free(abi_type);
                try arg_type_attrs.append(self.parent.allocator, mlir.oraStringAttrCreate(self.parent.context, strRef(abi_type)));
            }
            const arg_types_attr = mlir.oraArrayAttrCreate(self.parent.context, @intCast(arg_type_attrs.items.len), if (arg_type_attrs.items.len == 0) null else arg_type_attrs.items.ptr);

            var return_type_attrs: [1]mlir.MlirAttribute = .{undefined};
            const abi_return = try abi_support.externReturnAbiType(self.parent.allocator, resolved.method.return_type);
            defer self.parent.allocator.free(abi_return);
            return_type_attrs[0] = mlir.oraStringAttrCreate(self.parent.context, strRef(abi_return));
            const return_types_attr = mlir.oraArrayAttrCreate(self.parent.context, 1, &return_type_attrs);

            const signature = try abi_support.signatureForMethod(
                self.parent.allocator,
                resolved.method.name,
                resolved.method.receiver_kind != .none,
                resolved.method.param_types,
            );
            defer self.parent.allocator.free(signature);
            const selector_text = try abi_support.keccakSelectorHex(self.parent.allocator, signature);
            defer self.parent.allocator.free(selector_text);
            const selector_value = try std.fmt.parseUnsigned(u32, selector_text[2..], 16);
            const selector_type = mlir.oraIntegerTypeCreate(self.parent.context, 32);
            const selector_attr = mlir.oraIntegerAttrCreateI64FromType(selector_type, selector_value);

            const encode_op = mlir.oraAbiEncodeOpCreate(
                self.parent.context,
                loc,
                selector_attr,
                arg_types_attr,
                if (encode_args.items.len == 0) null else encode_args.items.ptr,
                encode_args.items.len,
                defaultIntegerType(self.parent.context),
            );
            if (mlir.oraOperationIsNull(encode_op)) return error.MlirOperationCreationFailed;
            const calldata = appendValueOp(self.block, encode_op);

            const target = try self.lowerExpr(proxy.address_expr, locals);
            const gas = try self.lowerExpr(proxy.gas_expr, locals);

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
            appendOp(self.block, external_call_op);
            const success = mlir.oraOperationGetResult(external_call_op, 0);
            const returndata = mlir.oraOperationGetResult(external_call_op, 1);

            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, success, &[_]mlir.MlirType{result_type}, 1, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            const decode_op = mlir.oraAbiDecodeOpCreate(self.parent.context, loc, return_types_attr, returndata, payload_type);
            if (mlir.oraOperationIsNull(decode_op)) return error.MlirOperationCreationFailed;
            appendOp(then_block, decode_op);
            const decoded = mlir.oraOperationGetResult(decode_op, 0);
            const ok_op = mlir.oraErrorOkOpCreate(self.parent.context, loc, decoded, result_type);
            if (mlir.oraOperationIsNull(ok_op)) return error.MlirOperationCreationFailed;
            if (self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.exprType(expr_id))) {
                mlir.oraOperationSetAttributeByName(ok_op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
            }
            appendOp(then_block, ok_op);
            try support.appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(ok_op, 0)});

            const error_result = try @This().lowerExternErrorResult(self, expr_id, resolved.method, else_block, loc, returndata);
            try support.appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{error_result});

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
            var selector_attrs: [1]mlir.MlirAttribute = .{
                mlir.oraStringAttrCreate(self.parent.context, strRef("u256")),
            };
            const selector_types_attr = mlir.oraArrayAttrCreate(self.parent.context, 1, &selector_attrs);
            const decode_op = mlir.oraAbiDecodeOpCreate(
                self.parent.context,
                loc,
                selector_types_attr,
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
            var return_type_attrs: [1]mlir.MlirAttribute = .{
                mlir.oraStringAttrCreate(self.parent.context, strRef("tuple")),
            };
            const return_types_attr = mlir.oraArrayAttrCreate(self.parent.context, 1, &return_type_attrs);
            const payload_offset = appendValueOp(block, createIntegerConstant(self.parent.context, loc, defaultIntegerType(self.parent.context), 4));
            const payload_ptr_op = mlir.oraArithAddIOpCreate(self.parent.context, loc, returndata, payload_offset);
            if (mlir.oraOperationIsNull(payload_ptr_op)) return error.MlirOperationCreationFailed;
            appendOp(block, payload_ptr_op);
            const payload_returndata = mlir.oraOperationGetResult(payload_ptr_op, 0);
            const decode_op = mlir.oraAbiDecodeOpCreate(self.parent.context, loc, return_types_attr, payload_returndata, lowered_error_type);
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
            for (trait_interface.methods) |method| {
                if (std.mem.eql(u8, method.name, field.name)) return .{
                    .field = field,
                    .trait_name = base_type.external_proxy.trait_name,
                    .method = method,
                };
            }
            return null;
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
                resolved.target_name,
                call,
                @This().currentContractParentBlock(self),
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
            if (mlir.oraOperationIsNull(op)) return try self.defaultValue(result_type, call.range);
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try self.defaultValue(defaultIntegerType(self.parent.context), call.range);
            }
            return appendValueOp(self.block, op);
        }

        const ResolvedTraitBoundMethodCall = struct {
            impl_item_id: ast.ItemId,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
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
                for (trait_interface.methods) |method| {
                    if (!std.mem.eql(u8, method.name, field.name) or method.receiver_kind != .value_self) continue;
                    if (matched_trait != null) return null;
                    matched_trait = bound.trait_name;
                }
            }
            const trait_name = matched_trait orelse return null;
            const impl_item_id = self.parent.item_index.lookupImpl(trait_name, target_name) orelse return null;
            const impl_item = self.parent.file.item(impl_item_id).Impl;
            for (impl_item.methods) |method_item_id| {
                const item = self.parent.file.item(method_item_id).*;
                if (item != .Function) continue;
                const has_self = @This().functionHasRuntimeSelf(self, item.Function);
                if (!std.mem.eql(u8, item.Function.name, field.name) or !has_self) continue;
                return .{
                    .impl_item_id = impl_item_id,
                    .method_item_id = method_item_id,
                    .function = item.Function,
                    .target_name = target_name,
                    .receiver_expr = field.base,
                };
            }
            return null;
        }

        fn resolveConcreteValueMethodCall(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            target_name: []const u8,
        ) ?ResolvedTraitBoundMethodCall {
            var matched_impl_item_id: ?ast.ItemId = null;
            var matched_method_item_id: ?ast.ItemId = null;
            var matched_function: ?ast.FunctionItem = null;

            for (self.parent.file.items, 0..) |item, item_index| {
                if (item != .Impl) continue;
                const impl_item = item.Impl;
                if (!std.mem.eql(u8, impl_item.target_name, target_name)) continue;
                for (impl_item.methods) |method_item_id| {
                    const method_item = self.parent.file.item(method_item_id).*;
                    if (method_item != .Function) continue;
                    if (!std.mem.eql(u8, method_item.Function.name, field.name) or
                        !@This().functionHasRuntimeSelf(self, method_item.Function)) continue;
                    if (matched_impl_item_id != null) return null;
                    matched_impl_item_id = ast.ItemId.fromIndex(item_index);
                    matched_method_item_id = method_item_id;
                    matched_function = method_item.Function;
                }
            }

            return .{
                .impl_item_id = matched_impl_item_id orelse return null,
                .method_item_id = matched_method_item_id orelse return null,
                .function = matched_function orelse return null,
                .target_name = target_name,
                .receiver_expr = field.base,
            };
        }

        const ResolvedAssociatedImplMethodCall = struct {
            impl_item_id: ast.ItemId,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
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
                resolved.target_name,
                call,
                @This().currentContractParentBlock(self),
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
            if (mlir.oraOperationIsNull(op)) return try self.defaultValue(result_type, call.range);
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return try self.defaultValue(defaultIntegerType(self.parent.context), call.range);
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

            for (self.parent.typecheck.impl_interfaces) |impl_interface| {
                if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
                const impl_item = self.parent.file.item(impl_interface.impl_item_id).Impl;
                for (impl_interface.methods, 0..) |method, index| {
                    if (!std.mem.eql(u8, method.name, field.name) or method.receiver_kind != .none) continue;
                    if (matched_impl_item_id != null) return null;
                    matched_impl_item_id = impl_interface.impl_item_id;
                    matched_method_item_id = impl_item.methods[index];
                    matched_function = self.parent.file.item(impl_item.methods[index]).Function;
                }
            }

            return .{
                .impl_item_id = matched_impl_item_id orelse return null,
                .method_item_id = matched_method_item_id orelse return null,
                .function = matched_function orelse return null,
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
                for (trait_interface.methods) |method| {
                    if (!std.mem.eql(u8, method.name, field.name) or method.receiver_kind != .none) continue;
                    if (matched_trait != null) return null;
                    matched_trait = bound.trait_name;
                }
            }
            const trait_name = matched_trait orelse return null;
            const concrete_type = self.parent.substitutedType(type_param_name) orelse return null;
            const target_name = concrete_type.name() orelse return null;
            return @This().resolveAssociatedMethodFromLookup(self, trait_name, target_name, field.name);
        }

        fn resolveAssociatedMethodFromLookup(self: *FunctionLowerer, trait_name: []const u8, target_name: []const u8, method_name: []const u8) ?ResolvedAssociatedImplMethodCall {
            const impl_item_id = self.parent.item_index.lookupImpl(trait_name, target_name) orelse return null;
            const impl_item = self.parent.file.item(impl_item_id).Impl;
            for (impl_item.methods) |method_item_id| {
                const item = self.parent.file.item(method_item_id).*;
                if (item != .Function) continue;
                const has_self = @This().functionHasRuntimeSelf(self, item.Function);
                if (!std.mem.eql(u8, item.Function.name, method_name) or has_self) continue;
                return .{
                    .impl_item_id = impl_item_id,
                    .method_item_id = method_item_id,
                    .function = item.Function,
                    .target_name = target_name,
                };
            }
            return null;
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
            for (function.parameters) |parameter| {
                if (parameter.is_comptime) continue;
                return std.mem.eql(u8, self.parent.patternName(parameter.pattern) orelse "", "self");
            }
            return false;
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
            const target_file = try query.ast_file(query.context, resolved.module_id);
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
            const contract_id = function.parent_contract orelse return null;
            const block = self.parent.contract_body_blocks[contract_id.index()];
            if (mlir.oraBlockIsNull(block)) return null;
            return block;
        }

        fn currentContractItemId(self: *FunctionLowerer) ?ast.ItemId {
            const function = self.function orelse return null;
            return function.parent_contract;
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
                    const target_module_id = @This().importedModuleForExpr(self, field.base) orelse break :blk null;
                    const target_item_id = (query.lookup_item(query.context, target_module_id, field.name) catch null) orelse break :blk null;
                    const target_file = try query.ast_file(query.context, target_module_id);
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
            return self.defaultValue(self.parent.lowerExprType(expr_id), builtin.range);
        }

        fn lowerDivisionBuiltin(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            builtin: ast.BuiltinExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(builtin.range);
            const result_type = self.parent.lowerExprType(expr_id);
            if (builtin.args.len < 2) return self.defaultValue(result_type, builtin.range);

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
            const is_signed = @This().isSignedIntegerExpr(self, builtin.args[0]) or
                (self.parent.file.expression(builtin.args[0]).* == .IntegerLiteral and @This().isSignedIntegerExpr(self, builtin.args[1]));
            const div_op = if (is_signed)
                mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs)
            else
                mlir.oraArithDivUIOpCreate(self.parent.context, loc, lhs, rhs);
            if (mlir.oraOperationIsNull(div_op)) return self.defaultValue(result_type, builtin.range);
            const quotient = appendValueOp(self.block, div_op);

            const rem_op = if (is_signed)
                mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs)
            else
                mlir.oraArithRemUIOpCreate(self.parent.context, loc, lhs, rhs);
            if (mlir.oraOperationIsNull(rem_op)) return self.defaultValue(result_type, builtin.range);
            const remainder = appendValueOp(self.block, rem_op);

            if (std.mem.eql(u8, builtin.name, "divmod")) {
                const tuple_op = mlir.oraTupleCreateOpCreate(
                    self.parent.context,
                    loc,
                    &[_]mlir.MlirValue{ quotient, remainder },
                    2,
                    result_type,
                );
                if (mlir.oraOperationIsNull(tuple_op)) return self.defaultValue(result_type, builtin.range);
                return appendValueOp(self.block, tuple_op);
            }

            if (std.mem.eql(u8, builtin.name, "divTrunc")) return quotient;

            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            const remainder_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", remainder, zero));

            if (std.mem.eql(u8, builtin.name, "divExact")) {
                const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, appendValueOp(self.block, self.createCompareOp(loc, "eq", remainder, zero)), strRef("exact division requires zero remainder"));
                if (mlir.oraOperationIsNull(assert_op)) return self.defaultValue(result_type, builtin.range);
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
            value = try @This().convertBuiltinCastValue(self, value, concrete_target, builtin.range, checked);

            if (!mlir.oraTypeIsNull(target_ref_base) and !mlir.oraTypeEqual(mlir.oraValueGetType(value), target_type)) {
                const wrap_op = mlir.oraBaseToRefinementOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    value,
                    target_type,
                    self.block,
                );
                if (mlir.oraOperationIsNull(wrap_op)) return value;
                return mlir.oraOperationGetResult(wrap_op, 0);
            }
            return value;
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
                    if (!mlir.oraOperationIsNull(wrap_op)) return mlir.oraOperationGetResult(wrap_op, 0);
                }
                return value;
            }

            const bitcast = mlir.oraArithBitcastOpCreate(
                self.parent.context,
                self.parent.location(builtin.range),
                value,
                concrete_target,
            );
            if (mlir.oraOperationIsNull(bitcast)) return value;
            const casted = appendValueOp(self.block, bitcast);

            if (!mlir.oraTypeIsNull(target_ref_base)) {
                const wrap_op = mlir.oraBaseToRefinementOpCreate(
                    self.parent.context,
                    self.parent.location(builtin.range),
                    casted,
                    target_type,
                    self.block,
                );
                if (!mlir.oraOperationIsNull(wrap_op)) return mlir.oraOperationGetResult(wrap_op, 0);
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
            range: source.TextRange,
            checked: bool,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, target_type)) return value;

            const loc = self.parent.location(range);
            const value_is_int = mlir.oraTypeIsAInteger(value_type);
            const target_is_int = mlir.oraTypeIsAInteger(target_type);

            if (mlir.oraTypeIsAddressType(value_type) and target_is_int and mlir.oraIntegerTypeGetWidth(target_type) == 160) {
                const op = mlir.oraAddrToI160OpCreate(self.parent.context, loc, value);
                if (mlir.oraOperationIsNull(op)) return value;
                return appendValueOp(self.block, op);
            }

            if (value_is_int and mlir.oraTypeIsAddressType(target_type) and mlir.oraIntegerTypeGetWidth(value_type) == 160) {
                const op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, value);
                if (mlir.oraOperationIsNull(op)) return value;
                return appendValueOp(self.block, op);
            }

            if (!(value_is_int and target_is_int)) return value;

            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const target_width = mlir.oraIntegerTypeGetWidth(target_type);
            if (value_width == target_width) {
                const op = mlir.oraArithBitcastOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return value;
                return appendValueOp(self.block, op);
            }

            if (value_width < target_width) {
                const op = if (mlir.oraIntegerTypeIsSigned(value_type))
                    mlir.oraArithExtSIOpCreate(self.parent.context, loc, value, target_type)
                else
                    mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return value;
                return appendValueOp(self.block, op);
            }

            if (checked) {
                try @This().emitCastOverflowCheck(self, value, target_type, range);
            }

            const trunc = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, target_type);
            if (mlir.oraOperationIsNull(trunc)) return value;
            return appendValueOp(self.block, trunc);
        }

        fn emitCastOverflowCheck(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            target_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!void {
            const value_type = mlir.oraValueGetType(value);
            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const target_width = mlir.oraIntegerTypeGetWidth(target_type);
            if (target_width >= value_width) return;

            const loc = self.parent.location(range);
            if (!mlir.oraIntegerTypeIsSigned(target_type)) {
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
            if (builtin.args.len < expected_args) return self.defaultValue(result_type, builtin.range);

            const lhs = try @This().unwrapRefinementForCast(self, try self.lowerExpr(builtin.args[0], locals), exprRange(self.parent.file, builtin.args[0]));
            const lhs_type = mlir.oraValueGetType(lhs);
            const is_signed = mlir.oraTypeIsAInteger(lhs_type) and mlir.oraIntegerTypeIsSigned(lhs_type);

            var value: mlir.MlirValue = lhs;
            var overflow_flag: mlir.MlirValue = undefined;

            if (std.mem.eql(u8, builtin.name, "negWithOverflow")) {
                const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, lhs_type, 0));
                const sub_op = mlir.oraArithSubIOpCreate(self.parent.context, loc, zero, lhs);
                if (mlir.oraOperationIsNull(sub_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, sub_op);
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
                if (mlir.oraOperationIsNull(add_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, add_op);
                overflow_flag = if (is_signed)
                    try @This().computeSignedAddOverflow(self, value, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", value, lhs));
            } else if (std.mem.eql(u8, builtin.name, "subWithOverflow")) {
                const sub_op = mlir.oraArithSubIOpCreate(self.parent.context, loc, lhs, rhs);
                if (mlir.oraOperationIsNull(sub_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, sub_op);
                overflow_flag = if (is_signed)
                    try @This().computeSignedSubOverflow(self, value, lhs, rhs, loc)
                else
                    appendValueOp(self.block, self.createCompareOp(loc, "ult", lhs, rhs));
            } else if (std.mem.eql(u8, builtin.name, "mulWithOverflow")) {
                const mul_op = mlir.oraArithMulIOpCreate(self.parent.context, loc, lhs, rhs);
                if (mlir.oraOperationIsNull(mul_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, mul_op);
                overflow_flag = if (is_signed)
                    try @This().computeSignedMulOverflow(self, value, lhs, rhs, lhs_type, loc)
                else
                    try @This().computeUnsignedMulOverflow(self, value, lhs, rhs, lhs_type, loc);
            } else if (std.mem.eql(u8, builtin.name, "powerWithOverflow")) {
                const power = try self.lowerPowerWithOverflow(lhs, rhs, lhs_type, builtin.range);
                value = power.value;
                overflow_flag = power.overflow;
            } else if (std.mem.eql(u8, builtin.name, "shlWithOverflow")) {
                const shl_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, lhs, rhs);
                if (mlir.oraOperationIsNull(shl_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, shl_op);
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
                if (mlir.oraOperationIsNull(shr_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, shr_op);
                overflow_flag = try @This().makeFalse(self, loc);
            } else if (std.mem.eql(u8, builtin.name, "divWithOverflow") or std.mem.eql(u8, builtin.name, "modWithOverflow")) {
                const arith_op = if (std.mem.eql(u8, builtin.name, "divWithOverflow"))
                    (if (is_signed) mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithDivUIOpCreate(self.parent.context, loc, lhs, rhs))
                else
                    (if (is_signed) mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs) else mlir.oraArithRemUIOpCreate(self.parent.context, loc, lhs, rhs));
                if (mlir.oraOperationIsNull(arith_op)) return self.defaultValue(result_type, builtin.range);
                value = appendValueOp(self.block, arith_op);
                overflow_flag = try @This().makeFalse(self, loc);
            } else {
                return self.defaultValue(result_type, builtin.range);
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
            if (mlir.oraOperationIsNull(op)) return self.defaultValue(result_type, range);
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
            if (expr_type.kind() == .anonymous_struct) {
                return try @This().lowerAnonymousStructLiteral(self, expr_id, struct_literal, expr_type.anonymous_struct.fields, locals);
            }
            const concrete_name = expr_type.name() orelse struct_literal.type_name;
            const struct_item_id = self.parent.item_index.lookup(concrete_name) orelse {
                if (self.parent.typecheck.instantiatedStructByName(concrete_name)) |instantiated| {
                    return try @This().lowerInstantiatedStructLiteral(self, expr_id, struct_literal, instantiated.fields, concrete_name, locals);
                }
                return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, &.{}, self.parent.lowerExprType(expr_id)));
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
                    const field_value = try self.lowerExpr(init.value, locals);
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
                    return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, &.{}, self.parent.lowerExprType(expr_id)));
                },
            };

            const result_type = self.parent.lowerExprType(expr_id);
            var operands: std.ArrayList(mlir.MlirValue) = .{};
            for (struct_item.fields) |decl_field| {
                const init = findStructFieldInit(struct_literal.fields, decl_field.name) orelse {
                    return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type));
                };
                const field_type = self.parent.lowerTypeExpr(decl_field.type_expr);
                const raw_value = try self.lowerExpr(init.value, locals);
                const value = try self.convertValueForFlow(raw_value, field_type, init.range);
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
            if (mlir.oraOperationIsNull(op)) {
                return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type));
            }
            return appendValueOp(self.block, op);
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
            for (fields) |decl_field| {
                const init = findStructFieldInit(struct_literal.fields, decl_field.name) orelse {
                    return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type));
                };
                const field_type = self.parent.lowerSemaType(decl_field.ty, init.range);
                const raw_value = try self.lowerExpr(init.value, locals);
                const value = try self.convertValueForFlow(raw_value, field_type, init.range);
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
            if (mlir.oraOperationIsNull(op)) {
                return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type));
            }
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
            for (fields) |decl_field| {
                const init = findStructFieldInit(struct_literal.fields, decl_field.name) orelse {
                    return appendValueOp(self.block, try self.createAggregatePlaceholder("ora.tuple.create", struct_literal.range, operands.items, result_type));
                };
                const field_type = self.parent.lowerSemaType(decl_field.ty, init.range);
                const raw_value = try self.lowerExpr(init.value, locals);
                const value = try self.convertValueForFlow(raw_value, field_type, init.range);
                try operands.append(self.parent.allocator, value);
            }
            const op = mlir.oraTupleCreateOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                if (operands.items.len == 0) null else operands.items.ptr,
                operands.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) {
                const placeholder = try self.createAggregatePlaceholder("ora.tuple.create", struct_literal.range, operands.items, result_type);
                return appendValueOp(self.block, placeholder);
            }
            return appendValueOp(self.block, op);
        }

        fn lowerFieldExpr(self: *FunctionLowerer, expr_id: ast.ExprId, field: ast.FieldExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const result_type = self.parent.lowerExprType(expr_id);
            if (try @This().lowerBuiltinFieldExpr(self, expr_id, field, result_type)) |value| {
                return value;
            }
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
                if (@This().enumFieldOrdinal(self, field.base, field.name)) |ordinal| {
                    return appendValueOp(
                        self.block,
                        createIntegerConstant(self.parent.context, self.parent.location(field.range), result_type, ordinal),
                    );
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
            if (@This().anonymousStructFieldIndex(base_type, field.name)) |tuple_index| {
                const op = mlir.oraTupleExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    tuple_index,
                    result_type,
                );
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
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
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
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

        fn anonymousStructFieldIndex(base_type: sema.Type, field_name: []const u8) ?i64 {
            if (base_type.kind() != .anonymous_struct) return null;
            for (base_type.anonymous_struct.fields, 0..) |field, index| {
                if (std.mem.eql(u8, field.name, field_name)) return @intCast(index);
            }
            return null;
        }

        fn enumFieldOrdinal(self: *FunctionLowerer, base_expr: ast.ExprId, field_name: []const u8) ?i64 {
            const base_type = self.parent.typecheck.exprType(base_expr);
            const enum_name = base_type.name() orelse return null;
            const item_id = self.parent.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |item| item,
                else => return null,
            };
            for (enum_item.variants, 0..) |variant, index| {
                if (std.mem.eql(u8, variant.name, field_name)) return @intCast(index);
            }
            return null;
        }

        fn lowerBuiltinFieldExpr(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            field: ast.FieldExpr,
            result_type: mlir.MlirType,
        ) anyerror!?mlir.MlirValue {
            const path = try @This().fieldExprPath(self, expr_id);
            if (try @This().lowerBuiltinFieldCall(self, field, result_type, path)) |value| {
                return value;
            }
            return null;
        }

        fn lowerBuiltinFieldCall(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            result_type: mlir.MlirType,
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
                    result_type;

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

        fn importedModuleForExpr(self: *FunctionLowerer, expr_id: ast.ExprId) ?source.ModuleId {
            const query = self.parent.module_query orelse return null;
            return switch (self.parent.file.expression(expr_id).*) {
                .Name => |name| query.resolve_import_alias(query.context, self.parent.module_id, name.name) catch null,
                .Field => |field| blk: {
                    const base_module_id = @This().importedModuleForExpr(self, field.base) orelse break :blk null;
                    break :blk query.resolve_import_alias(query.context, base_module_id, field.name) catch null;
                },
                .Group => |group| @This().importedModuleForExpr(self, group.expr),
                else => null,
            };
        }

        fn lowerImportedFieldExpr(
            self: *FunctionLowerer,
            field: ast.FieldExpr,
            result_type: mlir.MlirType,
        ) anyerror!?mlir.MlirValue {
            const query = self.parent.module_query orelse return null;
            const target_module_id = @This().importedModuleForExpr(self, field.base) orelse return null;
            const target_item_id = (query.lookup_item(query.context, target_module_id, field.name) catch null) orelse return null;
            const target_file = try query.ast_file(query.context, target_module_id);
            const target_typecheck = try query.module_typecheck(query.context, target_module_id);
            const target_const_eval = try query.const_eval(query.context, target_module_id);

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
                .Contract => |contract| {
                    for (contract.members) |member_id| {
                        const member = self.parent.file.item(member_id).*;
                        switch (member) {
                            .Field => |contract_field| {
                                if (!std.mem.eql(u8, contract_field.name, field.name)) continue;
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
                            .Constant => |constant| {
                                if (std.mem.eql(u8, constant.name, field.name)) {
                                    return try self.lowerExpr(constant.value, locals);
                                }
                            },
                            .Function => |function| {
                                if (std.mem.eql(u8, function.name, field.name)) {
                                    const op = mlir.oraFunctionRefOpCreate(
                                        self.parent.context,
                                        self.parent.location(field.range),
                                        strRef(function.name),
                                        result_type,
                                    );
                                    if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                                }
                            },
                            .Struct => |struct_item| {
                                if (std.mem.eql(u8, struct_item.name, field.name)) return null;
                            },
                            .Enum => |enum_item| {
                                if (std.mem.eql(u8, enum_item.name, field.name)) return null;
                            },
                            .Bitfield => |bitfield_item| {
                                if (std.mem.eql(u8, bitfield_item.name, field.name)) return null;
                            },
                            else => {},
                        }
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

        fn findStructFieldInit(fields: []const ast.StructFieldInit, name: []const u8) ?ast.StructFieldInit {
            for (fields) |field| {
                if (std.mem.eql(u8, field.name, name)) return field;
            }
            return null;
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
            var attrs: [1]mlir.MlirNamedAttribute = .{namedStringAttr(self.parent.context, "value", text)};
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
            var attrs: [1]mlir.MlirNamedAttribute = .{namedTypeAttr(self.parent.context, "ora.type", result_type)};
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

        pub fn defaultValue(self: *FunctionLowerer, ty: mlir.MlirType, range: source.TextRange) anyerror!mlir.MlirValue {
            if (mlir.oraTypeIsNull(ty)) return error.MlirOperationCreationFailed;
            if (self.typeIsVoid(ty)) {
                const seed = appendValueOp(
                    self.block,
                    createIntegerConstant(self.parent.context, self.parent.location(range), boolType(self.parent.context), 0),
                );
                const none_cast = mlir.oraUnrealizedConversionCastOpCreate(
                    self.parent.context,
                    self.parent.location(range),
                    seed,
                    ty,
                );
                if (mlir.oraOperationIsNull(none_cast)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, none_cast);
            }

            if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(ty))) {
                const payload_type = mlir.oraErrorUnionTypeGetSuccessType(ty);
                const payload = try self.defaultValue(payload_type, range);
                const op = mlir.oraErrorOkOpCreate(self.parent.context, self.parent.location(range), payload, ty);
                if (!mlir.oraOperationIsNull(op)) {
                    if (self.function != null and self.parent.errorUnionRequiresWideCarrier(self.parent.typecheck.body_types[self.function.?.body.index()])) {
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.parent.context, true));
                    }
                    return appendValueOp(self.block, op);
                }
                return payload;
            }

            const refinement_base = mlir.oraRefinementTypeGetBaseType(ty);
            if (!mlir.oraTypeIsNull(refinement_base)) {
                const op = try self.createAggregatePlaceholder("ora.default_value", range, &.{}, ty);
                return appendValueOp(self.block, op);
            }

            if (mlir.oraTypeIsAddressType(ty)) {
                return try @This().createZeroAddressValue(self, range);
            }

            if (mlir.oraTypeEqual(ty, stringType(self.parent.context))) {
                const op = mlir.oraStringConstantOpCreate(self.parent.context, self.parent.location(range), strRef(""), ty);
                mlir.oraOperationSetAttributeByName(
                    op,
                    strRef("length"),
                    mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(self.parent.context, 32), 0),
                );
                return appendValueOp(self.block, op);
            }

            if (mlir.oraTypeEqual(ty, bytesType(self.parent.context))) {
                const op = mlir.oraBytesConstantOpCreate(self.parent.context, self.parent.location(range), strRef("0x"), ty);
                mlir.oraOperationSetAttributeByName(
                    op,
                    strRef("length"),
                    mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(self.parent.context, 32), 0),
                );
                return appendValueOp(self.block, op);
            }

            const op = if (mlir.oraTypeIsAInteger(ty) or mlir.oraTypeIsIntegerType(ty))
                createIntegerConstant(self.parent.context, self.parent.location(range), ty, 0)
            else
                try self.createAggregatePlaceholder("ora.default_value", range, &.{}, ty);
            return appendValueOp(self.block, op);
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
