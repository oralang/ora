const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const const_values = @import("../sema/const_values.zig");
const source = @import("../source/mod.zig");
const hir_locals = @import("locals.zig");
const support = @import("support.zig");

const addressType = support.addressType;
const appendOp = support.appendOp;
const appendScfYieldValues = support.appendScfYieldValues;
const appendValueOp = support.appendValueOp;
const boolType = support.boolType;
const bytesType = support.bytesType;
const cmpPredicate = support.cmpPredicate;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const exprRange = support.exprRange;
const nullStringRef = support.nullStringRef;
const namedStringAttr = support.namedStringAttr;
const namedTypeAttr = support.namedTypeAttr;
const parseIntLiteral = support.parseIntLiteral;
const strRef = support.strRef;
const stringType = support.stringType;
const LocalEnv = hir_locals.LocalEnv;

pub fn mixin(FunctionLowerer: type, Lowerer: type) type {
    _ = Lowerer;
    return struct {
        pub fn lowerExpr(self: *FunctionLowerer, expr_id: ast.ExprId, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const expr = self.parent.file.expression(expr_id).*;
            const loc = self.parent.location(exprRange(self.parent.file, expr_id));
            return switch (expr) {
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
                    if (mlir.oraOperationIsNull(op)) {
                        const placeholder = try self.createValuePlaceholder("ora.string_const", literal.text, literal.range, stringType(self.parent.context));
                        break :blk appendValueOp(self.block, placeholder);
                    }
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
                    if (mlir.oraAttributeIsNull(addr_attr)) {
                        const placeholder = try self.createValuePlaceholder("ora.address_const", literal.text, literal.range, addressType(self.parent.context));
                        break :blk appendValueOp(self.block, placeholder);
                    }
                    const int_const = mlir.oraArithConstantOpCreate(self.parent.context, self.parent.location(literal.range), i160_type, addr_attr);
                    if (mlir.oraOperationIsNull(int_const)) {
                        const placeholder = try self.createValuePlaceholder("ora.address_const", literal.text, literal.range, addressType(self.parent.context));
                        break :blk appendValueOp(self.block, placeholder);
                    }
                    const i160_value = appendValueOp(self.block, int_const);
                    const addr_op = mlir.oraI160ToAddrOpCreate(self.parent.context, self.parent.location(literal.range), i160_value);
                    if (mlir.oraOperationIsNull(addr_op)) {
                        const placeholder = try self.createValuePlaceholder("ora.address_const", literal.text, literal.range, addressType(self.parent.context));
                        break :blk appendValueOp(self.block, placeholder);
                    }
                    break :blk appendValueOp(self.block, addr_op);
                },
                .BytesLiteral => |literal| blk: {
                    const op = mlir.oraBytesConstantOpCreate(
                        self.parent.context,
                        self.parent.location(literal.range),
                        strRef(literal.text),
                        bytesType(self.parent.context),
                    );
                    if (mlir.oraOperationIsNull(op)) {
                        const placeholder = try self.createValuePlaceholder("ora.bytes_const", literal.text, literal.range, bytesType(self.parent.context));
                        break :blk appendValueOp(self.block, placeholder);
                    }
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
                    var operands: std.ArrayList(mlir.MlirValue) = .{};
                    for (tuple.elements) |element| try operands.append(self.parent.allocator, try self.lowerExpr(element, locals));
                    const result_type = self.parent.lowerExprType(expr_id);
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
                    const op = try @This().lowerStructLiteral(self, expr_id, struct_literal, locals);
                    break :blk appendValueOp(self.block, op);
                },
                .Switch => |switch_expr| try self.lowerSwitchExpr(expr_id, switch_expr, locals),
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
                    const op = mlir.oraErrorReturnOpCreate(
                        self.parent.context,
                        loc,
                        strRef(error_return.name),
                        if (args.items.len == 0) null else args.items.ptr,
                        args.items.len,
                        result_type,
                    );
                    if (!mlir.oraOperationIsNull(op)) break :blk appendValueOp(self.block, op);
                    const placeholder = try self.createAggregatePlaceholder("ora.error.return", error_return.range, args.items, result_type);
                    break :blk appendValueOp(self.block, placeholder);
                },
                .Field => |field| blk: {
                    const op = try @This().lowerFieldExpr(self, expr_id, field, locals);
                    break :blk appendValueOp(self.block, op);
                },
                .Index => |index| blk: {
                    const base = try self.lowerExpr(index.base, locals);
                    const key = try self.lowerExpr(index.index, locals);
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
                    const base_type = self.parent.typecheck.exprType(index.base);
                    if (base_type == .tuple) {
                        const tuple_index = @This().constTupleIndex(self, index.index) orelse {
                            const op = try self.createAggregatePlaceholder("ora.index_access", index.range, &.{base, key}, self.parent.lowerExprType(expr_id));
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
                        const op = mlir.oraMapGetOpCreate(self.parent.context, self.parent.location(index.range), base, key, result_type);
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
                    .item => |item_id| {
                        const item = self.parent.file.item(item_id).*;
                        switch (item) {
                            .Field => |field| {
                                if (field.binding_kind == .immutable) {
                                    if (field.value) |value| return self.lowerExpr(value, locals);
                                }
                                const result_type = if (field.type_expr) |type_expr| self.parent.lowerTypeExpr(type_expr) else self.parent.lowerExprType(expr_id);
                                const op = switch (field.storage_class) {
                                    .storage => mlir.oraSLoadOpCreate(self.parent.context, self.parent.location(field.range), strRef(field.name), result_type),
                                    .memory => mlir.oraMLoadOpCreate(self.parent.context, self.parent.location(field.range), strRef(field.name), result_type),
                                    .tstore => mlir.oraTLoadOpCreate(self.parent.context, self.parent.location(field.range), strRef(field.name), result_type),
                                    .none => std.mem.zeroes(mlir.MlirOperation),
                                };
                                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                            },
                            .Constant => |constant| return self.lowerExpr(constant.value, locals),
                            .Function => {
                                const result_type = self.parent.lowerExprType(expr_id);
                                const function_name = switch (self.parent.file.item(binding.item).*) {
                                    .Function => |function| function.name,
                                    else => "",
                                };
                                const op = mlir.oraFunctionRefOpCreate(
                                    self.parent.context,
                                    self.parent.location(name.range),
                                    strRef(function_name),
                                    result_type,
                                );
                                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                                const placeholder = try self.createAggregatePlaceholder("ora.function_ref", name.range, &.{}, result_type);
                                return appendValueOp(self.block, placeholder);
                            },
                            else => {},
                        }
                    },
                }
            }

            if (locals.lookupName(name.name)) |local_id| {
                if (locals.getValue(local_id)) |value| return value;
            }

            return self.defaultValue(self.parent.lowerExprType(expr_id), name.range);
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

            const try_stmt = mlir.oraTryStmtOpCreate(self.parent.context, loc, &[_]mlir.MlirType{result_type}, 1);
            if (mlir.oraOperationIsNull(try_stmt)) return operand;
            appendOp(self.block, try_stmt);

            const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
            const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);
            if (mlir.oraBlockIsNull(try_block) or mlir.oraBlockIsNull(catch_block)) {
                return error.MlirOperationCreationFailed;
            }

            const unwrap = mlir.oraErrorUnwrapOpCreate(self.parent.context, loc, operand, result_type);
            if (mlir.oraOperationIsNull(unwrap)) return operand;
            appendOp(try_block, unwrap);
            try support.appendOraYieldValues(self.parent.context, try_block, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap, 0)});

            var catch_lowerer = self.*;
            catch_lowerer.block = catch_block;
            const fallback = try catch_lowerer.defaultValue(result_type, range);
            try support.appendOraYieldValues(self.parent.context, catch_block, loc, &[_]mlir.MlirValue{fallback});

            return mlir.oraOperationGetResult(try_stmt, 0);
        }

        pub fn lowerBinary(self: *FunctionLowerer, expr_id: ast.ExprId, binary: ast.BinaryExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const lhs = try self.lowerExpr(binary.lhs, locals);
            const rhs = try self.lowerExpr(binary.rhs, locals);
            const loc = self.parent.location(binary.range);
            const result_type = self.parent.lowerExprType(expr_id);

            if (binary.op == .pow) {
                return @This().lowerCheckedPower(self, lhs, rhs, result_type, binary.range);
            }

            const op = switch (binary.op) {
                .add => mlir.oraArithAddIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_add => mlir.oraAddWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .sub => mlir.oraArithSubIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_sub => mlir.oraSubWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .mul => mlir.oraArithMulIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_mul => mlir.oraMulWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .div => mlir.oraArithDivSIOpCreate(self.parent.context, loc, lhs, rhs),
                .mod => mlir.oraArithRemSIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_pow => mlir.oraPowerOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .bit_and, .and_and => mlir.oraArithAndIOpCreate(self.parent.context, loc, lhs, rhs),
                .bit_or, .or_or => mlir.oraArithOrIOpCreate(self.parent.context, loc, lhs, rhs),
                .bit_xor => mlir.oraArithXorIOpCreate(self.parent.context, loc, lhs, rhs),
                .shl => mlir.oraArithShlIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_shl => mlir.oraShlWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .shr => mlir.oraArithShrSIOpCreate(self.parent.context, loc, lhs, rhs),
                .wrapping_shr => mlir.oraShrWrappingOpCreate(self.parent.context, loc, lhs, rhs, result_type),
                .eq => self.createCompareOp(loc, "eq", lhs, rhs),
                .ne => self.createCompareOp(loc, "ne", lhs, rhs),
                .lt => self.createCompareOp(loc, "slt", lhs, rhs),
                .le => self.createCompareOp(loc, "sle", lhs, rhs),
                .gt => self.createCompareOp(loc, "sgt", lhs, rhs),
                .ge => self.createCompareOp(loc, "sge", lhs, rhs),
                .pow => unreachable,
            };

            if (mlir.oraOperationIsNull(op)) {
                return self.defaultValue(result_type, binary.range);
            }
            return appendValueOp(self.block, op);
        }

        fn lowerCheckedPower(
            self: *FunctionLowerer,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const is_signed = mlir.oraTypeIsAInteger(result_type) and mlir.oraIntegerTypeIsSigned(result_type);
            const value_ty = result_type;
            const bool_ty = boolType(self.parent.context);
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            var result = one;
            var factor = lhs;
            var overflow = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, bool_ty, 0));

            const bit_width: u32 = if (mlir.oraTypeIsAInteger(value_ty))
                @intCast(mlir.oraIntegerTypeGetWidth(value_ty))
            else
                256;

            var bit_index: u32 = 0;
            while (bit_index < bit_width) : (bit_index += 1) {
                const shift_amount = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_index)));
                const shifted = try @This().createArithShiftRightUnsigned(self, rhs, shift_amount, loc);
                const current_bit = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, shifted, one));
                const bit_is_set = appendValueOp(self.block, self.createCompareOp(loc, "ne", current_bit, zero));

                const multiplied = appendValueOp(self.block, mlir.oraArithMulIOpCreate(self.parent.context, loc, result, factor));
                const mul_overflow = try @This().computeMulOverflowFlag(self, multiplied, result, factor, value_ty, is_signed, range);
                const overflow_with_mul = appendValueOp(self.block, mlir.oraArithOrIOpCreate(self.parent.context, loc, overflow, mul_overflow));
                const after_mul = try @This().selectPowerState(self, bit_is_set, multiplied, overflow_with_mul, result, overflow, value_ty, bool_ty, loc);
                result = after_mul.value;
                overflow = after_mul.overflow;

                if (bit_index + 1 < bit_width) {
                    const next_shift_amount = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_index + 1)));
                    const remaining = try @This().createArithShiftRightUnsigned(self, rhs, next_shift_amount, loc);
                    const has_more_bits = appendValueOp(self.block, self.createCompareOp(loc, "ne", remaining, zero));

                    const squared = appendValueOp(self.block, mlir.oraArithMulIOpCreate(self.parent.context, loc, factor, factor));
                    const sq_overflow = try @This().computeMulOverflowFlag(self, squared, factor, factor, value_ty, is_signed, range);
                    const overflow_with_square = appendValueOp(self.block, mlir.oraArithOrIOpCreate(self.parent.context, loc, overflow, sq_overflow));
                    const after_square = try @This().selectPowerState(self, has_more_bits, squared, overflow_with_square, factor, overflow, value_ty, bool_ty, loc);
                    factor = after_square.value;
                    overflow = after_square.overflow;
                }
            }

            const power_op = mlir.oraPowerOpCreate(self.parent.context, loc, lhs, rhs, value_ty);
            if (mlir.oraOperationIsNull(power_op)) return self.defaultValue(result_type, range);
            const power_value = appendValueOp(self.block, power_op);
            try @This().emitOverflowAssert(self, overflow, "checked power overflow", range);
            return power_value;
        }

        fn createArithShiftRightUnsigned(self: *FunctionLowerer, lhs: mlir.MlirValue, rhs: mlir.MlirValue, loc: mlir.MlirLocation) anyerror!mlir.MlirValue {
            const op = mlir.oraArithShrUIOpCreate(self.parent.context, loc, lhs, rhs);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn selectPowerState(
            self: *FunctionLowerer,
            condition: mlir.MlirValue,
            then_value: mlir.MlirValue,
            then_overflow: mlir.MlirValue,
            else_value: mlir.MlirValue,
            else_overflow: mlir.MlirValue,
            value_ty: mlir.MlirType,
            flag_ty: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!struct { value: mlir.MlirValue, overflow: mlir.MlirValue } {
            const result_types = [_]mlir.MlirType{ value_ty, flag_ty };
            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, condition, &result_types, result_types.len, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{ then_value, then_overflow });
            try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{ else_value, else_overflow });

            return .{
                .value = mlir.oraOperationGetResult(if_op, 0),
                .overflow = mlir.oraOperationGetResult(if_op, 1),
            };
        }

        fn computeMulOverflowFlag(
            self: *FunctionLowerer,
            product: mlir.MlirValue,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            value_ty: mlir.MlirType,
            is_signed: bool,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            const rhs_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", rhs, zero));

            if (!is_signed) {
                const quot_op = mlir.oraArithDivUIOpCreate(self.parent.context, loc, product, rhs);
                if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
                mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
                const quotient = appendValueOp(self.block, quot_op);
                const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", quotient, lhs));
                const and_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, mismatch, rhs_non_zero);
                if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, and_op);
            }

            const bit_width: u32 = if (mlir.oraTypeIsAInteger(value_ty))
                @intCast(mlir.oraIntegerTypeGetWidth(value_ty))
            else
                256;
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));
            const shift_amt = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_width - 1)));
            const min_int_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, one, shift_amt);
            if (mlir.oraOperationIsNull(min_int_op)) return error.MlirOperationCreationFailed;
            const min_int = appendValueOp(self.block, min_int_op);
            const neg1 = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, -1));

            const lhs_is_min = appendValueOp(self.block, self.createCompareOp(loc, "eq", lhs, min_int));
            const rhs_is_neg1 = appendValueOp(self.block, self.createCompareOp(loc, "eq", rhs, neg1));
            const special_case = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, lhs_is_min, rhs_is_neg1));

            const quot_op = mlir.oraArithDivSIOpCreate(self.parent.context, loc, product, rhs);
            if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
            mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
            const quotient = appendValueOp(self.block, quot_op);
            const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", quotient, lhs));
            const general_case = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, mismatch, rhs_non_zero));
            const overflow_op = mlir.oraArithOrIOpCreate(self.parent.context, loc, special_case, general_case);
            if (mlir.oraOperationIsNull(overflow_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, overflow_op);
        }

        fn emitOverflowAssert(self: *FunctionLowerer, overflow_flag: mlir.MlirValue, message: []const u8, range: source.TextRange) anyerror!void {
            const loc = self.parent.location(range);
            const true_val = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
            const not_op = mlir.oraArithXorIOpCreate(self.parent.context, loc, overflow_flag, true_val);
            if (mlir.oraOperationIsNull(not_op)) return error.MlirOperationCreationFailed;
            const no_overflow = appendValueOp(self.block, not_op);
            const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, no_overflow, strRef(message));
            if (mlir.oraOperationIsNull(assert_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, assert_op);
        }

        pub fn createCompareOp(self: *FunctionLowerer, loc: mlir.MlirLocation, predicate: []const u8, lhs: mlir.MlirValue, rhs: mlir.MlirValue) mlir.MlirOperation {
            const lhs_type = mlir.oraValueGetType(lhs);
            const rhs_type = mlir.oraValueGetType(rhs);
            if (mlir.oraTypeIsAddressType(lhs_type) or mlir.oraTypeIsAddressType(rhs_type)) {
                return mlir.oraCmpOpCreate(self.parent.context, loc, strRef(predicate), lhs, rhs, boolType(self.parent.context));
            }
            return mlir.oraArithCmpIOpCreate(self.parent.context, loc, cmpPredicate(predicate), lhs, rhs);
        }

        pub fn lowerCall(self: *FunctionLowerer, expr_id: ast.ExprId, call: ast.CallExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            var args: std.ArrayList(mlir.MlirValue) = .{};
            for (call.args) |arg| {
                try args.append(self.parent.allocator, try self.lowerExpr(arg, locals));
            }

            const callee_name = switch (self.parent.file.expression(call.callee).*) {
                .Name => |name| name.name,
                else => null,
            } orelse return self.defaultValue(self.parent.lowerExprType(expr_id), call.range);

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
            if (mlir.oraOperationIsNull(op)) return self.defaultValue(result_type, call.range);
            if (self.typeIsVoid(result_type)) {
                appendOp(self.block, op);
                return self.defaultValue(defaultIntegerType(self.parent.context), call.range);
            }
            return appendValueOp(self.block, op);
        }

        pub fn lowerBuiltin(self: *FunctionLowerer, expr_id: ast.ExprId, builtin: ast.BuiltinExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            if (std.mem.eql(u8, builtin.name, "cast") and builtin.args.len > 0) {
                return self.lowerExpr(builtin.args[0], locals);
            }
            if (builtin.args.len >= 2 and (std.mem.eql(u8, builtin.name, "divTrunc") or
                std.mem.eql(u8, builtin.name, "divFloor") or
                std.mem.eql(u8, builtin.name, "divCeil") or
                std.mem.eql(u8, builtin.name, "divExact")))
            {
                const lhs = try self.lowerExpr(builtin.args[0], locals);
                const rhs = try self.lowerExpr(builtin.args[1], locals);
                const op = mlir.oraArithDivSIOpCreate(self.parent.context, self.parent.location(builtin.range), lhs, rhs);
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
            }
            if (builtin.args.len > 0 and std.mem.eql(u8, builtin.name, "truncate")) {
                return self.lowerExpr(builtin.args[0], locals);
            }
            return self.defaultValue(self.parent.lowerExprType(expr_id), builtin.range);
        }

        fn lowerStructLiteral(
            self: *FunctionLowerer,
            expr_id: ast.ExprId,
            struct_literal: ast.StructLiteralExpr,
            locals: *LocalEnv,
        ) anyerror!mlir.MlirOperation {
            const struct_item_id = self.parent.item_index.lookup(struct_literal.type_name) orelse {
                return self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, &.{}, self.parent.lowerExprType(expr_id));
            };
            const struct_item = switch (self.parent.file.item(struct_item_id).*) {
                .Struct => |item| item,
                else => return self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, &.{}, self.parent.lowerExprType(expr_id)),
            };

            const result_type = self.parent.lowerExprType(expr_id);
            var operands: std.ArrayList(mlir.MlirValue) = .{};
            for (struct_item.fields) |decl_field| {
                const init = findStructFieldInit(struct_literal.fields, decl_field.name) orelse {
                    return self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type);
                };
                try operands.append(self.parent.allocator, try self.lowerExpr(init.value, locals));
            }

            const op = mlir.oraStructInstantiateOpCreate(
                self.parent.context,
                self.parent.location(struct_literal.range),
                strRef(struct_literal.type_name),
                if (operands.items.len == 0) null else operands.items.ptr,
                operands.items.len,
                result_type,
            );
            if (mlir.oraOperationIsNull(op)) {
                return self.createAggregatePlaceholder("ora.struct.create", struct_literal.range, operands.items, result_type);
            }
            return op;
        }

        fn lowerFieldExpr(self: *FunctionLowerer, expr_id: ast.ExprId, field: ast.FieldExpr, locals: *LocalEnv) anyerror!mlir.MlirOperation {
            const base_type = self.parent.typecheck.exprType(field.base);
            if (base_type == .enum_) {
                const enum_name = base_type.name() orelse {
                    return self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, self.parent.lowerExprType(expr_id));
                };
                const result_type = self.parent.lowerExprType(expr_id);
                const op = mlir.oraEnumConstantOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    strRef(enum_name),
                    strRef(field.name),
                    result_type,
                );
                if (!mlir.oraOperationIsNull(op)) return op;
                return self.createAggregatePlaceholder("ora.field_access", field.range, &.{}, result_type);
            }

            const base = try self.lowerExpr(field.base, locals);
            if (base_type == .struct_) {
                const result_type = self.parent.lowerExprType(expr_id);
                const op = mlir.oraStructFieldExtractOpCreate(
                    self.parent.context,
                    self.parent.location(field.range),
                    base,
                    strRef(field.name),
                    result_type,
                );
                if (!mlir.oraOperationIsNull(op)) return op;
            }
            return self.createAggregatePlaceholder("ora.field_access", field.range, &.{base}, self.parent.lowerExprType(expr_id));
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
            for (array.elements, 0..) |element, index| {
                const element_value = try self.lowerExpr(element, locals);
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
            const value = self.parent.const_eval.values[expr_id.index()] orelse return null;
            return switch (value) {
                .integer => |integer| const_values.positiveShiftAmount(integer),
                else => null,
            };
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
            if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(ty))) {
                const payload_type = mlir.oraErrorUnionTypeGetSuccessType(ty);
                const payload = try self.defaultValue(payload_type, range);
                const op = mlir.oraErrorOkOpCreate(self.parent.context, self.parent.location(range), payload, ty);
                if (!mlir.oraOperationIsNull(op)) return appendValueOp(self.block, op);
                return payload;
            }

            const op = if (mlir.oraTypeIsAddressType(ty))
                try self.createValuePlaceholder("ora.address.zero", "0x0", range, ty)
            else
                createIntegerConstant(self.parent.context, self.parent.location(range), ty, 0);
            return appendValueOp(self.block, op);
        }

        pub fn typeIsVoid(self: *const FunctionLowerer, ty: mlir.MlirType) bool {
            const none = mlir.oraNoneTypeCreate(self.parent.context);
            return mlir.oraTypeEqual(ty, none);
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
