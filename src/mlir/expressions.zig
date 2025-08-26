const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Expression lowering system for converting Ora expressions to MLIR operations
pub const ExpressionLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const @import("types.zig").TypeMapper,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const @import("types.zig").TypeMapper) ExpressionLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
        };
    }

    /// Main dispatch function for lowering expressions
    pub fn lowerExpression(self: *const ExpressionLowerer, expr: *const lib.ast.Expressions.ExprNode) c.MlirValue {
        switch (expr.*) {
            .Literal => |lit| return self.lowerLiteral(lit),
            .Binary => |bin| return self.lowerBinary(bin),
            .Unary => |unary| return self.lowerUnary(unary),
            .Identifier => |ident| return self.lowerIdentifier(ident),
            // TODO: Implement other expression types
            else => {
                const ty = c.mlirIntegerTypeGet(self.ctx, 256);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(expr.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                const attr = c.mlirIntegerAttrGet(ty, 0);
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
        }
    }

    /// Lower literal expressions
    pub fn lowerLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.ExprNode) c.MlirValue {
        // Use the existing literal lowering logic from lower.zig
        switch (literal.*) {
            .Literal => |lit| switch (lit) {
                .Integer => |int| {
                    const ty = c.mlirIntegerTypeGet(self.ctx, 256);
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(int.span));
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                    const parsed: i64 = std.fmt.parseInt(i64, int.value, 0) catch 0;
                    const attr = c.mlirIntegerAttrGet(ty, parsed);
                    const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(self.block, op);
                    return c.mlirOperationGetResult(op, 0);
                },
                .Bool => |bool_lit| {
                    const ty = c.mlirIntegerTypeGet(self.ctx, 1);
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(bool_lit.span));
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                    const default_value: i64 = if (bool_lit.value) 1 else 0;
                    const attr = c.mlirIntegerAttrGet(ty, default_value);
                    const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(self.block, op);
                    return c.mlirOperationGetResult(op, 0);
                },
                else => {
                    // For other literal types, return a default value
                    const ty = c.mlirIntegerTypeGet(self.ctx, 256);
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(literal.*.Literal.span));
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                    const attr = c.mlirIntegerAttrGet(ty, 0);
                    const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(self.block, op);
                    return c.mlirOperationGetResult(op, 0);
                },
            },
            else => {
                // For non-literal expressions, delegate to main lowering
                return self.lowerExpression(literal);
            },
        }
    }

    /// Lower identifier expressions (variables, function names, etc.)
    pub fn lowerIdentifier(self: *const ExpressionLowerer, identifier: *const lib.ast.Expressions.IdentifierNode) c.MlirValue {
        // For now, return a dummy value
        // TODO: Implement identifier lowering with symbol table integration
        const ty = c.mlirIntegerTypeGet(self.ctx, 256);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(identifier.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
        const attr = c.mlirIntegerAttrGet(ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Lower binary operator expressions
    pub fn lowerBinaryOp(self: *const ExpressionLowerer, binary_op: *const lib.ast.Expressions.BinaryOpNode) c.MlirValue {
        // TODO: Implement binary operator lowering
        // For now, return a dummy value
        const ty = c.mlirIntegerTypeGet(self.ctx, 256);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(binary_op.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
        const attr = c.mlirIntegerAttrGet(ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Lower unary operator expressions
    pub fn lowerUnaryOp(self: *const ExpressionLowerer, unary_op: *const lib.ast.Expressions.UnaryOpNode) c.MlirValue {
        // TODO: Implement unary operator lowering
        // For now, return a dummy value
        const ty = c.mlirIntegerTypeGet(self.ctx, 256);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(unary_op.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
        const attr = c.mlirIntegerAttrGet(ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Lower binary expressions with all operators
    pub fn lowerBinary(self: *const ExpressionLowerer, bin: *const lib.ast.Expressions.BinaryNode) c.MlirValue {
        const lhs = self.lowerExpression(bin.lhs);
        const rhs = self.lowerExpression(bin.rhs);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, 256);

        switch (bin.operator) {
            // Arithmetic operators
            .Plus => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Minus => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Star => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Slash => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.divsi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Percent => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.remsi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .StarStar => {
                // Power operation - for now use multiplication as placeholder
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },

            // Comparison operators
            .EqualEqual => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const eq_attr = c.mlirStringRefCreateFromCString("eq");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const eq_attr_value = c.mlirStringAttrGet(self.ctx, eq_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, eq_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .BangEqual => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const ne_attr = c.mlirStringRefCreateFromCString("ne");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const ne_attr_value = c.mlirStringAttrGet(self.ctx, ne_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, ne_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Less => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const ult_attr = c.mlirStringRefCreateFromCString("ult");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const ult_attr_value = c.mlirStringAttrGet(self.ctx, ult_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, ult_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .LessEqual => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const ule_attr = c.mlirStringRefCreateFromCString("ule");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const ule_attr_value = c.mlirStringAttrGet(self.ctx, ule_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, ule_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Greater => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const ugt_attr = c.mlirStringRefCreateFromCString("ugt");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const ugt_attr_value = c.mlirStringAttrGet(self.ctx, ugt_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, ugt_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .GreaterEqual => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const uge_attr = c.mlirStringRefCreateFromCString("uge");
                const predicate_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const uge_attr_value = c.mlirStringAttrGet(self.ctx, uge_attr);
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, uge_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },

            // Logical operators
            .And => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.andi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Or => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.ori"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },

            // Bitwise operators
            .BitwiseAnd => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.andi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .BitwiseOr => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.ori"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .BitwiseXor => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .LeftShift => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.shli"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .RightShift => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.shrsi"), self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },

            // Comma operator - just return the right operand
            .Comma => {
                return rhs;
            },
        }
    }

    /// Lower unary expressions
    pub fn lowerUnary(self: *const ExpressionLowerer, unary: *const lib.ast.Expressions.UnaryNode) c.MlirValue {
        const operand = self.lowerExpression(unary.operand);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, 256);

        switch (unary.operator) {
            .Minus => {
                // Unary minus: -x
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), self.fileLoc(unary.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                    // Subtract from zero: 0 - x = -x
                    self.createConstant(0, unary.span),
                    operand,
                }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Bang => {
                // Logical NOT: !x
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), self.fileLoc(unary.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                    operand,
                    // XOR with 1: x ^ 1 = !x (for boolean values)
                    self.createConstant(1, unary.span),
                }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .BitNot => {
                // Bitwise NOT: ~x
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), self.fileLoc(unary.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                    operand,
                    // XOR with -1: x ^ (-1) = ~x
                    self.createConstant(-1, unary.span),
                }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
        }
    }

    /// Create a constant value
    pub fn createConstant(self: *const ExpressionLowerer, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
        const ty = c.mlirIntegerTypeGet(self.ctx, 256);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
        const attr = c.mlirIntegerAttrGet(ty, @intCast(value));
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        const attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Create arithmetic addition operation (arith.addi)
    pub fn createAddI(self: *const ExpressionLowerer, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        const result_type = c.mlirValueGetType(lhs);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), self.fileLoc(span));

        // Add operands
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, operands.ptr);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add overflow flags attribute
        const overflow_attr = c.mlirStringRefCreateFromCString("none");
        const overflow_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("overflowFlags"));
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(overflow_id, overflow_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Create arithmetic comparison operation (arith.cmpi)
    pub fn createCmpI(self: *const ExpressionLowerer, lhs: c.MlirValue, rhs: c.MlirValue, predicate: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        const result_type = c.mlirIntegerTypeGet(self.ctx, 1); // i1 for comparison result
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), self.fileLoc(span));

        // Add operands
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, operands.ptr);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add predicate attribute
        const pred_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(predicate.ptr));
        const pred_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Helper function to create file location
    fn fileLoc(self: *const ExpressionLowerer, span: anytype) c.MlirLocation {
        const fname = c.mlirStringRefCreateFromCString("input.ora");
        return c.mlirLocationFileLineColGet(self.ctx, fname, span.line, span.column);
    }
};
