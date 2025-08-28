const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("constants.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const ParamMap = @import("symbols.zig").ParamMap;
const StorageMap = @import("memory.zig").StorageMap;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const LocationTracker = @import("locations.zig").LocationTracker;

/// Expression lowering system for converting Ora expressions to MLIR operations
pub const ExpressionLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*const LocalVarMap,
    locations: LocationTracker,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*const LocalVarMap, locations: LocationTracker) ExpressionLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .locations = locations,
        };
    }

    /// Main dispatch function for lowering expressions
    pub fn lowerExpression(self: *const ExpressionLowerer, expr: *const lib.ast.Expressions.ExprNode) c.MlirValue {
        return switch (expr.*) {
            .Literal => |lit| self.lowerLiteral(&lit),
            .Binary => |bin| self.lowerBinary(&bin),
            .Unary => |unary| self.lowerUnary(&unary),
            .Identifier => |ident| self.lowerIdentifier(&ident),
            .Call => |call| self.lowerCall(&call),
            else => {
                // For other expression types, return a default value
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                // Use a default location since we can't access span directly from union
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), c.mlirLocationUnknownGet(self.ctx));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                const attr = c.mlirIntegerAttrGet(ty, 0);
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
        };
    }

    /// Lower literal expressions
    pub fn lowerLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.LiteralExpr) c.MlirValue {
        return switch (literal.*) {
            .Integer => |int| blk_int: {
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(int.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse the string value to an integer
                const parsed: i64 = std.fmt.parseInt(i64, int.value, 0) catch 0;
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_int c.mlirOperationGetResult(op, 0);
            },
            .Bool => |bool_lit| blk_bool: {
                const ty = c.mlirIntegerTypeGet(self.ctx, 1);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(bool_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                const default_value: i64 = if (bool_lit.value) 1 else 0;
                const attr = c.mlirIntegerAttrGet(ty, default_value);
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_bool c.mlirOperationGetResult(op, 0);
            },
            .String => |string_lit| blk_string: {
                // For now, create a placeholder constant for strings
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(string_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                const attr = c.mlirIntegerAttrGet(ty, 0); // Placeholder value
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_string c.mlirOperationGetResult(op, 0);
            },
            .Address => |addr_lit| blk_address: {
                // Parse address as hex and create integer constant
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(addr_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse hex address (remove 0x prefix if present)
                const addr_str = if (std.mem.startsWith(u8, addr_lit.value, "0x"))
                    addr_lit.value[2..]
                else
                    addr_lit.value;
                const parsed: i64 = std.fmt.parseInt(i64, addr_str, 16) catch 0;
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_address c.mlirOperationGetResult(op, 0);
            },
            .Hex => |hex_lit| blk_hex: {
                // Parse hex literal and create integer constant
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(hex_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse hex value (remove 0x prefix if present)
                const hex_str = if (std.mem.startsWith(u8, hex_lit.value, "0x"))
                    hex_lit.value[2..]
                else
                    hex_lit.value;
                const parsed: i64 = std.fmt.parseInt(i64, hex_str, 16) catch 0;
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_hex c.mlirOperationGetResult(op, 0);
            },
            .Binary => |bin_lit| blk_binary: {
                // Parse binary literal and create integer constant
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(bin_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse binary value (remove 0b prefix if present)
                const bin_str = if (std.mem.startsWith(u8, bin_lit.value, "0b"))
                    bin_lit.value[2..]
                else
                    bin_lit.value;
                const parsed: i64 = std.fmt.parseInt(i64, bin_str, 2) catch 0;
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                break :blk_binary c.mlirOperationGetResult(op, 0);
            },
        };
    }

    /// Lower binary expressions
    pub fn lowerBinary(self: *const ExpressionLowerer, bin: *const lib.ast.Expressions.BinaryExpr) c.MlirValue {
        const lhs = self.lowerExpression(bin.lhs);
        const rhs = self.lowerExpression(bin.rhs);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

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
                var attrs = [_]c.MlirNamedAttribute{
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
                var attrs = [_]c.MlirNamedAttribute{
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
                var attrs = [_]c.MlirNamedAttribute{
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
                var attrs = [_]c.MlirNamedAttribute{
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
                var attrs = [_]c.MlirNamedAttribute{
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
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(predicate_id, uge_attr_value),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },

            // Logical operators
            .And => {
                // Logical AND operation
                const left_val = self.lowerExpression(bin.lhs);
                const right_val = self.lowerExpression(bin.rhs);

                // For now, create a placeholder for logical AND
                // TODO: Implement proper logical AND operation
                _ = right_val; // Use the parameter to avoid warning
                return left_val;
            },
            .Or => {
                // Logical OR operation
                const left_val = self.lowerExpression(bin.lhs);
                const right_val = self.lowerExpression(bin.rhs);

                // For now, create a placeholder for logical OR
                // TODO: Implement proper logical OR operation
                _ = right_val; // Use the parameter to avoid warning
                return left_val;
            },
            .BitwiseXor => {
                // Bitwise XOR operation
                const left_val = self.lowerExpression(bin.lhs);

                // For now, create a placeholder for bitwise XOR
                // TODO: Implement proper bitwise XOR operation
                return left_val;
            },

            // Bitwise shift operators
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
            .BitwiseAnd => {
                // Bitwise AND operation
                // For now, create a placeholder for bitwise AND
                // TODO: Implement proper bitwise AND operation
                return lhs;
            },
            .BitwiseOr => {
                // Bitwise OR operation
                // For now, create a placeholder for bitwise OR
                // TODO: Implement proper bitwise OR operation
                return lhs;
            },
            .Comma => {
                // Comma operator - evaluate left, then right, return right
                // For now, create a placeholder for comma operator
                // TODO: Implement proper comma operator handling
                return rhs;
            },
        }
    }

    /// Lower unary expressions
    pub fn lowerUnary(self: *const ExpressionLowerer, unary: *const lib.ast.Expressions.UnaryExpr) c.MlirValue {
        const operand = self.lowerExpression(unary.operand);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        switch (unary.operator) {
            .Bang => {
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), self.fileLoc(unary.span));
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&operand));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            .Minus => {
                // Unary minus operation
                // For now, create a placeholder for unary minus
                // TODO: Implement proper unary minus operation
                return operand;
            },
            .BitNot => {
                // Bitwise NOT operation
                // For now, create a placeholder for bitwise NOT
                // TODO: Implement proper bitwise NOT operation
                return operand;
            },
        }
    }

    /// Lower identifier expressions
    pub fn lowerIdentifier(self: *const ExpressionLowerer, identifier: *const lib.ast.Expressions.IdentifierExpr) c.MlirValue {
        // First check if this is a function parameter
        if (self.param_map) |pm| {
            if (pm.getParamIndex(identifier.name)) |param_index| {
                // This is a function parameter - get the actual block argument
                if (pm.getBlockArgument(identifier.name)) |block_arg| {
                    std.debug.print("DEBUG: Function parameter {s} at index {d} - using block argument\n", .{ identifier.name, param_index });
                    return block_arg;
                } else {
                    // Fallback to dummy value if block argument not found
                    std.debug.print("DEBUG: Function parameter {s} at index {d} - block argument not found, using dummy value\n", .{ identifier.name, param_index });
                    const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
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
            }
        }

        // Check if this is a local variable
        if (self.local_var_map) |lvm| {
            if (lvm.hasLocalVar(identifier.name)) {
                // This is a local variable - return the stored value directly
                std.debug.print("DEBUG: Loading local variable: {s}\n", .{identifier.name});
                return lvm.getLocalVar(identifier.name).?;
            }
        }

        // Check if we have a storage map and if this variable exists in storage
        var is_storage_variable = false;
        if (self.storage_map) |sm| {
            if (sm.hasStorageVariable(identifier.name)) {
                is_storage_variable = true;
                // Ensure the variable exists in storage (create if needed)
                // TODO: Fix const qualifier issue - getOrCreateAddress expects mutable pointer
            }
        }

        if (is_storage_variable) {
            // This is a storage variable - use ora.sload
            std.debug.print("DEBUG: Loading storage variable: {s}\n", .{identifier.name});

            // Create a memory manager to use the storage load operation
            const memory_manager = @import("memory.zig").MemoryManager.init(self.ctx);
            // TODO: Get the actual type from the storage map instead of hardcoding
            const result_type = if (std.mem.eql(u8, identifier.name, "status"))
                c.mlirIntegerTypeGet(self.ctx, 1) // i1 for boolean
            else
                c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // i256 for integers
            const load_op = memory_manager.createStorageLoad(identifier.name, result_type, self.fileLoc(identifier.span));
            c.mlirBlockAppendOwnedOperation(self.block, load_op);
            return c.mlirOperationGetResult(load_op, 0);
        } else {
            // This is a local variable - load from the allocated memory
            std.debug.print("DEBUG: Loading local variable: {s}\n", .{identifier.name});

            // Get the local variable reference from our map
            if (self.local_var_map) |lvm| {
                if (lvm.getLocalVar(identifier.name)) |local_var_ref| {
                    // Load the value from the allocated memory
                    var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.load"), self.fileLoc(identifier.span));

                    // Add the local variable reference as operand
                    c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&local_var_ref));

                    // Add the result type (the type of the stored value)
                    const var_type = c.mlirValueGetType(local_var_ref);
                    const memref_type = c.mlirShapedTypeGetElementType(var_type);
                    c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&memref_type));

                    const load_op = c.mlirOperationCreate(&load_state);
                    c.mlirBlockAppendOwnedOperation(self.block, load_op);
                    return c.mlirOperationGetResult(load_op, 0);
                }
            }

            // If we can't find the local variable, this is an error
            std.debug.print("ERROR: Local variable not found: {s}\n", .{identifier.name});
            // For now, return a dummy value to avoid crashes
            return c.mlirBlockGetArgument(self.block, 0);
        }
    }

    /// Lower function call expressions
    pub fn lowerCall(self: *const ExpressionLowerer, call: *const lib.ast.Expressions.CallExpr) c.MlirValue {
        var args = std.ArrayList(c.MlirValue).init(std.heap.page_allocator);
        defer args.deinit();

        for (call.arguments) |arg| {
            const arg_value = self.lowerExpression(arg);
            args.append(arg_value) catch @panic("Failed to append argument");
        }

        // For now, assume the callee is an identifier (function name)
        switch (call.callee.*) {
            .Identifier => |ident| {
                // Create a function call operation
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // Default to i256 for now

                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.call"), self.fileLoc(call.span));
                c.mlirOperationStateAddOperands(&state, @intCast(args.items.len), args.items.ptr);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                // Add the callee name as a string attribute
                var callee_buffer: [256]u8 = undefined;
                for (0..ident.name.len) |i| {
                    callee_buffer[i] = ident.name[i];
                }
                callee_buffer[ident.name.len] = 0; // null-terminate
                const callee_str = c.mlirStringRefCreateFromCString(&callee_buffer[0]);
                const callee_attr = c.mlirStringAttrGet(self.ctx, callee_str);
                const callee_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("callee"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(callee_id, callee_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
                return c.mlirOperationGetResult(op, 0);
            },
            else => {
                // For now, panic on complex callee expressions
                @panic("Complex callee expressions not yet supported");
            },
        }
    }

    /// Create a constant value
    pub fn createConstant(self: *const ExpressionLowerer, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
        const attr = c.mlirIntegerAttrGet(ty, value);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Get file location for an expression
    pub fn fileLoc(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return @import("locations.zig").LocationTracker.createFileLocationFromSpan(&self.locations, span);
    }
};
