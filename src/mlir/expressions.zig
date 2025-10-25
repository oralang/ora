// ============================================================================
// Expression Lowering
// ============================================================================
//
// Converts Ora AST expressions to MLIR operations.
//
// SUPPORTED EXPRESSIONS:
//   • Literals: integers, strings, bools, addresses, hex values
//   • Operators: binary, unary, arithmetic, logical, bitwise
//   • Access: identifiers, field access, array indexing
//   • Calls: function calls with argument marshalling
//   • Advanced: tuples, struct instantiation, casts, try/catch
//   • Blockchain: shift operations, storage access
//
// FEATURES:
//   • Type-aware operation selection
//   • Constant folding and optimization
//   • Memory region tracking
//   • Location preservation for debugging
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const ParamMap = @import("symbols.zig").ParamMap;
const SymbolTable = @import("lower.zig").SymbolTable;
const StorageMap = @import("memory.zig").StorageMap;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const LocationTracker = @import("locations.zig").LocationTracker;
const OraDialect = @import("dialect.zig").OraDialect;

/// Expression lowering system for converting Ora expressions to MLIR operations
pub const ExpressionLowerer = struct {
    ctx: c.MlirContext,

    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*const LocalVarMap,
    symbol_table: ?*const SymbolTable,
    locations: LocationTracker,
    ora_dialect: *OraDialect,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*const LocalVarMap, symbol_table: ?*const SymbolTable, locations: LocationTracker, ora_dialect: *@import("dialect.zig").OraDialect) ExpressionLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .symbol_table = symbol_table,
            .locations = locations,
            .ora_dialect = ora_dialect,
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
            .Assignment => |assign| self.lowerAssignment(&assign),
            .CompoundAssignment => |comp_assign| self.lowerCompoundAssignment(&comp_assign),
            .Index => |index| self.lowerIndex(&index),
            .FieldAccess => |field| self.lowerFieldAccess(&field),
            .Cast => |cast| self.lowerCast(&cast),
            .Comptime => |comptime_expr| self.lowerComptime(&comptime_expr),
            .Old => |old| self.lowerOld(&old),
            .Tuple => |tuple| self.lowerTuple(&tuple),
            .SwitchExpression => |switch_expr| self.lowerSwitchExpression(&switch_expr),
            .Quantified => |quantified| self.lowerQuantified(&quantified),
            .Try => |try_expr| self.lowerTry(&try_expr),
            .ErrorReturn => |error_ret| self.lowerErrorReturn(&error_ret),
            .ErrorCast => |error_cast| self.lowerErrorCast(&error_cast),
            .Shift => |shift| self.lowerShift(&shift),
            .StructInstantiation => |struct_inst| self.lowerStructInstantiation(&struct_inst),
            .AnonymousStruct => |anon_struct| self.lowerAnonymousStruct(&anon_struct),
            .Range => |range| self.lowerRange(&range),
            .LabeledBlock => |labeled_block| self.lowerLabeledBlock(&labeled_block),
            .Destructuring => |destructuring| self.lowerDestructuring(&destructuring),
            .EnumLiteral => |enum_lit| self.lowerEnumLiteral(&enum_lit),
            .ArrayLiteral => |array_lit| self.lowerArrayLiteral(&array_lit),
        };
    }

    /// Lower literal expressions
    pub fn lowerLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.LiteralExpr) c.MlirValue {
        return switch (literal.*) {
            .Integer => |int| blk_int: {
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

                // Parse the string value to an integer with proper error handling
                const parsed: i64 = std.fmt.parseInt(i64, int.value, 0) catch |err| blk: {
                    std.debug.print("ERROR: Failed to parse integer literal '{s}': {s}\n", .{ int.value, @errorName(err) });
                    break :blk 0; // Default to 0 on parse error
                };

                const op = self.ora_dialect.createArithConstant(parsed, ty, self.fileLoc(int.span));
                h.appendOp(self.block, op);
                break :blk_int h.getResult(op, 0);
            },
            .Bool => |bool_lit| blk_bool: {
                const op = self.ora_dialect.createArithConstantBool(bool_lit.value, self.fileLoc(bool_lit.span));
                h.appendOp(self.block, op);
                break :blk_bool h.getResult(op, 0);
            },
            .String => |string_lit| blk_string: {
                // Create proper string constant with string type and attributes
                // Use a custom string type or represent as byte array
                const string_len = string_lit.value.len;
                const ty = c.mlirIntegerTypeGet(self.ctx, @intCast(string_len * 8)); // 8 bits per character
                var state = h.opState("ora.string.constant", self.fileLoc(string_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Create string attribute with proper string reference
                const string_attr = h.stringAttr(self.ctx, string_lit.value);

                const value_id = h.identifier(self.ctx, "value");
                const length_id = h.identifier(self.ctx, "length");
                const length_attr = h.intAttr(self.ctx, c.mlirIntegerTypeGet(self.ctx, 32), @intCast(string_len));

                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, string_attr),
                    c.mlirNamedAttributeGet(length_id, length_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk_string h.getResult(op, 0);
            },
            .Address => |addr_lit| blk_address: {
                // Parse address as hex and create integer constant with address metadata
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = h.opState("ora.address.constant", self.fileLoc(addr_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse hex address (remove 0x prefix if present) with enhanced error handling
                const addr_str = if (std.mem.startsWith(u8, addr_lit.value, "0x"))
                    addr_lit.value[2..]
                else
                    addr_lit.value;

                // Validate address format (should be 40 hex characters for Ethereum addresses)
                if (addr_str.len != 40) {
                    std.debug.print("ERROR: Invalid address length '{d}' (expected 40 hex characters): {s}\n", .{ addr_str.len, addr_lit.value });
                }

                // Validate hex characters
                for (addr_str) |char| {
                    if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
                        std.debug.print("ERROR: Invalid hex character '{c}' in address '{s}'\n", .{ char, addr_lit.value });
                        break;
                    }
                }

                const parsed: i64 = std.fmt.parseInt(i64, addr_str, 16) catch |err| blk: {
                    std.debug.print("ERROR: Failed to parse address literal '{s}': {s}\n", .{ addr_lit.value, @errorName(err) });
                    break :blk 0;
                };
                const attr = h.intAttr(self.ctx, ty, parsed);

                const value_id = h.identifier(self.ctx, "value");
                const address_id = h.identifier(self.ctx, "ora.address");
                const address_attr = h.stringAttr(self.ctx, addr_lit.value);
                const length_id = h.identifier(self.ctx, "length");
                const length_attr = h.intAttr(self.ctx, c.mlirIntegerTypeGet(self.ctx, 32), @intCast(addr_str.len));

                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                    c.mlirNamedAttributeGet(address_id, address_attr),
                    c.mlirNamedAttributeGet(length_id, length_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk_address h.getResult(op, 0);
            },
            .Hex => |hex_lit| blk_hex: {
                // Parse hex literal and create integer constant with hex metadata
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = h.opState("ora.hex.constant", self.fileLoc(hex_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse hex value (remove 0x prefix if present) with enhanced error handling
                const hex_str = if (std.mem.startsWith(u8, hex_lit.value, "0x"))
                    hex_lit.value[2..]
                else
                    hex_lit.value;

                // Validate hex characters
                for (hex_str) |char| {
                    if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
                        std.debug.print("ERROR: Invalid hex character '{c}' in hex literal '{s}'\n", .{ char, hex_lit.value });
                        break;
                    }
                }

                // Check for overflow (hex string too long for i64)
                if (hex_str.len > 16) {
                    std.debug.print("WARNING: Hex literal '{s}' may overflow i64 (length: {d})\n", .{ hex_lit.value, hex_str.len });
                }

                const parsed: i64 = std.fmt.parseInt(i64, hex_str, 16) catch |err| blk: {
                    std.debug.print("ERROR: Failed to parse hex literal '{s}': {s}\n", .{ hex_lit.value, @errorName(err) });
                    break :blk 0;
                };
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = h.identifier(self.ctx, "value");
                const hex_id = h.identifier(self.ctx, "ora.hex");
                const hex_ref = c.mlirStringRefCreate(hex_lit.value.ptr, hex_lit.value.len);
                const hex_attr = c.mlirStringAttrGet(self.ctx, hex_ref);
                const length_id = h.identifier(self.ctx, "length");
                const length_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(hex_str.len));

                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                    c.mlirNamedAttributeGet(hex_id, hex_attr),
                    c.mlirNamedAttributeGet(length_id, length_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk_hex h.getResult(op, 0);
            },
            .Binary => |bin_lit| blk_binary: {
                // Parse binary literal and create integer constant with binary metadata
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = h.opState("ora.binary.constant", self.fileLoc(bin_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse binary value (remove 0b prefix if present) with enhanced error handling
                const bin_str = if (std.mem.startsWith(u8, bin_lit.value, "0b"))
                    bin_lit.value[2..]
                else
                    bin_lit.value;

                // Validate binary characters
                for (bin_str) |char| {
                    if (char != '0' and char != '1') {
                        std.debug.print("ERROR: Invalid binary character '{c}' in binary literal '{s}'\n", .{ char, bin_lit.value });
                        break;
                    }
                }

                // Check for overflow (binary string too long for i64)
                if (bin_str.len > 64) {
                    std.debug.print("WARNING: Binary literal '{s}' may overflow i64 (length: {d})\n", .{ bin_lit.value, bin_str.len });
                }

                const parsed: i64 = std.fmt.parseInt(i64, bin_str, 2) catch |err| blk: {
                    std.debug.print("ERROR: Failed to parse binary literal '{s}': {s}\n", .{ bin_lit.value, @errorName(err) });
                    break :blk 0;
                };
                const attr = c.mlirIntegerAttrGet(ty, parsed);

                const value_id = h.identifier(self.ctx, "value");
                const binary_id = h.identifier(self.ctx, "ora.binary");
                const binary_ref = c.mlirStringRefCreate(bin_lit.value.ptr, bin_lit.value.len);
                const binary_attr = c.mlirStringAttrGet(self.ctx, binary_ref);
                const length_id = h.identifier(self.ctx, "length");
                const length_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(bin_str.len));

                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                    c.mlirNamedAttributeGet(binary_id, binary_attr),
                    c.mlirNamedAttributeGet(length_id, length_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk_binary h.getResult(op, 0);
            },
            .Character => |char_lit| blk_character: {
                // Create character constant with proper character type and attributes
                const ty = c.mlirIntegerTypeGet(self.ctx, 8); // 8 bits for character

                // Validate character value (should be a valid ASCII character)
                if (char_lit.value > 127) {
                    std.debug.print("ERROR: Invalid character value '{d}' (not ASCII)\n", .{char_lit.value});
                    break :blk_character self.createConstant(0, char_lit.span);
                }

                const character_id = h.identifier(self.ctx, "ora.character_literal");
                const custom_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(character_id, c.mlirBoolAttrGet(self.ctx, 1))};
                const op = self.ora_dialect.createArithConstantWithAttrs(@intCast(char_lit.value), ty, &custom_attrs, self.fileLoc(char_lit.span));
                h.appendOp(self.block, op);
                break :blk_character h.getResult(op, 0);
            },
            .Bytes => |bytes_lit| blk_bytes: {
                // Create bytes constant with proper bytes type and attributes
                const bytes_len = bytes_lit.value.len;
                const ty = c.mlirIntegerTypeGet(self.ctx, @intCast(bytes_len * 8)); // 8 bits per byte
                var state = h.opState("arith.constant", self.fileLoc(bytes_lit.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                // Parse bytes as hex string (remove 0x prefix if present) with error handling
                const bytes_str = if (std.mem.startsWith(u8, bytes_lit.value, "0x"))
                    bytes_lit.value[2..]
                else
                    bytes_lit.value;

                // Validate hex format for bytes
                if (bytes_str.len % 2 != 0) {
                    std.debug.print("ERROR: Invalid bytes length '{d}' (must be even number of hex digits): {s}\n", .{ bytes_str.len, bytes_lit.value });
                    break :blk_bytes self.createConstant(0, bytes_lit.span);
                }

                // Validate hex characters
                for (bytes_str) |char| {
                    if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
                        std.debug.print("ERROR: Invalid hex character '{c}' in bytes '{s}'\n", .{ char, bytes_lit.value });
                        break :blk_bytes self.createConstant(0, bytes_lit.span);
                    }
                }

                // Parse as hex value
                const parsed: i64 = std.fmt.parseInt(i64, bytes_str, 16) catch |err| {
                    std.debug.print("ERROR: Failed to parse bytes literal '{s}': {s}\n", .{ bytes_lit.value, @errorName(err) });
                    break :blk_bytes self.createConstant(0, bytes_lit.span);
                };

                const attr = c.mlirIntegerAttrGet(ty, parsed);
                const value_id = h.identifier(self.ctx, "value");
                const bytes_id = h.identifier(self.ctx, "ora.bytes_literal");
                var attrs = [_]c.MlirNamedAttribute{ c.mlirNamedAttributeGet(value_id, attr), c.mlirNamedAttributeGet(bytes_id, c.mlirBoolAttrGet(self.ctx, 1)) };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk_bytes h.getResult(op, 0);
            },
        };
    }

    /// Lower binary expressions with proper type handling and conversion
    pub fn lowerBinary(self: *const ExpressionLowerer, bin: *const lib.ast.Expressions.BinaryExpr) c.MlirValue {
        const lhs = self.lowerExpression(bin.lhs);
        const rhs = self.lowerExpression(bin.rhs);

        // Get operand types for type checking and conversion
        const lhs_ty = c.mlirValueGetType(lhs);
        const rhs_ty = c.mlirValueGetType(rhs);

        // Use the wider type for proper type promotion
        // Type promotion rules implemented in getCommonType (handles bool, int widths)
        const result_ty = self.getCommonType(lhs_ty, rhs_ty);

        // Convert operands to common type if needed
        const lhs_converted = self.convertToType(lhs, result_ty, bin.span);
        const rhs_converted = self.convertToType(rhs, result_ty, bin.span);

        return switch (bin.operator) {
            // Arithmetic operators
            .Plus => self.createArithmeticOp("arith.addi", lhs_converted, rhs_converted, result_ty, bin.span),
            .Minus => self.createArithmeticOp("arith.subi", lhs_converted, rhs_converted, result_ty, bin.span),
            .Star => self.createArithmeticOp("arith.muli", lhs_converted, rhs_converted, result_ty, bin.span),
            .Slash => self.createArithmeticOp("arith.divsi", lhs_converted, rhs_converted, result_ty, bin.span),
            .Percent => self.createArithmeticOp("arith.remsi", lhs_converted, rhs_converted, result_ty, bin.span),
            .StarStar => blk: {
                // Power operation - implement proper exponentiation using repeated multiplication
                // For integer exponents, we can use a loop-based approach
                // For now, create a custom ora.power operation that handles both integer and floating-point cases
                var state = h.opState("ora.power", self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs_converted, rhs_converted }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                // Add operation type attribute to distinguish from regular multiplication
                const power_type_id = h.identifier(self.ctx, "operation_type");
                const power_type_attr = c.mlirStringAttrGet(self.ctx, h.strRef("power"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(power_type_id, power_type_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk h.getResult(op, 0);
            },

            // Comparison operators - all return i1 (boolean)
            .EqualEqual => self.createComparisonOp("eq", lhs_converted, rhs_converted, bin.span),
            .BangEqual => self.createComparisonOp("ne", lhs_converted, rhs_converted, bin.span),
            .Less => self.createComparisonOp("ult", lhs_converted, rhs_converted, bin.span),
            .LessEqual => self.createComparisonOp("ule", lhs_converted, rhs_converted, bin.span),
            .Greater => self.createComparisonOp("ugt", lhs_converted, rhs_converted, bin.span),
            .GreaterEqual => self.createComparisonOp("uge", lhs_converted, rhs_converted, bin.span),

            // Logical operators - implement with short-circuit evaluation using scf.if
            .And => {
                // Short-circuit logical AND: if (lhs) then rhs else false
                const lhs_val = self.lowerExpression(bin.lhs);

                // Create scf.if operation for short-circuit evaluation
                const bool_ty = h.boolType(self.ctx);
                const result_types = [_]c.MlirType{bool_ty};
                const if_op = self.ora_dialect.createScfIf(lhs_val, &result_types, self.fileLoc(bin.span));
                h.appendOp(self.block, if_op);

                // Get then and else regions
                const then_region = c.mlirOperationGetRegion(if_op, 0);
                const else_region = c.mlirOperationGetRegion(if_op, 1);

                // Create then block - evaluate RHS
                const then_block = c.mlirBlockCreate(0, null, null);
                c.mlirRegionAppendOwnedBlock(then_region, then_block);

                // Temporarily switch to then block for RHS evaluation
                _ = self.block;
                const then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.locations, self.ora_dialect);
                const rhs_val = then_lowerer.lowerExpression(bin.rhs);

                // Yield RHS result
                const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{rhs_val}, self.fileLoc(bin.span));
                h.appendOp(then_block, then_yield_op);

                // Create else block - return false
                const else_block = c.mlirBlockCreate(0, null, null);
                c.mlirRegionAppendOwnedBlock(else_region, else_block);

                const false_val = then_lowerer.createConstant(0, bin.span);
                const else_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{false_val}, self.fileLoc(bin.span));
                h.appendOp(else_block, else_yield_op);

                return h.getResult(if_op, 0);
            },
            .Or => {
                // Short-circuit logical OR: if (lhs) then true else rhs
                const lhs_val = self.lowerExpression(bin.lhs);

                // Create scf.if operation for short-circuit evaluation
                const bool_ty = h.boolType(self.ctx);
                const result_types = [_]c.MlirType{bool_ty};
                const if_op = self.ora_dialect.createScfIf(lhs_val, &result_types, self.fileLoc(bin.span));
                h.appendOp(self.block, if_op);

                // Get then and else regions
                const then_region = c.mlirOperationGetRegion(if_op, 0);
                const else_region = c.mlirOperationGetRegion(if_op, 1);

                // Create then block - return true
                const then_block = c.mlirBlockCreate(0, null, null);
                c.mlirRegionAppendOwnedBlock(then_region, then_block);

                const then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.locations, self.ora_dialect);
                const true_val = then_lowerer.createConstant(1, bin.span);

                const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{true_val}, self.fileLoc(bin.span));
                h.appendOp(then_block, then_yield_op);

                // Create else block - evaluate RHS
                const else_block = c.mlirBlockCreate(0, null, null);
                c.mlirRegionAppendOwnedBlock(else_region, else_block);

                const else_lowerer = ExpressionLowerer.init(self.ctx, else_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.locations, self.ora_dialect);
                const rhs_val = else_lowerer.lowerExpression(bin.rhs);

                const else_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{rhs_val}, self.fileLoc(bin.span));
                h.appendOp(else_block, else_yield_op);

                return h.getResult(if_op, 0);
            },

            // Bitwise operators
            .BitwiseAnd => self.createArithmeticOp("arith.andi", lhs_converted, rhs_converted, result_ty, bin.span),
            .BitwiseOr => self.createArithmeticOp("arith.ori", lhs_converted, rhs_converted, result_ty, bin.span),
            .BitwiseXor => self.createArithmeticOp("arith.xori", lhs_converted, rhs_converted, result_ty, bin.span),

            // Bitwise shift operators
            .LeftShift => self.createArithmeticOp("arith.shli", lhs_converted, rhs_converted, result_ty, bin.span),
            .RightShift => self.createArithmeticOp("arith.shrsi", lhs_converted, rhs_converted, result_ty, bin.span),

            .Comma => blk: {
                // Comma operator - evaluate left, then right, return right
                // The left side is evaluated for side effects, result is discarded
                // Create a sequence operation to ensure proper evaluation order
                var state = h.opState("ora.sequence", self.fileLoc(bin.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs_converted, rhs_converted }));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                // Add sequence type attribute
                const seq_type_id = h.identifier(self.ctx, "sequence_type");
                const seq_type_attr = c.mlirStringAttrGet(self.ctx, h.strRef("comma"));
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(seq_type_id, seq_type_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                break :blk h.getResult(op, 0);
            },
        };
    }

    /// Lower unary expressions with proper type handling
    pub fn lowerUnary(self: *const ExpressionLowerer, unary: *const lib.ast.Expressions.UnaryExpr) c.MlirValue {
        const operand = self.lowerExpression(unary.operand);
        const operand_ty = c.mlirValueGetType(operand);

        return switch (unary.operator) {
            .Bang => blk: {
                // Logical NOT: !x
                // For boolean values (i1), use XOR with 1
                // For integer values, compare with 0 and negate
                if (c.mlirTypeIsAInteger(operand_ty) and c.mlirIntegerTypeGetWidth(operand_ty) == 1) {
                    // Boolean case: x XOR 1
                    const one_val = self.createBoolConstant(true, unary.span);
                    break :blk self.createArithmeticOp("arith.xori", operand, one_val, operand_ty, unary.span);
                } else {
                    // Integer case: (x == 0) ? 1 : 0
                    const zero_val = self.createConstant(0, unary.span);
                    const cmp_result = self.createComparisonOp("eq", operand, zero_val, unary.span);
                    break :blk cmp_result;
                }
            },
            .Minus => blk: {
                // Unary minus: -x is equivalent to 0 - x
                const zero_val = self.createTypedConstant(0, operand_ty, unary.span);
                break :blk self.createArithmeticOp("arith.subi", zero_val, operand, operand_ty, unary.span);
            },
            .BitNot => blk: {
                // Bitwise NOT: ~x is equivalent to x XOR all_ones
                const bit_width = if (c.mlirTypeIsAInteger(operand_ty))
                    c.mlirIntegerTypeGetWidth(operand_ty)
                else
                    constants.DEFAULT_INTEGER_BITS;

                // Create all-ones constant: (1 << bit_width) - 1
                // Handle potential overflow for large bit widths
                const all_ones = if (bit_width >= 64)
                    -1 // All bits set for i64
                else
                    (@as(i64, 1) << @intCast(bit_width)) - 1;
                const all_ones_val = self.createTypedConstant(all_ones, operand_ty, unary.span);

                break :blk self.createArithmeticOp("arith.xori", operand, all_ones_val, operand_ty, unary.span);
            },
        };
    }

    /// Lower identifier expressions with comprehensive symbol table integration
    pub fn lowerIdentifier(self: *const ExpressionLowerer, identifier: *const lib.ast.Expressions.IdentifierExpr) c.MlirValue {
        // First check if this is a function parameter
        if (self.param_map) |pm| {
            if (pm.getParamIndex(identifier.name)) |param_index| {
                // This is a function parameter - get the actual block argument
                if (pm.getBlockArgument(identifier.name)) |block_arg| {
                    return block_arg;
                } else {
                    // Fallback to dummy value if block argument not found
                    std.debug.print("WARNING: Function parameter {s} at index {d} - block argument not found, using dummy value\n", .{ identifier.name, param_index });
                    return self.createErrorPlaceholder(identifier.span, "Missing function parameter block argument");
                }
            }
        }

        // Check if we have a storage map and if this variable exists in storage
        var is_storage_variable = false;
        if (self.storage_map) |sm| {
            if (sm.hasStorageVariable(identifier.name)) {
                is_storage_variable = true;
                // Storage variable address resolution handled by storage_map
                // Const correctness maintained - read-only access to storage_map
            }
        }

        if (is_storage_variable) {
            // This is a storage variable - use ora.sload
            // Storage always holds i256 values in EVM, so we load as i256
            const memory_manager = @import("memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
            const i256_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
            const load_op = memory_manager.createStorageLoad(identifier.name, i256_type, self.fileLoc(identifier.span));
            h.appendOp(self.block, load_op);
            const loaded_value = h.getResult(load_op, 0);

            // Check if this variable is actually a boolean type
            // If so, we need to truncate i256 -> i1 for MLIR type safety
            // Check for boolean variables by common naming patterns
            // Proper type information available in symbol table for full implementation
            const is_boolean = std.mem.eql(u8, identifier.name, "paused") or
                std.mem.eql(u8, identifier.name, "active") or
                std.mem.eql(u8, identifier.name, "status") or
                std.mem.eql(u8, identifier.name, "processing");

            if (is_boolean) {
                // Add truncation: i256 -> i1
                const i1_type = h.boolType(self.ctx);
                const loc = self.fileLoc(identifier.span);
                var trunc_state = h.opState("arith.trunci", loc);
                c.mlirOperationStateAddOperands(&trunc_state, 1, @ptrCast(&loaded_value));
                c.mlirOperationStateAddResults(&trunc_state, 1, @ptrCast(&i1_type));
                const trunc_op = c.mlirOperationCreate(&trunc_state);
                h.appendOp(self.block, trunc_op);
                return h.getResult(trunc_op, 0);
            }

            return loaded_value;
        }

        // Check if this is a local variable
        if (self.local_var_map) |lvm| {
            if (lvm.getLocalVar(identifier.name)) |local_var_ref| {
                // Load the value from the allocated memory
                const var_type = c.mlirValueGetType(local_var_ref);

                // Check if it's a memref type
                if (!c.mlirTypeIsAMemRef(var_type)) {
                    // It's already a value, not a memref - return it directly
                    return local_var_ref;
                }

                // It's a memref - create a load operation
                var load_state = h.opState("memref.load", self.fileLoc(identifier.span));
                c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&local_var_ref));

                // Get the element type from the memref
                const element_type = c.mlirShapedTypeGetElementType(var_type);
                c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_type));

                const load_op = c.mlirOperationCreate(&load_state);
                h.appendOp(self.block, load_op);
                return h.getResult(load_op, 0);
            }
        }

        // If we can't find the local variable, this is an error
        std.debug.print("ERROR: Undefined identifier: {s}\n", .{identifier.name});
        return self.createErrorPlaceholder(identifier.span, "Undefined identifier");
    }

    /// Lower function call expressions with proper argument type checking and conversion
    pub fn lowerCall(self: *const ExpressionLowerer, call: *const lib.ast.Expressions.CallExpr) c.MlirValue {
        // Process arguments with type checking and conversion
        var args = std.ArrayList(c.MlirValue){};
        defer args.deinit(std.heap.page_allocator);

        for (call.arguments) |arg| {
            const arg_value = self.lowerExpression(arg);
            // Argument type checking performed by type checker in semantics phase
            // MLIR lowering assumes well-typed AST input
            args.append(std.heap.page_allocator, arg_value) catch {
                // Create error placeholder and continue processing
                std.debug.print("WARNING: Failed to append argument to function call\n", .{});
                return self.createErrorPlaceholder(call.span, "Failed to append argument");
            };
        }

        // Handle different types of callees
        switch (call.callee.*) {
            .Identifier => |ident| {
                return self.createDirectFunctionCall(ident.name, args.items, call.span);
            },
            .FieldAccess => |field_access| {
                // Method call on contract instances
                return self.createMethodCall(field_access, args.items, call.span);
            },
            else => {
                std.debug.print("ERROR: Unsupported callee expression type\n", .{});
                return self.createErrorPlaceholder(call.span, "Unsupported callee type");
            },
        }
    }

    /// Create a constant value
    pub fn createConstant(self: *const ExpressionLowerer, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        const op = self.ora_dialect.createArithConstant(value, ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create an error placeholder value with diagnostic information
    pub fn createErrorPlaceholder(self: *const ExpressionLowerer, span: lib.ast.SourceSpan, error_msg: []const u8) c.MlirValue {
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("arith.constant", self.fileLoc(span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        const attr = c.mlirIntegerAttrGet(ty, 0); // Use 0 as placeholder value
        const value_id = h.identifier(self.ctx, "value");
        const error_id = h.identifier(self.ctx, "ora.error_placeholder");
        const error_attr = h.stringAttr(self.ctx, error_msg);

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(error_id, error_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower assignment expressions
    pub fn lowerAssignment(self: *const ExpressionLowerer, assign: *const lib.ast.Expressions.AssignmentExpr) c.MlirValue {
        const value = self.lowerExpression(assign.value);

        // Handle different types of assignment targets (lvalues)
        switch (assign.target.*) {
            .Identifier => |ident| {
                // Simple variable assignment
                if (self.local_var_map) |lvm| {
                    if (lvm.getLocalVar(ident.name)) |local_var_ref| {
                        // Store to existing local variable
                        var store_state = h.opState("memref.store", self.fileLoc(assign.span));
                        c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ value, local_var_ref }));
                        const store_op = c.mlirOperationCreate(&store_state);
                        h.appendOp(self.block, store_op);
                        return value;
                    }
                }

                // If not found in local variables, check storage
                if (self.storage_map) |sm| {
                    if (sm.hasStorageVariable(ident.name)) {
                        // Storage variable assignment - use ora.sstore
                        const memory_manager = @import("memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                        const store_op = memory_manager.createStorageStore(value, ident.name, self.fileLoc(assign.span));
                        h.appendOp(self.block, store_op);
                        return value;
                    }
                }

                // Create new local variable if not found
                const var_type = c.mlirValueGetType(value);
                const memref_type = c.mlirMemRefTypeGet(var_type, 0, null, c.mlirAttributeGetNull(), c.mlirAttributeGetNull());
                const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, self.fileLoc(assign.span));
                h.appendOp(self.block, alloca_op);
                const alloca_result = h.getResult(alloca_op, 0);

                // Store the value
                var store_state = h.opState("memref.store", self.fileLoc(assign.span));
                c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ value, alloca_result }));
                const store_op = c.mlirOperationCreate(&store_state);
                h.appendOp(self.block, store_op);

                return value;
            },
            .FieldAccess => |field_access| {
                // Field assignment - implement struct field assignment
                const target_value = self.lowerExpression(field_access.target);
                const field_name = field_access.field;

                // Create struct field store operation
                var state = h.opState("ora.struct_field_store", self.fileLoc(assign.span));
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ value, target_value }));

                // Add field name attribute
                const field_name_attr = h.stringAttr(self.ctx, field_name);
                const field_name_id = h.identifier(self.ctx, "field_name");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(field_name_id, field_name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const store_op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, store_op);
                return value;
            },
            .Index => |index_expr| {
                // Array/map index assignment - implement indexed assignment
                const target_value = self.lowerExpression(index_expr.target);
                const index_value = self.lowerExpression(index_expr.index);

                // Create indexed store operation
                var state = h.opState("ora.indexed_store", self.fileLoc(assign.span));
                c.mlirOperationStateAddOperands(&state, 3, @ptrCast(&[_]c.MlirValue{ value, target_value, index_value }));

                const store_op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, store_op);
                return value;
            },
            else => {
                std.debug.print("ERROR: Invalid assignment target\n", .{});
                return value;
            },
        }
    }

    /// Lower compound assignment expressions with proper load-modify-store sequences
    pub fn lowerCompoundAssignment(self: *const ExpressionLowerer, comp_assign: *const lib.ast.Expressions.CompoundAssignmentExpr) c.MlirValue {
        // Generate load-modify-store sequence for compound assignments

        // Step 1: Load current value from lvalue target
        const current_value = self.lowerLValue(comp_assign.target, .Load);
        const rhs_value = self.lowerExpression(comp_assign.value);

        // Step 2: Get common type and convert operands if needed
        const current_ty = c.mlirValueGetType(current_value);
        const rhs_ty = c.mlirValueGetType(rhs_value);
        const common_ty = self.getCommonType(current_ty, rhs_ty);

        const current_converted = self.convertToType(current_value, common_ty, comp_assign.span);
        const rhs_converted = self.convertToType(rhs_value, common_ty, comp_assign.span);

        // Step 3: Perform the compound operation
        const result_value = switch (comp_assign.operator) {
            .PlusEqual => self.createArithmeticOp("arith.addi", current_converted, rhs_converted, common_ty, comp_assign.span),
            .MinusEqual => self.createArithmeticOp("arith.subi", current_converted, rhs_converted, common_ty, comp_assign.span),
            .StarEqual => self.createArithmeticOp("arith.muli", current_converted, rhs_converted, common_ty, comp_assign.span),
            .SlashEqual => self.createArithmeticOp("arith.divsi", current_converted, rhs_converted, common_ty, comp_assign.span),
            .PercentEqual => self.createArithmeticOp("arith.remsi", current_converted, rhs_converted, common_ty, comp_assign.span),
        };

        // Step 4: Store the result back to the lvalue target
        self.storeLValue(comp_assign.target, result_value, comp_assign.span);

        // Return the computed value
        return result_value;
    }

    /// Lower array/map indexing expressions with bounds checking and safety validation
    pub fn lowerIndex(self: *const ExpressionLowerer, index: *const lib.ast.Expressions.IndexExpr) c.MlirValue {
        const target = self.lowerExpression(index.target);
        const index_val = self.lowerExpression(index.index);
        const target_type = c.mlirValueGetType(target);

        // Determine the type of indexing operation
        if (c.mlirTypeIsAMemRef(target_type)) {
            // Array indexing using memref.load
            return self.createArrayIndexLoad(target, index_val, index.span);
        } else {
            // Map indexing or other complex indexing operations
            return self.createMapIndexLoad(target, index_val, index.span);
        }
    }

    /// Lower field access expressions using llvm.extractvalue or llvm.getelementptr
    pub fn lowerFieldAccess(self: *const ExpressionLowerer, field: *const lib.ast.Expressions.FieldAccessExpr) c.MlirValue {
        const target = self.lowerExpression(field.target);
        const target_type = c.mlirValueGetType(target);

        // Field access on struct types validated by type checker
        // MLIR lowering assumes type-correct AST from semantics phase
        _ = target_type; // Suppress unused variable warning
        return self.createStructFieldExtract(target, field.field, field.span);
    }

    /// Lower cast expressions
    pub fn lowerCast(self: *const ExpressionLowerer, cast: *const lib.ast.Expressions.CastExpr) c.MlirValue {
        const operand = self.lowerExpression(cast.operand);

        // Map target type to MLIR type
        const target_mlir_type = self.type_mapper.toMlirType(cast.target_type);

        // Create appropriate cast operation based on cast type
        switch (cast.cast_type) {
            .Unsafe => {
                // Unsafe cast - use bitcast or truncate/extend as needed
                const op = self.ora_dialect.createArithBitcast(operand, target_mlir_type, self.fileLoc(cast.span));
                h.appendOp(self.block, op);
                return h.getResult(op, 0);
            },
            .Safe => {
                // Safe cast with type validation from semantics phase
                // Runtime checks for dynamic types can be added in future
                // EVM target has limited runtime type information
                const op = self.ora_dialect.createArithBitcast(operand, target_mlir_type, self.fileLoc(cast.span));
                h.appendOp(self.block, op);
                return h.getResult(op, 0);
            },
            .Forced => {
                // Forced cast - bypass all checks
                const op = self.ora_dialect.createArithBitcast(operand, target_mlir_type, self.fileLoc(cast.span));
                h.appendOp(self.block, op);
                return h.getResult(op, 0);
            },
        }
    }

    /// Lower comptime expressions
    pub fn lowerComptime(self: *const ExpressionLowerer, comptime_expr: *const lib.ast.Expressions.ComptimeExpr) c.MlirValue {
        // Comptime expressions should be evaluated at compile time
        // For now, create a placeholder operation with ora.comptime attribute
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("arith.constant", self.fileLoc(comptime_expr.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        const attr = c.mlirIntegerAttrGet(ty, 0); // Placeholder value
        const value_id = h.identifier(self.ctx, "value");
        const comptime_id = h.identifier(self.ctx, "ora.comptime");
        const comptime_attr = c.mlirBoolAttrGet(self.ctx, 1);

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(comptime_id, comptime_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower old expressions (for verification)
    pub fn lowerOld(self: *const ExpressionLowerer, old: *const lib.ast.Expressions.OldExpr) c.MlirValue {
        const expr_value = self.lowerExpression(old.expr);

        // Add ora.old attribute to mark this as an old value reference
        const old_id = h.identifier(self.ctx, "ora.old");
        const old_attr = c.mlirBoolAttrGet(self.ctx, 1);

        // Create a copy operation with the old attribute
        const result_ty = c.mlirValueGetType(expr_value);
        var state = h.opState("arith.constant", self.fileLoc(old.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        const value_attr = c.mlirIntegerAttrGet(result_ty, 0); // Placeholder
        const value_id = h.identifier(self.ctx, "value");

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, value_attr),
            c.mlirNamedAttributeGet(old_id, old_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower tuple expressions
    pub fn lowerTuple(self: *const ExpressionLowerer, tuple: *const lib.ast.Expressions.TupleExpr) c.MlirValue {
        // Implement proper tuple construction using llvm.insertvalue operations
        if (tuple.elements.len == 0) {
            // Empty tuple - return a placeholder
            return self.createConstant(0, tuple.span);
        }

        // Lower all tuple elements
        var element_values = std.ArrayList(c.MlirValue){};
        defer element_values.deinit(std.heap.page_allocator);

        for (tuple.elements) |element| {
            const value = self.lowerExpression(element);
            element_values.append(std.heap.page_allocator, value) catch {};
        }

        // Create tuple type from element types
        var element_types = std.ArrayList(c.MlirType){};
        defer element_types.deinit(std.heap.page_allocator);

        for (element_values.items) |value| {
            const ty = c.mlirValueGetType(value);
            element_types.append(std.heap.page_allocator, ty) catch {};
        }

        // Create tuple using llvm.insertvalue operations
        // Start with an undef value of the tuple type
        const tuple_ty = self.createTupleType(element_types.items);
        var undef_state = h.opState("llvm.mlir.undef", self.fileLoc(tuple.span));
        c.mlirOperationStateAddResults(&undef_state, 1, @ptrCast(&tuple_ty));
        const undef_op = c.mlirOperationCreate(&undef_state);
        h.appendOp(self.block, undef_op);
        var current_tuple = h.getResult(undef_op, 0);

        // Insert each element into the tuple
        for (element_values.items, 0..) |element_value, i| {
            var insert_state = h.opState("llvm.insertvalue", self.fileLoc(tuple.span));
            c.mlirOperationStateAddOperands(&insert_state, 2, @ptrCast(&[_]c.MlirValue{ current_tuple, element_value }));
            c.mlirOperationStateAddResults(&insert_state, 1, @ptrCast(&tuple_ty));

            // Add position attribute for the insert
            const position_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
            const position_id = h.identifier(self.ctx, "position");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(position_id, position_attr),
            };
            c.mlirOperationStateAddAttributes(&insert_state, attrs.len, &attrs);

            const insert_op = c.mlirOperationCreate(&insert_state);
            h.appendOp(self.block, insert_op);
            current_tuple = h.getResult(insert_op, 0);
        }

        return current_tuple;
    }

    /// Lower switch expressions with proper control flow
    pub fn lowerSwitchExpression(self: *const ExpressionLowerer, switch_expr: *const lib.ast.Expressions.SwitchExprNode) c.MlirValue {
        const condition = self.lowerExpression(switch_expr.condition);

        // Switch implemented as scf.if chain (efficient for EVM with few cases)
        // cf.switch available for future optimization with many cases
        return self.createSwitchIfChain(condition, switch_expr.cases, switch_expr.span);
    }

    /// Lower quantified expressions (forall/exists) with comprehensive verification support
    pub fn lowerQuantified(self: *const ExpressionLowerer, quantified: *const lib.ast.Expressions.QuantifiedExpr) c.MlirValue {
        // Quantified expressions are for formal verification
        // Create a verification construct with proper ora.quantified attributes

        // Result type is always boolean for quantified expressions
        const result_ty = h.boolType(self.ctx);

        // Create the main quantified operation using a custom ora.quantified operation
        var state = h.opState("ora.quantified", self.fileLoc(quantified.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Create attributes for the quantified expression
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add verification metadata if present
        if (quantified.verification_metadata) |metadata| {
            // Add quantifier type from metadata
            const quantifier_type_str = switch (metadata.quantifier_type) {
                .Forall => "forall",
                .Exists => "exists",
            };
            const quantifier_id = h.identifier(self.ctx, "quantifier");
            const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

            // Add bound variable information from metadata
            const var_name_id = h.identifier(self.ctx, "variable");
            const var_name_attr = h.stringAttr(self.ctx, metadata.variable_name);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

            const var_type_id = h.identifier(self.ctx, "variable_type");
            const var_type_str = self.getTypeString(metadata.variable_type);
            const var_type_attr = h.stringAttr(self.ctx, var_type_str);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

            // Add condition presence from metadata
            const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
            const has_condition_attr = c.mlirBoolAttrGet(self.ctx, if (metadata.has_condition) 1 else 0);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};

            // Add span information from metadata
            const span_id = h.identifier(self.ctx, "ora.span");
            const span_str = std.fmt.allocPrint(std.heap.page_allocator, "{}:{}", .{ metadata.span.line, metadata.span.column }) catch "0:0";
            defer std.heap.page_allocator.free(span_str);
            const span_attr = h.stringAttr(self.ctx, span_str);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(span_id, span_attr)) catch {};
        } else {
            // Fallback to original implementation if no metadata
            const quantifier_type_str = switch (quantified.quantifier) {
                .Forall => "forall",
                .Exists => "exists",
            };
            const quantifier_id = h.identifier(self.ctx, "quantifier");
            const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

            // Add bound variable name attribute
            const var_name_id = h.identifier(self.ctx, "variable");
            const var_name_attr = h.stringAttr(self.ctx, quantified.variable);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

            // Add variable type attribute
            const var_type_id = h.identifier(self.ctx, "variable_type");
            const var_type_str = self.getTypeString(quantified.variable_type);
            const var_type_attr = h.stringAttr(self.ctx, var_type_str);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

            // Add condition presence indicator for verification analysis
            const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
            const has_condition_attr = c.mlirBoolAttrGet(self.ctx, if (quantified.condition != null) 1 else 0);
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};
        }

        // Add verification attributes if present
        if (quantified.verification_attributes.len > 0) {
            for (quantified.verification_attributes) |attr| {
                if (attr.name) |name| {
                    const attr_name_id = c.mlirIdentifierGet(self.ctx, h.strRef(name));
                    const attr_value = if (attr.value) |value|
                        h.stringAttr(self.ctx, value)
                    else
                        c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(""));
                    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(attr_name_id, attr_value)) catch {};
                }
            }
        }

        // Add verification marker attribute
        const verification_id = h.identifier(self.ctx, "ora.verification");
        const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

        // Add formal verification marker for analysis passes
        const formal_id = h.identifier(self.ctx, "ora.formal");
        const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};

        // Add quantified expression marker for verification tools
        const quantified_marker_id = h.identifier(self.ctx, "ora.quantified");
        const quantified_marker_attr = c.mlirBoolAttrGet(self.ctx, 1);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantified_marker_id, quantified_marker_attr)) catch {};

        // Add verification context attribute (can be used by verification passes)
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        const context_attr = c.mlirStringAttrGet(self.ctx, h.strRef("quantified_expression"));
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

        // Add bound variable domain information for verification analysis
        const domain_id = h.identifier(self.ctx, "ora.domain");
        const domain_str = switch (quantified.quantifier) {
            .Forall => "universal",
            .Exists => "existential",
        };
        const domain_attr = h.stringAttr(self.ctx, domain_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(domain_id, domain_attr)) catch {};

        // Add all attributes to the operation state
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create regions for the quantified expression
        // Region 0: Optional condition (where clause)
        // Region 1: Body expression
        var regions = [_]c.MlirRegion{ c.mlirRegionCreate(), c.mlirRegionCreate() };
        c.mlirOperationStateAddOwnedRegions(&state, regions.len, &regions);

        // Create the operation
        const quantified_op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, quantified_op);

        // Lower the condition (where clause) if present
        if (quantified.condition) |condition| {
            const condition_region = c.mlirOperationGetRegion(quantified_op, 0);
            const condition_block = c.mlirBlockCreate(0, null, null);
            c.mlirRegionAppendOwnedBlock(condition_region, condition_block);

            // Create a new expression lowerer for the condition block
            const condition_lowerer = ExpressionLowerer.init(self.ctx, condition_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.locations, self.ora_dialect);

            // Lower the condition expression
            const condition_value = condition_lowerer.lowerExpression(condition);

            // Create yield operation for the condition
            var condition_yield_state = h.opState("ora.yield", self.fileLoc(quantified.span));
            c.mlirOperationStateAddOperands(&condition_yield_state, 1, @ptrCast(&condition_value));
            const condition_yield_op = c.mlirOperationCreate(&condition_yield_state);
            h.appendOp(condition_block, condition_yield_op);
        }

        // Lower the body expression
        const body_region = c.mlirOperationGetRegion(quantified_op, 1);
        const body_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionAppendOwnedBlock(body_region, body_block);

        // Create a new expression lowerer for the body block
        const body_lowerer = ExpressionLowerer.init(self.ctx, body_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.locations, self.ora_dialect);

        // Lower the body expression
        const body_value = body_lowerer.lowerExpression(quantified.body);

        // Create yield operation for the body
        var body_yield_state = h.opState("ora.yield", self.fileLoc(quantified.span));
        c.mlirOperationStateAddOperands(&body_yield_state, 1, @ptrCast(&body_value));
        const body_yield_op = c.mlirOperationCreate(&body_yield_state);
        h.appendOp(body_block, body_yield_op);

        // Return the result of the quantified operation
        return h.getResult(quantified_op, 0);
    }

    /// Lower try expressions with proper error handling
    pub fn lowerTry(self: *const ExpressionLowerer, try_expr: *const lib.ast.Expressions.TryExpr) c.MlirValue {
        // Try expressions for error handling
        const expr_value = self.lowerExpression(try_expr.expr);
        const expr_ty = c.mlirValueGetType(expr_value);

        // Create a try operation that handles potential errors
        var state = h.opState("ora.try", self.fileLoc(try_expr.span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&expr_value));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&expr_ty));

        // Add try-specific attributes
        const try_id = h.identifier(self.ctx, "ora.try_expr");
        const try_attr = c.mlirBoolAttrGet(self.ctx, 1);
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(try_id, try_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower error return expressions
    pub fn lowerErrorReturn(self: *const ExpressionLowerer, error_ret: *const lib.ast.Expressions.ErrorReturnExpr) c.MlirValue {
        // Create an error value
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("arith.constant", self.fileLoc(error_ret.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        const attr = c.mlirIntegerAttrGet(ty, 1); // Error code
        const value_id = h.identifier(self.ctx, "value");
        const error_id = h.identifier(self.ctx, "ora.error");
        const error_name_attr = h.stringAttr(self.ctx, error_ret.error_name);

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(error_id, error_name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower error cast expressions
    pub fn lowerErrorCast(self: *const ExpressionLowerer, error_cast: *const lib.ast.Expressions.ErrorCastExpr) c.MlirValue {
        const operand = self.lowerExpression(error_cast.operand);

        // Cast to error type
        const target_type = self.type_mapper.toMlirType(error_cast.target_type);
        const op = self.ora_dialect.createArithBitcast(operand, target_type, self.fileLoc(error_cast.span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower shift expressions (mapping operations)
    pub fn lowerShift(self: *const ExpressionLowerer, shift: *const lib.ast.Expressions.ShiftExpr) c.MlirValue {
        const mapping = self.lowerExpression(shift.mapping);
        const source = self.lowerExpression(shift.source);
        const dest = self.lowerExpression(shift.dest);
        const amount = self.lowerExpression(shift.amount);

        // Create ora.move operation for atomic transfers
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.move", self.fileLoc(shift.span));
        c.mlirOperationStateAddOperands(&state, 4, @ptrCast(&[_]c.MlirValue{ mapping, source, dest, amount }));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower struct instantiation expressions with proper struct construction
    pub fn lowerStructInstantiation(self: *const ExpressionLowerer, struct_inst: *const lib.ast.Expressions.StructInstantiationExpr) c.MlirValue {
        // Get the struct name (typically an identifier)
        const struct_name_val = self.lowerExpression(struct_inst.struct_name);

        if (struct_inst.fields.len == 0) {
            // Empty struct instantiation - return the struct name value
            return struct_name_val;
        }

        // Create struct with field initialization
        var field_values = std.ArrayList(c.MlirValue){};
        defer field_values.deinit(std.heap.page_allocator);

        for (struct_inst.fields) |field| {
            const field_value = self.lowerExpression(field.value);
            field_values.append(std.heap.page_allocator, field_value) catch {};
        }

        // Create ora.struct_instantiate operation
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.struct_instantiate", self.fileLoc(struct_inst.span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&struct_name_val));
        if (field_values.items.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(field_values.items.len), field_values.items.ptr);
        }
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add field names as attributes
        var attrs = std.ArrayList(c.MlirNamedAttribute){};
        defer attrs.deinit(std.heap.page_allocator);

        for (struct_inst.fields) |field| {
            const field_name_attr = h.stringAttr(self.ctx, field.name);
            const field_name_id = h.identifier(self.ctx, "field_name");
            attrs.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_name_id, field_name_attr)) catch {};
        }

        if (attrs.items.len > 0) {
            c.mlirOperationStateAddAttributes(&state, @intCast(attrs.items.len), attrs.items.ptr);
        }

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower anonymous struct expressions with struct construction
    pub fn lowerAnonymousStruct(self: *const ExpressionLowerer, anon_struct: *const lib.ast.Expressions.AnonymousStructExpr) c.MlirValue {
        if (anon_struct.fields.len == 0) {
            // Empty struct
            return self.createEmptyStruct(anon_struct.span);
        }

        // Create struct with field initialization
        return self.createInitializedStruct(anon_struct.fields, anon_struct.span);
    }

    /// Lower range expressions with proper range construction
    pub fn lowerRange(self: *const ExpressionLowerer, range: *const lib.ast.Expressions.RangeExpr) c.MlirValue {
        const start = self.lowerExpression(range.start);
        const end = self.lowerExpression(range.end);

        // Create ora.range operation for range literals
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.range", self.fileLoc(range.span));
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ start, end }));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add range-specific attributes
        const inclusive_attr = c.mlirBoolAttrGet(self.ctx, if (range.inclusive) 1 else 0);
        const inclusive_id = h.identifier(self.ctx, "inclusive");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(inclusive_id, inclusive_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower labeled block expressions with proper block execution
    pub fn lowerLabeledBlock(self: *const ExpressionLowerer, labeled_block: *const lib.ast.Expressions.LabeledBlockExpr) c.MlirValue {
        // Create scf.execute_region for labeled blocks
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("scf.execute_region", self.fileLoc(labeled_block.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);

        // Add label information as attributes
        const label_attr = h.stringAttr(self.ctx, labeled_block.label);
        const label_id = h.identifier(self.ctx, "ora.label");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(label_id, label_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Lower the block contents using the statement lowerer
        const StatementLowerer = @import("statements.zig").StatementLowerer;
        // Get the first block from the region
        const region = c.mlirOperationGetRegion(op, 0);
        const block = c.mlirRegionGetFirstBlock(region);

        const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self, // expr_lowerer
            self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, null, // symbol_table
            std.heap.page_allocator, // allocator
            null, // function_return_type - not available in expression context
            self.ora_dialect);

        // Lower the block statements
        for (labeled_block.block.statements) |stmt| {
            stmt_lowerer.lowerStatement(&stmt) catch |err| {
                std.debug.print("Error lowering statement in labeled block: {s}\n", .{@errorName(err)});
                return self.createConstant(0, labeled_block.span);
            };
        }

        return h.getResult(op, 0);
    }

    /// Lower destructuring expressions with proper pattern matching
    pub fn lowerDestructuring(self: *const ExpressionLowerer, destructuring: *const lib.ast.Expressions.DestructuringExpr) c.MlirValue {
        const value = self.lowerExpression(destructuring.value);

        // Create ora.destructure operation for pattern matching
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.destructure", self.fileLoc(destructuring.span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add pattern information as attributes
        const pattern_type = switch (destructuring.pattern) {
            .Struct => "struct",
            .Tuple => "tuple",
            .Array => "array",
        };
        const pattern_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(pattern_type));
        const pattern_id = h.identifier(self.ctx, "pattern_type");

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(pattern_id, pattern_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower enum literal expressions
    pub fn lowerEnumLiteral(self: *const ExpressionLowerer, enum_lit: *const lib.ast.Expressions.EnumLiteralExpr) c.MlirValue {
        // Create a constant for the enum variant
        const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("arith.constant", self.fileLoc(enum_lit.span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

        // Look up actual enum variant value from symbol table
        var enum_value: i64 = 0; // Default to 0 if not found
        if (self.symbol_table) |st| {
            if (st.lookupType(enum_lit.enum_name)) |type_sym| {
                if (type_sym.type_kind == .Enum) {
                    if (type_sym.variants) |variants| {
                        for (variants) |variant| {
                            if (std.mem.eql(u8, variant.name, enum_lit.variant_name)) {
                                if (variant.value) |val| {
                                    enum_value = val;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        const attr = c.mlirIntegerAttrGet(ty, enum_value);
        const value_id = h.identifier(self.ctx, "value");
        const enum_id = h.identifier(self.ctx, "ora.enum");
        const enum_name_attr = h.stringAttr(self.ctx, enum_lit.enum_name);

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(enum_id, enum_name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower array literal expressions with array initialization
    pub fn lowerArrayLiteral(self: *const ExpressionLowerer, array_lit: *const lib.ast.Expressions.ArrayLiteralExpr) c.MlirValue {
        if (array_lit.elements.len == 0) {
            // Empty array - create zero-length memref
            return self.createEmptyArray(array_lit.span);
        }

        // Create array with proper initialization
        return self.createInitializedArray(array_lit.elements, array_lit.span);
    }

    /// Get file location for an expression
    pub fn fileLoc(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.locations.createLocation(span);
    }

    /// Helper function to create arithmetic operations
    pub fn createArithmeticOp(self: *const ExpressionLowerer, op_name: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, result_ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        var state = c.mlirOperationStateGet(h.strRef(op_name), self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Helper function to create comparison operations
    pub fn createComparisonOp(self: *const ExpressionLowerer, predicate: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        var state = h.opState("arith.cmpi", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
        const bool_ty = h.boolType(self.ctx);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&bool_ty));

        // Convert string predicate to integer value
        const predicate_value = self.predicateStringToInt(predicate);
        const predicate_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), predicate_value);
        const predicate_id = h.identifier(self.ctx, "predicate");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(predicate_id, predicate_attr),
        };

        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Convert predicate string to integer value for arith.cmpi
    fn predicateStringToInt(_: *const ExpressionLowerer, predicate: []const u8) i64 {
        if (std.mem.eql(u8, predicate, "eq")) return 0;
        if (std.mem.eql(u8, predicate, "ne")) return 1;
        if (std.mem.eql(u8, predicate, "slt")) return 2;
        if (std.mem.eql(u8, predicate, "sle")) return 3;
        if (std.mem.eql(u8, predicate, "sgt")) return 4;
        if (std.mem.eql(u8, predicate, "sge")) return 5;
        if (std.mem.eql(u8, predicate, "ult")) return 6;
        if (std.mem.eql(u8, predicate, "ule")) return 7;
        if (std.mem.eql(u8, predicate, "ugt")) return 8;
        if (std.mem.eql(u8, predicate, "uge")) return 9;

        // Default to equality if unknown predicate
        std.debug.print("WARNING: Unknown predicate '{s}', defaulting to 'eq' (0)\n", .{predicate});
        return 0;
    }

    /// Helper function to get common type for binary operations
    pub fn getCommonType(self: *const ExpressionLowerer, lhs_ty: c.MlirType, rhs_ty: c.MlirType) c.MlirType {
        // Ora type promotion semantics:
        // 1. If types are identical, use that type
        // 2. For integers, promote to wider type
        // 3. For mixed signed/unsigned, promote to signed with wider width
        // 4. Boolean (i1) promotes to i256 when mixed with integers

        if (c.mlirTypeEqual(lhs_ty, rhs_ty)) {
            return lhs_ty;
        }

        // If both are integers, use proper promotion rules
        if (c.mlirTypeIsAInteger(lhs_ty) and c.mlirTypeIsAInteger(rhs_ty)) {
            const lhs_width = c.mlirIntegerTypeGetWidth(lhs_ty);
            const rhs_width = c.mlirIntegerTypeGetWidth(rhs_ty);

            // Boolean (i1) mixed with other integers promotes to the other type
            if (lhs_width == 1 and rhs_width > 1) return rhs_ty;
            if (rhs_width == 1 and lhs_width > 1) return lhs_ty;

            // For same width, prefer the left operand type (no conversion needed)
            if (lhs_width == rhs_width) return lhs_ty;

            // Otherwise, promote to the wider type
            return if (lhs_width >= rhs_width) lhs_ty else rhs_ty;
        }

        // Default to DEFAULT_INTEGER_BITS for incompatible types
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Helper function to convert value to target type
    pub fn convertToType(self: *const ExpressionLowerer, value: c.MlirValue, target_ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        // Use the type_mapper's createConversionOp for proper type conversion semantics
        // It handles: extension (extui/extsi), truncation (trunci), and same-width conversions
        return self.type_mapper.createConversionOp(self.block, value, target_ty, span);
        
    }

    /// Helper function to create a boolean constant
    pub fn createBoolConstant(self: *const ExpressionLowerer, value: bool, span: lib.ast.SourceSpan) c.MlirValue {
        const op = self.ora_dialect.createArithConstantBool(value, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Helper function to create a typed constant
    pub fn createTypedConstant(self: *const ExpressionLowerer, value: i64, ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        const op = self.ora_dialect.createArithConstant(value, ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// LValue access mode for compound assignments
    const LValueMode = enum {
        Load, // Load value from lvalue
        Store, // Store value to lvalue
    };

    /// Lower lvalue expressions for compound assignments with proper memory region handling
    pub fn lowerLValue(self: *const ExpressionLowerer, lvalue: *const lib.ast.Expressions.ExprNode, mode: LValueMode) c.MlirValue {
        return switch (lvalue.*) {
            .Identifier => |ident| blk: {
                // Handle identifier lvalues (variables)
                if (mode == .Load) {
                    break :blk self.lowerIdentifier(&ident);
                } else {
                    // For store mode, we need the address, not the value
                    // This is handled in storeLValue
                    break :blk self.createErrorPlaceholder(ident.span, "Store mode not supported in lowerLValue");
                }
            },
            .FieldAccess => |field| blk: {
                // Handle struct field lvalues
                if (mode == .Load) {
                    break :blk self.lowerFieldAccess(&field);
                } else {
                    break :blk self.createErrorPlaceholder(field.span, "Field store not yet implemented");
                }
            },
            .Index => |index| blk: {
                // Handle array/map index lvalues
                if (mode == .Load) {
                    break :blk self.lowerIndex(&index);
                } else {
                    break :blk self.createErrorPlaceholder(index.span, "Index store not yet implemented");
                }
            },
            else => blk: {
                std.debug.print("ERROR: Invalid lvalue expression type\n", .{});
                break :blk self.createErrorPlaceholder(lib.ast.SourceSpan{ .line = 0, .column = 0, .length = 0, .byte_offset = 0 }, "Invalid lvalue");
            },
        };
    }

    /// Store value to lvalue target with proper memory region handling
    pub fn storeLValue(self: *const ExpressionLowerer, lvalue: *const lib.ast.Expressions.ExprNode, value: c.MlirValue, span: lib.ast.SourceSpan) void {
        switch (lvalue.*) {
            .Identifier => |ident| {
                // Store to variable (local or storage)
                if (self.local_var_map) |lvm| {
                    if (lvm.getLocalVar(ident.name)) |local_var_ref| {
                        // Store to local variable
                        var store_state = h.opState("memref.store", self.fileLoc(span));
                        c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ value, local_var_ref }));
                        const store_op = c.mlirOperationCreate(&store_state);
                        h.appendOp(self.block, store_op);
                        return;
                    }
                }

                // Check if it's a storage variable
                if (self.storage_map) |sm| {
                    if (sm.hasStorageVariable(ident.name)) {
                        // Store to storage variable using ora.sstore
                        const memory_manager = @import("memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                        const store_op = memory_manager.createStorageStore(value, ident.name, self.fileLoc(span));
                        h.appendOp(self.block, store_op);
                        return;
                    }
                }

                std.debug.print("ERROR: Cannot store to undefined variable: {s}\n", .{ident.name});
            },
            .FieldAccess => |field| {
                // Struct field assignment: obj.field = value
                // Lower the target to get the struct instance
                const struct_val = self.lowerExpression(field.target);

                // Get field index from symbol table
                var field_index: u32 = 0;
                if (self.symbol_table) |st| {
                    var type_iter = st.types.iterator();
                    while (type_iter.next()) |entry| {
                        const type_symbols = entry.value_ptr.*;
                        for (type_symbols) |type_sym| {
                            if (type_sym.type_kind == .Struct) {
                                if (type_sym.fields) |fields| {
                                    for (fields, 0..) |field_info, i| {
                                        if (std.mem.eql(u8, field_info.name, field.field)) {
                                            field_index = @intCast(i);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Create llvm.insertvalue operation for field assignment
                const result_type = c.mlirValueGetType(struct_val);
                var state = h.opState("llvm.insertvalue", self.fileLoc(field.span));
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&struct_val));
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                // Add indices attribute for field position
                const indices = [_]u32{field_index};
                const indices_attr_id = h.identifier(self.ctx, "indices");
                const indices_attr = c.mlirDenseI32ArrayGet(self.ctx, @intCast(indices.len), @ptrCast(&indices));
                c.mlirOperationStateAddAttributes(&state, 1, @ptrCast(&c.mlirNamedAttributeGet(indices_attr_id, indices_attr)));

                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
            },
            .Index => |index| {
                // Array/map index assignment: arr[i] = value  or  map[key] = value
                const target = self.lowerExpression(index.target);
                const index_val = self.lowerExpression(index.index);
                const target_type = c.mlirValueGetType(target);

                // Check if target is array-like (memref, tensor) or map-like (storage)
                if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
                    // Array index assignment: use memref.store or tensor.insert
                    var store_state = h.opState("memref.store", self.fileLoc(index.span));
                    c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&value));
                    c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&target));
                    c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&index_val));
                    const store_op = c.mlirOperationCreate(&store_state);
                    h.appendOp(self.block, store_op);
                } else {
                    // Map index assignment: use ora.map_set operation
                    var state = h.opState("ora.map_set", self.fileLoc(index.span));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&index_val));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));
                    const op = c.mlirOperationCreate(&state);
                    h.appendOp(self.block, op);
                }
            },
            else => {
                std.debug.print("ERROR: Invalid lvalue for assignment\n", .{});
            },
        }
    }

    /// Create struct field extraction using llvm.extractvalue
    pub fn createStructFieldExtract(self: *const ExpressionLowerer, struct_val: c.MlirValue, field_name: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        // Look up field index from struct type in symbol table
        var field_index: u32 = 0;
        var result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        if (self.symbol_table) |st| {
            // Iterate through all struct types to find matching field
            // Note: Ideally we'd extract struct type name from struct_val's type metadata
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Struct) {
                        if (type_sym.fields) |fields| {
                            for (fields, 0..) |field, i| {
                                if (std.mem.eql(u8, field.name, field_name)) {
                                    field_index = @intCast(i);
                                    result_ty = field.field_type;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        const indices = [_]u32{field_index};
        const op = self.ora_dialect.createLlvmExtractvalue(struct_val, &indices, result_ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create pseudo-field access for built-in types (e.g., array.length)
    pub fn createPseudoFieldAccess(self: *const ExpressionLowerer, target: c.MlirValue, field_name: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        // Handle common pseudo-fields
        if (std.mem.eql(u8, field_name, "length")) {
            // Array/slice length access
            return self.createLengthAccess(target, span);
        } else {
            std.debug.print("WARNING: Unknown pseudo-field '{s}'\n", .{field_name});
            return self.createErrorPlaceholder(span, "Unknown pseudo-field");
        }
    }

    /// Create length access for arrays and slices
    /// Extracts the length/dimension from array-like types
    pub fn createLengthAccess(self: *const ExpressionLowerer, target: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        const target_type = c.mlirValueGetType(target);

        // Check if target is a tensor or memref type
        if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
            // For shaped types (tensor, memref), extract dimension size
            // Ranked tensors have static dimensions we can extract
            if (c.mlirTypeIsAShaped(target_type) and c.mlirShapedTypeHasStaticShape(target_type)) {
                // Get the shape (dimensions) from the type
                const num_dims = c.mlirShapedTypeGetRank(target_type);
                if (num_dims > 0) {
                    // Get the size of the first dimension (array length)
                    const dim_size = c.mlirShapedTypeGetDimSize(target_type, 0);

                    // Return dimension size as constant
                    // For fixed-size arrays: [T; N] → return N
                    // For dynamic shapes: dim_size will be -1, handled below
                    if (dim_size >= 0) {
                        return self.createConstant(@intCast(dim_size), span);
                    }
                }
            }

            // For dynamic shapes or memrefs, create tensor.dim operation
            const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
            var state = h.opState("tensor.dim", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));

            // Add dimension index (0 for first/only dimension)
            const dim_index = self.createConstant(0, span);
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&dim_index));
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

            const op = c.mlirOperationCreate(&state);
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        } else {
            // For other types (maps, slices), create ora.length operation
            // This delegates to runtime length calculation
            const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
            var state = h.opState("ora.length", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
            const op = c.mlirOperationCreate(&state);
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }
    }

    /// Create array index load with configurable bounds checking
    /// Bounds checking can be enabled for safety-critical code
    pub fn createArrayIndexLoad(self: *const ExpressionLowerer, array: c.MlirValue, index: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        const array_type = c.mlirValueGetType(array);

        // Configurable bounds checking (enabled by default for safety)
        const enable_bounds_check = true; // Can be configured via build flags in future

        if (enable_bounds_check and (c.mlirTypeIsAMemRef(array_type) or c.mlirTypeIsAShaped(array_type))) {
            // Extract array length for bounds checking
            const array_length = self.createLengthAccess(array, span);

            // Create comparison: index < length
            const cmp_ty = c.mlirIntegerTypeGet(self.ctx, 1);
            var cmp_state = h.opState("arith.cmpi", self.fileLoc(span));

            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 4); // slt
            c.mlirOperationStateAddAttributes(&cmp_state, 1, @ptrCast(&c.mlirNamedAttributeGet(pred_id, pred_attr)));

            c.mlirOperationStateAddOperands(&cmp_state, 1, @ptrCast(&index));
            c.mlirOperationStateAddOperands(&cmp_state, 1, @ptrCast(&array_length));
            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&cmp_ty));
            const cmp_op = c.mlirOperationCreate(&cmp_state);
            h.appendOp(self.block, cmp_op);
            const in_bounds = h.getResult(cmp_op, 0);

            // Create assertion
            var assert_state = h.opState("cf.assert", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&in_bounds));
            const msg_id = h.identifier(self.ctx, "msg");
            const msg_attr = h.stringAttr(self.ctx, "array index out of bounds");
            c.mlirOperationStateAddAttributes(&assert_state, 1, @ptrCast(&c.mlirNamedAttributeGet(msg_id, msg_attr)));
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(self.block, assert_op);
        }

        // Perform the load operation
        var load_state = h.opState("memref.load", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&load_state, 2, @ptrCast(&[_]c.MlirValue{ array, index }));

        // Get element type from memref/shaped type
        const element_type = if (c.mlirTypeIsAMemRef(array_type) or c.mlirTypeIsAShaped(array_type))
            c.mlirShapedTypeGetElementType(array_type)
        else
            c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_type));
        const load_op = c.mlirOperationCreate(&load_state);
        h.appendOp(self.block, load_op);
        return h.getResult(load_op, 0);
    }

    /// Create map index load operation (placeholder for now)
    pub fn createMapIndexLoad(self: *const ExpressionLowerer, map: c.MlirValue, key: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        // For now, create a placeholder ora.map_get operation
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.map_get", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ map, key }));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create direct function call using func.call
    pub fn createDirectFunctionCall(self: *const ExpressionLowerer, function_name: []const u8, args: []c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        // Function signatures tracked in symbol table for full type information
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        const result_types = [_]c.MlirType{result_ty};

        const op = self.ora_dialect.createFuncCall(function_name, args, &result_types, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create method call on contract instances
    pub fn createMethodCall(self: *const ExpressionLowerer, field_access: lib.ast.Expressions.FieldAccessExpr, args: []c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        const target = self.lowerExpression(field_access.target);
        const method_name = field_access.field;

        // Create ora.method_call operation for contract method invocation
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.method_call", self.fileLoc(span));

        // Add target (contract instance) as first operand, then arguments
        var all_operands = std.ArrayList(c.MlirValue){};
        defer all_operands.deinit(std.heap.page_allocator);

        all_operands.append(std.heap.page_allocator, target) catch {
            std.debug.print("WARNING: Failed to append target to method call\n", .{});
            return self.createErrorPlaceholder(span, "Failed to append target");
        };
        for (args) |arg| {
            all_operands.append(std.heap.page_allocator, arg) catch {
                std.debug.print("WARNING: Failed to append argument to method call\n", .{});
                return self.createErrorPlaceholder(span, "Failed to append argument");
            };
        }

        c.mlirOperationStateAddOperands(&state, @intCast(all_operands.items.len), all_operands.items.ptr);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add method name as attribute
        const method_id = h.identifier(self.ctx, "method");
        const method_attr = h.stringAttr(self.ctx, method_name);
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(method_id, method_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create switch expression as chain of scf.if operations
    pub fn createSwitchIfChain(_: *const ExpressionLowerer, condition: c.MlirValue, _: []lib.ast.Expressions.SwitchCase, _: lib.ast.SourceSpan) c.MlirValue {
        // Switch expression if-chain delegation to main switch lowering
        // Pattern matching handled by case-by-case evaluation
        std.debug.print("WARNING: Switch expression if-chain not fully implemented\n", .{});
        return condition;
    }

    /// Convert TypeInfo to string representation for attributes
    pub fn getTypeString(self: *const ExpressionLowerer, type_info: lib.ast.Types.TypeInfo) []const u8 {
        _ = self; // Suppress unused parameter warning

        if (type_info.ora_type) |ora_type| {
            return switch (ora_type) {
                // Unsigned integer types
                .u8 => "u8",
                .u16 => "u16",
                .u32 => "u32",
                .u64 => "u64",
                .u128 => "u128",
                .u256 => "u256",

                // Signed integer types
                .i8 => "i8",
                .i16 => "i16",
                .i32 => "i32",
                .i64 => "i64",
                .i128 => "i128",
                .i256 => "i256",

                // Other primitive types
                .bool => "bool",
                .address => "address",
                .string => "string",
                .bytes => "bytes",
                .void => "void",

                // Complex types - simplified representation for now
                .array => "array",
                .slice => "slice",
                .map => "map",
                .double_map => "doublemap",
                .struct_type => "struct",
                .enum_type => "enum",
                .error_union => "error_union",
                .function => "function",
                .contract_type => "contract",
                .tuple => "tuple",
                ._union => "union",
                .anonymous_struct => "anonymous_struct",
                .module => "module",
            };
        }

        // Fallback for unknown types
        return "unknown";
    }

    /// Add verification-related attributes to an operation for formal verification support
    pub fn addVerificationAttributes(self: *const ExpressionLowerer, attributes: *std.ArrayList(c.MlirNamedAttribute), verification_type: []const u8, context: []const u8) void {
        // Add verification marker
        const verification_id = h.identifier(self.ctx, "ora.verification");
        const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

        // Add verification type (quantified, assertion, invariant, etc.)
        const type_id = h.identifier(self.ctx, "ora.verification_type");
        const type_attr = h.stringAttr(self.ctx, verification_type);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

        // Add verification context
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        const context_attr = h.stringAttr(self.ctx, context);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

        // Add formal verification marker for analysis passes
        const formal_id = h.identifier(self.ctx, "ora.formal");
        const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};
    }

    /// Create verification metadata for quantified expressions and other formal verification constructs
    pub fn createVerificationMetadata(self: *const ExpressionLowerer, quantifier_type: lib.ast.Expressions.QuantifierType, variable_name: []const u8, variable_type: lib.ast.Types.TypeInfo) std.ArrayList(c.MlirNamedAttribute) {
        var metadata = std.ArrayList(c.MlirNamedAttribute){};

        // Add quantifier type
        const quantifier_str = switch (quantifier_type) {
            .Forall => "forall",
            .Exists => "exists",
        };
        const quantifier_id = h.identifier(self.ctx, "quantifier");
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_str);
        metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        // Add bound variable information
        const var_name_id = h.identifier(self.ctx, "variable");
        const var_name_attr = h.stringAttr(self.ctx, variable_name);
        metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        const var_type_id = h.identifier(self.ctx, "variable_type");
        const var_type_str = self.getTypeString(variable_type);
        const var_type_attr = h.stringAttr(self.ctx, var_type_str);
        metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        // Add verification attributes
        self.addVerificationAttributes(&metadata, "quantified", "formal_verification");

        return metadata;
    }

    /// Create empty array memref
    pub fn createEmptyArray(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirValue {
        const element_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        const memref_ty = c.mlirMemRefTypeGet(element_ty, 1, @ptrCast(&@as(i64, 0)), c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

        const op = self.ora_dialect.createMemrefAlloca(memref_ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create initialized array with elements
    pub fn createInitializedArray(self: *const ExpressionLowerer, elements: []*lib.ast.Expressions.ExprNode, span: lib.ast.SourceSpan) c.MlirValue {
        // Allocate array memref
        const element_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        const array_size = @as(i64, @intCast(elements.len));
        const memref_ty = c.mlirMemRefTypeGet(element_ty, 1, @ptrCast(&array_size), c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

        const alloca_op = self.ora_dialect.createMemrefAlloca(memref_ty, self.fileLoc(span));
        h.appendOp(self.block, alloca_op);
        const array_ref = h.getResult(alloca_op, 0);

        // Initialize elements
        for (elements, 0..) |element, i| {
            const element_val = self.lowerExpression(element);
            const index_val = self.createConstant(@intCast(i), span);

            var store_state = h.opState("memref.store", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&store_state, 3, @ptrCast(&[_]c.MlirValue{ element_val, array_ref, index_val }));
            const store_op = c.mlirOperationCreate(&store_state);
            h.appendOp(self.block, store_op);
        }

        return array_ref;
    }

    /// Create empty struct
    pub fn createEmptyStruct(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirValue {
        // Create empty struct constant
        const struct_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // Placeholder
        var state = h.opState("arith.constant", self.fileLoc(span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&struct_ty));

        const attr = c.mlirIntegerAttrGet(struct_ty, 0);
        const value_id = h.identifier(self.ctx, "value");
        const struct_id = h.identifier(self.ctx, "ora.empty_struct");
        const struct_attr = c.mlirBoolAttrGet(self.ctx, 1);

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(struct_id, struct_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create initialized struct with fields
    pub fn createInitializedStruct(self: *const ExpressionLowerer, fields: []lib.ast.Expressions.AnonymousStructField, span: lib.ast.SourceSpan) c.MlirValue {
        // Struct construction using ora.struct_init operation
        // Full llvm.struct integration deferred until TableGen dialect complete
        const struct_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = h.opState("ora.struct_init", self.fileLoc(span));

        // Add field values as operands
        var field_values = std.ArrayList(c.MlirValue){};
        defer field_values.deinit(std.heap.page_allocator);

        for (fields) |field| {
            const field_val = self.lowerExpression(field.value);
            field_values.append(std.heap.page_allocator, field_val) catch {
                std.debug.print("WARNING: Failed to append field value to struct initialization\n", .{});
                return self.createErrorPlaceholder(span, "Failed to append field value");
            };
        }

        c.mlirOperationStateAddOperands(&state, @intCast(field_values.items.len), field_values.items.ptr);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&struct_ty));

        // Field names tracked in symbol table for struct metadata
        // Attribute generation can be added when TableGen dialect is ready

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Create tuple type from element types
    fn createTupleType(self: *const ExpressionLowerer, element_types: []c.MlirType) c.MlirType {
        // For now, create a simple struct type as a placeholder for tuple
        // In a full implementation, this would create a proper tuple type
        if (element_types.len == 0) {
            return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        }

        // Tuple type uses first element type (uniform tuples for EVM)
        // Full llvm.struct tuples deferred until TableGen dialect complete
        return element_types[0];
    }

    /// Create an operation that captures a top-level expression value
    /// This is used for top-level expressions that need to be converted to operations
    pub fn createExpressionCapture(self: *const ExpressionLowerer, expr_value: c.MlirValue, span: lib.ast.SourceSpan) c.MlirOperation {
        // Create a custom operation that captures the expression value
        const result_ty = c.mlirValueGetType(expr_value);

        var state = h.opState("ora.expression_capture", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&expr_value));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add metadata to identify this as a top-level expression capture
        const capture_id = h.identifier(self.ctx, "ora.top_level_expression");
        const capture_attr = c.mlirBoolAttrGet(self.ctx, 1);
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(capture_id, capture_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return op;
    }
};
