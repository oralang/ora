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
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const ParamMap = @import("../symbols.zig").ParamMap;
const SymbolTable = @import("../lower.zig").SymbolTable;
const StorageMap = @import("../memory.zig").StorageMap;
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const builtins = lib.semantics.builtins;
const expr_helpers = @import("helpers.zig");
const expr_literals = @import("literals.zig");
const expr_operators = @import("operators.zig");
const expr_access = @import("access.zig");
const expr_calls = @import("calls.zig");
const expr_assignments = @import("assignments.zig");
const expr_advanced = @import("advanced.zig");
const ErrorHandler = @import("../error_handling.zig").ErrorHandler;

/// Expression lowering system for converting Ora expressions to MLIR operations
pub const ExpressionLowerer = struct {
    ctx: c.MlirContext,

    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*const LocalVarMap,
    symbol_table: ?*const SymbolTable,
    builtin_registry: ?*const builtins.BuiltinRegistry,
    error_handler: ?*ErrorHandler,
    locations: LocationTracker,
    ora_dialect: *OraDialect,
    in_try_block: bool = false,
    current_function_return_type: ?c.MlirType = null,
    current_function_return_type_info: ?lib.ast.Types.TypeInfo = null,
    refinement_base_cache: ?*std.AutoHashMap(usize, c.MlirValue) = null,
    refinement_guard_cache: ?*std.AutoHashMap(u128, void) = null,
    prefer_refinement_base_cache: bool = false,
    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*const LocalVarMap, symbol_table: ?*const SymbolTable, builtin_registry: ?*const builtins.BuiltinRegistry, error_handler: ?*ErrorHandler, locations: LocationTracker, ora_dialect: *OraDialect) ExpressionLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .symbol_table = symbol_table,
            .builtin_registry = builtin_registry,
            .error_handler = error_handler,
            .locations = locations,
            .ora_dialect = ora_dialect,
            .in_try_block = false,
            .current_function_return_type = null,
            .current_function_return_type_info = null,
            .refinement_base_cache = null,
            .refinement_guard_cache = null,
            .prefer_refinement_base_cache = false,
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

    /// Extract integer value from a literal expression (for switch case patterns)
    fn extractIntegerFromLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.LiteralExpr) ?i64 {
        _ = self;
        return expr_literals.extractIntegerFromLiteral(literal);
    }

    /// Extract integer value from an expression (for switch case patterns)
    fn extractIntegerFromExpr(self: *const ExpressionLowerer, expr: *const lib.ast.Expressions.ExprNode) ?i64 {
        _ = self;
        return expr_literals.extractIntegerFromExpr(expr);
    }

    /// Lower literal expressions
    pub fn lowerLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.LiteralExpr) c.MlirValue {
        return expr_literals.lowerLiteral(self.ctx, self.block, self.type_mapper, self.ora_dialect, self.locations, literal);
    }

    /// Lower binary expressions with proper type handling and conversion
    pub fn lowerBinary(self: *const ExpressionLowerer, bin: *const lib.ast.Expressions.BinaryExpr) c.MlirValue {
        return expr_operators.lowerBinary(self, bin);
    }

    /// Lower unary expressions with proper type handling
    pub fn lowerUnary(self: *const ExpressionLowerer, unary: *const lib.ast.Expressions.UnaryExpr) c.MlirValue {
        return expr_operators.lowerUnary(self, unary);
    }

    /// Lower identifier expressions with comprehensive symbol table integration
    pub fn lowerIdentifier(self: *const ExpressionLowerer, identifier: *const lib.ast.Expressions.IdentifierExpr) c.MlirValue {
        return expr_access.lowerIdentifier(self, identifier);
    }

    /// Lower index expressions
    pub fn lowerIndex(self: *const ExpressionLowerer, index: *const lib.ast.Expressions.IndexExpr) c.MlirValue {
        return expr_access.lowerIndex(self, index);
    }

    /// Lower field access expressions
    pub fn lowerFieldAccess(self: *const ExpressionLowerer, field: *const lib.ast.Expressions.FieldAccessExpr) c.MlirValue {
        return expr_access.lowerFieldAccess(self, field);
    }

    /// Lower function call expressions with proper argument type checking and conversion
    pub fn lowerCall(self: *const ExpressionLowerer, call: *const lib.ast.Expressions.CallExpr) c.MlirValue {
        return expr_calls.lowerCall(self, call);
    }

    /// Process a normal (non-builtin) function call
    pub fn processNormalCall(self: *const ExpressionLowerer, call: *const lib.ast.Expressions.CallExpr) c.MlirValue {
        return expr_calls.processNormalCall(self, call);
    }

    /// Lower builtin function call
    pub fn lowerBuiltinCall(self: *const ExpressionLowerer, builtin_info: *const builtins.BuiltinInfo, call: *const lib.ast.Expressions.CallExpr) c.MlirValue {
        return expr_calls.lowerBuiltinCall(self, builtin_info, call);
    }

    /// Create direct function call
    pub fn createDirectFunctionCall(self: *const ExpressionLowerer, function_name: []const u8, args: []c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_calls.createDirectFunctionCall(self, function_name, args, span);
    }

    /// Create method call
    pub fn createMethodCall(self: *const ExpressionLowerer, field_access: lib.ast.Expressions.FieldAccessExpr, args: []c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_calls.createMethodCall(self, field_access, args, span);
    }

    /// Create a constant value
    pub fn createConstant(self: *const ExpressionLowerer, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_helpers.createConstant(self.ctx, self.block, self.ora_dialect, self.locations, value, span);
    }

    /// Create an error placeholder value with diagnostic information
    pub fn createErrorPlaceholder(self: *const ExpressionLowerer, span: lib.ast.SourceSpan, error_msg: []const u8) c.MlirValue {
        return expr_helpers.createErrorPlaceholder(self.ctx, self.block, self.locations, span, error_msg);
    }

    pub fn reportLoweringError(self: *const ExpressionLowerer, span: lib.ast.SourceSpan, message: []const u8, suggestion: ?[]const u8) c.MlirValue {
        if (self.error_handler) |handler| {
            handler.reportError(.InternalError, span, message, suggestion) catch {};
            return self.createErrorPlaceholder(span, message);
        }
        @panic(message);
    }

    /// Lower assignment expressions
    pub fn lowerAssignment(self: *const ExpressionLowerer, assign: *const lib.ast.Expressions.AssignmentExpr) c.MlirValue {
        return expr_assignments.lowerAssignment(self, assign);
    }

    /// Lower compound assignment expressions with proper load-modify-store sequences
    pub fn lowerCompoundAssignment(self: *const ExpressionLowerer, comp_assign: *const lib.ast.Expressions.CompoundAssignmentExpr) c.MlirValue {
        return expr_assignments.lowerCompoundAssignment(self, comp_assign);
    }

    /// Lower lvalue expression
    pub fn lowerLValue(self: *const ExpressionLowerer, lvalue: *const lib.ast.Expressions.ExprNode, mode: expr_assignments.LValueMode) c.MlirValue {
        return expr_assignments.lowerLValue(self, lvalue, mode);
    }

    /// Store value to lvalue target
    pub fn storeLValue(self: *const ExpressionLowerer, lvalue: *const lib.ast.Expressions.ExprNode, value: c.MlirValue, span: lib.ast.SourceSpan) void {
        expr_assignments.storeLValue(self, lvalue, value, span);
    }

    /// Lower builtin constant (e.g., std.constants.ZERO_ADDRESS)
    fn lowerBuiltinConstant(self: *const ExpressionLowerer, builtin_info: *const builtins.BuiltinInfo, span: lib.ast.SourceSpan) c.MlirValue {
        const ty = self.type_mapper.toMlirType(.{
            .ora_type = builtin_info.return_type,
        });

        // special handling for specific constants
        if (std.mem.eql(u8, builtin_info.full_path, "std.constants.ZERO_ADDRESS")) {
            // zero_address should return !ora.address type (not i160)
            // when used in comparisons, it will be converted to i160 via ora.addr.to.i160
            // create arith.constant as i160, then convert to !ora.address
            const i160_ty = c.oraIntegerTypeCreate(self.ctx, 160);
            const const_op = self.ora_dialect.createArithConstant(0, i160_ty, self.fileLoc(span));
            h.appendOp(self.block, const_op);
            const i160_value = h.getResult(const_op, 0);

            // convert i160 to !ora.address using ora.i160.to.addr
            const addr_op = c.oraI160ToAddrOpCreate(self.ctx, self.fileLoc(span), i160_value);
            h.appendOp(self.block, addr_op);
            return h.getResult(addr_op, 0);
        }

        if (std.mem.eql(u8, builtin_info.full_path, "std.constants.U256_MAX")) {
            // create i256 constant -1 (all 1s in two's complement = max unsigned)
            const op = self.ora_dialect.createArithConstant(-1, ty, self.fileLoc(span));
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        if (std.mem.eql(u8, builtin_info.full_path, "std.constants.U128_MAX")) {
            const op = self.ora_dialect.createArithConstant(-1, ty, self.fileLoc(span));
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        if (std.mem.eql(u8, builtin_info.full_path, "std.constants.U64_MAX")) {
            const op = self.ora_dialect.createArithConstant(-1, ty, self.fileLoc(span));
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        if (std.mem.eql(u8, builtin_info.full_path, "std.constants.U32_MAX")) {
            const op = self.ora_dialect.createArithConstant(-1, ty, self.fileLoc(span));
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        // fallback: return 0
        const op = self.ora_dialect.createArithConstant(0, ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    /// Lower cast expressions
    pub fn lowerCast(self: *const ExpressionLowerer, cast: *const lib.ast.Expressions.CastExpr) c.MlirValue {
        return expr_advanced.lowerCast(self, cast);
    }

    /// Lower comptime expressions
    pub fn lowerComptime(self: *const ExpressionLowerer, comptime_expr: *const lib.ast.Expressions.ComptimeExpr) c.MlirValue {
        return expr_advanced.lowerComptime(self, comptime_expr);
    }

    /// Lower old expressions (for verification)
    pub fn lowerOld(self: *const ExpressionLowerer, old: *const lib.ast.Expressions.OldExpr) c.MlirValue {
        return expr_advanced.lowerOld(self, old);
    }

    /// Lower tuple expressions
    pub fn lowerTuple(self: *const ExpressionLowerer, tuple: *const lib.ast.Expressions.TupleExpr) c.MlirValue {
        return expr_advanced.lowerTuple(self, tuple);
    }

    /// Create a default value for a given MLIR type
    pub fn createDefaultValueForType(self: *const ExpressionLowerer, mlir_type: c.MlirType, loc: c.MlirLocation) !c.MlirValue {
        return expr_advanced.createDefaultValueForType(self, mlir_type, loc);
    }

    /// Lower switch expressions
    pub fn lowerSwitchExpression(self: *const ExpressionLowerer, switch_expr: *const lib.ast.Expressions.SwitchExprNode) c.MlirValue {
        return expr_advanced.lowerSwitchExpression(self, switch_expr);
    }

    /// Lower quantified expressions
    pub fn lowerQuantified(self: *const ExpressionLowerer, quantified: *const lib.ast.Expressions.QuantifiedExpr) c.MlirValue {
        return expr_advanced.lowerQuantified(self, quantified);
    }

    /// Lower try expressions
    pub fn lowerTry(self: *const ExpressionLowerer, try_expr: *const lib.ast.Expressions.TryExpr) c.MlirValue {
        return expr_advanced.lowerTry(self, try_expr);
    }

    /// Lower error return expressions
    pub fn lowerErrorReturn(self: *const ExpressionLowerer, error_ret: *const lib.ast.Expressions.ErrorReturnExpr) c.MlirValue {
        return expr_advanced.lowerErrorReturn(self, error_ret);
    }

    /// Lower error cast expressions
    pub fn lowerErrorCast(self: *const ExpressionLowerer, error_cast: *const lib.ast.Expressions.ErrorCastExpr) c.MlirValue {
        return expr_advanced.lowerErrorCast(self, error_cast);
    }

    /// Lower shift expressions
    pub fn lowerShift(self: *const ExpressionLowerer, shift: *const lib.ast.Expressions.ShiftExpr) c.MlirValue {
        return expr_advanced.lowerShift(self, shift);
    }

    /// Lower struct instantiation expressions
    pub fn lowerStructInstantiation(self: *const ExpressionLowerer, struct_inst: *const lib.ast.Expressions.StructInstantiationExpr) c.MlirValue {
        return expr_advanced.lowerStructInstantiation(self, struct_inst);
    }

    /// Lower anonymous struct expressions
    pub fn lowerAnonymousStruct(self: *const ExpressionLowerer, anon_struct: *const lib.ast.Expressions.AnonymousStructExpr) c.MlirValue {
        return expr_advanced.lowerAnonymousStruct(self, anon_struct);
    }

    /// Lower range expressions
    pub fn lowerRange(self: *const ExpressionLowerer, range: *const lib.ast.Expressions.RangeExpr) c.MlirValue {
        return expr_advanced.lowerRange(self, range);
    }

    /// Lower labeled block expressions
    pub fn lowerLabeledBlock(self: *const ExpressionLowerer, labeled_block: *const lib.ast.Expressions.LabeledBlockExpr) c.MlirValue {
        return expr_advanced.lowerLabeledBlock(self, labeled_block);
    }

    /// Lower destructuring expressions
    pub fn lowerDestructuring(self: *const ExpressionLowerer, destructuring: *const lib.ast.Expressions.DestructuringExpr) c.MlirValue {
        return expr_advanced.lowerDestructuring(self, destructuring);
    }

    /// Lower enum literal expressions
    pub fn lowerEnumLiteral(self: *const ExpressionLowerer, enum_lit: *const lib.ast.Expressions.EnumLiteralExpr) c.MlirValue {
        return expr_advanced.lowerEnumLiteral(self, enum_lit);
    }

    /// Lower array literal expressions
    pub fn lowerArrayLiteral(self: *const ExpressionLowerer, array_lit: *const lib.ast.Expressions.ArrayLiteralExpr) c.MlirValue {
        return expr_advanced.lowerArrayLiteral(self, array_lit);
    }

    /// Get file location for an expression
    pub fn fileLoc(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.locations.createLocation(span);
    }

    /// Helper function to create arithmetic operations
    pub fn createArithmeticOp(self: *const ExpressionLowerer, op_name: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, _: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_helpers.createArithmeticOp(self.ctx, self.block, self.type_mapper, self.ora_dialect, self.locations, self.refinement_base_cache, op_name, lhs, rhs, span);
    }

    /// Helper function to create comparison operations
    pub fn createComparisonOp(self: *const ExpressionLowerer, predicate: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_helpers.createComparisonOp(self.ctx, self.block, self.locations, self.refinement_base_cache, predicate, lhs, rhs, span);
    }

    /// Helper function to get common type for binary operations
    pub fn getCommonType(self: *const ExpressionLowerer, lhs_ty: c.MlirType, rhs_ty: c.MlirType) c.MlirType {
        return expr_helpers.getCommonType(self.ctx, lhs_ty, rhs_ty);
    }

    /// Helper function to convert value to target type
    pub fn convertToType(self: *const ExpressionLowerer, value: c.MlirValue, target_ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        const value_type = c.oraValueGetType(value);
        const loc = self.fileLoc(span);

        // refinement -> base conversion with source span
        const refinement_base = c.oraRefinementTypeGetBaseType(value_type);
        if (refinement_base.ptr != null and c.oraTypeEqual(refinement_base, target_ty)) {
            const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
            if (convert_op.ptr != null) {
                return h.getResult(convert_op, 0);
            }
        }

        // base -> refinement conversion with source span
        const target_ref_base = c.oraRefinementTypeGetBaseType(target_ty);
        if (target_ref_base.ptr != null and c.oraTypeEqual(target_ref_base, value_type)) {
            const convert_op = c.oraBaseToRefinementOpCreate(self.ctx, loc, value, target_ty, self.block);
            if (convert_op.ptr != null) {
                return h.getResult(convert_op, 0);
            }
        }

        // refinement -> refinement conversion with source span
        if (refinement_base.ptr != null and target_ref_base.ptr != null) {
            const to_base_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, self.block);
            if (to_base_op.ptr != null) {
                const base_val = h.getResult(to_base_op, 0);
                const base_converted = self.type_mapper.createConversionOp(self.block, base_val, target_ref_base, span);
                const to_ref_op = c.oraBaseToRefinementOpCreate(self.ctx, loc, base_converted, target_ty, self.block);
                if (to_ref_op.ptr != null) {
                    return h.getResult(to_ref_op, 0);
                }
            }
        }

        return self.type_mapper.createConversionOp(self.block, value, target_ty, span);
    }

    /// Check if a value is a zero constant
    /// This is a simplified check - we check if the value type is i160 and the value is 0
    /// For more accurate detection, we'd need to trace back to the defining operation
    fn isZeroConstant(_: *const ExpressionLowerer, value: c.MlirValue) bool {
        // simplified approach: check if the value type is i160 (which zero address constants would be)
        // and assume it's zero if it's a small integer type
        // this is a heuristic - in practice, we'd need to check the actual constant value
        // for now, we'll rely on the type system to handle this correctly
        const value_ty = c.oraValueGetType(value);

        // check if it's i160 (zero address would be i160)
        if (c.oraTypeIsAInteger(value_ty)) {
            const width = c.oraIntegerTypeGetWidth(value_ty);
            // if it's i160, it could be a zero address constant
            // we'll be conservative and only treat it as zero if it's explicitly i160
            // the actual zero detection would require checking the defining operation
            // for now, return false to be safe - the comparison will work correctly anyway
            _ = width;
        }

        // for now, return false - we'll handle zero address comparisons differently
        // by checking if ZERO_ADDRESS constant is being used
        return false;
    }

    /// Helper function to create a boolean constant
    pub fn createBoolConstant(self: *const ExpressionLowerer, value: bool, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_helpers.createBoolConstant(self.ctx, self.block, self.ora_dialect, self.locations, value, span);
    }

    /// Helper function to create a typed constant
    pub fn createTypedConstant(self: *const ExpressionLowerer, value: i64, ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_helpers.createTypedConstant(self.ctx, self.block, self.ora_dialect, self.locations, value, ty, span);
    }

    /// Create struct field extraction using ora.struct_field_extract
    pub fn createStructFieldExtract(self: *const ExpressionLowerer, struct_val: c.MlirValue, field_name: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.createStructFieldExtract(self, struct_val, field_name, span);
    }

    /// Create pseudo-field access for built-in types (e.g., array.length)
    pub fn createPseudoFieldAccess(self: *const ExpressionLowerer, target: c.MlirValue, field_name: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.createPseudoFieldAccess(self, target, field_name, span);
    }

    /// Create length access for arrays and slices
    pub fn createLengthAccess(self: *const ExpressionLowerer, target: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.createLengthAccess(self, target, span);
    }

    /// Convert index value to MLIR index type
    pub fn convertIndexToIndexType(self: *const ExpressionLowerer, index: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.convertIndexToIndexType(self, index, span);
    }

    /// Create array index load with bounds checking
    pub fn createArrayIndexLoad(self: *const ExpressionLowerer, array: c.MlirValue, index: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.createArrayIndexLoad(self, array, index, span);
    }

    /// Create map index load operation
    pub fn createMapIndexLoad(self: *const ExpressionLowerer, map: c.MlirValue, key: c.MlirValue, result_type: ?c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_access.createMapIndexLoad(self, map, key, result_type, span);
    }

    /// Create switch expression as chain of scf.if operations
    pub fn createSwitchIfChain(self: *const ExpressionLowerer, condition: c.MlirValue, cases: []lib.ast.Expressions.SwitchCase, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_advanced.createSwitchIfChain(self, condition, cases, span);
    }

    /// Convert TypeInfo to string representation for attributes
    pub fn getTypeString(self: *const ExpressionLowerer, type_info: lib.ast.Types.TypeInfo) []const u8 {
        return expr_advanced.getTypeString(self, type_info);
    }

    /// Add verification-related attributes
    pub fn addVerificationAttributes(self: *const ExpressionLowerer, attributes: *std.ArrayList(c.MlirNamedAttribute), verification_type: []const u8, context: []const u8) void {
        expr_advanced.addVerificationAttributes(self, attributes, verification_type, context);
    }

    /// Create verification metadata
    pub fn createVerificationMetadata(self: *const ExpressionLowerer, quantifier_type: lib.ast.Expressions.QuantifierType, variable_name: []const u8, variable_type: lib.ast.Types.TypeInfo) std.ArrayList(c.MlirNamedAttribute) {
        return expr_advanced.createVerificationMetadata(self, quantifier_type, variable_name, variable_type);
    }

    /// Create empty array memref
    pub fn createEmptyArray(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_advanced.createEmptyArray(self, span);
    }

    /// Create initialized array with elements
    pub fn createInitializedArray(self: *const ExpressionLowerer, elements: []*lib.ast.Expressions.ExprNode, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_advanced.createInitializedArray(self, elements, span);
    }

    /// Create empty struct
    pub fn createEmptyStruct(self: *const ExpressionLowerer, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_advanced.createEmptyStruct(self, span);
    }

    /// Create initialized struct with fields
    pub fn createInitializedStruct(self: *const ExpressionLowerer, fields: []lib.ast.Expressions.AnonymousStructField, span: lib.ast.SourceSpan) c.MlirValue {
        return expr_advanced.createInitializedStruct(self, fields, span);
    }

    /// Create tuple type from element types
    pub fn createTupleType(self: *const ExpressionLowerer, element_types: []c.MlirType) c.MlirType {
        return expr_advanced.createTupleType(self, element_types);
    }

    /// Create an operation that captures a top-level expression value
    pub fn createExpressionCapture(self: *const ExpressionLowerer, expr_value: c.MlirValue, span: lib.ast.SourceSpan) c.MlirOperation {
        return expr_advanced.createExpressionCapture(self, expr_value, span);
    }
};
