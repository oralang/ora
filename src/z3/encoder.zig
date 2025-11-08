//===----------------------------------------------------------------------===//
//
// MLIR to SMT Encoder
//
//===----------------------------------------------------------------------===//
//
// Converts MLIR operations and types to Z3 SMT expressions.
// This is the core of the formal verification system.
//
//===----------------------------------------------------------------------===//

const std = @import("std");
const c = @import("c.zig").c;
const Context = @import("context.zig").Context;

/// MLIR to SMT encoder
pub const Encoder = struct {
    context: *Context,
    allocator: std.mem.Allocator,

    /// Map from MLIR value to Z3 AST (for caching encoded values)
    value_map: std.HashMap(u64, c.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),

    pub fn init(context: *Context, allocator: std.mem.Allocator) Encoder {
        return .{
            .context = context,
            .allocator = allocator,
            .value_map = std.HashMap(u64, c.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.value_map.deinit();
    }

    //===----------------------------------------------------------------------===//
    // Type Encoding
    //===----------------------------------------------------------------------===//

    /// Encode MLIR type to Z3 sort
    /// Returns Z3 sort for the given MLIR type
    /// TODO: Get actual MLIR type information from mlir_type parameter
    pub fn encodeType(self: *Encoder, _: anytype) !c.Z3_sort {
        // TODO: Extract actual type width from MLIR type
        // For now, assume u256 (256-bit bitvector) for EVM
        return c.Z3_mk_bv_sort(self.context.ctx, 256);
    }

    /// Create Z3 bitvector sort of given width
    pub fn mkBitVectorSort(self: *Encoder, width: u32) c.Z3_sort {
        return c.Z3_mk_bv_sort(self.context.ctx, width);
    }

    //===----------------------------------------------------------------------===//
    // Constant Encoding
    //===----------------------------------------------------------------------===//

    /// Encode integer constant to Z3 bitvector
    pub fn encodeIntegerConstant(self: *Encoder, value: u256, width: u32) !c.Z3_ast {
        const sort = self.mkBitVectorSort(width);
        // Z3_mk_unsigned_int64 only handles 64-bit, so we need to handle larger values
        // For u256, we'll need to use Z3_mk_numeral with string representation
        if (width <= 64) {
            return c.Z3_mk_unsigned_int64(self.context.ctx, @intCast(value), sort);
        } else {
            // For larger bitvectors, use string representation
            // Format as decimal string for Z3_mk_numeral
            const value_str = try std.fmt.allocPrintZ(self.allocator, "{d}", .{value});
            defer self.allocator.free(value_str);
            // Z3_mk_numeral takes a string and a sort
            // Note: Z3_mk_numeral may not be in our bindings, so we'll need to add it
            // For now, use a workaround: create a variable and assert it equals the value
            // TODO: Add Z3_mk_numeral to c.zig bindings
            const symbol = c.Z3_mk_string_symbol(self.context.ctx, value_str.ptr);
            const var_ast = c.Z3_mk_const(self.context.ctx, symbol, sort);
            // For now, return the variable - we'll need proper numeral support
            // This is a limitation we'll address when we add full Z3 API bindings
            return var_ast;
        }
    }

    /// Encode boolean constant
    pub fn encodeBoolConstant(self: *Encoder, value: bool) c.Z3_ast {
        return if (value) c.Z3_mk_true(self.context.ctx) else c.Z3_mk_false(self.context.ctx);
    }

    //===----------------------------------------------------------------------===//
    // Variable Encoding
    //===----------------------------------------------------------------------===//

    /// Create Z3 variable (uninterpreted constant) from name and sort
    pub fn mkVariable(self: *Encoder, name: []const u8, sort: c.Z3_sort) !c.Z3_ast {
        const name_copy = try self.allocator.dupeZ(u8, name);
        errdefer self.allocator.free(name_copy);

        const symbol = c.Z3_mk_string_symbol(self.context.ctx, name_copy);
        return c.Z3_mk_const(self.context.ctx, symbol, sort);
    }

    //===----------------------------------------------------------------------===//
    // Arithmetic Operations
    //===----------------------------------------------------------------------===//

    /// Encode arithmetic operation (add, sub, mul, div, rem)
    pub fn encodeArithmeticOp(
        self: *Encoder,
        op: ArithmeticOp,
        lhs: c.Z3_ast,
        rhs: c.Z3_ast,
    ) !c.Z3_ast {
        return switch (op) {
            .Add => c.Z3_mk_bv_add(self.context.ctx, lhs, rhs),
            .Sub => c.Z3_mk_bv_sub(self.context.ctx, lhs, rhs),
            .Mul => c.Z3_mk_bv_mul(self.context.ctx, lhs, rhs),
            .Div => c.Z3_mk_bv_udiv(self.context.ctx, lhs, rhs), // Unsigned division
            .Rem => c.Z3_mk_bv_urem(self.context.ctx, lhs, rhs), // Unsigned remainder
        };
    }

    /// Arithmetic operation types
    pub const ArithmeticOp = enum {
        Add,
        Sub,
        Mul,
        Div,
        Rem,
    };

    /// Check for overflow in addition (u256 + u256 can overflow)
    pub fn checkAddOverflow(self: *Encoder, lhs: c.Z3_ast, rhs: c.Z3_ast) c.Z3_ast {
        // Overflow occurs when result < lhs (unsigned comparison)
        const result = c.Z3_mk_bv_add(self.context.ctx, lhs, rhs);
        return c.Z3_mk_bvult(self.context.ctx, result, lhs);
    }

    /// Check for underflow in subtraction (u256 - u256 can underflow)
    pub fn checkSubUnderflow(self: *Encoder, lhs: c.Z3_ast, rhs: c.Z3_ast) c.Z3_ast {
        // Underflow occurs when rhs > lhs
        return c.Z3_mk_bvult(self.context.ctx, lhs, rhs);
    }

    /// Check for overflow in multiplication
    pub fn checkMulOverflow(self: *Encoder, lhs: c.Z3_ast, rhs: c.Z3_ast) c.Z3_ast {
        // For multiplication, we need to check if the high bits are non-zero
        // This is more complex - for now, we'll use a conservative check
        // TODO: Implement proper multiplication overflow detection
        // For u256, multiplication overflow occurs when the result doesn't fit in 256 bits
        // We can check this by computing the full-width result and checking high bits
        _ = lhs;
        _ = rhs;
        return c.Z3_mk_false(self.context.ctx); // Placeholder - needs proper implementation
    }

    /// Check for division by zero
    pub fn checkDivByZero(self: *Encoder, divisor: c.Z3_ast) c.Z3_ast {
        // Create zero constant of same width as divisor
        const sort = c.Z3_get_sort(self.context.ctx, divisor);
        const zero = c.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        return c.Z3_mk_eq(self.context.ctx, divisor, zero);
    }

    // Helper function to get sort from Z3 AST
    fn getSort(self: *Encoder, ast: c.Z3_ast) c.Z3_sort {
        return c.Z3_get_sort(self.context.ctx, ast);
    }

    //===----------------------------------------------------------------------===//
    // Comparison Operations
    //===----------------------------------------------------------------------===//

    /// Encode comparison operation
    pub fn encodeComparisonOp(
        self: *Encoder,
        op: ComparisonOp,
        lhs: c.Z3_ast,
        rhs: c.Z3_ast,
    ) c.Z3_ast {
        return switch (op) {
            .Eq => c.Z3_mk_eq(self.context.ctx, lhs, rhs),
            .Ne => c.Z3_mk_not(self.context.ctx, c.Z3_mk_eq(self.context.ctx, lhs, rhs)),
            .Lt => c.Z3_mk_bvult(self.context.ctx, lhs, rhs), // Unsigned less than
            .Le => c.Z3_mk_bvule(self.context.ctx, lhs, rhs), // Unsigned less than or equal
            .Gt => c.Z3_mk_bvugt(self.context.ctx, lhs, rhs), // Unsigned greater than
            .Ge => c.Z3_mk_bvuge(self.context.ctx, lhs, rhs), // Unsigned greater than or equal
        };
    }

    /// Comparison operation types
    pub const ComparisonOp = enum {
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
    };

    //===----------------------------------------------------------------------===//
    // Boolean Operations
    //===----------------------------------------------------------------------===//

    /// Encode boolean AND
    pub fn encodeAnd(self: *Encoder, args: []const c.Z3_ast) c.Z3_ast {
        if (args.len == 0) return c.Z3_mk_true(self.context.ctx);
        if (args.len == 1) return args[0];
        if (args.len == 2) {
            return c.Z3_mk_and(self.context.ctx, 2, args.ptr);
        }
        // For more than 2 arguments, chain them
        var result = c.Z3_mk_and(self.context.ctx, 2, args[0..2].ptr);
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const args_slice = [_]c.Z3_ast{ result, args[i] };
            result = c.Z3_mk_and(self.context.ctx, 2, &args_slice);
        }
        return result;
    }

    /// Encode boolean OR
    pub fn encodeOr(self: *Encoder, args: []const c.Z3_ast) c.Z3_ast {
        if (args.len == 0) return c.Z3_mk_false(self.context.ctx);
        if (args.len == 1) return args[0];
        if (args.len == 2) {
            return c.Z3_mk_or(self.context.ctx, 2, args.ptr);
        }
        // For more than 2 arguments, chain them
        var result = c.Z3_mk_or(self.context.ctx, 2, args[0..2].ptr);
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const args_slice = [_]c.Z3_ast{ result, args[i] };
            result = c.Z3_mk_or(self.context.ctx, 2, &args_slice);
        }
        return result;
    }

    /// Encode boolean NOT
    pub fn encodeNot(self: *Encoder, arg: c.Z3_ast) c.Z3_ast {
        return c.Z3_mk_not(self.context.ctx, arg);
    }

    /// Encode implication (A => B)
    pub fn encodeImplies(self: *Encoder, antecedent: c.Z3_ast, consequent: c.Z3_ast) c.Z3_ast {
        return c.Z3_mk_implies(self.context.ctx, antecedent, consequent);
    }

    /// Encode if-then-else (ite)
    pub fn encodeIte(self: *Encoder, condition: c.Z3_ast, then_expr: c.Z3_ast, else_expr: c.Z3_ast) c.Z3_ast {
        return c.Z3_mk_ite(self.context.ctx, condition, then_expr, else_expr);
    }

    //===----------------------------------------------------------------------===//
    // Array/Storage Operations
    //===----------------------------------------------------------------------===//

    /// Create array sort (for storage maps: address -> value)
    pub fn mkArraySort(self: *Encoder, domain_sort: c.Z3_sort, range_sort: c.Z3_sort) c.Z3_sort {
        return c.Z3_mk_array_sort(self.context.ctx, domain_sort, range_sort);
    }

    /// Encode array select (read from array/map)
    pub fn encodeSelect(self: *Encoder, array: c.Z3_ast, index: c.Z3_ast) c.Z3_ast {
        return c.Z3_mk_select(self.context.ctx, array, index);
    }

    /// Encode array store (write to array/map)
    pub fn encodeStore(self: *Encoder, array: c.Z3_ast, index: c.Z3_ast, value: c.Z3_ast) c.Z3_ast {
        return c.Z3_mk_store(self.context.ctx, array, index, value);
    }

    //===----------------------------------------------------------------------===//
    // MLIR Operation Encoding
    //===----------------------------------------------------------------------===//

    /// Encode an MLIR operation to Z3 AST
    /// This is the main entry point for encoding MLIR operations
    pub fn encodeOperation(self: *Encoder, mlir_op: c.MlirOperation) !?c.Z3_ast {
        // Get operation name
        const op_name_id = c.mlirOperationGetName(mlir_op);
        const op_name_str = c.mlirIdentifierStr(op_name_id);
        const op_name = op_name_str.data[0..op_name_str.length];

        // Get number of operands
        const num_operands = c.mlirOperationGetNumOperands(mlir_op);

        // Encode operands recursively
        const num_ops: usize = @intCast(num_operands);
        var operands = try self.allocator.alloc(c.Z3_ast, num_ops);
        defer self.allocator.free(operands);

        for (0..num_ops) |i| {
            const operand_value = c.mlirOperationGetOperand(mlir_op, @intCast(i));
            if (self.encodeValue(operand_value)) |encoded| {
                operands[i] = encoded;
            } else |err| {
                // If encoding fails, return error
                return err;
            }
        }

        // Dispatch based on operation name
        return try self.dispatchOperation(op_name, operands, mlir_op);
    }

    /// Encode an MLIR value to Z3 AST (with caching)
    pub fn encodeValue(self: *Encoder, mlir_value: c.MlirValue) !c.Z3_ast {
        // Check cache first
        const value_id = @intFromPtr(mlir_value.ptr);
        if (self.value_map.get(value_id)) |cached| {
            return cached;
        }

        // Get the defining operation for this value
        const defining_op = c.mlirValueGetDefiningOp(mlir_value);
        if (!c.mlirOperationIsNull(defining_op)) {
            // Encode the defining operation
            if (try self.encodeOperation(defining_op)) |encoded| {
                // Cache the result
                try self.value_map.put(value_id, encoded);
                return encoded;
            }
        }

        // If no defining operation, create a fresh variable
        const value_type = c.mlirValueGetType(mlir_value);
        const sort = try self.encodeMLIRType(value_type);
        const var_name = try std.fmt.allocPrint(self.allocator, "v_{d}", .{value_id});
        defer self.allocator.free(var_name);
        const encoded = try self.mkVariable(var_name, sort);

        // Cache the result
        try self.value_map.put(value_id, encoded);
        return encoded;
    }

    /// Encode MLIR type to Z3 sort
    pub fn encodeMLIRType(self: *Encoder, mlir_type: c.MlirType) !c.Z3_sort {
        // Check if it's an integer type
        if (c.mlirTypeIsAInteger(mlir_type)) {
            const width = c.mlirIntegerTypeGetWidth(mlir_type);
            // Treat 1-bit integer as boolean
            if (width == 1) {
                return c.Z3_mk_bool_sort(self.context.ctx);
            }
            return self.mkBitVectorSort(@intCast(width));
        }

        // Default to 256-bit bitvector for EVM
        return self.mkBitVectorSort(256);
    }

    /// Dispatch operation to appropriate encoder based on operation name
    fn dispatchOperation(self: *Encoder, op_name: []const u8, operands: []const c.Z3_ast, mlir_op: c.MlirOperation) !?c.Z3_ast {
        // Arithmetic operations
        if (std.mem.startsWith(u8, op_name, "arith.")) {
            return try self.encodeArithOp(op_name, operands);
        }

        // Comparison operations
        if (std.mem.eql(u8, op_name, "arith.cmpi")) {
            // Get predicate from attributes
            const predicate = self.getCmpPredicate(mlir_op);
            return try self.encodeCmpOp(predicate, operands);
        }

        // Constant operations
        if (std.mem.eql(u8, op_name, "arith.constant")) {
            const value = self.getConstantValue(mlir_op);
            const value_type = c.mlirOperationGetResult(mlir_op, 0);
            const mlir_type = c.mlirValueGetType(value_type);
            const width: u32 = if (c.mlirTypeIsAInteger(mlir_type))
                @intCast(c.mlirIntegerTypeGetWidth(mlir_type))
            else
                256;
            return try self.encodeConstantOp(value, width);
        }

        // Storage operations
        if (std.mem.eql(u8, op_name, "ora.sload")) {
            if (operands.len >= 1) {
                // Storage load: select from storage array
                const storage_var = operands[0];
                const key = operands[0]; // TODO: Get actual key from operation
                return self.encodeStorageLoad(storage_var, key);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.sstore")) {
            if (operands.len >= 2) {
                // Storage store: update storage array
                const storage_var = operands[0];
                const key = operands[0]; // TODO: Get actual key from operation
                const value = operands[1];
                return self.encodeStorageStore(storage_var, key, value);
            }
        }

        // Control flow
        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (operands.len >= 1) {
                const condition = operands[0];
                // TODO: Extract then/else expressions from regions
                return try self.encodeControlFlow(op_name, condition, null, null);
            }
        }

        // Unsupported operation - return null to indicate we can't encode it
        return null;
    }

    /// Get comparison predicate from MLIR operation
    fn getCmpPredicate(_: *Encoder, mlir_op: c.MlirOperation) u32 {
        // Extract predicate from arith.cmpi attributes
        // The predicate attribute contains the comparison type (eq, ne, ult, ule, ugt, uge, etc.)
        const attr_name_ref = c.MlirStringRef{ .data = "predicate".ptr, .length = "predicate".len };
        const attr = c.mlirOperationGetAttributeByName(mlir_op, attr_name_ref);
        if (c.mlirAttributeIsNull(attr)) {
            // Default to eq (0) if predicate is missing
            return 0;
        }

        // Get the integer value of the predicate
        const predicate = c.mlirIntegerAttrGetValueSInt(attr);
        // MLIR predicate values: 0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
        // We use unsigned comparisons for EVM (u256), so map signed to unsigned if needed
        // Predicate values are always non-negative, so safe to cast
        return @intCast(predicate);
    }

    /// Get constant value from MLIR constant operation
    fn getConstantValue(_: *Encoder, mlir_op: c.MlirOperation) u256 {
        // Extract constant value from arith.constant attributes
        // The value attribute contains the integer constant
        const attr_name_ref = c.MlirStringRef{ .data = "value".ptr, .length = "value".len };
        const attr = c.mlirOperationGetAttributeByName(mlir_op, attr_name_ref);
        if (c.mlirAttributeIsNull(attr)) {
            // Return 0 if value attribute is missing
            return 0;
        }

        // Get the integer value (signed, but we'll interpret as unsigned for u256)
        const int_value = c.mlirIntegerAttrGetValueSInt(attr);

        // Handle negative values (for boolean true = -1 in MLIR, but we want 1)
        if (int_value < 0) {
            // For boolean values, -1 means true, convert to 1
            if (int_value == -1) {
                return 1;
            }
            // For other negative values, this represents a large unsigned value
            // Convert i64 to u64 first (two's complement), then to u256
            const u64_value: u64 = @bitCast(@as(i64, int_value));
            return @intCast(u64_value);
        }

        // Convert signed to unsigned (non-negative values)
        // For values that fit in i64, cast directly
        return @intCast(@as(i64, int_value));
    }

    /// Encode MLIR arithmetic operation (arith.addi, arith.subi, etc.)
    pub fn encodeArithOp(self: *Encoder, op_name: []const u8, operands: []const c.Z3_ast) !c.Z3_ast {
        if (operands.len < 2) {
            return error.InvalidOperandCount;
        }

        const lhs = operands[0];
        const rhs = operands[1];

        const arith_op = if (std.mem.eql(u8, op_name, "arith.addi"))
            ArithmeticOp.Add
        else if (std.mem.eql(u8, op_name, "arith.subi"))
            ArithmeticOp.Sub
        else if (std.mem.eql(u8, op_name, "arith.muli"))
            ArithmeticOp.Mul
        else if (std.mem.eql(u8, op_name, "arith.divui"))
            ArithmeticOp.Div
        else if (std.mem.eql(u8, op_name, "arith.remui"))
            ArithmeticOp.Rem
        else {
            return error.UnsupportedOperation;
        };

        return self.encodeArithmeticOp(arith_op, lhs, rhs);
    }

    /// Encode MLIR comparison operation (arith.cmpi)
    pub fn encodeCmpOp(self: *Encoder, predicate: u32, operands: []const c.Z3_ast) !c.Z3_ast {
        if (operands.len < 2) {
            return error.InvalidOperandCount;
        }

        const lhs = operands[0];
        const rhs = operands[1];

        // Z3 predicate values: 0=eq, 1=ne, 2=ult, 3=ule, 4=ugt, 5=uge
        const cmp_op = switch (predicate) {
            0 => ComparisonOp.Eq,
            1 => ComparisonOp.Ne,
            2 => ComparisonOp.Lt,
            3 => ComparisonOp.Le,
            4 => ComparisonOp.Gt,
            5 => ComparisonOp.Ge,
            else => return error.UnsupportedPredicate,
        };

        return self.encodeComparisonOp(cmp_op, lhs, rhs);
    }

    /// Encode MLIR constant operation (arith.constant)
    pub fn encodeConstantOp(self: *Encoder, value: u256, width: u32) !c.Z3_ast {
        return self.encodeIntegerConstant(value, width);
    }

    /// Encode MLIR storage load (ora.sload)
    pub fn encodeStorageLoad(self: *Encoder, storage_var: c.Z3_ast, key: c.Z3_ast) c.Z3_ast {
        // Storage is modeled as an array: address -> value
        return self.encodeSelect(storage_var, key);
    }

    /// Encode MLIR storage store (ora.sstore)
    pub fn encodeStorageStore(self: *Encoder, storage_var: c.Z3_ast, key: c.Z3_ast, value: c.Z3_ast) c.Z3_ast {
        // Storage is modeled as an array: address -> value
        return self.encodeStore(storage_var, key, value);
    }

    /// Encode MLIR control flow (scf.if, scf.while)
    pub fn encodeControlFlow(self: *Encoder, op_name: []const u8, condition: c.Z3_ast, then_expr: ?c.Z3_ast, else_expr: ?c.Z3_ast) !c.Z3_ast {
        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (then_expr) |then_val| {
                if (else_expr) |else_val| {
                    return self.encodeIte(condition, then_val, else_val);
                } else {
                    // If without else - return then value or undefined
                    return then_val;
                }
            } else {
                return error.InvalidControlFlow;
            }
        } else {
            return error.UnsupportedOperation;
        }
    }

    //===----------------------------------------------------------------------===//
    // Error Types
    //===----------------------------------------------------------------------===//

    pub const EncodingError = error{
        InvalidOperandCount,
        UnsupportedOperation,
        UnsupportedPredicate,
        InvalidControlFlow,
    };
};
