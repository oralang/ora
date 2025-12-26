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
const z3 = @import("c.zig");
const mlir = @import("mlir_c_api").c;
const Context = @import("context.zig").Context;

/// MLIR to SMT encoder
pub const Encoder = struct {
    context: *Context,
    allocator: std.mem.Allocator,

    /// Map from MLIR value to Z3 AST (for caching encoded values)
    value_map: std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),

    pub fn init(context: *Context, allocator: std.mem.Allocator) Encoder {
        return .{
            .context = context,
            .allocator = allocator,
            .value_map = std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.value_map.deinit();
    }

    //===----------------------------------------------------------------------===//
    // type Encoding
    //===----------------------------------------------------------------------===//

    /// Encode MLIR type to Z3 sort
    pub fn encodeType(self: *Encoder, mlir_type: anytype) EncodeError!z3.Z3_sort {
        if (@TypeOf(mlir_type) == mlir.MlirType) {
            return self.encodeMLIRType(mlir_type);
        }
        return self.mkBitVectorSort(256);
    }

    /// Create Z3 bitvector sort of given width
    pub fn mkBitVectorSort(self: *Encoder, width: u32) z3.Z3_sort {
        return z3.Z3_mk_bv_sort(self.context.ctx, width);
    }

    //===----------------------------------------------------------------------===//
    // constant Encoding
    //===----------------------------------------------------------------------===//

    /// Encode integer constant to Z3 bitvector
    pub fn encodeIntegerConstant(self: *Encoder, value: u256, width: u32) EncodeError!z3.Z3_ast {
        const sort = self.mkBitVectorSort(width);
        // z3_mk_unsigned_int64 only handles 64-bit, so we need to handle larger values
        // for u256, we'll need to use Z3_mk_numeral with string representation
        if (width <= 64) {
            return z3.Z3_mk_unsigned_int64(self.context.ctx, @intCast(value), sort);
        } else {
            // for larger bitvectors, use string representation
            // format as decimal string for Z3_mk_numeral
            const value_str = try std.fmt.allocPrint(self.allocator, "{d}", .{value});
            defer self.allocator.free(value_str);
            const value_z = try self.allocator.dupeZ(u8, value_str);
            defer self.allocator.free(value_z);
            return z3.Z3_mk_numeral(self.context.ctx, value_z.ptr, sort);
        }
    }

    /// Encode boolean constant
    pub fn encodeBoolConstant(self: *Encoder, value: bool) z3.Z3_ast {
        return if (value) z3.Z3_mk_true(self.context.ctx) else z3.Z3_mk_false(self.context.ctx);
    }

    //===----------------------------------------------------------------------===//
    // variable Encoding
    //===----------------------------------------------------------------------===//

    /// Create Z3 variable (uninterpreted constant) from name and sort
    pub fn mkVariable(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        const name_copy = try self.allocator.dupeZ(u8, name);
        errdefer self.allocator.free(name_copy);

        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name_copy);
        return z3.Z3_mk_const(self.context.ctx, symbol, sort);
    }

    //===----------------------------------------------------------------------===//
    // arithmetic Operations
    //===----------------------------------------------------------------------===//

    /// Encode arithmetic operation (add, sub, mul, div, rem)
    pub fn encodeArithmeticOp(
        self: *Encoder,
        op: ArithmeticOp,
        lhs: z3.Z3_ast,
        rhs: z3.Z3_ast,
    ) !z3.Z3_ast {
        return switch (op) {
            .Add => z3.Z3_mk_bv_add(self.context.ctx, lhs, rhs),
            .Sub => z3.Z3_mk_bv_sub(self.context.ctx, lhs, rhs),
            .Mul => z3.Z3_mk_bv_mul(self.context.ctx, lhs, rhs),
            .Div => z3.Z3_mk_bv_udiv(self.context.ctx, lhs, rhs), // Unsigned division
            .Rem => z3.Z3_mk_bv_urem(self.context.ctx, lhs, rhs), // Unsigned remainder
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

    pub const BitwiseOp = enum {
        And,
        Or,
        Xor,
    };

    pub const ShiftOp = enum {
        Shl,
        ShrSigned,
        ShrUnsigned,
    };

    pub fn encodeBitwiseOp(self: *Encoder, op: BitwiseOp, lhs: z3.Z3_ast, rhs: z3.Z3_ast) !z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (kind == z3.Z3_BOOL_SORT) {
            return switch (op) {
                .And => z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs }),
                .Or => z3.Z3_mk_or(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs }),
                .Xor => z3.Z3_mk_xor(self.context.ctx, lhs, rhs),
            };
        }

        return switch (op) {
            .And => z3.Z3_mk_bvand(self.context.ctx, lhs, rhs),
            .Or => z3.Z3_mk_bvor(self.context.ctx, lhs, rhs),
            .Xor => z3.Z3_mk_bvxor(self.context.ctx, lhs, rhs),
        };
    }

    pub fn encodeShiftOp(self: *Encoder, op: ShiftOp, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        return switch (op) {
            .Shl => z3.Z3_mk_bvshl(self.context.ctx, lhs, rhs),
            .ShrSigned => z3.Z3_mk_bvashr(self.context.ctx, lhs, rhs),
            .ShrUnsigned => z3.Z3_mk_bvlshr(self.context.ctx, lhs, rhs),
        };
    }

    /// Check for overflow in addition (u256 + u256 can overflow)
    pub fn checkAddOverflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        // overflow occurs when result < lhs (unsigned comparison)
        const result = z3.Z3_mk_bv_add(self.context.ctx, lhs, rhs);
        return z3.Z3_mk_bvult(self.context.ctx, result, lhs);
    }

    /// Check for underflow in subtraction (u256 - u256 can underflow)
    pub fn checkSubUnderflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        // underflow occurs when rhs > lhs
        return z3.Z3_mk_bvult(self.context.ctx, lhs, rhs);
    }

    /// Check for overflow in multiplication
    pub fn checkMulOverflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        // for multiplication, we need to check if the high bits are non-zero
        // this is more complex - for now, we'll use a conservative check
        // todo: Implement proper multiplication overflow detection
        // for u256, multiplication overflow occurs when the result doesn't fit in 256 bits
        // we can check this by computing the full-width result and checking high bits
        _ = lhs;
        _ = rhs;
        return z3.Z3_mk_false(self.context.ctx); // Placeholder - needs proper implementation
    }

    /// Check for division by zero
    pub fn checkDivByZero(self: *Encoder, divisor: z3.Z3_ast) z3.Z3_ast {
        // create zero constant of same width as divisor
        const sort = z3.Z3_get_sort(self.context.ctx, divisor);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        return z3.Z3_mk_eq(self.context.ctx, divisor, zero);
    }

    // helper function to get sort from Z3 AST
    fn getSort(self: *Encoder, ast: z3.Z3_ast) z3.Z3_sort {
        return z3.Z3_get_sort(self.context.ctx, ast);
    }

    //===----------------------------------------------------------------------===//
    // comparison Operations
    //===----------------------------------------------------------------------===//

    /// Encode comparison operation
    pub fn encodeComparisonOp(
        self: *Encoder,
        op: ComparisonOp,
        lhs: z3.Z3_ast,
        rhs: z3.Z3_ast,
    ) z3.Z3_ast {
        return switch (op) {
            .Eq => z3.Z3_mk_eq(self.context.ctx, lhs, rhs),
            .Ne => z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, lhs, rhs)),
            .Lt => z3.Z3_mk_bvult(self.context.ctx, lhs, rhs), // Unsigned less than
            .Le => z3.Z3_mk_bvule(self.context.ctx, lhs, rhs), // Unsigned less than or equal
            .Gt => z3.Z3_mk_bvugt(self.context.ctx, lhs, rhs), // Unsigned greater than
            .Ge => z3.Z3_mk_bvuge(self.context.ctx, lhs, rhs), // Unsigned greater than or equal
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
    // boolean Operations
    //===----------------------------------------------------------------------===//

    /// Encode boolean AND
    pub fn encodeAnd(self: *Encoder, args: []const z3.Z3_ast) z3.Z3_ast {
        if (args.len == 0) return z3.Z3_mk_true(self.context.ctx);
        if (args.len == 1) return args[0];
        if (args.len == 2) {
            return z3.Z3_mk_and(self.context.ctx, 2, args.ptr);
        }
        // for more than 2 arguments, chain them
        var result = z3.Z3_mk_and(self.context.ctx, 2, args[0..2].ptr);
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const args_slice = [_]z3.Z3_ast{ result, args[i] };
            result = z3.Z3_mk_and(self.context.ctx, 2, &args_slice);
        }
        return result;
    }

    /// Encode boolean OR
    pub fn encodeOr(self: *Encoder, args: []const z3.Z3_ast) z3.Z3_ast {
        if (args.len == 0) return z3.Z3_mk_false(self.context.ctx);
        if (args.len == 1) return args[0];
        if (args.len == 2) {
            return z3.Z3_mk_or(self.context.ctx, 2, args.ptr);
        }
        // for more than 2 arguments, chain them
        var result = z3.Z3_mk_or(self.context.ctx, 2, args[0..2].ptr);
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const args_slice = [_]z3.Z3_ast{ result, args[i] };
            result = z3.Z3_mk_or(self.context.ctx, 2, &args_slice);
        }
        return result;
    }

    /// Encode boolean NOT
    pub fn encodeNot(self: *Encoder, arg: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_not(self.context.ctx, arg);
    }

    /// Encode implication (A => B)
    pub fn encodeImplies(self: *Encoder, antecedent: z3.Z3_ast, consequent: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_implies(self.context.ctx, antecedent, consequent);
    }

    /// Encode if-then-else (ite)
    pub fn encodeIte(self: *Encoder, condition: z3.Z3_ast, then_expr: z3.Z3_ast, else_expr: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_ite(self.context.ctx, condition, then_expr, else_expr);
    }

    //===----------------------------------------------------------------------===//
    // array/Storage Operations
    //===----------------------------------------------------------------------===//

    /// Create array sort (for storage maps: address -> value)
    pub fn mkArraySort(self: *Encoder, domain_sort: z3.Z3_sort, range_sort: z3.Z3_sort) z3.Z3_sort {
        return z3.Z3_mk_array_sort(self.context.ctx, domain_sort, range_sort);
    }

    /// Encode array select (read from array/map)
    pub fn encodeSelect(self: *Encoder, array: z3.Z3_ast, index: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_select(self.context.ctx, array, index);
    }

    /// Encode array store (write to array/map)
    pub fn encodeStore(self: *Encoder, array: z3.Z3_ast, index: z3.Z3_ast, value: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_store(self.context.ctx, array, index, value);
    }

    //===----------------------------------------------------------------------===//
    // mlir Operation Encoding
    //===----------------------------------------------------------------------===//

    /// Encode an MLIR operation to Z3 AST
    /// This is the main entry point for encoding MLIR operations
    pub fn encodeOperation(self: *Encoder, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        // get operation name
        const op_name_ref = mlir.oraOperationGetName(mlir_op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        // get number of operands
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);

        // encode operands recursively
        const num_ops: usize = @intCast(num_operands);
        var operands = try self.allocator.alloc(z3.Z3_ast, num_ops);
        defer self.allocator.free(operands);

        for (0..num_ops) |i| {
            const operand_value = mlir.oraOperationGetOperand(mlir_op, @intCast(i));
            operands[i] = try self.encodeValue(operand_value);
        }

        // dispatch based on operation name
        return try self.dispatchOperation(op_name, operands, mlir_op);
    }

    /// Encode an MLIR value to Z3 AST (with caching)
    pub fn encodeValue(self: *Encoder, mlir_value: mlir.MlirValue) EncodeError!z3.Z3_ast {
        // check cache first
        const value_id = @intFromPtr(mlir_value.ptr);
        if (self.value_map.get(value_id)) |cached| {
            return cached;
        }

        if (mlir.oraValueIsAOpResult(mlir_value)) {
            const defining_op = mlir.oraOpResultGetOwner(mlir_value);
            if (!mlir.oraOperationIsNull(defining_op)) {
                if (self.encodeOperation(defining_op)) |encoded| {
                    try self.value_map.put(value_id, encoded);
                    return encoded;
                } else |err| {
                    if (err != error.UnsupportedOperation) return err;
                }
            }
        }

        // if no defining operation, create a fresh variable
        const value_type = mlir.oraValueGetType(mlir_value);
        const sort = try self.encodeMLIRType(value_type);
        const var_name = try std.fmt.allocPrint(self.allocator, "v_{d}", .{value_id});
        defer self.allocator.free(var_name);
        const encoded = try self.mkVariable(var_name, sort);

        // cache the result
        try self.value_map.put(value_id, encoded);
        return encoded;
    }

    /// Encode MLIR type to Z3 sort
    pub fn encodeMLIRType(self: *Encoder, mlir_type: mlir.MlirType) EncodeError!z3.Z3_sort {
        if (mlir.oraTypeIsAddressType(mlir_type)) {
            return self.mkBitVectorSort(160);
        }

        if (mlir.oraTypeIsIntegerType(mlir_type)) {
            const builtin = mlir.oraTypeToBuiltin(mlir_type);
            const width = mlir.oraIntegerTypeGetWidth(builtin);
            if (width == 1) {
                return z3.Z3_mk_bool_sort(self.context.ctx);
            }
            return self.mkBitVectorSort(@intCast(width));
        }

        // check if it's a builtin integer type
        if (mlir.oraTypeIsAInteger(mlir_type)) {
            const width = mlir.oraIntegerTypeGetWidth(mlir_type);
            if (width == 1) {
                return z3.Z3_mk_bool_sort(self.context.ctx);
            }
            return self.mkBitVectorSort(@intCast(width));
        }

        const map_value_type = mlir.oraMapTypeGetValueType(mlir_type);
        if (!mlir.oraTypeIsNull(map_value_type)) {
            const value_sort = try self.encodeMLIRType(map_value_type);
            const key_sort = self.mkBitVectorSort(256);
            return self.mkArraySort(key_sort, value_sort);
        }

        // default to 256-bit bitvector for EVM
        return self.mkBitVectorSort(256);
    }

    /// Dispatch operation to appropriate encoder based on operation name
    fn dispatchOperation(self: *Encoder, op_name: []const u8, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        // comparison operations
        if (std.mem.eql(u8, op_name, "arith.cmpi")) {
            // get predicate from attributes
            const predicate = self.getCmpPredicate(mlir_op);
            return try self.encodeCmpOp(predicate, operands);
        }

        // constant operations
        if (std.mem.eql(u8, op_name, "arith.constant")) {
            const value = self.getConstantValue(mlir_op);
            const value_type = mlir.oraOperationGetResult(mlir_op, 0);
            const mlir_type = mlir.oraValueGetType(value_type);
            const width: u32 = if (mlir.oraTypeIsAInteger(mlir_type))
                @intCast(mlir.oraIntegerTypeGetWidth(mlir_type))
            else
                256;
            return try self.encodeConstantOp(value, width);
        }

        if (std.mem.eql(u8, op_name, "arith.andi") or std.mem.eql(u8, op_name, "arith.ori") or std.mem.eql(u8, op_name, "arith.xori")) {
            if (operands.len < 2) return error.InvalidOperandCount;
            const bw_op = if (std.mem.eql(u8, op_name, "arith.andi"))
                BitwiseOp.And
            else if (std.mem.eql(u8, op_name, "arith.ori"))
                BitwiseOp.Or
            else
                BitwiseOp.Xor;
            return try self.encodeBitwiseOp(bw_op, operands[0], operands[1]);
        }

        if (std.mem.eql(u8, op_name, "arith.shli") or std.mem.eql(u8, op_name, "arith.shrsi") or std.mem.eql(u8, op_name, "arith.shrui")) {
            if (operands.len < 2) return error.InvalidOperandCount;
            const shift_op = if (std.mem.eql(u8, op_name, "arith.shli"))
                ShiftOp.Shl
            else if (std.mem.eql(u8, op_name, "arith.shrsi"))
                ShiftOp.ShrSigned
            else
                ShiftOp.ShrUnsigned;
            return self.encodeShiftOp(shift_op, operands[0], operands[1]);
        }

        // arithmetic operations
        if (std.mem.startsWith(u8, op_name, "arith.")) {
            return try self.encodeArithOp(op_name, operands);
        }

        if (std.mem.eql(u8, op_name, "ora.add") or
            std.mem.eql(u8, op_name, "ora.sub") or
            std.mem.eql(u8, op_name, "ora.mul") or
            std.mem.eql(u8, op_name, "ora.div") or
            std.mem.eql(u8, op_name, "ora.rem"))
        {
            if (operands.len < 2) return error.InvalidOperandCount;
            const arith_op = if (std.mem.eql(u8, op_name, "ora.add"))
                ArithmeticOp.Add
            else if (std.mem.eql(u8, op_name, "ora.sub"))
                ArithmeticOp.Sub
            else if (std.mem.eql(u8, op_name, "ora.mul"))
                ArithmeticOp.Mul
            else if (std.mem.eql(u8, op_name, "ora.div"))
                ArithmeticOp.Div
            else
                ArithmeticOp.Rem;
            return self.encodeArithmeticOp(arith_op, operands[0], operands[1]);
        }

        // storage operations
        if (std.mem.eql(u8, op_name, "ora.sload")) {
            if (operands.len >= 1) {
                // storage load: select from storage array
                const storage_var = operands[0];
                const key = operands[0]; // TODO: Get actual key from operation
                return self.encodeStorageLoad(storage_var, key);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.sstore")) {
            if (operands.len >= 2) {
                // storage store: update storage array
                const storage_var = operands[0];
                const key = operands[0]; // TODO: Get actual key from operation
                const value = operands[1];
                return self.encodeStorageStore(storage_var, key, value);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.map_get")) {
            if (operands.len >= 2) {
                return self.encodeSelect(operands[0], operands[1]);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.map_store")) {
            if (operands.len >= 3) {
                const stored = self.encodeStore(operands[0], operands[1], operands[2]);
                const num_results = mlir.oraOperationGetNumResults(mlir_op);
                if (num_results > 0) return stored;
            }
        }

        if (std.mem.eql(u8, op_name, "ora.addr.to.i160")) {
            if (operands.len >= 1) {
                return z3.Z3_mk_extract(self.context.ctx, 159, 0, operands[0]);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.struct_field_extract")) {
            if (operands.len >= 1) {
                const field_name = self.getStringAttr(mlir_op, "field_name") orelse "field";
                const struct_sort = z3.Z3_get_sort(self.context.ctx, operands[0]);
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                return try self.applyFieldFunction(field_name, struct_sort, result_sort, operands[0]);
            }
        }

        // control flow
        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (operands.len >= 1) {
                const condition = operands[0];
                const then_expr = try self.extractIfYield(mlir_op, 0);
                const else_expr = try self.extractIfYield(mlir_op, 1);
                return try self.encodeControlFlow(op_name, condition, then_expr, else_expr);
            }
        }

        // unsupported operation - caller may treat as unknown symbol.
        return error.UnsupportedOperation;
    }

    fn extractIfYield(self: *Encoder, mlir_op: mlir.MlirOperation, region_index: u32) EncodeError!?z3.Z3_ast {
        const region = mlir.oraOperationGetRegion(mlir_op, region_index);
        if (mlir.oraRegionIsNull(region)) return null;
        const block = mlir.oraRegionGetFirstBlock(region);
        if (mlir.oraBlockIsNull(block)) return null;

        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.yield")) {
                const num_operands = mlir.oraOperationGetNumOperands(current);
                if (num_operands >= 1) {
                    const value = mlir.oraOperationGetOperand(current, 0);
                    return try self.encodeValue(value);
                }
                return null;
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn getOperationName(_: *Encoder, op: mlir.MlirOperation) mlir.MlirStringRef {
        return mlir.oraOperationGetName(op);
    }

    fn getStringAttr(_: *Encoder, op: mlir.MlirOperation, name: []const u8) ?[]const u8 {
        const attr_name_ref = mlir.oraStringRefCreate(name.ptr, name.len);
        const attr = mlir.oraOperationGetAttributeByName(op, attr_name_ref);
        if (mlir.oraAttributeIsNull(attr)) return null;
        const value = mlir.oraStringAttrGetValue(attr);
        if (value.data == null or value.length == 0) return null;
        return value.data[0..value.length];
    }

    fn applyFieldFunction(
        self: *Encoder,
        field_name: []const u8,
        struct_sort: z3.Z3_sort,
        result_sort: z3.Z3_sort,
        struct_value: z3.Z3_ast,
    ) !z3.Z3_ast {
        const fn_name = try std.fmt.allocPrint(self.allocator, "ora.field.{s}", .{field_name});
        defer self.allocator.free(fn_name);
        const fn_name_z = try self.allocator.dupeZ(u8, fn_name);
        defer self.allocator.free(fn_name_z);
        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, fn_name_z.ptr);
        const domain = [_]z3.Z3_sort{struct_sort};
        const func_decl = z3.Z3_mk_func_decl(self.context.ctx, symbol, 1, &domain, result_sort);
        return z3.Z3_mk_app(self.context.ctx, func_decl, 1, &[_]z3.Z3_ast{struct_value});
    }

    /// Get comparison predicate from MLIR operation
    fn getCmpPredicate(_: *Encoder, mlir_op: mlir.MlirOperation) u32 {
        // extract predicate from arith.cmpi attributes
        // the predicate attribute contains the comparison type (eq, ne, ult, ule, ugt, uge, etc.)
        const attr_name_ref = mlir.oraStringRefCreate("predicate".ptr, "predicate".len);
        const attr = mlir.oraOperationGetAttributeByName(mlir_op, attr_name_ref);
        if (mlir.oraAttributeIsNull(attr)) {
            // default to eq (0) if predicate is missing
            return 0;
        }

        // get the integer value of the predicate
        const predicate = mlir.oraIntegerAttrGetValueSInt(attr);
        // mlir predicate values: 0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
        // we use unsigned comparisons for EVM (u256), so map signed to unsigned if needed
        // predicate values are always non-negative, so safe to cast
        return @intCast(predicate);
    }

    /// Get constant value from MLIR constant operation
    fn getConstantValue(_: *Encoder, mlir_op: mlir.MlirOperation) u256 {
        // extract constant value from arith.constant attributes
        // the value attribute contains the integer constant
        const attr_name_ref = mlir.oraStringRefCreate("value".ptr, "value".len);
        const attr = mlir.oraOperationGetAttributeByName(mlir_op, attr_name_ref);
        if (mlir.oraAttributeIsNull(attr)) {
            // return 0 if value attribute is missing
            return 0;
        }

        // get the integer value (signed, but we'll interpret as unsigned for u256)
        const int_value = mlir.oraIntegerAttrGetValueSInt(attr);

        // handle negative values (for boolean true = -1 in MLIR, but we want 1)
        if (int_value < 0) {
            // for boolean values, -1 means true, convert to 1
            if (int_value == -1) {
                return 1;
            }
            // for other negative values, this represents a large unsigned value
            // convert i64 to u64 first (two's complement), then to u256
            const u64_value: u64 = @bitCast(@as(i64, int_value));
            return @intCast(u64_value);
        }

        // convert signed to unsigned (non-negative values)
        // for values that fit in i64, cast directly
        return @intCast(@as(i64, int_value));
    }

    /// Encode MLIR arithmetic operation (arith.addi, arith.subi, etc.)
    pub fn encodeArithOp(self: *Encoder, op_name: []const u8, operands: []const z3.Z3_ast) !z3.Z3_ast {
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
    pub fn encodeCmpOp(self: *Encoder, predicate: u32, operands: []const z3.Z3_ast) !z3.Z3_ast {
        if (operands.len < 2) {
            return error.InvalidOperandCount;
        }

        const lhs = operands[0];
        const rhs = operands[1];

        return switch (predicate) {
            0 => self.encodeComparisonOp(.Eq, lhs, rhs),
            1 => self.encodeComparisonOp(.Ne, lhs, rhs),
            2 => z3.Z3_mk_bvslt(self.context.ctx, lhs, rhs),
            3 => z3.Z3_mk_bvsle(self.context.ctx, lhs, rhs),
            4 => z3.Z3_mk_bvsgt(self.context.ctx, lhs, rhs),
            5 => z3.Z3_mk_bvsge(self.context.ctx, lhs, rhs),
            6 => self.encodeComparisonOp(.Lt, lhs, rhs),
            7 => self.encodeComparisonOp(.Le, lhs, rhs),
            8 => self.encodeComparisonOp(.Gt, lhs, rhs),
            9 => self.encodeComparisonOp(.Ge, lhs, rhs),
            else => return error.UnsupportedPredicate,
        };
    }

    /// Encode MLIR constant operation (arith.constant)
    pub fn encodeConstantOp(self: *Encoder, value: u256, width: u32) EncodeError!z3.Z3_ast {
        if (width == 1) {
            return self.encodeBoolConstant(value != 0);
        }
        return self.encodeIntegerConstant(value, width);
    }

    /// Encode MLIR storage load (ora.sload)
    pub fn encodeStorageLoad(self: *Encoder, storage_var: z3.Z3_ast, key: z3.Z3_ast) z3.Z3_ast {
        // storage is modeled as an array: address -> value
        return self.encodeSelect(storage_var, key);
    }

    /// Encode MLIR storage store (ora.sstore)
    pub fn encodeStorageStore(self: *Encoder, storage_var: z3.Z3_ast, key: z3.Z3_ast, value: z3.Z3_ast) z3.Z3_ast {
        // storage is modeled as an array: address -> value
        return self.encodeStore(storage_var, key, value);
    }

    /// Encode MLIR control flow (scf.if, scf.while)
    pub fn encodeControlFlow(self: *Encoder, op_name: []const u8, condition: z3.Z3_ast, then_expr: ?z3.Z3_ast, else_expr: ?z3.Z3_ast) EncodeError!z3.Z3_ast {
        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (then_expr) |then_val| {
                if (else_expr) |else_val| {
                    return self.encodeIte(condition, then_val, else_val);
                } else {
                    // if without else - return then value or undefined
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
    // error Types
    //===----------------------------------------------------------------------===//

    pub const EncodingError = error{
        InvalidOperandCount,
        UnsupportedOperation,
        UnsupportedPredicate,
        InvalidControlFlow,
    };

    pub const EncodeError = EncodingError || std.mem.Allocator.Error;
};
