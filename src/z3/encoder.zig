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
    /// Map from global storage name to Z3 AST (for consistent storage symbols)
    global_map: std.StringHashMap(z3.Z3_ast),
    /// Keep Z3 symbol names alive for the life of the encoder.
    string_storage: std.ArrayList([]u8),
    /// Pending constraints emitted during encoding (e.g., error.unwrap validity).
    pending_constraints: std.ArrayList(z3.Z3_ast),
    /// Cache of error_union tuple sorts by MLIR type pointer.
    error_union_sorts: std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),

    pub fn init(context: *Context, allocator: std.mem.Allocator) Encoder {
        return .{
            .context = context,
            .allocator = allocator,
            .value_map = std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .global_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .string_storage = std.ArrayList([]u8){},
            .pending_constraints = std.ArrayList(z3.Z3_ast){},
            .error_union_sorts = std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Encoder) void {
        var it = self.global_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_map.deinit();
        for (self.string_storage.items) |buf| {
            self.allocator.free(buf);
        }
        self.string_storage.deinit(self.allocator);
        self.pending_constraints.deinit(self.allocator);
        self.error_union_sorts.deinit();
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
        try self.string_storage.append(self.allocator, name_copy[0 .. name.len + 1]);

        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name_copy);
        return z3.Z3_mk_const(self.context.ctx, symbol, sort);
    }

    fn mkSymbol(self: *Encoder, name: []const u8) EncodeError!z3.Z3_symbol {
        const name_copy = try self.allocator.dupeZ(u8, name);
        errdefer self.allocator.free(name_copy);
        try self.string_storage.append(self.allocator, name_copy[0 .. name.len + 1]);
        return z3.Z3_mk_string_symbol(self.context.ctx, name_copy);
    }

    fn mkUndefValue(self: *Encoder, sort: z3.Z3_sort, label: []const u8, id: u64) EncodeError!z3.Z3_ast {
        const name = try std.fmt.allocPrint(self.allocator, "undef_{s}_{d}", .{ label, id });
        defer self.allocator.free(name);
        return try self.mkVariable(name, sort);
    }

    fn addConstraint(self: *Encoder, constraint: z3.Z3_ast) void {
        self.pending_constraints.append(self.allocator, constraint) catch {};
    }

    pub fn takeConstraints(self: *Encoder, allocator: std.mem.Allocator) ![]z3.Z3_ast {
        if (self.pending_constraints.items.len == 0) return &[_]z3.Z3_ast{};
        const slice = try allocator.dupe(z3.Z3_ast, self.pending_constraints.items);
        self.pending_constraints.clearRetainingCapacity();
        return slice;
    }

    fn getOrCreateGlobal(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.global_map.get(name)) |existing| {
            return existing;
        }
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);

        const global_name = try std.fmt.allocPrint(self.allocator, "g_{s}", .{name});
        defer self.allocator.free(global_name);
        const ast = try self.mkVariable(global_name, sort);
        try self.global_map.put(key, ast);
        return ast;
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
        const lhs_sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const rhs_sort = z3.Z3_get_sort(self.context.ctx, rhs);
        const lhs_kind = z3.Z3_get_sort_kind(self.context.ctx, lhs_sort);
        const rhs_kind = z3.Z3_get_sort_kind(self.context.ctx, rhs_sort);

        // If both are boolean, use boolean operations
        if (lhs_kind == z3.Z3_BOOL_SORT and rhs_kind == z3.Z3_BOOL_SORT) {
            return switch (op) {
                .And => z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs }),
                .Or => z3.Z3_mk_or(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs }),
                .Xor => z3.Z3_mk_xor(self.context.ctx, lhs, rhs),
            };
        }

        // If mixed types (one bool, one bitvec), convert bool to bitvec
        var lhs_bv = lhs;
        var rhs_bv = rhs;
        if (lhs_kind == z3.Z3_BOOL_SORT and rhs_kind == z3.Z3_BV_SORT) {
            // Convert lhs bool to bitvec of same width as rhs
            const width = z3.Z3_get_bv_sort_size(self.context.ctx, rhs_sort);
            const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, z3.Z3_mk_bv_sort(self.context.ctx, width));
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, z3.Z3_mk_bv_sort(self.context.ctx, width));
            lhs_bv = z3.Z3_mk_ite(self.context.ctx, lhs, one, zero);
        } else if (rhs_kind == z3.Z3_BOOL_SORT and lhs_kind == z3.Z3_BV_SORT) {
            // Convert rhs bool to bitvec of same width as lhs
            const width = z3.Z3_get_bv_sort_size(self.context.ctx, lhs_sort);
            const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, z3.Z3_mk_bv_sort(self.context.ctx, width));
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, z3.Z3_mk_bv_sort(self.context.ctx, width));
            rhs_bv = z3.Z3_mk_ite(self.context.ctx, rhs, one, zero);
        }

        return switch (op) {
            .And => z3.Z3_mk_bvand(self.context.ctx, lhs_bv, rhs_bv),
            .Or => z3.Z3_mk_bvor(self.context.ctx, lhs_bv, rhs_bv),
            .Xor => z3.Z3_mk_bvxor(self.context.ctx, lhs_bv, rhs_bv),
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
                    return err;
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
        const refinement_base = mlir.oraRefinementTypeGetBaseType(mlir_type);
        if (!mlir.oraTypeIsNull(refinement_base)) {
            return self.encodeMLIRType(refinement_base);
        }
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

        const success_type = mlir.oraErrorUnionTypeGetSuccessType(mlir_type);
        if (!mlir.oraTypeIsNull(success_type)) {
            const eu = try self.getErrorUnionSort(mlir_type, success_type);
            return eu.sort;
        }

        const map_value_type = mlir.oraMapTypeGetValueType(mlir_type);
        if (!mlir.oraTypeIsNull(map_value_type)) {
            const value_sort = try self.encodeMLIRType(map_value_type);
            // Get the actual key type from the map (e.g., address is 160 bits, not 256)
            const map_key_type = mlir.oraMapTypeGetKeyType(mlir_type);
            const key_sort = if (!mlir.oraTypeIsNull(map_key_type))
                try self.encodeMLIRType(map_key_type)
            else
                self.mkBitVectorSort(256); // fallback to 256-bit
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

        if (std.mem.eql(u8, op_name, "ora.cmp")) {
            if (operands.len < 2) return error.InvalidOperandCount;
            const predicate = self.getStringAttr(mlir_op, "predicate") orelse "eq";
            if (std.mem.eql(u8, predicate, "eq")) return self.encodeComparisonOp(.Eq, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "ne")) return self.encodeComparisonOp(.Ne, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "lt") or std.mem.eql(u8, predicate, "ult")) return self.encodeComparisonOp(.Lt, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "le") or std.mem.eql(u8, predicate, "ule")) return self.encodeComparisonOp(.Le, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "gt") or std.mem.eql(u8, predicate, "ugt")) return self.encodeComparisonOp(.Gt, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "ge") or std.mem.eql(u8, predicate, "uge")) return self.encodeComparisonOp(.Ge, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "slt")) return z3.Z3_mk_bvslt(self.context.ctx, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "sle")) return z3.Z3_mk_bvsle(self.context.ctx, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "sgt")) return z3.Z3_mk_bvsgt(self.context.ctx, operands[0], operands[1]);
            if (std.mem.eql(u8, predicate, "sge")) return z3.Z3_mk_bvsge(self.context.ctx, operands[0], operands[1]);
            return error.UnsupportedPredicate;
        }

        // constant operations
        if (std.mem.eql(u8, op_name, "arith.constant")) {
            const value_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("value", 5));
            const value_type = mlir.oraOperationGetResult(mlir_op, 0);
            const mlir_type = mlir.oraValueGetType(value_type);
            const is_addr = mlir.oraTypeIsAddressType(mlir_type);
            const value = self.parseConstAttrValue(value_attr) orelse if (is_addr) 0 else self.getConstantValue(mlir_op);
            const width: u32 = if (is_addr)
                160
            else if (mlir.oraTypeIsAInteger(mlir_type))
                @intCast(mlir.oraIntegerTypeGetWidth(mlir_type))
            else
                256;
            return try self.encodeConstantOp(value, width);
        }

        if (std.mem.eql(u8, op_name, "ora.const")) {
            const value_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("value", 5));
            const value = self.parseConstAttrValue(value_attr) orelse return error.UnsupportedOperation;
            const value_type = mlir.oraOperationGetResult(mlir_op, 0);
            const mlir_type = mlir.oraValueGetType(value_type);
            const width: u32 = if (mlir.oraTypeIsAInteger(mlir_type))
                @intCast(mlir.oraIntegerTypeGetWidth(mlir_type))
            else if (mlir.oraTypeIsAddressType(mlir_type))
                160
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

        if (std.mem.eql(u8, op_name, "arith.bitcast")) {
            if (operands.len < 1) return error.InvalidOperandCount;
            return operands[0];
        }

        if (std.mem.eql(u8, op_name, "arith.extui") or
            std.mem.eql(u8, op_name, "arith.extsi") or
            std.mem.eql(u8, op_name, "arith.trunci") or
            std.mem.eql(u8, op_name, "arith.index_cast") or
            std.mem.eql(u8, op_name, "arith.index_castui") or
            std.mem.eql(u8, op_name, "arith.index_castsi"))
        {
            return try self.encodeUnaryIntCast(mlir_op, op_name, operands);
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
            const global_name = self.getStringAttr(mlir_op, "global") orelse return error.UnsupportedOperation;
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results < 1) return error.UnsupportedOperation;
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_type = mlir.oraValueGetType(result_value);
            const result_sort = try self.encodeMLIRType(result_type);
            return try self.getOrCreateGlobal(global_name, result_sort);
        }

        if (std.mem.eql(u8, op_name, "ora.sstore")) {
            if (operands.len < 1) return error.InvalidOperandCount;
            const global_name = self.getStringAttr(mlir_op, "global") orelse return error.UnsupportedOperation;
            const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
            const operand_type = mlir.oraValueGetType(operand_value);
            const operand_sort = try self.encodeMLIRType(operand_type);
            _ = try self.getOrCreateGlobal(global_name, operand_sort);
            return operands[0];
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

        if (std.mem.eql(u8, op_name, "ora.refinement_to_base")) {
            if (operands.len >= 1) {
                return operands[0];
            }
        }

        if (std.mem.eql(u8, op_name, "ora.base_to_refinement")) {
            if (operands.len >= 1) {
                return operands[0];
            }
        }

        if (std.mem.eql(u8, op_name, "func.call")) {
            return try self.encodeFuncCall(mlir_op, operands);
        }

        if (std.mem.eql(u8, op_name, "ora.error.unwrap") or
            std.mem.eql(u8, op_name, "ora.error.ok") or
            std.mem.eql(u8, op_name, "ora.error.err") or
            std.mem.eql(u8, op_name, "ora.error.is_error") or
            std.mem.eql(u8, op_name, "ora.error.get_error"))
        {
            return try self.encodeErrorUnionOp(op_name, operands, mlir_op);
        }

        if (std.mem.eql(u8, op_name, "ora.i160.to.addr")) {
            if (operands.len >= 1) {
                return operands[0];
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

        if (std.mem.eql(u8, op_name, "memref.alloca")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "memref_alloca", op_id);
            }
        }

        if (std.mem.eql(u8, op_name, "memref.load")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "memref_load", op_id);
            }
        }

        // unsupported operation - caller may treat as unknown symbol.
        std.debug.print("z3 encoder unsupported op: {s}\n", .{op_name});
        return error.UnsupportedOperation;
    }

    const ErrorUnionSort = struct {
        sort: z3.Z3_sort,
        ctor: z3.Z3_func_decl,
        proj_is_error: z3.Z3_func_decl,
        proj_ok: z3.Z3_func_decl,
        proj_err: z3.Z3_func_decl,
        ok_sort: z3.Z3_sort,
        err_sort: z3.Z3_sort,
    };

    fn getErrorUnionSort(self: *Encoder, mlir_type: mlir.MlirType, success_type: mlir.MlirType) EncodeError!ErrorUnionSort {
        const key = @intFromPtr(mlir_type.ptr);
        if (self.error_union_sorts.get(key)) |cached| {
            return cached;
        }

        const ok_sort = try self.encodeMLIRType(success_type);
        const err_sort = self.mkBitVectorSort(256);

        const tuple_name = try std.fmt.allocPrint(self.allocator, "ora.error_union.{d}", .{key});
        defer self.allocator.free(tuple_name);
        const tuple_symbol = try self.mkSymbol(tuple_name);

        const field_names = [_][]const u8{ "is_error", "ok", "err" };
        var field_symbols: [3]z3.Z3_symbol = undefined;
        field_symbols[0] = try self.mkSymbol(field_names[0]);
        field_symbols[1] = try self.mkSymbol(field_names[1]);
        field_symbols[2] = try self.mkSymbol(field_names[2]);
        const field_sorts = [_]z3.Z3_sort{
            z3.Z3_mk_bool_sort(self.context.ctx),
            ok_sort,
            err_sort,
        };

        var ctor: z3.Z3_func_decl = undefined;
        var projections: [3]z3.Z3_func_decl = undefined;
        const sort = z3.Z3_mk_tuple_sort(
            self.context.ctx,
            tuple_symbol,
            3,
            &field_symbols,
            &field_sorts,
            &ctor,
            &projections,
        );

        const eu = ErrorUnionSort{
            .sort = sort,
            .ctor = ctor,
            .proj_is_error = projections[0],
            .proj_ok = projections[1],
            .proj_err = projections[2],
            .ok_sort = ok_sort,
            .err_sort = err_sort,
        };
        try self.error_union_sorts.put(key, eu);
        return eu;
    }

    fn encodeErrorUnionOp(self: *Encoder, op_name: []const u8, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        if (operands.len < 1) return error.InvalidOperandCount;
        const op_id = @intFromPtr(mlir_op.ptr);

        if (std.mem.eql(u8, op_name, "ora.error.ok")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_type = mlir.oraValueGetType(result_value);
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(result_type);
            if (mlir.oraTypeIsNull(success_type)) return error.UnsupportedOperation;
            const eu = try self.getErrorUnionSort(result_type, success_type);
            const is_err = self.encodeBoolConstant(false);
            const err_val = try self.mkUndefValue(eu.err_sort, "err", op_id);
            return z3.Z3_mk_app(self.context.ctx, eu.ctor, 3, &[_]z3.Z3_ast{ is_err, operands[0], err_val });
        }

        if (std.mem.eql(u8, op_name, "ora.error.err")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_type = mlir.oraValueGetType(result_value);
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(result_type);
            if (mlir.oraTypeIsNull(success_type)) return error.UnsupportedOperation;
            const eu = try self.getErrorUnionSort(result_type, success_type);
            const is_err = self.encodeBoolConstant(true);
            const ok_val = try self.mkUndefValue(eu.ok_sort, "ok", op_id);
            return z3.Z3_mk_app(self.context.ctx, eu.ctor, 3, &[_]z3.Z3_ast{ is_err, ok_val, operands[0] });
        }

        const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
        const operand_type = mlir.oraValueGetType(operand_value);
        const success_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
        if (mlir.oraTypeIsNull(success_type)) return error.UnsupportedOperation;
        const eu = try self.getErrorUnionSort(operand_type, success_type);

        const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{operands[0]});

        if (std.mem.eql(u8, op_name, "ora.error.is_error")) {
            return is_err;
        }

        if (std.mem.eql(u8, op_name, "ora.error.unwrap")) {
            const ok_val = z3.Z3_mk_app(self.context.ctx, eu.proj_ok, 1, &[_]z3.Z3_ast{operands[0]});
            self.addConstraint(z3.Z3_mk_not(self.context.ctx, is_err));
            return ok_val;
        }

        if (std.mem.eql(u8, op_name, "ora.error.get_error")) {
            const err_val = z3.Z3_mk_app(self.context.ctx, eu.proj_err, 1, &[_]z3.Z3_ast{operands[0]});
            self.addConstraint(is_err);
            return err_val;
        }

        return error.UnsupportedOperation;
    }

    fn encodeFuncCall(self: *Encoder, mlir_op: mlir.MlirOperation, operands: []const z3.Z3_ast) EncodeError!z3.Z3_ast {
        const num_results = mlir.oraOperationGetNumResults(mlir_op);
        if (num_results < 1) return error.UnsupportedOperation;
        const result_value = mlir.oraOperationGetResult(mlir_op, 0);
        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);

        const callee = self.getStringAttr(mlir_op, "callee") orelse "call";
        const op_id = @intFromPtr(mlir_op.ptr);
        const name = try std.fmt.allocPrint(self.allocator, "{s}_{d}", .{ callee, op_id });
        defer self.allocator.free(name);
        const symbol = try self.mkSymbol(name);

        var domain = try self.allocator.alloc(z3.Z3_sort, operands.len);
        defer self.allocator.free(domain);
        for (operands, 0..) |opnd, i| {
            domain[i] = z3.Z3_get_sort(self.context.ctx, opnd);
        }

        const func_decl = z3.Z3_mk_func_decl(self.context.ctx, symbol, @intCast(operands.len), domain.ptr, result_sort);
        return z3.Z3_mk_app(self.context.ctx, func_decl, @intCast(operands.len), operands.ptr);
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

    fn parseConstAttrValue(_: *Encoder, attr: mlir.MlirAttribute) ?u256 {
        if (mlir.oraAttributeIsNull(attr)) return null;
        const str = mlir.oraStringAttrGetValue(attr);
        if (str.data != null and str.length > 0) {
            const slice = str.data[0..str.length];
            if (slice.len >= 2 and slice[0] == '0' and (slice[1] == 'x' or slice[1] == 'X')) {
                return std.fmt.parseUnsigned(u256, slice[2..], 16) catch null;
            }
            return std.fmt.parseUnsigned(u256, slice, 10) catch null;
        }
        const int_value = mlir.oraIntegerAttrGetValueSInt(attr);
        if (int_value < 0) {
            if (int_value == -1) return 1;
            const u64_value: u64 = @bitCast(@as(i64, int_value));
            return @intCast(u64_value);
        }
        return @intCast(@as(i64, int_value));
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
            return error.UnsupportedOperation;
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

    fn getTypeBitWidth(_: *Encoder, ty: mlir.MlirType) ?u32 {
        if (mlir.oraTypeIsIntegerType(ty)) {
            const builtin = mlir.oraTypeToBuiltin(ty);
            return @intCast(mlir.oraIntegerTypeGetWidth(builtin));
        }
        if (mlir.oraTypeIsAInteger(ty)) {
            return @intCast(mlir.oraIntegerTypeGetWidth(ty));
        }
        return null;
    }

    fn encodeUnaryIntCast(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        op_name: []const u8,
        operands: []const z3.Z3_ast,
    ) !z3.Z3_ast {
        if (operands.len < 1) return error.InvalidOperandCount;
        const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
        const operand_type = mlir.oraValueGetType(operand_value);
        const result_value = mlir.oraOperationGetResult(mlir_op, 0);
        const result_type = mlir.oraValueGetType(result_value);

        const in_width = self.getTypeBitWidth(operand_type) orelse return operands[0];
        const out_width = self.getTypeBitWidth(result_type) orelse return operands[0];
        if (in_width == out_width) return operands[0];

        // Convert Bool sort to BitVec<1> if needed (Z3_mk_zero_ext/sign_ext require bitvector operands)
        var operand = operands[0];
        const sort = z3.Z3_get_sort(self.context.ctx, operand);
        const kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (kind == z3.Z3_BOOL_SORT) {
            // Convert bool to bitvector<1>: ite(bool, bv1(1), bv1(0))
            const bv1_sort = self.mkBitVectorSort(1);
            const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, bv1_sort);
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, bv1_sort);
            operand = z3.Z3_mk_ite(self.context.ctx, operand, one, zero);
        }

        if (out_width > in_width) {
            const extend = out_width - in_width;
            const is_signed = std.mem.eql(u8, op_name, "arith.extsi") or std.mem.eql(u8, op_name, "arith.index_castsi");
            return if (is_signed)
                z3.Z3_mk_sign_ext(self.context.ctx, extend, operand)
            else
                z3.Z3_mk_zero_ext(self.context.ctx, extend, operand);
        }

        const high: u32 = out_width - 1;
        return z3.Z3_mk_extract(self.context.ctx, high, 0, operand);
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
