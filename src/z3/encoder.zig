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
    const CallSlotState = struct {
        name: []const u8,
        pre: z3.Z3_ast,
        sort: z3.Z3_sort,
        is_write: bool,
        post: ?z3.Z3_ast = null,
    };

    const TensorDimKey = struct {
        source_value_id: u64,
        dim_value_id: u64,
        is_old: bool,
    };

    const QuantifiedBinding = struct {
        name: []u8,
        ast: z3.Z3_ast,
    };

    context: *Context,
    allocator: std.mem.Allocator,
    verify_calls: bool,
    verify_state: bool,
    max_summary_inline_depth: u32,

    /// Map from MLIR value to Z3 AST (for caching encoded values)
    value_map: std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    /// Cache for old() encodings of MLIR values (separate from current-state encodings).
    value_map_old: std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    /// Explicit MLIR value bindings (used for call summary argument substitution).
    value_bindings: std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    /// Map from global storage name to Z3 AST (for consistent storage symbols)
    global_map: std.StringHashMap(z3.Z3_ast),
    /// Map from global storage name to Z3 AST (for old() storage symbols)
    global_old_map: std.StringHashMap(z3.Z3_ast),
    /// Snapshot of global storage symbols at function entry.
    global_entry_map: std.StringHashMap(z3.Z3_ast),
    /// Map from environment symbol name (e.g. msg.sender) to Z3 AST.
    env_map: std.StringHashMap(z3.Z3_ast),
    /// Scalar memref local state threaded during verification extraction.
    memref_map: std.AutoHashMap(u64, z3.Z3_ast),
    /// Scalar memref local state for old() mode.
    memref_old_map: std.AutoHashMap(u64, z3.Z3_ast),
    /// Stable symbolic dimensions for tensor.dim/memref.dim on dynamic shapes.
    tensor_dim_map: std.AutoHashMap(TensorDimKey, z3.Z3_ast),
    /// Map from function symbol name to MLIR function operation.
    function_ops: std.StringHashMap(mlir.MlirOperation),
    /// Map from struct symbol name to comma-separated field names in declaration order.
    struct_field_names_csv: std.StringHashMap([]u8),
    /// Calls already materialized for current-state summary (avoid double state transitions).
    materialized_calls: std.AutoHashMap(u64, void),
    /// Active summary inlining stack for recursion guard.
    inline_function_stack: std.ArrayList([]u8),
    /// Keep Z3 symbol names alive for the life of the encoder.
    string_storage: std.ArrayList([]u8),
    /// Pending constraints emitted during encoding (e.g., error.unwrap validity).
    pending_constraints: std.ArrayList(z3.Z3_ast),
    /// Pending safety obligations emitted during encoding (e.g., div-by-zero, overflow).
    pending_obligations: std.ArrayList(z3.Z3_ast),
    /// Cache of error_union tuple sorts by MLIR type pointer.
    error_union_sorts: std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    /// Stack of active quantified variable bindings (innermost binding last).
    quantified_bindings: std.ArrayList(QuantifiedBinding),

    pub fn init(context: *Context, allocator: std.mem.Allocator) Encoder {
        return .{
            .context = context,
            .allocator = allocator,
            .verify_calls = true,
            .verify_state = true,
            .max_summary_inline_depth = 8,
            .value_map = std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .value_map_old = std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .value_bindings = std.HashMap(u64, z3.Z3_ast, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .global_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .global_old_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .global_entry_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .env_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .memref_map = std.AutoHashMap(u64, z3.Z3_ast).init(allocator),
            .memref_old_map = std.AutoHashMap(u64, z3.Z3_ast).init(allocator),
            .tensor_dim_map = std.AutoHashMap(TensorDimKey, z3.Z3_ast).init(allocator),
            .function_ops = std.StringHashMap(mlir.MlirOperation).init(allocator),
            .struct_field_names_csv = std.StringHashMap([]u8).init(allocator),
            .materialized_calls = std.AutoHashMap(u64, void).init(allocator),
            .inline_function_stack = std.ArrayList([]u8){},
            .string_storage = std.ArrayList([]u8){},
            .pending_constraints = std.ArrayList(z3.Z3_ast){},
            .pending_obligations = std.ArrayList(z3.Z3_ast){},
            .error_union_sorts = std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .quantified_bindings = std.ArrayList(QuantifiedBinding){},
        };
    }

    pub fn setVerifyCalls(self: *Encoder, enabled: bool) void {
        self.verify_calls = enabled;
    }

    pub fn setVerifyState(self: *Encoder, enabled: bool) void {
        self.verify_state = enabled;
    }

    pub fn resetFunctionState(self: *Encoder) void {
        var g_it = self.global_map.iterator();
        while (g_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_map.clearRetainingCapacity();

        var old_it = self.global_old_map.iterator();
        while (old_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_old_map.clearRetainingCapacity();

        var entry_it = self.global_entry_map.iterator();
        while (entry_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_entry_map.clearRetainingCapacity();

        var env_it = self.env_map.iterator();
        while (env_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.env_map.clearRetainingCapacity();

        self.memref_map.clearRetainingCapacity();
        self.memref_old_map.clearRetainingCapacity();
        self.tensor_dim_map.clearRetainingCapacity();

        self.value_map.clearRetainingCapacity();
        self.value_map_old.clearRetainingCapacity();
        self.value_bindings.clearRetainingCapacity();
        self.materialized_calls.clearRetainingCapacity();
        for (self.inline_function_stack.items) |fn_name| {
            self.allocator.free(fn_name);
        }
        self.inline_function_stack.clearRetainingCapacity();
        for (self.quantified_bindings.items) |binding| {
            self.allocator.free(binding.name);
        }
        self.quantified_bindings.clearRetainingCapacity();
    }

    pub fn deinit(self: *Encoder) void {
        var it = self.global_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_map.deinit();
        var old_it = self.global_old_map.iterator();
        while (old_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_old_map.deinit();
        var entry_it = self.global_entry_map.iterator();
        while (entry_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.global_entry_map.deinit();
        var env_it = self.env_map.iterator();
        while (env_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.env_map.deinit();
        self.memref_map.deinit();
        self.memref_old_map.deinit();
        self.tensor_dim_map.deinit();
        var fn_it = self.function_ops.iterator();
        while (fn_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.function_ops.deinit();
        var struct_it = self.struct_field_names_csv.iterator();
        while (struct_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.struct_field_names_csv.deinit();
        self.materialized_calls.deinit();
        for (self.inline_function_stack.items) |fn_name| {
            self.allocator.free(fn_name);
        }
        self.inline_function_stack.deinit(self.allocator);
        for (self.string_storage.items) |buf| {
            self.allocator.free(buf);
        }
        self.string_storage.deinit(self.allocator);
        self.pending_constraints.deinit(self.allocator);
        self.pending_obligations.deinit(self.allocator);
        self.error_union_sorts.deinit();
        for (self.quantified_bindings.items) |binding| {
            self.allocator.free(binding.name);
        }
        self.quantified_bindings.deinit(self.allocator);
        self.value_map.deinit();
        self.value_map_old.deinit();
        self.value_bindings.deinit();
    }

    pub fn registerFunctionOperation(self: *Encoder, func_op: mlir.MlirOperation) !void {
        const name = self.getStringAttr(func_op, "sym_name") orelse return;
        if (self.function_ops.contains(name)) return;
        const key = try self.allocator.dupe(u8, name);
        try self.function_ops.put(key, func_op);
    }

    pub fn registerStructDeclOperation(self: *Encoder, struct_op: mlir.MlirOperation) !void {
        const name = self.getStringAttr(struct_op, "sym_name") orelse
            self.getStringAttr(struct_op, "name") orelse return;
        if (self.struct_field_names_csv.contains(name)) return;

        const fields_attr = mlir.oraOperationGetAttributeByName(struct_op, mlir.oraStringRefCreate("ora.field_names", 15));
        var csv_builder = std.ArrayList(u8){};
        defer csv_builder.deinit(self.allocator);
        if (!mlir.oraAttributeIsNull(fields_attr)) {
            const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(fields_attr));
            for (0..count) |i| {
                const field_attr = mlir.oraArrayAttrGetElement(fields_attr, i);
                if (mlir.oraAttributeIsNull(field_attr)) continue;
                const field_ref = mlir.oraStringAttrGetValue(field_attr);
                if (field_ref.data == null or field_ref.length == 0) continue;
                if (csv_builder.items.len > 0) try csv_builder.append(self.allocator, ',');
                try csv_builder.appendSlice(self.allocator, field_ref.data[0..field_ref.length]);
            }
        }

        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        const value = try self.allocator.dupe(u8, csv_builder.items);
        errdefer self.allocator.free(value);
        try self.struct_field_names_csv.put(key, value);
    }

    fn copyFunctionRegistryFrom(self: *Encoder, other: *const Encoder) !void {
        var it = other.function_ops.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (self.function_ops.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            try self.function_ops.put(key, entry.value_ptr.*);
        }
    }

    fn copyStructRegistryFrom(self: *Encoder, other: *const Encoder) !void {
        var it = other.struct_field_names_csv.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (self.struct_field_names_csv.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            const value = try self.allocator.dupe(u8, entry.value_ptr.*);
            try self.struct_field_names_csv.put(key, value);
        }
    }

    fn copyInlineStackFrom(self: *Encoder, other: *const Encoder) !void {
        for (other.inline_function_stack.items) |fn_name| {
            try self.inline_function_stack.append(self.allocator, try self.allocator.dupe(u8, fn_name));
        }
    }

    fn copyEnvMapFrom(self: *Encoder, other: *const Encoder) !void {
        var it = other.env_map.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (self.env_map.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            try self.env_map.put(key, entry.value_ptr.*);
        }
    }

    fn copyGlobalStateMapFrom(self: *Encoder, other: *const Encoder, use_old: bool) !void {
        const source = if (use_old) &other.global_old_map else &other.global_map;
        const destination = if (use_old) &self.global_old_map else &self.global_map;
        var it = source.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (destination.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            try destination.put(key, entry.value_ptr.*);
        }
    }

    fn copyGlobalEntryMapFrom(self: *Encoder, other: *const Encoder) !void {
        var it = other.global_entry_map.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (self.global_entry_map.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            try self.global_entry_map.put(key, entry.value_ptr.*);
        }
    }

    fn inlineStackContains(self: *const Encoder, name: []const u8) bool {
        for (self.inline_function_stack.items) |fn_name| {
            if (std.mem.eql(u8, fn_name, name)) return true;
        }
        return false;
    }

    fn pushInlineFunction(self: *Encoder, name: []const u8) !void {
        try self.inline_function_stack.append(self.allocator, try self.allocator.dupe(u8, name));
    }

    fn pushQuantifiedBinding(self: *Encoder, name: []const u8, ast: z3.Z3_ast) !void {
        try self.quantified_bindings.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, name),
            .ast = ast,
        });
    }

    fn popQuantifiedBinding(self: *Encoder) void {
        const popped = self.quantified_bindings.pop() orelse return;
        self.allocator.free(popped.name);
    }

    fn lookupQuantifiedBinding(self: *const Encoder, name: []const u8, expected_sort: z3.Z3_sort) ?z3.Z3_ast {
        var idx = self.quantified_bindings.items.len;
        while (idx > 0) : (idx -= 1) {
            const binding = self.quantified_bindings.items[idx - 1];
            if (!std.mem.eql(u8, binding.name, name)) continue;
            const binding_sort = z3.Z3_get_sort(self.context.ctx, binding.ast);
            if (binding_sort == expected_sort) return binding.ast;
        }
        return null;
    }

    pub fn bindValue(self: *Encoder, value: mlir.MlirValue, ast: z3.Z3_ast) !void {
        const value_id = @intFromPtr(value.ptr);
        try self.value_bindings.put(value_id, ast);
    }

    pub fn encodeValueOld(self: *Encoder, mlir_value: mlir.MlirValue) EncodeError!z3.Z3_ast {
        return self.encodeValueWithMode(mlir_value, .Old);
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

    /// Get or create a global storage symbol for "old" values
    fn getOrCreateOldGlobal(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.global_old_map.get(name)) |existing| {
            return existing;
        }

        const prefixed = try std.fmt.allocPrint(self.allocator, "old_{s}", .{name});
        defer self.allocator.free(prefixed);
        const symbol = try self.mkVariable(prefixed, sort);
        const key = try self.allocator.dupe(u8, name);
        try self.global_old_map.put(key, symbol);

        // old(x) is the entry-state value of x; tie it to the entry snapshot.
        const entry_symbol = try self.getOrCreateGlobalEntry(name, sort);
        const eq = z3.Z3_mk_eq(self.context.ctx, symbol, entry_symbol);
        self.addConstraint(eq);
        return symbol;
    }

    fn getOrCreateGlobalEntry(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.global_entry_map.get(name)) |existing| {
            return existing;
        }

        // If current state doesn't exist yet, create a dedicated entry symbol.
        // getOrCreateGlobal() will reuse this symbol as the initial current state.
        const entry_name = try std.fmt.allocPrint(self.allocator, "g_entry_{s}", .{name});
        defer self.allocator.free(entry_name);
        const entry_symbol = try self.mkVariable(entry_name, sort);
        const key = try self.allocator.dupe(u8, name);
        try self.global_entry_map.put(key, entry_symbol);
        return entry_symbol;
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

    fn getValueConstUnsigned(self: *Encoder, value: mlir.MlirValue, width: u32) ?u256 {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;

        const name_ref = mlir.oraOperationGetName(owner);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (!std.mem.eql(u8, op_name, "arith.constant") and !std.mem.eql(u8, op_name, "ora.const")) {
            return null;
        }

        const value_attr = mlir.oraOperationGetAttributeByName(
            owner,
            mlir.oraStringRefCreate("value".ptr, "value".len),
        );
        return self.parseConstAttrValue(value_attr, width);
    }

    fn encodeUnsignedToSort(self: *Encoder, value: u64, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        const value_str = try std.fmt.allocPrint(self.allocator, "{d}", .{value});
        defer self.allocator.free(value_str);
        const value_z = try self.allocator.dupeZ(u8, value_str);
        defer self.allocator.free(value_z);
        return z3.Z3_mk_numeral(self.context.ctx, value_z.ptr, sort);
    }

    fn getOrCreateShapedDimSymbol(self: *Encoder, key: TensorDimKey, result_sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.tensor_dim_map.get(key)) |existing| {
            return existing;
        }

        const mode_tag = if (key.is_old) "old" else "cur";
        const name = try std.fmt.allocPrint(
            self.allocator,
            "shape_dim_{d}_{d}_{s}",
            .{ key.source_value_id, key.dim_value_id, mode_tag },
        );
        defer self.allocator.free(name);
        const dim_ast = try self.mkVariable(name, result_sort);
        try self.tensor_dim_map.put(key, dim_ast);
        return dim_ast;
    }

    fn encodeShapedDimOp(self: *Encoder, mlir_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const num_results = mlir.oraOperationGetNumResults(mlir_op);
        if (num_results < 1) return error.UnsupportedOperation;
        const result_value = mlir.oraOperationGetResult(mlir_op, 0);
        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);

        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        if (num_operands < 1) return error.InvalidOperandCount;

        const source_value = mlir.oraOperationGetOperand(mlir_op, 0);
        var dim_value_id: u64 = 0;
        var dim_index: ?u256 = null;
        if (num_operands >= 2) {
            const dim_value = mlir.oraOperationGetOperand(mlir_op, 1);
            dim_value_id = @intFromPtr(dim_value.ptr);
            dim_index = self.getValueConstUnsigned(dim_value, 64);
        }

        const source_type = mlir.oraValueGetType(source_value);
        if (!mlir.oraTypeIsNull(source_type) and mlir.oraTypeIsAShaped(source_type)) {
            if (dim_index) |idx| {
                const rank_i64 = mlir.oraShapedTypeGetRank(source_type);
                const rank_u64: u64 = if (rank_i64 > 0) @intCast(rank_i64) else 0;
                if (idx < rank_u64) {
                    const dim_axis: i64 = @intCast(idx);
                    const dim_size = mlir.oraShapedTypeGetDimSize(source_type, dim_axis);
                    if (dim_size >= 0) {
                        return try self.encodeUnsignedToSort(@intCast(dim_size), result_sort);
                    }
                }
            }
        }

        const key = TensorDimKey{
            .source_value_id = @intFromPtr(source_value.ptr),
            .dim_value_id = dim_value_id,
            .is_old = mode == .Old,
        };
        return try self.getOrCreateShapedDimSymbol(key, result_sort);
    }

    fn addConstraint(self: *Encoder, constraint: z3.Z3_ast) void {
        self.pending_constraints.append(self.allocator, constraint) catch {};
    }

    fn addObligation(self: *Encoder, obligation: z3.Z3_ast) void {
        self.pending_obligations.append(self.allocator, obligation) catch {};
    }

    pub fn takeConstraints(self: *Encoder, allocator: std.mem.Allocator) ![]z3.Z3_ast {
        if (self.pending_constraints.items.len == 0) return &[_]z3.Z3_ast{};
        const slice = try allocator.dupe(z3.Z3_ast, self.pending_constraints.items);
        self.pending_constraints.clearRetainingCapacity();
        return slice;
    }

    pub fn takeObligations(self: *Encoder, allocator: std.mem.Allocator) ![]z3.Z3_ast {
        if (self.pending_obligations.items.len == 0) return &[_]z3.Z3_ast{};
        const slice = try allocator.dupe(z3.Z3_ast, self.pending_obligations.items);
        self.pending_obligations.clearRetainingCapacity();
        return slice;
    }

    fn getOrCreateGlobal(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.global_map.get(name)) |existing| {
            return existing;
        }

        // If old(name) was encoded first, reuse the entry snapshot as the
        // initial current-state value.
        if (self.global_entry_map.get(name)) |entry_symbol| {
            const key = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(key);
            try self.global_map.put(key, entry_symbol);
            return entry_symbol;
        }

        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);

        const global_name = try std.fmt.allocPrint(self.allocator, "g_{s}", .{name});
        defer self.allocator.free(global_name);
        const ast = try self.mkVariable(global_name, sort);
        try self.global_map.put(key, ast);

        const entry_key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(entry_key);
        try self.global_entry_map.put(entry_key, ast);
        return ast;
    }

    fn getOrCreateEnv(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        if (self.env_map.get(name)) |existing| {
            const existing_sort = z3.Z3_get_sort(self.context.ctx, existing);
            self.addEnvironmentConstraints(name, existing, existing_sort);
            return existing;
        }
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);

        const env_name = try std.fmt.allocPrint(self.allocator, "env_{s}", .{name});
        defer self.allocator.free(env_name);
        const ast = try self.mkVariable(env_name, sort);
        self.addEnvironmentConstraints(name, ast, sort);
        try self.env_map.put(key, ast);
        return ast;
    }

    fn addEnvironmentConstraints(self: *Encoder, name: []const u8, ast: z3.Z3_ast, sort: z3.Z3_sort) void {
        // EVM caller is always a non-zero address in runtime semantics.
        if (std.mem.eql(u8, name, "evm_caller")) {
            self.addNonZeroBitVectorConstraint(ast, sort);
        }
    }

    fn addNonZeroBitVectorConstraint(self: *Encoder, ast: z3.Z3_ast, sort: z3.Z3_sort) void {
        if (z3.Z3_get_sort_kind(self.context.ctx, sort) != z3.Z3_BV_SORT) return;
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const non_zero = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, ast, zero));
        self.addConstraint(non_zero);
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
            .DivUnsigned => z3.Z3_mk_bv_udiv(self.context.ctx, lhs, rhs),
            .RemUnsigned => z3.Z3_mk_bv_urem(self.context.ctx, lhs, rhs),
            .DivSigned => z3.Z3_mk_bvsdiv(self.context.ctx, lhs, rhs),
            .RemSigned => z3.Z3_mk_bvsrem(self.context.ctx, lhs, rhs),
        };
    }

    /// Arithmetic operation types
    pub const ArithmeticOp = enum {
        Add,
        Sub,
        Mul,
        DivUnsigned,
        RemUnsigned,
        DivSigned,
        RemSigned,
    };

    fn emitArithmeticSafetyObligations(self: *Encoder, op: ArithmeticOp, lhs: z3.Z3_ast, rhs: z3.Z3_ast) void {
        switch (op) {
            // Division/remainder: always check divisor != 0
            .DivUnsigned, .DivSigned, .RemUnsigned, .RemSigned => {
                const non_zero_divisor = z3.Z3_mk_not(self.context.ctx, self.checkDivByZero(rhs));
                self.addObligation(non_zero_divisor);
            },
            // For ora.add/sub/mul (old-style ops without ora.assert), add overflow obligations.
            // For arith.addi/subi/muli, overflow obligations come from ora.assert ops instead.
            .Add => {
                const no_overflow = z3.Z3_mk_not(self.context.ctx, self.checkAddOverflow(lhs, rhs));
                self.addObligation(no_overflow);
            },
            .Sub => {
                const no_underflow = z3.Z3_mk_not(self.context.ctx, self.checkSubUnderflow(lhs, rhs));
                self.addObligation(no_underflow);
            },
            .Mul => {
                const no_overflow = z3.Z3_mk_not(self.context.ctx, self.checkMulOverflow(lhs, rhs));
                self.addObligation(no_overflow);
            },
        }
    }

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
        // Obligation: shift amount must be < bit_width (undefined behavior otherwise)
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const sort_kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (sort_kind == z3.Z3_BV_SORT) {
            const width = z3.Z3_get_bv_sort_size(self.context.ctx, sort);
            const max_shift = z3.Z3_mk_unsigned_int64(self.context.ctx, width, sort);
            const in_range = z3.Z3_mk_bvult(self.context.ctx, rhs, max_shift);
            self.addObligation(in_range);
        }
        return switch (op) {
            .Shl => z3.Z3_mk_bvshl(self.context.ctx, lhs, rhs),
            .ShrSigned => z3.Z3_mk_bvashr(self.context.ctx, lhs, rhs),
            .ShrUnsigned => z3.Z3_mk_bvlshr(self.context.ctx, lhs, rhs),
        };
    }

    /// Encode modular exponentiation over bitvectors using exponentiation by squaring.
    /// This is exact for EVM-style arithmetic: all intermediate results wrap to bit-width.
    pub fn encodePowerOp(self: *Encoder, base_in: z3.Z3_ast, exponent_in: z3.Z3_ast) EncodeError!z3.Z3_ast {
        const base_sort = z3.Z3_get_sort(self.context.ctx, base_in);
        if (z3.Z3_get_sort_kind(self.context.ctx, base_sort) != z3.Z3_BV_SORT) {
            return error.UnsupportedOperation;
        }

        const width = z3.Z3_get_bv_sort_size(self.context.ctx, base_sort);
        const base = self.coerceAstToSort(base_in, base_sort);
        const exponent = self.coerceAstToSort(exponent_in, base_sort);

        const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, base_sort);
        const bit_sort = self.mkBitVectorSort(1);
        const bit_one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, bit_sort);

        var result = one;
        var factor = base;
        var bit_index: u32 = 0;
        while (bit_index < width) : (bit_index += 1) {
            const bit = z3.Z3_mk_extract(self.context.ctx, bit_index, bit_index, exponent);
            const bit_is_set = z3.Z3_mk_eq(self.context.ctx, bit, bit_one);
            const multiplied = z3.Z3_mk_bv_mul(self.context.ctx, result, factor);
            result = z3.Z3_mk_ite(self.context.ctx, bit_is_set, multiplied, result);
            factor = z3.Z3_mk_bv_mul(self.context.ctx, factor, factor);
        }

        return result;
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
        // Unsigned overflow check:
        // overflow iff lhs != 0 and ((lhs * rhs) / lhs) != rhs (in modular arithmetic).
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const lhs_non_zero = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, lhs, zero));
        const product = z3.Z3_mk_bv_mul(self.context.ctx, lhs, rhs);
        const recovered_rhs = z3.Z3_mk_bv_udiv(self.context.ctx, product, lhs);
        const matches_rhs = z3.Z3_mk_eq(self.context.ctx, recovered_rhs, rhs);
        const overflow_if_non_zero = z3.Z3_mk_not(self.context.ctx, matches_rhs);
        return z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs_non_zero, overflow_if_non_zero });
    }

    /// Check for signed overflow in addition.
    /// Overflow iff operands have same sign and result has different sign:
    ///   ((result ^ a) & (result ^ b))[MSB] == 1
    pub fn checkSignedAddOverflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const ctx = self.context.ctx;
        const result = z3.Z3_mk_bv_add(ctx, lhs, rhs);
        const xor_ra = z3.Z3_mk_bvxor(ctx, result, lhs);
        const xor_rb = z3.Z3_mk_bvxor(ctx, result, rhs);
        const both = z3.Z3_mk_bvand(ctx, xor_ra, xor_rb);
        const sort = z3.Z3_get_sort(ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(ctx, 0, sort);
        return z3.Z3_mk_bvslt(ctx, both, zero); // MSB set → overflow
    }

    /// Check for signed overflow in subtraction.
    /// Overflow iff operands have different sign and result has different sign from lhs:
    ///   ((a ^ b) & (result ^ a))[MSB] == 1
    pub fn checkSignedSubOverflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const ctx = self.context.ctx;
        const result = z3.Z3_mk_bv_sub(ctx, lhs, rhs);
        const xor_ab = z3.Z3_mk_bvxor(ctx, lhs, rhs);
        const xor_ra = z3.Z3_mk_bvxor(ctx, result, lhs);
        const both = z3.Z3_mk_bvand(ctx, xor_ab, xor_ra);
        const sort = z3.Z3_get_sort(ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(ctx, 0, sort);
        return z3.Z3_mk_bvslt(ctx, both, zero);
    }

    /// Check for signed overflow in multiplication.
    /// Overflow iff b != 0 && sdiv(a*b, b) != a, with special case for MIN_INT * -1.
    pub fn checkSignedMulOverflow(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const ctx = self.context.ctx;
        const sort = z3.Z3_get_sort(ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(ctx, 0, sort);
        const rhs_nz = z3.Z3_mk_not(ctx, z3.Z3_mk_eq(ctx, rhs, zero));
        const product = z3.Z3_mk_bv_mul(ctx, lhs, rhs);
        const recovered = z3.Z3_mk_bvsdiv(ctx, product, rhs);
        const mismatch = z3.Z3_mk_not(ctx, z3.Z3_mk_eq(ctx, recovered, lhs));
        return z3.Z3_mk_and(ctx, 2, &[_]z3.Z3_ast{ rhs_nz, mismatch });
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
        const lhs_sort = z3.Z3_get_sort(self.context.ctx, lhs);
        if (z3.Z3_is_string_sort(self.context.ctx, lhs_sort)) {
            return switch (op) {
                .Eq => z3.Z3_mk_eq(self.context.ctx, lhs, rhs),
                .Ne => z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, lhs, rhs)),
                .Lt => z3.Z3_mk_str_lt(self.context.ctx, lhs, rhs),
                .Le => z3.Z3_mk_str_le(self.context.ctx, lhs, rhs),
                .Gt => z3.Z3_mk_str_lt(self.context.ctx, rhs, lhs),
                .Ge => z3.Z3_mk_str_le(self.context.ctx, rhs, lhs),
            };
        }
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

    fn coerceAstToSort(self: *Encoder, ast: z3.Z3_ast, target_sort: z3.Z3_sort) z3.Z3_ast {
        const src_sort = z3.Z3_get_sort(self.context.ctx, ast);
        if (src_sort == target_sort) return ast;

        const src_kind = z3.Z3_get_sort_kind(self.context.ctx, src_sort);
        const dst_kind = z3.Z3_get_sort_kind(self.context.ctx, target_sort);

        if (src_kind == z3.Z3_BOOL_SORT and dst_kind == z3.Z3_BV_SORT) {
            const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, target_sort);
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, target_sort);
            return z3.Z3_mk_ite(self.context.ctx, ast, one, zero);
        }

        if (src_kind == z3.Z3_BV_SORT and dst_kind == z3.Z3_BOOL_SORT) {
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, src_sort);
            return z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, ast, zero));
        }

        if (src_kind == z3.Z3_BV_SORT and dst_kind == z3.Z3_BV_SORT) {
            const src_width = z3.Z3_get_bv_sort_size(self.context.ctx, src_sort);
            const dst_width = z3.Z3_get_bv_sort_size(self.context.ctx, target_sort);
            if (src_width == dst_width) return ast;
            if (src_width < dst_width) {
                return z3.Z3_mk_zero_ext(self.context.ctx, dst_width - src_width, ast);
            }
            return z3.Z3_mk_extract(self.context.ctx, dst_width - 1, 0, ast);
        }

        return ast;
    }

    fn encodeTensorExtractOp(self: *Encoder, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        if (operands.len < 2) return error.InvalidOperandCount;

        var current = operands[0];
        for (operands[1..]) |raw_index| {
            const current_sort = z3.Z3_get_sort(self.context.ctx, current);
            if (!self.isArraySort(current_sort)) return error.UnsupportedOperation;
            const index_sort = z3.Z3_get_array_sort_domain(self.context.ctx, current_sort);
            const index = self.coerceAstToSort(raw_index, index_sort);
            current = self.encodeSelect(current, index);
        }

        const num_results = mlir.oraOperationGetNumResults(mlir_op);
        if (num_results < 1) return current;
        const result_value = mlir.oraOperationGetResult(mlir_op, 0);
        const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
        return self.coerceAstToSort(current, result_sort);
    }

    fn encodeTensorInsertOp(self: *Encoder, operands: []const z3.Z3_ast) EncodeError!z3.Z3_ast {
        // tensor.insert %value into %tensor[%i, ...]
        // operands: value, tensor, idx0, idx1, ...
        if (operands.len < 3) return error.InvalidOperandCount;

        const value = operands[0];
        const tensor = operands[1];
        const indices = operands[2..];

        var containers = try self.allocator.alloc(z3.Z3_ast, indices.len);
        defer self.allocator.free(containers);
        var cast_indices = try self.allocator.alloc(z3.Z3_ast, indices.len);
        defer self.allocator.free(cast_indices);

        var cursor = tensor;
        for (indices, 0..) |raw_index, i| {
            const container_sort = z3.Z3_get_sort(self.context.ctx, cursor);
            if (!self.isArraySort(container_sort)) return error.UnsupportedOperation;

            containers[i] = cursor;
            const index_sort = z3.Z3_get_array_sort_domain(self.context.ctx, container_sort);
            cast_indices[i] = self.coerceAstToSort(raw_index, index_sort);

            if (i + 1 < indices.len) {
                cursor = self.encodeSelect(cursor, cast_indices[i]);
            }
        }

        const leaf_container = containers[indices.len - 1];
        const leaf_sort = z3.Z3_get_sort(self.context.ctx, leaf_container);
        const leaf_range = z3.Z3_get_array_sort_range(self.context.ctx, leaf_sort);
        var updated = self.encodeStore(
            leaf_container,
            cast_indices[indices.len - 1],
            self.coerceAstToSort(value, leaf_range),
        );

        var depth = indices.len - 1;
        while (depth > 0) {
            depth -= 1;
            const parent = containers[depth];
            const parent_sort = z3.Z3_get_sort(self.context.ctx, parent);
            const parent_range = z3.Z3_get_array_sort_range(self.context.ctx, parent_sort);
            updated = self.encodeStore(parent, cast_indices[depth], self.coerceAstToSort(updated, parent_range));
        }

        return updated;
    }

    /// Emit quantified frame condition for a single array store:
    /// forall k. k != index -> select(post, k) == select(pre, k)
    fn addArrayStoreFrameConstraint(self: *Encoder, pre: z3.Z3_ast, index: z3.Z3_ast, post: z3.Z3_ast, op_id: u64) EncodeError!void {
        const key_sort = z3.Z3_get_sort(self.context.ctx, index);
        const key_name = try std.fmt.allocPrint(self.allocator, "frame_k_{d}", .{op_id});
        defer self.allocator.free(key_name);
        const key = try self.mkVariable(key_name, key_sort);

        const neq_key = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, key, index));
        const pre_val = z3.Z3_mk_select(self.context.ctx, pre, key);
        const post_val = z3.Z3_mk_select(self.context.ctx, post, key);
        const eq_val = z3.Z3_mk_eq(self.context.ctx, post_val, pre_val);
        const body = z3.Z3_mk_implies(self.context.ctx, neq_key, eq_val);

        var bounds = [_]z3.Z3_app{z3.Z3_to_app(self.context.ctx, key)};
        const frame = z3.Z3_mk_forall_const(self.context.ctx, 0, 1, &bounds, 0, null, body);
        self.addConstraint(frame);
    }

    fn isArraySort(self: *Encoder, sort: z3.Z3_sort) bool {
        return z3.Z3_get_sort_kind(self.context.ctx, sort) == z3.Z3_ARRAY_SORT;
    }

    /// Emit quantified frame equality for array-typed state:
    /// forall k. select(post, k) == select(pre, k)
    fn addArrayEqualityFrameConstraint(self: *Encoder, pre: z3.Z3_ast, post: z3.Z3_ast, frame_id: u64) EncodeError!void {
        const pre_sort = z3.Z3_get_sort(self.context.ctx, pre);
        if (!self.isArraySort(pre_sort)) return;
        const key_sort = z3.Z3_get_array_sort_domain(self.context.ctx, pre_sort);
        const key_name = try std.fmt.allocPrint(self.allocator, "frame_eq_k_{d}", .{frame_id});
        defer self.allocator.free(key_name);
        const key = try self.mkVariable(key_name, key_sort);

        const pre_val = z3.Z3_mk_select(self.context.ctx, pre, key);
        const post_val = z3.Z3_mk_select(self.context.ctx, post, key);
        const eq_val = z3.Z3_mk_eq(self.context.ctx, post_val, pre_val);

        var bounds = [_]z3.Z3_app{z3.Z3_to_app(self.context.ctx, key)};
        const frame = z3.Z3_mk_forall_const(self.context.ctx, 0, 1, &bounds, 0, null, eq_val);
        self.addConstraint(frame);
    }

    //===----------------------------------------------------------------------===//
    // mlir Operation Encoding
    //===----------------------------------------------------------------------===//

    const EncodeMode = enum { Current, Old };

    /// Encode an MLIR operation to Z3 AST
    /// This is the main entry point for encoding MLIR operations
    pub fn encodeOperation(self: *Encoder, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        return self.encodeOperationWithMode(mlir_op, .Current);
    }

    fn encodeOperationWithMode(self: *Encoder, mlir_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!z3.Z3_ast {
        // get operation name
        const op_name_ref = mlir.oraOperationGetName(mlir_op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        // Quantified ops must establish bindings before encoding operands.
        if (std.mem.eql(u8, op_name, "ora.quantified")) {
            return try self.encodeQuantifiedOp(mlir_op, mode);
        }

        // get number of operands
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);

        // encode operands recursively
        const num_ops: usize = @intCast(num_operands);
        var operands = try self.allocator.alloc(z3.Z3_ast, num_ops);
        defer self.allocator.free(operands);

        for (0..num_ops) |i| {
            const operand_value = mlir.oraOperationGetOperand(mlir_op, @intCast(i));
            operands[i] = try self.encodeValueWithMode(operand_value, mode);
        }

        // dispatch based on operation name
        return try self.dispatchOperation(op_name, operands, mlir_op, mode);
    }

    /// Encode an MLIR value to Z3 AST (with caching)
    pub fn encodeValue(self: *Encoder, mlir_value: mlir.MlirValue) EncodeError!z3.Z3_ast {
        return self.encodeValueWithMode(mlir_value, .Current);
    }

    /// Encode an MLIR value to Z3 AST (with caching), using the provided mode.
    fn encodeValueWithMode(self: *Encoder, mlir_value: mlir.MlirValue, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const value_id = @intFromPtr(mlir_value.ptr);
        if (self.value_bindings.get(value_id)) |bound| {
            return bound;
        }

        // check cache first
        if (mode == .Current) {
            if (self.value_map.get(value_id)) |cached| {
                return cached;
            }
        } else {
            if (self.value_map_old.get(value_id)) |cached| {
                return cached;
            }
        }

        if (mlir.oraValueIsAOpResult(mlir_value)) {
            const defining_op = mlir.oraOpResultGetOwner(mlir_value);
            if (!mlir.oraOperationIsNull(defining_op)) {
                const result_index = self.getResultIndex(defining_op, mlir_value) orelse 0;
                const encoded = try self.encodeOperationResultWithMode(defining_op, result_index, mode);
                if (mode == .Current) {
                    try self.value_map.put(value_id, encoded);
                } else {
                    try self.value_map_old.put(value_id, encoded);
                }
                return encoded;
            }
        }

        // if no defining operation, create a fresh variable
        const value_type = mlir.oraValueGetType(mlir_value);
        const sort = try self.encodeMLIRType(value_type);
        const var_name = try std.fmt.allocPrint(self.allocator, "v_{d}", .{value_id});
        defer self.allocator.free(var_name);
        const encoded = try self.mkVariable(var_name, sort);
        try self.addBlockArgumentConstraints(mlir_value, encoded, mode);

        // cache the result
        // For block arguments, keep a single symbol shared across modes.
        try self.value_map.put(value_id, encoded);
        return encoded;
    }

    fn getResultIndex(_: *Encoder, op: mlir.MlirOperation, value: mlir.MlirValue) ?u32 {
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(op));
        var i: u32 = 0;
        while (i < num_results) : (i += 1) {
            const candidate = mlir.oraOperationGetResult(op, @intCast(i));
            if (candidate.ptr == value.ptr) return i;
        }
        return null;
    }

    fn addBlockArgumentConstraints(self: *Encoder, value: mlir.MlirValue, ast: z3.Z3_ast, mode: EncodeMode) EncodeError!void {
        if (!mlir.mlirValueIsABlockArgument(value)) return;

        const owner_block = mlir.mlirBlockArgumentGetOwner(value);
        if (mlir.oraBlockIsNull(owner_block)) return;
        const parent_op = mlir.mlirBlockGetParentOperation(owner_block);
        if (mlir.oraOperationIsNull(parent_op)) return;

        const op_name_ref = mlir.oraOperationGetName(parent_op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        // scf.for body arg0 is induction variable. Constrain it by loop bounds.
        if (std.mem.eql(u8, op_name, "scf.for")) {
            const arg_no = mlir.mlirBlockArgumentGetArgNumber(value);
            if (arg_no != 0) return;

            const num_operands = mlir.oraOperationGetNumOperands(parent_op);
            if (num_operands < 2) return;
            const lower_bound = mlir.oraOperationGetOperand(parent_op, 0);
            const upper_bound = mlir.oraOperationGetOperand(parent_op, 1);

            const lb_ast = try self.encodeValueWithMode(lower_bound, mode);
            const ub_ast = try self.encodeValueWithMode(upper_bound, mode);
            const unsigned_cmp = self.getScfForUnsignedCmp(parent_op);
            const lower_ok = if (unsigned_cmp)
                z3.Z3_mk_bvuge(self.context.ctx, ast, lb_ast)
            else
                z3.Z3_mk_bvsge(self.context.ctx, ast, lb_ast);
            const upper_ok = if (unsigned_cmp)
                z3.Z3_mk_bvult(self.context.ctx, ast, ub_ast)
            else
                z3.Z3_mk_bvslt(self.context.ctx, ast, ub_ast);
            self.addConstraint(lower_ok);
            self.addConstraint(upper_ok);
        }
    }

    fn getScfForUnsignedCmp(self: *Encoder, loop_op: mlir.MlirOperation) bool {
        const printed = mlir.oraOperationPrintToString(loop_op);
        defer if (printed.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(printed);
        };

        if (printed.data != null and printed.length > 0) {
            const text = printed.data[0..printed.length];
            if (std.mem.indexOf(u8, text, "unsignedCmp = true") != null) return true;
            if (std.mem.indexOf(u8, text, "unsignedCmp = false") != null) return false;
        }

        const unsigned_attr = mlir.oraOperationGetAttributeByName(
            loop_op,
            mlir.oraStringRefCreate("unsignedCmp".ptr, "unsignedCmp".len),
        );
        if (mlir.oraAttributeIsNull(unsigned_attr)) return false;

        _ = self;
        return mlir.oraIntegerAttrGetValueSInt(unsigned_attr) != 0;
    }

    fn encodeOperationResultWithMode(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!z3.Z3_ast {
        const op_name_ref = mlir.oraOperationGetName(mlir_op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        if (std.mem.eql(u8, op_name, "scf.if")) {
            return try self.encodeScfIfResult(mlir_op, result_index, mode);
        }

        if (std.mem.eql(u8, op_name, "scf.while") or
            std.mem.eql(u8, op_name, "scf.for") or
            std.mem.eql(u8, op_name, "scf.execute_region"))
        {
            const operands = try self.encodeOperationOperandsWithMode(mlir_op, mode);
            defer self.allocator.free(operands);
            return try self.encodeStructuredControlResult(mlir_op, op_name, operands, result_index);
        }

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            const operands = try self.encodeOperationOperandsWithMode(mlir_op, mode);
            defer self.allocator.free(operands);
            return try self.encodeFuncCallResult(mlir_op, operands, result_index, mode);
        }

        if (result_index == 0) {
            return self.encodeOperationWithMode(mlir_op, mode);
        }

        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(mlir_op));
        if (result_index >= num_results) return error.UnsupportedOperation;
        const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);
        const op_id = @intFromPtr(mlir_op.ptr);
        const label = try std.fmt.allocPrint(self.allocator, "op_result_{d}", .{result_index});
        defer self.allocator.free(label);
        return try self.mkUndefValue(result_sort, label, op_id);
    }

    fn encodeScfIfResult(self: *Encoder, mlir_op: mlir.MlirOperation, result_index: u32, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        if (num_operands < 1) return error.InvalidOperandCount;
        const condition_value = mlir.oraOperationGetOperand(mlir_op, 0);
        const condition = try self.encodeValueWithMode(condition_value, mode);
        const then_expr = try self.extractIfYield(mlir_op, 0, result_index, mode);
        const else_expr = try self.extractIfYield(mlir_op, 1, result_index, mode);
        return try self.encodeControlFlow("scf.if", condition, then_expr, else_expr);
    }

    fn encodeOperationOperandsWithMode(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError![]z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        const operand_count: usize = @intCast(num_operands);
        var operands = try self.allocator.alloc(z3.Z3_ast, operand_count);
        for (0..operand_count) |i| {
            const operand_value = mlir.oraOperationGetOperand(mlir_op, @intCast(i));
            operands[i] = try self.encodeValueWithMode(operand_value, mode);
        }
        return operands;
    }

    fn encodeStructuredControlResult(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        op_name: []const u8,
        operands: []const z3.Z3_ast,
        result_index: u32,
    ) EncodeError!z3.Z3_ast {
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(mlir_op));
        if (num_results == 0) {
            return self.encodeBoolConstant(true);
        }
        if (result_index >= num_results) return error.InvalidOperandCount;
        const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);

        const op_id = @intFromPtr(mlir_op.ptr);
        const fn_name = try std.fmt.allocPrint(self.allocator, "{s}_summary_{d}_{d}", .{ op_name, op_id, result_index });
        defer self.allocator.free(fn_name);
        const symbol = try self.mkSymbol(fn_name);

        var domain = try self.allocator.alloc(z3.Z3_sort, operands.len);
        defer self.allocator.free(domain);
        for (operands, 0..) |opnd, i| {
            domain[i] = z3.Z3_get_sort(self.context.ctx, opnd);
        }
        const func_decl = z3.Z3_mk_func_decl(self.context.ctx, symbol, @intCast(operands.len), domain.ptr, result_sort);
        return z3.Z3_mk_app(self.context.ctx, func_decl, @intCast(operands.len), operands.ptr);
    }

    /// Encode MLIR type to Z3 sort
    pub fn encodeMLIRType(self: *Encoder, mlir_type: mlir.MlirType) EncodeError!z3.Z3_sort {
        if (mlir.oraTypeIsNull(mlir_type)) return error.UnsupportedOperation;

        const refinement_base = mlir.oraRefinementTypeGetBaseType(mlir_type);
        if (!mlir.oraTypeIsNull(refinement_base)) {
            return self.encodeMLIRType(refinement_base);
        }
        if (mlir.oraTypeIsAddressType(mlir_type)) {
            return self.mkBitVectorSort(160);
        }
        {
            const type_ctx = mlir.mlirTypeGetContext(mlir_type);
            const string_ty = mlir.oraStringTypeGet(type_ctx);
            if (!mlir.oraTypeIsNull(string_ty) and mlir.oraTypeEqual(mlir_type, string_ty)) {
                return z3.Z3_mk_string_sort(self.context.ctx);
            }
            const bytes_ty = mlir.oraBytesTypeGet(type_ctx);
            if (!mlir.oraTypeIsNull(bytes_ty) and mlir.oraTypeEqual(mlir_type, bytes_ty)) {
                // Model dynamic bytes as canonicalized hex strings in SMT.
                return z3.Z3_mk_string_sort(self.context.ctx);
            }
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

        if (mlir.oraTypeIsAShaped(mlir_type)) {
            const elem_type = mlir.oraShapedTypeGetElementType(mlir_type);
            if (mlir.oraTypeIsNull(elem_type)) return self.mkBitVectorSort(256);

            var sort = try self.encodeMLIRType(elem_type);
            const rank_i = mlir.oraShapedTypeGetRank(mlir_type);
            if (rank_i <= 0) return sort;

            const rank: usize = @intCast(rank_i);
            const index_sort = self.mkBitVectorSort(256);
            for (0..rank) |_| {
                sort = self.mkArraySort(index_sort, sort);
            }
            return sort;
        }

        // default to 256-bit bitvector for EVM
        return self.mkBitVectorSort(256);
    }

    /// Dispatch operation to appropriate encoder based on operation name
    fn dispatchOperation(self: *Encoder, op_name: []const u8, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!z3.Z3_ast {
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
            if (self.getStringAttr(mlir_op, "ora.bound_variable")) |bound_var_name| {
                const num_results = mlir.oraOperationGetNumResults(mlir_op);
                if (num_results < 1) return error.UnsupportedOperation;
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const sort = try self.encodeMLIRType(result_type);
                if (self.lookupQuantifiedBinding(bound_var_name, sort)) |bound| {
                    return bound;
                }
                return try self.mkVariable(bound_var_name, sort);
            }
            const value_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("value", 5));
            const value_type = mlir.oraOperationGetResult(mlir_op, 0);
            const mlir_type = mlir.oraValueGetType(value_type);
            const is_addr = mlir.oraTypeIsAddressType(mlir_type);
            const width: u32 = if (is_addr)
                160
            else if (mlir.oraTypeIsAInteger(mlir_type))
                @intCast(mlir.oraIntegerTypeGetWidth(mlir_type))
            else
                256;
            const value = self.parseConstAttrValue(value_attr, width) orelse if (is_addr) 0 else self.getConstantValue(mlir_op, width);
            return try self.encodeConstantOp(value, width);
        }

        if (std.mem.eql(u8, op_name, "ora.const")) {
            const value_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("value", 5));
            const value_type = mlir.oraOperationGetResult(mlir_op, 0);
            const mlir_type = mlir.oraValueGetType(value_type);
            const width: u32 = if (mlir.oraTypeIsAInteger(mlir_type))
                @intCast(mlir.oraIntegerTypeGetWidth(mlir_type))
            else if (mlir.oraTypeIsAddressType(mlir_type))
                160
            else
                256;
            const value = self.parseConstAttrValue(value_attr, width) orelse return error.UnsupportedOperation;
            return try self.encodeConstantOp(value, width);
        }

        if (std.mem.eql(u8, op_name, "ora.string.constant")) {
            const value = self.getStringAttr(mlir_op, "value") orelse return error.UnsupportedOperation;
            const value_z = try self.allocator.dupeZ(u8, value);
            defer self.allocator.free(value_z);
            return z3.Z3_mk_string(self.context.ctx, value_z.ptr);
        }

        if (std.mem.eql(u8, op_name, "ora.bytes.constant")) {
            const value = self.getStringAttr(mlir_op, "value") orelse return error.UnsupportedOperation;
            const canonical = try self.normalizeBytesHexLiteral(value);
            defer self.allocator.free(canonical);
            const value_z = try self.allocator.dupeZ(u8, canonical);
            defer self.allocator.free(value_z);
            return z3.Z3_mk_string(self.context.ctx, value_z.ptr);
        }

        if (std.mem.eql(u8, op_name, "ora.old")) {
            const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
            if (num_operands < 1) return error.InvalidOperandCount;
            const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
            return self.encodeValueWithMode(operand_value, .Old);
        }

        if (std.mem.eql(u8, op_name, "ora.quantified")) {
            return try self.encodeQuantifiedOp(mlir_op, mode);
        }

        if (std.mem.eql(u8, op_name, "ora.variable_placeholder")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results < 1) return error.UnsupportedOperation;
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_type = mlir.oraValueGetType(result_value);
            const sort = try self.encodeMLIRType(result_type);
            const name = self.getStringAttr(mlir_op, "name") orelse "placeholder";
            if (self.lookupQuantifiedBinding(name, sort)) |bound| {
                return bound;
            }
            return try self.mkVariable(name, sort);
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
            return try self.encodeArithOp(op_name, operands, mlir_op);
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
                ArithmeticOp.DivUnsigned
            else
                ArithmeticOp.RemUnsigned;
            self.emitArithmeticSafetyObligations(arith_op, operands[0], operands[1]);
            return self.encodeArithmeticOp(arith_op, operands[0], operands[1]);
        }

        if (std.mem.eql(u8, op_name, "ora.power")) {
            if (operands.len < 2) return error.InvalidOperandCount;
            return try self.encodePowerOp(operands[0], operands[1]);
        }

        // storage operations
        if (std.mem.eql(u8, op_name, "ora.sload")) {
            const global_name = self.getStringAttr(mlir_op, "global") orelse return error.UnsupportedOperation;
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results < 1) return error.UnsupportedOperation;
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_type = mlir.oraValueGetType(result_value);
            const result_sort = try self.encodeMLIRType(result_type);
            if (!self.verify_state) {
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "sload", op_id);
            }
            if (mode == .Old) {
                return try self.getOrCreateOldGlobal(global_name, result_sort);
            }
            return try self.getOrCreateGlobal(global_name, result_sort);
        }

        if (std.mem.eql(u8, op_name, "ora.sstore")) {
            if (operands.len < 1) return error.InvalidOperandCount;
            if (!self.verify_state) {
                return operands[0];
            }
            const global_name = self.getStringAttr(mlir_op, "global") orelse return error.UnsupportedOperation;
            const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
            const operand_type = mlir.oraValueGetType(operand_value);
            const operand_sort = try self.encodeMLIRType(operand_type);
            if (mode == .Old) {
                _ = try self.getOrCreateOldGlobal(global_name, operand_sort);
            } else if (self.global_map.getPtr(global_name)) |existing| {
                existing.* = operands[0];
            } else {
                const key = try self.allocator.dupe(u8, global_name);
                try self.global_map.put(key, operands[0]);
            }
            return operands[0];
        }

        if (std.mem.eql(u8, op_name, "ora.map_get")) {
            if (operands.len >= 2) {
                if (!self.verify_state) {
                    const num_results = mlir.oraOperationGetNumResults(mlir_op);
                    if (num_results < 1) return error.UnsupportedOperation;
                    const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                    const result_type = mlir.oraValueGetType(result_value);
                    const result_sort = try self.encodeMLIRType(result_type);
                    const op_id = @intFromPtr(mlir_op.ptr);
                    return try self.mkUndefValue(result_sort, "map_get", op_id);
                }
                const map_sort = z3.Z3_get_sort(self.context.ctx, operands[0]);
                if (!self.isArraySort(map_sort)) return error.UnsupportedOperation;
                const key_sort = z3.Z3_get_array_sort_domain(self.context.ctx, map_sort);
                const key = self.coerceAstToSort(operands[1], key_sort);
                return self.encodeSelect(operands[0], key);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.map_store")) {
            if (operands.len >= 3) {
                if (!self.verify_state) {
                    const num_results = mlir.oraOperationGetNumResults(mlir_op);
                    if (num_results > 0) return operands[2];
                    return self.encodeBoolConstant(true);
                }
                const map_sort = z3.Z3_get_sort(self.context.ctx, operands[0]);
                if (!self.isArraySort(map_sort)) return error.UnsupportedOperation;
                const key_sort = z3.Z3_get_array_sort_domain(self.context.ctx, map_sort);
                const value_sort = z3.Z3_get_array_sort_range(self.context.ctx, map_sort);
                const key = self.coerceAstToSort(operands[1], key_sort);
                const value = self.coerceAstToSort(operands[2], value_sort);
                const stored = self.encodeStore(operands[0], key, value);
                const op_id = @intFromPtr(mlir_op.ptr);
                self.addArrayStoreFrameConstraint(operands[0], key, stored, op_id) catch {};
                if (mode == .Current) {
                    const map_operand = mlir.oraOperationGetOperand(mlir_op, 0);
                    const map_operand_id = @intFromPtr(map_operand.ptr);
                    try self.value_bindings.put(map_operand_id, stored);
                    try self.value_map.put(map_operand_id, stored);
                    if (self.resolveGlobalNameFromMapOperand(map_operand)) |global_name| {
                        if (self.global_map.getPtr(global_name)) |existing| {
                            existing.* = stored;
                        } else {
                            const global_key = try self.allocator.dupe(u8, global_name);
                            try self.global_map.put(global_key, stored);
                        }
                    }
                }
                const num_results = mlir.oraOperationGetNumResults(mlir_op);
                if (num_results > 0) return stored;
                return self.encodeBoolConstant(true);
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

        // ---- ora.assert: encode condition as a verification obligation ----
        // Handles checked-arithmetic overflow assertions, runtime guards, etc.
        if (std.mem.eql(u8, op_name, "ora.assert")) {
            if (operands.len >= 1) {
                const condition = self.coerceToBool(operands[0]);
                self.addObligation(condition);
                return condition;
            }
        }

        // ---- ora.assume: encode condition as a known-true assumption ----
        // Handles bitfield read bounds, control-flow-derived facts, etc.
        if (std.mem.eql(u8, op_name, "ora.assume")) {
            if (operands.len >= 1) {
                const condition = self.coerceToBool(operands[0]);
                self.addConstraint(condition);
                return condition;
            }
        }

        // ---- ora.struct_instantiate: model as a constrained constructor ----
        // Uses struct declaration field order to tie field extracts to operands.
        if (std.mem.eql(u8, op_name, "ora.struct_instantiate")) {
            if (mlir.oraOperationGetNumResults(mlir_op) >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                const struct_var_name = try std.fmt.allocPrint(self.allocator, "struct_instantiate_{d}", .{op_id});
                defer self.allocator.free(struct_var_name);
                const struct_val = try self.mkVariable(struct_var_name, result_sort);

                var field_names_str = self.getStringAttr(mlir_op, "ora.field_names");
                if (field_names_str == null) {
                    if (self.getStringAttr(mlir_op, "struct_name")) |struct_name| {
                        field_names_str = self.struct_field_names_csv.get(struct_name);
                    }
                }

                for (operands, 0..) |operand, i| {
                    const field_attr_name = try self.resolveFieldNameForIndex(field_names_str, i);
                    defer self.allocator.free(field_attr_name);
                    const field_sort = z3.Z3_get_sort(self.context.ctx, operand);
                    const accessor = try self.applyFieldFunction(field_attr_name, result_sort, field_sort, struct_val);
                    const eq = z3.Z3_mk_eq(self.context.ctx, accessor, operand);
                    self.addConstraint(eq);
                }
                return struct_val;
            }
        }

        // ---- ora.struct_init: model as an uninterpreted constructor ----
        // Used by @addWithOverflow etc. to return (value, overflow) tuples.
        if (std.mem.eql(u8, op_name, "ora.struct_init")) {
            if (mlir.oraOperationGetNumResults(mlir_op) >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                // Build a struct-like value; bind each operand as a named field
                // so ora.struct_field_extract can recover them.
                const struct_var_name = try std.fmt.allocPrint(self.allocator, "struct_init_{d}", .{op_id});
                defer self.allocator.free(struct_var_name);
                const struct_val = try self.mkVariable(struct_var_name, result_sort);
                // Use field names from attribute if available, otherwise fall back to field_N
                const field_names_str = self.getStringAttr(mlir_op, "ora.field_names");
                for (operands, 0..) |operand, i| {
                    const field_attr_name = try self.resolveFieldNameForIndex(field_names_str, i);
                    defer self.allocator.free(field_attr_name);
                    const field_sort = z3.Z3_get_sort(self.context.ctx, operand);
                    const accessor = try self.applyFieldFunction(field_attr_name, result_sort, field_sort, struct_val);
                    const eq = z3.Z3_mk_eq(self.context.ctx, accessor, operand);
                    self.addConstraint(eq);
                }
                return struct_val;
            }
        }

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            return try self.encodeFuncCall(mlir_op, operands, mode);
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
                const then_expr = try self.extractIfYield(mlir_op, 0, 0, mode);
                const else_expr = try self.extractIfYield(mlir_op, 1, 0, mode);
                return try self.encodeControlFlow(op_name, condition, then_expr, else_expr);
            }
        }

        if (std.mem.eql(u8, op_name, "scf.while") or
            std.mem.eql(u8, op_name, "scf.for") or
            std.mem.eql(u8, op_name, "scf.execute_region"))
        {
            return try self.encodeStructuredControlResult(mlir_op, op_name, operands, 0);
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            // Conservative summary for try/catch-style control flow.
            // Result values are modeled as structured-control summaries.
            return try self.encodeStructuredControlResult(mlir_op, op_name, operands, 0);
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

        if (std.mem.eql(u8, op_name, "memref.store")) {
            // Track scalar local memref state (rank-0 alloca pattern used by lowered locals).
            if (operands.len < 2) return error.InvalidOperandCount;
            const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
            if (num_operands == 2) {
                const memref_value = mlir.oraOperationGetOperand(mlir_op, 1);
                const memref_id = @intFromPtr(memref_value.ptr);
                if (mode == .Old) {
                    try self.memref_old_map.put(memref_id, operands[0]);
                } else {
                    try self.memref_map.put(memref_id, operands[0]);
                }
            }
            return operands[0];
        }

        if (std.mem.eql(u8, op_name, "memref.load")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);

                // Scalar memref load: read from tracked local state if present.
                const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
                if (num_operands == 1) {
                    const memref_value = mlir.oraOperationGetOperand(mlir_op, 0);
                    const memref_id = @intFromPtr(memref_value.ptr);
                    const map = if (mode == .Old) &self.memref_old_map else &self.memref_map;
                    if (map.get(memref_id)) |stored| {
                        return stored;
                    }

                    const op_id_unknown = @intFromPtr(mlir_op.ptr);
                    const fresh = try self.mkUndefValue(result_sort, "memref_load", op_id_unknown);
                    try map.put(memref_id, fresh);
                    return fresh;
                }

                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "memref_load", op_id);
            }
        }

        if (std.mem.eql(u8, op_name, "tensor.dim") or std.mem.eql(u8, op_name, "memref.dim")) {
            return try self.encodeShapedDimOp(mlir_op, mode);
        }

        if (std.mem.eql(u8, op_name, "tensor.insert")) {
            return try self.encodeTensorInsertOp(operands);
        }

        if (std.mem.eql(u8, op_name, "tensor.extract")) {
            return try self.encodeTensorExtractOp(operands, mlir_op);
        }

        if (std.mem.eql(u8, op_name, "ora.evm.origin")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "evm_origin", op_id);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.evm.caller")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                return try self.getOrCreateEnv("evm_caller", result_sort);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.evm.timestamp")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "evm_timestamp", op_id);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.tload")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const op_id = @intFromPtr(mlir_op.ptr);
                return try self.mkUndefValue(result_sort, "tload", op_id);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.tstore")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results == 0) {
                return self.encodeBoolConstant(true);
            }
        }

        // unsupported operation - fail verification/compilation immediately.
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

    fn encodeFuncCall(self: *Encoder, mlir_op: mlir.MlirOperation, operands: []const z3.Z3_ast, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(mlir_op));
        if (num_results == 0) {
            if (mode == .Current and self.verify_calls) {
                try self.materializeCallSummaryCurrent(mlir_op, operands);
            }
            return self.encodeBoolConstant(true);
        }
        return self.encodeFuncCallResult(mlir_op, operands, 0, mode);
    }

    fn encodeFuncCallResult(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        operands: []const z3.Z3_ast,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!z3.Z3_ast {
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(mlir_op));
        if (num_results < 1 or result_index >= num_results) return error.UnsupportedOperation;

        const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
        const result_id = @intFromPtr(result_value.ptr);

        if (mode == .Current) {
            if (self.value_map.get(result_id)) |cached| {
                return cached;
            }
            if (self.verify_calls) {
                try self.materializeCallSummaryCurrent(mlir_op, operands);
                if (self.value_map.get(result_id)) |cached_after| {
                    return cached_after;
                }
            }
        }

        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);
        if (!self.verify_calls) {
            const op_id = @intFromPtr(mlir_op.ptr);
            const label = try std.fmt.allocPrint(self.allocator, "call_r{d}", .{result_index});
            defer self.allocator.free(label);
            return try self.mkUndefValue(result_sort, label, op_id);
        }

        const owned_callee = try self.resolveCalleeName(mlir_op);
        defer if (owned_callee) |name| self.allocator.free(name);
        const callee = owned_callee orelse "call";
        if (mode == .Old) {
            if (self.function_ops.get(callee)) |func_op| {
                if (try self.tryInlinePureCallResult(callee, func_op, operands, result_index, mode)) |inlined| {
                    return inlined;
                }
            }
        }
        return try self.encodeCallResultUFSymbol(callee, operands, &[_]CallSlotState{}, result_index, result_sort);
    }

    fn encodeCallResultUFSymbol(
        self: *Encoder,
        callee: []const u8,
        operands: []const z3.Z3_ast,
        slots: []const CallSlotState,
        result_index: u32,
        result_sort: z3.Z3_sort,
    ) EncodeError!z3.Z3_ast {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(callee);
        var idx_copy = result_index;
        hasher.update(std.mem.asBytes(&idx_copy));
        const result_sort_id = @intFromPtr(result_sort);
        hasher.update(std.mem.asBytes(&result_sort_id));

        for (operands) |opnd| {
            const sort_ptr = @intFromPtr(z3.Z3_get_sort(self.context.ctx, opnd));
            hasher.update(std.mem.asBytes(&sort_ptr));
        }
        for (slots) |slot| {
            hasher.update(slot.name);
            const slot_sort_ptr = @intFromPtr(slot.sort);
            hasher.update(std.mem.asBytes(&slot_sort_ptr));
        }
        const signature_hash = hasher.final();

        const fn_name = try std.fmt.allocPrint(self.allocator, "{s}_{x}_r{d}", .{ callee, signature_hash, result_index });
        defer self.allocator.free(fn_name);
        const symbol = try self.mkSymbol(fn_name);

        const domain_len = operands.len + slots.len;
        var domain = try self.allocator.alloc(z3.Z3_sort, domain_len);
        defer self.allocator.free(domain);
        var args = try self.allocator.alloc(z3.Z3_ast, domain_len);
        defer self.allocator.free(args);

        var cursor: usize = 0;
        for (operands) |opnd| {
            domain[cursor] = z3.Z3_get_sort(self.context.ctx, opnd);
            args[cursor] = opnd;
            cursor += 1;
        }
        for (slots) |slot| {
            domain[cursor] = slot.sort;
            args[cursor] = slot.pre;
            cursor += 1;
        }

        const func_decl = z3.Z3_mk_func_decl(self.context.ctx, symbol, @intCast(domain_len), domain.ptr, result_sort);
        return z3.Z3_mk_app(self.context.ctx, func_decl, @intCast(domain_len), args.ptr);
    }

    fn encodeCallStateTransitionUFSymbol(
        self: *Encoder,
        callee: []const u8,
        slot: CallSlotState,
        operands: []const z3.Z3_ast,
        slots: []const CallSlotState,
    ) EncodeError!z3.Z3_ast {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(callee);
        hasher.update(slot.name);
        const range_sort_ptr = @intFromPtr(slot.sort);
        hasher.update(std.mem.asBytes(&range_sort_ptr));
        for (operands) |opnd| {
            const sort_ptr = @intFromPtr(z3.Z3_get_sort(self.context.ctx, opnd));
            hasher.update(std.mem.asBytes(&sort_ptr));
        }
        for (slots) |state_slot| {
            hasher.update(state_slot.name);
            const slot_sort_ptr = @intFromPtr(state_slot.sort);
            hasher.update(std.mem.asBytes(&slot_sort_ptr));
        }
        const signature_hash = hasher.final();

        const fn_name = try std.fmt.allocPrint(self.allocator, "{s}_state_{s}_{x}", .{ callee, slot.name, signature_hash });
        defer self.allocator.free(fn_name);
        const symbol = try self.mkSymbol(fn_name);

        const domain_len = operands.len + slots.len;
        var domain = try self.allocator.alloc(z3.Z3_sort, domain_len);
        defer self.allocator.free(domain);
        var args = try self.allocator.alloc(z3.Z3_ast, domain_len);
        defer self.allocator.free(args);

        var cursor: usize = 0;
        for (operands) |opnd| {
            domain[cursor] = z3.Z3_get_sort(self.context.ctx, opnd);
            args[cursor] = opnd;
            cursor += 1;
        }
        for (slots) |state_slot| {
            domain[cursor] = state_slot.sort;
            args[cursor] = state_slot.pre;
            cursor += 1;
        }

        const decl = z3.Z3_mk_func_decl(self.context.ctx, symbol, @intCast(domain_len), domain.ptr, slot.sort);
        return z3.Z3_mk_app(self.context.ctx, decl, @intCast(domain_len), args.ptr);
    }

    fn collectFunctionWriteInfo(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        write_slots: *std.ArrayList([]u8),
        writes_unknown: *bool,
    ) EncodeError!void {
        writes_unknown.* = false;
        var visited_funcs = std.AutoHashMap(u64, void).init(self.allocator);
        defer visited_funcs.deinit();
        try self.collectFunctionWriteInfoRecursive(func_op, write_slots, writes_unknown, &visited_funcs);
    }

    fn appendWriteSlotUnique(self: *Encoder, write_slots: *std.ArrayList([]u8), slot_name: []const u8) EncodeError!void {
        for (write_slots.items) |existing| {
            if (std.mem.eql(u8, existing, slot_name)) return;
        }
        try write_slots.append(self.allocator, try self.allocator.dupe(u8, slot_name));
    }

    fn functionHasWriteEffect(self: *Encoder, func_op: mlir.MlirOperation) bool {
        _ = self;
        const effect_attr = mlir.oraOperationGetAttributeByName(func_op, mlir.oraStringRefCreate("ora.effect", 10));
        if (mlir.oraAttributeIsNull(effect_attr)) return false;
        const effect_ref = mlir.oraStringAttrGetValue(effect_attr);
        if (effect_ref.data == null or effect_ref.length == 0) return false;
        const effect = effect_ref.data[0..effect_ref.length];
        return std.mem.eql(u8, effect, "writes") or std.mem.eql(u8, effect, "readwrites");
    }

    fn collectFunctionWriteInfoRecursive(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        write_slots: *std.ArrayList([]u8),
        writes_unknown: *bool,
        visited_funcs: *std.AutoHashMap(u64, void),
    ) EncodeError!void {
        const func_id = @intFromPtr(func_op.ptr);
        if (visited_funcs.contains(func_id)) return;
        try visited_funcs.put(func_id, {});

        const slots_before = write_slots.items.len;
        const has_write_effect = self.functionHasWriteEffect(func_op);

        // Prefer explicit metadata when available.
        const slots_attr = mlir.oraOperationGetAttributeByName(func_op, mlir.oraStringRefCreate("ora.write_slots", 15));
        if (!mlir.oraAttributeIsNull(slots_attr)) {
            const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(slots_attr));
            for (0..count) |i| {
                const elem = mlir.oraArrayAttrGetElement(slots_attr, i);
                if (mlir.oraAttributeIsNull(elem)) continue;
                const slot_ref = mlir.oraStringAttrGetValue(elem);
                if (slot_ref.data == null or slot_ref.length == 0) continue;
                const slot_name = slot_ref.data[0..slot_ref.length];
                try self.appendWriteSlotUnique(write_slots, slot_name);
            }
        }

        // Also scan function body and transitive callees to recover missing metadata.
        try self.collectWriteInfoFromOperation(func_op, write_slots, writes_unknown, visited_funcs);

        // If function is marked as writing but we still couldn't recover any slots,
        // conservatively model as unknown write set.
        if (has_write_effect and write_slots.items.len == slots_before) {
            writes_unknown.* = true;
        }
    }

    fn collectWriteInfoFromOperation(
        self: *Encoder,
        op: mlir.MlirOperation,
        write_slots: *std.ArrayList([]u8),
        writes_unknown: *bool,
        visited_funcs: *std.AutoHashMap(u64, void),
    ) EncodeError!void {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.sstore")) {
            const global_name = self.getStringAttr(op, "global") orelse {
                writes_unknown.* = true;
                return;
            };
            try self.appendWriteSlotUnique(write_slots, global_name);
        } else if (std.mem.eql(u8, op_name, "ora.map_store")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands < 1) {
                writes_unknown.* = true;
            } else {
                const map_operand = mlir.oraOperationGetOperand(op, 0);
                if (self.resolveGlobalNameFromMapOperand(map_operand)) |global_name| {
                    try self.appendWriteSlotUnique(write_slots, global_name);
                } else {
                    // Could be a non-global map or unresolved storage origin.
                    writes_unknown.* = true;
                }
            }
        } else if (self.verify_calls and
            (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")))
        {
            const owned_callee = try self.resolveCalleeName(op);
            defer if (owned_callee) |name| self.allocator.free(name);
            if (owned_callee) |callee| {
                if (self.function_ops.get(callee)) |callee_op| {
                    try self.collectFunctionWriteInfoRecursive(callee_op, write_slots, writes_unknown, visited_funcs);
                } else {
                    // Unknown callee target: conservatively treat as unknown state write.
                    writes_unknown.* = true;
                }
            } else {
                writes_unknown.* = true;
            }
        }

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            const region = mlir.oraOperationGetRegion(op, region_idx);
            if (mlir.oraRegionIsNull(region)) continue;
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) {
                var nested = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(nested)) {
                    try self.collectWriteInfoFromOperation(nested, write_slots, writes_unknown, visited_funcs);
                    nested = mlir.oraOperationGetNextInBlock(nested);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    fn inferGlobalSortFromFunction(self: *Encoder, func_op: mlir.MlirOperation, slot_name: []const u8) ?z3.Z3_sort {
        var found: ?z3.Z3_sort = null;
        self.findGlobalSortInOperation(func_op, slot_name, &found);
        return found;
    }

    fn findGlobalSortInOperation(
        self: *Encoder,
        op: mlir.MlirOperation,
        slot_name: []const u8,
        out: *?z3.Z3_sort,
    ) void {
        if (out.* != null) return;

        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "ora.sload") or std.mem.eql(u8, op_name, "ora.sstore")) {
                const global_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("global", 6));
                if (!mlir.oraAttributeIsNull(global_attr)) {
                    const global_ref = mlir.oraStringAttrGetValue(global_attr);
                    if (global_ref.data != null and global_ref.length > 0) {
                        const global_name = global_ref.data[0..global_ref.length];
                        if (std.mem.eql(u8, global_name, slot_name)) {
                            if (std.mem.eql(u8, op_name, "ora.sload")) {
                                const num_results = mlir.oraOperationGetNumResults(op);
                                if (num_results >= 1) {
                                    const result_value = mlir.oraOperationGetResult(op, 0);
                                    out.* = self.encodeMLIRType(mlir.oraValueGetType(result_value)) catch null;
                                    if (out.* != null) return;
                                }
                            } else {
                                const num_operands = mlir.oraOperationGetNumOperands(op);
                                if (num_operands >= 1) {
                                    const value = mlir.oraOperationGetOperand(op, 0);
                                    out.* = self.encodeMLIRType(mlir.oraValueGetType(value)) catch null;
                                    if (out.* != null) return;
                                }
                            }
                        }
                    }
                }
            }
        }

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            const region = mlir.oraOperationGetRegion(op, region_idx);
            if (mlir.oraRegionIsNull(region)) continue;
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) {
                var nested = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(nested)) {
                    self.findGlobalSortInOperation(nested, slot_name, out);
                    if (out.* != null) return;
                    nested = mlir.oraOperationGetNextInBlock(nested);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    fn findFunctionReturnOp(self: *Encoder, func_op: mlir.MlirOperation) ?mlir.MlirOperation {
        var out = mlir.MlirOperation{ .ptr = null };
        self.findReturnInOperation(func_op, &out);
        if (mlir.oraOperationIsNull(out)) return null;
        return out;
    }

    fn findReturnInOperation(self: *Encoder, op: mlir.MlirOperation, out: *mlir.MlirOperation) void {
        if (!mlir.oraOperationIsNull(out.*)) return;

        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "ora.return") or std.mem.eql(u8, op_name, "func.return")) {
                out.* = op;
                return;
            }
        }

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            const region = mlir.oraOperationGetRegion(op, region_idx);
            if (mlir.oraRegionIsNull(region)) continue;
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) {
                var nested = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(nested)) {
                    self.findReturnInOperation(nested, out);
                    if (!mlir.oraOperationIsNull(out.*)) return;
                    nested = mlir.oraOperationGetNextInBlock(nested);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    fn appendSlotState(
        self: *Encoder,
        slots: *std.ArrayList(CallSlotState),
        name: []const u8,
        is_write: bool,
        func_op: ?mlir.MlirOperation,
    ) EncodeError!void {
        for (slots.items) |*existing| {
            if (std.mem.eql(u8, existing.name, name)) {
                existing.is_write = existing.is_write or is_write;
                return;
            }
        }

        const slot_name = try self.allocator.dupe(u8, name);
        var pre: z3.Z3_ast = undefined;
        var sort: z3.Z3_sort = undefined;
        if (self.global_map.get(slot_name)) |existing| {
            pre = existing;
            sort = z3.Z3_get_sort(self.context.ctx, existing);
        } else {
            const inferred_sort = if (func_op) |fop|
                self.inferGlobalSortFromFunction(fop, slot_name)
            else
                null;
            sort = inferred_sort orelse self.mkBitVectorSort(256);
            pre = try self.getOrCreateGlobal(slot_name, sort);
        }

        try slots.append(self.allocator, .{
            .name = slot_name,
            .pre = pre,
            .sort = sort,
            .is_write = is_write,
            .post = null,
        });
    }

    fn sortSlotStates(_: *Encoder, slots: []CallSlotState) void {
        const Ctx = struct {};
        std.mem.sort(CallSlotState, slots, Ctx{}, struct {
            fn lessThan(_: Ctx, lhs: CallSlotState, rhs: CallSlotState) bool {
                return std.mem.lessThan(u8, lhs.name, rhs.name);
            }
        }.lessThan);
    }

    fn freeSlotStates(self: *Encoder, slots: []CallSlotState) void {
        for (slots) |slot| {
            self.allocator.free(slot.name);
        }
    }

    fn tryInlinePureCallResult(
        self: *Encoder,
        callee: []const u8,
        func_op: mlir.MlirOperation,
        operands: []const z3.Z3_ast,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        // Old-mode fallbacks should not mutate caller state. Inline only pure callees.
        if (self.functionHasWriteEffect(func_op)) return null;
        if (self.inlineStackContains(callee)) return null;
        if (self.inline_function_stack.items.len >= self.max_summary_inline_depth) return null;

        var summary_encoder = Encoder.init(self.context, self.allocator);
        defer summary_encoder.deinit();
        summary_encoder.setVerifyCalls(self.verify_calls);
        summary_encoder.setVerifyState(self.verify_state);
        summary_encoder.max_summary_inline_depth = self.max_summary_inline_depth;
        try summary_encoder.copyFunctionRegistryFrom(self);
        try summary_encoder.copyStructRegistryFrom(self);
        try summary_encoder.copyInlineStackFrom(self);
        try summary_encoder.copyEnvMapFrom(self);
        try summary_encoder.copyGlobalStateMapFrom(self, mode == .Old);
        try summary_encoder.copyGlobalEntryMapFrom(self);
        try summary_encoder.pushInlineFunction(callee);

        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return null;
        const entry_block = mlir.oraRegionGetFirstBlock(body_region);
        if (mlir.oraBlockIsNull(entry_block)) return null;

        const arg_count = mlir.oraBlockGetNumArguments(entry_block);
        const bind_count = @min(arg_count, operands.len);
        for (0..bind_count) |i| {
            const arg_value = mlir.oraBlockGetArgument(entry_block, i);
            try summary_encoder.bindValue(arg_value, operands[i]);
        }

        const return_op = self.findFunctionReturnOp(func_op) orelse return null;
        const ret_operands = mlir.oraOperationGetNumOperands(return_op);
        if (result_index >= ret_operands) return null;
        const ret_value = mlir.oraOperationGetOperand(return_op, result_index);

        const encoded = summary_encoder.encodeValueWithMode(ret_value, mode) catch return null;

        const extra_constraints = try summary_encoder.takeConstraints(self.allocator);
        defer if (extra_constraints.len > 0) self.allocator.free(extra_constraints);
        for (extra_constraints) |cst| self.addConstraint(cst);

        const extra_obligations = try summary_encoder.takeObligations(self.allocator);
        defer if (extra_obligations.len > 0) self.allocator.free(extra_obligations);
        for (extra_obligations) |obl| self.addObligation(obl);

        return encoded;
    }

    fn tryInlineFunctionCallSummary(
        self: *Encoder,
        callee: []const u8,
        func_op: mlir.MlirOperation,
        operands: []const z3.Z3_ast,
        slots: []CallSlotState,
        result_exprs: []?z3.Z3_ast,
    ) EncodeError!bool {
        // Stop recursive/non-terminating expansion; fall back to UF summaries.
        if (self.inlineStackContains(callee)) return false;
        if (self.inline_function_stack.items.len >= self.max_summary_inline_depth) return false;

        var summary_encoder = Encoder.init(self.context, self.allocator);
        defer summary_encoder.deinit();
        summary_encoder.setVerifyCalls(self.verify_calls);
        summary_encoder.setVerifyState(self.verify_state);
        summary_encoder.max_summary_inline_depth = self.max_summary_inline_depth;
        try summary_encoder.copyFunctionRegistryFrom(self);
        try summary_encoder.copyStructRegistryFrom(self);
        try summary_encoder.copyInlineStackFrom(self);
        try summary_encoder.copyEnvMapFrom(self);
        try summary_encoder.pushInlineFunction(callee);

        for (slots) |slot| {
            const g_key = try summary_encoder.allocator.dupe(u8, slot.name);
            try summary_encoder.global_map.put(g_key, slot.pre);
            const old_key = try summary_encoder.allocator.dupe(u8, slot.name);
            try summary_encoder.global_old_map.put(old_key, slot.pre);
            const entry_key = try summary_encoder.allocator.dupe(u8, slot.name);
            try summary_encoder.global_entry_map.put(entry_key, slot.pre);
        }

        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return false;
        const entry_block = mlir.oraRegionGetFirstBlock(body_region);
        if (mlir.oraBlockIsNull(entry_block)) return false;

        const arg_count = mlir.oraBlockGetNumArguments(entry_block);
        const bind_count = @min(arg_count, operands.len);
        for (0..bind_count) |i| {
            const arg_value = mlir.oraBlockGetArgument(entry_block, i);
            try summary_encoder.bindValue(arg_value, operands[i]);
        }

        // Materialize stateful effects from the callee body so summary state
        // reflects sstore/map_store updates before we snapshot post-state.
        summary_encoder.encodeStateEffectsInOperation(func_op);

        const return_op = self.findFunctionReturnOp(func_op) orelse return false;
        const ret_operands = mlir.oraOperationGetNumOperands(return_op);
        const result_count = @min(ret_operands, result_exprs.len);
        if (result_count == 0 and result_exprs.len > 0) return false;

        var any_result = false;
        for (0..result_count) |i| {
            const ret_value = mlir.oraOperationGetOperand(return_op, i);
            const encoded = summary_encoder.encodeValue(ret_value) catch return false;
            result_exprs[i] = encoded;
            any_result = true;
        }

        const extra_constraints = try summary_encoder.takeConstraints(self.allocator);
        defer if (extra_constraints.len > 0) self.allocator.free(extra_constraints);
        for (extra_constraints) |cst| self.addConstraint(cst);

        const extra_obligations = try summary_encoder.takeObligations(self.allocator);
        defer if (extra_obligations.len > 0) self.allocator.free(extra_obligations);
        for (extra_obligations) |obl| self.addObligation(obl);

        for (slots) |*slot| {
            if (summary_encoder.global_map.get(slot.name)) |post| {
                slot.post = post;
            }
        }

        return any_result or slots.len > 0;
    }

    fn encodeStateEffectsInOperation(self: *Encoder, op: mlir.MlirOperation) void {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "ora.sstore") or
                std.mem.eql(u8, op_name, "ora.map_store") or
                std.mem.eql(u8, op_name, "func.call") or
                std.mem.eql(u8, op_name, "call"))
            {
                _ = self.encodeOperation(op) catch {};
            }
        }

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            const region = mlir.oraOperationGetRegion(op, region_idx);
            if (mlir.oraRegionIsNull(region)) continue;
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) {
                var nested = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(nested)) {
                    self.encodeStateEffectsInOperation(nested);
                    nested = mlir.oraOperationGetNextInBlock(nested);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    fn materializeCallSummaryCurrent(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        operands: []const z3.Z3_ast,
    ) EncodeError!void {
        const call_id = @intFromPtr(mlir_op.ptr);
        if (self.materialized_calls.contains(call_id)) return;

        const owned_callee = try self.resolveCalleeName(mlir_op);
        defer if (owned_callee) |name| self.allocator.free(name);
        const callee = owned_callee orelse "call";
        const func_op = self.function_ops.get(callee);

        var write_slots = std.ArrayList([]u8){};
        defer {
            for (write_slots.items) |slot_name| self.allocator.free(slot_name);
            write_slots.deinit(self.allocator);
        }
        var writes_unknown = false;
        if (self.verify_state) {
            if (func_op) |fop| {
                try self.collectFunctionWriteInfo(fop, &write_slots, &writes_unknown);
            }
        }

        var slots = std.ArrayList(CallSlotState){};
        defer {
            self.freeSlotStates(slots.items);
            slots.deinit(self.allocator);
        }

        if (self.verify_state) {
            var g_it = self.global_map.iterator();
            while (g_it.next()) |entry| {
                try self.appendSlotState(&slots, entry.key_ptr.*, false, func_op);
            }

            for (write_slots.items) |slot_name| {
                try self.appendSlotState(&slots, slot_name, true, func_op);
            }

            if (writes_unknown) {
                for (slots.items) |*slot| {
                    slot.is_write = true;
                }
            }
        }

        self.sortSlotStates(slots.items);

        const num_results: usize = @intCast(mlir.oraOperationGetNumResults(mlir_op));
        var result_exprs = try self.allocator.alloc(?z3.Z3_ast, num_results);
        defer self.allocator.free(result_exprs);
        for (0..num_results) |i| result_exprs[i] = null;

        if (func_op) |fop| {
            _ = try self.tryInlineFunctionCallSummary(callee, fop, operands, slots.items, result_exprs);
        }

        for (0..num_results) |i| {
            const result_value = mlir.oraOperationGetResult(mlir_op, i);
            const result_type = mlir.oraValueGetType(result_value);
            const result_sort = try self.encodeMLIRType(result_type);
            const encoded = if (result_exprs[i]) |expr|
                expr
            else
                try self.encodeCallResultUFSymbol(callee, operands, slots.items, @intCast(i), result_sort);
            const result_id = @intFromPtr(result_value.ptr);
            try self.value_map.put(result_id, encoded);
        }

        if (self.verify_state) {
            for (slots.items, 0..) |slot, slot_idx| {
                var post = slot.post orelse slot.pre;
                if (slot.is_write and slot.post == null) {
                    post = try self.encodeCallStateTransitionUFSymbol(callee, slot, operands, slots.items);
                } else if (!slot.is_write) {
                    self.addConstraint(z3.Z3_mk_eq(self.context.ctx, post, slot.pre));
                    const slot_sort = z3.Z3_get_sort(self.context.ctx, post);
                    if (self.isArraySort(slot_sort)) {
                        const call_id_u64: u64 = @intCast(@intFromPtr(mlir_op.ptr));
                        const frame_id = @as(u64, @intCast(slot_idx)) ^ (call_id_u64 << 8);
                        self.addArrayEqualityFrameConstraint(slot.pre, post, frame_id) catch {};
                    }
                }

                if (self.global_map.getPtr(slot.name)) |existing| {
                    existing.* = post;
                } else {
                    const key = try self.allocator.dupe(u8, slot.name);
                    try self.global_map.put(key, post);
                }
            }
        }

        try self.materialized_calls.put(call_id, {});
    }

    fn resolveCalleeName(self: *Encoder, mlir_op: mlir.MlirOperation) EncodeError!?[]u8 {
        if (self.getStringAttr(mlir_op, "callee")) |callee| {
            return try self.allocator.dupe(u8, callee);
        }

        const printed = mlir.oraOperationPrintToString(mlir_op);
        defer if (printed.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(printed);
        };
        if (printed.data == null or printed.length == 0) return null;

        const text = printed.data[0..printed.length];
        const at_pos = std.mem.indexOfScalar(u8, text, '@') orelse return null;
        var end = at_pos + 1;
        while (end < text.len) : (end += 1) {
            const ch = text[end];
            if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '$')) {
                break;
            }
        }
        if (end <= at_pos + 1) return null;
        return try self.allocator.dupe(u8, text[at_pos + 1 .. end]);
    }

    fn resolveGlobalNameFromMapOperand(self: *Encoder, value: mlir.MlirValue) ?[]const u8 {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;

        const name_ref = mlir.oraOperationGetName(owner);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.sload")) {
            return self.getStringAttr(owner, "global");
        }

        if (std.mem.eql(u8, op_name, "ora.refinement_to_base") or
            std.mem.eql(u8, op_name, "ora.base_to_refinement") or
            std.mem.eql(u8, op_name, "arith.bitcast") or
            std.mem.eql(u8, op_name, "arith.extui") or
            std.mem.eql(u8, op_name, "arith.extsi") or
            std.mem.eql(u8, op_name, "arith.trunci") or
            std.mem.eql(u8, op_name, "arith.index_cast") or
            std.mem.eql(u8, op_name, "arith.index_castui") or
            std.mem.eql(u8, op_name, "arith.index_castsi") or
            std.mem.eql(u8, op_name, "builtin.unrealized_conversion_cast") or
            std.mem.eql(u8, op_name, "tensor.cast"))
        {
            const num_operands = mlir.oraOperationGetNumOperands(owner);
            if (num_operands < 1) return null;
            const inner = mlir.oraOperationGetOperand(owner, 0);
            return self.resolveGlobalNameFromMapOperand(inner);
        }

        return null;
    }

    fn extractIfYield(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        region_index: u32,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
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
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (num_operands > result_index) {
                    const value = mlir.oraOperationGetOperand(current, result_index);
                    return try self.encodeValueWithMode(value, mode);
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

    fn resolveFieldNameForIndex(self: *Encoder, field_names_csv: ?[]const u8, index: usize) ![]u8 {
        if (field_names_csv) |csv| {
            var it = std.mem.splitScalar(u8, csv, ',');
            var idx: usize = 0;
            while (it.next()) |part| {
                if (idx == index) {
                    const trimmed = std.mem.trim(u8, part, " \t\n\r");
                    if (trimmed.len > 0) {
                        return self.allocator.dupe(u8, trimmed);
                    }
                    break;
                }
                idx += 1;
            }
        }
        return std.fmt.allocPrint(self.allocator, "field_{d}", .{index});
    }

    fn coerceToBool(self: *Encoder, ast: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, ast);
        const kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (kind == z3.Z3_BOOL_SORT) return ast;
        if (kind == z3.Z3_BV_SORT) {
            const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
            return z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, ast, zero));
        }
        return ast;
    }

    fn quantifiedVarSortFromTypeString(self: *Encoder, type_name: []const u8) z3.Z3_sort {
        if (std.mem.eql(u8, type_name, "bool")) {
            return z3.Z3_mk_bool_sort(self.context.ctx);
        }
        if (std.mem.eql(u8, type_name, "address")) {
            return self.mkBitVectorSort(160);
        }
        if (std.mem.eql(u8, type_name, "string")) {
            return z3.Z3_mk_string_sort(self.context.ctx);
        }
        if (std.mem.eql(u8, type_name, "bytes")) {
            return z3.Z3_mk_string_sort(self.context.ctx);
        }

        if (type_name.len >= 2 and (type_name[0] == 'u' or type_name[0] == 'i')) {
            const width = std.fmt.parseInt(u32, type_name[1..], 10) catch 256;
            return self.mkBitVectorSort(width);
        }

        // Default to EVM word width for unknown/refinement type spellings.
        return self.mkBitVectorSort(256);
    }

    fn encodeQuantifiedOp(self: *Encoder, mlir_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        if (num_operands < 1) return error.InvalidOperandCount;
        const quantifier = self.getStringAttr(mlir_op, "quantifier") orelse
            self.getStringAttr(mlir_op, "ora.quantifier") orelse
            "forall";
        const variable = self.getStringAttr(mlir_op, "variable") orelse
            self.getStringAttr(mlir_op, "ora.bound_variable") orelse
            "q";
        const variable_type = self.getStringAttr(mlir_op, "variable_type") orelse
            self.getStringAttr(mlir_op, "ora.variable_type") orelse
            "u256";

        const var_sort = self.quantifiedVarSortFromTypeString(variable_type);
        const bound_var = try self.mkVariable(variable, var_sort);
        try self.pushQuantifiedBinding(variable, bound_var);
        defer self.popQuantifiedBinding();

        const body_value = mlir.oraOperationGetOperand(mlir_op, num_operands - 1);
        var quantified_body = self.coerceToBool(try self.encodeValueWithMode(body_value, mode));

        if (num_operands > 1) {
            const condition_value = mlir.oraOperationGetOperand(mlir_op, 0);
            const condition = self.coerceToBool(try self.encodeValueWithMode(condition_value, mode));
            if (std.ascii.eqlIgnoreCase(quantifier, "exists")) {
                quantified_body = z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ condition, quantified_body });
            } else {
                quantified_body = z3.Z3_mk_implies(self.context.ctx, condition, quantified_body);
            }
        }

        var bounds = [_]z3.Z3_app{z3.Z3_to_app(self.context.ctx, bound_var)};
        if (std.ascii.eqlIgnoreCase(quantifier, "exists")) {
            return z3.Z3_mk_exists_const(self.context.ctx, 0, 1, &bounds, 0, null, quantified_body);
        }
        return z3.Z3_mk_forall_const(self.context.ctx, 0, 1, &bounds, 0, null, quantified_body);
    }

    fn bitMaskForWidth(_: *Encoder, width: u32) u256 {
        if (width == 0) return 0;
        if (width >= 256) return std.math.maxInt(u256);
        return (@as(u256, 1) << @intCast(width)) - 1;
    }

    fn normalizeUnsignedToWidth(self: *Encoder, value: u256, width: u32) u256 {
        return value & self.bitMaskForWidth(width);
    }

    fn negateModuloWidth(self: *Encoder, abs_value: u256, width: u32) u256 {
        const mask = self.bitMaskForWidth(width);
        const narrowed = abs_value & mask;
        if (narrowed == 0) return 0;
        return ((~narrowed) +% 1) & mask;
    }

    fn normalizeSignedIntToWidth(self: *Encoder, value: i64, width: u32) u256 {
        if (value >= 0) {
            return self.normalizeUnsignedToWidth(@intCast(value), width);
        }
        const signed_wide: i128 = value;
        const abs_wide: i128 = -signed_wide;
        const abs_u128: u128 = @intCast(abs_wide);
        return self.negateModuloWidth(@intCast(abs_u128), width);
    }

    fn parseLiteralToWidth(self: *Encoder, literal: []const u8, width: u32) ?u256 {
        if (literal.len == 0) return null;
        var negative = false;
        var body = literal;

        if (body[0] == '-') {
            negative = true;
            body = body[1..];
        }
        if (body.len == 0) return null;

        var base: u8 = 10;
        if (body.len >= 2 and body[0] == '0' and (body[1] == 'x' or body[1] == 'X')) {
            base = 16;
            body = body[2..];
            if (body.len == 0) return null;
        }

        const parsed = std.fmt.parseUnsigned(u256, body, base) catch return null;
        if (negative) return self.negateModuloWidth(parsed, width);
        return self.normalizeUnsignedToWidth(parsed, width);
    }

    fn normalizeBytesHexLiteral(self: *Encoder, literal: []const u8) EncodeError![]u8 {
        var hex = literal;
        if (hex.len >= 2 and hex[0] == '0' and (hex[1] == 'x' or hex[1] == 'X')) {
            hex = hex[2..];
        }

        if (hex.len % 2 != 0) return error.UnsupportedOperation;

        const normalized = try self.allocator.alloc(u8, hex.len);
        errdefer self.allocator.free(normalized);

        for (hex, 0..) |ch, i| {
            const lower = std.ascii.toLower(ch);
            if (!((lower >= '0' and lower <= '9') or (lower >= 'a' and lower <= 'f'))) {
                return error.UnsupportedOperation;
            }
            normalized[i] = lower;
        }

        return normalized;
    }

    fn parseConstAttrValue(self: *Encoder, attr: mlir.MlirAttribute, width: u32) ?u256 {
        if (mlir.oraAttributeIsNull(attr)) return null;
        const str = mlir.oraStringAttrGetValue(attr);
        if (str.data != null and str.length > 0) {
            const slice = str.data[0..str.length];
            return self.parseLiteralToWidth(slice, width);
        }
        const int_str = mlir.oraIntegerAttrGetValueString(attr);
        defer if (int_str.data != null) @import("mlir_c_api").freeStringRef(int_str);
        if (int_str.data != null and int_str.length > 0) {
            const slice = int_str.data[0..int_str.length];
            return self.parseLiteralToWidth(slice, width);
        }
        return self.normalizeSignedIntToWidth(mlir.oraIntegerAttrGetValueSInt(attr), width);
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
    fn getConstantValue(self: *Encoder, mlir_op: mlir.MlirOperation, width: u32) u256 {
        // extract constant value from arith.constant attributes
        // the value attribute contains the integer constant
        const attr_name_ref = mlir.oraStringRefCreate("value".ptr, "value".len);
        const attr = mlir.oraOperationGetAttributeByName(mlir_op, attr_name_ref);
        return self.parseConstAttrValue(attr, width) orelse 0;
    }

    /// Encode MLIR arithmetic operation (arith.addi, arith.subi, etc.)
    pub fn encodeArithOp(self: *Encoder, op_name: []const u8, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation) !z3.Z3_ast {
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
            ArithmeticOp.DivUnsigned
        else if (std.mem.eql(u8, op_name, "arith.divsi"))
            ArithmeticOp.DivSigned
        else if (std.mem.eql(u8, op_name, "arith.remui"))
            ArithmeticOp.RemUnsigned
        else if (std.mem.eql(u8, op_name, "arith.remsi"))
            ArithmeticOp.RemSigned
        else {
            return error.UnsupportedOperation;
        };

        const guard_internal = if (self.getStringAttr(mlir_op, "ora.guard_internal")) |value|
            std.mem.eql(u8, value, "true")
        else
            false;
        if (!guard_internal) {
            // For arith.* ops, only emit div-by-zero obligations here.
            // Overflow obligations for add/sub/mul are emitted by ora.assert ops
            // that the MLIR lowering places alongside checked arithmetic.
            // Wrapping operators (+%, -%, *%) don't have ora.assert, so they
            // correctly get no overflow obligation.
            switch (arith_op) {
                .DivUnsigned, .DivSigned, .RemUnsigned, .RemSigned => {
                    const non_zero = z3.Z3_mk_not(self.context.ctx, self.checkDivByZero(rhs));
                    self.addObligation(non_zero);
                },
                else => {},
            }
        }
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
