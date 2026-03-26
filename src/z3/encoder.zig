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
const mlir_helpers = @import("mlir_helpers.zig");

const finite_scf_for_unroll_limit: usize = 8;
const finite_scf_while_unroll_limit: usize = 16;

/// MLIR to SMT encoder
pub const Encoder = struct {
    const SwitchCaseMetadata = struct {
        case_kinds: []i64,
        case_values: []i64,
        range_starts: []i64,
        range_ends: []i64,
        default_case_index: ?i64,

        fn deinit(self: *SwitchCaseMetadata, allocator: std.mem.Allocator) void {
            allocator.free(self.case_kinds);
            allocator.free(self.case_values);
            allocator.free(self.range_starts);
            allocator.free(self.range_ends);
        }
    };

    const TupleValue = struct {
        elements: []z3.Z3_ast,
    };
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

    pub const TrackedMemrefState = struct {
        value: z3.Z3_ast,
        initialized: z3.Z3_ast,
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
    /// Local tuple element cache for ora.tuple_create / ora.tuple_extract.
    tuple_values: std.AutoHashMap(u64, TupleValue),
    /// Map from global storage name to Z3 AST (for consistent storage symbols)
    global_map: std.StringHashMap(z3.Z3_ast),
    /// Map from global storage name to Z3 AST (for old() storage symbols)
    global_old_map: std.StringHashMap(z3.Z3_ast),
    /// Snapshot of global storage symbols at function entry.
    global_entry_map: std.StringHashMap(z3.Z3_ast),
    /// Global storage roots actually written by this encoder instance.
    written_global_slots: std.StringHashMap(void),
    /// Map from environment symbol name (e.g. msg.sender) to Z3 AST.
    env_map: std.StringHashMap(z3.Z3_ast),
    /// Scalar memref local state threaded during verification extraction.
    memref_map: std.AutoHashMap(u64, TrackedMemrefState),
    /// Scalar memref local state for old() mode.
    memref_old_map: std.AutoHashMap(u64, TrackedMemrefState),
    /// Stable symbolic dimensions for tensor.dim/memref.dim on dynamic shapes.
    tensor_dim_map: std.AutoHashMap(TensorDimKey, z3.Z3_ast),
    /// Map from function symbol name to MLIR function operation.
    function_ops: std.StringHashMap(mlir.MlirOperation),
    /// Map from struct symbol name to comma-separated field names in declaration order.
    struct_field_names_csv: std.StringHashMap([]u8),
    /// Map from struct symbol name to its declaration op for field type lookup.
    struct_decl_ops: std.StringHashMap(mlir.MlirOperation),
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
    /// Set once encoding drops or skips any verification-relevant information.
    encoding_degraded: bool,
    /// Human-readable reason for the first degradation encountered.
    encoding_degraded_reason: ?[]const u8,
    /// Cache of error_union tuple sorts by MLIR type pointer.
    error_union_sorts: std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    /// Stack of active quantified variable bindings (innermost binding last).
    quantified_bindings: std.ArrayList(QuantifiedBinding),
    /// Active branch assumptions used during pure-return extraction.
    return_path_assumptions: std.ArrayList(z3.Z3_ast),

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
            .tuple_values = std.AutoHashMap(u64, TupleValue).init(allocator),
            .global_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .global_old_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .global_entry_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .written_global_slots = std.StringHashMap(void).init(allocator),
            .env_map = std.StringHashMap(z3.Z3_ast).init(allocator),
            .memref_map = std.AutoHashMap(u64, TrackedMemrefState).init(allocator),
            .memref_old_map = std.AutoHashMap(u64, TrackedMemrefState).init(allocator),
            .tensor_dim_map = std.AutoHashMap(TensorDimKey, z3.Z3_ast).init(allocator),
            .function_ops = std.StringHashMap(mlir.MlirOperation).init(allocator),
            .struct_field_names_csv = std.StringHashMap([]u8).init(allocator),
            .struct_decl_ops = std.StringHashMap(mlir.MlirOperation).init(allocator),
            .materialized_calls = std.AutoHashMap(u64, void).init(allocator),
            .inline_function_stack = std.ArrayList([]u8){},
            .string_storage = std.ArrayList([]u8){},
            .pending_constraints = std.ArrayList(z3.Z3_ast){},
            .pending_obligations = std.ArrayList(z3.Z3_ast){},
            .encoding_degraded = false,
            .encoding_degraded_reason = null,
            .error_union_sorts = std.HashMap(u64, ErrorUnionSort, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .quantified_bindings = std.ArrayList(QuantifiedBinding){},
            .return_path_assumptions = std.ArrayList(z3.Z3_ast){},
        };
    }

    pub fn setVerifyCalls(self: *Encoder, enabled: bool) void {
        self.verify_calls = enabled;
    }

    pub fn setVerifyState(self: *Encoder, enabled: bool) void {
        self.verify_state = enabled;
    }

    pub fn clearDegradation(self: *Encoder) void {
        self.encoding_degraded = false;
        self.encoding_degraded_reason = null;
    }

    pub fn isDegraded(self: *const Encoder) bool {
        return self.encoding_degraded;
    }

    pub fn degradationReason(self: *const Encoder) ?[]const u8 {
        return self.encoding_degraded_reason;
    }

    pub fn noteDegradation(self: *Encoder, reason: []const u8) void {
        self.recordDegradation(reason);
    }

    pub fn getOrCreateCurrentGlobal(self: *Encoder, name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        return try self.getOrCreateGlobal(name, sort);
    }

    fn recordDegradation(self: *Encoder, reason: []const u8) void {
        self.encoding_degraded = true;
        if (self.encoding_degraded_reason == null) {
            self.encoding_degraded_reason = reason;
        }
    }

    fn allocPersistentMessage(self: *Encoder, message: []const u8) ![]const u8 {
        const copy = try self.allocator.dupeZ(u8, message);
        try self.string_storage.append(self.allocator, copy[0 .. message.len + 1]);
        return copy[0..message.len];
    }

    fn getOperationLocationText(self: *Encoder, op: mlir.MlirOperation) !?[]const u8 {
        const loc = mlir.oraOperationGetLocation(op);
        if (mlir.oraLocationIsNull(loc)) return null;

        const loc_ref = mlir.oraLocationPrintToString(loc);
        defer @import("mlir_c_api").freeStringRef(loc_ref);
        if (loc_ref.data == null or loc_ref.length == 0) return null;

        const loc_text = loc_ref.data[0..loc_ref.length];
        return try self.allocPersistentMessage(loc_text);
    }

    fn formatDegradationAtOp(self: *Encoder, op: mlir.MlirOperation, reason: []const u8) []const u8 {
        const loc_text = self.getOperationLocationText(op) catch null;
        if (loc_text) |loc| {
            const owned = std.fmt.allocPrint(self.allocator, "{s} at {s}", .{ reason, loc }) catch return reason;
            defer self.allocator.free(owned);
            return self.allocPersistentMessage(owned) catch reason;
        }
        return reason;
    }

    fn recordCalleeResultDegradation(self: *Encoder, call_op: mlir.MlirOperation, callee: []const u8, reason: []const u8) void {
        const loc_text = self.getOperationLocationText(call_op) catch null;
        const full_reason = if (loc_text) |loc|
            std.fmt.allocPrint(self.allocator, "{s} for callee '{s}' at {s}", .{ reason, callee, loc })
        else
            std.fmt.allocPrint(self.allocator, "{s} for callee '{s}'", .{ reason, callee });

        if (full_reason) |owned| {
            defer self.allocator.free(owned);
            const persistent = self.allocPersistentMessage(owned) catch {
                self.recordDegradation(reason);
                return;
            };
            self.recordDegradation(persistent);
            return;
        } else |_| {}

        self.recordDegradation(reason);
    }

    fn astEquivalent(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) bool {
        return z3.c.Z3_is_eq_ast(self.context.ctx, lhs, rhs);
    }

    fn astSimplifiesToBool(self: *Encoder, ast: z3.Z3_ast) ?bool {
        const simplified = z3.Z3_simplify(self.context.ctx, ast);
        if (self.astEquivalent(simplified, self.boolTrue())) return true;
        if (self.astEquivalent(simplified, self.boolFalse())) return false;
        return null;
    }

    fn boolTrue(self: *Encoder) z3.Z3_ast {
        return z3.Z3_mk_true(self.context.ctx);
    }

    fn boolFalse(self: *Encoder) z3.Z3_ast {
        return z3.Z3_mk_false(self.context.ctx);
    }

    fn isBoolConst(self: *Encoder, ast: z3.Z3_ast, expected: bool) bool {
        const target = if (expected) self.boolTrue() else self.boolFalse();
        return astEquivalent(self, ast, target);
    }

    fn activeReturnPathContains(self: *Encoder, needle: z3.Z3_ast) bool {
        for (self.return_path_assumptions.items) |assume| {
            if (astEquivalent(self, assume, needle)) return true;
        }
        return false;
    }

    fn astContainsConjunct(self: *Encoder, haystack: z3.Z3_ast, needle: z3.Z3_ast) bool {
        if (astEquivalent(self, haystack, needle)) return true;
        if (z3.Z3_get_ast_kind(self.context.ctx, haystack) != z3.Z3_APP_AST) return false;
        const app = z3.Z3_to_app(self.context.ctx, haystack);
        const decl = z3.Z3_get_app_decl(self.context.ctx, app);
        if (z3.Z3_get_decl_kind(self.context.ctx, decl) != z3.Z3_OP_AND) return false;
        const num_args = z3.Z3_get_app_num_args(self.context.ctx, app);
        var idx: c_uint = 0;
        while (idx < num_args) : (idx += 1) {
            if (self.astContainsConjunct(z3.Z3_get_app_arg(self.context.ctx, app, idx), needle)) return true;
        }
        return false;
    }

    fn activeReturnPathImplies(self: *Encoder, needle: z3.Z3_ast) bool {
        for (self.return_path_assumptions.items) |assume| {
            if (self.astContainsConjunct(assume, needle)) return true;
        }
        if (self.return_path_assumptions.items.len == 0) return false;

        const solver = z3.Z3_mk_solver(self.context.ctx) orelse return false;
        z3.Z3_solver_inc_ref(self.context.ctx, solver);
        defer z3.Z3_solver_dec_ref(self.context.ctx, solver);

        for (self.return_path_assumptions.items) |assume| {
            z3.Z3_solver_assert(self.context.ctx, solver, self.coerceToBool(assume));
        }
        z3.Z3_solver_assert(self.context.ctx, solver, z3.Z3_mk_not(self.context.ctx, self.coerceToBool(needle)));
        return z3.Z3_solver_check(self.context.ctx, solver) == z3.Z3_L_FALSE;
    }

    pub fn mergeInitPredicate(self: *Encoder, condition: z3.Z3_ast, then_init: z3.Z3_ast, else_init: z3.Z3_ast) z3.Z3_ast {
        const cond = self.coerceToBool(condition);
        if (astEquivalent(self, then_init, else_init)) return then_init;
        if (self.isBoolConst(then_init, true) and self.isBoolConst(else_init, false)) return cond;
        if (self.isBoolConst(then_init, false) and self.isBoolConst(else_init, true)) {
            return z3.Z3_mk_not(self.context.ctx, cond);
        }
        return self.encodeIte(cond, then_init, else_init);
    }

    fn anyResultSlotMissing(result_exprs: []const ?z3.Z3_ast) bool {
        for (result_exprs) |expr| {
            if (expr == null) return true;
        }
        return false;
    }

    fn markGlobalSlotWritten(self: *Encoder, name: []const u8) EncodeError!void {
        if (self.written_global_slots.contains(name)) return;
        try self.written_global_slots.put(try self.allocator.dupe(u8, name), {});
    }

    fn putOwnedStringAst(
        self: *Encoder,
        map: *std.StringHashMap(z3.Z3_ast),
        name: []const u8,
        value: z3.Z3_ast,
    ) EncodeError!void {
        if (map.getPtr(name)) |existing| {
            existing.* = value;
            return;
        }
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try map.put(key, value);
    }

    fn putOwnedStringVoid(
        self: *Encoder,
        map: *std.StringHashMap(void),
        name: []const u8,
    ) EncodeError!void {
        if (map.contains(name)) return;
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try map.put(key, {});
    }

    fn putTupleValue(self: *Encoder, result_id: u64, elements: []z3.Z3_ast) EncodeError!void {
        if (self.tuple_values.getPtr(result_id)) |existing| {
            self.allocator.free(existing.elements);
            existing.* = .{ .elements = elements };
            return;
        }
        try self.tuple_values.put(result_id, .{ .elements = elements });
    }

    fn hasWrittenGlobalSlot(self: *const Encoder, name: []const u8) bool {
        return self.written_global_slots.contains(name);
    }

    const StateSnapshot = struct {
        global_map: std.StringHashMap(z3.Z3_ast),
        memref_map: std.AutoHashMap(u64, TrackedMemrefState),
        written_global_slots: std.StringHashMap(void),

        fn init(allocator: std.mem.Allocator) StateSnapshot {
            return .{
                .global_map = std.StringHashMap(z3.Z3_ast).init(allocator),
                .memref_map = std.AutoHashMap(u64, TrackedMemrefState).init(allocator),
                .written_global_slots = std.StringHashMap(void).init(allocator),
            };
        }

        fn deinit(self: *StateSnapshot, allocator: std.mem.Allocator) void {
            var g_it = self.global_map.iterator();
            while (g_it.next()) |entry| allocator.free(entry.key_ptr.*);
            self.global_map.deinit();

            self.memref_map.deinit();

            var w_it = self.written_global_slots.iterator();
            while (w_it.next()) |entry| allocator.free(entry.key_ptr.*);
            self.written_global_slots.deinit();
        }
    };

    fn captureStateSnapshot(self: *Encoder) !StateSnapshot {
        var snap = StateSnapshot.init(self.allocator);
        errdefer snap.deinit(self.allocator);

        var g_it = self.global_map.iterator();
        while (g_it.next()) |entry| {
            try snap.global_map.put(try self.allocator.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }

        var m_it = self.memref_map.iterator();
        while (m_it.next()) |entry| {
            try snap.memref_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var w_it = self.written_global_slots.iterator();
        while (w_it.next()) |entry| {
            try snap.written_global_slots.put(try self.allocator.dupe(u8, entry.key_ptr.*), {});
        }

        return snap;
    }

    fn clearCurrentStateMaps(self: *Encoder) void {
        var g_it = self.global_map.iterator();
        while (g_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.global_map.clearRetainingCapacity();

        self.memref_map.clearRetainingCapacity();

        var w_it = self.written_global_slots.iterator();
        while (w_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.written_global_slots.clearRetainingCapacity();
    }

    fn restoreStateSnapshot(self: *Encoder, snap: *const StateSnapshot) !void {
        self.clearCurrentStateMaps();

        var g_it = snap.global_map.iterator();
        while (g_it.next()) |entry| {
            try self.global_map.put(try self.allocator.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }

        var m_it = snap.memref_map.iterator();
        while (m_it.next()) |entry| {
            try self.memref_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var w_it = snap.written_global_slots.iterator();
        while (w_it.next()) |entry| {
            try self.written_global_slots.put(try self.allocator.dupe(u8, entry.key_ptr.*), {});
        }
    }

    fn stateSnapshotsEquivalent(self: *Encoder, lhs: *const StateSnapshot, rhs: *const StateSnapshot) bool {
        if (lhs.global_map.count() != rhs.global_map.count()) return false;
        if (lhs.memref_map.count() != rhs.memref_map.count()) return false;
        if (lhs.written_global_slots.count() != rhs.written_global_slots.count()) return false;

        var lhs_g = lhs.global_map.iterator();
        while (lhs_g.next()) |entry| {
            const rhs_value = rhs.global_map.get(entry.key_ptr.*) orelse return false;
            if (!self.astEquivalent(entry.value_ptr.*, rhs_value)) return false;
        }

        var lhs_m = lhs.memref_map.iterator();
        while (lhs_m.next()) |entry| {
            const rhs_value = rhs.memref_map.get(entry.key_ptr.*) orelse return false;
            if (!self.astEquivalent(entry.value_ptr.*.value, rhs_value.value)) return false;
            if (!self.astEquivalent(entry.value_ptr.*.initialized, rhs_value.initialized)) return false;
        }

        var lhs_w = lhs.written_global_slots.iterator();
        while (lhs_w.next()) |entry| {
            if (!rhs.written_global_slots.contains(entry.key_ptr.*)) return false;
        }

        return true;
    }

    fn appendUniqueStateName(
        self: *Encoder,
        names: *std.ArrayList([]const u8),
        name: []const u8,
    ) !void {
        for (names.items) |existing| {
            if (std.mem.eql(u8, existing, name)) return;
        }
        try names.append(self.allocator, name);
    }

    fn mergeStateSnapshotsIf(
        self: *Encoder,
        condition: z3.Z3_ast,
        base: *const StateSnapshot,
        then_state: *const StateSnapshot,
        else_state: *const StateSnapshot,
    ) !void {
        self.clearCurrentStateMaps();

        var global_names = std.ArrayList([]const u8){};
        defer global_names.deinit(self.allocator);

        var base_g_it = base.global_map.iterator();
        while (base_g_it.next()) |entry| try self.appendUniqueStateName(&global_names, entry.key_ptr.*);
        var then_g_it = then_state.global_map.iterator();
        while (then_g_it.next()) |entry| try self.appendUniqueStateName(&global_names, entry.key_ptr.*);
        var else_g_it = else_state.global_map.iterator();
        while (else_g_it.next()) |entry| try self.appendUniqueStateName(&global_names, entry.key_ptr.*);

        for (global_names.items) |name| {
            const base_opt = base.global_map.get(name);
            const then_opt = then_state.global_map.get(name);
            const else_opt = else_state.global_map.get(name);
            const fallback = base_opt orelse blk: {
                const branch_val = then_opt orelse else_opt orelse break :blk null;
                const branch_sort = z3.Z3_get_sort(self.context.ctx, branch_val);
                break :blk try self.getOrCreateCurrentGlobal(name, branch_sort);
            } orelse continue;
            const then_val = then_opt orelse fallback;
            const else_val = else_opt orelse fallback;
            const merged = if (then_val == else_val)
                then_val
            else
                self.encodeIte(condition, then_val, else_val);
            try self.putOwnedStringAst(&self.global_map, name, merged);
        }

        var mem_keys = std.AutoHashMap(u64, void).init(self.allocator);
        defer mem_keys.deinit();

        var base_m_it = base.memref_map.iterator();
        while (base_m_it.next()) |entry| try mem_keys.put(entry.key_ptr.*, {});
        var then_m_it = then_state.memref_map.iterator();
        while (then_m_it.next()) |entry| try mem_keys.put(entry.key_ptr.*, {});
        var else_m_it = else_state.memref_map.iterator();
        while (else_m_it.next()) |entry| try mem_keys.put(entry.key_ptr.*, {});

        var key_it = mem_keys.iterator();
        while (key_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const base_opt = base.memref_map.get(key);
            const then_opt = then_state.memref_map.get(key);
            const else_opt = else_state.memref_map.get(key);
            const fallback_value = if (base_opt) |base_state|
                base_state.value
            else if (then_opt) |then_state_entry|
                then_state_entry.value
            else if (else_opt) |else_state_entry|
                else_state_entry.value
            else
                continue;
            const fallback_init = if (base_opt) |base_state|
                base_state.initialized
            else
                self.boolFalse();
            const then_state_entry = then_opt orelse TrackedMemrefState{
                .value = fallback_value,
                .initialized = fallback_init,
            };
            const else_state_entry = else_opt orelse TrackedMemrefState{
                .value = fallback_value,
                .initialized = fallback_init,
            };
            const then_val = then_state_entry.value;
            const else_val = else_state_entry.value;
            const merged = if (then_val == else_val)
                then_val
            else
                self.encodeIte(condition, then_val, else_val);
            try self.memref_map.put(key, .{
                .value = merged,
                .initialized = blk: {
                    const merged_init = self.mergeInitPredicate(condition, then_state_entry.initialized, else_state_entry.initialized);
                    if (self.astSimplifiesToBool(merged_init)) |const_init| {
                        break :blk if (const_init) self.boolTrue() else self.boolFalse();
                    }
                    break :blk merged_init;
                },
            });
        }

        var written_names = std.ArrayList([]const u8){};
        defer written_names.deinit(self.allocator);
        var then_w_it = then_state.written_global_slots.iterator();
        while (then_w_it.next()) |entry| try self.appendUniqueStateName(&written_names, entry.key_ptr.*);
        var else_w_it = else_state.written_global_slots.iterator();
        while (else_w_it.next()) |entry| try self.appendUniqueStateName(&written_names, entry.key_ptr.*);
        for (written_names.items) |name| {
            try self.putOwnedStringVoid(&self.written_global_slots, name);
        }
    }

    fn mergeStateSnapshotsMany(
        self: *Encoder,
        conditions: []const z3.Z3_ast,
        base: *const StateSnapshot,
        branch_states: []const StateSnapshot,
    ) !void {
        self.clearCurrentStateMaps();

        var global_names = std.ArrayList([]const u8){};
        defer global_names.deinit(self.allocator);

        var base_g_it = base.global_map.iterator();
        while (base_g_it.next()) |entry| try self.appendUniqueStateName(&global_names, entry.key_ptr.*);
        for (branch_states) |*branch_state| {
            var branch_it = branch_state.global_map.iterator();
            while (branch_it.next()) |entry| try self.appendUniqueStateName(&global_names, entry.key_ptr.*);
        }

        for (global_names.items) |name| {
            const base_opt = base.global_map.get(name);
            var fallback = base_opt orelse blk: {
                for (branch_states) |*branch_state| {
                    if (branch_state.global_map.get(name)) |value| {
                        const branch_sort = z3.Z3_get_sort(self.context.ctx, value);
                        break :blk try self.getOrCreateCurrentGlobal(name, branch_sort);
                    }
                }
                continue;
            };

            var idx = branch_states.len;
            while (idx > 0) {
                idx -= 1;
                const branch_val = branch_states[idx].global_map.get(name) orelse fallback;
                fallback = if (branch_val == fallback)
                    branch_val
                else
                    self.encodeIte(conditions[idx], branch_val, fallback);
            }

            try self.putOwnedStringAst(&self.global_map, name, fallback);
        }

        var mem_keys = std.AutoHashMap(u64, void).init(self.allocator);
        defer mem_keys.deinit();

        var base_m_it = base.memref_map.iterator();
        while (base_m_it.next()) |entry| try mem_keys.put(entry.key_ptr.*, {});
        for (branch_states) |*branch_state| {
            var branch_it = branch_state.memref_map.iterator();
            while (branch_it.next()) |entry| try mem_keys.put(entry.key_ptr.*, {});
        }

        var key_it = mem_keys.iterator();
        while (key_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const base_opt = base.memref_map.get(key);
            var fallback_value = if (base_opt) |base_state|
                base_state.value
            else blk: {
                for (branch_states) |*branch_state| {
                    if (branch_state.memref_map.get(key)) |state| break :blk state.value;
                }
                continue;
            };
            var fallback_init = if (base_opt) |base_state|
                base_state.initialized
            else
                self.boolFalse();

            var idx = branch_states.len;
            while (idx > 0) {
                idx -= 1;
                const branch_state_entry = branch_states[idx].memref_map.get(key) orelse TrackedMemrefState{
                    .value = fallback_value,
                    .initialized = fallback_init,
                };
                const branch_val = branch_state_entry.value;
                fallback_value = if (branch_val == fallback_value)
                    branch_val
                else
                    self.encodeIte(conditions[idx], branch_val, fallback_value);
                fallback_init = self.mergeInitPredicate(conditions[idx], branch_state_entry.initialized, fallback_init);
            }

            try self.memref_map.put(key, .{
                .value = fallback_value,
                .initialized = blk: {
                    if (self.astSimplifiesToBool(fallback_init)) |const_init| {
                        break :blk if (const_init) self.boolTrue() else self.boolFalse();
                    }
                    break :blk fallback_init;
                },
            });
        }

        var written_names = std.ArrayList([]const u8){};
        defer written_names.deinit(self.allocator);
        for (branch_states) |*branch_state| {
            var written_it = branch_state.written_global_slots.iterator();
            while (written_it.next()) |entry| try self.appendUniqueStateName(&written_names, entry.key_ptr.*);
        }
        for (written_names.items) |name| {
            try self.putOwnedStringVoid(&self.written_global_slots, name);
        }
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

        var written_it = self.written_global_slots.iterator();
        while (written_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.written_global_slots.clearRetainingCapacity();

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
        var tuple_it = self.tuple_values.iterator();
        while (tuple_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.elements);
        }
        self.tuple_values.clearRetainingCapacity();
        self.materialized_calls.clearRetainingCapacity();
        for (self.inline_function_stack.items) |fn_name| {
            self.allocator.free(fn_name);
        }
        self.inline_function_stack.clearRetainingCapacity();
        for (self.quantified_bindings.items) |binding| {
            self.allocator.free(binding.name);
        }
        self.quantified_bindings.clearRetainingCapacity();
        self.return_path_assumptions.clearRetainingCapacity();
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

        var written_it = self.written_global_slots.iterator();
        while (written_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.written_global_slots.deinit();
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
        var struct_decl_it = self.struct_decl_ops.iterator();
        while (struct_decl_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.struct_decl_ops.deinit();
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
        self.return_path_assumptions.deinit(self.allocator);
        var tuple_it = self.tuple_values.iterator();
        while (tuple_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.elements);
        }
        self.tuple_values.deinit();
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

        try self.registerStructDeclAlias(name, csv_builder.items, struct_op);

        const canonical_type_key = try std.fmt.allocPrint(self.allocator, "!ora.struct<\"{s}\">", .{name});
        defer self.allocator.free(canonical_type_key);
        try self.registerStructDeclAlias(canonical_type_key, csv_builder.items, struct_op);
    }

    fn registerStructDeclAlias(self: *Encoder, alias: []const u8, field_names_csv: []const u8, struct_op: mlir.MlirOperation) !void {
        if (!self.struct_field_names_csv.contains(alias)) {
            const key = try self.allocator.dupe(u8, alias);
            errdefer self.allocator.free(key);
            const value = try self.allocator.dupe(u8, field_names_csv);
            errdefer self.allocator.free(value);
            try self.struct_field_names_csv.put(key, value);
        }
        if (!self.struct_decl_ops.contains(alias)) {
            const decl_key = try self.allocator.dupe(u8, alias);
            errdefer self.allocator.free(decl_key);
            try self.struct_decl_ops.put(decl_key, struct_op);
        }
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
        var decl_it = other.struct_decl_ops.iterator();
        while (decl_it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (self.struct_decl_ops.contains(name)) continue;
            const key = try self.allocator.dupe(u8, name);
            try self.struct_decl_ops.put(key, entry.value_ptr.*);
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

    fn copyReturnPathAssumptionsFrom(self: *Encoder, other: *const Encoder) !void {
        for (other.return_path_assumptions.items) |assume| {
            try self.return_path_assumptions.append(self.allocator, assume);
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

    fn unbindValue(self: *Encoder, value: mlir.MlirValue) void {
        _ = self.value_bindings.remove(@intFromPtr(value.ptr));
    }

    fn scalarMemrefStoreValue(
        self: *Encoder,
        store_op: mlir.MlirOperation,
        memref_value: mlir.MlirValue,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        if (!self.operationNameEq(store_op, "memref.store")) return null;
        const num_operands = mlir.oraOperationGetNumOperands(store_op);
        if (num_operands != 2) return null;
        const store_memref = mlir.oraOperationGetOperand(store_op, 1);
        if (!mlir.mlirValueEqual(store_memref, memref_value)) return null;
        return try self.encodeValueWithMode(mlir.oraOperationGetOperand(store_op, 0), mode);
    }

    fn tryRecoverDominatingScalarMemrefStoreInBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        before_op: mlir.MlirOperation,
        memref_value: mlir.MlirValue,
        mode: EncodeMode,
    ) EncodeError!?TrackedMemrefState {
        if (mlir.oraBlockIsNull(block)) return null;
        var current = mlir.oraBlockGetFirstOperation(block);
        var recovered: ?TrackedMemrefState = null;
        while (!mlir.oraOperationIsNull(current) and !mlir.mlirOperationEqual(current, before_op)) {
            if (try self.scalarMemrefStoreValue(current, memref_value, mode)) |stored_value| {
                recovered = .{
                    .value = stored_value,
                    .initialized = self.boolTrue(),
                };
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return recovered;
    }

    fn tryRecoverDominatingScalarMemrefState(
        self: *Encoder,
        load_op: mlir.MlirOperation,
        memref_value: mlir.MlirValue,
        mode: EncodeMode,
    ) EncodeError!?TrackedMemrefState {
        const owner_block = mlir.mlirOperationGetBlock(load_op);
        if (mlir.oraBlockIsNull(owner_block)) return null;

        if (try self.tryRecoverDominatingScalarMemrefStoreInBlock(owner_block, load_op, memref_value, mode)) |state| {
            return state;
        }

        const parent_op = mlir.mlirBlockGetParentOperation(owner_block);
        if (mlir.oraOperationIsNull(parent_op)) return null;
        if (!self.operationNameEq(parent_op, "scf.while")) return null;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(parent_op);
        if (!mlir.oraBlockIsNull(before_block) and before_block.ptr == owner_block.ptr) {
            const parent_block = mlir.mlirOperationGetBlock(parent_op);
            if (mlir.oraBlockIsNull(parent_block)) return null;
            return try self.tryRecoverDominatingScalarMemrefStoreInBlock(parent_block, parent_op, memref_value, mode);
        }

        return null;
    }

    fn encodeShapedReadFromBase(
        self: *Encoder,
        base: z3.Z3_ast,
        indices: []const z3.Z3_ast,
        result_sort: z3.Z3_sort,
    ) EncodeError!z3.Z3_ast {
        var current = base;
        for (indices) |raw_index| {
            const current_sort = z3.Z3_get_sort(self.context.ctx, current);
            if (!self.isArraySort(current_sort)) return error.UnsupportedOperation;
            const index_sort = z3.Z3_get_array_sort_domain(self.context.ctx, current_sort);
            const index = self.coerceAstToSort(raw_index, index_sort);
            current = self.encodeSelect(current, index);
        }
        return self.coerceAstToSort(current, result_sort);
    }

    fn encodeShapedWriteToBase(
        self: *Encoder,
        base: z3.Z3_ast,
        value: z3.Z3_ast,
        indices: []const z3.Z3_ast,
    ) EncodeError!z3.Z3_ast {
        if (indices.len == 0) return value;

        var containers = try self.allocator.alloc(z3.Z3_ast, indices.len);
        defer self.allocator.free(containers);
        var cast_indices = try self.allocator.alloc(z3.Z3_ast, indices.len);
        defer self.allocator.free(cast_indices);

        var cursor = base;
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

        if (indices.len == 1) return updated;

        var rev_index = indices.len - 1;
        while (rev_index > 0) {
            rev_index -= 1;
            const container = containers[rev_index];
            updated = self.encodeStore(container, cast_indices[rev_index], updated);
        }

        return updated;
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
            // Re-assert old(x) == entry(x) on every access so each annotation
            // query that mentions old(...) carries the linkage constraint,
            // even if the old symbol was materialized by an earlier assertion.
            const entry_symbol_existing = try self.getOrCreateGlobalEntry(name, sort);
            const eq_existing = z3.Z3_mk_eq(self.context.ctx, existing, entry_symbol_existing);
            self.addConstraint(eq_existing);
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

    fn degradeToUndef(self: *Encoder, sort: z3.Z3_sort, label: []const u8, id: u64, reason: []const u8) EncodeError!z3.Z3_ast {
        self.recordDegradation(reason);
        return try self.mkUndefValue(sort, label, id);
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
        self.pending_constraints.append(self.allocator, constraint) catch {
            self.recordDegradation("failed to record SMT constraint");
        };
    }

    fn addObligation(self: *Encoder, obligation: z3.Z3_ast) void {
        const guarded_obligation = if (self.return_path_assumptions.items.len == 0)
            obligation
        else
            self.encodeImplies(self.encodeAnd(self.return_path_assumptions.items), obligation);
        self.pending_obligations.append(self.allocator, guarded_obligation) catch {
            self.recordDegradation("failed to record SMT obligation");
        };
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
        // overflow iff rhs != 0 and lhs > MAX / rhs.
        //
        // This is equivalent to the usual mathematical overflow condition for
        // fixed-width unsigned integers, but it is much easier for Z3 than the
        // older recovery-division identity ((lhs * rhs) / lhs) != rhs.
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const rhs_non_zero = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, rhs, zero));

        const width = z3.Z3_get_bv_sort_size(self.context.ctx, sort);
        const max_value = if (width <= 64) blk: {
            const max_u64: u64 = if (width == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(width)) - 1;
            break :blk z3.Z3_mk_unsigned_int64(self.context.ctx, max_u64, sort);
        } else blk: {
            var remaining = width;
            var assembled: ?z3.Z3_ast = null;
            while (remaining > 0) {
                const chunk_width: u32 = if (assembled == null and remaining % 64 != 0) remaining % 64 else @min(remaining, 64);
                const chunk_sort = z3.Z3_mk_bv_sort(self.context.ctx, chunk_width);
                const chunk_value: u64 = if (chunk_width == 64)
                    std.math.maxInt(u64)
                else
                    (@as(u64, 1) << @intCast(chunk_width)) - 1;
                const chunk = z3.Z3_mk_unsigned_int64(self.context.ctx, chunk_value, chunk_sort);
                assembled = if (assembled) |existing|
                    z3.Z3_mk_concat(self.context.ctx, existing, chunk)
                else
                    chunk;
                remaining -= chunk_width;
            }
            break :blk assembled orelse zero;
        };
        const max_div_rhs = z3.Z3_mk_bv_udiv(self.context.ctx, max_value, rhs);
        const lhs_too_large = z3.Z3_mk_bvugt(self.context.ctx, lhs, max_div_rhs);
        return z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ rhs_non_zero, lhs_too_large });
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
        const coerced = self.coerceComparisonOperands(lhs, rhs);
        return switch (op) {
            .Eq => z3.Z3_mk_eq(self.context.ctx, coerced.lhs, coerced.rhs),
            .Ne => z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, coerced.lhs, coerced.rhs)),
            .Lt => z3.Z3_mk_bvult(self.context.ctx, coerced.lhs, coerced.rhs), // Unsigned less than
            .Le => z3.Z3_mk_bvule(self.context.ctx, coerced.lhs, coerced.rhs), // Unsigned less than or equal
            .Gt => z3.Z3_mk_bvugt(self.context.ctx, coerced.lhs, coerced.rhs), // Unsigned greater than
            .Ge => z3.Z3_mk_bvuge(self.context.ctx, coerced.lhs, coerced.rhs), // Unsigned greater than or equal
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
        return z3.Z3_mk_not(self.context.ctx, self.coerceToBool(arg));
    }

    /// Encode implication (A => B)
    pub fn encodeImplies(self: *Encoder, antecedent: z3.Z3_ast, consequent: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_implies(self.context.ctx, self.coerceToBool(antecedent), self.coerceToBool(consequent));
    }

    /// Encode if-then-else (ite)
    pub fn encodeIte(self: *Encoder, condition: z3.Z3_ast, then_expr: z3.Z3_ast, else_expr: z3.Z3_ast) z3.Z3_ast {
        return z3.Z3_mk_ite(self.context.ctx, self.coerceToBool(condition), then_expr, else_expr);
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
                const result_index = self.getResultIndex(defining_op, mlir_value) orelse {
                    self.recordDegradation("failed to resolve defining result index");
                    return error.UnsupportedOperation;
                };
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
        const var_name = try self.resolveValueName(mlir_value, value_id);
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

    fn operationNameEq(_: *Encoder, op: mlir.MlirOperation, expected: []const u8) bool {
        if (mlir.oraOperationIsNull(op)) return false;
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data == null or name_ref.length == 0) return false;
        const op_name = name_ref.data[0..name_ref.length];
        return std.mem.eql(u8, op_name, expected);
    }

    fn tryGetConstBoolValue(self: *Encoder, value: mlir.MlirValue) ?bool {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const value_type = mlir.oraValueGetType(value);
        if ((getTypeBitWidth(self, value_type) orelse 0) != 1) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (!self.operationNameEq(owner, "arith.constant")) return null;
        const value_attr = mlir.oraOperationGetAttributeByName(owner, mlir.oraStringRefCreate("value", 5));
        if (self.parseConstAttrValue(value_attr, 1)) |parsed| {
            return parsed != 0;
        }
        const printed = mlir.oraOperationPrintToString(owner);
        defer if (printed.data != null) @import("mlir_c_api").freeStringRef(printed);
        if (printed.data != null and printed.length > 0) {
            const text = printed.data[0..printed.length];
            if (std.mem.indexOf(u8, text, " true") != null) return true;
            if (std.mem.indexOf(u8, text, " false") != null) return false;
        }
        return null;
    }

    fn tryGetConstIntValue(self: *Encoder, value: mlir.MlirValue) ?u256 {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (!self.operationNameEq(owner, "arith.constant")) return null;
        const value_attr = mlir.oraOperationGetAttributeByName(owner, mlir.oraStringRefCreate("value", 5));
        const value_type = mlir.oraValueGetType(value);
        const width = getTypeBitWidth(self, value_type) orelse 256;
        return self.parseConstAttrValue(value_attr, width);
    }

    fn getStringAttrValue(_: *Encoder, attr: mlir.MlirAttribute) ?[]const u8 {
        if (mlir.oraAttributeIsNull(attr)) return null;
        const value = mlir.oraStringAttrGetValue(attr);
        if (value.data == null or value.length == 0) return null;
        return value.data[0..value.length];
    }

    fn opPrintContains(_: *Encoder, op: mlir.MlirOperation, needle: []const u8) bool {
        const printed = mlir.oraOperationPrintToString(op);
        defer if (printed.data != null) @import("mlir_c_api").freeStringRef(printed);
        if (printed.data == null or printed.length == 0) return false;
        return std.mem.indexOf(u8, printed.data[0..printed.length], needle) != null;
    }

    fn isI1ConstantValue(self: *Encoder, value: mlir.MlirValue) bool {
        if (!mlir.oraValueIsAOpResult(value)) return false;
        const owner = mlir.oraOpResultGetOwner(value);
        if (!self.operationNameEq(owner, "arith.constant")) return false;
        const value_type = mlir.oraValueGetType(value);
        return (getTypeBitWidth(self, value_type) orelse 0) == 1;
    }

    fn tryEncodeCheckedUnsignedMulAssert(self: *Encoder, condition_value: mlir.MlirValue, mode: EncodeMode) EncodeError!?z3.Z3_ast {
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const xori_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(xori_op, "arith.xori")) return null;
        if (mlir.oraOperationGetNumOperands(xori_op) != 2) return null;

        var and_value: ?mlir.MlirValue = null;
        var found_bool_inversion = false;
        for (0..2) |i| {
            const operand = mlir.oraOperationGetOperand(xori_op, i);
            if (self.isI1ConstantValue(operand)) {
                found_bool_inversion = true;
            } else if (mlir.oraValueIsAOpResult(operand) and self.operationNameEq(mlir.oraOpResultGetOwner(operand), "arith.andi")) {
                and_value = operand;
            }
        }
        if (!found_bool_inversion or and_value == null) return null;

        const and_op = mlir.oraOpResultGetOwner(and_value.?);
        if (mlir.oraOperationGetNumOperands(and_op) != 2) return null;

        var overflow_cmp: ?mlir.MlirOperation = null;
        var rhs_non_zero_cmp: ?mlir.MlirOperation = null;
        for (0..2) |i| {
            const operand = mlir.oraOperationGetOperand(and_op, i);
            if (!mlir.oraValueIsAOpResult(operand)) return null;
            const owner = mlir.oraOpResultGetOwner(operand);
            if (!self.operationNameEq(owner, "arith.cmpi")) return null;

            const lhs = mlir.oraOperationGetOperand(owner, 0);
            const rhs = mlir.oraOperationGetOperand(owner, 1);
            if (self.tryGetConstBoolValue(lhs) != null or self.tryGetConstBoolValue(rhs) != null) return null;

            const lhs_const_zero = self.tryGetConstIntValue(lhs);
            const rhs_const_zero = self.tryGetConstIntValue(rhs);
            if ((lhs_const_zero != null and lhs_const_zero.? == 0) or (rhs_const_zero != null and rhs_const_zero.? == 0)) {
                rhs_non_zero_cmp = owner;
                continue;
            }

            var div_value: ?mlir.MlirValue = null;
            var base_value: ?mlir.MlirValue = null;
            if (mlir.oraValueIsAOpResult(lhs) and self.operationNameEq(mlir.oraOpResultGetOwner(lhs), "arith.divui")) {
                div_value = lhs;
                base_value = rhs;
            } else if (mlir.oraValueIsAOpResult(rhs) and self.operationNameEq(mlir.oraOpResultGetOwner(rhs), "arith.divui")) {
                div_value = rhs;
                base_value = lhs;
            }
            if (div_value == null or base_value == null) return null;
            overflow_cmp = owner;
            if (!mlir.mlirValueEqual(base_value.?, mlir.oraOperationGetOperand(mlir.oraOpResultGetOwner(div_value.?), 0)) and !mlir.mlirValueEqual(base_value.?, mlir.oraOperationGetOperand(mlir.oraOpResultGetOwner(div_value.?), 1))) {
                // Defer exact base/mul matching to the div inspection below.
            }
        }
        if (overflow_cmp == null or rhs_non_zero_cmp == null) return null;

        const rhs_non_zero_op = rhs_non_zero_cmp.?;
        const overflow_op = overflow_cmp.?;
        const nz_lhs = mlir.oraOperationGetOperand(rhs_non_zero_op, 0);
        const nz_rhs = mlir.oraOperationGetOperand(rhs_non_zero_op, 1);
        const lhs_is_zero = if (self.tryGetConstIntValue(nz_lhs)) |value| value == 0 else false;
        const rhs_is_zero = if (self.tryGetConstIntValue(nz_rhs)) |value| value == 0 else false;
        const rhs_value =
            if (lhs_is_zero and !rhs_is_zero)
                nz_rhs
            else if (rhs_is_zero and !lhs_is_zero)
                nz_lhs
            else
                return null;

        const ov_lhs = mlir.oraOperationGetOperand(overflow_op, 0);
        const ov_rhs = mlir.oraOperationGetOperand(overflow_op, 1);
        var div_value: ?mlir.MlirValue = null;
        var lhs_value: ?mlir.MlirValue = null;
        if (mlir.oraValueIsAOpResult(ov_lhs) and self.operationNameEq(mlir.oraOpResultGetOwner(ov_lhs), "arith.divui")) {
            div_value = ov_lhs;
            lhs_value = ov_rhs;
        } else if (mlir.oraValueIsAOpResult(ov_rhs) and self.operationNameEq(mlir.oraOpResultGetOwner(ov_rhs), "arith.divui")) {
            div_value = ov_rhs;
            lhs_value = ov_lhs;
        } else {
            return null;
        }

        const div_op = mlir.oraOpResultGetOwner(div_value.?);
        if (mlir.oraOperationGetNumOperands(div_op) != 2) return null;
        const mul_value = mlir.oraOperationGetOperand(div_op, 0);
        const div_rhs_value = mlir.oraOperationGetOperand(div_op, 1);
        if (!mlir.mlirValueEqual(div_rhs_value, rhs_value)) return null;
        if (!mlir.oraValueIsAOpResult(mul_value)) return null;

        const mul_op = mlir.oraOpResultGetOwner(mul_value);
        if (!self.operationNameEq(mul_op, "arith.muli")) return null;
        if (mlir.oraOperationGetNumOperands(mul_op) != 2) return null;
        const mul_lhs = mlir.oraOperationGetOperand(mul_op, 0);
        const mul_rhs = mlir.oraOperationGetOperand(mul_op, 1);

        var matched_lhs: ?mlir.MlirValue = null;
        if (mlir.mlirValueEqual(mul_lhs, lhs_value.?)) {
            if (!mlir.mlirValueEqual(mul_rhs, rhs_value)) return null;
            matched_lhs = mul_lhs;
        } else if (mlir.mlirValueEqual(mul_rhs, lhs_value.?)) {
            if (!mlir.mlirValueEqual(mul_lhs, rhs_value)) return null;
            matched_lhs = mul_rhs;
        } else {
            return null;
        }

        const lhs_ast = try self.encodeValueWithMode(matched_lhs.?, mode);
        const rhs_ast = try self.encodeValueWithMode(rhs_value, mode);
        const overflow = self.checkMulOverflow(lhs_ast, rhs_ast);
        return z3.Z3_mk_not(self.context.ctx, overflow);
    }

    fn tryEncodeCheckedAddAssert(self: *Encoder, condition_value: mlir.MlirValue, mode: EncodeMode) EncodeError!?z3.Z3_ast {
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const xori_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(xori_op, "arith.xori")) return null;
        if (mlir.oraOperationGetNumOperands(xori_op) != 2) return null;

        var cmp_value: ?mlir.MlirValue = null;
        var found_bool_inversion = false;
        for (0..2) |i| {
            const operand = mlir.oraOperationGetOperand(xori_op, i);
            if (self.isI1ConstantValue(operand)) {
                found_bool_inversion = true;
            } else if (mlir.oraValueIsAOpResult(operand) and self.operationNameEq(mlir.oraOpResultGetOwner(operand), "arith.cmpi")) {
                cmp_value = operand;
            }
        }
        if (!found_bool_inversion or cmp_value == null) return null;

        const cmp_op = mlir.oraOpResultGetOwner(cmp_value.?);
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;
        const predicate = self.getCmpPredicate(cmp_op) catch return null;
        if (predicate != 6) return null; // ult

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        if (!mlir.oraValueIsAOpResult(cmp_lhs)) return null;
        const add_op = mlir.oraOpResultGetOwner(cmp_lhs);
        if (!self.operationNameEq(add_op, "arith.addi")) return null;
        if (mlir.oraOperationGetNumOperands(add_op) != 2) return null;
        const lhs_value = mlir.oraOperationGetOperand(add_op, 0);
        const rhs_value = mlir.oraOperationGetOperand(add_op, 1);
        if (!mlir.mlirValueEqual(cmp_rhs, lhs_value) and !mlir.mlirValueEqual(cmp_rhs, rhs_value)) return null;

        const lhs_ast = try self.encodeValueWithMode(lhs_value, mode);
        const rhs_ast = try self.encodeValueWithMode(rhs_value, mode);
        const overflow = self.checkAddOverflow(lhs_ast, rhs_ast);
        return z3.Z3_mk_not(self.context.ctx, overflow);
    }

    fn tryEncodeCheckedSubAssert(self: *Encoder, condition_value: mlir.MlirValue, mode: EncodeMode) EncodeError!?z3.Z3_ast {
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const xori_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(xori_op, "arith.xori")) return null;
        if (mlir.oraOperationGetNumOperands(xori_op) != 2) return null;

        var cmp_value: ?mlir.MlirValue = null;
        var found_bool_inversion = false;
        for (0..2) |i| {
            const operand = mlir.oraOperationGetOperand(xori_op, i);
            if (self.isI1ConstantValue(operand)) {
                found_bool_inversion = true;
            } else if (mlir.oraValueIsAOpResult(operand) and self.operationNameEq(mlir.oraOpResultGetOwner(operand), "arith.cmpi")) {
                cmp_value = operand;
            }
        }
        if (!found_bool_inversion or cmp_value == null) return null;

        const cmp_op = mlir.oraOpResultGetOwner(cmp_value.?);
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;
        const predicate = self.getCmpPredicate(cmp_op) catch return null;
        if (predicate != 6) return null; // ult

        const lhs_value = mlir.oraOperationGetOperand(cmp_op, 0);
        const rhs_value = mlir.oraOperationGetOperand(cmp_op, 1);

        const lhs_ast = try self.encodeValueWithMode(lhs_value, mode);
        const rhs_ast = try self.encodeValueWithMode(rhs_value, mode);
        const underflow = self.checkSubUnderflow(lhs_ast, rhs_ast);
        return z3.Z3_mk_not(self.context.ctx, underflow);
    }

    pub fn tryEncodeAssertCondition(self: *Encoder, assert_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!?z3.Z3_ast {
        if (!self.operationNameEq(assert_op, "ora.assert") and !self.operationNameEq(assert_op, "cf.assert")) return null;
        if (mlir.oraOperationGetNumOperands(assert_op) < 1) return null;
        const message_attr = mlir.oraOperationGetAttributeByName(assert_op, mlir.oraStringRefCreate("message", 7));
        const has_checked_mul_message =
            if (self.getStringAttrValue(message_attr)) |message|
                std.mem.eql(u8, message, "checked multiplication overflow")
            else
                self.opPrintContains(assert_op, "\"checked multiplication overflow\"");
        if (has_checked_mul_message) {
            const condition_value = mlir.oraOperationGetOperand(assert_op, 0);
            const specialized = try self.tryEncodeCheckedUnsignedMulAssert(condition_value, mode);
            if (specialized == null) {
                const debug_enabled = blk: {
                    const value = std.process.getEnvVarOwned(self.allocator, "ORA_Z3_DEBUG_ASSERT_SIMPLIFY") catch null;
                    defer if (value) |buf| self.allocator.free(buf);
                    break :blk if (value) |buf|
                        std.mem.eql(u8, buf, "1") or std.ascii.eqlIgnoreCase(buf, "true")
                    else
                        false;
                };
                if (debug_enabled) {
                    const printed = mlir.oraOperationPrintToString(assert_op);
                    defer if (printed.data != null) @import("mlir_c_api").freeStringRef(printed);
                    if (printed.data != null and printed.length > 0) {
                        std.debug.print("smt-debug: assert simplify miss: {s}\n", .{printed.data[0..printed.length]});
                    }
                }
            }
            return specialized;
        }

        const has_checked_add_message =
            if (self.getStringAttrValue(message_attr)) |message|
                std.mem.eql(u8, message, "checked addition overflow")
            else
                self.opPrintContains(assert_op, "\"checked addition overflow\"");
        if (has_checked_add_message) {
            const condition_value = mlir.oraOperationGetOperand(assert_op, 0);
            return try self.tryEncodeCheckedAddAssert(condition_value, mode);
        }

        const has_checked_sub_message =
            if (self.getStringAttrValue(message_attr)) |message|
                std.mem.eql(u8, message, "checked subtraction overflow")
            else
                self.opPrintContains(assert_op, "\"checked subtraction overflow\"");
        if (has_checked_sub_message) {
            const condition_value = mlir.oraOperationGetOperand(assert_op, 0);
            return try self.tryEncodeCheckedSubAssert(condition_value, mode);
        }
        return null;
    }

    /// Try to resolve a human-readable name for an MLIR value.
    /// For function parameters (block arguments of func ops), extracts the
    /// parameter name from the enclosing function's printed form.
    /// Falls back to "v_{pointer_id}".
    fn resolveValueName(self: *Encoder, value: mlir.MlirValue, value_id: u64) ![]u8 {
        if (mlir.mlirValueIsABlockArgument(value)) {
            const owner_block = mlir.mlirBlockArgumentGetOwner(value);
            if (!mlir.oraBlockIsNull(owner_block)) {
                const parent_op = mlir.mlirBlockGetParentOperation(owner_block);
                if (!mlir.oraOperationIsNull(parent_op)) {
                    const op_name_ref = mlir.oraOperationGetName(parent_op);
                    if (op_name_ref.data != null and op_name_ref.length > 0) {
                        const op_name = op_name_ref.data[0..op_name_ref.length];
                        if (std.mem.eql(u8, op_name, "func.func") or std.mem.eql(u8, op_name, "ora.func_decl")) {
                            const arg_no = mlir.mlirBlockArgumentGetArgNumber(value);
                            // Try to read parameter name from printed function signature
                            if (self.extractFuncParamName(parent_op, @intCast(arg_no))) |name| {
                                return std.fmt.allocPrint(self.allocator, "{s}", .{name});
                            }
                            // Fallback: use positional name
                            return std.fmt.allocPrint(self.allocator, "arg{d}", .{arg_no});
                        }
                    }
                }
            }
        }
        return std.fmt.allocPrint(self.allocator, "v_{d}", .{value_id});
    }

    /// Extract the parameter name at a given index from a func op's sym_name and
    /// printed representation. We look for the ora.param_names attribute first,
    /// then fall back to parsing the MLIR printed form.
    fn extractFuncParamName(self: *Encoder, func_op: mlir.MlirOperation, arg_index: u32) ?[]const u8 {
        _ = self;
        // Try ora.param_names array attribute (if present on the function)
        const param_names_attr = mlir.oraOperationGetAttributeByName(
            func_op,
            mlir.oraStringRefCreate("ora.param_names", 15),
        );
        if (!mlir.oraAttributeIsNull(param_names_attr)) {
            // It's an array attribute; extract the element at arg_index
            const num_elements: u32 = @intCast(mlir.oraArrayAttrGetNumElements(param_names_attr));
            if (arg_index < num_elements) {
                const elem = mlir.oraArrayAttrGetElement(param_names_attr, @intCast(arg_index));
                if (!mlir.oraAttributeIsNull(elem)) {
                    const str = mlir.oraStringAttrGetValue(elem);
                    if (str.data != null and str.length > 0) {
                        return str.data[0..str.length];
                    }
                }
            }
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
            const unsigned_cmp = mlir_helpers.getScfForUnsignedCmp(parent_op);
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
            return;
        }

        if (std.mem.eql(u8, op_name, "scf.while")) {
            const after_block = mlir.oraScfWhileOpGetAfterBlock(parent_op);
            if (mlir.oraBlockIsNull(after_block) or owner_block.ptr != after_block.ptr) return;

            const arg_no = mlir.mlirBlockArgumentGetArgNumber(value);
            const before_block = mlir.oraScfWhileOpGetBeforeBlock(parent_op);
            if (mlir.oraBlockIsNull(before_block)) return;

            var op = mlir.oraBlockGetFirstOperation(before_block);
            while (!mlir.oraOperationIsNull(op)) {
                const cond_name_ref = mlir.oraOperationGetName(op);
                defer @import("mlir_c_api").freeStringRef(cond_name_ref);
                const cond_name = if (cond_name_ref.data == null or cond_name_ref.length == 0)
                    ""
                else
                    cond_name_ref.data[0..cond_name_ref.length];

                if (std.mem.eql(u8, cond_name, "scf.condition")) {
                    const num_operands = mlir.oraOperationGetNumOperands(op);
                    if (num_operands < arg_no + 2) return;

                    const continue_value = mlir.oraOperationGetOperand(op, 0);
                    const carried_value = mlir.oraOperationGetOperand(op, @as(usize, @intCast(arg_no + 1)));
                    const continue_ast = try self.encodeValueWithMode(continue_value, mode);
                    const carried_ast = try self.encodeValueWithMode(carried_value, mode);

                    self.addConstraint(self.coerceBoolean(continue_ast));
                    self.addConstraint(z3.Z3_mk_eq(self.context.ctx, ast, carried_ast));
                    return;
                }
                op = mlir.oraOperationGetNextInBlock(op);
            }
        }
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

        if (std.mem.eql(u8, op_name, "ora.switch_expr") or std.mem.eql(u8, op_name, "ora.switch")) {
            return try self.encodeOraSwitchExprResult(mlir_op, result_index, mode);
        }

        if (std.mem.eql(u8, op_name, "scf.while") or
            std.mem.eql(u8, op_name, "scf.for"))
        {
            if (std.mem.eql(u8, op_name, "scf.while")) {
                const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
                const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
                const op_id = @intFromPtr(mlir_op.ptr);
                return (try self.tryExtractZeroIterationScfWhileResult(mlir_op, result_index, mode)) orelse
                    (try self.tryExtractCanonicalUnsignedScfWhileResult(mlir_op, result_index, mode)) orelse
                    (try self.tryExtractCanonicalSignedScfWhileResult(mlir_op, result_index, mode)) orelse
                    (try self.tryExtractCanonicalIncrementScfWhileResult(mlir_op, result_index, mode)) orelse
                    (try self.tryExtractCanonicalDecrementScfWhileResult(mlir_op, result_index, mode)) orelse
                    (try self.tryExtractFiniteScfWhileResult(mlir_op, result_index, mode)) orelse
                    try self.degradeToUndef(result_sort, "scf_while_result", op_id, "scf.while result requires loop summary");
            }

            if (std.mem.eql(u8, op_name, "scf.for")) {
                const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
                const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
                const op_id = @intFromPtr(mlir_op.ptr);
                if (try self.tryExtractCanonicalScfForDerivedResult(mlir_op, result_index, mode)) |derived_result| {
                    return derived_result;
                }
                if (try self.tryExtractCanonicalIncrementScfForResult(mlir_op, result_index, mode)) |increment_result| {
                    return increment_result;
                }
                if (try self.tryExtractCanonicalDecrementScfForResult(mlir_op, result_index, mode)) |decrement_result| {
                    return decrement_result;
                }
                if (try self.tryExtractIdentityScfForResult(mlir_op, result_index, mode)) |identity_result| {
                    return identity_result;
                }
                return (try self.tryExtractFiniteScfForResult(mlir_op, result_index, mode)) orelse
                    try self.degradeToUndef(result_sort, "scf_for_result", op_id, "scf.for result requires loop summary");
            }

            const operands = try self.encodeOperationOperandsWithMode(mlir_op, mode);
            defer self.allocator.free(operands);
            return try self.encodeStructuredControlResult(mlir_op, op_name, operands, result_index);
        }

        if (std.mem.eql(u8, op_name, "scf.execute_region")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
            const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
            const op_id = @intFromPtr(mlir_op.ptr);
            return (try self.extractRegionYield(mlir_op, 0, result_index, mode)) orelse
                try self.degradeToUndef(result_sort, "execute_region_result", op_id, "scf.execute_region result missing region yield");
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
            const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
            const op_id = @intFromPtr(mlir_op.ptr);
            if (try self.tryStmtAlwaysEntersCatch(mlir_op, mode)) {
                return (try self.extractRegionYield(mlir_op, 1, result_index, mode)) orelse
                    try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result missing catch-region yield");
            }
            if (!self.tryStmtMayEnterCatch(mlir_op)) {
                return (try self.extractRegionYield(mlir_op, 0, result_index, mode)) orelse
                    try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result missing try-region yield");
            }
            if (try self.tryExtractTryRegionCatchPredicate(mlir_op, mode)) |catch_pred| {
                if (try self.extractRegionYieldValue(mlir_op, 0, result_index)) |yield_value| {
                    if (try self.trySummarizeTryValue(yield_value, mode)) |summary| {
                        if (try self.extractRegionYield(mlir_op, 1, result_index, mode)) |catch_expr| {
                            return try self.encodeControlFlow("scf.if", catch_pred, catch_expr, summary.ok_expr);
                        }
                    }
                }
            }
            if (try self.tryExtractDirectErrorUnwrapTryStmtResult(mlir_op, result_index, mode)) |encoded| {
                return encoded;
            }
            return (try self.tryExtractEquivalentTryStmtResult(mlir_op, result_index, mode)) orelse
                try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result requires exact catch summary");
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
        return try self.degradeToUndef(result_sort, label, op_id, "failed to encode non-zero operation result precisely");
    }

    fn encodeScfIfResult(self: *Encoder, mlir_op: mlir.MlirOperation, result_index: u32, mode: EncodeMode) EncodeError!z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        if (num_operands < 1) return error.InvalidOperandCount;
        const condition_value = mlir.oraOperationGetOperand(mlir_op, 0);
        const condition = try self.encodeValueWithMode(condition_value, mode);
        const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
        const result_type = mlir.oraValueGetType(result_value);
        const result_sort = try self.encodeMLIRType(result_type);
        const op_id = @intFromPtr(mlir_op.ptr);
        const then_expr = try self.extractRegionYield(mlir_op, 0, result_index, mode);
        const else_expr = try self.extractRegionYield(mlir_op, 1, result_index, mode);
        if (then_expr == null or else_expr == null) {
            return try self.degradeToUndef(result_sort, "scf_if_result", op_id, "scf.if result missing branch yield");
        }
        return try self.encodeControlFlow("scf.if", condition, then_expr, else_expr);
    }

    fn encodeOraSwitchExprResult(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(mlir_op);
        if (num_operands < 1) return error.InvalidOperandCount;

        const scrutinee_value = mlir.oraOperationGetOperand(mlir_op, 0);
        const scrutinee = try self.encodeValueWithMode(scrutinee_value, mode);

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(mlir_op));
        var metadata = try self.getSwitchCaseMetadata(mlir_op, num_regions);
        defer metadata.deinit(self.allocator);

        const result_value = mlir.oraOperationGetResult(mlir_op, @intCast(result_index));
        const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
        const op_id = @intFromPtr(mlir_op.ptr);
        var branch_exprs = try self.allocator.alloc(z3.Z3_ast, num_regions);
        defer self.allocator.free(branch_exprs);
        var branch_predicates = try self.allocator.alloc(z3.Z3_ast, num_regions);
        defer self.allocator.free(branch_predicates);

        var default_expr: ?z3.Z3_ast = null;
        if (metadata.default_case_index) |default_case_index| {
            if (default_case_index >= 0) {
                default_expr = try self.extractRegionYield(mlir_op, @intCast(default_case_index), result_index, mode);
            }
        }

        var remaining = self.encodeBoolConstant(true);

        var region_index = num_regions;
        while (region_index > 0) {
            region_index -= 1;

            const branch_expr = (try self.extractRegionYield(mlir_op, @intCast(region_index), result_index, mode)) orelse default_expr orelse
                try self.degradeToUndef(result_sort, "switch_expr", op_id, "switch expression result initialized from unconstrained fallback");
            const raw_predicate = try self.buildOraSwitchCasePredicate(
                scrutinee,
                metadata.case_kinds,
                metadata.case_values,
                metadata.range_starts,
                metadata.range_ends,
                region_index,
            );
            const effective_predicate = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                remaining
            else
                self.encodeAnd(&.{ remaining, raw_predicate });
            branch_exprs[region_index] = branch_expr;
            branch_predicates[region_index] = effective_predicate;

            if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
            }
        }

        const exhaustive_without_default = metadata.default_case_index == null and self.switchCasesCoverI1Domain(scrutinee_value, metadata);

        var merged = blk: {
            if (default_expr) |expr| break :blk expr;
            if (exhaustive_without_default or self.astEquivalent(remaining, self.encodeBoolConstant(false))) {
                var chosen: ?z3.Z3_ast = null;
                var idx: usize = 0;
                while (idx < num_regions) : (idx += 1) {
                    if (!self.astEquivalent(branch_predicates[idx], self.encodeBoolConstant(false))) {
                        chosen = branch_exprs[idx];
                        break;
                    }
                }
                if (chosen) |expr| break :blk expr;
            }
            break :blk try self.degradeToUndef(result_sort, "switch_expr", op_id, "switch expression result initialized from unconstrained fallback");
        };

        region_index = num_regions;
        while (region_index > 0) {
            region_index -= 1;
            merged = self.encodeIte(branch_predicates[region_index], branch_exprs[region_index], merged);
        }

        return merged;
    }

    fn switchCasesCoverI1Domain(
        self: *Encoder,
        scrutinee_value: mlir.MlirValue,
        metadata: SwitchCaseMetadata,
    ) bool {
        const scrutinee_ty = mlir.oraValueGetType(scrutinee_value);
        if ((self.getTypeBitWidth(scrutinee_ty) orelse 0) != 1) return false;

        var saw_zero = false;
        var saw_one = false;
        for (metadata.case_kinds, metadata.case_values) |kind, value| {
            if (kind != 0) return false;
            if (value == 0) saw_zero = true;
            if (value == 1) saw_one = true;
        }
        return saw_zero and saw_one;
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
        self.recordDegradation("structured control result encoded via opaque summary");
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
            const index_ty = mlir.oraIndexTypeCreate(type_ctx);
            if (!mlir.oraTypeIsNull(index_ty) and mlir.oraTypeEqual(mlir_type, index_ty)) {
                return self.mkBitVectorSort(256);
            }
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
            const predicate = try self.getCmpPredicate(mlir_op);
            return try self.encodeCmpOp(predicate, operands);
        }

        if (std.mem.eql(u8, op_name, "ora.cmp")) {
            if (operands.len < 2) return error.InvalidOperandCount;
            const predicate = self.getStringAttr(mlir_op, "predicate") orelse {
                self.recordDegradation("ora.cmp missing predicate");
                return error.UnsupportedOperation;
            };
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
            const value = self.parseConstAttrValue(value_attr, width) orelse blk: {
                self.recordDegradation("failed to parse MLIR constant attribute");
                break :blk if (is_addr) 0 else self.getConstantValue(mlir_op, width);
            };
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

        if (std.mem.eql(u8, op_name, "ora.add_wrapping") or
            std.mem.eql(u8, op_name, "ora.sub_wrapping") or
            std.mem.eql(u8, op_name, "ora.mul_wrapping"))
        {
            if (operands.len < 2) return error.InvalidOperandCount;
            const arith_op = if (std.mem.eql(u8, op_name, "ora.add_wrapping"))
                ArithmeticOp.Add
            else if (std.mem.eql(u8, op_name, "ora.sub_wrapping"))
                ArithmeticOp.Sub
            else
                ArithmeticOp.Mul;
            return self.encodeArithmeticOp(arith_op, operands[0], operands[1]);
        }

        if (std.mem.eql(u8, op_name, "ora.shl_wrapping") or
            std.mem.eql(u8, op_name, "ora.shr_wrapping"))
        {
            if (operands.len < 2) return error.InvalidOperandCount;
            return if (std.mem.eql(u8, op_name, "ora.shl_wrapping"))
                z3.Z3_mk_bvshl(self.context.ctx, operands[0], operands[1])
            else
                z3.Z3_mk_bvlshr(self.context.ctx, operands[0], operands[1]);
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
                try self.markGlobalSlotWritten(global_name);
            } else {
                const key = try self.allocator.dupe(u8, global_name);
                try self.global_map.put(key, operands[0]);
                try self.markGlobalSlotWritten(global_name);
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
                self.addArrayStoreFrameConstraint(operands[0], key, stored, op_id) catch {
                    self.recordDegradation("failed to encode map store frame constraint");
                };
                if (mode == .Current) {
                    const map_operand = mlir.oraOperationGetOperand(mlir_op, 0);
                    const map_operand_id = @intFromPtr(map_operand.ptr);
                    try self.value_bindings.put(map_operand_id, stored);
                    try self.value_map.put(map_operand_id, stored);
                    if (self.resolveGlobalNameFromMapOperand(map_operand)) |global_name| {
                        if (self.global_map.getPtr(global_name)) |existing| {
                            existing.* = stored;
                            try self.markGlobalSlotWritten(global_name);
                        } else {
                            const global_key = try self.allocator.dupe(u8, global_name);
                            try self.global_map.put(global_key, stored);
                            try self.markGlobalSlotWritten(global_name);
                        }
                    } else {
                        // Nested map write (e.g., root[a][b] = v) may target a map_get
                        // result. Rebuild and thread the root global map update so later
                        // reads observe the inner write.
                        try self.tryUpdateNestedGlobalMapStore(map_operand, stored, mode, op_id);
                    }
                }
                const num_results = mlir.oraOperationGetNumResults(mlir_op);
                if (num_results > 0) return stored;
                return self.encodeBoolConstant(true);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.tuple_create")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results < 1) return error.UnsupportedOperation;
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_id = @intFromPtr(result_value.ptr);
            const elements = try self.allocator.dupe(z3.Z3_ast, operands);
            try self.putTupleValue(result_id, elements);
            return if (operands.len > 0) operands[0] else self.encodeBoolConstant(true);
        }

        if (std.mem.eql(u8, op_name, "ora.tuple_extract")) {
            const tuple_operand = mlir.oraOperationGetOperand(mlir_op, 0);
            const tuple_id = @intFromPtr(tuple_operand.ptr);
            if (self.tuple_values.get(tuple_id)) |tuple| {
                const index_attr = mlir.oraOperationGetAttributeByName(mlir_op, .{ .data = "index".ptr, .length = 5 });
                if (!mlir.oraAttributeIsNull(index_attr)) {
                    const index: usize = @intCast(mlir.oraIntegerAttrGetValueSInt(index_attr));
                    if (index < tuple.elements.len) return tuple.elements[index];
                }
            }
            return error.UnsupportedOperation;
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
        // Handles checked-arithmetic overflow assertions and runtime guards.
        //
        // Note: cf.assert is handled in verification extraction where tagged
        // requires/ensures are interpreted with their dedicated semantics.
        if (std.mem.eql(u8, op_name, "ora.assert")) {
            if (operands.len >= 1) {
                const condition = if (try self.tryEncodeAssertCondition(mlir_op, .Current)) |specialized|
                    specialized
                else
                    self.coerceToBool(operands[0]);
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
                var field_names_owned: ?[]u8 = null;
                defer if (field_names_owned) |owned| self.allocator.free(owned);
                if (field_names_str == null) {
                    const names_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("ora.field_names", 15));
                    field_names_owned = try self.buildFieldNamesCsvFromAttr(names_attr);
                    field_names_str = field_names_owned;
                }
                if (field_names_str == null) {
                    if (self.getStringAttr(mlir_op, "struct_name")) |struct_name| {
                        field_names_str = self.struct_field_names_csv.get(struct_name);
                    }
                }
                if (field_names_str == null) {
                    field_names_owned = try self.lookupStructFieldNamesInScope(mlir_op, result_type);
                    field_names_str = field_names_owned;
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
                var field_names_str = self.getStringAttr(mlir_op, "ora.field_names");
                var field_names_owned: ?[]u8 = null;
                defer if (field_names_owned) |owned| self.allocator.free(owned);
                if (field_names_str == null) {
                    const names_attr = mlir.oraOperationGetAttributeByName(mlir_op, mlir.oraStringRefCreate("ora.field_names", 15));
                    field_names_owned = try self.buildFieldNamesCsvFromAttr(names_attr);
                    field_names_str = field_names_owned;
                }
                if (field_names_str == null) {
                    field_names_owned = try self.lookupStructFieldNamesInScope(mlir_op, result_type);
                    field_names_str = field_names_owned;
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

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            return try self.encodeFuncCall(mlir_op, operands, mode);
        }

        if (std.mem.eql(u8, op_name, "ora.error.return")) {
            return try self.encodeErrorReturnOp(operands, mlir_op);
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
                const field_name = self.getStringAttr(mlir_op, "field_name") orelse {
                    self.recordDegradation("struct_field_extract missing field_name");
                    return error.UnsupportedOperation;
                };
                const struct_sort = z3.Z3_get_sort(self.context.ctx, operands[0]);
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                return try self.applyFieldFunction(field_name, struct_sort, result_sort, operands[0]);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.struct_field_update")) {
            if (operands.len >= 2 and mlir.oraOperationGetNumResults(mlir_op) >= 1) {
                const source_value = mlir.oraOperationGetOperand(mlir_op, 0);
                const field_name = self.getStringAttr(mlir_op, "field_name") orelse {
                    self.recordDegradation("struct_field_update missing field_name");
                    return error.UnsupportedOperation;
                };
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const source_sort = z3.Z3_get_sort(self.context.ctx, operands[0]);
                const op_id = @intFromPtr(mlir_op.ptr);
                const struct_var_name = try std.fmt.allocPrint(self.allocator, "struct_field_update_{d}", .{op_id});
                defer self.allocator.free(struct_var_name);
                const struct_val = try self.mkVariable(struct_var_name, result_sort);
                const field_sort = z3.Z3_get_sort(self.context.ctx, operands[1]);
                const accessor = try self.applyFieldFunction(field_name, result_sort, field_sort, struct_val);
                const eq = z3.Z3_mk_eq(self.context.ctx, accessor, operands[1]);
                self.addConstraint(eq);

                var field_names_csv = self.lookupStructFieldNames(result_type) catch blk: {
                    self.recordDegradation("failed to resolve struct field names for struct update");
                    break :blk null;
                };
                var field_names_csv_owned = false;
                defer if (field_names_csv_owned and field_names_csv != null) self.allocator.free(field_names_csv.?);
                if (field_names_csv == null) {
                    field_names_csv = self.lookupStructFieldNamesInScope(mlir_op, result_type) catch blk: {
                        self.recordDegradation("failed to resolve scoped struct field names for struct update");
                        break :blk null;
                    };
                    field_names_csv_owned = field_names_csv != null;
                }
                if (field_names_csv == null) {
                    field_names_csv = self.tryLookupStructFieldNamesFromValue(source_value) catch blk: {
                        self.recordDegradation("failed to recover struct field names from source value");
                        break :blk null;
                    };
                    field_names_csv_owned = field_names_csv != null;
                }
                if (field_names_csv) |csv| {
                    var struct_update_failed = false;
                    var it = std.mem.splitScalar(u8, csv, ',');
                    var field_index: usize = 0;
                    while (it.next()) |part| : (field_index += 1) {
                        const trimmed = std.mem.trim(u8, part, " \t\n\r");
                        if (trimmed.len == 0 or std.mem.eql(u8, trimmed, field_name)) continue;
                        var unchanged_type = self.lookupStructFieldType(result_type, field_index) catch blk: {
                            self.recordDegradation("failed to resolve struct field type for struct update");
                            break :blk mlir.MlirType{ .ptr = null };
                        };
                        if (mlir.oraTypeIsNull(unchanged_type)) {
                            unchanged_type = self.lookupStructFieldTypeInScope(mlir_op, result_type, field_index);
                        }
                        if (mlir.oraTypeIsNull(unchanged_type)) {
                            unchanged_type = self.tryLookupStructFieldTypeFromValue(source_value, field_index) catch blk: {
                                self.recordDegradation("failed to recover struct field type from source value");
                                break :blk mlir.MlirType{ .ptr = null };
                            };
                        }
                        if (mlir.oraTypeIsNull(unchanged_type)) {
                            self.recordDegradation("missing struct field type metadata for struct update");
                            struct_update_failed = true;
                            continue;
                        }
                        const unchanged_sort = self.encodeMLIRType(unchanged_type) catch {
                            self.recordDegradation("failed to encode unchanged struct field sort");
                            struct_update_failed = true;
                            continue;
                        };
                        const updated_accessor = self.applyFieldFunction(trimmed, result_sort, unchanged_sort, struct_val) catch {
                            self.recordDegradation("failed to encode updated struct field accessor");
                            struct_update_failed = true;
                            continue;
                        };
                        const original_accessor = self.applyFieldFunction(trimmed, source_sort, unchanged_sort, operands[0]) catch {
                            self.recordDegradation("failed to encode original struct field accessor");
                            struct_update_failed = true;
                            continue;
                        };
                        self.addConstraint(z3.Z3_mk_eq(self.context.ctx, updated_accessor, original_accessor));
                    }
                    if (struct_update_failed) {
                        self.recordDegradation("struct update frame constraints are incomplete");
                    }
                } else {
                    self.recordDegradation("missing struct declaration metadata for struct update");
                }
                return struct_val;
            }
        }

        // control flow
        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (operands.len >= 1) {
                const condition = operands[0];
                const then_expr = try self.extractRegionYield(mlir_op, 0, 0, mode);
                const else_expr = try self.extractRegionYield(mlir_op, 1, 0, mode);
                return try self.encodeControlFlow(op_name, condition, then_expr, else_expr);
            }
        }

        if (std.mem.eql(u8, op_name, "scf.while") or
            std.mem.eql(u8, op_name, "scf.for"))
        {
            if (std.mem.eql(u8, op_name, "scf.while")) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
                const op_id = @intFromPtr(mlir_op.ptr);
                return (try self.tryExtractZeroIterationScfWhileResult(mlir_op, 0, mode)) orelse
                    (try self.tryExtractCanonicalUnsignedScfWhileResult(mlir_op, 0, mode)) orelse
                    (try self.tryExtractCanonicalSignedScfWhileResult(mlir_op, 0, mode)) orelse
                    (try self.tryExtractCanonicalIncrementScfWhileResult(mlir_op, 0, mode)) orelse
                    (try self.tryExtractCanonicalDecrementScfWhileResult(mlir_op, 0, mode)) orelse
                    (try self.tryExtractFiniteScfWhileResult(mlir_op, 0, mode)) orelse
                    try self.degradeToUndef(result_sort, "scf_while_result", op_id, "scf.while result requires loop summary");
            }

            if (std.mem.eql(u8, op_name, "scf.for")) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
                const op_id = @intFromPtr(mlir_op.ptr);
                if (try self.tryExtractCanonicalScfForDerivedResult(mlir_op, 0, mode)) |derived_result| {
                    return derived_result;
                }
                if (try self.tryExtractCanonicalIncrementScfForResult(mlir_op, 0, mode)) |increment_result| {
                    return increment_result;
                }
                if (try self.tryExtractCanonicalDecrementScfForResult(mlir_op, 0, mode)) |decrement_result| {
                    return decrement_result;
                }
                if (try self.tryExtractIdentityScfForResult(mlir_op, 0, mode)) |identity_result| {
                    return identity_result;
                }
                return (try self.tryExtractFiniteScfForResult(mlir_op, 0, mode)) orelse
                    try self.degradeToUndef(result_sort, "scf_for_result", op_id, "scf.for result requires loop summary");
            }

            return try self.encodeStructuredControlResult(mlir_op, op_name, operands, 0);
        }

        if (std.mem.eql(u8, op_name, "scf.execute_region")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
            const op_id = @intFromPtr(mlir_op.ptr);
            return (try self.extractRegionYield(mlir_op, 0, 0, mode)) orelse
                try self.degradeToUndef(result_sort, "execute_region_result", op_id, "scf.execute_region result missing region yield");
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const result_value = mlir.oraOperationGetResult(mlir_op, 0);
            const result_sort = try self.encodeMLIRType(mlir.oraValueGetType(result_value));
            const op_id = @intFromPtr(mlir_op.ptr);
            if (try self.tryStmtAlwaysEntersCatch(mlir_op, mode)) {
                return (try self.extractRegionYield(mlir_op, 1, 0, mode)) orelse
                    try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result missing catch-region yield");
            }
            if (!self.tryStmtMayEnterCatch(mlir_op)) {
                return (try self.extractRegionYield(mlir_op, 0, 0, mode)) orelse
                    try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result missing try-region yield");
            }
            if (try self.tryExtractTryRegionCatchPredicate(mlir_op, mode)) |catch_pred| {
                if (try self.extractRegionYieldValue(mlir_op, 0, 0)) |yield_value| {
                    if (try self.trySummarizeTryValue(yield_value, mode)) |summary| {
                        if (try self.extractRegionYield(mlir_op, 1, 0, mode)) |catch_expr| {
                            return try self.encodeControlFlow("scf.if", catch_pred, catch_expr, summary.ok_expr);
                        }
                    }
                }
            }
            return (try self.tryExtractEquivalentTryStmtResult(mlir_op, 0, mode)) orelse
                try self.degradeToUndef(result_sort, "try_stmt_result", op_id, "ora.try_stmt result requires exact catch summary");
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
                const tracked = TrackedMemrefState{
                    .value = operands[0],
                    .initialized = self.boolTrue(),
                };
                if (mode == .Old) {
                    try self.memref_old_map.put(memref_id, tracked);
                } else {
                    try self.memref_map.put(memref_id, tracked);
                }
            } else if (num_operands > 2) {
                const memref_value = mlir.oraOperationGetOperand(mlir_op, 1);
                const memref_id = @intFromPtr(memref_value.ptr);
                const map = if (mode == .Old) &self.memref_old_map else &self.memref_map;
                const base = if (map.get(memref_id)) |tracked|
                    tracked.value
                else
                    operands[1];
                const updated = try self.encodeShapedWriteToBase(base, operands[0], operands[2..]);
                try map.put(memref_id, .{
                    .value = updated,
                    .initialized = self.boolTrue(),
                });
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
                        if (self.isBoolConst(stored.initialized, true) or self.activeReturnPathImplies(stored.initialized)) {
                            return stored.value;
                        }
                        if (self.isBoolConst(stored.initialized, false)) {
                            if (try self.tryRecoverDominatingScalarMemrefState(mlir_op, memref_value, mode)) |recovered| {
                                try map.put(memref_id, recovered);
                                return recovered.value;
                            }
                        }
                        const op_id_unknown_conditional = @intFromPtr(mlir_op.ptr);
                        const fresh_conditional = try self.degradeToUndef(
                            result_sort,
                            "memref_load",
                            op_id_unknown_conditional,
                            self.formatDegradationAtOp(mlir_op, "memref.load read from conditionally initialized tracked local state"),
                        );
                        try map.put(memref_id, .{
                            .value = fresh_conditional,
                            .initialized = stored.initialized,
                        });
                        return fresh_conditional;
                    }

                    if (try self.tryRecoverDominatingScalarMemrefState(mlir_op, memref_value, mode)) |recovered| {
                        try map.put(memref_id, recovered);
                        return recovered.value;
                    }

                    const op_id_unknown = @intFromPtr(mlir_op.ptr);
                    const fresh = try self.degradeToUndef(
                        result_sort,
                        "memref_load",
                        op_id_unknown,
                        self.formatDegradationAtOp(mlir_op, "memref.load read from uninitialized tracked local state"),
                    );
                    try map.put(memref_id, .{
                        .value = fresh,
                        .initialized = self.boolFalse(),
                    });
                    return fresh;
                }

                if (num_operands > 1) {
                    const memref_value = mlir.oraOperationGetOperand(mlir_op, 0);
                    const memref_id = @intFromPtr(memref_value.ptr);
                    const map = if (mode == .Old) &self.memref_old_map else &self.memref_map;
                    if (map.get(memref_id)) |stored| {
                        if (self.isBoolConst(stored.initialized, true) or self.activeReturnPathImplies(stored.initialized)) {
                            return try self.encodeShapedReadFromBase(stored.value, operands[1..], result_sort);
                        }
                    }
                    return try self.encodeTensorExtractOp(operands, mlir_op);
                }

                const op_id = @intFromPtr(mlir_op.ptr);
                const shape_reason = try std.fmt.allocPrint(
                    self.allocator,
                    "memref.load with unsupported operand shape (operands={d})",
                    .{num_operands},
                );
                defer self.allocator.free(shape_reason);
                return try self.degradeToUndef(
                    result_sort,
                    "memref_load",
                    op_id,
                    self.formatDegradationAtOp(mlir_op, shape_reason),
                );
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
                return try self.getOrCreateEnv("evm_origin", result_sort);
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
                return try self.getOrCreateEnv("evm_timestamp", result_sort);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.tload")) {
            const num_results = mlir.oraOperationGetNumResults(mlir_op);
            if (num_results >= 1) {
                const result_value = mlir.oraOperationGetResult(mlir_op, 0);
                const result_type = mlir.oraValueGetType(result_value);
                const result_sort = try self.encodeMLIRType(result_type);
                const key = self.getStringAttr(mlir_op, "key") orelse return error.UnsupportedOperation;
                const transient_name = try self.transientSlotName(key);
                defer self.allocator.free(transient_name);
                if (mode == .Old) {
                    return try self.getOrCreateOldGlobal(transient_name, result_sort);
                }
                return try self.getOrCreateGlobal(transient_name, result_sort);
            }
        }

        if (std.mem.eql(u8, op_name, "ora.tstore")) {
            if (operands.len < 1) return error.InvalidOperandCount;
            const key = self.getStringAttr(mlir_op, "key") orelse return error.UnsupportedOperation;
            const transient_name = try self.transientSlotName(key);
            defer self.allocator.free(transient_name);

            const operand_value = mlir.oraOperationGetOperand(mlir_op, 0);
            const operand_type = mlir.oraValueGetType(operand_value);
            const operand_sort = try self.encodeMLIRType(operand_type);
            if (mode == .Old) {
                _ = try self.getOrCreateOldGlobal(transient_name, operand_sort);
            } else if (self.global_map.getPtr(transient_name)) |existing| {
                existing.* = operands[0];
                try self.markGlobalSlotWritten(transient_name);
            } else {
                try self.global_map.put(try self.allocator.dupe(u8, transient_name), operands[0]);
                try self.markGlobalSlotWritten(transient_name);
            }
            return operands[0];
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

    fn encodeErrorReturnOp(self: *Encoder, operands: []const z3.Z3_ast, mlir_op: mlir.MlirOperation) EncodeError!z3.Z3_ast {
        const num_results = mlir.oraOperationGetNumResults(mlir_op);
        if (num_results < 1) return error.UnsupportedOperation;

        const result_value = mlir.oraOperationGetResult(mlir_op, 0);
        const result_type = mlir.oraValueGetType(result_value);
        const success_type = mlir.oraErrorUnionTypeGetSuccessType(result_type);
        const op_id = @intFromPtr(mlir_op.ptr);

        if (!mlir.oraTypeIsNull(success_type)) {
            const eu = try self.getErrorUnionSort(result_type, success_type);
            const is_err = self.encodeBoolConstant(true);
            const ok_val = try self.mkUndefValue(eu.ok_sort, "ok", op_id);
            const err_val = if (operands.len >= 1)
                try self.coerceAstToSortOrUndef(operands[0], eu.err_sort, "error_return", op_id)
            else
                try self.encodeErrorIdValue(mlir_op, eu.err_sort, op_id);
            return z3.Z3_mk_app(self.context.ctx, eu.ctor, 3, &[_]z3.Z3_ast{ is_err, ok_val, err_val });
        }

        const result_sort = try self.encodeMLIRType(result_type);
        if (operands.len >= 1) {
            return try self.coerceAstToSortOrUndef(operands[0], result_sort, "error_return", op_id);
        }
        return try self.encodeErrorIdValue(mlir_op, result_sort, op_id);
    }

    fn coerceAstToSortOrUndef(self: *Encoder, value: z3.Z3_ast, target_sort: z3.Z3_sort, prefix: []const u8, op_id: usize) EncodeError!z3.Z3_ast {
        const value_sort = z3.Z3_get_sort(self.context.ctx, value);
        if (value_sort == target_sort) return value;
        return try self.degradeToUndef(target_sort, prefix, op_id, "failed to coerce AST to target sort");
    }

    fn encodeErrorIdValue(self: *Encoder, mlir_op: mlir.MlirOperation, target_sort: z3.Z3_sort, op_id: usize) EncodeError!z3.Z3_ast {
        const sym_name = self.getStringAttr(mlir_op, "sym_name") orelse return try self.degradeToUndef(target_sort, "error_id", op_id, "missing error symbol name");
        const error_id = self.lookupErrorDeclId(mlir_op, sym_name) orelse return try self.degradeToUndef(target_sort, "error_id", op_id, "failed to resolve error declaration id");
        if (z3.Z3_get_sort_kind(self.context.ctx, target_sort) == z3.Z3_BV_SORT) {
            const width = z3.Z3_get_bv_sort_size(self.context.ctx, target_sort);
            return try self.encodeIntegerConstant(error_id, width);
        }
        return try self.degradeToUndef(target_sort, "error_id", op_id, "unsupported error id target sort");
    }

    fn lookupErrorDeclId(self: *Encoder, op: mlir.MlirOperation, sym_name: []const u8) ?u256 {
        var current_block = mlir.mlirOperationGetBlock(op);
        while (!mlir.oraBlockIsNull(current_block)) {
            const parent_op = mlir.mlirBlockGetParentOperation(current_block);
            if (mlir.oraOperationIsNull(parent_op)) break;
            const parent_block = mlir.mlirOperationGetBlock(parent_op);
            if (mlir.oraBlockIsNull(parent_block)) {
                current_block = current_block;
                break;
            }
            current_block = parent_block;
        }

        var current = mlir.oraBlockGetFirstOperation(current_block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const op_name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "ora.error.decl")) {
                const candidate_name = self.getStringAttr(current, "sym_name") orelse "";
                if (std.mem.eql(u8, candidate_name, sym_name)) {
                    const attr = mlir.oraOperationGetAttributeByName(current, mlir.oraStringRefCreate("ora.error_id".ptr, "ora.error_id".len));
                    if (!mlir.oraAttributeIsNull(attr)) {
                        return @intCast(mlir.oraIntegerAttrGetValueSInt(attr));
                    }
                }
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
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

        const callee = try self.getOpaqueCalleeKey(mlir_op);
        defer self.allocator.free(callee);
        if (mode == .Old) {
            if (self.function_ops.get(callee)) |func_op| {
                if (try self.tryInlinePureCallResult(mlir_op, callee, func_op, operands, result_index, mode)) |inlined| {
                    return inlined;
                }
                self.recordCalleeResultDegradation(mlir_op, callee, "failed to encode known pure callee result exactly");
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

    fn getOpaqueCalleeKey(self: *Encoder, mlir_op: mlir.MlirOperation) ![]u8 {
        if (try self.resolveCalleeName(mlir_op)) |callee| {
            return callee;
        }
        return try std.fmt.allocPrint(self.allocator, "call_site_{d}", .{@intFromPtr(mlir_op.ptr)});
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

    fn transientSlotName(self: *Encoder, key: []const u8) EncodeError![]u8 {
        return try std.fmt.allocPrint(self.allocator, "transient:{s}", .{key});
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
        } else if (std.mem.eql(u8, op_name, "ora.tstore")) {
            const key = self.getStringAttr(op, "key") orelse {
                writes_unknown.* = true;
                return;
            };
            const transient_name = try self.transientSlotName(key);
            defer self.allocator.free(transient_name);
            try self.appendWriteSlotUnique(write_slots, transient_name);
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

    fn operationMayWriteTrackedState(self: *Encoder, op: mlir.MlirOperation) EncodeError!bool {
        var write_slots = std.ArrayList([]u8){};
        defer {
            for (write_slots.items) |slot_name| self.allocator.free(slot_name);
            write_slots.deinit(self.allocator);
        }
        var writes_unknown = false;
        var visited_funcs = std.AutoHashMap(u64, void).init(self.allocator);
        defer visited_funcs.deinit();
        try self.collectWriteInfoFromOperation(op, &write_slots, &writes_unknown, &visited_funcs);
        return writes_unknown or write_slots.items.len > 0;
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
            if (std.mem.eql(u8, op_name, "ora.tload") or std.mem.eql(u8, op_name, "ora.tstore")) {
                const key_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("key", 3));
                if (!mlir.oraAttributeIsNull(key_attr)) {
                    const key_ref = mlir.oraStringAttrGetValue(key_attr);
                    if (key_ref.data != null and key_ref.length > 0 and std.mem.startsWith(u8, slot_name, "transient:")) {
                        const key_name = key_ref.data[0..key_ref.length];
                        const transient_name = slot_name["transient:".len..];
                        if (std.mem.eql(u8, key_name, transient_name)) {
                            if (std.mem.eql(u8, op_name, "ora.tload")) {
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
        var state = ReturnSearchState{};
        self.findUniqueReturnInOperation(func_op, &state);
        if (state.multiple or mlir.oraOperationIsNull(state.first)) return null;
        return state.first;
    }

    const ReturnSearchState = struct {
        first: mlir.MlirOperation = .{ .ptr = null },
        multiple: bool = false,
    };

    fn findUniqueReturnInOperation(self: *Encoder, op: mlir.MlirOperation, state: *ReturnSearchState) void {
        if (state.multiple) return;

        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "ora.return") or std.mem.eql(u8, op_name, "func.return")) {
                if (mlir.oraOperationIsNull(state.first)) {
                    state.first = op;
                } else {
                    state.multiple = true;
                }
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
                    self.findUniqueReturnInOperation(nested, state);
                    if (state.multiple) return;
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
            sort = inferred_sort orelse blk: {
                self.recordDegradation("failed to infer global sort for call summary slot");
                break :blk self.mkBitVectorSort(256);
            };
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
        call_op: mlir.MlirOperation,
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
        try summary_encoder.copyReturnPathAssumptionsFrom(self);
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

        var callee_requires = std.ArrayList(z3.Z3_ast){};
        defer callee_requires.deinit(self.allocator);
        try summary_encoder.collectRequiresForSummary(func_op, &callee_requires);

        const encoded = try summary_encoder.extractFunctionReturnExpr(func_op, result_index, mode) orelse return null;
        if (summary_encoder.isDegraded()) {
            self.recordCalleeResultDegradation(
                call_op,
                callee,
                summary_encoder.degradationReason() orelse "summary encoder degraded while inlining pure call result",
            );
        }

        const extra_constraints = try summary_encoder.takeConstraints(self.allocator);
        defer if (extra_constraints.len > 0) self.allocator.free(extra_constraints);
        for (extra_constraints) |cst| self.addConstraint(cst);

        const extra_obligations = try summary_encoder.takeObligations(self.allocator);
        defer if (extra_obligations.len > 0) self.allocator.free(extra_obligations);
        if (!self.functionIsExternallyVerified(func_op)) {
            self.addSummaryObligations(extra_obligations, callee_requires.items, summary_encoder.return_path_assumptions.items);
        }

        return encoded;
    }

    fn extractFunctionReturnExpr(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return null;
        const entry_block = mlir.oraRegionGetFirstBlock(body_region);
        if (mlir.oraBlockIsNull(entry_block)) return null;
        return try self.extractReturnedExprFromBlock(entry_block, result_index, mode);
    }

    fn extractFunctionReturnValue(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        result_index: u32,
    ) EncodeError!?mlir.MlirValue {
        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return null;
        const entry_block = mlir.oraRegionGetFirstBlock(body_region);
        if (mlir.oraBlockIsNull(entry_block)) return null;
        return try self.extractReturnedValueFromBlock(entry_block, result_index);
    }

    fn extractReturnedValueFromBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        result_index: u32,
    ) EncodeError!?mlir.MlirValue {
        if (mlir.oraBlockIsNull(block)) return null;
        return try self.extractReturnedValueFromSequence(
            mlir.oraBlockGetFirstOperation(block),
            result_index,
        );
    }

    fn extractReturnedValueFromSequence(
        self: *Encoder,
        start_op: mlir.MlirOperation,
        result_index: u32,
    ) EncodeError!?mlir.MlirValue {
        _ = self;
        var current = start_op;
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = mlir.oraOperationGetName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (result_index >= num_operands) return null;
                return mlir.oraOperationGetOperand(current, result_index);
            }

            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn tryLookupStructFieldNamesFromFunctionReturn(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        result_index: u32,
    ) EncodeError!?[]u8 {
        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return null;
        return try self.tryLookupStructFieldNamesFromReturnBlock(
            mlir.oraRegionGetFirstBlock(body_region),
            result_index,
        );
    }

    fn tryLookupStructFieldNamesFromReturnBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        result_index: u32,
    ) EncodeError!?[]u8 {
        if (mlir.oraBlockIsNull(block)) return null;
        return try self.tryLookupStructFieldNamesFromReturnSequence(
            mlir.oraBlockGetFirstOperation(block),
            result_index,
        );
    }

    fn tryLookupStructFieldNamesFromReturnSequence(
        self: *Encoder,
        start_op: mlir.MlirOperation,
        result_index: u32,
    ) EncodeError!?[]u8 {
        var current = start_op;
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            const next = mlir.oraOperationGetNextInBlock(current);

            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (result_index >= num_operands) return null;
                return try self.tryLookupStructFieldNamesFromValue(mlir.oraOperationGetOperand(current, result_index));
            }

            if (std.mem.eql(u8, name, "scf.if")) {
                const fallthrough = try self.tryLookupStructFieldNamesFromReturnSequence(next, result_index);
                var then_csv = try self.tryLookupStructFieldNamesFromReturnBlock(mlir.oraScfIfOpGetThenBlock(current), result_index);
                var else_csv = try self.tryLookupStructFieldNamesFromReturnBlock(mlir.oraScfIfOpGetElseBlock(current), result_index);
                if (then_csv == null) then_csv = fallthrough;
                if (else_csv == null) else_csv = fallthrough;
                return try self.mergeRecoveredFieldNames(then_csv, else_csv);
            }

            if (std.mem.eql(u8, name, "ora.conditional_return")) {
                const fallthrough = try self.tryLookupStructFieldNamesFromReturnSequence(next, result_index);
                var then_csv = try self.tryLookupStructFieldNamesFromReturnBlock(mlir.oraConditionalReturnOpGetThenBlock(current), result_index);
                var else_csv = try self.tryLookupStructFieldNamesFromReturnBlock(mlir.oraConditionalReturnOpGetElseBlock(current), result_index);
                if (then_csv == null) then_csv = fallthrough;
                if (else_csv == null) else_csv = fallthrough;
                return try self.mergeRecoveredFieldNames(then_csv, else_csv);
            }

            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                const fallthrough = try self.tryLookupStructFieldNamesFromReturnSequence(next, result_index);
                const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(current));
                var recovered = fallthrough;
                for (0..num_regions) |region_index| {
                    const region = mlir.oraOperationGetRegion(current, @intCast(region_index));
                    const branch_block = mlir.oraRegionGetFirstBlock(region);
                    const branch_csv = try self.tryLookupStructFieldNamesFromReturnBlock(branch_block, result_index);
                    recovered = try self.mergeRecoveredFieldNames(recovered, if (branch_csv != null) branch_csv else fallthrough);
                }
                return recovered;
            }

            if (std.mem.eql(u8, name, "scf.execute_region")) {
                const region = mlir.oraOperationGetRegion(current, 0);
                if (!mlir.oraRegionIsNull(region)) {
                    const nested = try self.tryLookupStructFieldNamesFromReturnBlock(mlir.oraRegionGetFirstBlock(region), result_index);
                    if (nested != null) return nested;
                }
            }

            current = next;
        }
        return null;
    }

    fn tryLookupStructFieldTypeFromFunctionReturn(
        self: *Encoder,
        func_op: mlir.MlirOperation,
        result_index: u32,
        index: usize,
    ) EncodeError!mlir.MlirType {
        const body_region = mlir.oraOperationGetRegion(func_op, 0);
        if (mlir.oraRegionIsNull(body_region)) return .{ .ptr = null };
        return try self.tryLookupStructFieldTypeFromReturnBlock(
            mlir.oraRegionGetFirstBlock(body_region),
            result_index,
            index,
        );
    }

    fn tryLookupStructFieldTypeFromReturnBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        result_index: u32,
        index: usize,
    ) EncodeError!mlir.MlirType {
        if (mlir.oraBlockIsNull(block)) return .{ .ptr = null };
        return try self.tryLookupStructFieldTypeFromReturnSequence(
            mlir.oraBlockGetFirstOperation(block),
            result_index,
            index,
        );
    }

    fn tryLookupStructFieldTypeFromReturnSequence(
        self: *Encoder,
        start_op: mlir.MlirOperation,
        result_index: u32,
        index: usize,
    ) EncodeError!mlir.MlirType {
        var current = start_op;
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            const next = mlir.oraOperationGetNextInBlock(current);

            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (result_index >= num_operands) return .{ .ptr = null };
                return try self.tryLookupStructFieldTypeFromValue(mlir.oraOperationGetOperand(current, result_index), index);
            }

            if (std.mem.eql(u8, name, "scf.if")) {
                const fallthrough = try self.tryLookupStructFieldTypeFromReturnSequence(next, result_index, index);
                var then_ty = try self.tryLookupStructFieldTypeFromReturnBlock(mlir.oraScfIfOpGetThenBlock(current), result_index, index);
                var else_ty = try self.tryLookupStructFieldTypeFromReturnBlock(mlir.oraScfIfOpGetElseBlock(current), result_index, index);
                if (mlir.oraTypeIsNull(then_ty)) then_ty = fallthrough;
                if (mlir.oraTypeIsNull(else_ty)) else_ty = fallthrough;
                return self.mergeRecoveredFieldTypes(then_ty, else_ty);
            }

            if (std.mem.eql(u8, name, "ora.conditional_return")) {
                const fallthrough = try self.tryLookupStructFieldTypeFromReturnSequence(next, result_index, index);
                var then_ty = try self.tryLookupStructFieldTypeFromReturnBlock(mlir.oraConditionalReturnOpGetThenBlock(current), result_index, index);
                var else_ty = try self.tryLookupStructFieldTypeFromReturnBlock(mlir.oraConditionalReturnOpGetElseBlock(current), result_index, index);
                if (mlir.oraTypeIsNull(then_ty)) then_ty = fallthrough;
                if (mlir.oraTypeIsNull(else_ty)) else_ty = fallthrough;
                return self.mergeRecoveredFieldTypes(then_ty, else_ty);
            }

            if (std.mem.eql(u8, name, "ora.switch") or std.mem.eql(u8, name, "ora.switch_expr")) {
                const fallthrough = try self.tryLookupStructFieldTypeFromReturnSequence(next, result_index, index);
                const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(current));
                var recovered = fallthrough;
                for (0..num_regions) |region_index| {
                    const region = mlir.oraOperationGetRegion(current, @intCast(region_index));
                    const branch_block = mlir.oraRegionGetFirstBlock(region);
                    const branch_ty = try self.tryLookupStructFieldTypeFromReturnBlock(branch_block, result_index, index);
                    recovered = self.mergeRecoveredFieldTypes(recovered, if (!mlir.oraTypeIsNull(branch_ty)) branch_ty else fallthrough);
                }
                return recovered;
            }

            if (std.mem.eql(u8, name, "scf.execute_region")) {
                const region = mlir.oraOperationGetRegion(current, 0);
                if (!mlir.oraRegionIsNull(region)) {
                    const nested = try self.tryLookupStructFieldTypeFromReturnBlock(mlir.oraRegionGetFirstBlock(region), result_index, index);
                    if (!mlir.oraTypeIsNull(nested)) return nested;
                }
            }

            current = next;
        }
        return .{ .ptr = null };
    }

    fn extractReturnedExprFromBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        if (mlir.oraBlockIsNull(block)) return null;
        return try self.extractReturnedExprFromSequence(
            mlir.oraBlockGetFirstOperation(block),
            result_index,
            mode,
        );
    }

    fn extractReturnedExprFromSequence(
        self: *Encoder,
        start_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        var current = start_op;
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "scf.for")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                if (mlir.oraOperationIsNull(next_op)) {
                    if (try self.tryExtractGuaranteedFirstIterationScfForReturnedExpr(current, result_index, mode)) |loop_returned_expr| {
                        return loop_returned_expr;
                    }
                }
            }

            if (std.mem.eql(u8, name, "scf.while")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                if (mlir.oraOperationIsNull(next_op)) {
                    if (try self.tryExtractGuaranteedFirstIterationScfWhileReturnedExpr(current, result_index, mode)) |loop_returned_expr| {
                        return loop_returned_expr;
                    }
                    const nested_expr = (try self.tryExtractZeroIterationScfWhileResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalUnsignedScfWhileResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalSignedScfWhileResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalIncrementScfWhileResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalDecrementScfWhileResult(current, result_index, mode)) orelse
                        (try self.tryExtractFiniteScfWhileResult(current, result_index, mode));
                    if (nested_expr != null) return nested_expr;
                }
            }

            self.applyLocalReturnExtractionStateEffect(current, name, mode) catch |err| switch (err) {
                error.UnsupportedOperation,
                error.InvalidOperandCount,
                error.InvalidControlFlow,
                error.UnsupportedPredicate,
                => self.recordDegradation("failed to replay local state effect during pure return extraction"),
                error.OutOfMemory => return error.OutOfMemory,
            };

            if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (result_index >= num_operands) return null;
                const ret_value = mlir.oraOperationGetOperand(current, result_index);
                return try self.encodeValueWithMode(ret_value, mode);
            }

            if (std.mem.eql(u8, name, "scf.if")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                const fallthrough_expr = try self.extractReturnedExprFromSequence(next_op, result_index, mode);
                const branch_expr = try self.extractScfIfReturnedExpr(current, result_index, mode, fallthrough_expr);
                if (branch_expr != null) return branch_expr;
            }

            if (std.mem.eql(u8, name, "ora.conditional_return")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                var fallthrough_expr: ?z3.Z3_ast = null;
                if (!mlir.oraOperationIsNull(next_op) and mlir.oraOperationGetNumOperands(current) >= 1) {
                    const condition_value = mlir.oraOperationGetOperand(current, 0);
                    const condition = try self.encodeValueWithMode(condition_value, mode);
                    const saved_len = self.return_path_assumptions.items.len;
                    defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
                    try self.return_path_assumptions.append(
                        self.allocator,
                        z3.Z3_mk_not(self.context.ctx, self.coerceToBool(condition)),
                    );
                    fallthrough_expr = try self.extractReturnedExprFromSequence(next_op, result_index, mode);
                } else {
                    fallthrough_expr = try self.extractReturnedExprFromSequence(next_op, result_index, mode);
                }
                const branch_expr = try self.extractConditionalReturnExpr(current, result_index, mode, fallthrough_expr);
                if (branch_expr != null) return branch_expr;
            }

            if (std.mem.eql(u8, name, "ora.switch")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                const fallthrough_expr = try self.extractReturnedExprFromSequence(next_op, result_index, mode);
                const branch_expr = try self.extractSwitchReturnedExpr(current, result_index, mode, fallthrough_expr);
                if (branch_expr != null) return branch_expr;
            }

            if (std.mem.eql(u8, name, "scf.execute_region")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                if (mlir.oraOperationIsNull(next_op)) {
                    const region = mlir.oraOperationGetRegion(current, 0);
                    if (!mlir.oraRegionIsNull(region)) {
                        const block = mlir.oraRegionGetFirstBlock(region);
                        const nested_expr = try self.extractReturnedExprFromBlock(block, result_index, mode);
                        if (nested_expr != null) return nested_expr;
                    }
                }
            }

            if (std.mem.eql(u8, name, "scf.for")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                if (mlir.oraOperationIsNull(next_op)) {
                    const nested_expr = (try self.tryExtractCanonicalScfForDerivedResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalIncrementScfForResult(current, result_index, mode)) orelse
                        (try self.tryExtractCanonicalDecrementScfForResult(current, result_index, mode)) orelse
                        (try self.tryExtractIdentityScfForResult(current, result_index, mode)) orelse
                        (try self.tryExtractFiniteScfForResult(current, result_index, mode));
                    if (nested_expr != null) return nested_expr;
                }
            }

            if (std.mem.eql(u8, name, "ora.try_stmt")) {
                const next_op = mlir.oraOperationGetNextInBlock(current);
                if (mlir.oraOperationIsNull(next_op)) {
                    if ((self.tryStmtAlwaysEntersCatch(current, mode) catch false)) {
                        const catch_region = mlir.oraOperationGetRegion(current, 1);
                        if (!mlir.oraRegionIsNull(catch_region)) {
                            const catch_block = mlir.oraRegionGetFirstBlock(catch_region);
                            const nested_expr = try self.extractReturnedExprFromBlock(catch_block, result_index, mode);
                            if (nested_expr != null) return nested_expr;
                        }
                    } else if (!self.tryStmtMayEnterCatch(current)) {
                        const try_block = mlir.oraTryStmtOpGetTryBlock(current);
                        const nested_expr = try self.extractReturnedExprFromBlock(try_block, result_index, mode);
                        if (nested_expr != null) return nested_expr;
                    }
                }
            }

            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn applyLocalReturnExtractionStateEffect(
        self: *Encoder,
        op: mlir.MlirOperation,
        op_name: []const u8,
        mode: EncodeMode,
    ) EncodeError!void {
        _ = mode;

        if (std.mem.eql(u8, op_name, "memref.store") or
            std.mem.eql(u8, op_name, "ora.sstore") or
            std.mem.eql(u8, op_name, "ora.map_store") or
            std.mem.eql(u8, op_name, "ora.tstore"))
        {
            _ = try self.encodeOperation(op);
            return;
        }

        if (std.mem.eql(u8, op_name, "ora.assert") or
            std.mem.eql(u8, op_name, "func.call") or
            std.mem.eql(u8, op_name, "call"))
        {
            _ = try self.encodeOperation(op);
            return;
        }

        if (std.mem.eql(u8, op_name, "scf.if") or
            std.mem.eql(u8, op_name, "ora.conditional_return") or
            std.mem.eql(u8, op_name, "ora.switch") or
            std.mem.eql(u8, op_name, "ora.switch_expr") or
            std.mem.eql(u8, op_name, "scf.execute_region") or
            std.mem.eql(u8, op_name, "scf.for") or
            std.mem.eql(u8, op_name, "scf.while") or
            std.mem.eql(u8, op_name, "ora.try_stmt"))
        {
            self.encodeStateEffectsInOperation(op);
            return;
        }
    }

    fn tryExtractGuaranteedFirstIterationScfForReturnedExpr(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 3) return null;

        const lb_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 0)) orelse return null;
        const ub_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 1)) orelse return null;
        const step_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 2)) orelse return null;
        if (step_const <= 0 or lb_const >= ub_const) return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;
        return try self.extractReturnedExprFromBlock(body, result_index, mode);
    }

    fn tryExtractGuaranteedFirstIterationScfWhileReturnedExpr(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const before_bind_count = try self.bindScfWhileBeforeArgs(while_op, mode);
        defer self.unbindScfWhileBeforeArgs(while_op, before_bind_count);

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return null;

        self.invalidateBlockValueCaches(before_block);
        self.invalidateBlockValueCaches(after_block);

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) < 1) return null;
        const condition_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), mode);
        if (!self.isBoolConst(condition_ast, true)) return null;

        const after_bind_count = try self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, mode);
        defer self.unbindScfWhileAfterArgs(while_op, after_bind_count);
        return try self.extractReturnedExprFromBlock(after_block, result_index, mode);
    }

    fn regionMayEnterCatch(self: *Encoder, region: mlir.MlirRegion) bool {
        if (mlir.oraRegionIsNull(region)) return false;
        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) {
            var nested = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(nested)) {
                if (self.operationMayEnterCatch(nested)) return true;
                nested = mlir.oraOperationGetNextInBlock(nested);
            }
            block = mlir.oraBlockGetNextInRegion(block);
        }
        return false;
    }

    fn valueIsStaticallyErrorUnionError(
        self: *Encoder,
        value: mlir.MlirValue,
        mode: EncodeMode,
    ) EncodeError!?bool {
        const value_type = mlir.oraValueGetType(value);
        const success_type = mlir.oraErrorUnionTypeGetSuccessType(value_type);
        if (mlir.oraTypeIsNull(success_type)) return null;

        const encoded = try self.encodeValueWithMode(value, mode);
        const eu = try self.getErrorUnionSort(value_type, success_type);
        const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{encoded});
        return self.astSimplifiesToBool(is_err);
    }

    fn regionAlwaysEntersCatch(
        self: *Encoder,
        region: mlir.MlirRegion,
        mode: EncodeMode,
    ) EncodeError!bool {
        if (mlir.oraRegionIsNull(region)) return false;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) {
            var op = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(op)) {
                if (try self.operationAlwaysEntersCatch(op, mode)) return true;

                const name_ref = mlir.oraOperationGetName(op);
                defer @import("mlir_c_api").freeStringRef(name_ref);
                const op_name = if (name_ref.data == null or name_ref.length == 0)
                    ""
                else
                    name_ref.data[0..name_ref.length];

                if (std.mem.eql(u8, op_name, "ora.return") or
                    std.mem.eql(u8, op_name, "func.return") or
                    std.mem.eql(u8, op_name, "ora.yield") or
                    std.mem.eql(u8, op_name, "scf.yield"))
                {
                    return false;
                }

                op = mlir.oraOperationGetNextInBlock(op);
            }
            block = mlir.oraBlockGetNextInRegion(block);
        }

        return false;
    }

    fn operationAlwaysEntersCatch(
        self: *Encoder,
        op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!bool {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.error.unwrap")) {
            if (mlir.oraOperationGetNumOperands(op) < 1) return false;
            const operand = mlir.oraOperationGetOperand(op, 0);
            return (try self.valueIsStaticallyErrorUnionError(operand, mode)) == true;
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const catch_region = mlir.oraOperationGetRegion(op, 1);
            if (!self.regionMayEnterCatch(catch_region)) return false;
            return try self.tryStmtAlwaysEntersCatch(op, mode);
        }

        if (std.mem.eql(u8, op_name, "scf.execute_region")) {
            return try self.regionAlwaysEntersCatch(mlir.oraOperationGetRegion(op, 0), mode);
        }

        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (mlir.oraOperationGetNumOperands(op) < 1) return false;
            const condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(op, 0), mode);
            const selected_region: u32 = if ((self.astSimplifiesToBool(condition) orelse return false)) 0 else 1;
            return try self.regionAlwaysEntersCatch(mlir.oraOperationGetRegion(op, selected_region), mode);
        }

        if (std.mem.eql(u8, op_name, "ora.conditional_return")) {
            if (mlir.oraOperationGetNumOperands(op) < 1) return false;
            const condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(op, 0), mode);
            const selected_region: u32 = if ((self.astSimplifiesToBool(condition) orelse return false)) 0 else 1;
            return try self.regionAlwaysEntersCatch(mlir.oraOperationGetRegion(op, selected_region), mode);
        }

        if (std.mem.eql(u8, op_name, "ora.switch")) {
            const case_index = self.getSelectedOraSwitchCaseIndex(op) orelse return false;
            return try self.regionAlwaysEntersCatch(mlir.oraOperationGetRegion(op, @intCast(case_index)), mode);
        }

        return false;
    }

    fn operationMayEnterCatch(self: *Encoder, op: mlir.MlirOperation) bool {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.error.unwrap") or
            std.mem.eql(u8, op_name, "ora.try_catch"))
        {
            return true;
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const catch_region = mlir.oraOperationGetRegion(op, 1);
            if (!self.regionMayEnterCatch(catch_region)) return false;
            return self.tryStmtMayEnterCatch(op);
        }

        if (std.mem.eql(u8, op_name, "scf.if")) {
            if (mlir.oraOperationGetNumOperands(op) >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                if (self.getValueConstUnsigned(condition_value, 1)) |cond| {
                    return self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, if (cond != 0) 0 else 1));
                }
            }
            return self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, 0)) or
                self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, 1));
        }

        if (std.mem.eql(u8, op_name, "ora.conditional_return")) {
            if (mlir.oraOperationGetNumOperands(op) >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                if (self.getValueConstUnsigned(condition_value, 1)) |cond| {
                    return self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, if (cond != 0) 0 else 1));
                }
            }
            return self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, 0)) or
                self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, 1));
        }

        if (std.mem.eql(u8, op_name, "ora.switch")) {
            if (self.getSelectedOraSwitchCaseIndex(op)) |case_index| {
                return self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, @intCast(case_index)));
            }

            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
            for (0..num_regions) |region_idx| {
                if (self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, @intCast(region_idx)))) return true;
            }
            return false;
        }

        if (std.mem.eql(u8, op_name, "scf.for")) {
            if (self.isZeroIterationScfFor(op)) return false;
        }

        if (std.mem.eql(u8, op_name, "scf.while")) {
            if ((self.isStaticallyFalseScfWhile(op, .Current) catch false)) return false;
        }

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            const num_results = mlir.oraOperationGetNumResults(op);
            var idx: usize = 0;
            while (idx < num_results) : (idx += 1) {
                const result_ty = mlir.oraValueGetType(mlir.oraOperationGetResult(op, @intCast(idx)));
                if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(result_ty))) return true;
            }
        }

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            if (self.regionMayEnterCatch(mlir.oraOperationGetRegion(op, region_idx))) return true;
        }

        return false;
    }

    fn getSelectedOraSwitchCaseIndex(self: *Encoder, op: mlir.MlirOperation) ?usize {
        if (mlir.oraOperationGetNumOperands(op) < 1) return null;

        const scrutinee_value = mlir.oraOperationGetOperand(op, 0);
        const scrutinee_ty = mlir.oraValueGetType(scrutinee_value);
        const bit_width = self.getTypeBitWidth(scrutinee_ty) orelse return null;
        const scrutinee_const = self.getValueConstUnsigned(scrutinee_value, bit_width) orelse return null;

        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        if (num_regions == 0) return null;

        var metadata = self.getSwitchCaseMetadata(op, num_regions) catch return null;
        defer metadata.deinit(self.allocator);

        for (0..num_regions) |case_index| {
            switch (metadata.case_kinds[case_index]) {
                0 => {
                    const case_value = std.math.cast(u64, metadata.case_values[case_index]) orelse continue;
                    if (scrutinee_const == case_value) return case_index;
                },
                1 => {
                    const start_value = std.math.cast(u64, metadata.range_starts[case_index]) orelse continue;
                    const end_value = std.math.cast(u64, metadata.range_ends[case_index]) orelse continue;
                    if (scrutinee_const >= start_value and scrutinee_const <= end_value) return case_index;
                },
                2 => {},
                else => {},
            }
        }

        if (metadata.default_case_index) |default_case_index| {
            return std.math.cast(usize, default_case_index);
        }

        return null;
    }

    fn tryStmtMayEnterCatch(self: *Encoder, try_stmt: mlir.MlirOperation) bool {
        const try_region = mlir.oraOperationGetRegion(try_stmt, 0);
        if (mlir.oraRegionIsNull(try_region)) return false;
        var block = mlir.oraRegionGetFirstBlock(try_region);
        while (!mlir.oraBlockIsNull(block)) {
            var op = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(op)) {
                if (self.operationMayEnterCatch(op)) return true;
                op = mlir.oraOperationGetNextInBlock(op);
            }
            block = mlir.oraBlockGetNextInRegion(block);
        }
        return false;
    }

    fn tryStmtAlwaysEntersCatch(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!bool {
        const try_region = mlir.oraOperationGetRegion(try_stmt, 0);
        return try self.regionAlwaysEntersCatch(try_region, mode);
    }

    fn isZeroIterationScfFor(self: *Encoder, op: mlir.MlirOperation) bool {
        if (mlir.oraOperationGetNumOperands(op) < 3) return false;

        const lb_value = mlir.oraOperationGetOperand(op, 0);
        const ub_value = mlir.oraOperationGetOperand(op, 1);
        const step_value = mlir.oraOperationGetOperand(op, 2);
        const lb = self.getValueConstUnsigned(lb_value, 64) orelse return false;
        const ub = self.getValueConstUnsigned(ub_value, 64) orelse return false;
        const step = self.getValueConstUnsigned(step_value, 64) orelse return false;
        if (step == 0) return false;
        return lb >= ub;
    }

    const FiniteScfForContext = struct {
        body: mlir.MlirBlock,
        num_body_args: usize,
        lower_bound: u64,
        step: u64,
        trip_count: usize,
    };

    fn getFiniteScfForContext(self: *Encoder, op: mlir.MlirOperation) ?FiniteScfForContext {
        if (mlir.oraOperationGetNumOperands(op) < 3) return null;

        const lb_value = mlir.oraOperationGetOperand(op, 0);
        const ub_value = mlir.oraOperationGetOperand(op, 1);
        const step_value = mlir.oraOperationGetOperand(op, 2);
        const lb_u256 = self.getValueConstUnsigned(lb_value, 64) orelse return null;
        const ub_u256 = self.getValueConstUnsigned(ub_value, 64) orelse return null;
        const step_u256 = self.getValueConstUnsigned(step_value, 64) orelse return null;
        const lb = std.math.cast(u64, lb_u256) orelse return null;
        const ub = std.math.cast(u64, ub_u256) orelse return null;
        const step = std.math.cast(u64, step_u256) orelse return null;
        if (step == 0) return null;

        var trip_count: usize = 0;
        var iter = lb;
        while (iter < ub) : (iter += step) {
            trip_count += 1;
            if (trip_count > finite_scf_for_unroll_limit) return null;
            if (iter > std.math.maxInt(u64) - step) break;
        }

        const body = mlir.oraScfForOpGetBodyBlock(op);
        if (mlir.oraBlockIsNull(body)) return null;
        const num_body_args: usize = @intCast(mlir.oraBlockGetNumArguments(body));
        if (num_body_args < 1) return null;

        return .{
            .body = body,
            .num_body_args = num_body_args,
            .lower_bound = lb,
            .step = step,
            .trip_count = trip_count,
        };
    }

    fn bindFiniteScfForLoopArgs(
        self: *Encoder,
        loop_ctx: FiniteScfForContext,
        iv_value: u64,
        carried_values: []const z3.Z3_ast,
    ) EncodeError!void {
        const iv = mlir.oraBlockGetArgument(loop_ctx.body, 0);
        try self.bindValue(iv, try self.encodeIntegerConstant(@intCast(iv_value), 256));

        var arg_index: usize = 1;
        while (arg_index < loop_ctx.num_body_args) : (arg_index += 1) {
            const body_arg = mlir.oraBlockGetArgument(loop_ctx.body, @intCast(arg_index));
            if (arg_index - 1 >= carried_values.len) return error.InvalidOperandCount;
            try self.bindValue(body_arg, carried_values[arg_index - 1]);
        }
    }

    fn unbindFiniteScfForLoopArgs(self: *Encoder, loop_ctx: FiniteScfForContext) void {
        var arg_index: usize = 0;
        while (arg_index < loop_ctx.num_body_args) : (arg_index += 1) {
            const body_arg = mlir.oraBlockGetArgument(loop_ctx.body, @intCast(arg_index));
            _ = self.value_bindings.remove(@intFromPtr(body_arg.ptr));
        }
    }

    fn getScfForInitCarriedValues(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError![]z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 3) return self.allocator.alloc(z3.Z3_ast, 0);
        const num_iter_args = num_operands - 3;
        var values = try self.allocator.alloc(z3.Z3_ast, num_iter_args);
        for (0..num_iter_args) |index| {
            values[index] = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, @intCast(index + 3)), mode);
        }
        return values;
    }

    fn extractScfForYieldValues(
        self: *Encoder,
        loop_ctx: FiniteScfForContext,
        mode: EncodeMode,
    ) EncodeError!?[]z3.Z3_ast {
        var current = mlir.oraBlockGetFirstOperation(loop_ctx.body);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.yield")) {
                const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(current));
                var values = try self.allocator.alloc(z3.Z3_ast, num_operands);
                errdefer self.allocator.free(values);
                for (0..num_operands) |index| {
                    values[index] = try self.encodeValueWithMode(mlir.oraOperationGetOperand(current, @intCast(index)), mode);
                }
                return values;
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn invalidateOperationResultCaches(self: *Encoder, op: mlir.MlirOperation) void {
        var result_index: u32 = 0;
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(op));
        while (result_index < num_results) : (result_index += 1) {
            const result = mlir.oraOperationGetResult(op, @intCast(result_index));
            const result_id = @intFromPtr(result.ptr);
            _ = self.value_map.remove(result_id);
            _ = self.value_map_old.remove(result_id);
        }

        var region_index: u32 = 0;
        const num_regions: u32 = @intCast(mlir.oraOperationGetNumRegions(op));
        while (region_index < num_regions) : (region_index += 1) {
            self.invalidateRegionValueCaches(mlir.oraOperationGetRegion(op, @intCast(region_index)));
        }
    }

    fn invalidateRegionValueCaches(self: *Encoder, region: mlir.MlirRegion) void {
        if (mlir.oraRegionIsNull(region)) return;
        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) {
            self.invalidateBlockValueCaches(block);
            block = mlir.oraBlockGetNextInRegion(block);
        }
    }

    fn invalidateBlockValueCaches(self: *Encoder, block: mlir.MlirBlock) void {
        if (mlir.oraBlockIsNull(block)) return;
        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            self.invalidateOperationResultCaches(current);
            current = mlir.oraOperationGetNextInBlock(current);
        }
    }

    fn tryExtractFiniteScfForResult(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const loop_ctx = self.getFiniteScfForContext(for_op) orelse return null;
        var carried = try self.getScfForInitCarriedValues(for_op, mode);
        defer self.allocator.free(carried);

        if (loop_ctx.trip_count == 0) {
            if (result_index >= carried.len) return null;
            return carried[result_index];
        }

        for (0..loop_ctx.trip_count) |iter_index| {
            try self.bindFiniteScfForLoopArgs(loop_ctx, loop_ctx.lower_bound + iter_index * loop_ctx.step, carried);
            self.invalidateBlockValueCaches(loop_ctx.body);
            if (try self.extractReturnedExprFromBlock(loop_ctx.body, result_index, mode)) |returned_expr| {
                self.unbindFiniteScfForLoopArgs(loop_ctx);
                return returned_expr;
            }
            const yielded = (try self.extractScfForYieldValues(loop_ctx, mode)) orelse {
                self.unbindFiniteScfForLoopArgs(loop_ctx);
                return null;
            };
            self.unbindFiniteScfForLoopArgs(loop_ctx);
            self.allocator.free(carried);
            carried = yielded;
        }

        if (result_index >= carried.len) return null;
        return carried[result_index];
    }

    fn tryExtractCanonicalIncrementScfForResult(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 4) return null;
        const num_iter_args = num_operands - 3;
        if (num_iter_args != 1 or result_index != 0) return null;

        const lb_value = mlir.oraOperationGetOperand(for_op, 0);
        const step_value = mlir.oraOperationGetOperand(for_op, 2);
        const step_const = self.tryGetConstIntValue(step_value);
        if (step_const == null or step_const.? == 0) return null;
        const step_u64 = std.math.cast(u64, step_const.?) orelse return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;
        if (mlir.oraBlockGetNumArguments(body) != 2) return null;
        const carried_arg = mlir.oraBlockGetArgument(body, 1);

        var current = mlir.oraBlockGetFirstOperation(body);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.return")) return null;
            if (std.mem.eql(u8, name, "scf.yield")) {
                if (mlir.oraOperationGetNumOperands(current) != 1) return null;
                const yielded = mlir.oraOperationGetOperand(current, 0);
                if (!mlir.oraValueIsAOpResult(yielded)) return null;
                const yield_owner = mlir.oraOpResultGetOwner(yielded);
                if (!self.operationNameEq(yield_owner, "arith.addi")) return null;
                if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return null;

                const add_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
                const add_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
                const lhs_const = self.tryGetConstIntValue(add_lhs);
                const rhs_const = self.tryGetConstIntValue(add_rhs);
                const delta_value =
                    if (lhs_const != null and lhs_const.? != 0 and mlir.mlirValueEqual(add_rhs, carried_arg))
                        add_lhs
                    else if (rhs_const != null and rhs_const.? != 0 and mlir.mlirValueEqual(add_lhs, carried_arg))
                        add_rhs
                    else
                        return null;

                const init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, 3), mode);
                const lb_ast = try self.encodeValueWithMode(lb_value, mode);
                const ub_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, 1), mode);
                const trip_count_ast = try self.encodeCanonicalPositiveStepScfForTripCount(lb_ast, ub_ast, step_u64);
                const delta_ast = try self.encodeValueWithMode(delta_value, mode);
                const delta_total_ast = try self.encodeArithmeticOp(.Mul, trip_count_ast, delta_ast);
                return try self.encodeArithmeticOp(.Add, init_ast, delta_total_ast);
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn tryExtractCanonicalScfForDerivedResult(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 4) return null;
        const num_iter_args = num_operands - 3;
        if (num_iter_args == 0 or result_index >= num_iter_args) return null;

        const lb_value = mlir.oraOperationGetOperand(for_op, 0);
        const ub_value = mlir.oraOperationGetOperand(for_op, 1);
        const step_value = mlir.oraOperationGetOperand(for_op, 2);
        const step_const = self.tryGetConstIntValue(step_value);
        if (step_const == null or step_const.? == 0) return null;
        const step_u64 = std.math.cast(u64, step_const.?) orelse return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;
        if (mlir.oraBlockGetNumArguments(body) != num_iter_args + 1) return null;

        var current = mlir.oraBlockGetFirstOperation(body);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.return")) return null;
            if (std.mem.eql(u8, name, "scf.yield")) {
                if (mlir.oraOperationGetNumOperands(current) != num_iter_args) return null;

                const target_after_arg = mlir.oraBlockGetArgument(body, result_index + 1);
                const target_yield = mlir.oraOperationGetOperand(current, result_index);
                const target_update = self.classifyCanonicalWhileCarriedUpdate(target_yield, target_after_arg) orelse return null;

                const init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, result_index + 3), mode);
                const lb_ast = try self.encodeValueWithMode(lb_value, mode);
                const ub_ast = try self.encodeValueWithMode(ub_value, mode);
                const trip_count_ast = try self.encodeCanonicalPositiveStepScfForTripCount(lb_ast, ub_ast, step_u64);

                return switch (target_update) {
                    .identity => init_ast,
                    .add_const => |delta| blk: {
                        const sort = z3.Z3_get_sort(self.context.ctx, init_ast);
                        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                        const total_delta = try self.encodeArithmeticOp(.Mul, trip_count_ast, delta_ast);
                        break :blk try self.encodeArithmeticOp(.Add, init_ast, total_delta);
                    },
                    .sub_const => |delta| blk: {
                        const sort = z3.Z3_get_sort(self.context.ctx, init_ast);
                        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                        const total_delta = try self.encodeArithmeticOp(.Mul, trip_count_ast, delta_ast);
                        break :blk try self.encodeArithmeticOp(.Sub, init_ast, total_delta);
                    },
                };
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn tryExtractCanonicalDecrementScfForResult(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 4) return null;
        const num_iter_args = num_operands - 3;
        if (num_iter_args != 1 or result_index != 0) return null;

        const lb_value = mlir.oraOperationGetOperand(for_op, 0);
        const step_value = mlir.oraOperationGetOperand(for_op, 2);
        const step_const = self.tryGetConstIntValue(step_value);
        if (step_const == null or step_const.? == 0) return null;
        const step_u64 = std.math.cast(u64, step_const.?) orelse return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;
        if (mlir.oraBlockGetNumArguments(body) != 2) return null;
        const carried_arg = mlir.oraBlockGetArgument(body, 1);

        var current = mlir.oraBlockGetFirstOperation(body);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.return")) return null;
            if (std.mem.eql(u8, name, "scf.yield")) {
                if (mlir.oraOperationGetNumOperands(current) != 1) return null;
                const yielded = mlir.oraOperationGetOperand(current, 0);
                if (!mlir.oraValueIsAOpResult(yielded)) return null;
                const yield_owner = mlir.oraOpResultGetOwner(yielded);
                if (!self.operationNameEq(yield_owner, "arith.subi")) return null;
                if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return null;

                const sub_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
                const sub_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
                const rhs_const = self.tryGetConstIntValue(sub_rhs);
                if (!mlir.mlirValueEqual(sub_lhs, carried_arg) or rhs_const == null or rhs_const.? == 0) return null;

                const init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, 3), mode);
                const lb_ast = try self.encodeValueWithMode(lb_value, mode);
                const ub_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, 1), mode);
                const trip_count_ast = try self.encodeCanonicalPositiveStepScfForTripCount(lb_ast, ub_ast, step_u64);
                const delta_ast = try self.encodeValueWithMode(sub_rhs, mode);
                const delta_total_ast = try self.encodeArithmeticOp(.Mul, trip_count_ast, delta_ast);
                return try self.encodeArithmeticOp(.Sub, init_ast, delta_total_ast);
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn encodeCanonicalPositiveStepScfForTripCount(
        self: *Encoder,
        lb_ast: z3.Z3_ast,
        ub_ast: z3.Z3_ast,
        step: u64,
    ) EncodeError!z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, ub_ast);
        if (z3.Z3_get_sort_kind(self.context.ctx, sort) != z3.Z3_BV_SORT) return error.UnsupportedOperation;

        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const ub_le_lb = z3.Z3_mk_bvule(self.context.ctx, ub_ast, lb_ast);
        if (step == 1) {
            const distance = z3.Z3_mk_bv_sub(self.context.ctx, ub_ast, lb_ast);
            return z3.Z3_mk_ite(self.context.ctx, ub_le_lb, zero, distance);
        }

        const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, sort);
        const step_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, step, sort);
        const distance = z3.Z3_mk_bv_sub(self.context.ctx, ub_ast, lb_ast);
        const adjusted_distance = z3.Z3_mk_bv_sub(self.context.ctx, distance, one);
        const quotient = z3.Z3_mk_bv_udiv(self.context.ctx, adjusted_distance, step_ast);
        const rounded = z3.Z3_mk_bv_add(self.context.ctx, quotient, one);
        return z3.Z3_mk_ite(self.context.ctx, ub_le_lb, zero, rounded);
    }

    fn tryExtractIdentityScfForResult(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 3) return null;
        const num_iter_args = num_operands - 3;
        if (result_index >= num_iter_args) return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;
        const num_body_args: usize = @intCast(mlir.oraBlockGetNumArguments(body));
        if (num_body_args != num_iter_args + 1) return null;

        var current = mlir.oraBlockGetFirstOperation(body);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.return")) return null;
            if (std.mem.eql(u8, name, "scf.yield")) {
                const num_yield_operands: usize = @intCast(mlir.oraOperationGetNumOperands(current));
                if (result_index >= num_yield_operands) return null;
                const yielded = mlir.oraOperationGetOperand(current, result_index);
                const carried_arg = mlir.oraBlockGetArgument(body, result_index + 1);
                if (!mlir.mlirValueEqual(yielded, carried_arg)) return null;
                return try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, result_index + 3), mode);
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn tryEncodeFiniteScfForStateEffects(self: *Encoder, op: mlir.MlirOperation) bool {
        const loop_ctx = self.getFiniteScfForContext(op) orelse return false;
        var carried = self.getScfForInitCarriedValues(op, .Current) catch return false;
        defer self.allocator.free(carried);

        for (0..loop_ctx.trip_count) |iter_index| {
            self.bindFiniteScfForLoopArgs(loop_ctx, loop_ctx.lower_bound + iter_index * loop_ctx.step, carried) catch return false;
            self.invalidateBlockValueCaches(loop_ctx.body);

            var current = mlir.oraBlockGetFirstOperation(loop_ctx.body);
            while (!mlir.oraOperationIsNull(current)) {
                const next = mlir.oraOperationGetNextInBlock(current);
                const name_ref = mlir.oraOperationGetName(current);
                defer @import("mlir_c_api").freeStringRef(name_ref);
                const op_name = if (name_ref.data == null or name_ref.length == 0)
                    ""
                else
                    name_ref.data[0..name_ref.length];
                if (std.mem.eql(u8, op_name, "scf.yield")) break;
                self.encodeStateEffectsInOperation(current);
                current = next;
            }
            if (self.isDegraded()) {
                self.unbindFiniteScfForLoopArgs(loop_ctx);
                return false;
            }

            const yielded = self.extractScfForYieldValues(loop_ctx, .Current) catch {
                self.unbindFiniteScfForLoopArgs(loop_ctx);
                return false;
            } orelse {
                self.unbindFiniteScfForLoopArgs(loop_ctx);
                return false;
            };
            self.unbindFiniteScfForLoopArgs(loop_ctx);
            self.allocator.free(carried);
            carried = yielded;
        }

        return true;
    }

    fn tryExtractZeroIterationScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return null;
        const bind_count = try self.bindScfWhileBeforeArgs(while_op, mode);
        defer self.unbindScfWhileBeforeArgs(while_op, bind_count);

        var current = mlir.oraBlockGetFirstOperation(before_block);
        while (!mlir.oraOperationIsNull(current)) {
            const next = mlir.oraOperationGetNextInBlock(current);
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "scf.condition")) {
                const num_operands = mlir.oraOperationGetNumOperands(current);
                if (num_operands < 1) return null;

                const condition_value = mlir.oraOperationGetOperand(current, 0);
                const condition_ast = try self.encodeValueWithMode(condition_value, mode);
                const false_ast = self.encodeBoolConstant(false);
                if (!self.astEquivalent(condition_ast, false_ast)) return null;

                const value_operand_index: u32 = result_index + 1;
                if (value_operand_index >= num_operands) return null;
                return try self.encodeValueWithMode(mlir.oraOperationGetOperand(current, value_operand_index), mode);
            }

            current = next;
        }

        return null;
    }

    fn bindScfWhileBeforeArgs(self: *Encoder, while_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!usize {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return 0;

        const num_args: usize = @intCast(mlir.oraBlockGetNumArguments(before_block));
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(while_op));
        const bind_count = @min(num_args, num_operands);
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(before_block, @intCast(index));
            const init_value = mlir.oraOperationGetOperand(while_op, @intCast(index));
            try self.bindValue(arg, try self.encodeValueWithMode(init_value, mode));
        }
        return bind_count;
    }

    fn getScfWhileInitValues(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError![]z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(while_op));
        var values = try self.allocator.alloc(z3.Z3_ast, num_operands);
        errdefer self.allocator.free(values);
        for (0..num_operands) |index| {
            values[index] = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, @intCast(index)), mode);
        }
        return values;
    }

    fn unbindScfWhileBeforeArgs(self: *Encoder, while_op: mlir.MlirOperation, bind_count: usize) void {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return;
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(before_block, @intCast(index));
            _ = self.value_bindings.remove(@intFromPtr(arg.ptr));
        }
    }

    fn findScfConditionOp(self: *Encoder, while_op: mlir.MlirOperation) ?mlir.MlirOperation {
        _ = self;
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return null;

        var current = mlir.oraBlockGetFirstOperation(before_block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = mlir.oraOperationGetName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.condition")) return current;
            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn bindScfWhileAfterArgsFromCondition(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        condition_op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!usize {
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(after_block)) return 0;

        const num_args: usize = @intCast(mlir.oraBlockGetNumArguments(after_block));
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(condition_op));
        const bind_count = @min(num_args, if (num_operands > 0) num_operands - 1 else 0);
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(after_block, @intCast(index));
            const value = mlir.oraOperationGetOperand(condition_op, @intCast(index + 1));
            try self.bindValue(arg, try self.encodeValueWithMode(value, mode));
        }
        return bind_count;
    }

    fn unbindScfWhileAfterArgs(self: *Encoder, while_op: mlir.MlirOperation, bind_count: usize) void {
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(after_block)) return;
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(after_block, @intCast(index));
            _ = self.value_bindings.remove(@intFromPtr(arg.ptr));
        }
    }

    fn extractScfWhileYieldValues(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!?[]z3.Z3_ast {
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(after_block)) return null;

        var current = mlir.oraBlockGetFirstOperation(after_block);
        while (!mlir.oraOperationIsNull(current)) {
            const next = mlir.oraOperationGetNextInBlock(current);
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.yield")) {
                const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(current));
                var values = try self.allocator.alloc(z3.Z3_ast, num_operands);
                errdefer self.allocator.free(values);
                for (0..num_operands) |index| {
                    values[index] = try self.encodeValueWithMode(mlir.oraOperationGetOperand(current, @intCast(index)), mode);
                }
                return values;
            }
            current = next;
        }

        return null;
    }

    fn bindScfWhileBeforeArgsFromYielded(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        yielded: []const z3.Z3_ast,
    ) EncodeError!usize {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return 0;

        const num_args: usize = @intCast(mlir.oraBlockGetNumArguments(before_block));
        const bind_count = @min(num_args, yielded.len);
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(before_block, @intCast(index));
            try self.bindValue(arg, yielded[index]);
        }
        return bind_count;
    }

    fn bindScfWhileBeforeArgsFromValues(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        values: []const z3.Z3_ast,
    ) EncodeError!usize {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return 0;

        const num_args: usize = @intCast(mlir.oraBlockGetNumArguments(before_block));
        const bind_count = @min(num_args, values.len);
        for (0..bind_count) |index| {
            const arg = mlir.oraBlockGetArgument(before_block, @intCast(index));
            try self.bindValue(arg, values[index]);
        }
        return bind_count;
    }

    fn tryExtractFiniteScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const condition_op = self.findScfConditionOp(while_op) orelse return null;

        var carried = try self.getScfWhileInitValues(while_op, mode);
        defer self.allocator.free(carried);

        var iteration_count: usize = 0;
        while (iteration_count < finite_scf_while_unroll_limit) : (iteration_count += 1) {
            const before_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, carried);
            self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetBeforeBlock(while_op));
            self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetAfterBlock(while_op));
            const condition_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), mode);
            if (self.astSimplifiesToBool(condition_ast)) |condition_const| {
                if (!condition_const) {
                    const num_operands = mlir.oraOperationGetNumOperands(condition_op);
                    const value_operand_index: u32 = result_index + 1;
                    if (value_operand_index >= num_operands) {
                        self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                        return null;
                    }
                    const result = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, value_operand_index), mode);
                    self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                    return result;
                }
            }
            if ((self.astSimplifiesToBool(condition_ast) orelse return null) != true) {
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return null;
            }

            const after_bind_count = try self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, mode);
            const yielded = (try self.extractScfWhileYieldValues(while_op, mode)) orelse {
                self.unbindScfWhileAfterArgs(while_op, after_bind_count);
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return null;
            };

            self.unbindScfWhileAfterArgs(while_op, after_bind_count);
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            self.allocator.free(carried);
            carried = yielded;
        }

        return null;
    }

    const CanonicalWhileCarriedUpdate = union(enum) {
        identity,
        add_const: u64,
        sub_const: u64,
    };

    fn tryEncodeCanonicalWhileAffineCarriedResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        after_block: mlir.MlirBlock,
        yield_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
        step_count: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const target_after_arg = mlir.oraBlockGetArgument(after_block, result_index);
        const target_yield = mlir.oraOperationGetOperand(yield_op, result_index);
        if (!mlir.oraValueIsAOpResult(target_yield)) return null;

        const owner = mlir.oraOpResultGetOwner(target_yield);
        const op_name_ref = mlir.oraOperationGetName(owner);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];
        if (!std.mem.eql(u8, op_name, "arith.addi") and !std.mem.eql(u8, op_name, "arith.subi")) return null;
        if (mlir.oraOperationGetNumOperands(owner) != 2) return null;

        const lhs = mlir.oraOperationGetOperand(owner, 0);
        const rhs = mlir.oraOperationGetOperand(owner, 1);
        var other_index_opt: ?u32 = null;
        var is_add = false;
        var is_sub = false;

        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(while_op));
        if (std.mem.eql(u8, op_name, "arith.addi")) {
            if (mlir.mlirValueEqual(lhs, target_after_arg)) {
                var idx: u32 = 0;
                while (idx < num_results) : (idx += 1) {
                    if (idx == result_index) continue;
                    if (mlir.mlirValueEqual(rhs, mlir.oraBlockGetArgument(after_block, idx))) {
                        other_index_opt = idx;
                        is_add = true;
                        break;
                    }
                }
            } else if (mlir.mlirValueEqual(rhs, target_after_arg)) {
                var idx: u32 = 0;
                while (idx < num_results) : (idx += 1) {
                    if (idx == result_index) continue;
                    if (mlir.mlirValueEqual(lhs, mlir.oraBlockGetArgument(after_block, idx))) {
                        other_index_opt = idx;
                        is_add = true;
                        break;
                    }
                }
            }
        } else if (std.mem.eql(u8, op_name, "arith.subi")) {
            if (mlir.mlirValueEqual(lhs, target_after_arg)) {
                var idx: u32 = 0;
                while (idx < num_results) : (idx += 1) {
                    if (idx == result_index) continue;
                    if (mlir.mlirValueEqual(rhs, mlir.oraBlockGetArgument(after_block, idx))) {
                        other_index_opt = idx;
                        is_sub = true;
                        break;
                    }
                }
            }
        }

        const other_index = other_index_opt orelse return null;
        const other_yield = mlir.oraOperationGetOperand(yield_op, other_index);
        const other_after_arg = mlir.oraBlockGetArgument(after_block, other_index);
        const other_update = self.classifyCanonicalWhileCarriedUpdate(other_yield, other_after_arg) orelse return null;
        if (other_update != .identity) return null;

        const target_init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, result_index), mode);
        const other_init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, other_index), mode);
        const total_delta = try self.encodeArithmeticOp(.Mul, step_count, other_init_ast);

        if (is_add) return try self.encodeArithmeticOp(.Add, target_init_ast, total_delta);
        if (is_sub) return try self.encodeArithmeticOp(.Sub, target_init_ast, total_delta);
        return null;
    }

    fn classifyCanonicalWhileCarriedUpdate(
        self: *Encoder,
        yielded_value: mlir.MlirValue,
        after_arg: mlir.MlirValue,
    ) ?CanonicalWhileCarriedUpdate {
        if (mlir.mlirValueEqual(yielded_value, after_arg)) return .identity;
        if (!mlir.oraValueIsAOpResult(yielded_value)) return null;

        const owner = mlir.oraOpResultGetOwner(yielded_value);
        if (self.operationNameEq(owner, "arith.addi")) {
            if (mlir.oraOperationGetNumOperands(owner) != 2) return null;
            const lhs = mlir.oraOperationGetOperand(owner, 0);
            const rhs = mlir.oraOperationGetOperand(owner, 1);
            const lhs_const = self.tryGetConstIntValue(lhs);
            const rhs_const = self.tryGetConstIntValue(rhs);
            if (lhs_const != null and lhs_const.? != 0 and mlir.mlirValueEqual(rhs, after_arg)) {
                const delta = std.math.cast(u64, lhs_const.?) orelse return null;
                return .{ .add_const = delta };
            }
            if (rhs_const != null and rhs_const.? != 0 and mlir.mlirValueEqual(lhs, after_arg)) {
                const delta = std.math.cast(u64, rhs_const.?) orelse return null;
                return .{ .add_const = delta };
            }
            return null;
        }

        if (self.operationNameEq(owner, "arith.subi")) {
            if (mlir.oraOperationGetNumOperands(owner) != 2) return null;
            const lhs = mlir.oraOperationGetOperand(owner, 0);
            const rhs = mlir.oraOperationGetOperand(owner, 1);
            const rhs_const = self.tryGetConstIntValue(rhs);
            if (!mlir.mlirValueEqual(lhs, after_arg) or rhs_const == null or rhs_const.? == 0) return null;
            const delta = std.math.cast(u64, rhs_const.?) orelse return null;
            return .{ .sub_const = delta };
        }

        return null;
    }

    fn tryExtractCanonicalUnsignedScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(while_op));
        if (num_operands == 0 or result_index >= num_operands) return null;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return null;
        if (mlir.oraBlockGetNumArguments(before_block) != num_operands or mlir.oraBlockGetNumArguments(after_block) != num_operands) return null;

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) != num_operands + 1) return null;

        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return null;
        const predicate = try self.getCmpPredicate(cmp_op);
        if (predicate != 6 and predicate != 8) return null; // ult / ugt only
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        var control_index_opt: ?usize = null;
        var bound_value: ?mlir.MlirValue = null;
        var control_predicate: ?u64 = null;
        for (0..num_operands) |idx| {
            const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(idx));
            if (mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) {
                control_index_opt = idx;
                bound_value = cmp_rhs;
                control_predicate = predicate;
                break;
            }
            if (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)) {
                control_index_opt = idx;
                bound_value = cmp_lhs;
                control_predicate = switch (predicate) {
                    6 => 8, // bound < control => control > bound
                    8 => 6, // bound > control => control < bound
                    else => null,
                };
                break;
            }
        }
        const control_index = control_index_opt orelse return null;
        const normalized_predicate = control_predicate orelse return null;

        for (0..num_operands) |idx| {
            const false_value = mlir.oraOperationGetOperand(condition_op, @intCast(idx + 1));
            const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(idx));
            if (!mlir.mlirValueEqual(false_value, before_arg)) return null;
        }

        const yield_op = self.findScfYieldOp(after_block) orelse return null;
        if (mlir.oraOperationGetNumOperands(yield_op) != num_operands) return null;

        const control_after_arg = mlir.oraBlockGetArgument(after_block, @intCast(control_index));
        const control_yield = mlir.oraOperationGetOperand(yield_op, @intCast(control_index));
        const control_update = self.classifyCanonicalWhileCarriedUpdate(control_yield, control_after_arg) orelse return null;

        const init_control_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, @intCast(control_index)), mode);
        const bound_ast = try self.encodeValueWithMode(bound_value.?, mode);
        const step_count = switch (control_update) {
            .add_const => |delta| if (normalized_predicate == 6) try self.encodeUnsignedPositiveStepCount(
                try self.encodeUnsignedPositiveDistance(init_control_ast, bound_ast),
                delta,
            ) else return null,
            .sub_const => |delta| if (normalized_predicate == 8) try self.encodeUnsignedPositiveStepCount(
                try self.encodeUnsignedPositiveDistance(bound_ast, init_control_ast),
                delta,
            ) else return null,
            .identity => return null,
        };

        const target_init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, result_index), mode);
        const target_after_arg = mlir.oraBlockGetArgument(after_block, result_index);
        const target_yield = mlir.oraOperationGetOperand(yield_op, result_index);
        if (self.classifyCanonicalWhileCarriedUpdate(target_yield, target_after_arg)) |target_update| {
            return switch (target_update) {
                .identity => target_init_ast,
                .add_const => |delta| blk: {
                    const sort = z3.Z3_get_sort(self.context.ctx, target_init_ast);
                    const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                    const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
                    break :blk try self.encodeArithmeticOp(.Add, target_init_ast, total_delta);
                },
                .sub_const => |delta| blk: {
                    const sort = z3.Z3_get_sort(self.context.ctx, target_init_ast);
                    const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                    const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
                    break :blk try self.encodeArithmeticOp(.Sub, target_init_ast, total_delta);
                },
            };
        }

        return try self.tryEncodeCanonicalWhileAffineCarriedResult(
            while_op,
            after_block,
            yield_op,
            result_index,
            mode,
            step_count,
        );
    }

    fn tryExtractCanonicalSignedScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(while_op));
        if (num_operands == 0 or result_index >= num_operands) return null;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return null;
        if (mlir.oraBlockGetNumArguments(before_block) != num_operands or mlir.oraBlockGetNumArguments(after_block) != num_operands) return null;

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) != num_operands + 1) return null;

        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return null;
        const predicate = try self.getCmpPredicate(cmp_op);
        if (predicate != 2 and predicate != 4) return null; // slt / sgt only
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        var control_index_opt: ?usize = null;
        var bound_value: ?mlir.MlirValue = null;
        var control_predicate: ?u64 = null;
        for (0..num_operands) |idx| {
            const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(idx));
            if (mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) {
                control_index_opt = idx;
                bound_value = cmp_rhs;
                control_predicate = predicate;
                break;
            }
            if (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)) {
                control_index_opt = idx;
                bound_value = cmp_lhs;
                control_predicate = switch (predicate) {
                    2 => 4,
                    4 => 2,
                    else => null,
                };
                break;
            }
        }
        const control_index = control_index_opt orelse return null;
        const normalized_predicate = control_predicate orelse return null;

        for (0..num_operands) |idx| {
            const false_value = mlir.oraOperationGetOperand(condition_op, @intCast(idx + 1));
            const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(idx));
            if (!mlir.mlirValueEqual(false_value, before_arg)) return null;
        }

        const yield_op = self.findScfYieldOp(after_block) orelse return null;
        if (mlir.oraOperationGetNumOperands(yield_op) != num_operands) return null;

        const control_after_arg = mlir.oraBlockGetArgument(after_block, @intCast(control_index));
        const control_yield = mlir.oraOperationGetOperand(yield_op, @intCast(control_index));
        const control_update = self.classifyCanonicalWhileCarriedUpdate(control_yield, control_after_arg) orelse return null;

        if (num_operands == 1 and result_index == 0) {
            switch (control_update) {
                .add_const => |delta| if (predicate == 2 and delta == 1) return null,
                .sub_const => |delta| if (predicate == 4 and delta == 1) return null,
                .identity => return null,
            }
        }

        const init_control_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, @intCast(control_index)), mode);
        const bound_ast = try self.encodeValueWithMode(bound_value.?, mode);
        const step_count = switch (control_update) {
            .add_const => |delta| if (normalized_predicate == 2) try self.encodeUnsignedPositiveStepCount(
                try self.encodeSignedPositiveDistance(init_control_ast, bound_ast),
                delta,
            ) else return null,
            .sub_const => |delta| if (normalized_predicate == 4) try self.encodeUnsignedPositiveStepCount(
                try self.encodeSignedPositiveDistance(bound_ast, init_control_ast),
                delta,
            ) else return null,
            .identity => return null,
        };

        const target_init_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(while_op, result_index), mode);
        const target_after_arg = mlir.oraBlockGetArgument(after_block, result_index);
        const target_yield = mlir.oraOperationGetOperand(yield_op, result_index);
        if (self.classifyCanonicalWhileCarriedUpdate(target_yield, target_after_arg)) |target_update| {
            return switch (target_update) {
                .identity => target_init_ast,
                .add_const => |delta| blk: {
                    const sort = z3.Z3_get_sort(self.context.ctx, target_init_ast);
                    const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                    const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
                    break :blk try self.encodeArithmeticOp(.Add, target_init_ast, total_delta);
                },
                .sub_const => |delta| blk: {
                    const sort = z3.Z3_get_sort(self.context.ctx, target_init_ast);
                    const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
                    const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
                    break :blk try self.encodeArithmeticOp(.Sub, target_init_ast, total_delta);
                },
            };
        }

        return try self.tryEncodeCanonicalWhileAffineCarriedResult(
            while_op,
            after_block,
            yield_op,
            result_index,
            mode,
            step_count,
        );
    }

    fn tryExtractCanonicalIncrementScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        if (result_index != 0) return null;
        if (mlir.oraOperationGetNumOperands(while_op) != 1) return null;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return null;
        if (mlir.oraBlockGetNumArguments(before_block) != 1 or mlir.oraBlockGetNumArguments(after_block) != 1) return null;

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) != 2) return null;

        const before_arg = mlir.oraBlockGetArgument(before_block, 0);
        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return null;
        const predicate = try self.getCmpPredicate(cmp_op);
        if (predicate != 2 and predicate != 6) return null; // slt / ult
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        const bound_value, const normalized_predicate = blk: {
            if (mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) {
                break :blk .{ cmp_rhs, predicate };
            }
            if (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)) {
                break :blk .{ cmp_lhs, switch (predicate) {
                    2 => @as(u64, 4),
                    6 => @as(u64, 8),
                    else => return null,
                } };
            }
            return null;
        };

        const false_value = mlir.oraOperationGetOperand(condition_op, 1);
        if (!mlir.mlirValueEqual(false_value, before_arg)) return null;

        const after_arg = mlir.oraBlockGetArgument(after_block, 0);
        const yield_op = self.findScfYieldOp(after_block) orelse return null;
        if (mlir.oraOperationGetNumOperands(yield_op) != 1) return null;
        const yield_value = mlir.oraOperationGetOperand(yield_op, 0);
        const yield_owner = if (mlir.oraValueIsAOpResult(yield_value))
            mlir.oraOpResultGetOwner(yield_value)
        else
            mlir.MlirOperation{ .ptr = null };
        if (mlir.oraOperationIsNull(yield_owner) or !self.operationNameEq(yield_owner, "arith.addi")) return null;
        if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return null;
        const add_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
        const add_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
        const one_from_lhs = self.tryGetConstIntValue(add_lhs);
        const one_from_rhs = self.tryGetConstIntValue(add_rhs);
        const delta_value =
            if (one_from_lhs != null and one_from_lhs.? != 0 and mlir.mlirValueEqual(add_rhs, after_arg))
                add_lhs
            else if (one_from_rhs != null and one_from_rhs.? != 0 and mlir.mlirValueEqual(add_lhs, after_arg))
                add_rhs
            else
                return null;
        const delta_const = self.tryGetConstIntValue(delta_value) orelse return null;
        const delta_u64 = std.math.cast(u64, delta_const) orelse return null;

        const init_value = mlir.oraOperationGetOperand(while_op, 0);
        const init_ast = try self.encodeValueWithMode(init_value, mode);
        const bound_ast = try self.encodeValueWithMode(bound_value, mode);

        if (normalized_predicate == 6) {
            return try self.encodeUnsignedCanonicalWhileIncrementResult(init_ast, bound_ast, delta_u64);
        }

        if (delta_u64 != 1) {
            return try self.encodeSignedCanonicalWhileIncrementResult(init_ast, bound_ast, delta_u64);
        }
        const before_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, &[_]z3.Z3_ast{init_ast});
        defer self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
        self.invalidateBlockValueCaches(before_block);
        self.invalidateBlockValueCaches(after_block);

        const initial_condition = try self.encodeValueWithMode(condition_value, mode);
        return try self.encodeControlFlow("scf.if", initial_condition, bound_ast, init_ast);
    }

    fn tryExtractCanonicalDecrementScfWhileResult(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        if (result_index != 0) return null;
        if (mlir.oraOperationGetNumOperands(while_op) != 1) return null;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return null;
        if (mlir.oraBlockGetNumArguments(before_block) != 1 or mlir.oraBlockGetNumArguments(after_block) != 1) return null;

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) != 2) return null;

        const before_arg = mlir.oraBlockGetArgument(before_block, 0);
        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return null;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return null;
        const predicate = try self.getCmpPredicate(cmp_op);
        if (predicate != 4 and predicate != 8) return null; // sgt / ugt
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return null;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        const bound_value, const normalized_predicate = blk: {
            if (mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) {
                break :blk .{ cmp_rhs, predicate };
            }
            if (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)) {
                break :blk .{ cmp_lhs, switch (predicate) {
                    4 => @as(u64, 2),
                    8 => @as(u64, 6),
                    else => return null,
                } };
            }
            return null;
        };

        const false_value = mlir.oraOperationGetOperand(condition_op, 1);
        if (!mlir.mlirValueEqual(false_value, before_arg)) return null;

        const after_arg = mlir.oraBlockGetArgument(after_block, 0);
        const yield_op = self.findScfYieldOp(after_block) orelse return null;
        if (mlir.oraOperationGetNumOperands(yield_op) != 1) return null;
        const yield_value = mlir.oraOperationGetOperand(yield_op, 0);
        const yield_owner = if (mlir.oraValueIsAOpResult(yield_value))
            mlir.oraOpResultGetOwner(yield_value)
        else
            mlir.MlirOperation{ .ptr = null };
        if (mlir.oraOperationIsNull(yield_owner) or !self.operationNameEq(yield_owner, "arith.subi")) return null;
        if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return null;
        const sub_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
        const sub_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
        const rhs_const = self.tryGetConstIntValue(sub_rhs);
        if (!mlir.mlirValueEqual(sub_lhs, after_arg) or rhs_const == null or rhs_const.? == 0) return null;
        const delta_u64 = std.math.cast(u64, rhs_const.?) orelse return null;

        const init_value = mlir.oraOperationGetOperand(while_op, 0);
        const init_ast = try self.encodeValueWithMode(init_value, mode);
        const bound_ast = try self.encodeValueWithMode(bound_value, mode);

        if (normalized_predicate == 8) {
            return try self.encodeUnsignedCanonicalWhileDecrementResult(init_ast, bound_ast, delta_u64);
        }

        // Swapped-compare forms normalize to slt/ult. Those are not valid
        // decrement closed forms because the control variable moves away from
        // the bound, so we must fail closed instead of routing them through the
        // signed decrement path.
        if (normalized_predicate != 4) return null;

        if (delta_u64 != 1) {
            return try self.encodeSignedCanonicalWhileDecrementResult(init_ast, bound_ast, delta_u64);
        }
        const before_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, &[_]z3.Z3_ast{init_ast});
        defer self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
        self.invalidateBlockValueCaches(before_block);
        self.invalidateBlockValueCaches(after_block);

        const initial_condition = try self.encodeValueWithMode(condition_value, mode);
        return try self.encodeControlFlow("scf.if", initial_condition, bound_ast, init_ast);
    }

    fn encodeUnsignedCanonicalWhileIncrementResult(
        self: *Encoder,
        init_ast: z3.Z3_ast,
        bound_ast: z3.Z3_ast,
        delta: u64,
    ) EncodeError!z3.Z3_ast {
        const distance = try self.encodeUnsignedPositiveDistance(init_ast, bound_ast);
        const step_count = try self.encodeUnsignedPositiveStepCount(distance, delta);
        const delta_sort = z3.Z3_get_sort(self.context.ctx, init_ast);
        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, delta_sort);
        const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
        return try self.encodeArithmeticOp(.Add, init_ast, total_delta);
    }

    fn encodeUnsignedCanonicalWhileDecrementResult(
        self: *Encoder,
        init_ast: z3.Z3_ast,
        bound_ast: z3.Z3_ast,
        delta: u64,
    ) EncodeError!z3.Z3_ast {
        const distance = try self.encodeUnsignedPositiveDistance(bound_ast, init_ast);
        const step_count = try self.encodeUnsignedPositiveStepCount(distance, delta);
        const delta_sort = z3.Z3_get_sort(self.context.ctx, init_ast);
        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, delta_sort);
        const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
        return try self.encodeArithmeticOp(.Sub, init_ast, total_delta);
    }

    fn encodeSignedCanonicalWhileIncrementResult(
        self: *Encoder,
        init_ast: z3.Z3_ast,
        bound_ast: z3.Z3_ast,
        delta: u64,
    ) EncodeError!z3.Z3_ast {
        const distance = try self.encodeSignedPositiveDistance(init_ast, bound_ast);
        const step_count = try self.encodeUnsignedPositiveStepCount(distance, delta);
        const delta_sort = z3.Z3_get_sort(self.context.ctx, init_ast);
        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, delta_sort);
        const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
        return try self.encodeArithmeticOp(.Add, init_ast, total_delta);
    }

    fn encodeSignedCanonicalWhileDecrementResult(
        self: *Encoder,
        init_ast: z3.Z3_ast,
        bound_ast: z3.Z3_ast,
        delta: u64,
    ) EncodeError!z3.Z3_ast {
        const distance = try self.encodeSignedPositiveDistance(bound_ast, init_ast);
        const step_count = try self.encodeUnsignedPositiveStepCount(distance, delta);
        const delta_sort = z3.Z3_get_sort(self.context.ctx, init_ast);
        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, delta_sort);
        const total_delta = try self.encodeArithmeticOp(.Mul, step_count, delta_ast);
        return try self.encodeArithmeticOp(.Sub, init_ast, total_delta);
    }

    fn encodeUnsignedPositiveDistance(
        self: *Encoder,
        lower_ast: z3.Z3_ast,
        upper_ast: z3.Z3_ast,
    ) EncodeError!z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, upper_ast);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const upper_le_lower = z3.Z3_mk_bvule(self.context.ctx, upper_ast, lower_ast);
        const raw_distance = z3.Z3_mk_bv_sub(self.context.ctx, upper_ast, lower_ast);
        return z3.Z3_mk_ite(self.context.ctx, upper_le_lower, zero, raw_distance);
    }

    fn encodeSignedPositiveDistance(
        self: *Encoder,
        lower_ast: z3.Z3_ast,
        upper_ast: z3.Z3_ast,
    ) EncodeError!z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, upper_ast);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const upper_le_lower = z3.Z3_mk_bvsle(self.context.ctx, upper_ast, lower_ast);
        const raw_distance = z3.Z3_mk_bv_sub(self.context.ctx, upper_ast, lower_ast);
        return z3.Z3_mk_ite(self.context.ctx, upper_le_lower, zero, raw_distance);
    }

    fn encodeUnsignedPositiveStepCount(
        self: *Encoder,
        distance_ast: z3.Z3_ast,
        delta: u64,
    ) EncodeError!z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, distance_ast);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        if (delta == 1) return distance_ast;

        const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, sort);
        const delta_ast = z3.Z3_mk_unsigned_int64(self.context.ctx, delta, sort);
        const distance_is_zero = z3.Z3_mk_eq(self.context.ctx, distance_ast, zero);
        const adjusted_distance = z3.Z3_mk_bv_sub(self.context.ctx, distance_ast, one);
        const quotient = z3.Z3_mk_bv_udiv(self.context.ctx, adjusted_distance, delta_ast);
        const rounded = z3.Z3_mk_bv_add(self.context.ctx, quotient, one);
        return z3.Z3_mk_ite(self.context.ctx, distance_is_zero, zero, rounded);
    }

    fn isCanonicalIncrementScfWhile(self: *Encoder, while_op: mlir.MlirOperation) bool {
        if (mlir.oraOperationGetNumOperands(while_op) != 1) return false;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return false;
        if (mlir.oraBlockGetNumArguments(before_block) != 1 or mlir.oraBlockGetNumArguments(after_block) != 1) return false;

        const condition_op = self.findScfConditionOp(while_op) orelse return false;
        if (mlir.oraOperationGetNumOperands(condition_op) != 2) return false;

        const before_arg = mlir.oraBlockGetArgument(before_block, 0);
        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return false;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return false;
        const predicate = self.getCmpPredicate(cmp_op) catch return false;
        if (predicate != 2 and predicate != 6) return false; // slt / ult
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return false;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        if (!((mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) or
            (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)))) return false;
        if (!mlir.mlirValueEqual(mlir.oraOperationGetOperand(condition_op, 1), before_arg)) return false;

        const after_arg = mlir.oraBlockGetArgument(after_block, 0);
        const yield_op = self.findScfYieldOp(after_block) orelse return false;
        if (mlir.oraOperationGetNumOperands(yield_op) != 1) return false;
        const yield_value = mlir.oraOperationGetOperand(yield_op, 0);
        if (!mlir.oraValueIsAOpResult(yield_value)) return false;
        const yield_owner = mlir.oraOpResultGetOwner(yield_value);
        if (!self.operationNameEq(yield_owner, "arith.addi")) return false;
        if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return false;

        const add_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
        const add_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
        const lhs_const = self.tryGetConstIntValue(add_lhs);
        const rhs_const = self.tryGetConstIntValue(add_rhs);
        if (predicate == 6) {
            return (lhs_const != null and lhs_const.? != 0 and mlir.mlirValueEqual(add_rhs, after_arg)) or
                (rhs_const != null and rhs_const.? != 0 and mlir.mlirValueEqual(add_lhs, after_arg));
        }
        return (lhs_const != null and lhs_const.? != 0 and mlir.mlirValueEqual(add_rhs, after_arg)) or
            (rhs_const != null and rhs_const.? != 0 and mlir.mlirValueEqual(add_lhs, after_arg));
    }

    fn isCanonicalDecrementScfWhile(self: *Encoder, while_op: mlir.MlirOperation) bool {
        if (mlir.oraOperationGetNumOperands(while_op) != 1) return false;

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) return false;
        if (mlir.oraBlockGetNumArguments(before_block) != 1 or mlir.oraBlockGetNumArguments(after_block) != 1) return false;

        const condition_op = self.findScfConditionOp(while_op) orelse return false;
        if (mlir.oraOperationGetNumOperands(condition_op) != 2) return false;

        const before_arg = mlir.oraBlockGetArgument(before_block, 0);
        const condition_value = mlir.oraOperationGetOperand(condition_op, 0);
        if (!mlir.oraValueIsAOpResult(condition_value)) return false;
        const cmp_op = mlir.oraOpResultGetOwner(condition_value);
        if (!self.operationNameEq(cmp_op, "arith.cmpi")) return false;
        const predicate = self.getCmpPredicate(cmp_op) catch return false;
        if (predicate != 4 and predicate != 8) return false; // sgt / ugt
        if (mlir.oraOperationGetNumOperands(cmp_op) != 2) return false;

        const cmp_lhs = mlir.oraOperationGetOperand(cmp_op, 0);
        const cmp_rhs = mlir.oraOperationGetOperand(cmp_op, 1);
        if (!((mlir.mlirValueEqual(cmp_lhs, before_arg) and !mlir.mlirValueEqual(cmp_rhs, before_arg)) or
            (mlir.mlirValueEqual(cmp_rhs, before_arg) and !mlir.mlirValueEqual(cmp_lhs, before_arg)))) return false;
        if (!mlir.mlirValueEqual(mlir.oraOperationGetOperand(condition_op, 1), before_arg)) return false;

        const after_arg = mlir.oraBlockGetArgument(after_block, 0);
        const yield_op = self.findScfYieldOp(after_block) orelse return false;
        if (mlir.oraOperationGetNumOperands(yield_op) != 1) return false;
        const yield_value = mlir.oraOperationGetOperand(yield_op, 0);
        if (!mlir.oraValueIsAOpResult(yield_value)) return false;
        const yield_owner = mlir.oraOpResultGetOwner(yield_value);
        if (!self.operationNameEq(yield_owner, "arith.subi")) return false;
        if (mlir.oraOperationGetNumOperands(yield_owner) != 2) return false;

        const sub_lhs = mlir.oraOperationGetOperand(yield_owner, 0);
        const sub_rhs = mlir.oraOperationGetOperand(yield_owner, 1);
        const rhs_const = self.tryGetConstIntValue(sub_rhs);
        if (predicate == 8) {
            return mlir.mlirValueEqual(sub_lhs, after_arg) and rhs_const != null and rhs_const.? != 0;
        }
        return mlir.mlirValueEqual(sub_lhs, after_arg) and rhs_const != null and rhs_const.? != 0;
    }

    fn allScfWhileResultsEncodeExactly(self: *Encoder, while_op: mlir.MlirOperation, mode: EncodeMode) bool {
        const num_results: u32 = @intCast(mlir.oraOperationGetNumResults(while_op));
        // A zero-result while contributes no value summary. For no-write state
        // summaries, that is already exact on the returning path.
        if (num_results == 0) return true;
        var result_idx: u32 = 0;
        while (result_idx < num_results) : (result_idx += 1) {
            const exact =
                (self.tryExtractZeroIterationScfWhileResult(while_op, result_idx, mode) catch null) orelse
                (self.tryExtractCanonicalUnsignedScfWhileResult(while_op, result_idx, mode) catch null) orelse
                (self.tryExtractCanonicalSignedScfWhileResult(while_op, result_idx, mode) catch null) orelse
                (self.tryExtractCanonicalIncrementScfWhileResult(while_op, result_idx, mode) catch null) orelse
                (self.tryExtractCanonicalDecrementScfWhileResult(while_op, result_idx, mode) catch null) orelse
                (self.tryExtractFiniteScfWhileResult(while_op, result_idx, mode) catch null);
            if (exact == null) return false;
        }
        return true;
    }

    fn findScfYieldOp(self: *Encoder, block: mlir.MlirBlock) ?mlir.MlirOperation {
        _ = self;
        if (mlir.oraBlockIsNull(block)) return null;
        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = mlir.oraOperationGetName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0) "" else name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.yield")) return current;
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn tryEncodeFiniteScfWhileStateEffects(self: *Encoder, while_op: mlir.MlirOperation) bool {
        const condition_op = self.findScfConditionOp(while_op) orelse return false;
        var carried = self.getScfWhileInitValues(while_op, .Current) catch return false;
        defer self.allocator.free(carried);

        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(after_block)) return false;

        var iteration_count: usize = 0;
        while (iteration_count < finite_scf_while_unroll_limit) : (iteration_count += 1) {
            const before_bind_count = self.bindScfWhileBeforeArgsFromValues(while_op, carried) catch return false;
            self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetBeforeBlock(while_op));
            self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetAfterBlock(while_op));
            const condition_ast = self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), .Current) catch {
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            };
            if (self.astSimplifiesToBool(condition_ast)) |condition_const| {
                if (!condition_const) {
                    self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                    return true;
                }
            }
            if ((self.astSimplifiesToBool(condition_ast) orelse return false) != true) {
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            }

            const after_bind_count = self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, .Current) catch {
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            };

            var current = mlir.oraBlockGetFirstOperation(after_block);
            while (!mlir.oraOperationIsNull(current)) {
                const next = mlir.oraOperationGetNextInBlock(current);
                const name_ref = mlir.oraOperationGetName(current);
                defer @import("mlir_c_api").freeStringRef(name_ref);
                const op_name = if (name_ref.data == null or name_ref.length == 0)
                    ""
                else
                    name_ref.data[0..name_ref.length];
                if (std.mem.eql(u8, op_name, "scf.yield")) break;
                self.encodeStateEffectsInOperation(current);
                current = next;
            }
            if (self.isDegraded()) {
                self.unbindScfWhileAfterArgs(while_op, after_bind_count);
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            }

            const yielded = (self.extractScfWhileYieldValues(while_op, .Current) catch {
                self.unbindScfWhileAfterArgs(while_op, after_bind_count);
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            }) orelse {
                self.unbindScfWhileAfterArgs(while_op, after_bind_count);
                self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
                return false;
            };

            self.unbindScfWhileAfterArgs(while_op, after_bind_count);
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            self.allocator.free(carried);
            carried = yielded;
        }

        return false;
    }

    fn isStaticallyFalseScfWhile(self: *Encoder, while_op: mlir.MlirOperation, mode: EncodeMode) EncodeError!bool {
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        if (mlir.oraBlockIsNull(before_block)) return false;
        const bind_count = try self.bindScfWhileBeforeArgs(while_op, mode);
        defer self.unbindScfWhileBeforeArgs(while_op, bind_count);

        var current = mlir.oraBlockGetFirstOperation(before_block);
        while (!mlir.oraOperationIsNull(current)) {
            const next = mlir.oraOperationGetNextInBlock(current);
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "scf.condition")) {
                const num_operands = mlir.oraOperationGetNumOperands(current);
                if (num_operands < 1) return false;

                const condition_value = mlir.oraOperationGetOperand(current, 0);
                const condition_ast = try self.encodeValueWithMode(condition_value, mode);
                return self.astEquivalent(condition_ast, self.encodeBoolConstant(false));
            }

            current = next;
        }

        return false;
    }

    fn tryExtractEquivalentTryStmtResult(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(try_stmt));
        if (num_regions < 2) return null;

        const try_expr = (try self.extractRegionYield(try_stmt, 0, result_index, mode)) orelse return null;
        const catch_expr = (try self.extractRegionYield(try_stmt, 1, result_index, mode)) orelse return null;
        if (!self.astEquivalent(try_expr, catch_expr)) return null;
        return try_expr;
    }

    const DirectTryUnwrapInfo = struct {
        is_err: z3.Z3_ast,
        ok_expr: z3.Z3_ast,
    };

    fn trySummarizeTryValue(
        self: *Encoder,
        value: mlir.MlirValue,
        mode: EncodeMode,
    ) EncodeError!?DirectTryUnwrapInfo {
        if (!mlir.oraValueIsAOpResult(value)) {
            return .{
                .is_err = self.encodeBoolConstant(false),
                .ok_expr = try self.encodeValueWithMode(value, mode),
            };
        }

        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;

        if (self.operationNameEq(owner, "ora.error.unwrap")) {
            const unwrap_operand = mlir.oraOperationGetOperand(owner, 0);
            const operand_type = mlir.oraValueGetType(unwrap_operand);
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
            if (mlir.oraTypeIsNull(success_type)) return null;

            const operand_expr = try self.encodeValueWithMode(unwrap_operand, mode);
            const eu = try self.getErrorUnionSort(operand_type, success_type);
            const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{operand_expr});
            const ok_expr = z3.Z3_mk_app(self.context.ctx, eu.proj_ok, 1, &[_]z3.Z3_ast{operand_expr});
            return .{ .is_err = is_err, .ok_expr = ok_expr };
        }

        if (self.operationNameEq(owner, "scf.if")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            if (mlir.oraOperationGetNumOperands(owner) < 1) return null;

            const condition_value = mlir.oraOperationGetOperand(owner, 0);
            const condition = try self.encodeValueWithMode(condition_value, mode);
            const then_value = try self.extractRegionYieldValue(owner, 0, result_index) orelse return null;
            const else_value = try self.extractRegionYieldValue(owner, 1, result_index) orelse return null;
            const then_summary = (try self.trySummarizeTryValue(then_value, mode)) orelse return null;
            const else_summary = (try self.trySummarizeTryValue(else_value, mode)) orelse return null;

            const catch_pred = self.encodeOr(&.{
                self.encodeAnd(&.{ condition, then_summary.is_err }),
                self.encodeAnd(&.{ self.encodeNot(condition), else_summary.is_err }),
            });
            const ok_expr = self.encodeIte(condition, then_summary.ok_expr, else_summary.ok_expr);
            return .{ .is_err = catch_pred, .ok_expr = ok_expr };
        }

        if (self.operationNameEq(owner, "scf.execute_region")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const yielded_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return null;
            return (try self.trySummarizeTryValue(yielded_value, mode)) orelse null;
        }

        if (self.operationNameEq(owner, "scf.for")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const exact =
                (try self.tryExtractFiniteScfForResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalIncrementScfForResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalScfForDerivedResult(owner, result_index, mode));
            if (exact) |ok_expr| {
                return .{ .is_err = self.encodeBoolConstant(false), .ok_expr = ok_expr };
            }
        }

        if (self.operationNameEq(owner, "scf.while")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const exact =
                (try self.tryExtractZeroIterationScfWhileResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalUnsignedScfWhileResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalSignedScfWhileResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalIncrementScfWhileResult(owner, result_index, mode)) orelse
                (try self.tryExtractCanonicalDecrementScfWhileResult(owner, result_index, mode)) orelse
                (try self.tryExtractFiniteScfWhileResult(owner, result_index, mode));
            if (exact) |ok_expr| {
                return .{ .is_err = self.encodeBoolConstant(false), .ok_expr = ok_expr };
            }
        }

        if (self.operationNameEq(owner, "ora.try_stmt")) {
            const catch_region = mlir.oraOperationGetRegion(owner, 1);
            if (!self.regionMayEnterCatch(catch_region)) {
                return .{
                    .is_err = self.encodeBoolConstant(false),
                    .ok_expr = try self.encodeValueWithMode(value, mode),
                };
            }
        }

        if (self.operationNameEq(owner, "ora.switch_expr") or self.operationNameEq(owner, "ora.switch")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            if (mlir.oraOperationGetNumOperands(owner) < 1) return null;

            const scrutinee_value = mlir.oraOperationGetOperand(owner, 0);
            const scrutinee = try self.encodeValueWithMode(scrutinee_value, mode);
            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(owner));
            var metadata = try self.getSwitchCaseMetadata(owner, num_regions);
            defer metadata.deinit(self.allocator);

            var branch_ok_exprs = try self.allocator.alloc(z3.Z3_ast, num_regions);
            defer self.allocator.free(branch_ok_exprs);
            var branch_err_preds = try self.allocator.alloc(z3.Z3_ast, num_regions);
            defer self.allocator.free(branch_err_preds);
            var branch_predicates = try self.allocator.alloc(z3.Z3_ast, num_regions);
            defer self.allocator.free(branch_predicates);

            var default_summary: ?DirectTryUnwrapInfo = null;
            if (metadata.default_case_index) |default_case_index| {
                if (default_case_index >= 0) {
                    const default_value = (try self.extractRegionYieldValue(owner, @intCast(default_case_index), result_index)) orelse return null;
                    default_summary = (try self.trySummarizeTryValue(default_value, mode)) orelse return null;
                }
            }

            var remaining = self.encodeBoolConstant(true);
            var region_index = num_regions;
            while (region_index > 0) {
                region_index -= 1;

                const branch_summary = blk: {
                    if (try self.extractRegionYieldValue(owner, @intCast(region_index), result_index)) |branch_value| {
                        if (try self.trySummarizeTryValue(branch_value, mode)) |summary| break :blk summary;
                    }
                    if (default_summary) |summary| break :blk summary;
                    return null;
                };

                const raw_predicate = try self.buildOraSwitchCasePredicate(
                    scrutinee,
                    metadata.case_kinds,
                    metadata.case_values,
                    metadata.range_starts,
                    metadata.range_ends,
                    region_index,
                );
                const effective_predicate = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                    remaining
                else
                    self.encodeAnd(&.{ remaining, raw_predicate });

                branch_ok_exprs[region_index] = branch_summary.ok_expr;
                branch_err_preds[region_index] = branch_summary.is_err;
                branch_predicates[region_index] = effective_predicate;

                if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                    remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
                }
            }

            var catch_pred = self.encodeBoolConstant(false);
            var merged_ok_expr = blk: {
                if (default_summary) |summary| break :blk summary.ok_expr;
                if (metadata.default_case_index == null and self.switchCasesCoverI1Domain(scrutinee_value, metadata)) {
                    var idx: usize = 0;
                    while (idx < num_regions) : (idx += 1) {
                        if (!self.astEquivalent(branch_predicates[idx], self.encodeBoolConstant(false))) {
                            break :blk branch_ok_exprs[idx];
                        }
                    }
                }
                return null;
            };

            region_index = num_regions;
            while (region_index > 0) {
                region_index -= 1;
                catch_pred = self.encodeOr(&.{ catch_pred, self.encodeAnd(&.{ branch_predicates[region_index], branch_err_preds[region_index] }) });
                merged_ok_expr = self.encodeIte(branch_predicates[region_index], branch_ok_exprs[region_index], merged_ok_expr);
            }

            return .{ .is_err = catch_pred, .ok_expr = merged_ok_expr };
        }

        return .{
            .is_err = self.encodeBoolConstant(false),
            .ok_expr = try self.encodeValueWithMode(value, mode),
        };
    }

    fn tryExtractCatchPredicateFromBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        if (mlir.oraBlockIsNull(block)) return continuation;
        return try self.tryExtractCatchPredicateFromSequence(
            mlir.oraBlockGetFirstOperation(block),
            mode,
            continuation,
        );
    }

    fn tryExtractCatchPredicateFromSequence(
        self: *Encoder,
        start_op: mlir.MlirOperation,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        if (mlir.oraOperationIsNull(start_op)) return continuation;

        const name_ref = self.getOperationName(start_op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        const next = mlir.oraOperationGetNextInBlock(start_op);

        if (std.mem.eql(u8, name, "ora.yield") or std.mem.eql(u8, name, "scf.yield")) {
            return continuation;
        }

        if (std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return")) {
            return self.encodeBoolConstant(false);
        }

        if (std.mem.eql(u8, name, "ora.error.unwrap")) {
            if (mlir.oraOperationGetNumOperands(start_op) < 1) return null;
            const operand = mlir.oraOperationGetOperand(start_op, 0);
            const operand_type = mlir.oraValueGetType(operand);
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
            if (mlir.oraTypeIsNull(success_type)) return null;

            const operand_expr = try self.encodeValueWithMode(operand, mode);
            const eu = try self.getErrorUnionSort(operand_type, success_type);
            const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{operand_expr});
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            return self.encodeOr(&.{ is_err, self.encodeAnd(&.{ self.encodeNot(is_err), rest }) });
        }

        if (std.mem.eql(u8, name, "scf.if")) {
            if (mlir.oraOperationGetNumOperands(start_op) < 1) return null;
            const condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(start_op, 0), mode);
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const then_pred = (try self.tryExtractCatchPredicateFromBlock(mlir.oraScfIfOpGetThenBlock(start_op), mode, rest)) orelse return null;
            const else_pred = (try self.tryExtractCatchPredicateFromBlock(mlir.oraScfIfOpGetElseBlock(start_op), mode, rest)) orelse return null;
            return self.encodeOr(&.{
                self.encodeAnd(&.{ condition, then_pred }),
                self.encodeAnd(&.{ self.encodeNot(condition), else_pred }),
            });
        }

        if (std.mem.eql(u8, name, "ora.conditional_return")) {
            if (mlir.oraOperationGetNumOperands(start_op) < 1) return null;
            const condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(start_op, 0), mode);
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const then_pred = (try self.tryExtractCatchPredicateFromBlock(mlir.oraConditionalReturnOpGetThenBlock(start_op), mode, rest)) orelse return null;
            const else_pred = (try self.tryExtractCatchPredicateFromBlock(mlir.oraConditionalReturnOpGetElseBlock(start_op), mode, rest)) orelse return null;
            return self.encodeOr(&.{
                self.encodeAnd(&.{ condition, then_pred }),
                self.encodeAnd(&.{ self.encodeNot(condition), else_pred }),
            });
        }

        if (std.mem.eql(u8, name, "ora.switch")) {
            if (mlir.oraOperationGetNumOperands(start_op) < 1) return null;
            const scrutinee = try self.encodeValueWithMode(mlir.oraOperationGetOperand(start_op, 0), mode);
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(start_op));
            if (num_regions == 0) return rest;

            var metadata = try self.getSwitchCaseMetadata(start_op, num_regions);
            defer metadata.deinit(self.allocator);

            var remaining = self.encodeBoolConstant(true);
            var combined = self.encodeBoolConstant(false);
            for (0..num_regions) |region_index| {
                const raw_predicate = try self.buildOraSwitchCasePredicate(
                    scrutinee,
                    metadata.case_kinds,
                    metadata.case_values,
                    metadata.range_starts,
                    metadata.range_ends,
                    region_index,
                );
                const branch_condition = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                    remaining
                else
                    self.encodeAnd(&.{ remaining, raw_predicate });
                const region = mlir.oraOperationGetRegion(start_op, @intCast(region_index));
                const branch_pred = (try self.tryExtractCatchPredicateFromBlock(mlir.oraRegionGetFirstBlock(region), mode, rest)) orelse return null;
                combined = self.encodeOr(&.{ combined, self.encodeAnd(&.{ branch_condition, branch_pred }) });
                if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                    remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
                }
            }
            return combined;
        }

        if (std.mem.eql(u8, name, "ora.switch_expr")) {
            if (mlir.oraOperationGetNumOperands(start_op) < 1) return null;
            if (mlir.oraOperationGetNumResults(start_op) < 1) return null;

            const scrutinee_value = mlir.oraOperationGetOperand(start_op, 0);
            const scrutinee = try self.encodeValueWithMode(scrutinee_value, mode);
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(start_op));
            if (num_regions == 0) return rest;

            var metadata = try self.getSwitchCaseMetadata(start_op, num_regions);
            defer metadata.deinit(self.allocator);

            var remaining = self.encodeBoolConstant(true);
            var combined = self.encodeBoolConstant(false);
            for (0..num_regions) |region_index| {
                const raw_predicate = try self.buildOraSwitchCasePredicate(
                    scrutinee,
                    metadata.case_kinds,
                    metadata.case_values,
                    metadata.range_starts,
                    metadata.range_ends,
                    region_index,
                );
                const branch_condition = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                    remaining
                else
                    self.encodeAnd(&.{ remaining, raw_predicate });

                const branch_value = (try self.extractRegionYieldValue(start_op, @intCast(region_index), 0)) orelse return null;
                const branch_summary = (try self.trySummarizeTryValue(branch_value, mode)) orelse return null;
                const branch_pred = self.encodeOr(&.{
                    branch_summary.is_err,
                    self.encodeAnd(&.{ self.encodeNot(branch_summary.is_err), rest }),
                });
                combined = self.encodeOr(&.{ combined, self.encodeAnd(&.{ branch_condition, branch_pred }) });

                if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                    remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
                }
            }
            return combined;
        }

        if (std.mem.eql(u8, name, "scf.execute_region")) {
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const region = mlir.oraOperationGetRegion(start_op, 0);
            if (mlir.oraRegionIsNull(region)) return null;
            const block = mlir.oraRegionGetFirstBlock(region);
            return try self.tryExtractCatchPredicateFromBlock(block, mode, rest);
        }

        if (std.mem.eql(u8, name, "ora.sstore") or
            std.mem.eql(u8, name, "ora.map_store") or
            std.mem.eql(u8, name, "ora.tstore") or
            std.mem.eql(u8, name, "memref.store"))
        {
            return try self.tryExtractCatchPredicateFromSequence(next, mode, continuation);
        }

        if (std.mem.eql(u8, name, "scf.for")) {
            if (try self.tryExtractFiniteScfForCatchPredicate(start_op, mode, continuation)) |pred| {
                return pred;
            }
            if (try self.tryExtractSingleIterationScfForCatchPredicate(start_op, mode, continuation)) |pred| {
                return pred;
            }
            return null;
        }

        if (std.mem.eql(u8, name, "scf.while")) {
            if (try self.tryExtractFiniteScfWhileCatchPredicate(start_op, mode, continuation)) |pred| {
                return pred;
            }
            if (try self.tryExtractSingleIterationScfWhileCatchPredicate(start_op, mode, continuation)) |pred| {
                return pred;
            }
            return null;
        }

        if (std.mem.eql(u8, name, "ora.try_stmt")) {
            const rest = (try self.tryExtractCatchPredicateFromSequence(next, mode, continuation)) orelse return null;
            const catch_region = mlir.oraOperationGetRegion(start_op, 1);
            if (!self.regionMayEnterCatch(catch_region)) {
                return rest;
            }
            const catch_block = mlir.oraRegionGetFirstBlock(catch_region);
            const catch_pred = (try self.tryExtractCatchPredicateFromBlock(catch_block, mode, rest)) orelse return null;
            if (try self.tryStmtAlwaysEntersCatch(start_op, mode)) {
                return catch_pred;
            }
            if (!self.tryStmtMayEnterCatch(start_op)) {
                return rest;
            }
            if (try self.tryExtractTryRegionCatchPredicate(start_op, mode)) |try_catch_pred| {
                return self.encodeOr(&.{
                    self.encodeAnd(&.{ try_catch_pred, catch_pred }),
                    self.encodeAnd(&.{ self.encodeNot(try_catch_pred), rest }),
                });
            }
            return null;
        }

        return try self.tryExtractCatchPredicateFromSequence(next, mode, continuation);
    }

    fn tryExtractFiniteScfForCatchPredicate(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const loop_ctx = self.getFiniteScfForContext(for_op) orelse return null;
        const carried = try self.getScfForInitCarriedValues(for_op, mode);
        defer self.allocator.free(carried);
        return try self.extractFiniteScfForCatchPredicate(loop_ctx, 0, carried, mode, continuation);
    }

    fn extractFiniteScfForCatchPredicate(
        self: *Encoder,
        loop_ctx: FiniteScfForContext,
        iter_index: usize,
        carried: []const z3.Z3_ast,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        if (iter_index >= loop_ctx.trip_count) return continuation;

        const iv_value = loop_ctx.lower_bound + iter_index * loop_ctx.step;
        try self.bindFiniteScfForLoopArgs(loop_ctx, iv_value, carried);
        self.invalidateBlockValueCaches(loop_ctx.body);
        const yielded = (try self.extractScfForYieldValues(loop_ctx, mode)) orelse {
            self.unbindFiniteScfForLoopArgs(loop_ctx);
            return null;
        };
        self.unbindFiniteScfForLoopArgs(loop_ctx);
        defer self.allocator.free(yielded);

        const rest = (try self.extractFiniteScfForCatchPredicate(loop_ctx, iter_index + 1, yielded, mode, continuation)) orelse return null;

        try self.bindFiniteScfForLoopArgs(loop_ctx, iv_value, carried);
        defer self.unbindFiniteScfForLoopArgs(loop_ctx);
        self.invalidateBlockValueCaches(loop_ctx.body);
        return try self.tryExtractCatchPredicateFromBlock(loop_ctx.body, mode, rest);
    }

    fn tryExtractFiniteScfWhileCatchPredicate(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const carried = try self.getScfWhileInitValues(while_op, mode);
        defer self.allocator.free(carried);
        return try self.extractFiniteScfWhileCatchPredicate(while_op, carried, mode, continuation, 0);
    }

    fn extractFiniteScfWhileCatchPredicate(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        carried: []const z3.Z3_ast,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
        depth: usize,
    ) EncodeError!?z3.Z3_ast {
        if (depth >= finite_scf_while_unroll_limit) return null;

        const condition_op = self.findScfConditionOp(while_op) orelse return null;

        const before_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, carried);
        self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetBeforeBlock(while_op));
        self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetAfterBlock(while_op));
        const condition_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), mode);
        const condition_const = self.astSimplifiesToBool(condition_ast) orelse {
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        };
        if (!condition_const) {
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return continuation;
        }

        const after_bind_count = try self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, mode);
        const yielded = (try self.extractScfWhileYieldValues(while_op, mode)) orelse {
            self.unbindScfWhileAfterArgs(while_op, after_bind_count);
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        };
        self.unbindScfWhileAfterArgs(while_op, after_bind_count);
        self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
        defer self.allocator.free(yielded);

        const rest = (try self.extractFiniteScfWhileCatchPredicate(while_op, yielded, mode, continuation, depth + 1)) orelse return null;

        const replay_before_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, carried);
        defer self.unbindScfWhileBeforeArgs(while_op, replay_before_bind_count);
        self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetBeforeBlock(while_op));
        self.invalidateBlockValueCaches(mlir.oraScfWhileOpGetAfterBlock(while_op));
        const replay_after_bind_count = try self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, mode);
        defer self.unbindScfWhileAfterArgs(while_op, replay_after_bind_count);
        return try self.tryExtractCatchPredicateFromBlock(mlir.oraScfWhileOpGetAfterBlock(while_op), mode, rest);
    }

    fn tryExtractSingleIterationScfForCatchPredicate(
        self: *Encoder,
        for_op: mlir.MlirOperation,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(for_op));
        if (num_operands < 3) return null;

        const lb_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 0)) orelse return null;
        const ub_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 1)) orelse return null;
        const step_const = self.tryGetConstIntValue(mlir.oraOperationGetOperand(for_op, 2)) orelse return null;
        if (step_const == 0 or lb_const >= ub_const) return null;
        if (step_const < ub_const - lb_const) return null;

        const body = mlir.oraScfForOpGetBodyBlock(for_op);
        if (mlir.oraBlockIsNull(body)) return null;

        const body_arg_count: usize = @intCast(mlir.oraBlockGetNumArguments(body));
        if (body_arg_count == 0) return null;

        const iv_ast = try self.encodeValueWithMode(mlir.oraOperationGetOperand(for_op, 0), mode);
        try self.bindValue(mlir.oraBlockGetArgument(body, 0), iv_ast);
        defer self.unbindValue(mlir.oraBlockGetArgument(body, 0));

        const iter_bind_count = @min(body_arg_count - 1, num_operands - 3);
        for (0..iter_bind_count) |idx| {
            const iter_value = mlir.oraOperationGetOperand(for_op, @intCast(idx + 3));
            const iter_ast = try self.encodeValueWithMode(iter_value, mode);
            try self.bindValue(mlir.oraBlockGetArgument(body, @intCast(idx + 1)), iter_ast);
        }
        defer {
            for (0..iter_bind_count) |idx| {
                self.unbindValue(mlir.oraBlockGetArgument(body, @intCast(idx + 1)));
            }
        }

        self.invalidateBlockValueCaches(body);
        return try self.tryExtractCatchPredicateFromBlock(body, mode, continuation);
    }

    fn tryExtractSingleIterationScfWhileCatchPredicate(
        self: *Encoder,
        while_op: mlir.MlirOperation,
        mode: EncodeMode,
        continuation: z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const before_bind_count = try self.bindScfWhileBeforeArgs(while_op, mode);

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
        const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
        if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) {
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        }

        self.invalidateBlockValueCaches(before_block);
        self.invalidateBlockValueCaches(after_block);

        const condition_op = self.findScfConditionOp(while_op) orelse return null;
        if (mlir.oraOperationGetNumOperands(condition_op) < 1) {
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        }
        const initial_condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), mode);
        if (!self.isBoolConst(initial_condition, true)) {
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        }

        const after_bind_count = try self.bindScfWhileAfterArgsFromCondition(while_op, condition_op, mode);
        const body_pred = (try self.tryExtractCatchPredicateFromBlock(after_block, mode, continuation)) orelse {
            self.unbindScfWhileAfterArgs(while_op, after_bind_count);
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        };
        const yielded = (try self.extractScfWhileYieldValues(while_op, mode)) orelse {
            self.unbindScfWhileAfterArgs(while_op, after_bind_count);
            self.unbindScfWhileBeforeArgs(while_op, before_bind_count);
            return null;
        };
        defer self.allocator.free(yielded);

        self.unbindScfWhileAfterArgs(while_op, after_bind_count);
        self.unbindScfWhileBeforeArgs(while_op, before_bind_count);

        const second_bind_count = try self.bindScfWhileBeforeArgsFromValues(while_op, yielded);
        defer self.unbindScfWhileBeforeArgs(while_op, second_bind_count);

        self.invalidateBlockValueCaches(before_block);
        self.invalidateBlockValueCaches(after_block);

        const next_condition = try self.encodeValueWithMode(mlir.oraOperationGetOperand(condition_op, 0), mode);
        if (!self.isBoolConst(next_condition, false)) return null;

        return body_pred;
    }

    fn tryExtractTryRegionCatchPredicate(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const try_region = mlir.oraOperationGetRegion(try_stmt, 0);
        if (mlir.oraRegionIsNull(try_region)) return null;
        const try_block = mlir.oraRegionGetFirstBlock(try_region);
        return try self.tryExtractCatchPredicateFromBlock(try_block, mode, self.encodeBoolConstant(false));
    }

    fn tryGetDirectErrorUnwrapPredicateFromBlock(
        self: *Encoder,
        block: mlir.MlirBlock,
        mode: EncodeMode,
    ) EncodeError!?DirectTryUnwrapInfo {
        if (mlir.oraBlockIsNull(block)) return null;

        var unwrap_op: ?mlir.MlirOperation = null;
        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.yield") or std.mem.eql(u8, name, "scf.yield")) {
                if (unwrap_op == null) return null;

                const unwrap_operand = mlir.oraOperationGetOperand(unwrap_op.?, 0);
                const operand_type = mlir.oraValueGetType(unwrap_operand);
                const success_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
                if (mlir.oraTypeIsNull(success_type)) return null;

                const operand_expr = try self.encodeValueWithMode(unwrap_operand, mode);
                const eu = try self.getErrorUnionSort(operand_type, success_type);
                const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{operand_expr});
                const ok_expr = z3.Z3_mk_app(self.context.ctx, eu.proj_ok, 1, &[_]z3.Z3_ast{operand_expr});
                return .{ .is_err = is_err, .ok_expr = ok_expr };
            }

            if (std.mem.eql(u8, name, "ora.error.unwrap")) {
                if (unwrap_op != null) return null;
                unwrap_op = current;
                current = mlir.oraOperationGetNextInBlock(current);
                continue;
            }

            if (std.mem.eql(u8, name, "scf.if") or
                std.mem.eql(u8, name, "ora.switch") or
                std.mem.eql(u8, name, "ora.conditional_return") or
                std.mem.eql(u8, name, "scf.execute_region") or
                std.mem.eql(u8, name, "scf.for") or
                std.mem.eql(u8, name, "scf.while") or
                std.mem.eql(u8, name, "ora.try_stmt") or
                std.mem.eql(u8, name, "ora.return") or
                std.mem.eql(u8, name, "func.return"))
            {
                return null;
            }

            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn tryGetDirectErrorUnwrapBlockInfo(
        self: *Encoder,
        block: mlir.MlirBlock,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?DirectTryUnwrapInfo {
        if (result_index != 0 or mlir.oraBlockIsNull(block)) return null;

        var unwrap_op: ?mlir.MlirOperation = null;
        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = self.getOperationName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, name, "ora.yield") or std.mem.eql(u8, name, "scf.yield")) {
                if (unwrap_op == null) return null;
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (num_operands <= result_index) return null;
                const yielded = mlir.oraOperationGetOperand(current, result_index);
                const unwrap_result = mlir.oraOperationGetResult(unwrap_op.?, 0);
                if (yielded.ptr != unwrap_result.ptr) return null;

                const unwrap_operand = mlir.oraOperationGetOperand(unwrap_op.?, 0);
                const operand_type = mlir.oraValueGetType(unwrap_operand);
                const success_type = mlir.oraErrorUnionTypeGetSuccessType(operand_type);
                if (mlir.oraTypeIsNull(success_type)) return null;

                const operand_expr = try self.encodeValueWithMode(unwrap_operand, mode);
                const eu = try self.getErrorUnionSort(operand_type, success_type);
                const is_err = z3.Z3_mk_app(self.context.ctx, eu.proj_is_error, 1, &[_]z3.Z3_ast{operand_expr});
                const ok_expr = z3.Z3_mk_app(self.context.ctx, eu.proj_ok, 1, &[_]z3.Z3_ast{operand_expr});
                return .{ .is_err = is_err, .ok_expr = ok_expr };
            }

            if (std.mem.eql(u8, name, "ora.error.unwrap")) {
                if (unwrap_op != null) return null;
                unwrap_op = current;
                current = mlir.oraOperationGetNextInBlock(current);
                continue;
            }

            if (std.mem.eql(u8, name, "scf.if") or
                std.mem.eql(u8, name, "ora.switch") or
                std.mem.eql(u8, name, "ora.conditional_return") or
                std.mem.eql(u8, name, "scf.execute_region") or
                std.mem.eql(u8, name, "scf.for") or
                std.mem.eql(u8, name, "scf.while") or
                std.mem.eql(u8, name, "ora.try_stmt") or
                std.mem.eql(u8, name, "ora.return") or
                std.mem.eql(u8, name, "func.return"))
            {
                return null;
            }

            current = mlir.oraOperationGetNextInBlock(current);
        }

        return null;
    }

    fn tryGetDirectErrorUnwrapTryInfo(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?DirectTryUnwrapInfo {
        const try_region = mlir.oraOperationGetRegion(try_stmt, 0);
        if (mlir.oraRegionIsNull(try_region)) return null;
        return try self.tryGetDirectErrorUnwrapBlockInfo(mlir.oraRegionGetFirstBlock(try_region), result_index, mode);
    }

    fn tryGetDirectErrorUnwrapTryPredicate(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        mode: EncodeMode,
    ) EncodeError!?DirectTryUnwrapInfo {
        const try_region = mlir.oraOperationGetRegion(try_stmt, 0);
        if (mlir.oraRegionIsNull(try_region)) return null;
        return try self.tryGetDirectErrorUnwrapPredicateFromBlock(mlir.oraRegionGetFirstBlock(try_region), mode);
    }

    fn tryExtractDirectErrorUnwrapTryStmtResult(
        self: *Encoder,
        try_stmt: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
    ) EncodeError!?z3.Z3_ast {
        const catch_expr = (try self.extractRegionYield(try_stmt, 1, result_index, mode)) orelse return null;
        const info = (try self.tryGetDirectErrorUnwrapTryInfo(try_stmt, result_index, mode)) orelse return null;
        return try self.encodeControlFlow("scf.if", info.is_err, catch_expr, info.ok_expr);
    }

    fn extractRegionYieldValue(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        region_index: u32,
        result_index: u32,
    ) EncodeError!?mlir.MlirValue {
        _ = self;
        const region = mlir.oraOperationGetRegion(mlir_op, region_index);
        if (mlir.oraRegionIsNull(region)) return null;
        const block = mlir.oraRegionGetFirstBlock(region);
        if (mlir.oraBlockIsNull(block)) return null;

        var current = mlir.oraBlockGetFirstOperation(block);
        while (!mlir.oraOperationIsNull(current)) {
            const name_ref = mlir.oraOperationGetName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, name, "scf.yield") or std.mem.eql(u8, name, "ora.yield")) {
                const num_operands: u32 = @intCast(mlir.oraOperationGetNumOperands(current));
                if (num_operands > result_index) {
                    return mlir.oraOperationGetOperand(current, result_index);
                }
                return null;
            }
            current = mlir.oraOperationGetNextInBlock(current);
        }
        return null;
    }

    fn extractScfIfReturnedExpr(
        self: *Encoder,
        if_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
        fallthrough_expr: ?z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(if_op);
        if (num_operands < 1) return null;
        const condition_value = mlir.oraOperationGetOperand(if_op, 0);
        const condition = try self.encodeValueWithMode(condition_value, mode);

        const saved_len = self.return_path_assumptions.items.len;
        defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);

        try self.return_path_assumptions.append(self.allocator, condition);
        var then_expr = try self.extractReturnedExprFromBlock(mlir.oraScfIfOpGetThenBlock(if_op), result_index, mode);

        self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
        try self.return_path_assumptions.append(self.allocator, z3.Z3_mk_not(self.context.ctx, self.coerceToBool(condition)));
        var else_expr = try self.extractReturnedExprFromBlock(mlir.oraScfIfOpGetElseBlock(if_op), result_index, mode);

        if (then_expr == null) then_expr = fallthrough_expr;
        if (else_expr == null) else_expr = fallthrough_expr;
        if (then_expr == null or else_expr == null) return null;
        return try self.encodeControlFlow("scf.if", condition, then_expr, else_expr);
    }

    fn extractConditionalReturnExpr(
        self: *Encoder,
        conditional_ret_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
        fallthrough_expr: ?z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(conditional_ret_op);
        if (num_operands < 1) return null;

        const condition_value = mlir.oraOperationGetOperand(conditional_ret_op, 0);
        const condition = try self.encodeValueWithMode(condition_value, mode);
        const saved_len = self.return_path_assumptions.items.len;
        defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);

        try self.return_path_assumptions.append(self.allocator, condition);
        var then_expr = try self.extractReturnedExprFromBlock(mlir.oraConditionalReturnOpGetThenBlock(conditional_ret_op), result_index, mode);

        self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
        try self.return_path_assumptions.append(self.allocator, z3.Z3_mk_not(self.context.ctx, self.coerceToBool(condition)));
        var else_expr = try self.extractReturnedExprFromBlock(mlir.oraConditionalReturnOpGetElseBlock(conditional_ret_op), result_index, mode);

        if (then_expr == null) then_expr = fallthrough_expr;
        if (else_expr == null) else_expr = fallthrough_expr;
        if (then_expr == null or else_expr == null) return null;
        return try self.encodeControlFlow("scf.if", condition, then_expr, else_expr);
    }

    fn extractSwitchReturnedExpr(
        self: *Encoder,
        switch_op: mlir.MlirOperation,
        result_index: u32,
        mode: EncodeMode,
        fallthrough_expr: ?z3.Z3_ast,
    ) EncodeError!?z3.Z3_ast {
        const num_operands = mlir.oraOperationGetNumOperands(switch_op);
        if (num_operands < 1) return null;

        const scrutinee_value = mlir.oraOperationGetOperand(switch_op, 0);
        const scrutinee = try self.encodeValueWithMode(scrutinee_value, mode);
        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(switch_op));
        if (num_regions == 0) return fallthrough_expr;

        var metadata = try self.getSwitchCaseMetadata(switch_op, num_regions);
        defer metadata.deinit(self.allocator);

        const saved_len = self.return_path_assumptions.items.len;
        defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);

        var branch_conditions = try self.allocator.alloc(z3.Z3_ast, num_regions);
        defer self.allocator.free(branch_conditions);
        var branch_exprs = try self.allocator.alloc(?z3.Z3_ast, num_regions);
        defer self.allocator.free(branch_exprs);

        var remaining = self.encodeBoolConstant(true);
        for (0..num_regions) |region_index| {
            const raw_predicate = try self.buildOraSwitchCasePredicate(
                scrutinee,
                metadata.case_kinds,
                metadata.case_values,
                metadata.range_starts,
                metadata.range_ends,
                region_index,
            );
            const effective_predicate = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                remaining
            else
                self.encodeAnd(&.{ remaining, raw_predicate });
            branch_conditions[region_index] = effective_predicate;

            self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
            try self.return_path_assumptions.append(self.allocator, effective_predicate);
            const region = mlir.oraOperationGetRegion(switch_op, @intCast(region_index));
            const block = mlir.oraRegionGetFirstBlock(region);
            branch_exprs[region_index] = try self.extractReturnedExprFromBlock(block, result_index, mode);

            if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
            }
        }

        var combined = fallthrough_expr;
        var index = num_regions;
        while (index > 0) {
            index -= 1;
            const branch_expr = branch_exprs[index] orelse fallthrough_expr;
            if (branch_expr == null) return null;
            combined = if (combined) |else_expr|
                try self.encodeControlFlow("scf.if", branch_conditions[index], branch_expr.?, else_expr)
            else
                branch_expr;
        }

        return combined;
    }

    fn tryInlineFunctionCallSummary(
        self: *Encoder,
        call_op: mlir.MlirOperation,
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
        try summary_encoder.copyReturnPathAssumptionsFrom(self);
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

        var callee_requires = std.ArrayList(z3.Z3_ast){};
        defer callee_requires.deinit(self.allocator);
        try summary_encoder.collectRequiresForSummary(func_op, &callee_requires);

        var any_result = false;
        for (0..result_exprs.len) |i| {
            const encoded = try summary_encoder.extractFunctionReturnExpr(func_op, @intCast(i), .Current);
            if (encoded) |expr| {
                result_exprs[i] = expr;
                any_result = true;
            }
        }

        // Result-only summaries can rely on exact returned-path replay for
        // obligations/local effects. Full state-summary materialization is
        // still required when the caller tracks state slots or when the callee
        // has no results.
        if (slots.len > 0 or result_exprs.len == 0) {
            // Materialize stateful effects from the callee body so summary state
            // reflects sstore/map_store updates before we snapshot post-state.
            summary_encoder.encodeStateEffectsInOperation(func_op);
        }

        if (summary_encoder.isDegraded()) {
            const reason = summary_encoder.degradationReason() orelse "summary encoder degraded while materializing call summary";
            if (result_exprs.len > 0) {
                self.recordCalleeResultDegradation(call_op, callee, reason);
            } else {
                self.recordDegradation(reason);
            }
        }

        const extra_constraints = try summary_encoder.takeConstraints(self.allocator);
        defer if (extra_constraints.len > 0) self.allocator.free(extra_constraints);
        for (extra_constraints) |cst| self.addConstraint(cst);

        const extra_obligations = try summary_encoder.takeObligations(self.allocator);
        defer if (extra_obligations.len > 0) self.allocator.free(extra_obligations);
        if (!self.functionIsExternallyVerified(func_op)) {
            self.addSummaryObligations(extra_obligations, callee_requires.items, summary_encoder.return_path_assumptions.items);
        }

        for (slots) |*slot| {
            if (summary_encoder.global_map.get(slot.name)) |post| {
                if (!slot.is_write or summary_encoder.hasWrittenGlobalSlot(slot.name)) {
                    slot.post = post;
                } else if (!summary_encoder.isDegraded()) {
                    slot.post = slot.pre;
                }
            } else if (slot.is_write and !summary_encoder.isDegraded() and !summary_encoder.hasWrittenGlobalSlot(slot.name)) {
                slot.post = slot.pre;
            }
        }

        return any_result or slots.len > 0;
    }

    fn collectRequiresForSummary(
        self: *Encoder,
        op: mlir.MlirOperation,
        out: *std.ArrayList(z3.Z3_ast),
    ) EncodeError!void {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "cf.assert")) {
                const requires_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("ora.requires", 12));
                if (!mlir.oraAttributeIsNull(requires_attr) and mlir.oraOperationGetNumOperands(op) >= 1) {
                    const condition_value = mlir.oraOperationGetOperand(op, 0);
                    const encoded = self.encodeValue(condition_value) catch blk: {
                        self.recordDegradation("failed to encode summary precondition");
                        break :blk null;
                    };
                    if (encoded) |cond| {
                        try out.append(self.allocator, self.coerceToBool(cond));
                    }
                }
            } else if (std.mem.eql(u8, op_name, "ora.requires")) {
                if (mlir.oraOperationGetNumOperands(op) >= 1) {
                    const condition_value = mlir.oraOperationGetOperand(op, 0);
                    const encoded = self.encodeValue(condition_value) catch blk: {
                        self.recordDegradation("failed to encode summary precondition");
                        break :blk null;
                    };
                    if (encoded) |cond| {
                        try out.append(self.allocator, self.coerceToBool(cond));
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
                    try self.collectRequiresForSummary(nested, out);
                    nested = mlir.oraOperationGetNextInBlock(nested);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    fn functionIsExternallyVerified(_: *Encoder, func_op: mlir.MlirOperation) bool {
        const visibility_attr = mlir.oraOperationGetAttributeByName(func_op, mlir.oraStringRefCreate("ora.visibility", 14));
        if (mlir.oraAttributeIsNull(visibility_attr)) return true;
        const visibility_ref = mlir.oraStringAttrGetValue(visibility_attr);
        if (visibility_ref.data == null or visibility_ref.length == 0) return true;
        const visibility = visibility_ref.data[0..visibility_ref.length];
        return std.mem.eql(u8, visibility, "pub") or
            std.mem.eql(u8, visibility, "public") or
            std.mem.eql(u8, visibility, "external");
    }

    fn addSummaryObligations(
        self: *Encoder,
        obligations: []const z3.Z3_ast,
        requires: []const z3.Z3_ast,
        path_guards: []const z3.Z3_ast,
    ) void {
        if (obligations.len == 0) return;
        if (requires.len == 0 and path_guards.len == 0) {
            for (obligations) |obl| self.addObligation(obl);
            return;
        }

        var guard_terms = std.ArrayList(z3.Z3_ast){};
        defer guard_terms.deinit(self.allocator);
        guard_terms.appendSlice(self.allocator, requires) catch {
            self.recordDegradation("failed to allocate summary obligation guards");
            for (obligations) |obl| self.addObligation(obl);
            return;
        };
        guard_terms.appendSlice(self.allocator, path_guards) catch {
            self.recordDegradation("failed to allocate summary obligation guards");
            for (obligations) |obl| self.addObligation(obl);
            return;
        };

        const precondition_guard = self.encodeAnd(guard_terms.items);
        for (obligations) |obl| {
            self.addObligation(self.encodeImplies(precondition_guard, obl));
        }
    }

    fn encodeStateEffectsInOperation(self: *Encoder, op: mlir.MlirOperation) void {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and name_ref.length > 0) {
            const op_name = name_ref.data[0..name_ref.length];
            if (std.mem.eql(u8, op_name, "func.func")) {
                self.encodeStateEffectsInRegions(op);
                return;
            }
            if (std.mem.eql(u8, op_name, "scf.if")) {
                if (mlir.oraOperationGetNumOperands(op) < 1) {
                    self.recordDegradation("scf.if missing condition while encoding state effects");
                    return;
                }

                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const condition = self.encodeValue(condition_value) catch {
                    self.recordDegradation("failed to encode scf.if condition for state summary");
                    return;
                };

                var base_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture base state for scf.if summary");
                    return;
                };
                defer base_state.deinit(self.allocator);

                var then_state = StateSnapshot.init(self.allocator);
                defer then_state.deinit(self.allocator);
                var else_state = StateSnapshot.init(self.allocator);
                defer else_state.deinit(self.allocator);

                self.restoreStateSnapshot(&base_state) catch {
                    self.recordDegradation("failed to restore base state for scf.if then-branch");
                    return;
                };
                const then_region = mlir.oraOperationGetRegion(op, 0);
                if (!mlir.oraRegionIsNull(then_region)) {
                    const saved_len = self.return_path_assumptions.items.len;
                    defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
                    self.return_path_assumptions.append(self.allocator, self.coerceToBool(condition)) catch {
                        self.recordDegradation("failed to record scf.if then-branch path assumption");
                        return;
                    };
                    self.encodeStateEffectsInRegion(then_region);
                }
                then_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture scf.if then-branch state summary");
                    return;
                };

                self.restoreStateSnapshot(&base_state) catch {
                    self.recordDegradation("failed to restore base state for scf.if else-branch");
                    return;
                };
                const else_region = mlir.oraOperationGetRegion(op, 1);
                if (!mlir.oraRegionIsNull(else_region)) {
                    const saved_len = self.return_path_assumptions.items.len;
                    defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
                    self.return_path_assumptions.append(self.allocator, z3.Z3_mk_not(self.context.ctx, self.coerceToBool(condition))) catch {
                        self.recordDegradation("failed to record scf.if else-branch path assumption");
                        return;
                    };
                    self.encodeStateEffectsInRegion(else_region);
                }
                else_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture scf.if else-branch state summary");
                    return;
                };

                self.mergeStateSnapshotsIf(condition, &base_state, &then_state, &else_state) catch {
                    self.recordDegradation("failed to merge scf.if branch state summary");
                    return;
                };
                return;
            }

            if (std.mem.eql(u8, op_name, "scf.execute_region")) {
                self.encodeStateEffectsInRegions(op);
                return;
            }

            if (std.mem.eql(u8, op_name, "scf.for")) {
                const loop_writes_state = self.operationMayWriteTrackedState(op) catch {
                    self.recordDegradation("failed to recover scf.for write set for state summary");
                    return;
                };
                if (!loop_writes_state) return;
                if (self.isZeroIterationScfFor(op)) return;
                if (self.tryEncodeFiniteScfForStateEffects(op)) return;
                self.recordDegradation("loop state summary is not encoded exactly");
                return;
            }

            if (std.mem.eql(u8, op_name, "scf.while")) {
                const loop_writes_state = self.operationMayWriteTrackedState(op) catch {
                    self.recordDegradation("failed to recover scf.while write set for state summary");
                    return;
                };
                if ((self.isStaticallyFalseScfWhile(op, .Current) catch false)) return;
                if (!loop_writes_state) return;
                if (self.tryEncodeFiniteScfWhileStateEffects(op)) return;
                self.recordDegradation("loop state summary is not encoded exactly");
                return;
            }

            if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
                if ((self.tryStmtAlwaysEntersCatch(op, .Current) catch false)) {
                    const catch_region = mlir.oraOperationGetRegion(op, 1);
                    if (!mlir.oraRegionIsNull(catch_region)) {
                        self.encodeStateEffectsInRegion(catch_region);
                    }
                    return;
                }

                if (!self.tryStmtMayEnterCatch(op)) {
                    const try_region = mlir.oraOperationGetRegion(op, 0);
                    if (!mlir.oraRegionIsNull(try_region)) {
                        self.encodeStateEffectsInRegion(try_region);
                    }
                    return;
                }

                if (self.tryGetDirectErrorUnwrapTryPredicate(op, .Current) catch null) |info| {
                    var base_state = self.captureStateSnapshot() catch {
                        self.recordDegradation("failed to capture base state for direct ora.try_stmt summary");
                        return;
                    };
                    defer base_state.deinit(self.allocator);

                    var try_state = StateSnapshot.init(self.allocator);
                    defer try_state.deinit(self.allocator);
                    var catch_state = StateSnapshot.init(self.allocator);
                    defer catch_state.deinit(self.allocator);

                    self.restoreStateSnapshot(&base_state) catch {
                        self.recordDegradation("failed to restore base state for direct ora.try_stmt try-summary");
                        return;
                    };
                    const try_region = mlir.oraOperationGetRegion(op, 0);
                    if (!mlir.oraRegionIsNull(try_region)) {
                        self.encodeStateEffectsInRegion(try_region);
                    }
                    try_state = self.captureStateSnapshot() catch {
                        self.recordDegradation("failed to capture direct ora.try_stmt try-state summary");
                        return;
                    };

                    self.restoreStateSnapshot(&base_state) catch {
                        self.recordDegradation("failed to restore base state for direct ora.try_stmt catch-summary");
                        return;
                    };
                    const catch_region = mlir.oraOperationGetRegion(op, 1);
                    if (!mlir.oraRegionIsNull(catch_region)) {
                        self.encodeStateEffectsInRegion(catch_region);
                    }
                    catch_state = self.captureStateSnapshot() catch {
                        self.recordDegradation("failed to capture direct ora.try_stmt catch-state summary");
                        return;
                    };

                    self.mergeStateSnapshotsIf(info.is_err, &base_state, &catch_state, &try_state) catch {
                        self.recordDegradation("failed to merge direct ora.try_stmt state summary");
                        return;
                    };
                    return;
                }

                var base_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture base state for ora.try_stmt summary");
                    return;
                };
                defer base_state.deinit(self.allocator);

                var try_state = StateSnapshot.init(self.allocator);
                defer try_state.deinit(self.allocator);
                var catch_state = StateSnapshot.init(self.allocator);
                defer catch_state.deinit(self.allocator);

                self.restoreStateSnapshot(&base_state) catch {
                    self.recordDegradation("failed to restore base state for ora.try_stmt try-summary");
                    return;
                };
                const try_region = mlir.oraOperationGetRegion(op, 0);
                if (!mlir.oraRegionIsNull(try_region)) {
                    self.encodeStateEffectsInRegion(try_region);
                }
                try_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture ora.try_stmt try-state summary");
                    return;
                };

                self.restoreStateSnapshot(&base_state) catch {
                    self.recordDegradation("failed to restore base state for ora.try_stmt catch-summary");
                    return;
                };
                const catch_region = mlir.oraOperationGetRegion(op, 1);
                if (!mlir.oraRegionIsNull(catch_region)) {
                    self.encodeStateEffectsInRegion(catch_region);
                }
                catch_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture ora.try_stmt catch-state summary");
                    return;
                };

                const branch_condition = self.tryExtractTryRegionCatchPredicate(op, .Current) catch {
                    self.recordDegradation("failed to extract branch-conditioned ora.try_stmt catch predicate");
                    return;
                };
                if (branch_condition) |catch_pred| {
                    self.mergeStateSnapshotsIf(catch_pred, &base_state, &catch_state, &try_state) catch {
                        self.recordDegradation("failed to merge branch-conditioned ora.try_stmt state summary");
                        return;
                    };
                    return;
                }

                if (self.stateSnapshotsEquivalent(&try_state, &catch_state)) {
                    self.restoreStateSnapshot(&try_state) catch {
                        self.recordDegradation("failed to restore equivalent ora.try_stmt state summary");
                    };
                    return;
                }

                self.recordDegradation("try state summary is not encoded exactly");
                return;
            }

            if (std.mem.eql(u8, op_name, "ora.switch") or std.mem.eql(u8, op_name, "ora.switch_expr")) {
                if (mlir.oraOperationGetNumOperands(op) < 1) {
                    self.recordDegradation("ora.switch missing scrutinee while encoding state effects");
                    return;
                }

                const scrutinee_value = mlir.oraOperationGetOperand(op, 0);
                const scrutinee = self.encodeValue(scrutinee_value) catch {
                    self.recordDegradation("failed to encode ora.switch scrutinee for state summary");
                    return;
                };

                const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
                var metadata = self.getSwitchCaseMetadata(op, num_regions) catch {
                    self.recordDegradation("failed to recover ora.switch case metadata for state summary");
                    return;
                };
                defer metadata.deinit(self.allocator);

                var base_state = self.captureStateSnapshot() catch {
                    self.recordDegradation("failed to capture base state for ora.switch summary");
                    return;
                };
                defer base_state.deinit(self.allocator);

                var branch_states = self.allocator.alloc(StateSnapshot, num_regions) catch {
                    self.recordDegradation("failed to allocate ora.switch branch summary state");
                    return;
                };
                defer self.allocator.free(branch_states);
                for (branch_states) |*branch_state| branch_state.* = StateSnapshot.init(self.allocator);
                defer {
                    for (branch_states) |*branch_state| branch_state.deinit(self.allocator);
                }

                var branch_conditions = self.allocator.alloc(z3.Z3_ast, num_regions) catch {
                    self.recordDegradation("failed to allocate ora.switch branch predicates");
                    return;
                };
                defer self.allocator.free(branch_conditions);

                var remaining = self.encodeBoolConstant(true);
                for (0..num_regions) |region_index| {
                    self.restoreStateSnapshot(&base_state) catch {
                        self.recordDegradation("failed to restore base state for ora.switch branch summary");
                        return;
                    };

                    const raw_predicate = self.buildOraSwitchCasePredicate(
                        scrutinee,
                        metadata.case_kinds,
                        metadata.case_values,
                        metadata.range_starts,
                        metadata.range_ends,
                        region_index,
                    ) catch {
                        self.recordDegradation("failed to encode ora.switch branch predicate for state summary");
                        return;
                    };
                    const effective_predicate = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_index)))
                        remaining
                    else
                        self.encodeAnd(&.{ remaining, raw_predicate });
                    branch_conditions[region_index] = effective_predicate;

                    const region = mlir.oraOperationGetRegion(op, @intCast(region_index));
                    self.encodeStateEffectsInRegion(region);
                    branch_states[region_index] = self.captureStateSnapshot() catch {
                        self.recordDegradation("failed to capture ora.switch branch state summary");
                        return;
                    };

                    if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_index))) {
                        remaining = self.encodeAnd(&.{ remaining, self.encodeNot(raw_predicate) });
                    }
                }

                self.mergeStateSnapshotsMany(branch_conditions, &base_state, branch_states) catch {
                    self.recordDegradation("failed to merge ora.switch branch state summary");
                    return;
                };
                return;
            }

            if (std.mem.eql(u8, op_name, "ora.sstore") or
                std.mem.eql(u8, op_name, "ora.tstore") or
                std.mem.eql(u8, op_name, "ora.map_store") or
                std.mem.eql(u8, op_name, "memref.store") or
                std.mem.eql(u8, op_name, "func.call") or
                std.mem.eql(u8, op_name, "call") or
                std.mem.eql(u8, op_name, "ora.assert"))
            {
                _ = self.encodeOperation(op) catch {
                    self.recordDegradation("failed to encode state effect operation");
                };
            }
        }

        self.encodeStateEffectsInRegions(op);
    }

    fn encodeStateEffectsInConditionalReturn(
        self: *Encoder,
        op: mlir.MlirOperation,
        next_op: mlir.MlirOperation,
    ) void {
        if (mlir.oraOperationGetNumOperands(op) < 1) {
            self.recordDegradation("ora.conditional_return missing condition while encoding state effects");
            return;
        }

        const condition_value = mlir.oraOperationGetOperand(op, 0);
        const condition = self.encodeValue(condition_value) catch {
            self.recordDegradation("failed to encode ora.conditional_return condition for state summary");
            return;
        };

        var base_state = self.captureStateSnapshot() catch {
            self.recordDegradation("failed to capture base state for ora.conditional_return summary");
            return;
        };
        defer base_state.deinit(self.allocator);

        var then_state = StateSnapshot.init(self.allocator);
        defer then_state.deinit(self.allocator);
        var else_state = StateSnapshot.init(self.allocator);
        defer else_state.deinit(self.allocator);

        self.restoreStateSnapshot(&base_state) catch {
            self.recordDegradation("failed to restore base state for ora.conditional_return then-branch");
            return;
        };
        const then_block = mlir.oraConditionalReturnOpGetThenBlock(op);
        if (!mlir.oraBlockIsNull(then_block)) {
            const saved_len = self.return_path_assumptions.items.len;
            defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
            self.return_path_assumptions.append(self.allocator, self.coerceToBool(condition)) catch {
                self.recordDegradation("failed to record ora.conditional_return then-branch path assumption");
                return;
            };
            self.encodeStateEffectsInBlockFrom(then_block, mlir.oraBlockGetFirstOperation(then_block));
            if (self.blockFallsThrough(then_block) and !mlir.oraOperationIsNull(next_op)) {
                const parent_block = mlir.mlirOperationGetBlock(op);
                if (!mlir.oraBlockIsNull(parent_block)) {
                    self.encodeStateEffectsInBlockFrom(parent_block, next_op);
                }
            }
        }
        then_state = self.captureStateSnapshot() catch {
            self.recordDegradation("failed to capture ora.conditional_return then-branch state summary");
            return;
        };

        self.restoreStateSnapshot(&base_state) catch {
            self.recordDegradation("failed to restore base state for ora.conditional_return else-branch");
            return;
        };
        const else_block = mlir.oraConditionalReturnOpGetElseBlock(op);
        if (!mlir.oraBlockIsNull(else_block)) {
            const saved_len = self.return_path_assumptions.items.len;
            defer self.return_path_assumptions.shrinkRetainingCapacity(saved_len);
            self.return_path_assumptions.append(self.allocator, z3.Z3_mk_not(self.context.ctx, self.coerceToBool(condition))) catch {
                self.recordDegradation("failed to record ora.conditional_return else-branch path assumption");
                return;
            };
            self.encodeStateEffectsInBlockFrom(else_block, mlir.oraBlockGetFirstOperation(else_block));
            if (self.blockFallsThrough(else_block) and !mlir.oraOperationIsNull(next_op)) {
                const parent_block = mlir.mlirOperationGetBlock(op);
                if (!mlir.oraBlockIsNull(parent_block)) {
                    self.encodeStateEffectsInBlockFrom(parent_block, next_op);
                }
            }
        }
        else_state = self.captureStateSnapshot() catch {
            self.recordDegradation("failed to capture ora.conditional_return else-branch state summary");
            return;
        };

        self.mergeStateSnapshotsIf(condition, &base_state, &then_state, &else_state) catch {
            self.recordDegradation("failed to merge ora.conditional_return branch state summary");
            return;
        };
    }

    fn encodeStateEffectsInBlockFrom(self: *Encoder, block: mlir.MlirBlock, start_op: mlir.MlirOperation) void {
        if (mlir.oraBlockIsNull(block)) return;
        var current = start_op;
        while (!mlir.oraOperationIsNull(current)) {
            const next = mlir.oraOperationGetNextInBlock(current);
            const name_ref = mlir.oraOperationGetName(current);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const op_name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, op_name, "ora.conditional_return")) {
                self.encodeStateEffectsInConditionalReturn(current, next);
                return;
            }

            self.encodeStateEffectsInOperation(current);
            current = next;
        }
    }

    fn encodeStateEffectsInRegion(self: *Encoder, region: mlir.MlirRegion) void {
        if (mlir.oraRegionIsNull(region)) return;
        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) {
            self.encodeStateEffectsInBlockFrom(block, mlir.oraBlockGetFirstOperation(block));
            block = mlir.oraBlockGetNextInRegion(block);
        }
    }

    fn encodeStateEffectsInRegions(self: *Encoder, op: mlir.MlirOperation) void {
        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(op));
        for (0..num_regions) |region_idx| {
            const region = mlir.oraOperationGetRegion(op, region_idx);
            self.encodeStateEffectsInRegion(region);
        }
    }

    fn materializeCallSummaryCurrent(
        self: *Encoder,
        mlir_op: mlir.MlirOperation,
        operands: []const z3.Z3_ast,
    ) EncodeError!void {
        const call_id = @intFromPtr(mlir_op.ptr);
        if (self.materialized_calls.contains(call_id)) return;

        const callee = try self.getOpaqueCalleeKey(mlir_op);
        defer self.allocator.free(callee);
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
                if (writes_unknown) {
                    self.recordDegradation("failed to recover known callee write set exactly");
                }
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
            const summarized = try self.tryInlineFunctionCallSummary(mlir_op, callee, fop, operands, slots.items, result_exprs);
            if (!summarized) {
                if (num_results > 0) {
                    self.recordCalleeResultDegradation(mlir_op, callee, "failed to encode known callee results exactly");
                }
                if (slots.items.len > 0) {
                    self.recordDegradation("failed to encode known callee state exactly");
                }
            }
        }

        for (0..num_results) |i| {
            const result_value = mlir.oraOperationGetResult(mlir_op, i);
            const result_type = mlir.oraValueGetType(result_value);
            const result_sort = try self.encodeMLIRType(result_type);
            const encoded = if (result_exprs[i]) |expr|
                expr
            else
                try self.encodeCallResultUFSymbol(callee, operands, slots.items, @intCast(i), result_sort);
            if (func_op != null and result_exprs[i] == null) {
                self.recordCalleeResultDegradation(mlir_op, callee, "known callee result fell back to opaque UF summary");
            }
            const result_id = @intFromPtr(result_value.ptr);
            try self.value_map.put(result_id, encoded);
        }

        if (self.verify_state) {
            for (slots.items, 0..) |slot, slot_idx| {
                const slot_preserved_exactly = slot.is_write and slot.post != null and self.astEquivalent(slot.post.?, slot.pre);
                var post = slot.post orelse slot.pre;
                if (slot.is_write and slot.post == null) {
                    if (func_op != null) {
                        self.recordDegradation("known callee state fell back to opaque UF summary");
                    }
                    post = try self.encodeCallStateTransitionUFSymbol(callee, slot, operands, slots.items);
                } else if (!slot.is_write) {
                    self.addConstraint(z3.Z3_mk_eq(self.context.ctx, post, slot.pre));
                    const slot_sort = z3.Z3_get_sort(self.context.ctx, post);
                    if (self.isArraySort(slot_sort)) {
                        const call_id_u64: u64 = @intCast(@intFromPtr(mlir_op.ptr));
                        const frame_id = @as(u64, @intCast(slot_idx)) ^ (call_id_u64 << 8);
                        self.addArrayEqualityFrameConstraint(slot.pre, post, frame_id) catch {
                            self.recordDegradation("failed to encode call-state frame constraint");
                        };
                    }
                }

                if (self.global_map.getPtr(slot.name)) |existing| {
                    existing.* = post;
                } else {
                    const key = try self.allocator.dupe(u8, slot.name);
                    try self.global_map.put(key, post);
                }
                if (slot.is_write and slot.post != null and !slot_preserved_exactly) {
                    try self.markGlobalSlotWritten(slot.name);
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

        // Nested map access: ora.map_get(parent_map, key) — the write
        // targets the root global that the parent map was loaded from.
        if (std.mem.eql(u8, op_name, "ora.map_get")) {
            const num_operands = mlir.oraOperationGetNumOperands(owner);
            if (num_operands < 1) return null;
            const parent_map = mlir.oraOperationGetOperand(owner, 0);
            return self.resolveGlobalNameFromMapOperand(parent_map);
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

    const MapGetAncestor = struct {
        parent_map: mlir.MlirValue,
        key: mlir.MlirValue,
    };

    fn isTransparentMapSourceOp(op_name: []const u8) bool {
        return std.mem.eql(u8, op_name, "ora.refinement_to_base") or
            std.mem.eql(u8, op_name, "ora.base_to_refinement") or
            std.mem.eql(u8, op_name, "arith.bitcast") or
            std.mem.eql(u8, op_name, "arith.extui") or
            std.mem.eql(u8, op_name, "arith.extsi") or
            std.mem.eql(u8, op_name, "arith.trunci") or
            std.mem.eql(u8, op_name, "arith.index_cast") or
            std.mem.eql(u8, op_name, "arith.index_castui") or
            std.mem.eql(u8, op_name, "arith.index_castsi") or
            std.mem.eql(u8, op_name, "builtin.unrealized_conversion_cast") or
            std.mem.eql(u8, op_name, "tensor.cast");
    }

    fn tryUpdateNestedGlobalMapStore(
        self: *Encoder,
        map_operand: mlir.MlirValue,
        stored_value: z3.Z3_ast,
        mode: EncodeMode,
        op_id: u64,
    ) EncodeError!void {
        // Collect map_get chain from leaf map operand to root map value.
        var chain = std.ArrayList(MapGetAncestor){};
        defer chain.deinit(self.allocator);

        var root_candidate = map_operand;
        while (mlir.oraValueIsAOpResult(root_candidate)) {
            const owner = mlir.oraOpResultGetOwner(root_candidate);
            if (mlir.oraOperationIsNull(owner)) break;

            const name_ref = mlir.oraOperationGetName(owner);
            defer @import("mlir_c_api").freeStringRef(name_ref);
            const op_name = if (name_ref.data == null or name_ref.length == 0)
                ""
            else
                name_ref.data[0..name_ref.length];

            if (std.mem.eql(u8, op_name, "ora.map_get")) {
                const num_operands = mlir.oraOperationGetNumOperands(owner);
                if (num_operands < 2) break;
                const parent_map = mlir.oraOperationGetOperand(owner, 0);
                const key = mlir.oraOperationGetOperand(owner, 1);
                try chain.append(self.allocator, .{ .parent_map = parent_map, .key = key });
                root_candidate = parent_map;
                continue;
            }

            if (isTransparentMapSourceOp(op_name)) {
                const num_operands = mlir.oraOperationGetNumOperands(owner);
                if (num_operands < 1) break;
                root_candidate = mlir.oraOperationGetOperand(owner, 0);
                continue;
            }

            break;
        }

        if (chain.items.len == 0) return;
        const global_name = self.resolveGlobalNameFromMapOperand(root_candidate) orelse return;

        // Rebuild parent stores from leaf -> root:
        // updated_leaf, then store into each parent map_get site.
        var updated = stored_value;
        for (chain.items, 0..) |ancestor, depth| {
            const parent_map_ast = try self.encodeValueWithMode(ancestor.parent_map, mode);
            const parent_sort = z3.Z3_get_sort(self.context.ctx, parent_map_ast);
            if (!self.isArraySort(parent_sort)) return;

            const key_sort = z3.Z3_get_array_sort_domain(self.context.ctx, parent_sort);
            const value_sort = z3.Z3_get_array_sort_range(self.context.ctx, parent_sort);
            const raw_key = try self.encodeValueWithMode(ancestor.key, mode);
            const parent_key = self.coerceAstToSort(raw_key, key_sort);
            const parent_value = self.coerceAstToSort(updated, value_sort);

            updated = self.encodeStore(parent_map_ast, parent_key, parent_value);
            self.addArrayStoreFrameConstraint(
                parent_map_ast,
                parent_key,
                updated,
                op_id +% @as(u64, @intCast(depth + 1)),
            ) catch {
                self.recordDegradation("failed to encode nested map-store frame constraint");
            };
        }

        if (self.global_map.getPtr(global_name)) |existing| {
            existing.* = updated;
        } else {
            const global_key = try self.allocator.dupe(u8, global_name);
            try self.global_map.put(global_key, updated);
        }
        try self.markGlobalSlotWritten(global_name);
    }

    fn extractRegionYield(
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
            if (std.mem.eql(u8, name, "scf.yield") or std.mem.eql(u8, name, "ora.yield")) {
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

    fn getSwitchCaseAttrValues(self: *Encoder, op: mlir.MlirOperation, name: []const u8) EncodeError![]i64 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate(name.ptr, name.len));
        if (mlir.oraAttributeIsNull(attr)) return try self.allocator.alloc(i64, 0);

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        var values = try self.allocator.alloc(i64, count);
        for (0..count) |i| {
            values[i] = mlir.oraIntegerAttrGetValueSInt(mlir.oraArrayAttrGetElement(attr, i));
        }
        return values;
    }

    fn getSwitchDefaultCaseIndex(_: *Encoder, op: mlir.MlirOperation) ?i64 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("default_case_index".ptr, "default_case_index".len));
        if (mlir.oraAttributeIsNull(attr)) return null;
        return mlir.oraIntegerAttrGetValueSInt(attr);
    }

    fn getSwitchCaseMetadata(self: *Encoder, op: mlir.MlirOperation, num_regions: usize) EncodeError!SwitchCaseMetadata {
        var metadata = SwitchCaseMetadata{
            .case_kinds = try self.allocator.alloc(i64, num_regions),
            .case_values = try self.allocator.alloc(i64, num_regions),
            .range_starts = try self.allocator.alloc(i64, num_regions),
            .range_ends = try self.allocator.alloc(i64, num_regions),
            .default_case_index = null,
        };
        errdefer metadata.deinit(self.allocator);

        @memset(metadata.case_kinds, 0);
        @memset(metadata.case_values, 0);
        @memset(metadata.range_starts, 0);
        @memset(metadata.range_ends, 0);

        const attr_case_kinds = try self.getSwitchCaseAttrValues(op, "case_kinds");
        defer self.allocator.free(attr_case_kinds);
        const attr_case_values = try self.getSwitchCaseAttrValues(op, "case_values");
        defer self.allocator.free(attr_case_values);
        const attr_range_starts = try self.getSwitchCaseAttrValues(op, "range_starts");
        defer self.allocator.free(attr_range_starts);
        const attr_range_ends = try self.getSwitchCaseAttrValues(op, "range_ends");
        defer self.allocator.free(attr_range_ends);
        const attr_default = self.getSwitchDefaultCaseIndex(op);

        if (attr_case_kinds.len >= num_regions) {
            @memcpy(metadata.case_kinds, attr_case_kinds[0..num_regions]);
            if (attr_case_values.len >= num_regions) @memcpy(metadata.case_values, attr_case_values[0..num_regions]);
            if (attr_range_starts.len >= num_regions) @memcpy(metadata.range_starts, attr_range_starts[0..num_regions]);
            if (attr_range_ends.len >= num_regions) @memcpy(metadata.range_ends, attr_range_ends[0..num_regions]);
            metadata.default_case_index = attr_default;
            return metadata;
        }

        try self.parseSwitchCaseMetadataFromPrint(op, &metadata);
        return metadata;
    }

    fn parseSwitchCaseMetadataFromPrint(self: *Encoder, op: mlir.MlirOperation, metadata: *SwitchCaseMetadata) EncodeError!void {
        const printed = mlir.oraOperationPrintToString(op);
        defer if (printed.data != null) @import("mlir_c_api").freeStringRef(printed);
        if (printed.data == null or printed.length == 0) return error.UnsupportedOperation;

        var case_index: usize = 0;
        var lines = std.mem.splitScalar(u8, printed.data[0..printed.length], '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, trimmed, "case ")) {
                if (case_index >= metadata.case_kinds.len) break;
                const body = trimmed["case ".len..];
                const arrow_index = std.mem.indexOf(u8, body, "=>") orelse continue;
                const pattern = std.mem.trim(u8, body[0..arrow_index], " \t");
                if (std.mem.indexOf(u8, pattern, "...")) |range_index| {
                    metadata.case_kinds[case_index] = 1;
                    metadata.range_starts[case_index] = try self.parseSwitchPatternInt(pattern[0..range_index]);
                    metadata.range_ends[case_index] = try self.parseSwitchPatternInt(pattern[range_index + 3 ..]);
                } else {
                    metadata.case_kinds[case_index] = 0;
                    metadata.case_values[case_index] = try self.parseSwitchPatternInt(pattern);
                }
                case_index += 1;
            } else if (std.mem.startsWith(u8, trimmed, "else =>")) {
                if (case_index >= metadata.case_kinds.len) break;
                metadata.case_kinds[case_index] = 2;
                metadata.default_case_index = @intCast(case_index);
                case_index += 1;
            }
        }

        if (case_index < metadata.case_kinds.len) return error.UnsupportedOperation;
    }

    fn parseSwitchPatternInt(_: *Encoder, text: []const u8) EncodeError!i64 {
        const trimmed = std.mem.trim(u8, text, " \t");
        if (std.mem.eql(u8, trimmed, "true")) return 1;
        if (std.mem.eql(u8, trimmed, "false")) return 0;
        return std.fmt.parseInt(i64, trimmed, 10) catch error.UnsupportedOperation;
    }

    fn buildOraSwitchCasePredicate(
        self: *Encoder,
        scrutinee: z3.Z3_ast,
        case_kinds: []const i64,
        case_values: []const i64,
        range_starts: []const i64,
        range_ends: []const i64,
        case_index: usize,
    ) EncodeError!z3.Z3_ast {
        if (case_index >= case_kinds.len) return error.UnsupportedOperation;
        const sort = z3.Z3_get_sort(self.context.ctx, scrutinee);
        return switch (case_kinds[case_index]) {
            0 => blk: {
                if (case_index >= case_values.len) return error.UnsupportedOperation;
                const case_value = try self.encodeScalarValueForSort(case_values[case_index], sort);
                break :blk self.encodeComparisonOp(.Eq, scrutinee, case_value);
            },
            1 => blk: {
                if (case_index >= range_starts.len or case_index >= range_ends.len) return error.UnsupportedOperation;
                const start_value = try self.encodeScalarValueForSort(range_starts[case_index], sort);
                const end_value = try self.encodeScalarValueForSort(range_ends[case_index], sort);
                const lower = self.encodeComparisonOp(.Ge, scrutinee, start_value);
                const upper = self.encodeComparisonOp(.Le, scrutinee, end_value);
                break :blk self.encodeAnd(&.{ lower, upper });
            },
            2 => self.encodeBoolConstant(true),
            else => error.UnsupportedOperation,
        };
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

    const MlirPrintCollector = struct {
        allocator: std.mem.Allocator,
        buffer: std.ArrayList(u8),
        failed: bool = false,
    };

    fn printMlirChunk(value: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
        const collector: *MlirPrintCollector = @ptrCast(@alignCast(user_data orelse return));
        if (value.data == null or value.length == 0) return;
        collector.buffer.appendSlice(collector.allocator, value.data[0..value.length]) catch {
            collector.failed = true;
        };
    }

    fn printMlirTypeOwned(self: *Encoder, ty: mlir.MlirType) ![]u8 {
        var collector = MlirPrintCollector{
            .allocator = self.allocator,
            .buffer = std.ArrayList(u8){},
        };
        errdefer collector.buffer.deinit(self.allocator);
        mlir.mlirTypePrint(ty, printMlirChunk, &collector);
        if (collector.failed) {
            self.recordDegradation("failed to collect printed MLIR type");
        }
        return try collector.buffer.toOwnedSlice(self.allocator);
    }

    fn printMlirAttributeOwned(self: *Encoder, attr: mlir.MlirAttribute) ![]u8 {
        var collector = MlirPrintCollector{
            .allocator = self.allocator,
            .buffer = std.ArrayList(u8){},
        };
        errdefer collector.buffer.deinit(self.allocator);
        mlir.mlirAttributePrint(attr, printMlirChunk, &collector);
        return try collector.buffer.toOwnedSlice(self.allocator);
    }

    fn parseStructTypeName(type_text: []const u8) ?[]const u8 {
        const trimmed = std.mem.trim(u8, type_text, " \t\n\r");
        const prefix = "!ora.struct<";
        if (!std.mem.startsWith(u8, trimmed, prefix) or trimmed.len <= prefix.len or trimmed[trimmed.len - 1] != '>') {
            return null;
        }
        const body = trimmed[prefix.len .. trimmed.len - 1];
        if (std.mem.indexOfScalar(u8, body, '"')) |first_quote| {
            const rest = body[first_quote + 1 ..];
            if (std.mem.indexOfScalar(u8, rest, '"')) |second_quote| {
                const quoted = rest[0..second_quote];
                if (quoted.len > 0) return quoted;
            }
        }
        const inner = std.mem.trim(u8, body, " \t\n\r\"");
        if (inner.len == 0) return null;
        return inner;
    }

    fn resolveRegisteredStructName(self: *Encoder, query: []const u8) ?[]const u8 {
        if (self.struct_decl_ops.contains(query) or self.struct_field_names_csv.contains(query)) {
            return query;
        }

        if (std.mem.lastIndexOfAny(u8, query, ".:")) |sep| {
            const suffix = query[sep + 1 ..];
            if (suffix.len > 0 and (self.struct_decl_ops.contains(suffix) or self.struct_field_names_csv.contains(suffix))) {
                return suffix;
            }
        }

        var it = self.struct_decl_ops.iterator();
        while (it.next()) |entry| {
            const registered = entry.key_ptr.*;
            if (std.mem.endsWith(u8, query, registered) or std.mem.endsWith(u8, registered, query)) {
                return registered;
            }
        }

        var field_it = self.struct_field_names_csv.iterator();
        while (field_it.next()) |entry| {
            const registered = entry.key_ptr.*;
            if (std.mem.endsWith(u8, query, registered) or std.mem.endsWith(u8, registered, query)) {
                return registered;
            }
        }

        return null;
    }

    fn lookupStructFieldNames(self: *Encoder, ty: mlir.MlirType) !?[]const u8 {
        const type_text = try self.printMlirTypeOwned(ty);
        defer self.allocator.free(type_text);
        if (self.struct_field_names_csv.get(type_text)) |direct| return direct;
        const parsed_name = parseStructTypeName(type_text) orelse return null;
        const struct_name = self.resolveRegisteredStructName(parsed_name) orelse parsed_name;
        return self.struct_field_names_csv.get(struct_name);
    }

    fn lookupStructFieldNamesInScope(
        self: *Encoder,
        scope_op: mlir.MlirOperation,
        ty: mlir.MlirType,
    ) !?[]u8 {
        const field_count = mlir.oraStructTypeGetFieldCountInScope(scope_op, ty);
        if (field_count == 0) return null;

        var csv_builder = std.ArrayList(u8){};
        defer csv_builder.deinit(self.allocator);

        for (0..field_count) |i| {
            const field_ref = mlir.oraStructTypeGetFieldNameInScope(scope_op, ty, i);
            if (field_ref.data == null or field_ref.length == 0) return null;
            if (csv_builder.items.len > 0) try csv_builder.append(self.allocator, ',');
            try csv_builder.appendSlice(self.allocator, field_ref.data[0..field_ref.length]);
        }

        if (csv_builder.items.len == 0) return null;
        return try csv_builder.toOwnedSlice(self.allocator);
    }

    fn blockFallsThrough(self: *Encoder, block: mlir.MlirBlock) bool {
        _ = self;
        if (mlir.oraBlockIsNull(block)) return true;
        var current = mlir.oraBlockGetFirstOperation(block);
        var last: mlir.MlirOperation = .{ .ptr = null };
        while (!mlir.oraOperationIsNull(current)) {
            last = current;
            current = mlir.oraOperationGetNextInBlock(current);
        }
        if (mlir.oraOperationIsNull(last)) return true;
        const name_ref = mlir.oraOperationGetName(last);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];
        return !(std.mem.eql(u8, name, "ora.return") or std.mem.eql(u8, name, "func.return"));
    }

    fn lookupStructFieldType(self: *Encoder, ty: mlir.MlirType, index: usize) !mlir.MlirType {
        const type_text = try self.printMlirTypeOwned(ty);
        defer self.allocator.free(type_text);
        const struct_decl = blk: {
            if (self.struct_decl_ops.get(type_text)) |direct| break :blk direct;
            const parsed_name = parseStructTypeName(type_text) orelse return .{ .ptr = null };
            const struct_name = self.resolveRegisteredStructName(parsed_name) orelse parsed_name;
            break :blk self.struct_decl_ops.get(struct_name) orelse return .{ .ptr = null };
        };
        const field_types_attr = mlir.oraOperationGetAttributeByName(struct_decl, mlir.oraStringRefCreate("ora.field_types", 15));
        if (mlir.oraAttributeIsNull(field_types_attr)) return .{ .ptr = null };
        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(field_types_attr));
        if (index >= count) return .{ .ptr = null };
        const field_type_attr = mlir.oraArrayAttrGetElement(field_types_attr, index);
        if (mlir.oraAttributeIsNull(field_type_attr)) return .{ .ptr = null };
        const field_type_text = try self.printMlirAttributeOwned(field_type_attr);
        defer self.allocator.free(field_type_text);
        var type_slice = std.mem.trim(u8, field_type_text, " \t\n\r");
        const type_prefix = "type<";
        if (std.mem.startsWith(u8, type_slice, type_prefix) and type_slice.len > type_prefix.len and type_slice[type_slice.len - 1] == '>') {
            type_slice = type_slice[type_prefix.len .. type_slice.len - 1];
        }
        return mlir.mlirTypeParseGet(mlir.mlirTypeGetContext(ty), mlir.oraStringRefCreate(type_slice.ptr, type_slice.len));
    }

    fn lookupStructFieldTypeInScope(
        self: *Encoder,
        scope_op: mlir.MlirOperation,
        ty: mlir.MlirType,
        index: usize,
    ) mlir.MlirType {
        _ = self;
        return mlir.oraStructTypeGetFieldTypeInScope(scope_op, ty, index);
    }

    fn buildFieldNamesCsvFromAttr(self: *Encoder, attr: mlir.MlirAttribute) !?[]u8 {
        if (mlir.oraAttributeIsNull(attr)) return null;

        const direct = mlir.oraStringAttrGetValue(attr);
        if (direct.data != null and direct.length > 0) {
            const copy = try self.allocator.dupe(u8, direct.data[0..direct.length]);
            return copy;
        }

        var csv_builder = std.ArrayList(u8){};
        defer csv_builder.deinit(self.allocator);

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        if (count == 0) return null;
        for (0..count) |i| {
            const field_attr = mlir.oraArrayAttrGetElement(attr, i);
            if (mlir.oraAttributeIsNull(field_attr)) continue;
            const field_ref = mlir.oraStringAttrGetValue(field_attr);
            if (field_ref.data == null or field_ref.length == 0) continue;
            if (csv_builder.items.len > 0) try csv_builder.append(self.allocator, ',');
            try csv_builder.appendSlice(self.allocator, field_ref.data[0..field_ref.length]);
        }
        if (csv_builder.items.len == 0) return null;
        const csv = try csv_builder.toOwnedSlice(self.allocator);
        return csv;
    }

    fn mergeRecoveredFieldNames(
        self: *Encoder,
        lhs: ?[]u8,
        rhs: ?[]u8,
    ) !?[]u8 {
        if (lhs == null) return rhs;
        if (rhs == null) return lhs;

        const lhs_csv = lhs.?;
        const rhs_csv = rhs.?;
        if (std.mem.eql(u8, lhs_csv, rhs_csv)) {
            self.allocator.free(rhs_csv);
            return lhs_csv;
        }

        self.allocator.free(lhs_csv);
        self.allocator.free(rhs_csv);
        return null;
    }

    fn mergeRecoveredFieldTypes(
        self: *Encoder,
        lhs: mlir.MlirType,
        rhs: mlir.MlirType,
    ) mlir.MlirType {
        _ = self;
        if (mlir.oraTypeIsNull(lhs)) return rhs;
        if (mlir.oraTypeIsNull(rhs)) return lhs;
        if (mlir.mlirTypeEqual(lhs, rhs)) return lhs;
        return .{ .ptr = null };
    }

    fn tryLookupStructFieldNamesFromValue(self: *Encoder, value: mlir.MlirValue) !?[]u8 {
        if (!mlir.oraValueIsAOpResult(value)) return null;
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return null;

        const name_ref = mlir.oraOperationGetName(owner);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.struct_init")) {
            const names_attr = mlir.oraOperationGetAttributeByName(owner, mlir.oraStringRefCreate("ora.field_names", 15));
            if (try self.buildFieldNamesCsvFromAttr(names_attr)) |csv| return csv;
            return try self.lookupStructFieldNamesInScope(owner, mlir.oraValueGetType(value));
        }

        if (std.mem.eql(u8, op_name, "ora.struct_instantiate")) {
            if (self.getStringAttr(owner, "struct_name")) |struct_name| {
                if (self.struct_field_names_csv.get(struct_name)) |csv| {
                    const copy = try self.allocator.dupe(u8, csv);
                    return copy;
                }
            }
            const names_attr = mlir.oraOperationGetAttributeByName(owner, mlir.oraStringRefCreate("ora.field_names", 15));
            if (try self.buildFieldNamesCsvFromAttr(names_attr)) |csv| return csv;
            return try self.lookupStructFieldNamesInScope(owner, mlir.oraValueGetType(value));
        }

        if (std.mem.eql(u8, op_name, "ora.struct_field_update")) {
            return try self.tryLookupStructFieldNamesFromValue(mlir.oraOperationGetOperand(owner, 0));
        }

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            const callee = self.resolveCalleeName(owner) catch return null;
            defer if (callee) |name| self.allocator.free(name);
            const func_op = if (callee) |name| self.function_ops.get(name) else null;
            if (func_op) |known_func| {
                const result_index = self.getResultIndex(owner, value) orelse return null;
                if (try self.extractFunctionReturnValue(known_func, result_index)) |returned_value| {
                    return try self.tryLookupStructFieldNamesFromValue(returned_value);
                }
                return try self.tryLookupStructFieldNamesFromFunctionReturn(known_func, result_index);
            }
        }

        if (std.mem.eql(u8, op_name, "scf.if")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const then_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return null;
            const else_value = (try self.extractRegionYieldValue(owner, 1, result_index)) orelse return null;
            const then_csv = try self.tryLookupStructFieldNamesFromValue(then_value);
            const else_csv = try self.tryLookupStructFieldNamesFromValue(else_value);
            return try self.mergeRecoveredFieldNames(then_csv, else_csv);
        }

        if (std.mem.eql(u8, op_name, "scf.execute_region")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const yielded_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return null;
            return try self.tryLookupStructFieldNamesFromValue(yielded_value);
        }

        if (std.mem.eql(u8, op_name, "ora.switch_expr") or std.mem.eql(u8, op_name, "ora.switch")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(owner));
            var recovered: ?[]u8 = null;
            for (0..num_regions) |region_index| {
                const branch_value = (try self.extractRegionYieldValue(owner, @intCast(region_index), result_index)) orelse continue;
                const branch_csv = try self.tryLookupStructFieldNamesFromValue(branch_value);
                recovered = try self.mergeRecoveredFieldNames(recovered, branch_csv);
                if (recovered == null and branch_csv != null) return null;
            }
            return recovered;
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const result_index = self.getResultIndex(owner, value) orelse return null;
            const try_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return null;
            const catch_value = (try self.extractRegionYieldValue(owner, 1, result_index)) orelse return null;
            const try_csv = try self.tryLookupStructFieldNamesFromValue(try_value);
            const catch_csv = try self.tryLookupStructFieldNamesFromValue(catch_value);
            return try self.mergeRecoveredFieldNames(try_csv, catch_csv);
        }

        return null;
    }

    fn tryLookupStructFieldTypeFromValue(self: *Encoder, value: mlir.MlirValue, index: usize) !mlir.MlirType {
        if (!mlir.oraValueIsAOpResult(value)) return .{ .ptr = null };
        const owner = mlir.oraOpResultGetOwner(value);
        if (mlir.oraOperationIsNull(owner)) return .{ .ptr = null };

        const name_ref = mlir.oraOperationGetName(owner);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        const op_name = if (name_ref.data == null or name_ref.length == 0)
            ""
        else
            name_ref.data[0..name_ref.length];

        if (std.mem.eql(u8, op_name, "ora.struct_init") or std.mem.eql(u8, op_name, "ora.struct_instantiate")) {
            const num_operands: usize = @intCast(mlir.oraOperationGetNumOperands(owner));
            if (index >= num_operands) return .{ .ptr = null };
            return mlir.oraValueGetType(mlir.oraOperationGetOperand(owner, @intCast(index)));
        }

        if (std.mem.eql(u8, op_name, "ora.struct_field_update")) {
            const field_names_csv = try self.tryLookupStructFieldNamesFromValue(value);
            defer if (field_names_csv) |csv| self.allocator.free(csv);

            const updated_field_name = self.getStringAttr(owner, "field_name") orelse return .{ .ptr = null };
            if (field_names_csv) |csv| {
                var it = std.mem.splitScalar(u8, csv, ',');
                var field_idx: usize = 0;
                while (it.next()) |part| : (field_idx += 1) {
                    const trimmed = std.mem.trim(u8, part, " \t\n\r");
                    if (!std.mem.eql(u8, trimmed, updated_field_name)) continue;
                    if (field_idx == index) {
                        return mlir.oraValueGetType(mlir.oraOperationGetOperand(owner, 1));
                    }
                    break;
                }
            }
            return try self.tryLookupStructFieldTypeFromValue(mlir.oraOperationGetOperand(owner, 0), index);
        }

        if (std.mem.eql(u8, op_name, "func.call") or std.mem.eql(u8, op_name, "call")) {
            const callee = self.resolveCalleeName(owner) catch return .{ .ptr = null };
            defer if (callee) |name| self.allocator.free(name);
            const func_op = if (callee) |name| self.function_ops.get(name) else null;
            if (func_op) |known_func| {
                const result_index = self.getResultIndex(owner, value) orelse return .{ .ptr = null };
                if (try self.extractFunctionReturnValue(known_func, result_index)) |returned_value| {
                    return try self.tryLookupStructFieldTypeFromValue(returned_value, index);
                }
                return try self.tryLookupStructFieldTypeFromFunctionReturn(known_func, result_index, index);
            }
        }

        if (std.mem.eql(u8, op_name, "scf.if")) {
            const result_index = self.getResultIndex(owner, value) orelse return .{ .ptr = null };
            const then_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return .{ .ptr = null };
            const else_value = (try self.extractRegionYieldValue(owner, 1, result_index)) orelse return .{ .ptr = null };
            return self.mergeRecoveredFieldTypes(
                try self.tryLookupStructFieldTypeFromValue(then_value, index),
                try self.tryLookupStructFieldTypeFromValue(else_value, index),
            );
        }

        if (std.mem.eql(u8, op_name, "scf.execute_region")) {
            const result_index = self.getResultIndex(owner, value) orelse return .{ .ptr = null };
            const yielded_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return .{ .ptr = null };
            return try self.tryLookupStructFieldTypeFromValue(yielded_value, index);
        }

        if (std.mem.eql(u8, op_name, "ora.switch_expr") or std.mem.eql(u8, op_name, "ora.switch")) {
            const result_index = self.getResultIndex(owner, value) orelse return .{ .ptr = null };
            const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(owner));
            var recovered: mlir.MlirType = .{ .ptr = null };
            for (0..num_regions) |region_index| {
                const branch_value = (try self.extractRegionYieldValue(owner, @intCast(region_index), result_index)) orelse continue;
                recovered = self.mergeRecoveredFieldTypes(
                    recovered,
                    try self.tryLookupStructFieldTypeFromValue(branch_value, index),
                );
                if (mlir.oraTypeIsNull(recovered)) return .{ .ptr = null };
            }
            return recovered;
        }

        if (std.mem.eql(u8, op_name, "ora.try_stmt")) {
            const result_index = self.getResultIndex(owner, value) orelse return .{ .ptr = null };
            const try_value = (try self.extractRegionYieldValue(owner, 0, result_index)) orelse return .{ .ptr = null };
            const catch_value = (try self.extractRegionYieldValue(owner, 1, result_index)) orelse return .{ .ptr = null };
            return self.mergeRecoveredFieldTypes(
                try self.tryLookupStructFieldTypeFromValue(try_value, index),
                try self.tryLookupStructFieldTypeFromValue(catch_value, index),
            );
        }

        return .{ .ptr = null };
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

    pub fn coerceBoolean(self: *Encoder, ast: z3.Z3_ast) z3.Z3_ast {
        return self.coerceToBool(ast);
    }

    pub fn encodeScalarValueForSort(self: *Encoder, value: i64, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        const kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (kind == z3.Z3_BOOL_SORT) {
            return self.encodeBoolConstant(value != 0);
        }
        if (kind == z3.Z3_BV_SORT) {
            const width = z3.Z3_get_bv_sort_size(self.context.ctx, sort);
            return self.encodeIntegerConstant(self.normalizeSignedIntToWidth(value, width), width);
        }
        return error.UnsupportedOperation;
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
    fn getCmpPredicate(self: *Encoder, mlir_op: mlir.MlirOperation) EncodeError!u32 {
        // extract predicate from arith.cmpi attributes
        // the predicate attribute contains the comparison type (eq, ne, ult, ule, ugt, uge, etc.)
        const attr_name_ref = mlir.oraStringRefCreate("predicate".ptr, "predicate".len);
        const attr = mlir.oraOperationGetAttributeByName(mlir_op, attr_name_ref);
        if (mlir.oraAttributeIsNull(attr)) {
            self.recordDegradation("arith.cmpi missing predicate");
            return error.UnsupportedOperation;
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
        return self.parseConstAttrValue(attr, width) orelse blk: {
            self.recordDegradation("failed to decode constant value");
            break :blk 0;
        };
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
        const type_ctx = mlir.mlirTypeGetContext(ty);
        const index_ty = mlir.oraIndexTypeCreate(type_ctx);
        if (!mlir.oraTypeIsNull(index_ty) and mlir.oraTypeEqual(ty, index_ty)) {
            return 256;
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

        const result_sort = try self.encodeMLIRType(result_type);
        const op_id = @intFromPtr(mlir_op.ptr);
        const in_width = self.getTypeBitWidth(operand_type) orelse return try self.degradeToUndef(result_sort, "cast_width", op_id, "failed to determine operand cast width");
        const out_width = self.getTypeBitWidth(result_type) orelse return try self.degradeToUndef(result_sort, "cast_width", op_id, "failed to determine result cast width");
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

        const coerced = self.coerceComparisonOperands(operands[0], operands[1]);
        const lhs = coerced.lhs;
        const rhs = coerced.rhs;

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

    fn coerceComparisonOperands(self: *Encoder, lhs: z3.Z3_ast, rhs: z3.Z3_ast) struct { lhs: z3.Z3_ast, rhs: z3.Z3_ast } {
        const lhs_sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const rhs_sort = z3.Z3_get_sort(self.context.ctx, rhs);
        if (lhs_sort == rhs_sort) return .{ .lhs = lhs, .rhs = rhs };

        const lhs_kind = z3.Z3_get_sort_kind(self.context.ctx, lhs_sort);
        const rhs_kind = z3.Z3_get_sort_kind(self.context.ctx, rhs_sort);
        if (lhs_kind == z3.Z3_BV_SORT and rhs_kind == z3.Z3_BV_SORT) {
            const lhs_width = z3.Z3_get_bv_sort_size(self.context.ctx, lhs_sort);
            const rhs_width = z3.Z3_get_bv_sort_size(self.context.ctx, rhs_sort);
            const target_sort = if (lhs_width >= rhs_width) lhs_sort else rhs_sort;
            return .{
                .lhs = self.coerceAstToSort(lhs, target_sort),
                .rhs = self.coerceAstToSort(rhs, target_sort),
            };
        }

        return .{
            .lhs = lhs,
            .rhs = self.coerceAstToSort(rhs, lhs_sort),
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
