//===----------------------------------------------------------------------===//
//
// Z3 Solver Interface
//
//===----------------------------------------------------------------------===//
//
// High-level interface to Z3 solver, handling queries and results.
//
//===----------------------------------------------------------------------===//

const std = @import("std");
const c = @import("c.zig");
const Context = @import("context.zig").Context;

/// Z3 Solver wrapper
pub const Solver = struct {
    context: *Context,
    solver: c.Z3_solver,
    allocator: std.mem.Allocator,

    pub fn init(context: *Context, allocator: std.mem.Allocator) !Solver {
        const solver = c.Z3_mk_solver(context.ctx) orelse return error.SolverInitFailed;
        c.Z3_solver_inc_ref(context.ctx, solver);

        return Solver{
            .context = context,
            .solver = solver,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Solver) void {
        c.Z3_solver_dec_ref(self.context.ctx, self.solver);
    }

    pub fn assert(self: *Solver, constraint: c.Z3_ast) void {
        c.Z3_solver_assert(self.context.ctx, self.solver, constraint);
    }

    pub fn assertChecked(self: *Solver, constraint: c.Z3_ast) !void {
        self.context.clearLastError();
        c.Z3_solver_assert(self.context.ctx, self.solver, constraint);
        try self.context.checkNoError();
    }

    pub fn check(self: *Solver) c.Z3_lbool {
        return c.Z3_solver_check(self.context.ctx, self.solver);
    }

    pub fn checkChecked(self: *Solver) !c.Z3_lbool {
        self.context.clearLastError();
        const status = c.Z3_solver_check(self.context.ctx, self.solver);
        try self.context.checkNoError();
        return status;
    }

    pub fn checkAssumptions(self: *Solver, assumptions: []const c.Z3_ast) c.Z3_lbool {
        return c.Z3_solver_check_assumptions(
            self.context.ctx,
            self.solver,
            @intCast(assumptions.len),
            if (assumptions.len == 0) null else assumptions.ptr,
        );
    }

    pub fn checkAssumptionsChecked(self: *Solver, assumptions: []const c.Z3_ast) !c.Z3_lbool {
        self.context.clearLastError();
        const status = c.Z3_solver_check_assumptions(
            self.context.ctx,
            self.solver,
            @intCast(assumptions.len),
            if (assumptions.len == 0) null else assumptions.ptr,
        );
        try self.context.checkNoError();
        return status;
    }

    pub fn getModel(self: *Solver) ?c.Z3_model {
        return c.Z3_solver_get_model(self.context.ctx, self.solver);
    }

    pub fn getModelChecked(self: *Solver) !?c.Z3_model {
        self.context.clearLastError();
        const model = c.Z3_solver_get_model(self.context.ctx, self.solver);
        try self.context.checkNoError();
        return model;
    }

    pub fn getProof(self: *Solver) ?c.Z3_ast {
        return c.Z3_solver_get_proof(self.context.ctx, self.solver);
    }

    pub fn getProofChecked(self: *Solver) !?c.Z3_ast {
        self.context.clearLastError();
        const proof = c.Z3_solver_get_proof(self.context.ctx, self.solver);
        try self.context.checkNoError();
        return proof;
    }

    pub fn getProofStringOwned(self: *Solver) !?[]u8 {
        const proof = try self.getProofChecked() orelse return null;
        const raw = c.Z3_ast_to_string(self.context.ctx, proof);
        if (raw == null) return null;
        return try self.allocator.dupe(u8, std.mem.span(raw));
    }

    pub fn getUnsatCore(self: *Solver) c.Z3_ast_vector {
        return c.Z3_solver_get_unsat_core(self.context.ctx, self.solver);
    }

    pub fn getUnsatCoreChecked(self: *Solver) !c.Z3_ast_vector {
        self.context.clearLastError();
        const core = c.Z3_solver_get_unsat_core(self.context.ctx, self.solver);
        try self.context.checkNoError();
        return core;
    }

    pub fn getUnsatCoreOwned(self: *Solver) ![]c.Z3_ast {
        const core = try self.getUnsatCoreChecked();
        c.Z3_ast_vector_inc_ref(self.context.ctx, core);
        defer c.Z3_ast_vector_dec_ref(self.context.ctx, core);

        const len: usize = @intCast(c.Z3_ast_vector_size(self.context.ctx, core));
        var result = try self.allocator.alloc(c.Z3_ast, len);
        errdefer self.allocator.free(result);

        for (0..len) |i| {
            result[i] = c.Z3_ast_vector_get(self.context.ctx, core, @intCast(i));
        }
        return result;
    }

    pub fn reset(self: *Solver) void {
        c.Z3_solver_reset(self.context.ctx, self.solver);
    }

    pub fn resetChecked(self: *Solver) !void {
        self.context.clearLastError();
        c.Z3_solver_reset(self.context.ctx, self.solver);
        try self.context.checkNoError();
    }

    pub fn loadFromSmtlib(self: *Solver, smtlib: [:0]const u8, decl_symbols: []const c.Z3_symbol, decls: []const c.Z3_func_decl) !void {
        self.context.clearLastError();
        const parsed = c.Z3_parse_smtlib2_string(
            self.context.ctx,
            smtlib.ptr,
            0,
            null,
            null,
            @intCast(decl_symbols.len),
            if (decl_symbols.len == 0) null else decl_symbols.ptr,
            if (decls.len == 0) null else decls.ptr,
        ) orelse return error.Z3ApiError;
        c.Z3_ast_vector_inc_ref(self.context.ctx, parsed);
        defer c.Z3_ast_vector_dec_ref(self.context.ctx, parsed);
        const count = c.Z3_ast_vector_size(self.context.ctx, parsed);
        if (count == 0 and std.mem.indexOf(u8, smtlib, "(check-sat)") == null) {
            return error.Z3ApiError;
        }
        try self.context.checkNoError();
        var idx: u32 = 0;
        while (idx < count) : (idx += 1) {
            const formula = c.Z3_ast_vector_get(self.context.ctx, parsed, idx);
            c.Z3_solver_assert(self.context.ctx, self.solver, formula);
            try self.context.checkNoError();
        }
    }

    pub fn push(self: *Solver) void {
        c.Z3_solver_push(self.context.ctx, self.solver);
    }

    pub fn pushChecked(self: *Solver) !void {
        self.context.clearLastError();
        c.Z3_solver_push(self.context.ctx, self.solver);
        try self.context.checkNoError();
    }

    pub fn pop(self: *Solver) void {
        c.Z3_solver_pop(self.context.ctx, self.solver, 1);
    }

    pub fn popChecked(self: *Solver) !void {
        self.context.clearLastError();
        c.Z3_solver_pop(self.context.ctx, self.solver, 1);
        try self.context.checkNoError();
    }

    pub fn setTimeoutMs(self: *Solver, timeout_ms: u32) !void {
        self.context.clearLastError();
        const params = c.Z3_mk_params(self.context.ctx) orelse return error.SolverInitFailed;
        c.Z3_params_inc_ref(self.context.ctx, params);
        defer c.Z3_params_dec_ref(self.context.ctx, params);

        const sym = c.Z3_mk_string_symbol(self.context.ctx, "timeout");
        c.Z3_params_set_uint(self.context.ctx, params, sym, timeout_ms);
        c.Z3_solver_set_params(self.context.ctx, self.solver, params);
        try self.context.checkNoError();
    }

    pub fn setRandomSeed(self: *Solver, seed: u32) !void {
        self.context.clearLastError();
        const params = c.Z3_mk_params(self.context.ctx) orelse return error.SolverInitFailed;
        c.Z3_params_inc_ref(self.context.ctx, params);
        defer c.Z3_params_dec_ref(self.context.ctx, params);

        const sym = c.Z3_mk_string_symbol(self.context.ctx, "random_seed");
        c.Z3_params_set_uint(self.context.ctx, params, sym, seed);
        c.Z3_solver_set_params(self.context.ctx, self.solver, params);
        try self.context.checkNoError();
    }

    pub fn mkFreshBoolProxy(self: *Solver, prefix: [:0]const u8) !c.Z3_ast {
        self.context.clearLastError();
        const bool_sort = c.Z3_mk_bool_sort(self.context.ctx);
        const proxy = c.Z3_mk_fresh_const(self.context.ctx, prefix.ptr, bool_sort) orelse return error.SolverInitFailed;
        try self.context.checkNoError();
        return proxy;
    }
};

const testing = std.testing;

test "loadFromSmtlib rejects malformed smtlib" {
    var context = try Context.init(testing.allocator);
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    try testing.expectError(error.Z3ApiError, solver.loadFromSmtlib("(assert", &.{}, &.{}));
}

test "setTimeoutMs configures solver without Z3 error" {
    var context = try Context.init(testing.allocator);
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    try solver.setTimeoutMs(50);
    try testing.expectEqual(@as(c.Z3_error_code, c.Z3_OK), context.lastErrorCode());
}

test "setRandomSeed configures solver without Z3 error" {
    var context = try Context.init(testing.allocator);
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    try solver.setRandomSeed(7);
    try testing.expectEqual(@as(c.Z3_error_code, c.Z3_OK), context.lastErrorCode());
}

test "checked solver operations succeed on simple SAT query" {
    var context = try Context.init(testing.allocator);
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    const bool_sort = c.Z3_mk_bool_sort(context.ctx);
    const sym = c.Z3_mk_string_symbol(context.ctx, "p");
    const decl = c.Z3_mk_const(context.ctx, sym, bool_sort);

    try solver.resetChecked();
    try solver.pushChecked();
    try solver.assertChecked(decl);
    try testing.expectEqual(@as(c.Z3_lbool, c.Z3_L_TRUE), try solver.checkChecked());
    try testing.expect(try solver.getModelChecked() != null);
    try solver.popChecked();
}

test "checkAssumptions and unsat core wrappers work on simple contradiction" {
    var context = try Context.init(testing.allocator);
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    const p = try solver.mkFreshBoolProxy("core_p");
    const q = try solver.mkFreshBoolProxy("core_q");
    const not_q = c.Z3_mk_not(context.ctx, q);

    try solver.assertChecked(c.Z3_mk_implies(context.ctx, p, q));
    try solver.assertChecked(c.Z3_mk_implies(context.ctx, p, not_q));

    const assumptions = [_]c.Z3_ast{p};
    try testing.expectEqual(@as(c.Z3_lbool, c.Z3_L_FALSE), try solver.checkAssumptionsChecked(&assumptions));

    const core = try solver.getUnsatCoreOwned();
    defer testing.allocator.free(core);
    try testing.expectEqual(@as(usize, 1), core.len);
    try testing.expect(core[0] == p);
}

test "proof wrappers expose a proof on simple contradiction when enabled" {
    var context = try Context.initWithOptions(testing.allocator, .{ .proofs_enabled = true });
    defer context.deinit();

    var solver = try Solver.init(&context, testing.allocator);
    defer solver.deinit();

    const bool_sort = c.Z3_mk_bool_sort(context.ctx);
    const sym = c.Z3_mk_string_symbol(context.ctx, "proof_p");
    const p = c.Z3_mk_const(context.ctx, sym, bool_sort);
    const not_p = c.Z3_mk_not(context.ctx, p);

    try solver.assertChecked(p);
    try solver.assertChecked(not_p);
    try testing.expectEqual(@as(c.Z3_lbool, c.Z3_L_FALSE), try solver.checkChecked());

    const proof = try solver.getProofStringOwned();
    defer if (proof) |raw| testing.allocator.free(raw);

    try testing.expect(proof != null);
    try testing.expect(proof.?.len > 0);
}
