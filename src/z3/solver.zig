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

    pub fn check(self: *Solver) c.Z3_lbool {
        return c.Z3_solver_check(self.context.ctx, self.solver);
    }

    pub fn getModel(self: *Solver) ?c.Z3_model {
        return c.Z3_solver_get_model(self.context.ctx, self.solver);
    }

    pub fn reset(self: *Solver) void {
        c.Z3_solver_reset(self.context.ctx, self.solver);
    }

    // todo: Add methods for:
    // - push/pop (incremental solving)
    // - timeout management
    // - unsat core extraction
};
