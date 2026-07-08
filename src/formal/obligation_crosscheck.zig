//! Cross-check canonical obligations across solver/export backends.
//!
//! This layer is intentionally policy-sized: it does not encode formulas
//! itself. It asks the canonical Z3 adapter for a verdict, asks the Lean emitter
//! to export the same manifest, and derives the conservative runtime-erasure
//! decision from those two facts.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const obligation = @import("obligation.zig");
const obligation_to_lean = @import("obligation_to_lean.zig");
const obligation_to_z3 = @import("obligation_to_z3.zig");

pub const RuntimeErasureDecision = enum(u8) {
    elide_runtime_check,
    keep_runtime_check,
};

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    obligation_id: obligation.Id,
    z3_status: obligation_to_z3.CheckStatus,
    lean_source: []const u8,
    runtime_erasure: RuntimeErasureDecision,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn crossCheckRuntimeGuard(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    id: obligation.Id,
) !Result {
    const target = obligation.findById(set.obligations, id) orelse return error.UnknownObligation;
    const guard = switch (target.kind) {
        .runtime_guard => |guard| guard,
        else => return error.ExpectedRuntimeGuard,
    };

    var z3_ctx = try z3_verification.Z3Context.init(allocator);
    defer z3_ctx.deinit();

    var z3_adapter = obligation_to_z3.Adapter.init(&z3_ctx, allocator, set);
    const z3_status = try z3_adapter.checkObligation(id);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var buffer = std.Io.Writer.Allocating.init(arena_allocator);
    const lean_source = blk: {
        errdefer buffer.deinit();
        try obligation_to_lean.writeModule(&buffer.writer, set, .{});
        break :blk try buffer.toOwnedSlice();
    };

    return .{
        .arena = arena,
        .obligation_id = id,
        .z3_status = z3_status,
        .lean_source = lean_source,
        .runtime_erasure = runtimeErasureDecision(guard.erasure, z3_status),
    };
}

fn runtimeErasureDecision(
    policy: obligation.GuardErasurePolicy,
    status: obligation_to_z3.CheckStatus,
) RuntimeErasureDecision {
    return switch (policy) {
        .always_runtime => .keep_runtime_check,
        .may_elide_if_proven => if (status == .proved)
            .elide_runtime_check
        else
            .keep_runtime_check,
    };
}
