//! Z3 prepared-query snapshot to obligation manifest adapter.
//!
//! This module consumes data-only rows exported by `z3.verification`. It does
//! not inspect Z3 ASTs, rebuild SMT, or infer formulas. Formula expansion is a
//! later slice; this adapter records the canonical query surface that Z3
//! already built.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const obligation = @import("obligation.zig");

pub const CollectResult = struct {
    arena: std.heap.ArenaAllocator,
    set: obligation.ObligationSet,

    pub fn deinit(self: *CollectResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn collectPreparedQueries(
    allocator: std.mem.Allocator,
    rows: []const z3_verification.PreparedQueryManifestRow,
) !CollectResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var queries: std.ArrayList(obligation.VerificationQuery) = .empty;
    errdefer queries.deinit(arena_allocator);

    for (rows, 0..) |row, index| {
        const id = std.math.cast(obligation.Id, index + 1) orelse return error.TooManyQueries;
        try queries.append(arena_allocator, .{
            .id = id,
            .owner = .{ .function = .{ .name = try arena_allocator.dupe(u8, row.function_name) } },
            .source = .{
                .file = try optionalFile(arena_allocator, row.file),
                .line = row.line,
                .column = row.column,
            },
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = queryKind(row.kind),
            .logical_role = logicalRole(row.obligation_kind),
            .guard_id = if (row.guard_id) |guard_id| try arena_allocator.dupe(u8, guard_id) else null,
            .fragment = queryFragment(row.fragment),
            .solver_logic = querySolverLogic(row.solver_logic),
            .constraint_count = row.constraint_count,
            .smtlib_hash = row.smtlib_hash,
        });
    }

    const set: obligation.ObligationSet = .{
        .queries = try queries.toOwnedSlice(arena_allocator),
    };

    return .{
        .arena = arena,
        .set = set,
    };
}

fn optionalFile(allocator: std.mem.Allocator, file: []const u8) !?[]const u8 {
    if (file.len == 0) return null;
    return try allocator.dupe(u8, file);
}

fn queryKind(kind: z3_verification.QueryKind) obligation.VerificationQueryKind {
    return switch (kind) {
        .Base => .base,
        .Obligation => .obligation,
        .LoopInvariantStep => .loop_invariant_step,
        .LoopBodySafety => .loop_body_safety,
        .LoopInvariantPost => .loop_invariant_post,
        .GuardSatisfy => .guard_satisfy,
        .GuardViolate => .guard_violate,
    };
}

fn queryFragment(fragment: z3_verification.QueryFragment) obligation.VerificationQueryFragment {
    return switch (fragment) {
        .unknown => .unknown,
        .qf_bv => .qf_bv,
        .qf_bv_array => .qf_bv_array,
        .aufbv => .aufbv,
        .aufbv_quantifiers => .aufbv_quantifiers,
        .other => .other,
    };
}

fn querySolverLogic(logic: z3_verification.QuerySolverLogic) obligation.VerificationSolverLogic {
    return switch (logic) {
        .all => .all,
        .qf_aufbv => .qf_aufbv,
    };
}

fn logicalRole(kind: ?z3_verification.AnnotationKind) ?obligation.LogicalRole {
    const annotation = kind orelse return null;
    const label = z3_verification.formalLogicalRoleLabel(annotation) orelse return null;
    return std.meta.stringToEnum(obligation.LogicalRole, label) orelse unreachable;
}
