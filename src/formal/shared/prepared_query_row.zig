//! Typed, solver-independent prepared-query boundary for source accounting.
//!
//! Z3 deliberately projects its larger internal manifest into this row. The
//! source-accounting adapter consumes only this module, so neither side relies
//! on a duck-typed field contract or imports the other's coordinator.

pub const QueryKind = enum(u8) {
    obligation,
    loop_invariant_step,
    loop_body_safety,
    loop_invariant_post,
    guard_satisfy,
    guard_violate,
};

pub const MatchStatus = enum(u8) {
    not_applicable,
    matched,
    missing,
    ambiguous,
};

pub const BoundaryRole = enum(u8) {
    proof_target,
    assumption_context,
};

pub const Row = struct {
    kind: QueryKind,
    function_name: []const u8,
    file: []const u8 = "",
    line: u32 = 0,
    column: u32 = 0,
    match_status: MatchStatus = .not_applicable,
    query_id: ?u32 = null,
    boundary_role: ?BoundaryRole = null,
    boundary_callee_name: ?[]const u8 = null,
    boundary_source_fact_id: ?u32 = null,
};
