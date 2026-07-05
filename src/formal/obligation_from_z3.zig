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
            .result = queryResult(row),
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

pub const OverlayResult = struct {
    arena: std.heap.ArenaAllocator,
    set: obligation.ObligationSet,

    pub fn deinit(self: *OverlayResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn overlayPreparedQueryResults(
    allocator: std.mem.Allocator,
    base: obligation.ObligationSet,
    rows: []const z3_verification.PreparedQueryManifestRow,
) !OverlayResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const row_matched = try arena_allocator.alloc(bool, rows.len);
    @memset(row_matched, false);

    var queries: std.ArrayList(obligation.VerificationQuery) = .empty;
    errdefer queries.deinit(arena_allocator);

    var diagnostics: std.ArrayList(obligation.ObligationDiagnostic) = .empty;
    errdefer diagnostics.deinit(arena_allocator);
    try diagnostics.appendSlice(arena_allocator, base.diagnostics);

    for (base.queries) |query| {
        var match_index: ?usize = null;
        var duplicate = false;
        for (rows, 0..) |row, index| {
            if (row_matched[index] and !queryCanSharePreparedRow(query, row)) continue;
            if (!queryMatchesPreparedRow(query, row)) continue;
            if (match_index != null) {
                duplicate = true;
                break;
            }
            match_index = index;
        }

        var merged_query = query;
        if (duplicate) {
            try appendDiagnostic(
                arena_allocator,
                &diagnostics,
                query.source,
                "multiple SMT prepared-query rows matched one formal query",
            );
        } else if (match_index) |index| {
            const row = rows[index];
            if (!queryCanSharePreparedRow(query, row)) {
                row_matched[index] = true;
            }
            merged_query.backend = .z3;
            merged_query.fragment = queryFragment(row.fragment);
            merged_query.solver_logic = querySolverLogic(row.solver_logic);
            merged_query.constraint_count = row.constraint_count;
            merged_query.smtlib_hash = row.smtlib_hash;
            merged_query.result = queryResult(row);
        } else if (queryRequiresPreparedRow(base, query)) {
            try appendUnmatchedFormalQueryDiagnostic(
                arena_allocator,
                &diagnostics,
                query,
            );
        }
        try queries.append(arena_allocator, merged_query);
    }

    for (rows, row_matched) |row, matched| {
        if (matched) continue;
        // Base satisfiability rows are owned by the SMT report gate. The MLIR
        // obligation manifest only has proof-target rows, so an unmatched base
        // row is not proof-manifest drift.
        if (row.kind == .Base) continue;
        // A clean proved row cannot be opened by a Lean proof and has already
        // been discharged by the SMT gate. Vacuity/caveat flags are still
        // artifact blockers, so those rows must remain visible.
        if (unmatchedCleanUnsatRow(row)) continue;
        try appendUnmatchedRowDiagnostic(arena_allocator, &diagnostics, row);
    }

    var merged = base;
    merged.queries = try queries.toOwnedSlice(arena_allocator);
    merged.diagnostics = try diagnostics.toOwnedSlice(arena_allocator);

    return .{
        .arena = arena,
        .set = merged,
    };
}

fn queryRequiresPreparedRow(set: obligation.ObligationSet, query: obligation.VerificationQuery) bool {
    if (query.backend == .lean) return false;
    if (query.obligation_ids.len == 0) return true;

    for (query.obligation_ids) |id| {
        const item = findObligation(set, id) orelse return true;
        if (item.kind != .effect_frame) return true;
    }
    return false;
}

fn findObligation(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Obligation {
    for (set.obligations) |item| {
        if (item.id == id) return item;
    }
    return null;
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

fn unmatchedCleanUnsatRow(row: z3_verification.PreparedQueryManifestRow) bool {
    return row.result_status == .unsat and
        !row.vacuous and
        !row.vacuity_unknown and
        !row.verified_with_caveats;
}

fn queryResult(row: z3_verification.PreparedQueryManifestRow) ?obligation.VerificationQueryResult {
    const status = row.result_status orelse return null;
    return .{
        .status = switch (status) {
            .sat => .sat,
            .unsat => .unsat,
            .unknown => .unknown,
        },
        .vacuous = row.vacuous,
        .vacuity_unknown = row.vacuity_unknown,
        .degraded = row.verified_with_caveats and !row.vacuous and !row.vacuity_unknown,
    };
}

fn logicalRole(kind: ?z3_verification.AnnotationKind) ?obligation.LogicalRole {
    const annotation = kind orelse return null;
    const label = z3_verification.formalLogicalRoleLabel(annotation) orelse return null;
    return std.meta.stringToEnum(obligation.LogicalRole, label) orelse unreachable;
}

fn queryMatchesPreparedRow(query: obligation.VerificationQuery, row: z3_verification.PreparedQueryManifestRow) bool {
    if (queryMatchesSourceGuardSatisfyRow(query, row)) return true;
    if (queryMatchesSourceGuardViolateRow(query, row)) return true;
    if (query.kind != queryKind(row.kind)) return false;
    if (!logicalRolesMatch(query.logical_role, logicalRole(row.obligation_kind))) return false;
    if (!optionalStringEqual(query.guard_id, row.guard_id)) return false;
    if (!ownerMatchesFunctionName(query.owner, row.function_name)) return false;
    if (!sourceMatchesPreparedRow(query.source, row)) return false;
    return true;
}

fn queryCanSharePreparedRow(query: obligation.VerificationQuery, row: z3_verification.PreparedQueryManifestRow) bool {
    return queryMatchesSourceGuardSatisfyRow(query, row);
}

fn queryMatchesSourceGuardSatisfyRow(
    query: obligation.VerificationQuery,
    row: z3_verification.PreparedQueryManifestRow,
) bool {
    if (query.kind != .guard_satisfy) return false;
    if (query.guard_id == null) return false;
    if (row.kind != .Base) return false;
    if (row.obligation_kind != null) return false;
    if (row.guard_id != null) return false;
    if (!ownerMatchesFunctionName(query.owner, row.function_name)) return false;
    if (!sourceMatchesPreparedRow(query.source, row)) return false;
    return true;
}

fn queryMatchesSourceGuardViolateRow(
    query: obligation.VerificationQuery,
    row: z3_verification.PreparedQueryManifestRow,
) bool {
    if (query.kind != .guard_violate) return false;
    if (row.kind != .Obligation) return false;
    const role = logicalRole(row.obligation_kind) orelse return false;
    if (role != .guard) return false;
    if (!optionalStringEqual(query.guard_id, row.guard_id)) return false;
    if (!ownerMatchesFunctionName(query.owner, row.function_name)) return false;
    if (!sourceMatchesPreparedRow(query.source, row)) return false;
    return true;
}

fn logicalRolesMatch(lhs: ?obligation.LogicalRole, rhs: ?obligation.LogicalRole) bool {
    if (lhs == rhs) return true;
    if (lhs == null and rhs != null and rhs.? == .guard) return true;
    // Full-mode Z3 treats untagged `ora.assert` as a contract-invariant
    // obligation, while the MLIR collector preserves the source-level role.
    if (lhs) |left| {
        if (rhs) |right| return left == .assert and right == .contract_invariant;
    }
    return false;
}

fn optionalStringEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

fn ownerMatchesFunctionName(owner: obligation.Owner, function_name: []const u8) bool {
    if (owner != .function) return function_name.len == 0;
    return std.mem.eql(u8, owner.function.name, function_name);
}

fn sourceMatchesPreparedRow(source: obligation.SourceRef, row: z3_verification.PreparedQueryManifestRow) bool {
    if (source.file) |file| {
        if (row.file.len != 0 and !sourceFileMatches(file, row.file)) return false;
    }
    if (source.line != 0 and row.line != 0 and source.line != row.line) return false;
    if (source.column != 0 and row.column != 0 and source.column != row.column) return false;
    return true;
}

fn sourceFileMatches(lhs: []const u8, rhs: []const u8) bool {
    if (std.mem.eql(u8, lhs, rhs)) return true;
    if (pathHasSuffix(lhs, rhs)) return true;
    if (pathHasSuffix(rhs, lhs)) return true;
    return false;
}

fn pathHasSuffix(path: []const u8, suffix: []const u8) bool {
    if (!std.mem.endsWith(u8, path, suffix)) return false;
    if (path.len == suffix.len) return true;
    const boundary = path[path.len - suffix.len - 1];
    return boundary == '/' or boundary == '\\';
}

fn appendUnmatchedRowDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    row: z3_verification.PreparedQueryManifestRow,
) !void {
    const message = try std.fmt.allocPrint(
        allocator,
        "SMT prepared-query row was not matched by the formal MLIR manifest: {s} in {s}",
        .{ @tagName(row.kind), row.function_name },
    );
    try diagnostics.append(allocator, .{
        .kind = .unmatched_report_row,
        .source = .{
            .file = try optionalFile(allocator, row.file),
            .line = row.line,
            .column = row.column,
        },
        .message = message,
    });
}

fn appendDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    source: obligation.SourceRef,
    message: []const u8,
) !void {
    try diagnostics.append(allocator, .{
        .kind = .unmatched_report_row,
        .source = source,
        .message = try allocator.dupe(u8, message),
    });
}

fn appendUnmatchedFormalQueryDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    query: obligation.VerificationQuery,
) !void {
    const role = if (query.logical_role) |logical_role| @tagName(logical_role) else "none";
    const guard_id = query.guard_id orelse "none";
    const message = try std.fmt.allocPrint(
        allocator,
        "formal query has no matching SMT prepared-query row: id={d} kind={s} role={s} owner={s} guard_id={s}",
        .{ query.id, @tagName(query.kind), role, ownerName(query.owner), guard_id },
    );
    try diagnostics.append(allocator, .{
        .kind = .unmatched_report_row,
        .source = query.source,
        .message = message,
    });
}

fn ownerName(owner: obligation.Owner) []const u8 {
    return switch (owner) {
        .module => |name| name,
        .contract => |name| name,
        .function => |function| function.name,
        .trait_method => |method| method.method_name,
        .statement => |statement| statement.function_name,
        .backend => |backend| backend.name,
    };
}
