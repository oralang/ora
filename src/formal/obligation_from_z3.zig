//! Z3 prepared-query snapshot to obligation manifest adapter.
//!
//! This module consumes data-only rows exported by `z3.verification`. It does
//! not inspect Z3 ASTs, rebuild SMT, or infer formulas. Formula expansion is a
//! later slice; this adapter records the canonical query surface that Z3
//! already built.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const obligation = @import("obligation.zig");
const obligation_to_z3 = @import("obligation_to_z3.zig");

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
    var canonical_context: ?z3_verification.Z3Context = null;
    defer if (canonical_context) |*ctx| ctx.deinit();

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
            try appendCanonicalSmtCrosscheckDiagnostic(
                arena_allocator,
                &diagnostics,
                base,
                query,
                row,
                &canonical_context,
            );
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

fn appendCanonicalSmtCrosscheckDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    row: z3_verification.PreparedQueryManifestRow,
    canonical_context: *?z3_verification.Z3Context,
) !void {
    if (!queryEligibleForCanonicalSmtCrosscheck(query)) return;
    switch (obligation_to_z3.queryCanonicalSupport(set, query)) {
        .supported => {},
        .unsupported => |reason| {
            try appendCanonicalUnavailableDiagnostic(allocator, diagnostics, set, query, @tagName(reason));
            return;
        },
    }

    if (canonical_context.* == null) {
        canonical_context.* = try z3_verification.Z3Context.init(allocator);
    }

    var adapter = obligation_to_z3.Adapter.init(&canonical_context.*.?, allocator, set);
    const canonical = adapter.queryHashForRow(query) catch |err| {
        try appendCanonicalUnavailableDiagnostic(allocator, diagnostics, set, query, @errorName(err));
        return;
    };

    if (canonical.constraint_count == row.constraint_count and canonical.smtlib_hash == row.smtlib_hash) return;
    try appendCanonicalMismatchDiagnostic(allocator, diagnostics, set, query, row, canonical);
}

fn queryEligibleForCanonicalSmtCrosscheck(
    query: obligation.VerificationQuery,
) bool {
    return query.kind == .obligation and query.obligation_ids.len == 1;
}

fn appendCanonicalUnavailableDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    reason: []const u8,
) !void {
    const message = try std.fmt.allocPrint(
        allocator,
        "canonical_z3_unavailable: query id={d} kind={s} reason={s}",
        .{ query.id, @tagName(query.kind), reason },
    );
    try diagnostics.append(allocator, .{
        .kind = .canonical_z3_unavailable,
        .source = query.source,
        .message = message,
        .blocks_artifacts = obligation_to_z3.queryCanonicalRequiredModePromoted(set, query),
    });
}

fn appendCanonicalMismatchDiagnostic(
    allocator: std.mem.Allocator,
    diagnostics: *std.ArrayList(obligation.ObligationDiagnostic),
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    row: z3_verification.PreparedQueryManifestRow,
    canonical: obligation_to_z3.QueryHash,
) !void {
    const message = try std.fmt.allocPrint(
        allocator,
        "canonical Z3 hash mismatch for query id={d}: live=0x{x}/c={d} canonical=0x{x}/c={d}",
        .{ query.id, row.smtlib_hash, row.constraint_count, canonical.smtlib_hash, canonical.constraint_count },
    );
    try diagnostics.append(allocator, .{
        .kind = .canonical_z3_mismatch,
        .source = query.source,
        .message = message,
        .blocks_artifacts = obligation_to_z3.queryCanonicalRequiredModePromoted(set, query),
    });
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
    if (row.formal_query_id) |id| {
        return query.id == id and
            obligation.equalIdSlices(query.assumption_ids, row.formal_assumption_ids) and
            obligation.equalIdSlices(query.obligation_ids, row.formal_obligation_ids);
    }
    if (row.formal_match_status == .missing or row.formal_match_status == .ambiguous) return false;
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
    if (row.kind == .GuardViolate) {
        // Explicit guard-violate prepared rows are outside the Lean proof-target
        // identity lane; they keep the old guard-erasure matching path.
    } else if (row.kind == .Obligation) {
        const role = logicalRole(row.obligation_kind) orelse return false;
        if (role != .guard) return false;
    } else return false;
    if (!optionalStringEqual(query.guard_id, row.guard_id)) return false;
    if (!ownerMatchesFunctionName(query.owner, row.function_name)) return false;
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
    const message = if (row.formal_match_status == .missing or row.formal_match_status == .ambiguous)
        try std.fmt.allocPrint(
            allocator,
            "SMT prepared-query row was not matched by formal identity: {s} in {s} status={s} key={s} smtlib_hash=0x{x}",
            .{
                @tagName(row.kind),
                row.function_name,
                @tagName(row.formal_match_status),
                row.formal_match_key orelse "<none>",
                row.smtlib_hash,
            },
        )
    else
        try std.fmt.allocPrint(
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

test "prepared row formal id matches query without fuzzy metadata" {
    const assumptions = [_]obligation.Id{ 1, 2 };
    const obligations = [_]obligation.Id{3};
    const query: obligation.VerificationQuery = .{
        .id = 42,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .{ .file = "contract.ora", .line = 10, .column = 5 },
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .logical_role = .ensures,
        .assumption_ids = &assumptions,
        .obligation_ids = &obligations,
    };
    const row: z3_verification.PreparedQueryManifestRow = .{
        .kind = .Obligation,
        .function_name = "different_function",
        .obligation_kind = .ContractInvariant,
        .file = "different.ora",
        .line = 99,
        .column = 1,
        .formal_query_id = 42,
        .formal_assumption_ids = &assumptions,
        .formal_obligation_ids = &obligations,
        .formal_match_status = .matched,
    };

    try std.testing.expect(queryMatchesPreparedRow(query, row));
}

test "prepared row formal id mismatch does not fall back to source matching" {
    const assumptions = [_]obligation.Id{1};
    const obligations = [_]obligation.Id{2};
    const query: obligation.VerificationQuery = .{
        .id = 7,
        .owner = .{ .function = .{ .name = "bounded" } },
        .source = .{ .file = "same.ora", .line = 4, .column = 9 },
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .logical_role = .ensures,
        .assumption_ids = &assumptions,
        .obligation_ids = &obligations,
    };
    const row: z3_verification.PreparedQueryManifestRow = .{
        .kind = .Obligation,
        .function_name = "bounded",
        .obligation_kind = .Ensures,
        .file = "same.ora",
        .line = 4,
        .column = 9,
        .formal_query_id = 8,
        .formal_assumption_ids = &assumptions,
        .formal_obligation_ids = &obligations,
        .formal_match_status = .matched,
    };

    try std.testing.expect(!queryMatchesPreparedRow(query, row));
}

test "missing formal identity disables fuzzy proof target matching" {
    const query: obligation.VerificationQuery = .{
        .id = 7,
        .owner = .{ .function = .{ .name = "bounded" } },
        .source = .{ .file = "same.ora", .line = 4, .column = 9 },
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .logical_role = .ensures,
        .obligation_ids = &[_]obligation.Id{2},
    };
    const row: z3_verification.PreparedQueryManifestRow = .{
        .kind = .Obligation,
        .function_name = "bounded",
        .obligation_kind = .Ensures,
        .file = "same.ora",
        .line = 4,
        .column = 9,
        .formal_match_status = .missing,
        .formal_match_key = "source_op=0x1 kind=obligation role=ensures guard=none",
    };

    try std.testing.expect(!queryMatchesPreparedRow(query, row));
}

test "guard violate rows remain outside proof target formal identity" {
    const query: obligation.VerificationQuery = .{
        .id = 9,
        .owner = .{ .function = .{ .name = "guarded" } },
        .source = .{ .file = "formal.ora", .line = 10, .column = 9 },
        .phase = .report,
        .origin = .source,
        .kind = .guard_violate,
        .guard_id = "guard:file.ora:10:9:11:guard_clause",
        .obligation_ids = &[_]obligation.Id{8},
    };
    const row: z3_verification.PreparedQueryManifestRow = .{
        .kind = .GuardViolate,
        .function_name = "guarded",
        .guard_id = "guard:file.ora:10:9:11:guard_clause",
        .file = "z3.ora",
        .line = 1,
        .column = 1,
    };

    try std.testing.expect(queryMatchesPreparedRow(query, row));
}

test "canonical SMT hash crosscheck accepts matching supported formula row" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
        .kind = .requires,
        .formula = .{ .term = 2 },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .assumption_ids = &assumption_ids,
    }};
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, std.testing.allocator, set);
    const canonical = try adapter.queryHash(3);
    const rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "checked",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = canonical.constraint_count,
        .smtlib_hash = canonical.smtlib_hash,
        .result_status = .unknown,
        .formal_query_id = 3,
        .formal_assumption_ids = &assumption_ids,
        .formal_obligation_ids = &obligation_ids,
        .formal_match_status = .matched,
    }};

    var overlay = try overlayPreparedQueryResults(std.testing.allocator, set, &rows);
    defer overlay.deinit();

    try std.testing.expectEqual(@as(usize, 0), overlay.set.diagnostics.len);
    try std.testing.expectEqual(canonical.smtlib_hash, overlay.set.queries[0].smtlib_hash.?);
}

test "canonical SMT hash crosscheck accepts storage old query when hash matches" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .eq, .lhs = 1, .rhs = 2 } },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "view_balance" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 3 } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "view_balance" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };
    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, std.testing.allocator, set);
    const canonical = try adapter.queryHash(2);

    const rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "view_balance",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = canonical.constraint_count,
        .smtlib_hash = canonical.smtlib_hash,
        .result_status = .unknown,
        .formal_query_id = 2,
        .formal_obligation_ids = &obligation_ids,
        .formal_match_status = .matched,
    }};

    var overlay = try overlayPreparedQueryResults(std.testing.allocator, set, &rows);
    defer overlay.deinit();

    try std.testing.expectEqual(@as(usize, 0), overlay.set.diagnostics.len);
    try std.testing.expectEqual(canonical.constraint_count, overlay.set.queries[0].constraint_count);
    try std.testing.expectEqual(canonical.smtlib_hash, overlay.set.queries[0].smtlib_hash.?);
}

test "canonical SMT hash crosscheck accepts formula combination row when hash matches" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .result,
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .add, .lhs = 1, .rhs = 3, .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .and_, .lhs = 5, .rhs = 6 } },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "combined" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 7 } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "combined" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .canonical_smt_crosscheck_required = true,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };
    try std.testing.expectEqual(
        obligation_to_z3.CanonicalPromotionShape.formula_combination,
        obligation_to_z3.queryCanonicalPromotionShape(set, queries[0]).?,
    );
    try std.testing.expect(obligation_to_z3.queryCanonicalRequiredModePromoted(set, queries[0]));

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, std.testing.allocator, set);
    const canonical = try adapter.queryHash(2);
    const rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "combined",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = canonical.constraint_count,
        .smtlib_hash = canonical.smtlib_hash,
        .result_status = .unknown,
        .formal_query_id = 2,
        .formal_obligation_ids = &obligation_ids,
        .formal_match_status = .matched,
    }};

    var overlay = try overlayPreparedQueryResults(std.testing.allocator, set, &rows);
    defer overlay.deinit();

    try std.testing.expectEqual(@as(usize, 0), overlay.set.diagnostics.len);
    try std.testing.expectEqual(canonical.constraint_count, overlay.set.queries[0].constraint_count);
    try std.testing.expectEqual(canonical.smtlib_hash, overlay.set.queries[0].smtlib_hash.?);
}

test "canonical SMT hash mismatch blocks only when crosscheck is required" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const optional_queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const optional_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &optional_queries,
        .terms = &terms,
    };
    try std.testing.expectEqual(
        obligation_to_z3.CanonicalPromotionShape.core_formula,
        obligation_to_z3.queryCanonicalPromotionShape(optional_set, optional_queries[0]).?,
    );
    try std.testing.expect(!obligation_to_z3.queryCanonicalRequiredModePromoted(optional_set, optional_queries[0]));

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, std.testing.allocator, optional_set);
    const canonical = try adapter.queryHash(2);
    const wrong_hash = canonical.smtlib_hash +% 1;
    const rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "checked",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = canonical.constraint_count,
        .smtlib_hash = wrong_hash,
        .result_status = .unsat,
        .formal_query_id = 2,
        .formal_obligation_ids = &obligation_ids,
        .formal_match_status = .matched,
    }};

    var optional_overlay = try overlayPreparedQueryResults(std.testing.allocator, optional_set, &rows);
    defer optional_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), optional_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_mismatch, optional_overlay.set.diagnostics[0].kind);
    try std.testing.expect(!optional_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expect(optional_overlay.set.artifactDecision().isAllowed());

    const required_queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .canonical_smt_crosscheck_required = true,
    }};
    const required_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &required_queries,
        .terms = &terms,
    };
    try std.testing.expect(obligation_to_z3.queryCanonicalRequiredModePromoted(required_set, required_queries[0]));
    var required_overlay = try overlayPreparedQueryResults(std.testing.allocator, required_set, &rows);
    defer required_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), required_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_mismatch, required_overlay.set.diagnostics[0].kind);
    try std.testing.expect(required_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expectEqual(obligation.ArtifactDecision{ .blocked = .blocking_diagnostic }, required_overlay.set.artifactDecision());
}

test "canonical SMT required mode blocks promoted rows when type support regresses" {
    const unsupported_type_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "felt" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const missing_type_terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "0", .ty = null } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 1 } },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "typed" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "typed" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .canonical_smt_crosscheck_required = true,
    }};
    const rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "typed",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = 1,
        .smtlib_hash = 0x9999,
        .result_status = .unsat,
        .formal_query_id = 2,
        .formal_obligation_ids = &obligation_ids,
        .formal_match_status = .matched,
    }};

    const unsupported_type_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &unsupported_type_terms,
    };
    try std.testing.expectEqual(
        obligation_to_z3.CanonicalPromotionShape.core_formula,
        obligation_to_z3.queryCanonicalPromotionShape(unsupported_type_set, queries[0]).?,
    );
    try std.testing.expect(obligation_to_z3.queryCanonicalRequiredModePromoted(unsupported_type_set, queries[0]));
    var unsupported_overlay = try overlayPreparedQueryResults(std.testing.allocator, unsupported_type_set, &rows);
    defer unsupported_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), unsupported_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_unavailable, unsupported_overlay.set.diagnostics[0].kind);
    try std.testing.expect(unsupported_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expectEqual(obligation.ArtifactDecision{ .blocked = .blocking_diagnostic }, unsupported_overlay.set.artifactDecision());

    const missing_type_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &missing_type_terms,
    };
    try std.testing.expectEqual(
        obligation_to_z3.CanonicalPromotionShape.core_formula,
        obligation_to_z3.queryCanonicalPromotionShape(missing_type_set, queries[0]).?,
    );
    try std.testing.expect(obligation_to_z3.queryCanonicalRequiredModePromoted(missing_type_set, queries[0]));
    var missing_overlay = try overlayPreparedQueryResults(std.testing.allocator, missing_type_set, &rows);
    defer missing_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), missing_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_unavailable, missing_overlay.set.diagnostics[0].kind);
    try std.testing.expect(missing_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expectEqual(obligation.ArtifactDecision{ .blocked = .blocking_diagnostic }, missing_overlay.set.artifactDecision());
}

test "canonical SMT required mode is capped by promotion table exclusions" {
    const amount_terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
    };
    const resource_obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "resource_lane" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "resource", .ordinal = 0 } },
        .kind = .{ .resource = .{
            .op = .move,
            .domain = "Token",
            .amount = .{ .term = 0 },
            .property = .amount_non_negative,
        } },
    }};
    const resource_obligation_ids = [_]obligation.Id{1};
    const resource_queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "resource_lane" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &resource_obligation_ids,
        .canonical_smt_crosscheck_required = true,
    }};
    const resource_set: obligation.ObligationSet = .{
        .obligations = &resource_obligations,
        .queries = &resource_queries,
        .terms = &amount_terms,
    };
    try std.testing.expectEqual(@as(?obligation_to_z3.CanonicalPromotionShape, null), obligation_to_z3.queryCanonicalPromotionShape(resource_set, resource_queries[0]));
    try std.testing.expect(!obligation_to_z3.queryCanonicalRequiredModePromoted(resource_set, resource_queries[0]));
    const resource_rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "resource_lane",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = 1,
        .smtlib_hash = 0x1234,
        .result_status = .unsat,
        .formal_query_id = 2,
        .formal_obligation_ids = &resource_obligation_ids,
        .formal_match_status = .matched,
    }};
    var resource_overlay = try overlayPreparedQueryResults(std.testing.allocator, resource_set, &resource_rows);
    defer resource_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), resource_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_unavailable, resource_overlay.set.diagnostics[0].kind);
    try std.testing.expect(!resource_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expect(resource_overlay.set.artifactDecision().isAllowed());

    const effect_obligations = [_]obligation.Obligation{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "effect_lane" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "effect_frame", .ordinal = 0 } },
        .kind = .{ .effect_frame = .{ .relation = .write_covered_by_modifies } },
    }};
    const effect_obligation_ids = [_]obligation.Id{3};
    const effect_queries = [_]obligation.VerificationQuery{.{
        .id = 4,
        .owner = .{ .function = .{ .name = "effect_lane" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &effect_obligation_ids,
        .canonical_smt_crosscheck_required = true,
    }};
    const effect_set: obligation.ObligationSet = .{
        .obligations = &effect_obligations,
        .queries = &effect_queries,
    };
    try std.testing.expectEqual(@as(?obligation_to_z3.CanonicalPromotionShape, null), obligation_to_z3.queryCanonicalPromotionShape(effect_set, effect_queries[0]));
    try std.testing.expect(!obligation_to_z3.queryCanonicalRequiredModePromoted(effect_set, effect_queries[0]));
    const effect_rows = [_]z3_verification.PreparedQueryManifestRow{.{
        .kind = .Obligation,
        .function_name = "effect_lane",
        .file = "",
        .line = 0,
        .column = 0,
        .constraint_count = 1,
        .smtlib_hash = 0x5678,
        .result_status = .unsat,
        .formal_query_id = 4,
        .formal_obligation_ids = &effect_obligation_ids,
        .formal_match_status = .matched,
    }};
    var effect_overlay = try overlayPreparedQueryResults(std.testing.allocator, effect_set, &effect_rows);
    defer effect_overlay.deinit();
    try std.testing.expectEqual(@as(usize, 1), effect_overlay.set.diagnostics.len);
    try std.testing.expectEqual(obligation.DiagnosticKind.canonical_z3_unavailable, effect_overlay.set.diagnostics[0].kind);
    try std.testing.expect(!effect_overlay.set.diagnostics[0].blocks_artifacts);
    try std.testing.expect(effect_overlay.set.artifactDecision().isAllowed());
}
