//! Projection of actual prepared formal query identities into source accounting.
//!
//! Query rows come from the already-overlaid `ObligationSet`; no query is
//! reconstructed from MLIR, SMT text, or annotations. Bindings must therefore
//! name an actual prepared query ID or collection fails closed.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");
const prepared_query = @import("ora_prepared_query_row");
const obligation = @import("obligation.zig");
const from_mlir = @import("source_accounting_from_mlir.zig");

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    evidence: accounting.SymbolicEvidence,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Identity of one query row in the verifier's finalized prepared-query list.
/// `formal_query_id` links the concrete verifier row back to the formal query
/// binding for its source operation; multiple concrete rows may share it.
pub const PreparedQueryIdentity = struct {
    producer_id: u32,
    formal_query_id: obligation.Id,
    kind: accounting.QueryKind,
};

/// Actual verifier row generated for a source-level symbolic call boundary.
/// Location and callee identity are carried by the verifier; the production
/// pipeline matches them to the evaluator's exact abandoned-fold replacement.
pub const PreparedBoundaryQueryIdentity = struct {
    producer_id: u32,
    kind: accounting.QueryKind,
    role: accounting.UseRole,
    file: []const u8,
    line: u32,
    column: u32,
    callee_name: []const u8,
    source_fact_id: u32,
};

pub const BoundaryBinding = struct {
    use_id: accounting.UseId,
    handling_kind: accounting.HandlingKind,
    prepared_queries: []const PreparedBoundaryQueryIdentity = &.{},
};

pub const PreparedIdentityView = struct {
    queries: []const PreparedQueryIdentity,
    boundary_queries: []const PreparedBoundaryQueryIdentity = &.{},
    runtime_function_names: []const []const u8,
};

pub const PreparedIdentity = struct {
    query_storage: []PreparedQueryIdentity,
    boundary_query_storage: []PreparedBoundaryQueryIdentity,
    function_storage: [][]const u8,
    queries: []const PreparedQueryIdentity,
    boundary_queries: []const PreparedBoundaryQueryIdentity,
    runtime_function_names: []const []const u8,

    pub fn deinit(self: *PreparedIdentity, allocator: std.mem.Allocator) void {
        allocator.free(self.query_storage);
        allocator.free(self.boundary_query_storage);
        allocator.free(self.function_storage);
        self.* = undefined;
    }

    pub fn view(self: *const PreparedIdentity) PreparedIdentityView {
        return .{
            .queries = self.queries,
            .boundary_queries = self.boundary_queries,
            .runtime_function_names = self.runtime_function_names,
        };
    }
};

/// Normalize the verifier-owned prepared-query manifest without re-deriving
/// any query. Row order is the verifier's deterministic producer identity.
pub fn collectPreparedIdentity(allocator: std.mem.Allocator, rows: []const prepared_query.Row) !PreparedIdentity {
    var queries: std.ArrayList(PreparedQueryIdentity) = .empty;
    defer queries.deinit(allocator);
    var boundary_queries: std.ArrayList(PreparedBoundaryQueryIdentity) = .empty;
    defer boundary_queries.deinit(allocator);
    var function_names: std.ArrayList([]const u8) = .empty;
    defer function_names.deinit(allocator);

    for (rows, 0..) |row, index| {
        if (row.function_name.len != 0) try function_names.append(allocator, row.function_name);
        if (row.boundary_role) |role| {
            const callee_name = row.boundary_callee_name orelse
                return error.SourceAccountingBoundaryQueryMissingCallee;
            const source_fact_id = row.boundary_source_fact_id orelse
                return error.SourceAccountingBoundaryQueryMissingSourceFactId;
            if (callee_name.len == 0 or row.file.len == 0 or row.line == 0 or row.column == 0) {
                return error.SourceAccountingBoundaryQueryMissingLocation;
            }
            const producer_id = std.math.cast(u32, index + 1) orelse
                return error.SourceAccountingPreparedQueryProducerIdOverflow;
            try boundary_queries.append(allocator, .{
                .producer_id = producer_id,
                .kind = accountingQueryKind(row.kind),
                .role = switch (role) {
                    .proof_target => .proof_target,
                    .assumption_context => .assumption_context,
                },
                .file = row.file,
                .line = row.line,
                .column = row.column,
                .callee_name = callee_name,
                .source_fact_id = source_fact_id,
            });
        }
        if (row.match_status != .matched) continue;
        const formal_query_id = row.query_id orelse
            return error.SourceAccountingMatchedPreparedQueryMissingFormalId;
        const producer_id = std.math.cast(u32, index + 1) orelse
            return error.SourceAccountingPreparedQueryProducerIdOverflow;
        try queries.append(allocator, .{
            .producer_id = producer_id,
            .formal_query_id = formal_query_id,
            .kind = accountingQueryKind(row.kind),
        });
    }

    std.mem.sort([]const u8, function_names.items, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.less);
    const unique_function_count = dedupeSortedStrings(function_names.items);
    const query_storage = try queries.toOwnedSlice(allocator);
    errdefer allocator.free(query_storage);
    const boundary_query_storage = try boundary_queries.toOwnedSlice(allocator);
    errdefer allocator.free(boundary_query_storage);
    const function_storage = try function_names.toOwnedSlice(allocator);
    return .{
        .query_storage = query_storage,
        .boundary_query_storage = boundary_query_storage,
        .function_storage = function_storage,
        .queries = query_storage,
        .boundary_queries = boundary_query_storage,
        .runtime_function_names = function_storage[0..unique_function_count],
    };
}

pub fn bindPreparedQueries(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    bindings: []const from_mlir.Binding,
    mlir_evidence: accounting.SymbolicEvidence,
    actual_prepared_queries: ?[]const PreparedQueryIdentity,
) !Result {
    return bindPreparedQueriesWithBoundaries(
        allocator,
        set,
        bindings,
        mlir_evidence,
        actual_prepared_queries,
        &.{},
    );
}

pub fn bindPreparedQueriesWithBoundaries(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    bindings: []const from_mlir.Binding,
    mlir_evidence: accounting.SymbolicEvidence,
    actual_prepared_queries: ?[]const PreparedQueryIdentity,
    boundary_bindings: []const BoundaryBinding,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .evidence = .{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    if (actual_prepared_queries) |actual| try validatePreparedQueries(set, actual);
    try validateBoundaryBindings(bindings, boundary_bindings);

    var queries: std.ArrayList(accounting.QueryEvidence) = .empty;
    for (bindings) |binding| for (binding.query_ids) |formal_query_id| {
        const formal_query = findQuery(set, formal_query_id) orelse return error.UnknownSourceAccountingPreparedQuery;
        if (actual_prepared_queries) |actual| {
            for (actual) |prepared| {
                if (prepared.formal_query_id != formal_query_id) continue;
                try appendQueryEvidence(arena, &queries, .{
                    .producer_id = prepared.producer_id,
                    .kind = prepared.kind,
                    .use_id = binding.use_id,
                });
            }
        } else {
            try appendQueryEvidence(arena, &queries, .{
                .producer_id = formal_query_id,
                .kind = queryKind(formal_query.kind),
                .use_id = binding.use_id,
            });
        }
    };

    var obligations: std.ArrayList(accounting.CoveredEvidence) = .empty;
    for (mlir_evidence.obligations) |row| {
        try obligations.append(arena, .{
            .id = row.id,
            .producer_id = row.producer_id,
            .covered_use_ids = try arena.dupe(accounting.UseId, row.covered_use_ids),
        });
    }
    var next_boundary_obligation_producer: u32 = 0;
    for (obligations.items) |row| {
        next_boundary_obligation_producer = @max(next_boundary_obligation_producer, row.producer_id);
    }
    var boundary_obligation_producers = std.AutoHashMap(u32, u32).init(allocator);
    defer boundary_obligation_producers.deinit();

    for (boundary_bindings) |binding| {
        for (binding.prepared_queries) |prepared| {
            try appendQueryEvidence(arena, &queries, .{
                .producer_id = prepared.producer_id,
                .kind = prepared.kind,
                .use_id = binding.use_id,
            });
            if (binding.handling_kind != .symbolic) continue;
            const producer = if (boundary_obligation_producers.get(prepared.producer_id)) |existing|
                existing
            else blk: {
                next_boundary_obligation_producer = std.math.add(u32, next_boundary_obligation_producer, 1) catch
                    return error.SourceAccountingBoundaryObligationProducerIdOverflow;
                _ = try from_mlir.accountingEvidenceId(.obligation, next_boundary_obligation_producer);
                try boundary_obligation_producers.put(prepared.producer_id, next_boundary_obligation_producer);
                break :blk next_boundary_obligation_producer;
            };
            try appendCoveredEvidence(arena, &obligations, producer, binding.use_id);
        }
    }

    std.mem.sort(accounting.CoveredEvidence, obligations.items, {}, coveredEvidenceLessThan);
    std.mem.sort(accounting.QueryEvidence, queries.items, {}, queryEvidenceLessThan);

    const handlings = try arena.alloc(accounting.HandlingRecord, bindings.len + boundary_bindings.len);
    for (bindings, handlings[0..bindings.len]) |binding, *handling| {
        handling.* = .{
            .id = try accounting.namespacedHandlingId(.symbolic, binding.use_id),
            .use_id = binding.use_id,
            .kind = binding.handling_kind,
            .obligation_ids = try mapIds(arena, .obligation, binding.obligation_ids),
            .assumption_ids = try mapIds(arena, .assumption, binding.assumption_ids),
            .query_ids = try mapPreparedQueryIds(arena, binding.query_ids, actual_prepared_queries),
            .runtime_check_ids = try mapIds(arena, .runtime_check, binding.runtime_check_ids),
            .frame_result_ids = try mapIds(arena, .frame_result, binding.frame_result_ids),
            .state_effect_ids = try mapIds(arena, .state_effect, binding.state_effect_ids),
        };
    }
    for (boundary_bindings, handlings[bindings.len..]) |binding, *handling| {
        handling.* = .{
            .id = try accounting.namespacedHandlingId(.symbolic, binding.use_id),
            .use_id = binding.use_id,
            .kind = binding.handling_kind,
            .obligation_ids = if (binding.handling_kind == .symbolic)
                try mapBoundaryObligationIds(arena, binding.prepared_queries, boundary_obligation_producers)
            else
                &.{},
            .query_ids = if (binding.handling_kind == .verification_disabled)
                &.{}
            else
                try mapBoundaryQueryIds(arena, binding.prepared_queries),
        };
    }
    std.mem.sort(accounting.HandlingRecord, handlings, {}, struct {
        fn less(_: void, lhs: accounting.HandlingRecord, rhs: accounting.HandlingRecord) bool {
            return lhs.use_id < rhs.use_id;
        }
    }.less);

    result.evidence = .{
        .obligations = try obligations.toOwnedSlice(arena),
        .assumptions = try cloneCovered(arena, mlir_evidence.assumptions),
        .queries = try queries.toOwnedSlice(arena),
        .runtime_checks = try cloneCovered(arena, mlir_evidence.runtime_checks),
        .frame_results = try cloneValidated(arena, mlir_evidence.frame_results),
        .state_effects = try cloneValidated(arena, mlir_evidence.state_effects),
        .handlings = handlings,
    };
    return result;
}

fn validateBoundaryBindings(
    bindings: []const from_mlir.Binding,
    boundary_bindings: []const BoundaryBinding,
) !void {
    // Strictly increasing use and producer IDs are part of the adapter's
    // determinism contract and make duplicates fail closed. Do not replace
    // these checks with a permissive sort: sorting here would accept
    // non-canonical producer input and could conceal duplicate identities.
    var previous_use_id: ?accounting.UseId = null;
    for (boundary_bindings) |binding| {
        if (binding.use_id == 0) return error.SourceAccountingBoundaryUseIdZero;
        if (previous_use_id) |previous| {
            if (binding.use_id <= previous) return error.SourceAccountingBoundaryBindingsNotCanonical;
        }
        previous_use_id = binding.use_id;
        for (bindings) |existing| {
            if (existing.use_id == binding.use_id) return error.DuplicateSourceAccountingBoundaryBinding;
        }
        switch (binding.handling_kind) {
            .symbolic, .assumption_incorporated => {},
            .verification_disabled => {
                if (binding.prepared_queries.len != 0) {
                    return error.VerificationDisabledBoundaryHasPreparedQueries;
                }
            },
            else => return error.InvalidSourceAccountingBoundaryHandling,
        }
        var previous_producer_id: ?u32 = null;
        for (binding.prepared_queries) |prepared| {
            if (prepared.producer_id == 0) return error.SourceAccountingPreparedQueryProducerIdZero;
            if (previous_producer_id) |previous| {
                if (prepared.producer_id <= previous) {
                    return error.SourceAccountingBoundaryQueryProducerIdsNotCanonical;
                }
            }
            previous_producer_id = prepared.producer_id;
            _ = try from_mlir.accountingEvidenceId(.query, prepared.producer_id);
            const expected_role: accounting.UseRole = switch (binding.handling_kind) {
                .symbolic => .proof_target,
                .assumption_incorporated => .assumption_context,
                .verification_disabled => unreachable,
                else => unreachable,
            };
            if (prepared.role != expected_role) return error.SourceAccountingBoundaryQueryRoleMismatch;
        }
    }
}

fn mapBoundaryQueryIds(
    allocator: std.mem.Allocator,
    prepared_queries: []const PreparedBoundaryQueryIdentity,
) ![]const accounting.EvidenceId {
    const result = try allocator.alloc(accounting.EvidenceId, prepared_queries.len);
    for (prepared_queries, result) |prepared, *id| {
        id.* = try from_mlir.accountingEvidenceId(.query, prepared.producer_id);
    }
    return result;
}

fn mapBoundaryObligationIds(
    allocator: std.mem.Allocator,
    prepared_queries: []const PreparedBoundaryQueryIdentity,
    producers: std.AutoHashMap(u32, u32),
) ![]const accounting.EvidenceId {
    const result = try allocator.alloc(accounting.EvidenceId, prepared_queries.len);
    for (prepared_queries, result) |prepared, *id| {
        const producer = producers.get(prepared.producer_id) orelse
            return error.MissingSourceAccountingBoundaryObligationProducer;
        id.* = try from_mlir.accountingEvidenceId(.obligation, producer);
    }
    return result;
}

fn appendCoveredEvidence(
    allocator: std.mem.Allocator,
    rows: *std.ArrayList(accounting.CoveredEvidence),
    producer_id: u32,
    use_id: accounting.UseId,
) !void {
    if (findCoveredRow(rows.items, producer_id)) |index| {
        try appendUseUnique(allocator, &rows.items[index].covered_use_ids, use_id);
        return;
    }
    const covered = try allocator.alloc(accounting.UseId, 1);
    covered[0] = use_id;
    try rows.append(allocator, .{
        .id = try from_mlir.accountingEvidenceId(.obligation, producer_id),
        .producer_id = producer_id,
        .covered_use_ids = covered,
    });
}

fn coveredEvidenceLessThan(_: void, lhs: accounting.CoveredEvidence, rhs: accounting.CoveredEvidence) bool {
    return lhs.id < rhs.id;
}

fn queryEvidenceLessThan(_: void, lhs: accounting.QueryEvidence, rhs: accounting.QueryEvidence) bool {
    return lhs.id < rhs.id;
}

fn mapPreparedQueryIds(
    allocator: std.mem.Allocator,
    formal_query_ids: []const obligation.Id,
    actual_prepared_queries: ?[]const PreparedQueryIdentity,
) ![]const accounting.EvidenceId {
    if (actual_prepared_queries == null) return mapIds(allocator, .query, formal_query_ids);
    var mapped: std.ArrayList(accounting.EvidenceId) = .empty;
    for (actual_prepared_queries.?) |prepared| {
        if (!containsId(formal_query_ids, prepared.formal_query_id)) continue;
        try mapped.append(allocator, try from_mlir.accountingEvidenceId(.query, prepared.producer_id));
    }
    return mapped.toOwnedSlice(allocator);
}

const QueryEvidenceInput = struct {
    producer_id: u32,
    kind: accounting.QueryKind,
    use_id: accounting.UseId,
};

fn appendQueryEvidence(
    allocator: std.mem.Allocator,
    rows: *std.ArrayList(accounting.QueryEvidence),
    input: QueryEvidenceInput,
) !void {
    if (findQueryRow(rows.items, input.producer_id)) |index| {
        if (rows.items[index].kind != input.kind) return error.SourceAccountingPreparedQueryKindConflict;
        try appendUseUnique(allocator, &rows.items[index].covered_use_ids, input.use_id);
        return;
    }
    const covered = try allocator.alloc(accounting.UseId, 1);
    covered[0] = input.use_id;
    try rows.append(allocator, .{
        .id = try from_mlir.accountingEvidenceId(.query, input.producer_id),
        .producer_id = input.producer_id,
        .kind = input.kind,
        .covered_use_ids = covered,
    });
}

fn validatePreparedQueries(set: obligation.ObligationSet, rows: []const PreparedQueryIdentity) !void {
    for (rows, 0..) |row, index| {
        if (row.producer_id == 0) return error.SourceAccountingPreparedQueryProducerIdZero;
        if (index != 0 and row.producer_id <= rows[index - 1].producer_id) {
            return error.SourceAccountingPreparedQueryProducerIdsNotCanonical;
        }
        if (findQuery(set, row.formal_query_id) == null) {
            return error.UnknownSourceAccountingPreparedQuery;
        }
        _ = try from_mlir.accountingEvidenceId(.query, row.producer_id);
    }
}

fn queryKind(kind: obligation.VerificationQueryKind) accounting.QueryKind {
    return switch (kind) {
        .base, .obligation => .obligation,
        .loop_invariant_step => .loop_invariant_step,
        .loop_body_safety => .loop_body_safety,
        .loop_invariant_post => .loop_invariant_post,
        .guard_satisfy => .guard_satisfy,
        .guard_violate => .guard_violate,
    };
}

fn accountingQueryKind(kind: prepared_query.QueryKind) accounting.QueryKind {
    return switch (kind) {
        .obligation => .obligation,
        .loop_invariant_step => .loop_invariant_step,
        .loop_body_safety => .loop_body_safety,
        .loop_invariant_post => .loop_invariant_post,
        .guard_satisfy => .guard_satisfy,
        .guard_violate => .guard_violate,
    };
}

fn dedupeSortedStrings(values: [][]const u8) usize {
    if (values.len == 0) return 0;
    var write_index: usize = 1;
    for (values[1..]) |value| {
        if (std.mem.eql(u8, value, values[write_index - 1])) continue;
        values[write_index] = value;
        write_index += 1;
    }
    return write_index;
}

fn mapIds(allocator: std.mem.Allocator, namespace: from_mlir.Namespace, ids: []const obligation.Id) ![]const accounting.EvidenceId {
    const mapped = try allocator.alloc(accounting.EvidenceId, ids.len);
    for (ids, mapped) |id, *value| value.* = try from_mlir.accountingEvidenceId(namespace, id);
    return mapped;
}

fn cloneCovered(allocator: std.mem.Allocator, rows: []const accounting.CoveredEvidence) ![]const accounting.CoveredEvidence {
    const cloned = try allocator.dupe(accounting.CoveredEvidence, rows);
    for (cloned) |*row| row.covered_use_ids = try allocator.dupe(accounting.UseId, row.covered_use_ids);
    return cloned;
}

fn cloneValidated(allocator: std.mem.Allocator, rows: []const accounting.ValidationEvidence) ![]const accounting.ValidationEvidence {
    const cloned = try allocator.dupe(accounting.ValidationEvidence, rows);
    for (cloned) |*row| row.covered_use_ids = try allocator.dupe(accounting.UseId, row.covered_use_ids);
    return cloned;
}

fn appendUseUnique(allocator: std.mem.Allocator, values: *[]const accounting.UseId, use_id: accounting.UseId) !void {
    for (values.*) |value| if (value == use_id) return;
    const expanded = try allocator.alloc(accounting.UseId, values.len + 1);
    @memcpy(expanded[0..values.len], values.*);
    expanded[values.len] = use_id;
    values.* = expanded;
}

fn findQueryRow(rows: []const accounting.QueryEvidence, producer_id: u32) ?usize {
    for (rows, 0..) |row, index| if (row.producer_id == producer_id) return index;
    return null;
}

fn findCoveredRow(rows: []const accounting.CoveredEvidence, producer_id: u32) ?usize {
    for (rows, 0..) |row, index| if (row.producer_id == producer_id) return index;
    return null;
}

fn findQuery(set: obligation.ObligationSet, id: obligation.Id) ?obligation.VerificationQuery {
    for (set.queries) |row| if (row.id == id) return row;
    return null;
}

fn containsId(ids: []const obligation.Id, id: obligation.Id) bool {
    for (ids) |candidate| if (candidate == id) return true;
    return false;
}

test "prepared query kind mapping is exhaustive" {
    inline for (std.meta.fields(obligation.VerificationQueryKind)) |field| {
        const kind: obligation.VerificationQueryKind = @enumFromInt(field.value);
        _ = queryKind(kind);
    }
}

test "prepared verifier identity preserves row order and repeated formal identity" {
    const rows = [_]prepared_query.Row{
        .{ .kind = .obligation, .function_name = "run" },
        .{ .kind = .obligation, .function_name = "run", .match_status = .matched, .query_id = 7 },
        .{ .kind = .loop_invariant_step, .function_name = "run", .match_status = .matched, .query_id = 7 },
    };
    var identity = try collectPreparedIdentity(std.testing.allocator, &rows);
    defer identity.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), identity.queries.len);
    try std.testing.expectEqual(@as(u32, 2), identity.queries[0].producer_id);
    try std.testing.expectEqual(@as(u32, 3), identity.queries[1].producer_id);
    try std.testing.expectEqual(@as(u32, 7), identity.queries[0].formal_query_id);
    try std.testing.expectEqual(@as(u32, 7), identity.queries[1].formal_query_id);
    try std.testing.expectEqual(accounting.QueryKind.obligation, identity.queries[0].kind);
    try std.testing.expectEqual(accounting.QueryKind.loop_invariant_step, identity.queries[1].kind);
    try std.testing.expectEqual(@as(usize, 1), identity.runtime_function_names.len);
    try std.testing.expectEqualStrings("run", identity.runtime_function_names[0]);
}

test "prepared verifier identity preserves symbolic call-boundary identity" {
    const rows = [_]prepared_query.Row{
        .{
            .kind = .obligation,
            .function_name = "run",
            .file = "main.ora",
            .line = 8,
            .column = 16,
            .boundary_role = .proof_target,
            .boundary_callee_name = "checked",
            .boundary_source_fact_id = 41,
        },
        .{
            .kind = .obligation,
            .function_name = "run",
            .file = "main.ora",
            .line = 8,
            .column = 16,
            .boundary_role = .assumption_context,
            .boundary_callee_name = "checked",
            .boundary_source_fact_id = 59,
        },
    };
    var identity = try collectPreparedIdentity(std.testing.allocator, &rows);
    defer identity.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), identity.boundary_queries.len);
    try std.testing.expectEqual(accounting.UseRole.proof_target, identity.boundary_queries[0].role);
    try std.testing.expectEqual(accounting.UseRole.assumption_context, identity.boundary_queries[1].role);
    try std.testing.expectEqualStrings("main.ora", identity.boundary_queries[0].file);
    try std.testing.expectEqualStrings("checked", identity.boundary_queries[0].callee_name);
    try std.testing.expectEqual(@as(u32, 41), identity.boundary_queries[0].source_fact_id);
}

test "symbolic call-boundary adapter binds actual queries and proof obligation" {
    const prepared = [_]PreparedBoundaryQueryIdentity{
        .{
            .producer_id = 4,
            .kind = .obligation,
            .role = .proof_target,
            .file = "main.ora",
            .line = 8,
            .column = 16,
            .callee_name = "checked",
            .source_fact_id = 41,
        },
    };
    const boundary_bindings = [_]BoundaryBinding{.{
        .use_id = 9,
        .handling_kind = .symbolic,
        .prepared_queries = &prepared,
    }};
    var result = try bindPreparedQueriesWithBoundaries(
        std.testing.allocator,
        .{},
        &.{},
        .{},
        null,
        &boundary_bindings,
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.evidence.obligations.len);
    try std.testing.expectEqual(@as(usize, 1), result.evidence.queries.len);
    try std.testing.expectEqual(@as(usize, 1), result.evidence.handlings.len);
    try std.testing.expectEqual(accounting.HandlingKind.symbolic, result.evidence.handlings[0].kind);
    try std.testing.expectEqualSlices(
        accounting.EvidenceId,
        &.{result.evidence.obligations[0].id},
        result.evidence.handlings[0].obligation_ids,
    );
    try std.testing.expectEqualSlices(
        accounting.EvidenceId,
        &.{result.evidence.queries[0].id},
        result.evidence.handlings[0].query_ids,
    );
}

test "adapter preserves actual prepared-query identity" {
    const queries = [_]obligation.VerificationQuery{.{
        .id = 7,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .loop_invariant_step,
    }};
    const query_ids = [_]obligation.Id{7};
    const bindings = [_]from_mlir.Binding{.{
        .use_id = 9,
        .handling_kind = .symbolic,
        .query_ids = &query_ids,
    }};
    const set: obligation.ObligationSet = .{ .queries = &queries };
    var mlir_result = try from_mlir.collect(std.testing.allocator, set, &bindings, .{});
    defer mlir_result.deinit();
    const prepared = [_]PreparedQueryIdentity{.{
        .producer_id = 11,
        .formal_query_id = 7,
        .kind = .loop_invariant_step,
    }};
    var result = try bindPreparedQueries(std.testing.allocator, set, &bindings, mlir_result.evidence, &prepared);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.evidence.queries.len);
    try std.testing.expectEqual(@as(u32, 11), result.evidence.queries[0].producer_id);
    try std.testing.expectEqual(
        try from_mlir.accountingEvidenceId(.query, 11),
        result.evidence.queries[0].id,
    );
    try std.testing.expectEqual(accounting.QueryKind.loop_invariant_step, result.evidence.queries[0].kind);
    try std.testing.expectEqualSlices(accounting.UseId, &.{9}, result.evidence.queries[0].covered_use_ids);

    const bad_ids = [_]obligation.Id{8};
    const bad_bindings = [_]from_mlir.Binding{.{
        .use_id = 9,
        .handling_kind = .symbolic,
        .query_ids = &bad_ids,
    }};
    try std.testing.expectError(
        error.UnknownSourceAccountingPreparedQuery,
        bindPreparedQueries(std.testing.allocator, set, &bad_bindings, .{}, &prepared),
    );
}

test "unmatched formal query is not reported as prepared evidence" {
    const queries = [_]obligation.VerificationQuery{.{
        .id = 7,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
    }};
    const query_ids = [_]obligation.Id{7};
    const bindings = [_]from_mlir.Binding{.{
        .use_id = 9,
        .handling_kind = .symbolic,
        .query_ids = &query_ids,
    }};
    var result = try bindPreparedQueries(
        std.testing.allocator,
        .{ .queries = &queries },
        &bindings,
        .{},
        &.{},
    );
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.evidence.queries.len);
    try std.testing.expectEqual(@as(usize, 0), result.evidence.handlings[0].query_ids.len);
}

test "one formal invariant identity preserves obligation and step query rows" {
    const queries = [_]obligation.VerificationQuery{.{
        .id = 7,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
    }};
    const query_ids = [_]obligation.Id{7};
    const bindings = [_]from_mlir.Binding{.{
        .use_id = 9,
        .handling_kind = .symbolic,
        .query_ids = &query_ids,
    }};
    const prepared = [_]PreparedQueryIdentity{
        .{ .producer_id = 3, .formal_query_id = 7, .kind = .obligation },
        .{ .producer_id = 4, .formal_query_id = 7, .kind = .loop_invariant_step },
    };
    var result = try bindPreparedQueries(
        std.testing.allocator,
        .{ .queries = &queries },
        &bindings,
        .{},
        &prepared,
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.evidence.queries.len);
    try std.testing.expectEqual(accounting.QueryKind.obligation, result.evidence.queries[0].kind);
    try std.testing.expectEqual(accounting.QueryKind.loop_invariant_step, result.evidence.queries[1].kind);
    try std.testing.expectEqualSlices(
        accounting.EvidenceId,
        &.{
            try from_mlir.accountingEvidenceId(.query, 3),
            try from_mlir.accountingEvidenceId(.query, 4),
        },
        result.evidence.handlings[0].query_ids,
    );
}
