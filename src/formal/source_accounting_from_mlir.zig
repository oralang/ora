//! Projection of source-linked MLIR obligation rows into accounting evidence.
//!
//! The adapter never matches by formula text, diagnostic text, or source line.
//! Its bindings are the identity seam that the compiler repair will populate
//! on formal operations. With no bindings it truthfully produces no evidence.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");
const obligation = @import("obligation.zig");

pub const ProducerInventory = struct {
    runtime_check_ids: []const u32 = &.{},
    state_effect_ids: []const u32 = &.{},
};

pub const Binding = struct {
    use_id: accounting.UseId,
    handling_kind: accounting.HandlingKind,
    obligation_ids: []const obligation.Id = &.{},
    assumption_ids: []const obligation.Id = &.{},
    query_ids: []const obligation.Id = &.{},
    runtime_check_ids: []const obligation.Id = &.{},
    frame_result_ids: []const obligation.Id = &.{},
    state_effect_ids: []const obligation.Id = &.{},
};

/// Symbolic producer namespaces accepted by this adapter. Concrete evaluator
/// evidence owns a separate namespace and cannot enter through MLIR bindings.
pub const Namespace = enum(u4) {
    obligation = @intFromEnum(accounting.EvidenceNamespace.obligation),
    assumption = @intFromEnum(accounting.EvidenceNamespace.assumption),
    query = @intFromEnum(accounting.EvidenceNamespace.query),
    runtime_check = @intFromEnum(accounting.EvidenceNamespace.runtime_check),
    frame_result = @intFromEnum(accounting.EvidenceNamespace.frame_result),
    state_effect = @intFromEnum(accounting.EvidenceNamespace.state_effect),
};

pub fn accountingEvidenceId(namespace: Namespace, producer_id: obligation.Id) !accounting.EvidenceId {
    const shared_namespace: accounting.EvidenceNamespace = @enumFromInt(@intFromEnum(namespace));
    return accounting.namespacedEvidenceId(shared_namespace, producer_id);
}

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    evidence: accounting.SymbolicEvidence,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn collect(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    bindings: []const Binding,
    producers: ProducerInventory,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .evidence = .{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    try validateBindingOrderAndReferences(set, bindings, producers);
    result.evidence = .{
        .obligations = try collectCovered(arena, set, bindings, .obligation),
        .assumptions = try collectCovered(arena, set, bindings, .assumption),
        .runtime_checks = try collectCovered(arena, set, bindings, .runtime_check),
        .frame_results = try collectValidated(arena, set, bindings, .frame_result),
        .state_effects = try collectProducerValidated(arena, bindings, .state_effect, producers.state_effect_ids),
    };
    return result;
}

fn validateBindingOrderAndReferences(
    set: obligation.ObligationSet,
    bindings: []const Binding,
    producers: ProducerInventory,
) !void {
    var previous_use_id: ?accounting.UseId = null;
    for (bindings) |binding| {
        if (previous_use_id) |previous| {
            if (binding.use_id <= previous) return error.SourceAccountingBindingsNotCanonical;
        }
        previous_use_id = binding.use_id;
        for (binding.obligation_ids) |id| if (findObligation(set, id) == null) return error.UnknownSourceAccountingObligation;
        for (binding.assumption_ids) |id| if (findAssumption(set, id) == null) return error.UnknownSourceAccountingAssumption;
        for (binding.query_ids) |id| if (findQuery(set, id) == null) return error.UnknownSourceAccountingQuery;
        for (binding.runtime_check_ids) |id| if (!containsId(producers.runtime_check_ids, id))
            return error.UnknownSourceAccountingRuntimeCheck;
        for (binding.frame_result_ids) |id| {
            const item = findObligation(set, id) orelse return error.UnknownSourceAccountingFrameResult;
            if (item.kind != .effect_frame) return error.InvalidSourceAccountingFrameResult;
        }
        for (binding.state_effect_ids) |id| if (!containsId(producers.state_effect_ids, id)) return error.UnknownSourceAccountingStateEffect;
    }
}

fn collectCovered(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    bindings: []const Binding,
    namespace: Namespace,
) ![]const accounting.CoveredEvidence {
    var rows: std.ArrayList(accounting.CoveredEvidence) = .empty;
    for (bindings) |binding| {
        const ids = idsFor(binding, namespace);
        for (ids) |producer_id| {
            if (findCoveredRow(rows.items, producer_id)) |index| {
                try appendUseUnique(allocator, &rows.items[index].covered_use_ids, binding.use_id);
                continue;
            }
            const covered = try allocator.alloc(accounting.UseId, 1);
            covered[0] = binding.use_id;
            try rows.append(allocator, .{
                .id = try accountingEvidenceId(namespace, producer_id),
                .producer_id = producer_id,
                .covered_use_ids = covered,
            });
        }
    }
    _ = set;
    std.mem.sort(accounting.CoveredEvidence, rows.items, {}, struct {
        fn less(_: void, lhs: accounting.CoveredEvidence, rhs: accounting.CoveredEvidence) bool {
            return lhs.id < rhs.id;
        }
    }.less);
    return rows.toOwnedSlice(allocator);
}

fn collectValidated(
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    bindings: []const Binding,
    namespace: Namespace,
) ![]const accounting.ValidationEvidence {
    var rows: std.ArrayList(accounting.ValidationEvidence) = .empty;
    for (bindings) |binding| {
        for (idsFor(binding, namespace)) |producer_id| {
            if (findValidationRow(rows.items, producer_id)) |index| {
                try appendUseUnique(allocator, &rows.items[index].covered_use_ids, binding.use_id);
                continue;
            }
            const covered = try allocator.alloc(accounting.UseId, 1);
            covered[0] = binding.use_id;
            try rows.append(allocator, .{
                .id = try accountingEvidenceId(namespace, producer_id),
                .producer_id = producer_id,
                .covered_use_ids = covered,
                // This states that the producer emitted the required modeled
                // row. Solver success remains outside source accounting.
                .valid = findObligation(set, producer_id) != null,
            });
        }
    }
    std.mem.sort(accounting.ValidationEvidence, rows.items, {}, struct {
        fn less(_: void, lhs: accounting.ValidationEvidence, rhs: accounting.ValidationEvidence) bool {
            return lhs.id < rhs.id;
        }
    }.less);
    return rows.toOwnedSlice(allocator);
}

fn collectProducerValidated(
    allocator: std.mem.Allocator,
    bindings: []const Binding,
    namespace: Namespace,
    producer_ids: []const u32,
) ![]const accounting.ValidationEvidence {
    var rows: std.ArrayList(accounting.ValidationEvidence) = .empty;
    for (bindings) |binding| {
        for (idsFor(binding, namespace)) |producer_id| {
            if (findValidationRow(rows.items, producer_id)) |index| {
                try appendUseUnique(allocator, &rows.items[index].covered_use_ids, binding.use_id);
                continue;
            }
            const covered = try allocator.alloc(accounting.UseId, 1);
            covered[0] = binding.use_id;
            try rows.append(allocator, .{
                .id = try accountingEvidenceId(namespace, producer_id),
                .producer_id = producer_id,
                .covered_use_ids = covered,
                .valid = containsId(producer_ids, producer_id),
            });
        }
    }
    std.mem.sort(accounting.ValidationEvidence, rows.items, {}, struct {
        fn less(_: void, lhs: accounting.ValidationEvidence, rhs: accounting.ValidationEvidence) bool {
            return lhs.id < rhs.id;
        }
    }.less);
    return rows.toOwnedSlice(allocator);
}

fn idsFor(binding: Binding, namespace: Namespace) []const obligation.Id {
    return switch (namespace) {
        .obligation => binding.obligation_ids,
        .assumption => binding.assumption_ids,
        .query => binding.query_ids,
        .runtime_check => binding.runtime_check_ids,
        .frame_result => binding.frame_result_ids,
        .state_effect => binding.state_effect_ids,
    };
}

fn appendUseUnique(allocator: std.mem.Allocator, values: *[]const accounting.UseId, use_id: accounting.UseId) !void {
    for (values.*) |value| if (value == use_id) return;
    const expanded = try allocator.alloc(accounting.UseId, values.len + 1);
    @memcpy(expanded[0..values.len], values.*);
    expanded[values.len] = use_id;
    values.* = expanded;
}

fn findCoveredRow(rows: []const accounting.CoveredEvidence, producer_id: obligation.Id) ?usize {
    for (rows, 0..) |row, index| if (row.producer_id == producer_id) return index;
    return null;
}

fn findValidationRow(rows: []const accounting.ValidationEvidence, producer_id: obligation.Id) ?usize {
    for (rows, 0..) |row, index| if (row.producer_id == producer_id) return index;
    return null;
}

fn findObligation(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Obligation {
    for (set.obligations) |row| if (row.id == id) return row;
    return null;
}

fn findAssumption(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Assumption {
    for (set.assumptions) |row| if (row.id == id) return row;
    return null;
}

fn findQuery(set: obligation.ObligationSet, id: obligation.Id) ?obligation.VerificationQuery {
    for (set.queries) |row| if (row.id == id) return row;
    return null;
}

fn containsId(ids: []const u32, id: u32) bool {
    for (ids) |candidate| if (candidate == id) return true;
    return false;
}

test "evidence namespaces cannot alias producer ids" {
    const obligation_id = try accountingEvidenceId(.obligation, 7);
    const query_id = try accountingEvidenceId(.query, 7);
    try std.testing.expect(obligation_id != query_id);
    try std.testing.expectError(error.SourceAccountingProducerIdOverflow, accountingEvidenceId(.query, 0x1000_0000));
}
