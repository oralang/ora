//! Production assembly of source-accounting inputs.
//!
//! This module is the only production seam that joins package, evaluator,
//! runtime-owner, MLIR, and prepared-query identities before entering the
//! compiler-kernel registry.  It owns no authorization policy.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");
const obligation = @import("obligation.zig");
const from_package = @import("source_accounting_from_package.zig");
const from_runtime = @import("source_accounting_from_runtime.zig");
const from_comptime = @import("source_accounting_from_comptime.zig");
const from_mlir = @import("source_accounting_from_mlir.zig");
const from_z3 = @import("source_accounting_from_z3.zig");
const obligation_from_mlir = @import("obligation_from_mlir.zig");
const registry = @import("kernel/registry.zig");

pub const PreparedQueryIdentity = from_z3.PreparedQueryIdentity;
pub const PreparedIdentity = from_z3.PreparedIdentityView;

pub const Result = struct {
    session: registry.SourceAccountingSession,
    finished: registry.SourceAccountingSession.FinishedView,

    pub fn deinit(self: *Result) void {
        self.session.deinit();
        self.* = undefined;
    }
};

pub fn run(
    allocator: std.mem.Allocator,
    compiler_db: anytype,
    package_id: anytype,
    root_module_id: anytype,
    const_eval: anytype,
    formal: *const obligation_from_mlir.CollectResult,
    mode: accounting.CompilationMode,
    prepared: ?PreparedIdentity,
) !Result {
    if (mode == .unverified_emit) {
        if (prepared != null) return error.SourceAccountingPreparedQueriesNotPermittedWhenVerificationDisabled;
    } else if (prepared == null) {
        return error.SourceAccountingPreparedQueriesRequired;
    }
    if (prepared) |identity| try validatePreparedIdentity(identity);

    var package = try from_package.collect(
        allocator,
        compiler_db,
        package_id,
        root_module_id,
    );
    defer package.deinit();

    var active_runtime_owners: std.ArrayList(obligation_from_mlir.RuntimeOwnerBinding) = .empty;
    defer active_runtime_owners.deinit(allocator);
    for (formal.runtime_owner_bindings) |owner| {
        const activation = std.meta.stringToEnum(accounting.TemplateActivation, owner.template_activation) orelse
            return error.UnknownSourceAccountingTemplateActivation;
        if (activation != .runtime_body) return error.InvalidRuntimeSourceAccountingActivation;
        const module = package.moduleForSourcePath(owner.module_path) orelse
            return error.UnknownRuntimeSourceAccountingModule;
        if (module.templateForOwner(owner.owner_key, activation) == null) continue;
        if (prepared) |identity| {
            const symbol = owner.symbol orelse return error.RuntimeSourceAccountingOwnerMissingSymbol;
            if (!containsString(identity.runtime_function_names, symbol)) continue;
        }
        try active_runtime_owners.append(allocator, owner);
    }

    var runtime = try from_runtime.collect(
        allocator,
        &package,
        active_runtime_owners.items,
        .{},
    );
    defer runtime.deinit();

    var comptime_result = try from_comptime.collectPackageFoldExpansions(
        allocator,
        &package,
        const_eval,
        .{
            .expansion = try nextId(accounting.ExpansionId, runtime.expansions.len),
            .use = try nextId(accounting.UseId, runtime.uses.len),
            .control_node = try nextId(accounting.ControlNodeId, runtime.control_nodes.len),
            .control_edge = try nextId(accounting.ControlEdgeId, runtime.control_edges.len),
        },
    );
    defer comptime_result.deinit();

    var symbolic_bindings = try from_runtime.collectSymbolicBindings(
        allocator,
        &package,
        &runtime,
        formal,
    );
    defer symbolic_bindings.deinit();

    const adapted_bindings = try adaptBindingsForMode(
        allocator,
        mode,
        package.inventory.typed_sites,
        runtime.uses,
        symbolic_bindings.bindings,
    );
    defer allocator.free(adapted_bindings);

    var mlir_evidence = try from_mlir.collect(
        allocator,
        formal.set,
        adapted_bindings,
        symbolic_bindings.producers,
    );
    defer mlir_evidence.deinit();

    var boundary_bindings = try collectBoundaryBindings(
        allocator,
        &compiler_db.sources,
        &package,
        comptime_result.expansions,
        comptime_result.uses,
        mode,
        if (prepared) |identity| identity.boundary_queries else &.{},
    );
    defer boundary_bindings.deinit();

    var symbolic_evidence = try from_z3.bindPreparedQueriesWithBoundaries(
        allocator,
        formal.set,
        adapted_bindings,
        mlir_evidence.evidence,
        if (prepared) |identity| identity.queries else null,
        boundary_bindings.bindings,
    );
    defer symbolic_evidence.deinit();

    const expansions = try concat(
        accounting.Expansion,
        allocator,
        runtime.expansions,
        comptime_result.expansions,
    );
    defer allocator.free(expansions);
    const uses = try concat(
        accounting.SourceFactUse,
        allocator,
        runtime.uses,
        comptime_result.uses,
    );
    defer allocator.free(uses);
    const control_nodes = try concat(
        accounting.ControlNode,
        allocator,
        runtime.control_nodes,
        comptime_result.control_nodes,
    );
    defer allocator.free(control_nodes);
    const control_edges = try concat(
        accounting.ControlEdge,
        allocator,
        runtime.control_edges,
        comptime_result.control_edges,
    );
    defer allocator.free(control_edges);

    var session = try registry.SourceAccountingSession.prepareFromSource(allocator, mode, .{
        .declared_sites = package.inventory.declared_sites,
        .typed_sites = package.inventory.typed_sites,
        .generated_fact_derivations = package.inventory.generated_fact_derivations,
        .owner_templates = package.inventory.owner_templates,
        .expansions = expansions,
        .uses = uses,
        .control_nodes = control_nodes,
        .control_edges = control_edges,
    });
    errdefer session.deinit();
    try session.bindComptimeEvidence(comptime_result.evidence);
    if (mode == .unverified_emit) {
        try session.bindVerificationDisabled(symbolic_evidence.evidence);
    } else {
        try session.bindSymbolicEvidence(symbolic_evidence.evidence);
    }
    const finished = try session.finishAccountingDecision();
    return .{ .session = session, .finished = finished };
}

fn adaptBindingsForMode(
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    typed_sites: []const accounting.TypedSite,
    uses: []const accounting.SourceFactUse,
    bindings: []const from_mlir.Binding,
) ![]const from_mlir.Binding {
    const adapted = try allocator.dupe(from_mlir.Binding, bindings);
    errdefer allocator.free(adapted);
    for (adapted) |*binding| {
        if (binding.handling_kind != .symbolic) continue;
        const use = findUse(uses, binding.use_id) orelse return error.UnknownSourceAccountingUse;
        if (use.role != .proof_target) return error.InvalidSymbolicSourceAccountingRole;
        const site = findTypedSite(typed_sites, use.site_id) orelse return error.UnknownSourceAccountingSite;
        const handling = try registry.sourceProofHandlingForMode(mode, site.origin, site.kind);
        if (handling == .symbolic) continue;
        binding.handling_kind = handling;
        binding.obligation_ids = &.{};
        binding.assumption_ids = &.{};
        binding.query_ids = &.{};
        binding.runtime_check_ids = &.{};
        binding.frame_result_ids = &.{};
        binding.state_effect_ids = &.{};
    }
    return adapted;
}

fn validatePreparedIdentity(identity: PreparedIdentity) !void {
    for (identity.queries, 0..) |query, index| {
        if (query.producer_id == 0) return error.SourceAccountingPreparedQueryProducerIdZero;
        if (index != 0 and query.producer_id <= identity.queries[index - 1].producer_id) {
            return error.SourceAccountingPreparedQueryProducerIdsNotCanonical;
        }
    }
    for (identity.boundary_queries, 0..) |query, index| {
        if (query.producer_id == 0) return error.SourceAccountingPreparedQueryProducerIdZero;
        if (index != 0 and query.producer_id <= identity.boundary_queries[index - 1].producer_id) {
            return error.SourceAccountingBoundaryQueryProducerIdsNotCanonical;
        }
        if (query.file.len == 0 or query.line == 0 or query.column == 0) {
            return error.SourceAccountingBoundaryQueryMissingLocation;
        }
        if (query.callee_name.len == 0) return error.SourceAccountingBoundaryQueryMissingCallee;
        switch (query.role) {
            .proof_target, .assumption_context => {},
            else => return error.InvalidSourceAccountingBoundaryQueryRole,
        }
    }
    for (identity.runtime_function_names, 0..) |name, index| {
        if (name.len == 0) return error.SourceAccountingRuntimeFunctionNameEmpty;
        if (index != 0 and std.mem.order(u8, identity.runtime_function_names[index - 1], name) != .lt) {
            return error.SourceAccountingRuntimeFunctionNamesNotCanonical;
        }
    }
}

const BoundaryBindings = struct {
    arena: std.heap.ArenaAllocator,
    bindings: []const from_z3.BoundaryBinding,

    fn deinit(self: *BoundaryBindings) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

fn collectBoundaryBindings(
    allocator: std.mem.Allocator,
    sources: anytype,
    package: *const from_package.Result,
    expansions: []const accounting.Expansion,
    uses: []const accounting.SourceFactUse,
    mode: accounting.CompilationMode,
    prepared_queries: []const from_z3.PreparedBoundaryQueryIdentity,
) !BoundaryBindings {
    var result: BoundaryBindings = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .bindings = &.{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();
    var bindings: std.ArrayList(from_z3.BoundaryBinding) = .empty;

    for (expansions) |expansion| {
        if (expansion.activation != .symbolic_call or expansion.disposition != .symbolic) continue;
        const template = package.template(expansion.template_id) orelse
            return error.UnknownSourceAccountingBoundaryTemplate;
        if (template.activation != .symbolic_call_boundary) {
            return error.InvalidSourceAccountingBoundaryTemplateActivation;
        }
        const callee_name = ownerFunctionName(template.owner_key) orelse
            return error.SourceAccountingBoundaryOwnerMissingFunction;
        const call_site = if (expansion.folded_call_site_chain.len == 0)
            return error.SourceAccountingBoundaryMissingCallSite
        else
            expansion.folded_call_site_chain[expansion.folded_call_site_chain.len - 1];
        const module = moduleForCanonicalPath(package, call_site.file) orelse
            return error.UnknownSourceAccountingBoundaryCallSiteModule;
        const location = sources.lineColumn(.{
            .file_id = module.file_id,
            .range = .empty(call_site.start),
        });

        for (uses) |use| {
            if (use.expansion_id != expansion.id) continue;
            const site = findTypedSite(package.inventory.typed_sites, use.site_id) orelse
                return error.UnknownSourceAccountingBoundarySite;
            const source_fact_id = site.source_fact_id orelse
                return error.SourceAccountingBoundarySiteMissingSourceFactId;
            const handling_kind: accounting.HandlingKind = if (mode == .unverified_emit)
                .verification_disabled
            else switch (use.role) {
                .proof_target => .symbolic,
                .assumption_context => .assumption_incorporated,
                else => return error.InvalidSourceAccountingBoundaryUseRole,
            };
            var matched: std.ArrayList(from_z3.PreparedBoundaryQueryIdentity) = .empty;
            if (mode != .unverified_emit) {
                for (prepared_queries) |query| {
                    if (query.role != use.role or
                        query.line != location.line or
                        query.column != location.column or
                        query.source_fact_id != source_fact_id or
                        !sourcePathMatches(query.file, call_site.file) or
                        !calleeMatchesOwner(query.callee_name, callee_name)) continue;
                    try matched.append(arena, query);
                }
            }
            try bindings.append(arena, .{
                .use_id = use.id,
                .handling_kind = handling_kind,
                .prepared_queries = try matched.toOwnedSlice(arena),
            });
        }
    }
    std.mem.sort(from_z3.BoundaryBinding, bindings.items, {}, struct {
        fn less(_: void, lhs: from_z3.BoundaryBinding, rhs: from_z3.BoundaryBinding) bool {
            return lhs.use_id < rhs.use_id;
        }
    }.less);
    result.bindings = try bindings.toOwnedSlice(arena);
    return result;
}

fn moduleForCanonicalPath(package: *const from_package.Result, path: []const u8) ?from_package.ModuleInventory {
    for (package.modules) |module| {
        if (std.mem.eql(u8, module.canonical_path, path)) return module;
    }
    return null;
}

fn ownerFunctionName(owner_key: []const u8) ?[]const u8 {
    const markers = [_][]const u8{ "/function:", "/trait_method:" };
    var best: ?usize = null;
    var best_marker_len: usize = 0;
    for (markers) |marker| {
        if (std.mem.lastIndexOf(u8, owner_key, marker)) |index| {
            if (best == null or index > best.?) {
                best = index;
                best_marker_len = marker.len;
            }
        }
    }
    const start = (best orelse return null) + best_marker_len;
    const remainder = owner_key[start..];
    const end = std.mem.indexOfScalar(u8, remainder, '/') orelse remainder.len;
    if (end == 0) return null;
    return remainder[0..end];
}

fn calleeMatchesOwner(callee_name: []const u8, owner_name: []const u8) bool {
    const unspecialized = if (std.mem.indexOf(u8, callee_name, "__")) |index|
        callee_name[0..index]
    else
        callee_name;
    const final_component = if (std.mem.lastIndexOfScalar(u8, unspecialized, '.')) |index|
        unspecialized[index + 1 ..]
    else
        unspecialized;
    return std.mem.eql(u8, final_component, owner_name);
}

fn sourcePathMatches(verifier_path: []const u8, canonical_path: []const u8) bool {
    if (std.mem.eql(u8, verifier_path, canonical_path)) return true;
    if (!std.mem.endsWith(u8, verifier_path, canonical_path)) return false;
    const prefix_len = verifier_path.len - canonical_path.len;
    return prefix_len != 0 and (verifier_path[prefix_len - 1] == '/' or verifier_path[prefix_len - 1] == '\\');
}

fn nextId(comptime T: type, count: usize) !T {
    const narrowed = std.math.cast(T, count) orelse return error.SourceAccountingIdOverflow;
    return std.math.add(T, 1, narrowed) catch return error.SourceAccountingIdOverflow;
}

fn concat(comptime T: type, allocator: std.mem.Allocator, lhs: []const T, rhs: []const T) ![]T {
    const len = std.math.add(usize, lhs.len, rhs.len) catch return error.SourceAccountingInventoryTooLarge;
    const result = try allocator.alloc(T, len);
    @memcpy(result[0..lhs.len], lhs);
    @memcpy(result[lhs.len..], rhs);
    return result;
}

fn containsString(values: []const []const u8, needle: []const u8) bool {
    var low: usize = 0;
    var high = values.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        switch (std.mem.order(u8, values[mid], needle)) {
            .lt => low = mid + 1,
            .gt => high = mid,
            .eq => return true,
        }
    }
    return false;
}

fn findUse(uses: []const accounting.SourceFactUse, id: accounting.UseId) ?accounting.SourceFactUse {
    for (uses) |use| if (use.id == id) return use;
    return null;
}

fn findTypedSite(sites: []const accounting.TypedSite, id: accounting.SiteId) ?accounting.TypedSite {
    for (sites) |site| if (site.id == id) return site;
    return null;
}

test "prepared source-accounting identity is canonical" {
    try validatePreparedIdentity(.{
        .queries = &.{
            .{ .producer_id = 1, .formal_query_id = 4, .kind = .obligation },
            .{ .producer_id = 2, .formal_query_id = 4, .kind = .loop_invariant_step },
        },
        .runtime_function_names = &.{ "alpha", "omega" },
    });
    try std.testing.expectError(
        error.SourceAccountingPreparedQueryProducerIdsNotCanonical,
        validatePreparedIdentity(.{
            .queries = &.{
                .{ .producer_id = 4, .formal_query_id = 1, .kind = .obligation },
                .{ .producer_id = 4, .formal_query_id = 1, .kind = .loop_invariant_step },
            },
            .runtime_function_names = &.{},
        }),
    );
    try std.testing.expectError(
        error.SourceAccountingRuntimeFunctionNamesNotCanonical,
        validatePreparedIdentity(.{ .queries = &.{}, .runtime_function_names = &.{ "omega", "alpha" } }),
    );
}
