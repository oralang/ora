//! Package-wide source accounting for compile-time fold attempts.
//!
//! The evaluator publishes a deterministic fold ledger with structural source
//! identities. This adapter joins that ledger to the independent package-wide
//! semantic inventory. It never scans call expressions, matches callee names,
//! re-evaluates predicates, or derives source facts from range containment.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");

pub const IdStarts = struct {
    expansion: accounting.ExpansionId = 1,
    use: accounting.UseId = 1,
    control_node: accounting.ControlNodeId = 1,
    control_edge: accounting.ControlEdgeId = 1,
};

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    expansions: []const accounting.Expansion,
    uses: []const accounting.SourceFactUse,
    control_nodes: []const accounting.ControlNode,
    control_edges: []const accounting.ControlEdge,
    evidence: accounting.ComptimeEvidence,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const InvocationExpansion = struct {
    invocation_id: u32,
    nearest_expansion_id: ?accounting.ExpansionId,
};

const Instantiation = struct {
    first_use_id: accounting.UseId,
    node_ids: []const accounting.ControlNodeId,
};

const TypedSiteIndex = struct {
    rows: []const accounting.TypedSite,

    fn init(rows: []const accounting.TypedSite) !TypedSiteIndex {
        if (rows.len < 2) return .{ .rows = rows };
        for (rows[1..], rows[0..rows.len -| 1]) |current, previous| {
            if (current.id == previous.id) return error.DuplicateComptimeTypedSiteIdentity;
            if (current.id < previous.id) return error.NonCanonicalComptimeTypedSiteOrder;
        }
        return .{ .rows = rows };
    }

    fn get(self: TypedSiteIndex, id: accounting.SiteId) ?accounting.TypedSite {
        var low: usize = 0;
        var high: usize = self.rows.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const row = self.rows[middle];
            if (row.id < id) {
                low = middle + 1;
            } else if (row.id > id) {
                high = middle;
            } else {
                return row;
            }
        }
        return null;
    }
};

/// Compile-time checked adapter contract:
///
/// - `package` exposes `inventory.typed_sites` plus `module`, `moduleForFile`,
///   and `template`; returned module rows expose `file_id`, `canonical_path`,
///   and `templateForOwner`.
/// - `const_eval.formal_folds` is the evaluator-owned fold ledger. Its rows
///   expose invocation/parent IDs, activation/disposition, caller/callee
///   module and file IDs, owner keys, call-site chains, generic/trait identity,
///   and formal trace events as defined by `sema.model.ComptimeFoldEvidence`.
///
/// These requirements are documented here because importing the concrete
/// package adapter would introduce a compiler-module cycle. Missing fields or
/// incompatible field types are compile errors at the call site.
pub fn collectPackageFoldExpansions(
    allocator: std.mem.Allocator,
    package: anytype,
    const_eval: anytype,
    starts: IdStarts,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .expansions = &.{},
        .uses = &.{},
        .control_nodes = &.{},
        .control_edges = &.{},
        .evidence = .{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();
    // The package adapter emits typed sites in strict global-ID order. Check
    // that contract once so binary lookup cannot turn malformed input into a
    // silent missing-site result.
    const typed_site_index = try TypedSiteIndex.init(package.inventory.typed_sites);

    const order = try arena.alloc(usize, const_eval.formal_folds.len);
    for (order, 0..) |*index, value| index.* = value;
    std.mem.sort(usize, order, const_eval.formal_folds, struct {
        fn less(folds: @TypeOf(const_eval.formal_folds), lhs: usize, rhs: usize) bool {
            return folds[lhs].invocation_id < folds[rhs].invocation_id;
        }
    }.less);

    var expansions: std.ArrayList(accounting.Expansion) = .empty;
    var uses: std.ArrayList(accounting.SourceFactUse) = .empty;
    var nodes: std.ArrayList(accounting.ControlNode) = .empty;
    var edges: std.ArrayList(accounting.ControlEdge) = .empty;
    var folds: std.ArrayList(accounting.FoldRecord) = .empty;
    var predicates: std.ArrayList(accounting.PredicateEvent) = .empty;
    var handlings: std.ArrayList(accounting.HandlingRecord) = .empty;
    var invocation_expansions: std.ArrayList(InvocationExpansion) = .empty;

    var next_expansion_id = starts.expansion;
    var next_use_id = starts.use;
    var next_node_id = starts.control_node;
    var next_edge_id = starts.control_edge;

    for (order, 0..) |fold_index, ordered_index| {
        const formal_fold = const_eval.formal_folds[fold_index];
        if (ordered_index != 0 and
            const_eval.formal_folds[order[ordered_index - 1]].invocation_id == formal_fold.invocation_id)
        {
            return error.DuplicateComptimeFoldInvocationIdentity;
        }

        const parent_expansion_id = if (formal_fold.parent_invocation_id) |parent_invocation_id| blk: {
            const parent = invocationExpansionById(invocation_expansions.items, parent_invocation_id) orelse
                return error.UnknownComptimeFoldParentInvocation;
            break :blk parent.nearest_expansion_id;
        } else null;

        const callee_module = package.module(formal_fold.callee_module_id) orelse
            return error.UnknownComptimeFoldCalleeModule;
        if (callee_module.file_id != formal_fold.callee_file_id) return error.ComptimeFoldCalleeFileMismatch;
        const comptime_template_id = callee_module.templateForOwner(
            formal_fold.callee_owner_key,
            .comptime_body,
        ) orelse {
            // A fold whose callee owns no formal source fact contributes no
            // accounting expansion. Retain its ancestry so a nested formal
            // callee links to the nearest represented fold, if any.
            try invocation_expansions.append(arena, .{
                .invocation_id = formal_fold.invocation_id,
                .nearest_expansion_id = parent_expansion_id,
            });
            continue;
        };
        const comptime_template = package.template(comptime_template_id) orelse
            return error.UnknownComptimeFoldOwnerTemplate;

        const root_module = package.module(formal_fold.root_runtime_module_id) orelse
            return error.UnknownComptimeFoldRootModule;
        const root_runtime_owner = try globalOwner(
            arena,
            root_module.canonical_path,
            formal_fold.root_runtime_owner_key,
        );
        const call_chain = try canonicalCallChain(arena, package, formal_fold.call_site_chain);
        const imported_module: ?[]const u8 = if (std.mem.eql(
            u8,
            root_module.canonical_path,
            callee_module.canonical_path,
        )) null else try arena.dupe(u8, callee_module.canonical_path);
        const generic_bindings = try cloneStrings(arena, formal_fold.generic_bindings);
        validateSortedBindings(generic_bindings) catch return error.InvalidComptimeFoldGenericBindings;
        const trait_implementation = try cloneOptionalString(arena, formal_fold.trait_implementation);
        const trait_method = try cloneOptionalString(arena, formal_fold.trait_method);
        if ((trait_implementation == null) != (trait_method == null)) {
            return error.IncompleteComptimeFoldTraitIdentity;
        }
        const identity = try foldIdentity(
            arena,
            root_runtime_owner,
            call_chain,
            callee_module.canonical_path,
            formal_fold.callee_owner_key,
            generic_bindings,
            trait_implementation,
            trait_method,
        );
        const expansion_id = next_expansion_id;
        next_expansion_id = addId(accounting.ExpansionId, next_expansion_id) catch
            return error.SourceAccountingExpansionIdOverflow;
        try invocation_expansions.append(arena, .{
            .invocation_id = formal_fold.invocation_id,
            .nearest_expansion_id = expansion_id,
        });

        switch (formal_fold.disposition) {
            .committed => {
                const activation: accounting.ActivationReason = switch (formal_fold.activation) {
                    .speculative => .speculative_fold,
                    .required => .required_comptime,
                };
                try expansions.append(arena, .{
                    .id = expansion_id,
                    .template_id = comptime_template.id,
                    .parent_expansion_id = parent_expansion_id,
                    .activation = activation,
                    .disposition = .fold_committed,
                    .root_runtime_owner = root_runtime_owner,
                    .folded_call_site_chain = call_chain,
                    .imported_module = imported_module,
                    .generic_bindings = generic_bindings,
                    .trait_implementation = trait_implementation,
                    .trait_method = trait_method,
                    .identity = identity,
                });
                const instantiated = try instantiateTemplate(
                    arena,
                    comptime_template,
                    expansion_id,
                    &next_use_id,
                    &next_node_id,
                    &next_edge_id,
                    &uses,
                    &nodes,
                    &edges,
                );
                try appendCommittedEvidence(
                    arena,
                    comptime_template,
                    typed_site_index,
                    expansion_id,
                    instantiated,
                    nodes.items,
                    edges.items,
                    formal_fold,
                    &folds,
                    &predicates,
                    &handlings,
                );
            },
            .abandoned => {
                if (formal_fold.activation != .speculative) return error.RequiredComptimeFoldWasAbandoned;
                const boundary_template_id = callee_module.templateForOwner(
                    formal_fold.callee_owner_key,
                    .symbolic_call_boundary,
                ) orelse return error.UnknownComptimeFoldBoundaryTemplate;
                const boundary_template = package.template(boundary_template_id) orelse
                    return error.UnknownComptimeFoldBoundaryTemplate;
                const replacement_id = next_expansion_id;
                next_expansion_id = addId(accounting.ExpansionId, next_expansion_id) catch
                    return error.SourceAccountingExpansionIdOverflow;
                try expansions.append(arena, .{
                    .id = expansion_id,
                    .template_id = comptime_template.id,
                    .parent_expansion_id = parent_expansion_id,
                    .replacement_expansion_id = replacement_id,
                    .activation = .speculative_fold,
                    .disposition = .fold_abandoned_to_symbolic,
                    .root_runtime_owner = root_runtime_owner,
                    .folded_call_site_chain = call_chain,
                    .imported_module = imported_module,
                    .generic_bindings = generic_bindings,
                    .trait_implementation = trait_implementation,
                    .trait_method = trait_method,
                    .identity = identity,
                });
                try expansions.append(arena, .{
                    .id = replacement_id,
                    .template_id = boundary_template.id,
                    .parent_expansion_id = parent_expansion_id,
                    .activation = .symbolic_call,
                    .disposition = .symbolic,
                    .root_runtime_owner = root_runtime_owner,
                    .folded_call_site_chain = call_chain,
                    .imported_module = imported_module,
                    .generic_bindings = generic_bindings,
                    .trait_implementation = trait_implementation,
                    .trait_method = trait_method,
                    .identity = identity,
                });
                _ = try instantiateTemplate(
                    arena,
                    boundary_template,
                    replacement_id,
                    &next_use_id,
                    &next_node_id,
                    &next_edge_id,
                    &uses,
                    &nodes,
                    &edges,
                );
            },
            .rejected => try expansions.append(arena, .{
                .id = expansion_id,
                .template_id = comptime_template.id,
                .parent_expansion_id = parent_expansion_id,
                .activation = switch (formal_fold.activation) {
                    .speculative => .speculative_fold,
                    .required => .required_comptime,
                },
                .disposition = .rejected,
                .root_runtime_owner = root_runtime_owner,
                .folded_call_site_chain = call_chain,
                .imported_module = imported_module,
                .generic_bindings = generic_bindings,
                .trait_implementation = trait_implementation,
                .trait_method = trait_method,
                .identity = identity,
            }),
        }
    }

    result.expansions = try expansions.toOwnedSlice(arena);
    result.uses = try uses.toOwnedSlice(arena);
    result.control_nodes = try nodes.toOwnedSlice(arena);
    result.control_edges = try edges.toOwnedSlice(arena);
    result.evidence = .{
        .folds = try folds.toOwnedSlice(arena),
        .predicate_events = try predicates.toOwnedSlice(arena),
        .handlings = try handlings.toOwnedSlice(arena),
    };
    return result;
}

fn instantiateTemplate(
    allocator: std.mem.Allocator,
    template: accounting.OwnerTemplate,
    expansion_id: accounting.ExpansionId,
    next_use_id: *accounting.UseId,
    next_node_id: *accounting.ControlNodeId,
    next_edge_id: *accounting.ControlEdgeId,
    uses: *std.ArrayList(accounting.SourceFactUse),
    nodes: *std.ArrayList(accounting.ControlNode),
    edges: *std.ArrayList(accounting.ControlEdge),
) !Instantiation {
    const first_use_id = next_use_id.*;
    const node_ids = try allocator.alloc(accounting.ControlNodeId, template.control_nodes.len);
    for (node_ids) |*node_id| {
        node_id.* = next_node_id.*;
        next_node_id.* = addId(accounting.ControlNodeId, next_node_id.*) catch
            return error.SourceAccountingControlNodeIdOverflow;
    }
    for (template.uses, 0..) |use_template, ordinal| {
        const use_id = next_use_id.*;
        next_use_id.* = addId(accounting.UseId, next_use_id.*) catch
            return error.SourceAccountingUseIdOverflow;
        try uses.append(allocator, .{
            .id = use_id,
            .site_id = use_template.site_id,
            .expansion_id = expansion_id,
            .template_ordinal = @intCast(ordinal),
            .role = use_template.role,
            .control_node_id = if (use_template.control_node_slot) |slot|
                nodeIdForSlot(template, node_ids, slot) orelse return error.UnknownComptimeControlNodeSlot
            else
                null,
        });
    }
    for (template.control_nodes, node_ids) |node_template, node_id| {
        const attached = try allocator.alloc(accounting.UseId, node_template.attached_use_ordinals.len);
        for (node_template.attached_use_ordinals, attached) |ordinal, *use_id| {
            if (ordinal >= template.uses.len) return error.InvalidComptimeUseOrdinal;
            use_id.* = std.math.add(accounting.UseId, first_use_id, ordinal) catch
                return error.SourceAccountingUseIdOverflow;
        }
        try nodes.append(allocator, .{
            .id = node_id,
            .expansion_id = expansion_id,
            .slot = node_template.slot,
            .kind = node_template.kind,
            .range = try cloneRange(allocator, node_template.range),
            .attached_use_ids = attached,
        });
    }
    for (template.control_edges) |edge_template| {
        const edge_id = next_edge_id.*;
        next_edge_id.* = addId(accounting.ControlEdgeId, next_edge_id.*) catch
            return error.SourceAccountingControlEdgeIdOverflow;
        try edges.append(allocator, .{
            .id = edge_id,
            .expansion_id = expansion_id,
            .from = nodeIdForSlot(template, node_ids, edge_template.from_slot) orelse
                return error.UnknownComptimeControlNodeSlot,
            .to = nodeIdForSlot(template, node_ids, edge_template.to_slot) orelse
                return error.UnknownComptimeControlNodeSlot,
            .kind = edge_template.kind,
        });
    }
    return .{ .first_use_id = first_use_id, .node_ids = node_ids };
}

fn appendCommittedEvidence(
    allocator: std.mem.Allocator,
    template: accounting.OwnerTemplate,
    typed_site_index: TypedSiteIndex,
    expansion_id: accounting.ExpansionId,
    instantiated: Instantiation,
    nodes: []const accounting.ControlNode,
    edges: []const accounting.ControlEdge,
    formal_fold: anytype,
    folds: *std.ArrayList(accounting.FoldRecord),
    predicates: *std.ArrayList(accounting.PredicateEvent),
    handlings: *std.ArrayList(accounting.HandlingRecord),
) !void {
    const entry_slot = template.entry_slot orelse return error.CommittedComptimeFoldMissingEntry;
    if (template.control_nodes.len == 0) return error.CommittedComptimeFoldMissingControlGraph;
    const entry_node_id = nodeIdForSlot(template, instantiated.node_ids, entry_slot) orelse
        return error.CommittedComptimeFoldMissingEntry;

    var trace: std.ArrayList(accounting.TraceEvent) = .empty;
    var evidence_by_use = try allocator.alloc(std.ArrayList(accounting.EvidenceId), template.uses.len);
    for (evidence_by_use) |*list| list.* = .empty;

    var current_node_id: ?accounting.ControlNodeId = null;
    var terminal_node_id: ?accounting.ControlNodeId = null;
    for (formal_fold.events) |event| switch (event.kind) {
        .enter_node => {
            const node_id = try resolveNodeId(
                template,
                instantiated.node_ids,
                event.node_kind,
                event.node_range,
            );
            if (current_node_id) |current| {
                const edge = uniqueEdge(edges, expansion_id, current, node_id) orelse
                    return error.ComptimeFoldTraceEdgeMissingOrAmbiguous;
                switch (edge.kind) {
                    .return_exit => try trace.append(allocator, .{ .kind = .return_exit, .node_id = current }),
                    .break_exit => try trace.append(allocator, .{ .kind = .break_exit, .node_id = current }),
                    .continue_backedge => try trace.append(allocator, .{ .kind = .continue_backedge, .node_id = current }),
                    else => {},
                }
                try trace.append(allocator, .{ .kind = .take_edge, .edge_id = edge.id });
            }
            try trace.append(allocator, .{ .kind = .enter_node, .node_id = node_id });
            current_node_id = node_id;
            terminal_node_id = switch (event.node_kind) {
                .success_exit, .error_exit => node_id,
                else => terminal_node_id,
            };
        },
        .predicate_check => {
            const node_id = current_node_id orelse return error.ComptimePredicateOutsideControlNode;
            const source_fact_id = event.source_fact_id orelse
                return error.MissingComptimeSourceFactIdentity;
            const value = event.predicate_value orelse return error.MissingComptimePredicateValue;
            var matched_use = false;
            for (template.uses, 0..) |use_template, ordinal| {
                const slot = use_template.control_node_slot orelse continue;
                const use_node_id = nodeIdForSlot(template, instantiated.node_ids, slot) orelse
                    return error.UnknownComptimeControlNodeSlot;
                if (use_node_id != node_id) continue;
                const site = typed_site_index.get(use_template.site_id) orelse
                    return error.UnknownComptimeTypedSite;
                if (site.source_fact_id != source_fact_id) continue;
                matched_use = true;
                const use_id = std.math.add(
                    accounting.UseId,
                    instantiated.first_use_id,
                    @as(u32, @intCast(ordinal)),
                ) catch return error.SourceAccountingUseIdOverflow;
                const evidence_id = try accounting.namespacedEvidenceId(
                    .concrete_predicate,
                    @intCast(predicates.items.len + 1),
                );
                try predicates.append(allocator, .{
                    .id = evidence_id,
                    .fold_id = expansion_id,
                    .use_id = use_id,
                    .node_id = node_id,
                    .value = value,
                });
                try evidence_by_use[ordinal].append(allocator, evidence_id);
                try trace.append(allocator, .{
                    .kind = .predicate_check,
                    .node_id = node_id,
                    .use_id = use_id,
                    .predicate_value = value,
                });
            }
            if (!matched_use) return error.UnmatchedComptimePredicateSourceFact;
        },
    };

    const terminal_id = terminal_node_id orelse return error.CommittedComptimeFoldMissingTerminal;
    const terminal = nodeById(nodes, terminal_id) orelse return error.UnknownComptimeTerminalNode;
    try trace.append(allocator, .{
        .kind = if (terminal.kind == .error_exit) .error_exit else .success_exit,
        .node_id = terminal_id,
    });
    try folds.append(allocator, .{
        .id = expansion_id,
        .expansion_id = expansion_id,
        .entry_node_id = entry_node_id,
        .terminal_node_id = terminal_id,
        .disposition = .committed,
        .events = try trace.toOwnedSlice(allocator),
    });

    for (template.uses, 0..) |_, ordinal| {
        const use_id = std.math.add(
            accounting.UseId,
            instantiated.first_use_id,
            @as(u32, @intCast(ordinal)),
        ) catch return error.SourceAccountingUseIdOverflow;
        const predicate_ids = try evidence_by_use[ordinal].toOwnedSlice(allocator);
        try handlings.append(allocator, .{
            .id = try accounting.namespacedHandlingId(
                .concrete,
                @intCast(handlings.items.len + 1),
            ),
            .use_id = use_id,
            .kind = if (predicate_ids.len == 0) .control_eliminated else .concrete_true,
            .predicate_event_ids = predicate_ids,
            .fold_id = expansion_id,
        });
    }
}

fn resolveNodeId(
    template: accounting.OwnerTemplate,
    node_ids: []const accounting.ControlNodeId,
    kind: anytype,
    range: anytype,
) !accounting.ControlNodeId {
    return switch (kind) {
        .owner_entry => if (template.entry_slot) |slot|
            nodeIdForSlot(template, node_ids, slot) orelse error.UnknownComptimeControlNodeSlot
        else
            error.CommittedComptimeFoldMissingEntry,
        .success_exit => uniqueNodeForKind(template, node_ids, .success_exit) orelse
            error.ComptimeFoldControlNodeMissingOrAmbiguous,
        .error_exit => uniqueNodeForKind(template, node_ids, .error_exit) orelse
            error.ComptimeFoldControlNodeMissingOrAmbiguous,
        .statement => uniqueNodeForRange(template, node_ids, range) orelse
            error.ComptimeFoldControlNodeMissingOrAmbiguous,
    };
}

fn uniqueNodeForKind(
    template: accounting.OwnerTemplate,
    node_ids: []const accounting.ControlNodeId,
    kind: accounting.ControlNodeKind,
) ?accounting.ControlNodeId {
    var matched: ?accounting.ControlNodeId = null;
    for (template.control_nodes, node_ids) |node, node_id| {
        if (node.kind != kind) continue;
        if (matched != null) return null;
        matched = node_id;
    }
    return matched;
}

fn uniqueNodeForRange(
    template: accounting.OwnerTemplate,
    node_ids: []const accounting.ControlNodeId,
    range: anytype,
) ?accounting.ControlNodeId {
    var matched: ?accounting.ControlNodeId = null;
    for (template.control_nodes, node_ids) |node, node_id| {
        if (node.range.start != range.start or node.range.end != range.end) continue;
        if (matched != null) return null;
        matched = node_id;
    }
    return matched;
}

fn uniqueEdge(
    edges: []const accounting.ControlEdge,
    expansion_id: accounting.ExpansionId,
    from: accounting.ControlNodeId,
    to: accounting.ControlNodeId,
) ?accounting.ControlEdge {
    var matched: ?accounting.ControlEdge = null;
    for (edges) |edge| {
        if (edge.expansion_id != expansion_id or edge.from != from or edge.to != to) continue;
        if (matched != null) return null;
        matched = edge;
    }
    return matched;
}

fn nodeIdForSlot(
    template: accounting.OwnerTemplate,
    node_ids: []const accounting.ControlNodeId,
    slot: u32,
) ?accounting.ControlNodeId {
    for (template.control_nodes, node_ids) |node, node_id| if (node.slot == slot) return node_id;
    return null;
}

fn nodeById(nodes: []const accounting.ControlNode, id: accounting.ControlNodeId) ?accounting.ControlNode {
    if (nodes.len == 0 or id < nodes[0].id) return null;
    const index: usize = @intCast(id - nodes[0].id);
    if (index >= nodes.len) return null;
    const node = nodes[index];
    // instantiateTemplate appends nodes in the same dense order in which it
    // allocates IDs. Keep this equality check fail-closed if that invariant is
    // ever weakened by a later reorder.
    return if (node.id == id) node else null;
}

fn invocationExpansionById(rows: []const InvocationExpansion, invocation_id: u32) ?InvocationExpansion {
    // Rows are appended exactly once while walking the invocation-ID-sorted
    // fold order. Binary lookup depends on that canonical order; the duplicate
    // check in the caller must remain ahead of all row construction.
    var low: usize = 0;
    var high: usize = rows.len;
    while (low < high) {
        const middle = low + (high - low) / 2;
        const row = rows[middle];
        if (row.invocation_id < invocation_id) {
            low = middle + 1;
        } else if (row.invocation_id > invocation_id) {
            high = middle;
        } else {
            return row;
        }
    }
    return null;
}

fn canonicalCallChain(allocator: std.mem.Allocator, package: anytype, chain: anytype) ![]const accounting.SourceRange {
    const result = try allocator.alloc(accounting.SourceRange, chain.len);
    for (chain, result) |call_site, *canonical| {
        const module = package.moduleForFile(call_site.file_id) orelse
            return error.UnknownComptimeFoldCallSiteModule;
        canonical.* = .{
            .file = try allocator.dupe(u8, module.canonical_path),
            .start = call_site.range.start,
            .end = call_site.range.end,
        };
    }
    return result;
}

fn globalOwner(allocator: std.mem.Allocator, module_path: []const u8, owner_key: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}::{s}", .{ module_path, owner_key });
}

fn foldIdentity(
    allocator: std.mem.Allocator,
    root_runtime_owner: []const u8,
    call_chain: []const accounting.SourceRange,
    callee_module_path: []const u8,
    callee_owner_key: []const u8,
    generic_bindings: []const []const u8,
    trait_implementation: ?[]const u8,
    trait_method: ?[]const u8,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writeIdentityPart(writer, "comptime-fold");
    try writeIdentityPart(writer, root_runtime_owner);
    for (call_chain) |call_site| {
        try writeIdentityPart(writer, call_site.file);
        try writer.print("{d}:{d};", .{ call_site.start, call_site.end });
    }
    try writeIdentityPart(writer, callee_module_path);
    try writeIdentityPart(writer, callee_owner_key);
    for (generic_bindings) |binding| try writeIdentityPart(writer, binding);
    try writeIdentityPart(writer, trait_implementation orelse "");
    try writeIdentityPart(writer, trait_method orelse "");
    return out.toOwnedSlice();
}

fn writeIdentityPart(writer: anytype, value: []const u8) !void {
    try writer.print("{d}:", .{value.len});
    try writer.writeAll(value);
}

fn validateSortedBindings(bindings: []const []const u8) !void {
    if (bindings.len < 2) return;
    for (bindings[1..], bindings[0..bindings.len -| 1]) |current, previous| {
        if (std.mem.order(u8, previous, current) != .lt) return error.InvalidComptimeFoldGenericBindings;
    }
}

fn cloneStrings(allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    const cloned = try allocator.alloc([]const u8, values.len);
    for (values, cloned) |value, *copy| copy.* = try allocator.dupe(u8, value);
    return cloned;
}

fn cloneOptionalString(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const u8 {
    return if (value) |text| try allocator.dupe(u8, text) else null;
}

fn cloneRange(allocator: std.mem.Allocator, range: accounting.SourceRange) !accounting.SourceRange {
    return .{
        .file = try allocator.dupe(u8, range.file),
        .start = range.start,
        .end = range.end,
    };
}

fn addId(comptime T: type, id: T) !T {
    return std.math.add(T, id, 1);
}

fn testTypedSite(id: accounting.SiteId) accounting.TypedSite {
    return .{
        .id = id,
        .origin = .source_syntax,
        .kind = .requires,
        .key = .{
            .path = "contract.ora",
            .owner = "module/contract:Contract/function:run",
            .range_start = id,
            .range_end = id + 1,
            .kind = .requires,
            .ordinal = id,
        },
    };
}

test "comptime typed-site index supports sparse canonical IDs" {
    const empty = try TypedSiteIndex.init(&.{});
    try std.testing.expect(empty.get(1) == null);

    const rows = [_]accounting.TypedSite{
        testTypedSite(2),
        testTypedSite(17),
        testTypedSite(41),
    };
    const index = try TypedSiteIndex.init(&rows);
    try std.testing.expectEqual(@as(accounting.SiteId, 2), index.get(2).?.id);
    try std.testing.expectEqual(@as(accounting.SiteId, 17), index.get(17).?.id);
    try std.testing.expectEqual(@as(accounting.SiteId, 41), index.get(41).?.id);
    try std.testing.expect(index.get(1) == null);
    try std.testing.expect(index.get(18) == null);
    try std.testing.expect(index.get(42) == null);
}

test "comptime typed-site index rejects duplicate and noncanonical IDs" {
    const duplicate = [_]accounting.TypedSite{
        testTypedSite(2),
        testTypedSite(2),
    };
    try std.testing.expectError(
        error.DuplicateComptimeTypedSiteIdentity,
        TypedSiteIndex.init(&duplicate),
    );

    const unordered = [_]accounting.TypedSite{
        testTypedSite(3),
        testTypedSite(2),
    };
    try std.testing.expectError(
        error.NonCanonicalComptimeTypedSiteOrder,
        TypedSiteIndex.init(&unordered),
    );
}

test "comptime invocation expansion lookup supports sparse canonical IDs" {
    const rows = [_]InvocationExpansion{
        .{ .invocation_id = 3, .nearest_expansion_id = 11 },
        .{ .invocation_id = 19, .nearest_expansion_id = null },
        .{ .invocation_id = 52, .nearest_expansion_id = 27 },
    };
    try std.testing.expectEqual(@as(?accounting.ExpansionId, 11), invocationExpansionById(&rows, 3).?.nearest_expansion_id);
    try std.testing.expectEqual(@as(?accounting.ExpansionId, null), invocationExpansionById(&rows, 19).?.nearest_expansion_id);
    try std.testing.expectEqual(@as(?accounting.ExpansionId, 27), invocationExpansionById(&rows, 52).?.nearest_expansion_id);
    try std.testing.expect(invocationExpansionById(&rows, 2) == null);
    try std.testing.expect(invocationExpansionById(&rows, 20) == null);
    try std.testing.expect(invocationExpansionById(&rows, 53) == null);
}

test "comptime control-node lookup uses dense canonical IDs fail closed" {
    const canonical = [_]accounting.ControlNode{
        .{ .id = 40, .expansion_id = 7, .slot = 0, .kind = .entry, .range = .{ .file = "contract.ora", .start = 1, .end = 2 } },
        .{ .id = 41, .expansion_id = 7, .slot = 1, .kind = .statement, .range = .{ .file = "contract.ora", .start = 3, .end = 4 } },
        .{ .id = 42, .expansion_id = 7, .slot = 2, .kind = .success_exit, .range = .{ .file = "contract.ora", .start = 5, .end = 6 } },
    };
    try std.testing.expectEqual(accounting.ControlNodeKind.entry, nodeById(&canonical, 40).?.kind);
    try std.testing.expectEqual(accounting.ControlNodeKind.statement, nodeById(&canonical, 41).?.kind);
    try std.testing.expectEqual(accounting.ControlNodeKind.success_exit, nodeById(&canonical, 42).?.kind);
    try std.testing.expect(nodeById(&canonical, 39) == null);
    try std.testing.expect(nodeById(&canonical, 43) == null);

    const noncanonical = [_]accounting.ControlNode{
        canonical[0],
        canonical[2],
    };
    try std.testing.expect(nodeById(&noncanonical, 41) == null);
}
