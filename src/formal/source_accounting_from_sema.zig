//! Typed source-site and owner-template inventory.
//!
//! This AST walk is independent from compile-time evaluation and from MLIR/Z3
//! obligation construction. Source sites are reconciled against the prior
//! `SyntaxTree` inventory by structural containment and fact kind.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    typed_sites: []const accounting.TypedSite,
    owner_templates: []const accounting.OwnerTemplate,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const Candidate = struct {
    kind: accounting.SourceFactKind,
    range: struct { start: u32, end: u32 },
    source_fact_id: ?u32,
    owner_hint: []const u8,
};

const Matched = struct {
    candidate: Candidate,
    site: accounting.TypedSite,
    template_owner: []const u8,
};

pub fn collect(
    allocator: std.mem.Allocator,
    file: anytype,
    canonical_path: []const u8,
    declared_sites: []const accounting.DeclaredSite,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .typed_sites = &.{},
        .owner_templates = &.{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    var candidates: std.ArrayList(Candidate) = .empty;
    for (file.root_items) |item_id| try collectItem(arena, file, item_id, "module", &candidates);
    std.mem.sort(Candidate, candidates.items, {}, lessCandidate);

    var declared_used = try arena.alloc(bool, declared_sites.len);
    @memset(declared_used, false);
    var matched: std.ArrayList(Matched) = .empty;
    var next_unknown_id: accounting.SiteId = 1;
    for (declared_sites) |site| next_unknown_id = @max(next_unknown_id, site.id + 1);

    for (candidates.items) |candidate| {
        const declared_index = bestDeclaredMatch(candidate, declared_sites, declared_used);
        const typed_site: accounting.TypedSite = if (declared_index) |index| blk: {
            declared_used[index] = true;
            const declared = declared_sites[index];
            break :blk .{
                .id = declared.id,
                .origin = .source_syntax,
                .kind = candidate.kind,
                .key = try cloneKey(arena, declared.key),
                .source_fact_id = candidate.source_fact_id,
                .declared_site_id = declared.id,
            };
        } else blk: {
            const id = next_unknown_id;
            next_unknown_id += 1;
            break :blk .{
                .id = id,
                .origin = .source_syntax,
                .kind = candidate.kind,
                .key = .{
                    .path = try arena.dupe(u8, canonical_path),
                    .owner = try arena.dupe(u8, candidate.owner_hint),
                    .range_start = candidate.range.start,
                    .range_end = candidate.range.end,
                    .kind = candidate.kind,
                    .ordinal = 0,
                },
                .source_fact_id = candidate.source_fact_id,
                .declared_site_id = null,
            };
        };
        try matched.append(arena, .{
            .candidate = candidate,
            .site = typed_site,
            .template_owner = try arena.dupe(u8, rootOwner(typed_site.key.owner)),
        });
    }

    std.mem.sort(Matched, matched.items, {}, lessMatched);
    const typed_sites = try arena.alloc(accounting.TypedSite, matched.items.len);
    for (matched.items, typed_sites) |row, *typed| typed.* = row.site;

    var template_owners: std.ArrayList([]const u8) = .empty;
    for (matched.items) |row| try template_owners.append(arena, row.template_owner);
    for (file.root_items) |item_id| try collectRuntimeOwnerKeys(arena, file, item_id, "module", &template_owners);
    std.mem.sort([]const u8, template_owners.items, {}, lessString);

    var templates: std.ArrayList(accounting.OwnerTemplate) = .empty;
    var previous_owner: ?[]const u8 = null;
    for (template_owners.items) |owner| {
        if (previous_owner != null and std.mem.eql(u8, previous_owner.?, owner)) continue;
        previous_owner = owner;
        var selected: std.ArrayList(Matched) = .empty;
        for (matched.items) |row| if (siteAppliesToTemplate(row, owner)) {
            try selected.append(arena, row);
        };
        if (selected.items.len == 0) continue;
        std.mem.sort(Matched, selected.items, {}, lessMatched);
        const activations = [_]accounting.TemplateActivation{
            .runtime_body,
            .symbolic_call_boundary,
            .comptime_body,
        };
        for (activations) |activation| {
            var use_templates: std.ArrayList(accounting.UseTemplate) = .empty;
            for (selected.items) |row| {
                try appendRequiredUseTemplates(arena, &use_templates, row.site, activation);
            }
            const owned_uses = try use_templates.toOwnedSlice(arena);
            const graph = if (activation == .symbolic_call_boundary)
                BuiltControlTemplate{}
            else
                try buildOwnerControlTemplate(
                    arena,
                    file,
                    canonical_path,
                    owner,
                    selected.items,
                    owned_uses,
                );
            try templates.append(arena, .{
                .id = @intCast(templates.items.len + 1),
                .owner_key = try arena.dupe(u8, owner),
                .activation = activation,
                .uses = owned_uses,
                .control_nodes = graph.nodes,
                .control_edges = graph.edges,
                .entry_slot = graph.entry_slot,
                .terminal_slots = graph.terminal_slots,
            });
        }
    }

    result.typed_sites = typed_sites;
    result.owner_templates = try templates.toOwnedSlice(arena);
    return result;
}

const BuiltControlTemplate = struct {
    nodes: []const accounting.ControlNodeTemplate = &.{},
    edges: []const accounting.ControlEdgeTemplate = &.{},
    entry_slot: ?u32 = null,
    terminal_slots: []const u32 = &.{},
};

const MutableControlNode = struct {
    slot: u32,
    kind: accounting.ControlNodeKind,
    range: accounting.SourceRange,
};

const JumpTarget = struct {
    label: ?[]const u8,
    break_slot: u32,
    continue_slot: ?u32,
};

const UseOccurrenceKey = struct {
    site_id: accounting.SiteId,
    role: accounting.UseRole,
};

fn lessString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}

fn siteAppliesToTemplate(row: Matched, owner: []const u8) bool {
    if (std.mem.eql(u8, row.template_owner, owner)) return true;
    if (row.site.kind != .contract_invariant or !std.mem.startsWith(u8, owner, row.template_owner) or
        owner.len <= row.template_owner.len or owner[row.template_owner.len] != '/') return false;
    return std.mem.indexOfPos(u8, owner, row.template_owner.len, "/function:") != null;
}

fn collectRuntimeOwnerKeys(
    allocator: std.mem.Allocator,
    file: anytype,
    item_id: anytype,
    parent_owner: []const u8,
    owners: *std.ArrayList([]const u8),
) anyerror!void {
    switch (file.item(item_id).*) {
        .Contract => |contract| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/contract:{s}", .{ parent_owner, contract.name });
            for (contract.members) |member_id| try collectRuntimeOwnerKeys(allocator, file, member_id, owner, owners);
        },
        .Function => |function| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/function:{s}", .{ parent_owner, function.name });
            try owners.append(allocator, owner);
        },
        .Impl => |impl_item| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/impl:{s}:{s}", .{ parent_owner, impl_item.trait_name, impl_item.target_name });
            for (impl_item.methods) |method_id| try collectRuntimeOwnerKeys(allocator, file, method_id, owner, owners);
        },
        else => {},
    }
}

fn appendRequiredUseTemplates(
    allocator: std.mem.Allocator,
    uses: *std.ArrayList(accounting.UseTemplate),
    site: accounting.TypedSite,
    activation: accounting.TemplateActivation,
) !void {
    for (accounting.templateRoles(site.kind, activation)) |role| {
        try uses.append(allocator, .{ .site_id = site.id, .role = role });
    }
}

fn buildOwnerControlTemplate(
    allocator: std.mem.Allocator,
    file: anytype,
    canonical_path: []const u8,
    owner: []const u8,
    sites: []const Matched,
    uses: []accounting.UseTemplate,
) !BuiltControlTemplate {
    var nodes: std.ArrayList(MutableControlNode) = .empty;
    var edges: std.ArrayList(accounting.ControlEdgeTemplate) = .empty;
    var built = false;
    for (file.root_items) |item_id| {
        if (try buildOwnerGraphForItem(
            allocator,
            file,
            item_id,
            "module",
            canonical_path,
            owner,
            &nodes,
            &edges,
        )) {
            built = true;
            break;
        }
    }
    if (!built) return .{};

    var attachments = try allocator.alloc(std.ArrayList(u32), nodes.items.len);
    for (attachments) |*list| list.* = .empty;
    var use_occurrences = std.AutoHashMap(UseOccurrenceKey, u32).init(allocator);
    defer use_occurrences.deinit();
    for (uses, 0..) |*use, ordinal| {
        const site = findMatchedSite(sites, use.site_id) orelse continue;
        const occurrence_entry = try use_occurrences.getOrPut(.{ .site_id = use.site_id, .role = use.role });
        if (!occurrence_entry.found_existing) occurrence_entry.value_ptr.* = 0;
        const occurrence = occurrence_entry.value_ptr.*;
        occurrence_entry.value_ptr.* += 1;
        const slot = chooseControlSlot(nodes.items, site.site, use.role, occurrence) orelse continue;
        use.control_node_slot = slot;
        try attachments[slot].append(allocator, @intCast(ordinal));
    }

    const frozen_nodes = try allocator.alloc(accounting.ControlNodeTemplate, nodes.items.len);
    for (nodes.items, frozen_nodes, 0..) |node, *frozen, index| frozen.* = .{
        .slot = node.slot,
        .kind = node.kind,
        .range = node.range,
        .attached_use_ordinals = try attachments[index].toOwnedSlice(allocator),
    };
    return .{
        .nodes = frozen_nodes,
        .edges = try edges.toOwnedSlice(allocator),
        .entry_slot = 0,
        .terminal_slots = try allocator.dupe(u32, &.{ 1, 2 }),
    };
}

fn buildOwnerGraphForItem(
    allocator: std.mem.Allocator,
    file: anytype,
    item_id: anytype,
    parent_owner: []const u8,
    canonical_path: []const u8,
    wanted_owner: []const u8,
    nodes: *std.ArrayList(MutableControlNode),
    edges: *std.ArrayList(accounting.ControlEdgeTemplate),
) anyerror!bool {
    return switch (file.item(item_id).*) {
        .Contract => |contract| blk: {
            const contract_owner = try std.fmt.allocPrint(allocator, "{s}/contract:{s}", .{ parent_owner, contract.name });
            for (contract.members) |member_id| {
                if (try buildOwnerGraphForItem(allocator, file, member_id, contract_owner, canonical_path, wanted_owner, nodes, edges)) break :blk true;
            }
            break :blk false;
        },
        .Function => |function| blk: {
            const function_owner = try std.fmt.allocPrint(allocator, "{s}/function:{s}", .{ parent_owner, function.name });
            if (!std.mem.eql(u8, function_owner, wanted_owner)) break :blk false;
            try buildBodyControlGraph(allocator, file, canonical_path, function.body, nodes, edges);
            break :blk true;
        },
        .Impl => |impl_item| blk: {
            const impl_owner = try std.fmt.allocPrint(allocator, "{s}/impl:{s}:{s}", .{ parent_owner, impl_item.trait_name, impl_item.target_name });
            for (impl_item.methods) |method_id| {
                if (try buildOwnerGraphForItem(allocator, file, method_id, impl_owner, canonical_path, wanted_owner, nodes, edges)) break :blk true;
            }
            break :blk false;
        },
        .GhostBlock => |ghost| blk: {
            if (!std.mem.eql(u8, parent_owner, wanted_owner)) break :blk false;
            try buildBodyControlGraph(allocator, file, canonical_path, ghost.body, nodes, edges);
            break :blk true;
        },
        else => false,
    };
}

fn buildBodyControlGraph(
    allocator: std.mem.Allocator,
    file: anytype,
    canonical_path: []const u8,
    body_id: anytype,
    nodes: *std.ArrayList(MutableControlNode),
    edges: *std.ArrayList(accounting.ControlEdgeTemplate),
) !void {
    const body_range = file.body(body_id).range;
    try nodes.append(allocator, try mutableNode(allocator, canonical_path, 0, .entry, body_range));
    try nodes.append(allocator, try mutableNode(allocator, canonical_path, 1, .success_exit, body_range));
    try nodes.append(allocator, try mutableNode(allocator, canonical_path, 2, .error_exit, body_range));

    const StmtId = @TypeOf(file.body(body_id).statements[0]);
    var slots = std.AutoHashMap(StmtId, u32).init(allocator);
    defer slots.deinit();
    try allocateBodyNodes(allocator, file, canonical_path, body_id, nodes, &slots);

    var targets: std.ArrayList(JumpTarget) = .empty;
    const body_entry = try connectBody(allocator, file, body_id, 1, .next, 1, 2, &slots, nodes, edges, &targets);
    try appendTemplateEdge(allocator, edges, 0, body_entry, .next);
}

fn allocateBodyNodes(
    allocator: std.mem.Allocator,
    file: anytype,
    canonical_path: []const u8,
    body_id: anytype,
    nodes: *std.ArrayList(MutableControlNode),
    slots: anytype,
) anyerror!void {
    for (file.body(body_id).statements) |statement_id| {
        const statement = file.statement(statement_id).*;
        const slot: u32 = @intCast(nodes.items.len);
        try slots.put(statement_id, slot);
        try nodes.append(allocator, try mutableNode(allocator, canonical_path, slot, controlKind(statement), statementRange(statement)));
        switch (statement) {
            .If => |row| {
                try allocateBodyNodes(allocator, file, canonical_path, row.then_body, nodes, slots);
                if (row.else_body) |else_body| try allocateBodyNodes(allocator, file, canonical_path, else_body, nodes, slots);
            },
            .While => |row| try allocateBodyNodes(allocator, file, canonical_path, row.body, nodes, slots),
            .For => |row| try allocateBodyNodes(allocator, file, canonical_path, row.body, nodes, slots),
            .Switch => |row| {
                for (row.arms) |arm| try allocateBodyNodes(allocator, file, canonical_path, arm.body, nodes, slots);
                if (row.else_body) |else_body| try allocateBodyNodes(allocator, file, canonical_path, else_body, nodes, slots);
            },
            .Try => |row| {
                try allocateBodyNodes(allocator, file, canonical_path, row.try_body, nodes, slots);
                if (row.catch_clause) |catch_clause| try allocateBodyNodes(allocator, file, canonical_path, catch_clause.body, nodes, slots);
            },
            .Block => |row| try allocateBodyNodes(allocator, file, canonical_path, row.body, nodes, slots),
            .LabeledBlock => |row| try allocateBodyNodes(allocator, file, canonical_path, row.body, nodes, slots),
            else => {},
        }
    }
}

fn connectBody(
    allocator: std.mem.Allocator,
    file: anytype,
    body_id: anytype,
    continuation: u32,
    continuation_kind: accounting.ControlEdgeKind,
    success_exit: u32,
    error_exit: u32,
    slots: anytype,
    nodes: *const std.ArrayList(MutableControlNode),
    edges: *std.ArrayList(accounting.ControlEdgeTemplate),
    targets: *std.ArrayList(JumpTarget),
) anyerror!u32 {
    var next = continuation;
    var next_kind = continuation_kind;
    const statements = file.body(body_id).statements;
    var index = statements.len;
    while (index > 0) {
        index -= 1;
        const statement_id = statements[index];
        const slot = slots.get(statement_id).?;
        const statement = file.statement(statement_id).*;
        switch (statement) {
            .Return => try appendTemplateEdge(allocator, edges, slot, success_exit, .return_exit),
            .Break => |jump| {
                const target = findJumpTarget(targets.items, jump.label, false) orelse next;
                try appendTemplateEdge(allocator, edges, slot, target, .break_exit);
            },
            .Continue => |jump| {
                const target = findJumpTarget(targets.items, jump.label, true) orelse next;
                try appendTemplateEdge(allocator, edges, slot, target, .continue_backedge);
            },
            .If => |row| {
                const then_entry = try connectBody(allocator, file, row.then_body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets);
                const else_entry = if (row.else_body) |else_body|
                    try connectBody(allocator, file, else_body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets)
                else
                    next;
                try appendTemplateEdge(allocator, edges, slot, then_entry, .branch_true);
                try appendTemplateEdge(allocator, edges, slot, else_entry, .branch_false);
            },
            .While => |row| {
                try targets.append(allocator, .{ .label = row.label, .break_slot = next, .continue_slot = slot });
                defer _ = targets.pop();
                const body_entry = try connectBody(allocator, file, row.body, slot, .backedge, success_exit, error_exit, slots, nodes, edges, targets);
                try appendTemplateEdge(allocator, edges, slot, body_entry, .loop_body);
                try appendTemplateEdge(allocator, edges, slot, next, .loop_exit);
            },
            .For => |row| {
                try targets.append(allocator, .{ .label = row.label, .break_slot = next, .continue_slot = slot });
                defer _ = targets.pop();
                const body_entry = try connectBody(allocator, file, row.body, slot, .backedge, success_exit, error_exit, slots, nodes, edges, targets);
                try appendTemplateEdge(allocator, edges, slot, body_entry, .loop_body);
                try appendTemplateEdge(allocator, edges, slot, next, .loop_exit);
            },
            .Switch => |row| {
                try targets.append(allocator, .{ .label = row.label, .break_slot = next, .continue_slot = if (row.label != null) slot else null });
                defer _ = targets.pop();
                for (row.arms) |arm| {
                    const arm_entry = try connectBody(allocator, file, arm.body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets);
                    try appendTemplateEdge(allocator, edges, slot, arm_entry, .branch_true);
                }
                const else_entry = if (row.else_body) |else_body|
                    try connectBody(allocator, file, else_body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets)
                else
                    next;
                try appendTemplateEdge(allocator, edges, slot, else_entry, .branch_false);
            },
            .Try => |row| {
                const try_entry = try connectBody(allocator, file, row.try_body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets);
                const catch_entry = if (row.catch_clause) |catch_clause|
                    try connectBody(allocator, file, catch_clause.body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets)
                else
                    error_exit;
                try appendTemplateEdge(allocator, edges, slot, try_entry, .success_exit);
                try appendTemplateEdge(allocator, edges, slot, catch_entry, .error_exit);
            },
            .Block => |row| {
                const body_entry = try connectBody(allocator, file, row.body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets);
                try appendTemplateEdge(allocator, edges, slot, body_entry, .next);
            },
            .LabeledBlock => |row| {
                try targets.append(allocator, .{ .label = row.label, .break_slot = next, .continue_slot = null });
                defer _ = targets.pop();
                const body_entry = try connectBody(allocator, file, row.body, next, next_kind, success_exit, error_exit, slots, nodes, edges, targets);
                try appendTemplateEdge(allocator, edges, slot, body_entry, .next);
            },
            else => try appendTemplateEdge(allocator, edges, slot, next, next_kind),
        }
        next = slot;
        next_kind = .next;
    }
    return next;
}

fn appendTemplateEdge(
    allocator: std.mem.Allocator,
    edges: *std.ArrayList(accounting.ControlEdgeTemplate),
    from: u32,
    to: u32,
    kind: accounting.ControlEdgeKind,
) !void {
    // The control graph records reachability, not switch-case labels. Multiple
    // empty arms can therefore collapse to the same structural edge. Keep one
    // canonical edge instead of manufacturing duplicate graph identities.
    for (edges.items) |edge| {
        if (edge.from_slot == from and edge.to_slot == to and edge.kind == kind) return;
    }
    try edges.append(allocator, .{
        .slot = @intCast(edges.items.len),
        .from_slot = from,
        .to_slot = to,
        .kind = kind,
    });
}

fn findJumpTarget(targets: []const JumpTarget, label: ?[]const u8, want_continue: bool) ?u32 {
    var index = targets.len;
    while (index > 0) {
        index -= 1;
        const target = targets[index];
        if (label) |name| {
            if (target.label == null or !std.mem.eql(u8, name, target.label.?)) continue;
        }
        if (want_continue) {
            if (target.continue_slot) |slot| return slot;
            if (label != null) return null;
        } else return target.break_slot;
    }
    return null;
}

fn mutableNode(
    allocator: std.mem.Allocator,
    canonical_path: []const u8,
    slot: u32,
    kind: accounting.ControlNodeKind,
    range: anytype,
) !MutableControlNode {
    return .{
        .slot = slot,
        .kind = kind,
        .range = .{
            .file = try allocator.dupe(u8, canonical_path),
            .start = range.start,
            .end = range.end,
        },
    };
}

fn controlKind(statement: anytype) accounting.ControlNodeKind {
    return switch (statement) {
        .If, .Switch, .Try => .branch,
        .While, .For => .loop_head,
        .Return => .return_exit,
        else => .statement,
    };
}

fn statementRange(statement: anytype) @TypeOf(switch (statement) {
    inline else => |payload| payload.range,
}) {
    return switch (statement) {
        inline else => |payload| payload.range,
    };
}

fn findMatchedSite(sites: []const Matched, id: accounting.SiteId) ?Matched {
    for (sites) |site| if (site.site.id == id) return site;
    return null;
}

fn chooseControlSlot(
    nodes: []const MutableControlNode,
    site: accounting.TypedSite,
    role: accounting.UseRole,
    role_occurrence: u32,
) ?u32 {
    switch (site.kind) {
        .ensures, .ensures_ok => return 1,
        .ensures_err => return 2,
        .requires, .guard, .modifies, .refinement_guard, .runtime_guard => return 0,
        .contract_invariant => return if (role == .assumption_context) 0 else if (role_occurrence == 0) 1 else 2,
        else => {},
    }
    var best: ?u32 = null;
    var best_width: u32 = std.math.maxInt(u32);
    for (nodes) |node| {
        if (site.kind == .loop_invariant and node.kind != .loop_head) continue;
        if (node.range.start > site.key.range_start or node.range.end < site.key.range_end) continue;
        const width = node.range.end - node.range.start;
        if (best == null or width < best_width) {
            best = node.slot;
            best_width = width;
        }
    }
    return best;
}

fn collectItem(
    allocator: std.mem.Allocator,
    file: anytype,
    item_id: anytype,
    parent_owner: []const u8,
    candidates: *std.ArrayList(Candidate),
) anyerror!void {
    switch (file.item(item_id).*) {
        .Contract => |contract| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/contract:{s}", .{ parent_owner, contract.name });
            for (contract.invariants) |invariant| try appendCandidate(candidates, allocator, .contract_invariant, invariant.range, invariant.source_fact_id, owner);
            for (contract.members) |member_id| try collectItem(allocator, file, member_id, owner, candidates);
        },
        .Function => |function| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/function:{s}", .{ parent_owner, function.name });
            for (function.clauses) |clause| {
                const kind = factKindForClause(clause.kind) orelse continue;
                try appendCandidate(candidates, allocator, kind, clause.range, clause.source_fact_id, owner);
            }
            try collectBody(allocator, file, function.body, owner, candidates);
        },
        .Trait => |trait_item| {
            const trait_owner = try std.fmt.allocPrint(allocator, "{s}/trait:{s}", .{ parent_owner, trait_item.name });
            for (trait_item.methods) |method| {
                const owner = try std.fmt.allocPrint(allocator, "{s}/trait_method:{s}", .{ trait_owner, method.name });
                for (method.clauses) |clause| {
                    const kind = factKindForClause(clause.kind) orelse continue;
                    try appendCandidate(candidates, allocator, kind, clause.range, clause.source_fact_id, owner);
                }
            }
            if (trait_item.ghost_block) |ghost_id| try collectItem(allocator, file, ghost_id, trait_owner, candidates);
        },
        .Impl => |impl_item| {
            const owner = try std.fmt.allocPrint(allocator, "{s}/impl:{s}:{s}", .{ parent_owner, impl_item.trait_name, impl_item.target_name });
            for (impl_item.methods) |method_id| try collectItem(allocator, file, method_id, owner, candidates);
        },
        .GhostBlock => |ghost| try collectBody(allocator, file, ghost.body, parent_owner, candidates),
        else => {},
    }
}

fn collectBody(
    allocator: std.mem.Allocator,
    file: anytype,
    body_id: anytype,
    owner: []const u8,
    candidates: *std.ArrayList(Candidate),
) anyerror!void {
    for (file.body(body_id).statements) |stmt_id| try collectStatement(allocator, file, stmt_id, owner, candidates);
}

fn collectStatement(
    allocator: std.mem.Allocator,
    file: anytype,
    stmt_id: anytype,
    owner: []const u8,
    candidates: *std.ArrayList(Candidate),
) anyerror!void {
    switch (file.statement(stmt_id).*) {
        .Assert => |stmt| try appendCandidate(candidates, allocator, .assert, stmt.range, stmt.source_fact_id, owner),
        .Assume => |stmt| try appendCandidate(candidates, allocator, .assume, stmt.range, stmt.source_fact_id, owner),
        .Havoc => |stmt| try appendCandidate(candidates, allocator, .havoc, stmt.range, stmt.source_fact_id, owner),
        .If => |stmt| {
            try collectBody(allocator, file, stmt.then_body, owner, candidates);
            if (stmt.else_body) |else_body| try collectBody(allocator, file, else_body, owner, candidates);
        },
        .While => |stmt| {
            const loop_owner = try std.fmt.allocPrint(allocator, "{s}/while@{d}", .{ owner, stmt.range.start });
            for (stmt.invariants) |invariant| try appendCandidate(candidates, allocator, .loop_invariant, invariant.range, invariant.source_fact_id, loop_owner);
            try collectBody(allocator, file, stmt.body, loop_owner, candidates);
        },
        .For => |stmt| {
            const loop_owner = try std.fmt.allocPrint(allocator, "{s}/for@{d}", .{ owner, stmt.range.start });
            for (stmt.invariants) |invariant| try appendCandidate(candidates, allocator, .loop_invariant, invariant.range, invariant.source_fact_id, loop_owner);
            try collectBody(allocator, file, stmt.body, loop_owner, candidates);
        },
        .Switch => |stmt| {
            const loop_owner = try std.fmt.allocPrint(allocator, "{s}/switch@{d}", .{ owner, stmt.range.start });
            for (stmt.invariants) |invariant| try appendCandidate(candidates, allocator, .loop_invariant, invariant.range, invariant.source_fact_id, loop_owner);
            for (stmt.arms) |arm| try collectBody(allocator, file, arm.body, loop_owner, candidates);
            if (stmt.else_body) |else_body| try collectBody(allocator, file, else_body, loop_owner, candidates);
        },
        .Try => |stmt| {
            try collectBody(allocator, file, stmt.try_body, owner, candidates);
            if (stmt.catch_clause) |catch_clause| try collectBody(allocator, file, catch_clause.body, owner, candidates);
        },
        .Block => |stmt| try collectBody(allocator, file, stmt.body, owner, candidates),
        .LabeledBlock => |stmt| try collectBody(allocator, file, stmt.body, owner, candidates),
        else => {},
    }
}

fn appendCandidate(
    candidates: *std.ArrayList(Candidate),
    allocator: std.mem.Allocator,
    kind: accounting.SourceFactKind,
    range: anytype,
    source_fact_id: ?u32,
    owner: []const u8,
) !void {
    try candidates.append(allocator, .{
        .kind = kind,
        .range = .{ .start = range.start, .end = range.end },
        .source_fact_id = source_fact_id,
        .owner_hint = try allocator.dupe(u8, owner),
    });
}

fn factKindForClause(kind: anytype) ?accounting.SourceFactKind {
    return switch (kind) {
        .requires => .requires,
        .guard => .guard,
        .ensures => .ensures,
        .ensures_ok => .ensures_ok,
        .ensures_err => .ensures_err,
        .modifies => .modifies,
        .invariant => null,
    };
}

fn bestDeclaredMatch(candidate: Candidate, declared: []const accounting.DeclaredSite, used: []const bool) ?usize {
    const source_fact_id = candidate.source_fact_id orelse return null;
    for (declared, used, 0..) |site, is_used, index| {
        if (is_used or site.key.kind != candidate.kind) continue;
        if (site.key.range_start != source_fact_id or
            site.key.range_start != candidate.range.start or
            site.key.range_end != candidate.range.end or
            !std.mem.eql(u8, site.key.owner, candidate.owner_hint)) continue;
        return index;
    }
    return null;
}

fn cloneKey(allocator: std.mem.Allocator, key: accounting.SiteKey) !accounting.SiteKey {
    var result = key;
    result.path = try allocator.dupe(u8, key.path);
    result.owner = try allocator.dupe(u8, key.owner);
    return result;
}

fn lessCandidate(_: void, lhs: Candidate, rhs: Candidate) bool {
    if (lhs.range.start != rhs.range.start) return lhs.range.start < rhs.range.start;
    if (lhs.range.end != rhs.range.end) return lhs.range.end < rhs.range.end;
    return @intFromEnum(lhs.kind) < @intFromEnum(rhs.kind);
}

fn lessMatched(_: void, lhs: Matched, rhs: Matched) bool {
    const owner_order = std.mem.order(u8, lhs.template_owner, rhs.template_owner);
    if (owner_order != .eq) return owner_order == .lt;
    if (lhs.site.key.range_start != rhs.site.key.range_start) return lhs.site.key.range_start < rhs.site.key.range_start;
    return lhs.site.id < rhs.site.id;
}

fn rootOwner(owner: []const u8) []const u8 {
    var end = owner.len;
    inline for (.{ "/while@", "/for@", "/switch@" }) |marker| {
        if (std.mem.indexOf(u8, owner, marker)) |index| end = @min(end, index);
    }
    return owner[0..end];
}

fn expressionRange(expr: anytype) @TypeOf(switch (expr) {
    inline else => |payload| payload.range,
}) {
    return switch (expr) {
        inline else => |payload| payload.range,
    };
}
