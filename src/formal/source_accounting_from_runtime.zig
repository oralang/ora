//! Runtime-owner expansion inventory from structurally identified HIR owners.
//!
//! Sema owns the expected source-fact uses. HIR contributes only the concrete
//! owner activation and specialization dimensions. This adapter instantiates
//! the complete expected template before any MLIR/Z3 evidence is considered,
//! so a fact operation that disappears during lowering remains a missing
//! handling instead of disappearing from the accounting denominator.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");
const from_mlir = @import("obligation_from_mlir.zig");
const from_package = @import("source_accounting_from_package.zig");
const symbolic_adapter = @import("source_accounting_from_mlir.zig");

pub const IdStarts = struct {
    expansion: accounting.ExpansionId = 1,
    use: accounting.UseId = 1,
    control_node: accounting.ControlNodeId = 1,
    control_edge: accounting.ControlEdgeId = 1,
};

pub const ExpansionBinding = struct {
    runtime_source_op_id: usize,
    expansion_id: accounting.ExpansionId,
    module_path: []const u8,
    owner_key: []const u8,
    activation: accounting.TemplateActivation,
    runtime_symbol: []const u8,
    specialization_bindings: []const []const u8 = &.{},
    trait_implementation: ?[]const u8 = null,
    trait_method: ?[]const u8 = null,
};

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    expansions: []const accounting.Expansion,
    uses: []const accounting.SourceFactUse,
    control_nodes: []const accounting.ControlNode,
    control_edges: []const accounting.ControlEdge,
    bindings: []const ExpansionBinding,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn expansionForOwner(
        self: *const Result,
        module_path: []const u8,
        owner_key: []const u8,
        activation: accounting.TemplateActivation,
        runtime_symbol: []const u8,
        specialization_bindings: []const []const u8,
        trait_implementation: ?[]const u8,
        trait_method: ?[]const u8,
    ) ?accounting.ExpansionId {
        for (self.bindings) |binding| {
            if (binding.activation == activation and
                std.mem.eql(u8, binding.module_path, module_path) and
                std.mem.eql(u8, binding.owner_key, owner_key) and
                std.mem.eql(u8, binding.runtime_symbol, runtime_symbol) and
                stringSlicesEqual(binding.specialization_bindings, specialization_bindings) and
                optionalStringsEqual(binding.trait_implementation, trait_implementation) and
                optionalStringsEqual(binding.trait_method, trait_method))
            {
                return binding.expansion_id;
            }
        }
        return null;
    }
};

pub const SymbolicBindings = struct {
    arena: std.heap.ArenaAllocator,
    bindings: []const symbolic_adapter.Binding,
    producers: symbolic_adapter.ProducerInventory,

    pub fn deinit(self: *SymbolicBindings) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const MutableSymbolicBinding = struct {
    use_id: accounting.UseId,
    handling_kind: accounting.HandlingKind,
    obligation_ids: std.ArrayList(u32) = .empty,
    assumption_ids: std.ArrayList(u32) = .empty,
    query_ids: std.ArrayList(u32) = .empty,
    runtime_check_ids: std.ArrayList(u32) = .empty,
    frame_result_ids: std.ArrayList(u32) = .empty,
    state_effect_ids: std.ArrayList(u32) = .empty,
};

pub fn collect(
    allocator: std.mem.Allocator,
    package: *const from_package.Result,
    runtime_owners: []const from_mlir.RuntimeOwnerBinding,
    starts: IdStarts,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .expansions = &.{},
        .uses = &.{},
        .control_nodes = &.{},
        .control_edges = &.{},
        .bindings = &.{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    const order = try arena.alloc(usize, runtime_owners.len);
    for (order, 0..) |*index, value| index.* = value;
    std.mem.sort(usize, order, runtime_owners, lessRuntimeOwner);

    var expansions: std.ArrayList(accounting.Expansion) = .empty;
    var uses: std.ArrayList(accounting.SourceFactUse) = .empty;
    var nodes: std.ArrayList(accounting.ControlNode) = .empty;
    var edges: std.ArrayList(accounting.ControlEdge) = .empty;
    var bindings: std.ArrayList(ExpansionBinding) = .empty;

    var next_expansion_id = starts.expansion;
    var next_use_id = starts.use;
    var next_node_id = starts.control_node;
    var next_edge_id = starts.control_edge;
    for (order, 0..) |runtime_index, ordered_index| {
        const runtime_owner = runtime_owners[runtime_index];
        const activation = std.meta.stringToEnum(accounting.TemplateActivation, runtime_owner.template_activation) orelse
            return error.UnknownSourceAccountingTemplateActivation;
        if (activation != .runtime_body) return error.InvalidRuntimeSourceAccountingActivation;
        if (ordered_index != 0 and runtimeOwnerDimensionsEqual(runtime_owners[order[ordered_index - 1]], runtime_owner)) {
            return error.DuplicateRuntimeSourceAccountingExpansion;
        }
        const runtime_symbol = runtime_owner.symbol orelse
            return error.RuntimeSourceAccountingOwnerMissingSymbol;

        const module = package.moduleForSourcePath(runtime_owner.module_path) orelse
            return error.UnknownRuntimeSourceAccountingModule;
        const template_id = module.templateForOwner(runtime_owner.owner_key, activation) orelse
            return error.UnknownRuntimeSourceAccountingTemplate;
        const template = package.template(template_id) orelse
            return error.UnknownRuntimeSourceAccountingTemplate;

        const expansion_id = next_expansion_id;
        next_expansion_id = std.math.add(accounting.ExpansionId, next_expansion_id, 1) catch
            return error.SourceAccountingExpansionIdOverflow;
        const identity = try runtimeIdentity(arena, module.canonical_path, runtime_owner);
        try expansions.append(arena, .{
            .id = expansion_id,
            .template_id = template.id,
            .activation = .runtime_owner,
            .disposition = .symbolic,
            .root_runtime_owner = try std.fmt.allocPrint(arena, "{s}::{s}", .{ module.canonical_path, runtime_symbol }),
            .generic_bindings = try cloneStrings(arena, runtime_owner.specialization_bindings),
            .trait_implementation = try cloneOptionalString(arena, runtime_owner.trait_implementation),
            .trait_method = try cloneOptionalString(arena, runtime_owner.trait_method),
            .identity = identity,
        });

        const first_use_id = next_use_id;
        const node_ids = try arena.alloc(accounting.ControlNodeId, template.control_nodes.len);
        for (node_ids) |*node_id| {
            node_id.* = next_node_id;
            next_node_id = std.math.add(accounting.ControlNodeId, next_node_id, 1) catch
                return error.SourceAccountingControlNodeIdOverflow;
        }
        for (template.uses, 0..) |use_template, ordinal| {
            const use_id = next_use_id;
            next_use_id = std.math.add(accounting.UseId, next_use_id, 1) catch
                return error.SourceAccountingUseIdOverflow;
            try uses.append(arena, .{
                .id = use_id,
                .site_id = use_template.site_id,
                .expansion_id = expansion_id,
                .template_ordinal = @intCast(ordinal),
                .role = use_template.role,
                .control_node_id = if (use_template.control_node_slot) |slot|
                    nodeIdForSlot(template, node_ids, slot) orelse return error.UnknownRuntimeSourceAccountingControlSlot
                else
                    null,
            });
        }
        for (template.control_nodes, node_ids) |node_template, node_id| {
            const attached = try arena.alloc(accounting.UseId, node_template.attached_use_ordinals.len);
            for (node_template.attached_use_ordinals, attached) |ordinal, *use_id| {
                if (ordinal >= template.uses.len) return error.InvalidRuntimeSourceAccountingUseOrdinal;
                use_id.* = std.math.add(accounting.UseId, first_use_id, ordinal) catch
                    return error.SourceAccountingUseIdOverflow;
            }
            try nodes.append(arena, .{
                .id = node_id,
                .expansion_id = expansion_id,
                .slot = node_template.slot,
                .kind = node_template.kind,
                .range = try cloneRange(arena, node_template.range),
                .attached_use_ids = attached,
            });
        }
        for (template.control_edges) |edge_template| {
            const edge_id = next_edge_id;
            next_edge_id = std.math.add(accounting.ControlEdgeId, next_edge_id, 1) catch
                return error.SourceAccountingControlEdgeIdOverflow;
            try edges.append(arena, .{
                .id = edge_id,
                .expansion_id = expansion_id,
                .from = nodeIdForSlot(template, node_ids, edge_template.from_slot) orelse
                    return error.UnknownRuntimeSourceAccountingControlSlot,
                .to = nodeIdForSlot(template, node_ids, edge_template.to_slot) orelse
                    return error.UnknownRuntimeSourceAccountingControlSlot,
                .kind = edge_template.kind,
            });
        }

        try bindings.append(arena, .{
            .runtime_source_op_id = runtime_owner.source_op_id,
            .expansion_id = expansion_id,
            .module_path = try arena.dupe(u8, runtime_owner.module_path),
            .owner_key = try arena.dupe(u8, runtime_owner.owner_key),
            .activation = activation,
            .runtime_symbol = try arena.dupe(u8, runtime_symbol),
            .specialization_bindings = try cloneStrings(arena, runtime_owner.specialization_bindings),
            .trait_implementation = try cloneOptionalString(arena, runtime_owner.trait_implementation),
            .trait_method = try cloneOptionalString(arena, runtime_owner.trait_method),
        });
    }

    result.expansions = try expansions.toOwnedSlice(arena);
    result.uses = try uses.toOwnedSlice(arena);
    result.control_nodes = try nodes.toOwnedSlice(arena);
    result.control_edges = try edges.toOwnedSlice(arena);
    result.bindings = try bindings.toOwnedSlice(arena);
    return result;
}

/// Join explicit source-fact operation identities to the expected runtime
/// uses instantiated above. The join is structural: module, owner activation,
/// specialization, source-fact id, fact kind, and role must all agree. The
/// adapter never matches formulas, diagnostics, or source-line containment.
pub fn collectSymbolicBindings(
    allocator: std.mem.Allocator,
    package: *const from_package.Result,
    runtime: *const Result,
    formal: *const from_mlir.CollectResult,
) !SymbolicBindings {
    var result: SymbolicBindings = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .bindings = &.{},
        .producers = .{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    var mutable: std.ArrayList(MutableSymbolicBinding) = .empty;
    for (formal.source_fact_bindings) |op_binding| {
        const module_path = op_binding.module_path orelse continue;
        const owner_key = op_binding.owner_key orelse continue;
        const activation_text = op_binding.template_activation orelse continue;
        const activation = std.meta.stringToEnum(accounting.TemplateActivation, activation_text) orelse
            return error.UnknownSourceAccountingTemplateActivation;
        if (activation != .runtime_body) continue;
        const module = package.moduleForSourcePath(module_path) orelse
            return error.UnknownRuntimeSourceAccountingModule;
        const kind = std.meta.stringToEnum(accounting.SourceFactKind, op_binding.kind) orelse
            return error.UnknownRuntimeSourceAccountingFactKind;
        const site_id = module.siteForSourceFact(op_binding.source_fact_id, kind) orelse
            return error.UnknownRuntimeSourceAccountingSite;

        if (kind == .contract_invariant and op_binding.runtime_symbol == null) {
            // A contract invariant is emitted once in the contract body, while
            // the verifier instantiates its entry/success/error obligations for
            // every concrete runtime function in that contract.
            for (runtime.bindings) |binding| {
                if (binding.activation != activation or
                    !std.mem.eql(u8, binding.module_path, module_path) or
                    !ownerIsDirectContractFunction(binding.owner_key, owner_key)) continue;
                try appendSourceFactBindings(
                    arena,
                    &mutable,
                    runtime.uses,
                    binding.expansion_id,
                    site_id,
                    kind,
                    op_binding,
                );
            }
            continue;
        }

        const expansion_id = runtime.expansionForOwner(
            module_path,
            owner_key,
            activation,
            op_binding.runtime_symbol orelse return error.RuntimeSourceAccountingFactMissingSymbol,
            op_binding.specialization_bindings,
            op_binding.trait_implementation,
            op_binding.trait_method,
        ) orelse continue;
        try appendSourceFactBindings(arena, &mutable, runtime.uses, expansion_id, site_id, kind, op_binding);
    }

    std.mem.sort(MutableSymbolicBinding, mutable.items, {}, struct {
        fn less(_: void, lhs: MutableSymbolicBinding, rhs: MutableSymbolicBinding) bool {
            return lhs.use_id < rhs.use_id;
        }
    }.less);
    const bindings = try arena.alloc(symbolic_adapter.Binding, mutable.items.len);
    for (mutable.items, bindings) |*source, *binding| {
        sortIds(source.obligation_ids.items);
        sortIds(source.assumption_ids.items);
        sortIds(source.query_ids.items);
        sortIds(source.runtime_check_ids.items);
        sortIds(source.frame_result_ids.items);
        sortIds(source.state_effect_ids.items);
        binding.* = .{
            .use_id = source.use_id,
            .handling_kind = source.handling_kind,
            .obligation_ids = try source.obligation_ids.toOwnedSlice(arena),
            .assumption_ids = try source.assumption_ids.toOwnedSlice(arena),
            .query_ids = try source.query_ids.toOwnedSlice(arena),
            .runtime_check_ids = try source.runtime_check_ids.toOwnedSlice(arena),
            .frame_result_ids = try source.frame_result_ids.toOwnedSlice(arena),
            .state_effect_ids = try source.state_effect_ids.toOwnedSlice(arena),
        };
    }
    const runtime_check_ids = try arena.alloc(u32, formal.runtime_check_producers.len);
    for (formal.runtime_check_producers, runtime_check_ids) |producer, *id| id.* = producer.id;
    sortIds(runtime_check_ids);
    if (runtime_check_ids.len > 1) {
        for (runtime_check_ids[1..], runtime_check_ids[0 .. runtime_check_ids.len - 1]) |current, previous| {
            if (current == previous) return error.DuplicateSourceAccountingRuntimeCheckProducer;
        }
    }
    const state_effect_ids = try arena.alloc(u32, formal.state_effect_producers.len);
    for (formal.state_effect_producers, state_effect_ids) |producer, *id| id.* = producer.id;
    sortIds(state_effect_ids);
    if (state_effect_ids.len > 1) {
        for (state_effect_ids[1..], state_effect_ids[0 .. state_effect_ids.len - 1]) |current, previous| {
            if (current == previous) return error.DuplicateSourceAccountingStateEffectProducer;
        }
    }

    result.bindings = bindings;
    result.producers = .{
        .runtime_check_ids = runtime_check_ids,
        .state_effect_ids = state_effect_ids,
    };
    return result;
}

fn appendSourceFactBindings(
    allocator: std.mem.Allocator,
    rows: *std.ArrayList(MutableSymbolicBinding),
    uses: []const accounting.SourceFactUse,
    expansion_id: accounting.ExpansionId,
    site_id: accounting.SiteId,
    kind: accounting.SourceFactKind,
    op_binding: from_mlir.SourceFactOpBinding,
) !void {
    for (op_binding.roles) |role_text| {
        const role = std.meta.stringToEnum(accounting.UseRole, role_text) orelse
            return error.UnknownRuntimeSourceAccountingUseRole;
        var match_count: usize = 0;
        for (uses) |use| {
            if (use.expansion_id != expansion_id or use.site_id != site_id or use.role != role) continue;
            match_count += 1;
            const handling_kind = handlingForRole(role);
            const index = try mutableBindingIndex(allocator, rows, use.id, handling_kind);
            const row = &rows.items[index];
            switch (role) {
                .proof_target => {
                    try appendIdsUnique(allocator, &row.obligation_ids, op_binding.obligation_ids);
                    try appendIdsUnique(allocator, &row.query_ids, op_binding.query_ids);
                },
                .assumption_context => {
                    try appendIdsUnique(allocator, &row.assumption_ids, op_binding.assumption_ids);
                    // Loop and contract contexts are incorporated directly in
                    // prepared queries rather than as owner-global assumptions.
                    if (op_binding.assumption_ids.len == 0) {
                        try appendIdsUnique(allocator, &row.query_ids, op_binding.query_ids);
                    }
                },
                .runtime_condition => try appendIdsUnique(allocator, &row.runtime_check_ids, op_binding.runtime_check_ids),
                .frame_directive => try appendIdsUnique(allocator, &row.frame_result_ids, op_binding.obligation_ids),
                .state_directive => try appendIdsUnique(allocator, &row.state_effect_ids, op_binding.state_effect_ids),
            }
        }
        if (match_count == 0 or (kind != .contract_invariant and match_count != 1)) {
            return error.UnknownOrAmbiguousRuntimeSourceAccountingUse;
        }
    }
}

fn ownerIsDirectContractFunction(owner_key: []const u8, contract_owner_key: []const u8) bool {
    if (!std.mem.startsWith(u8, owner_key, contract_owner_key)) return false;
    const suffix = owner_key[contract_owner_key.len..];
    if (!std.mem.startsWith(u8, suffix, "/function:")) return false;
    return std.mem.indexOfScalarPos(u8, suffix, 1, '/') == null;
}

fn handlingForRole(role: accounting.UseRole) accounting.HandlingKind {
    return switch (role) {
        .proof_target => .symbolic,
        .assumption_context => .assumption_incorporated,
        .runtime_condition => .runtime_enforced,
        .frame_directive => .frame_validated,
        .state_directive => .state_effect_incorporated,
    };
}

fn mutableBindingIndex(
    allocator: std.mem.Allocator,
    rows: *std.ArrayList(MutableSymbolicBinding),
    use_id: accounting.UseId,
    handling_kind: accounting.HandlingKind,
) !usize {
    for (rows.items, 0..) |row, index| {
        if (row.use_id != use_id) continue;
        if (row.handling_kind != handling_kind) return error.ConflictingRuntimeSourceAccountingHandling;
        return index;
    }
    try rows.append(allocator, .{ .use_id = use_id, .handling_kind = handling_kind });
    return rows.items.len - 1;
}

fn appendIdsUnique(allocator: std.mem.Allocator, target: *std.ArrayList(u32), ids: []const u32) !void {
    for (ids) |id| {
        var present = false;
        for (target.items) |current| if (current == id) {
            present = true;
            break;
        };
        if (!present) try target.append(allocator, id);
    }
}

fn sortIds(ids: []u32) void {
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
}

fn lessRuntimeOwner(runtime_owners: []const from_mlir.RuntimeOwnerBinding, lhs_index: usize, rhs_index: usize) bool {
    const lhs = runtime_owners[lhs_index];
    const rhs = runtime_owners[rhs_index];
    var order = std.mem.order(u8, lhs.module_path, rhs.module_path);
    if (order != .eq) return order == .lt;
    order = std.mem.order(u8, lhs.owner_key, rhs.owner_key);
    if (order != .eq) return order == .lt;
    order = compareOptionalStrings(lhs.symbol, rhs.symbol);
    if (order != .eq) return order == .lt;
    order = compareStringSlices(lhs.specialization_bindings, rhs.specialization_bindings);
    if (order != .eq) return order == .lt;
    order = compareOptionalStrings(lhs.trait_implementation, rhs.trait_implementation);
    if (order != .eq) return order == .lt;
    order = compareOptionalStrings(lhs.trait_method, rhs.trait_method);
    if (order != .eq) return order == .lt;
    return lhs.op_ordinal < rhs.op_ordinal;
}

fn runtimeOwnerDimensionsEqual(lhs: from_mlir.RuntimeOwnerBinding, rhs: from_mlir.RuntimeOwnerBinding) bool {
    return std.mem.eql(u8, lhs.module_path, rhs.module_path) and
        std.mem.eql(u8, lhs.owner_key, rhs.owner_key) and
        optionalStringsEqual(lhs.symbol, rhs.symbol) and
        stringSlicesEqual(lhs.specialization_bindings, rhs.specialization_bindings) and
        optionalStringsEqual(lhs.trait_implementation, rhs.trait_implementation) and
        optionalStringsEqual(lhs.trait_method, rhs.trait_method);
}

fn compareStringSlices(lhs: []const []const u8, rhs: []const []const u8) std.math.Order {
    for (lhs[0..@min(lhs.len, rhs.len)], rhs[0..@min(lhs.len, rhs.len)]) |left, right| {
        const order = std.mem.order(u8, left, right);
        if (order != .eq) return order;
    }
    return std.math.order(lhs.len, rhs.len);
}

fn stringSlicesEqual(lhs: []const []const u8, rhs: []const []const u8) bool {
    return compareStringSlices(lhs, rhs) == .eq;
}

fn compareOptionalStrings(lhs: ?[]const u8, rhs: ?[]const u8) std.math.Order {
    if (lhs == null and rhs == null) return .eq;
    if (lhs == null) return .lt;
    if (rhs == null) return .gt;
    return std.mem.order(u8, lhs.?, rhs.?);
}

fn optionalStringsEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    return compareOptionalStrings(lhs, rhs) == .eq;
}

fn nodeIdForSlot(
    template: accounting.OwnerTemplate,
    node_ids: []const accounting.ControlNodeId,
    slot: u32,
) ?accounting.ControlNodeId {
    for (template.control_nodes, node_ids) |node, node_id| if (node.slot == slot) return node_id;
    return null;
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

fn runtimeIdentity(
    allocator: std.mem.Allocator,
    canonical_module_path: []const u8,
    runtime_owner: from_mlir.RuntimeOwnerBinding,
) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writeIdentityPart(writer, "runtime");
    try writeIdentityPart(writer, canonical_module_path);
    try writeIdentityPart(writer, runtime_owner.owner_key);
    try writeIdentityPart(writer, runtime_owner.symbol orelse return error.RuntimeSourceAccountingOwnerMissingSymbol);
    for (runtime_owner.specialization_bindings) |binding| try writeIdentityPart(writer, binding);
    try writeIdentityPart(writer, runtime_owner.trait_implementation orelse "");
    try writeIdentityPart(writer, runtime_owner.trait_method orelse "");
    return out.toOwnedSlice();
}

fn writeIdentityPart(writer: anytype, value: []const u8) !void {
    try writer.print("{d}:", .{value.len});
    try writer.writeAll(value);
}
