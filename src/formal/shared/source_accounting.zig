//! Data-only source-formal accounting manifest.
//!
//! This module deliberately does not import syntax, sema, comptime, MLIR, Z3,
//! the CLI, or artifact policy. Producers populate rows; the compiler-kernel
//! checker in `../kernel/source_accounting_gate.zig` owns validation and policy.

pub const schema_version: u32 = 3;

pub const Id = u32;
pub const SiteId = Id;
pub const TemplateId = Id;
pub const ExpansionId = Id;
pub const UseId = Id;
pub const ControlNodeId = Id;
pub const ControlEdgeId = Id;
pub const FoldId = Id;
pub const EvidenceId = Id;
pub const HandlingId = Id;

/// Global evidence-ID partitions. Producer-local IDs occupy the low 28 bits;
/// the high nibble prevents unrelated proof producers from aliasing rows.
pub const EvidenceNamespace = enum(u4) {
    obligation = 1,
    assumption = 2,
    query = 3,
    runtime_check = 4,
    frame_result = 5,
    state_effect = 6,
    concrete_predicate = 7,
};

/// Global handling-ID partitions. Concrete and symbolic handling rows share
/// one kernel index, so producer-local IDs must be namespaced before entering
/// a manifest.
pub const HandlingNamespace = enum(u4) {
    symbolic = 1,
    concrete = 2,
};

pub fn namespacedEvidenceId(namespace: EvidenceNamespace, producer_id: u32) !EvidenceId {
    if (producer_id > 0x0fff_ffff) return error.SourceAccountingProducerIdOverflow;
    return (@as(u32, @intFromEnum(namespace)) << 28) | producer_id;
}

pub fn namespacedHandlingId(namespace: HandlingNamespace, producer_id: u32) !HandlingId {
    if (producer_id > 0x0fff_ffff) return error.SourceAccountingHandlingIdOverflow;
    return (@as(u32, @intFromEnum(namespace)) << 28) | producer_id;
}

pub const CompilationMode = enum(u8) {
    verified_full,
    verified_basic,
    unverified_emit,
};

pub const FactOrigin = enum(u8) {
    source_syntax,
    semantic_generated,
};

pub const SourceFactKind = enum(u8) {
    requires,
    guard,
    ensures,
    ensures_ok,
    ensures_err,
    loop_invariant,
    contract_invariant,
    assert,
    assume,
    havoc,
    modifies,
    refinement_guard,
    runtime_guard,
};

pub const UseRole = enum(u8) {
    proof_target,
    assumption_context,
    frame_directive,
    state_directive,
    runtime_condition,
};

pub const ActivationReason = enum(u8) {
    runtime_owner,
    symbolic_call,
    speculative_fold,
    required_comptime,
};

/// The semantic view instantiated from one owner. Runtime and compile-time
/// bodies have control-flow graphs; a symbolic call boundary has only the
/// caller/callee contract uses that exist at that boundary.
pub const TemplateActivation = enum(u8) {
    runtime_body,
    symbolic_call_boundary,
    comptime_body,
};

pub const ExpansionDisposition = enum(u8) {
    symbolic,
    fold_committed,
    fold_abandoned_to_symbolic,
    rejected,
};

pub const HandlingKind = enum(u8) {
    symbolic,
    concrete_true,
    runtime_enforced,
    control_eliminated,
    assumption_incorporated,
    frame_validated,
    state_effect_incorporated,
    reduced_scope_excluded,
    verification_disabled,
    rejected,
};

pub const QueryKind = enum(u8) {
    obligation,
    loop_invariant_step,
    loop_body_safety,
    loop_invariant_post,
    guard_satisfy,
    guard_violate,
    other,
};

pub const ControlNodeKind = enum(u8) {
    entry,
    statement,
    branch,
    loop_head,
    loop_body,
    return_exit,
    success_exit,
    error_exit,
};

pub const ControlEdgeKind = enum(u8) {
    next,
    branch_true,
    branch_false,
    loop_body,
    loop_exit,
    backedge,
    break_exit,
    continue_backedge,
    return_exit,
    success_exit,
    error_exit,
};

pub const TraceEventKind = enum(u8) {
    enter_node,
    take_edge,
    predicate_check,
    return_exit,
    break_exit,
    continue_backedge,
    success_exit,
    error_exit,
};

pub const FoldDisposition = enum(u8) {
    committed,
    abandoned,
};

pub const SourceRange = struct {
    file: []const u8,
    start: u32,
    end: u32,
};

/// Canonical structural identity. `path` is package/module relative, never an
/// absolute temporary path. `owner` includes the owner kind and canonical name.
pub const SiteKey = struct {
    path: []const u8,
    owner: []const u8,
    range_start: u32,
    range_end: u32,
    kind: SourceFactKind,
    ordinal: u32,
};

pub const DeclaredSite = struct {
    id: SiteId,
    key: SiteKey,
    label: ?[]const u8 = null,
};

pub const TypedSite = struct {
    id: SiteId,
    origin: FactOrigin,
    kind: SourceFactKind,
    key: SiteKey,
    source_fact_id: ?u32 = null,
    declared_site_id: ?SiteId = null,
    derivation_id: ?Id = null,
};

pub const GeneratedFactDerivation = struct {
    id: Id,
    site_id: SiteId,
    semantic_rule: []const u8,
    anchor: SourceRange,
    parent_identity: []const u8,
    ordinal: u32,
};

pub const UseTemplate = struct {
    site_id: SiteId,
    role: UseRole,
    /// Owner-scoped structural control slot. Expansions map the slot to their
    /// concrete `ControlNodeId`; null means the use is owner-level.
    control_node_slot: ?u32 = null,
};

/// Owner-scoped control-flow shape produced by semantic analysis.  Fold
/// adapters may instantiate these slots, but they may not invent a different
/// graph after observing the evaluator's path.
pub const ControlNodeTemplate = struct {
    slot: u32,
    kind: ControlNodeKind,
    range: SourceRange,
    attached_use_ordinals: []const u32 = &.{},
};

pub const ControlEdgeTemplate = struct {
    slot: u32,
    from_slot: u32,
    to_slot: u32,
    kind: ControlEdgeKind,
};

/// An owner template contains every semantic use syntactically/semantically
/// owned by one function, contract application, trait method, or comptime body.
pub const OwnerTemplate = struct {
    id: TemplateId,
    owner_key: []const u8,
    activation: TemplateActivation,
    uses: []const UseTemplate,
    control_nodes: []const ControlNodeTemplate = &.{},
    control_edges: []const ControlEdgeTemplate = &.{},
    entry_slot: ?u32 = null,
    terminal_slots: []const u32 = &.{},
};

pub fn templateActivation(reason: ActivationReason) TemplateActivation {
    return switch (reason) {
        .runtime_owner => .runtime_body,
        .symbolic_call => .symbolic_call_boundary,
        .speculative_fold, .required_comptime => .comptime_body,
    };
}

pub const Expansion = struct {
    id: ExpansionId,
    template_id: TemplateId,
    parent_expansion_id: ?ExpansionId = null,
    replacement_expansion_id: ?ExpansionId = null,
    activation: ActivationReason,
    disposition: ExpansionDisposition,
    root_runtime_owner: []const u8,
    folded_call_site_chain: []const SourceRange = &.{},
    imported_module: ?[]const u8 = null,
    generic_bindings: []const []const u8 = &.{},
    trait_implementation: ?[]const u8 = null,
    trait_method: ?[]const u8 = null,
    /// Canonical presentation key assembled from the structural dimensions
    /// above. Repeated evaluations at the same structural call site may share
    /// this string. Consumers key rows by `id` and validate every structural
    /// dimension; `identity` alone is not a unique database key.
    identity: []const u8,
};

pub const SourceFactUse = struct {
    id: UseId,
    site_id: SiteId,
    expansion_id: ExpansionId,
    template_ordinal: u32,
    role: UseRole,
    control_node_id: ?ControlNodeId = null,
};

pub const ControlNode = struct {
    id: ControlNodeId,
    expansion_id: ExpansionId,
    slot: u32,
    kind: ControlNodeKind,
    range: SourceRange,
    attached_use_ids: []const UseId = &.{},
};

pub const ControlEdge = struct {
    id: ControlEdgeId,
    expansion_id: ExpansionId,
    from: ControlNodeId,
    to: ControlNodeId,
    kind: ControlEdgeKind,
};

pub const TraceEvent = struct {
    kind: TraceEventKind,
    node_id: ?ControlNodeId = null,
    edge_id: ?ControlEdgeId = null,
    use_id: ?UseId = null,
    predicate_value: ?bool = null,
};

pub const FoldRecord = struct {
    id: FoldId,
    expansion_id: ExpansionId,
    entry_node_id: ControlNodeId,
    terminal_node_id: ControlNodeId,
    disposition: FoldDisposition,
    events: []const TraceEvent,
};

pub const PredicateEvent = struct {
    id: EvidenceId,
    fold_id: FoldId,
    use_id: UseId,
    node_id: ControlNodeId,
    value: bool,
};

pub const CoveredEvidence = struct {
    id: EvidenceId,
    /// Stable identity assigned by the producing obligation/query manifest.
    /// `id` is the globally namespaced accounting identity.
    producer_id: u32 = 0,
    covered_use_ids: []const UseId,
};

pub const QueryEvidence = struct {
    id: EvidenceId,
    producer_id: u32 = 0,
    kind: QueryKind,
    covered_use_ids: []const UseId,
};

pub const ValidationEvidence = struct {
    id: EvidenceId,
    producer_id: u32 = 0,
    covered_use_ids: []const UseId,
    valid: bool,
};

pub const HandlingRecord = struct {
    id: HandlingId,
    use_id: UseId,
    kind: HandlingKind,
    obligation_ids: []const EvidenceId = &.{},
    assumption_ids: []const EvidenceId = &.{},
    query_ids: []const EvidenceId = &.{},
    runtime_check_ids: []const EvidenceId = &.{},
    frame_result_ids: []const EvidenceId = &.{},
    state_effect_ids: []const EvidenceId = &.{},
    predicate_event_ids: []const EvidenceId = &.{},
    fold_id: ?FoldId = null,
    control_event_index: ?u32 = null,
    rejection_reason: ?[]const u8 = null,
};

pub const SourceInventory = struct {
    declared_sites: []const DeclaredSite = &.{},
    typed_sites: []const TypedSite = &.{},
    generated_fact_derivations: []const GeneratedFactDerivation = &.{},
    owner_templates: []const OwnerTemplate = &.{},
    expansions: []const Expansion = &.{},
    uses: []const SourceFactUse = &.{},
    control_nodes: []const ControlNode = &.{},
    control_edges: []const ControlEdge = &.{},
};

pub const ComptimeEvidence = struct {
    folds: []const FoldRecord = &.{},
    predicate_events: []const PredicateEvent = &.{},
    handlings: []const HandlingRecord = &.{},
};

pub const SymbolicEvidence = struct {
    obligations: []const CoveredEvidence = &.{},
    assumptions: []const CoveredEvidence = &.{},
    queries: []const QueryEvidence = &.{},
    runtime_checks: []const CoveredEvidence = &.{},
    frame_results: []const ValidationEvidence = &.{},
    state_effects: []const ValidationEvidence = &.{},
    handlings: []const HandlingRecord = &.{},
};

pub const Manifest = struct {
    version: u32 = schema_version,
    inventory: SourceInventory = .{},
    comptime_evidence: ComptimeEvidence = .{},
    symbolic: SymbolicEvidence = .{},
};

pub fn requiredRoles(kind: SourceFactKind) []const UseRole {
    return switch (kind) {
        .requires => &.{ .proof_target, .assumption_context, .runtime_condition },
        .guard => &.{ .proof_target, .runtime_condition },
        .ensures, .ensures_ok, .ensures_err => &.{ .proof_target, .assumption_context },
        .loop_invariant, .contract_invariant => &.{ .proof_target, .assumption_context },
        .assert => &.{ .proof_target, .runtime_condition },
        .assume => &.{.assumption_context},
        .havoc => &.{.state_directive},
        .modifies => &.{.frame_directive},
        .refinement_guard => &.{ .proof_target, .runtime_condition },
        .runtime_guard => &.{.runtime_condition},
    };
}

/// Roles required when a source fact is instantiated in one semantic view.
/// The union across the three views is `requiredRoles(kind)`. Repeated roles
/// are meaningful occurrences: runtime contract invariants have independent
/// success- and error-exit proof targets.
pub fn templateRoles(kind: SourceFactKind, activation: TemplateActivation) []const UseRole {
    return switch (activation) {
        .runtime_body => switch (kind) {
            .requires => &.{ .assumption_context, .runtime_condition },
            .guard => &.{ .proof_target, .runtime_condition },
            .ensures, .ensures_ok, .ensures_err => &.{.proof_target},
            .loop_invariant => &.{ .proof_target, .assumption_context },
            .contract_invariant => &.{ .proof_target, .proof_target, .assumption_context },
            .assert => &.{ .proof_target, .runtime_condition },
            .assume => &.{.assumption_context},
            .havoc => &.{.state_directive},
            .modifies => &.{.frame_directive},
            .refinement_guard => &.{ .proof_target, .runtime_condition },
            .runtime_guard => &.{.runtime_condition},
        },
        .symbolic_call_boundary => switch (kind) {
            .requires => &.{.proof_target},
            .ensures, .ensures_ok, .ensures_err => &.{.assumption_context},
            .guard,
            .loop_invariant,
            .contract_invariant,
            .assert,
            .assume,
            .havoc,
            .modifies,
            .refinement_guard,
            .runtime_guard,
            => &.{},
        },
        .comptime_body => requiredRoles(kind),
    };
}

pub fn roleRequired(kind: SourceFactKind, role: UseRole) bool {
    for (requiredRoles(kind)) |required| if (required == role) return true;
    return false;
}

test "source-accounting enum tags remain byte sized" {
    const std = @import("std");
    inline for (.{
        CompilationMode,
        FactOrigin,
        SourceFactKind,
        UseRole,
        ActivationReason,
        TemplateActivation,
        ExpansionDisposition,
        HandlingKind,
        QueryKind,
        ControlNodeKind,
        ControlEdgeKind,
        TraceEventKind,
        FoldDisposition,
        EvidenceNamespace,
        HandlingNamespace,
    }) |T| try std.testing.expectEqual(@as(usize, 1), @sizeOf(T));
}

test "handling namespaces cannot alias producer ids" {
    const std = @import("std");
    const symbolic = try namespacedHandlingId(.symbolic, 7);
    const concrete = try namespacedHandlingId(.concrete, 7);
    try std.testing.expect(symbolic != concrete);
    try std.testing.expectError(
        error.SourceAccountingHandlingIdOverflow,
        namespacedHandlingId(.concrete, 0x1000_0000),
    );
}

test "required role vocabulary is closed and nonempty" {
    const std = @import("std");
    inline for (std.meta.fields(SourceFactKind)) |field| {
        const kind: SourceFactKind = @enumFromInt(field.value);
        try std.testing.expect(requiredRoles(kind).len != 0);
        for (requiredRoles(kind), 0..) |role, index| {
            try std.testing.expect(roleRequired(kind, role));
            for (requiredRoles(kind)[index + 1 ..]) |other| try std.testing.expect(role != other);
        }
    }
}

test "activation-scoped roles conserve the complete role vocabulary" {
    const std = @import("std");
    inline for (std.meta.fields(SourceFactKind)) |field| {
        const kind: SourceFactKind = @enumFromInt(field.value);
        for (requiredRoles(kind)) |required| {
            var found = false;
            inline for (std.meta.fields(TemplateActivation)) |activation_field| {
                const activation: TemplateActivation = @enumFromInt(activation_field.value);
                for (templateRoles(kind, activation)) |role| found = found or role == required;
            }
            try std.testing.expect(found);
        }
        inline for (std.meta.fields(TemplateActivation)) |activation_field| {
            const activation: TemplateActivation = @enumFromInt(activation_field.value);
            for (templateRoles(kind, activation)) |role| try std.testing.expect(roleRequired(kind, role));
        }
    }
}
