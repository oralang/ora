//! Backend-neutral verification obligation manifest.
//!
//! This is not a replacement IR for Ora MLIR, HIR, or Z3. It is the stable
//! manifest layer that records which compiler facts must be checked, where they
//! came from, and which backend is responsible for checking them.
//!
//! Keep this file data-oriented. Builders, MLIR walkers, Z3 encoding, Lean
//! emission, and optimization/simplification belong in separate adapter files.

const std = @import("std");

pub const Id = u32;
pub const TermId = u32;

pub const ObligationSet = struct {
    obligations: []const Obligation = &.{},
    assumptions: []const Assumption = &.{},
    queries: []const VerificationQuery = &.{},
    proof_artifacts: []const ProofArtifact = &.{},
    diagnostics: []const ObligationDiagnostic = &.{},
    terms: []const Term = &.{},

    pub fn hasBlockingDiagnostic(self: ObligationSet) bool {
        for (self.diagnostics) |diagnostic| {
            if (diagnostic.blocks_artifacts) return true;
        }
        return false;
    }

    pub fn artifactDecision(self: ObligationSet) ArtifactDecision {
        for (self.diagnostics) |diagnostic| {
            if (diagnostic.blocks_artifacts) return .{ .blocked = .blocking_diagnostic };
        }
        self.validateTermReferences() catch return .{ .blocked = .invalid_dependency };
        self.validateIdReferences() catch return .{ .blocked = .invalid_dependency };

        for (self.queries) |query| {
            if (query.result) |result| {
                if (result.degraded) return .{ .blocked = .degraded_query };
                if (result.vacuity_unknown) return .{ .blocked = .unknown_query };
                if (result.status == .unknown) return .{ .blocked = .unknown_query };
                if (!querySucceeded(query, result)) {
                    return .{ .blocked = if (query.backend == .lean) .lean_required_failure else .failed_query };
                }
            } else if (query.obligation_ids.len > 0) {
                return .{ .blocked = if (query.backend == .lean) .lean_required_failure else .missing_proof };
            }
        }

        for (self.obligations) |item| {
            if (!item.blocksArtifacts()) continue;
            if (!self.hasSuccessfulProofFor(item)) {
                return .{ .blocked = if (item.required_backend == .lean) .lean_required_failure else .missing_proof };
            }
        }

        return .allowed;
    }

    pub fn validateTermReferences(self: ObligationSet) !void {
        for (self.assumptions) |assumption| {
            if (assumption.formula) |formula| try self.validateFormulaRef(formula);
        }
        for (self.obligations) |obligation| {
            try self.validateKindTermRefs(obligation.kind);
        }
        for (self.terms) |term| {
            try self.validateTerm(term);
        }
    }

    fn validateKindTermRefs(self: ObligationSet, kind: Kind) !void {
        switch (kind) {
            .logical => |logical| try self.validateFormulaRef(logical.formula),
            .runtime_guard => |guard| try self.validateFormulaRef(guard.formula),
            .resource => |resource| if (resource.amount) |amount| try self.validateFormulaRef(amount),
            .quantifier,
            .type_wf,
            .type_relation,
            .region_relation,
            .effect_frame,
            .filtered_input,
            .backend_fact,
            => {},
        }
    }

    fn validateFormulaRef(self: ObligationSet, formula: FormulaRef) !void {
        switch (formula) {
            .origin_value => {},
            .term => |id| try self.validateTermId(id),
        }
    }

    fn validateTerm(self: ObligationSet, term: Term) !void {
        switch (term) {
            .bool_lit,
            .int_lit,
            .variable,
            .result,
            .place_read,
            => {},
            .old => |id| try self.validateTermId(id),
            .unary => |unary| try self.validateTermId(unary.operand),
            .binary => |binary| {
                try self.validateTermId(binary.lhs);
                try self.validateTermId(binary.rhs);
            },
            .refinement_predicate => |predicate| {
                try self.validateTermId(predicate.value);
                for (predicate.args) |arg| try self.validateTermId(arg);
            },
            .quantified => |quantified| {
                if (quantified.condition) |condition| try self.validateTermId(condition);
                try self.validateTermId(quantified.body);
            },
        }
    }

    fn validateTermId(self: ObligationSet, id: TermId) !void {
        if (id >= self.terms.len) return error.InvalidTermReference;
    }

    pub fn validateIdReferences(self: ObligationSet) !void {
        for (self.obligations) |item| {
            for (item.dependencies) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            for (item.derived_from) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            switch (item.kind) {
                .filtered_input => |filtered| for (filtered.accepted_by) |id| {
                    if (!self.obligationIdExists(id) and !self.assumptionIdExists(id)) return error.InvalidDependency;
                },
                else => {},
            }
        }
        for (self.proof_artifacts) |artifact| {
            for (artifact.obligation_ids) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
        }
        for (self.queries) |query| {
            for (query.obligation_ids) |id| {
                if (!self.obligationIdExists(id)) return error.InvalidDependency;
            }
            for (query.assumption_ids) |id| {
                if (!self.assumptionIdExists(id)) return error.InvalidDependency;
            }
            if (query.proof_artifact_id) |artifact_id| {
                const artifact = self.proofArtifactById(artifact_id) orelse return error.InvalidDependency;
                if (query.backend != .lean) return error.InvalidDependency;
                for (query.obligation_ids) |id| {
                    if (!containsId(artifact.obligation_ids, id)) return error.InvalidDependency;
                }
            }
        }
    }

    fn obligationIdExists(self: ObligationSet, id: Id) bool {
        for (self.obligations) |item| {
            if (item.id == id) return true;
        }
        return false;
    }

    fn assumptionIdExists(self: ObligationSet, id: Id) bool {
        for (self.assumptions) |item| {
            if (item.id == id) return true;
        }
        return false;
    }

    fn proofArtifactById(self: ObligationSet, id: Id) ?ProofArtifact {
        for (self.proof_artifacts) |item| {
            if (item.id == id) return item;
        }
        return null;
    }

    fn hasSuccessfulProofFor(self: ObligationSet, item: Obligation) bool {
        for (self.queries) |query| {
            if (item.required_backend) |backend| {
                if (query.backend != backend) continue;
            }
            if (!containsId(query.obligation_ids, item.id)) continue;
            const result = query.result orelse continue;
            if (querySucceeded(query, result)) return true;
        }
        return false;
    }
};

pub const Obligation = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    kind: Kind,
    artifact_policy: ArtifactPolicy = .blocks_verified_artifacts,
    required_backend: ?VerificationBackend = null,
    dependencies: []const Id = &.{},
    derived_from: []const Id = &.{},

    pub fn blocksArtifacts(self: Obligation) bool {
        return self.artifact_policy == .blocks_verified_artifacts;
    }
};

pub const VerificationQuery = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    backend: VerificationBackend = .unspecified,
    kind: VerificationQueryKind,
    logical_role: ?LogicalRole = null,
    guard_id: ?[]const u8 = null,
    obligation_ids: []const Id = &.{},
    assumption_ids: []const Id = &.{},
    fragment: VerificationQueryFragment = .unknown,
    solver_logic: VerificationSolverLogic = .all,
    constraint_count: u32 = 0,
    smtlib_hash: ?u64 = null,
    proof_artifact_id: ?Id = null,
    result: ?VerificationQueryResult = null,
};

pub const VerificationBackend = enum(u8) {
    unspecified,
    z3,
    lean,
};

pub const VerificationQueryKind = enum(u8) {
    base,
    obligation,
    loop_invariant_step,
    loop_body_safety,
    loop_invariant_post,
    guard_satisfy,
    guard_violate,
};

pub const VerificationQueryFragment = enum(u8) {
    unknown,
    qf_bv,
    qf_bv_array,
    aufbv,
    aufbv_quantifiers,
    other,
};

pub const VerificationSolverLogic = enum(u8) {
    all,
    qf_aufbv,
};

pub const VerificationQueryResult = struct {
    status: VerificationQueryStatus,
    vacuous: bool = false,
    vacuity_unknown: bool = false,
    degraded: bool = false,
};

pub const VerificationQueryStatus = enum(u8) {
    sat,
    unsat,
    unknown,
    proved,
    failed,
};

pub const VerificationQuerySummary = struct {
    base: u32 = 0,
    obligation: u32 = 0,
    loop_invariant_step: u32 = 0,
    loop_body_safety: u32 = 0,
    loop_invariant_post: u32 = 0,
    guard_satisfy: u32 = 0,
    guard_violate: u32 = 0,

    pub fn add(self: *VerificationQuerySummary, kind: VerificationQueryKind) void {
        switch (kind) {
            .base => self.base += 1,
            .obligation => self.obligation += 1,
            .loop_invariant_step => self.loop_invariant_step += 1,
            .loop_body_safety => self.loop_body_safety += 1,
            .loop_invariant_post => self.loop_invariant_post += 1,
            .guard_satisfy => self.guard_satisfy += 1,
            .guard_violate => self.guard_violate += 1,
        }
    }

    pub fn fromQueries(queries: []const VerificationQuery) VerificationQuerySummary {
        var summary: VerificationQuerySummary = .{};
        for (queries) |query| summary.add(query.kind);
        return summary;
    }
};

pub const Assumption = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    phase: Phase,
    origin: Origin,
    kind: AssumptionKind,
    formula: ?FormulaRef = null,
};

pub const ProofArtifact = struct {
    id: Id,
    owner: Owner,
    source: SourceRef,
    kind: ProofArtifactKind = .userland_lean,
    module_name: []const u8,
    theorem_name: []const u8,
    path: ?[]const u8 = null,
    content_hash: ?u64 = null,
    obligation_ids: []const Id = &.{},
};

pub const ProofArtifactKind = enum(u8) {
    userland_lean,
};

pub const ObligationDiagnostic = struct {
    kind: DiagnosticKind,
    source: SourceRef,
    message: []const u8,
    blocks_artifacts: bool = true,
};

pub const DiagnosticKind = enum(u8) {
    unsupported,
    missing_type,
    missing_region,
    missing_effect_path,
    missing_formula,
    invalid_dependency,
};

pub const Phase = enum(u8) {
    sema,
    hir,
    ora_mlir,
    sir_mlir,
    sinora,
    report,
};

pub const ArtifactPolicy = enum(u8) {
    blocks_verified_artifacts,
    diagnostic_only,
};

pub const ArtifactDecision = union(enum) {
    allowed,
    blocked: ArtifactBlockReason,

    pub fn isAllowed(self: ArtifactDecision) bool {
        return switch (self) {
            .allowed => true,
            .blocked => false,
        };
    }
};

pub const ArtifactBlockReason = enum(u8) {
    blocking_diagnostic,
    invalid_dependency,
    unknown_query,
    degraded_query,
    failed_query,
    missing_proof,
    lean_required_failure,
};

fn containsId(ids: []const Id, needle: Id) bool {
    for (ids) |id| {
        if (id == needle) return true;
    }
    return false;
}

fn querySucceeded(query: VerificationQuery, result: VerificationQueryResult) bool {
    return switch (result.status) {
        .proved => true,
        .failed, .unknown => false,
        .sat => switch (query.backend) {
            .lean => false,
            else => switch (query.kind) {
                .base, .guard_satisfy => true,
                .obligation,
                .loop_invariant_step,
                .loop_body_safety,
                .loop_invariant_post,
                .guard_violate,
                => false,
            },
        },
        .unsat => switch (query.backend) {
            .lean => false,
            else => switch (query.kind) {
                .obligation,
                .loop_invariant_step,
                .loop_body_safety,
                .loop_invariant_post,
                .guard_violate,
                => true,
                .base, .guard_satisfy => false,
            },
        },
    };
}

fn expectArtifactBlocked(expected: ArtifactBlockReason, decision: ArtifactDecision) !void {
    switch (decision) {
        .allowed => return error.TestUnexpectedResult,
        .blocked => |actual| try std.testing.expectEqual(expected, actual),
    }
}

pub const OwnerTag = enum(u8) {
    module,
    contract,
    function,
    trait_method,
    statement,
    backend,
};

pub const Owner = union(OwnerTag) {
    module: []const u8,
    contract: []const u8,
    function: FunctionOwner,
    trait_method: TraitMethodOwner,
    statement: StatementOwner,
    backend: BackendOwner,
};

pub const FunctionOwner = struct {
    module: ?[]const u8 = null,
    contract: ?[]const u8 = null,
    name: []const u8,
};

pub const TraitMethodOwner = struct {
    trait_name: []const u8,
    method_name: []const u8,
    impl_name: ?[]const u8 = null,
};

pub const StatementOwner = struct {
    function_name: []const u8,
    ordinal: u32,
};

pub const BackendOwner = struct {
    component: BackendComponent,
    name: []const u8,
};

pub const BackendComponent = enum(u8) {
    dispatcher,
    oratosir,
    sinora,
    artifact_policy,
};

pub const SourceRef = struct {
    file: ?[]const u8 = null,
    line: u32 = 0,
    column: u32 = 0,
    byte_start: u32 = 0,
    byte_end: u32 = 0,

    pub fn generated() SourceRef {
        return .{};
    }
};

pub const OriginTag = enum(u8) {
    source,
    sema_fact,
    mlir_op,
    effect_slot,
    resource_op,
    backend_fact,
};

pub const Origin = union(OriginTag) {
    source,
    sema_fact: SemaFactRef,
    mlir_op: MlirStableRef,
    effect_slot: EffectRef,
    resource_op: ResourceOpRef,
    backend_fact: BackendFactRef,
};

pub const SemaFactRef = struct {
    kind: []const u8,
    ordinal: u32,
};

pub const MlirStableRef = struct {
    op_name: []const u8,
    symbol: ?[]const u8 = null,
    ordinal: u32 = 0,
};

pub const EffectRef = struct {
    slot: PlaceRef,
    access: EffectAccess,
};

pub const EffectAccess = enum(u8) {
    read,
    write,
    read_write,
    lock,
    unlock,
};

pub const ResourceOpRef = struct {
    op: ResourceOperation,
    domain: []const u8,
    ordinal: u32 = 0,
};

pub const BackendFactRef = struct {
    component: BackendComponent,
    fact: []const u8,
    ordinal: u32 = 0,
};

pub const KindTag = enum(u8) {
    logical,
    runtime_guard,
    type_wf,
    type_relation,
    region_relation,
    effect_frame,
    resource,
    quantifier,
    filtered_input,
    backend_fact,
};

pub const Kind = union(KindTag) {
    logical: LogicalGoal,
    runtime_guard: RuntimeGuardGoal,
    type_wf: TypeWellFormedGoal,
    type_relation: TypeRelationGoal,
    region_relation: RegionRelationGoal,
    effect_frame: EffectFrameGoal,
    resource: ResourceGoal,
    quantifier: QuantifierGoal,
    filtered_input: FilteredInputGoal,
    backend_fact: BackendFactGoal,
};

pub const LogicalGoal = struct {
    role: LogicalRole,
    formula: FormulaRef,
    arithmetic_safety: ?ArithmeticSafetyKind = null,
};

pub const LogicalRole = enum(u8) {
    invariant,
    requires,
    callee_precondition,
    ensures,
    ensures_ok,
    ensures_err,
    assert,
    guard,
    loop_invariant,
    contract_invariant,
    arithmetic_safety,
    refinement,
    imported_callee_obligation,
    imported_callee_ensures,
};

pub const ArithmeticSafetyKind = enum(u8) {
    addition_overflow,
    subtraction_overflow,
    multiplication_overflow,
    power_overflow,
    negation_overflow,
    division_by_zero,
    shift_amount_bounds,
};

pub const RuntimeGuardGoal = struct {
    guard_id: []const u8,
    formula: FormulaRef,
    erasure: GuardErasurePolicy = .may_elide_if_proven,
};

pub const GuardErasurePolicy = enum(u8) {
    may_elide_if_proven,
    always_runtime,
};

pub const TypeWellFormedGoal = struct {
    ty: TypeRef,
};

pub const TypeRelationGoal = struct {
    relation: TypeRelation,
    lhs: TypeRef,
    rhs: TypeRef,
};

pub const TypeRelation = enum(u8) {
    eql,
    assignable,
};

pub const RegionRelationGoal = struct {
    relation: RegionRelation,
    from: RegionRef,
    to: RegionRef,
};

pub const RegionRelation = enum(u8) {
    assignable,
    eql,
};

pub const EffectFrameGoal = struct {
    relation: EffectFrameRelation,
    declared: []const PlaceRef = &.{},
    actual: []const PlaceRef = &.{},
};

pub const EffectFrameRelation = enum(u8) {
    write_covered_by_modifies,
    read_preserved_by_frame,
    lock_covers_write,
    external_call_frame,
};

pub const ResourceGoal = struct {
    op: ResourceOperation,
    domain: []const u8,
    source: ?PlaceRef = null,
    destination: ?PlaceRef = null,
    amount: ?FormulaRef = null,
    property: ResourceProperty,
};

pub const ResourceOperation = enum(u8) {
    move,
    create,
    destroy,
};

pub const ResourceProperty = enum(u8) {
    amount_non_negative,
    source_sufficient,
    destination_no_overflow,
    same_place_net_zero,
    conservation,
    modifies_covered,
};

pub const QuantifierGoal = struct {
    quantifier: Quantifier,
    variable: []const u8,
    binder_type: TypeRef,
    binder_sort: QuantifierBinderSort,
    fragment: VerificationQueryFragment,
    pattern_status: QuantifierPatternStatus,
    degradation: ?QuantifierDegradation = null,
};

pub const QuantifierBinderSort = enum(u8) {
    bool,
    bit_vector,
    byte_sequence,
    opaque_unknown,
};

pub const QuantifierPatternStatus = enum(u8) {
    explicit,
    synthesized,
    absent,
};

pub const QuantifierDegradation = enum(u8) {
    unsupported_binder_type,
    malformed_binder_width,
};

pub const FilteredInputGoal = struct {
    value: VarRef,
    sink: PlaceRef,
    accepted_by: []const Id = &.{},
};

pub const BackendFactGoal = struct {
    component: BackendComponent,
    property: BackendProperty,
};

pub const BackendProperty = enum(u8) {
    complete,
    disjoint,
    ordered,
    preserves_selector_behavior,
    no_unknown_strategy,
    dependency_valid,
};

pub const AssumptionKind = enum(u8) {
    requires,
    assume,
    path_assume,
    env_assume,
    binding,
    two_state_linkage,
    frame,
    loop_invariant,
    callee_obligation,
    callee_ensures,
    ghost_axiom,
    goal,
};

pub const FormulaRefTag = enum(u8) {
    origin_value,
    term,
};

pub const FormulaRef = union(FormulaRefTag) {
    /// Formula is still owned by a compiler IR operation/value. Adapters may use
    /// this to encode directly from MLIR without duplicating that IR here.
    origin_value: ValueRef,
    /// Formula was projected into the small proof-term fragment below for
    /// serialization or Lean/Z3 export. This is not an optimization target.
    term: TermId,
};

pub const ValueRef = struct {
    origin: Origin,
    kind: ValueRefKind = .result,
    index: u32 = 0,
};

pub const ValueRefKind = enum(u8) {
    result,
    operand,
    derived,
};

pub const TermTag = enum(u8) {
    bool_lit,
    int_lit,
    variable,
    old,
    result,
    place_read,
    unary,
    binary,
    refinement_predicate,
    quantified,
};

pub const Term = union(TermTag) {
    bool_lit: bool,
    int_lit: IntegerLiteralTerm,
    variable: VarRef,
    old: TermId,
    result,
    place_read: PlaceRef,
    unary: UnaryTerm,
    binary: BinaryTerm,
    refinement_predicate: RefinementPredicateTerm,
    quantified: QuantifiedTerm,
};

pub const IntegerLiteralTerm = struct {
    value: []const u8,
    ty: ?TypeRef = null,
};

pub const VarRef = struct {
    name: []const u8,
    ty: ?TypeRef = null,
    region: ?RegionRef = null,
};

pub const UnaryTerm = struct {
    op: UnaryOp,
    operand: TermId,
};

pub const UnaryOp = enum(u8) {
    not,
    neg,
};

pub const BinaryTerm = struct {
    op: BinaryOp,
    lhs: TermId,
    rhs: TermId,
};

pub const BinaryOp = enum(u8) {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    add,
    sub,
    mul,
    div,
    mod,
    and_,
    or_,
    implies,
};

pub const RefinementPredicateTerm = struct {
    name: []const u8,
    value: TermId,
    args: []const TermId = &.{},
};

pub const QuantifiedTerm = struct {
    quantifier: Quantifier,
    binder: VarRef,
    condition: ?TermId = null,
    body: TermId,
};

pub const Quantifier = enum(u8) {
    forall,
    exists,
};

pub const PlaceRef = struct {
    root: []const u8,
    region: RegionRef,
    fields: []const []const u8 = &.{},
    keys: []const PlaceKey = &.{},
};

pub const PlaceKeyTag = enum(u8) {
    parameter,
    comptime_parameter,
    comptime_range_parameter,
    constant,
    msg_sender,
    tx_origin,
    unknown,
};

pub const PlaceKey = union(PlaceKeyTag) {
    parameter: u32,
    comptime_parameter: u32,
    comptime_range_parameter: u32,
    constant: []const u8,
    msg_sender,
    tx_origin,
    unknown,
};

pub const RegionRef = enum(u8) {
    none,
    storage,
    memory,
    transient,
    calldata,
};

pub const TypeRefTag = enum(u8) {
    spelling,
    compiler_type_id,
};

pub const TypeRef = union(TypeRefTag) {
    spelling: []const u8,
    compiler_type_id: u32,
};

test "manifest enum tags stay byte-sized" {
    inline for (.{
        DiagnosticKind,
        Phase,
        ArtifactPolicy,
        ArtifactBlockReason,
        OwnerTag,
        BackendComponent,
        OriginTag,
        EffectAccess,
        KindTag,
        VerificationBackend,
        VerificationQueryKind,
        VerificationQueryFragment,
        VerificationSolverLogic,
        VerificationQueryStatus,
        ProofArtifactKind,
        LogicalRole,
        ArithmeticSafetyKind,
        GuardErasurePolicy,
        ValueRefKind,
        TypeRelation,
        RegionRelation,
        EffectFrameRelation,
        ResourceOperation,
        ResourceProperty,
        QuantifierBinderSort,
        QuantifierPatternStatus,
        QuantifierDegradation,
        BackendProperty,
        AssumptionKind,
        FormulaRefTag,
        TermTag,
        UnaryOp,
        BinaryOp,
        Quantifier,
        PlaceKeyTag,
        RegionRef,
        TypeRefTag,
    }) |T| {
        try std.testing.expectEqual(@as(usize, 1), @sizeOf(T));
    }
}

test "verification query summary counts solver-facing categories" {
    const queries = [_]VerificationQuery{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "func.func", .symbol = "transfer" } },
            .kind = .base,
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
            .kind = .obligation,
            .logical_role = .ensures,
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .{ .mlir_op = .{ .op_name = "ora.refinement_guard", .symbol = "transfer" } },
            .kind = .guard_violate,
            .guard_id = "guard:transfer:amount",
        },
    };

    const summary = VerificationQuerySummary.fromQueries(&queries);
    try std.testing.expectEqual(@as(u32, 1), summary.base);
    try std.testing.expectEqual(@as(u32, 1), summary.obligation);
    try std.testing.expectEqual(@as(u32, 0), summary.guard_satisfy);
    try std.testing.expectEqual(@as(u32, 1), summary.guard_violate);
}

test "blocking diagnostics gate obligation set" {
    const diagnostics = [_]ObligationDiagnostic{
        .{
            .kind = .unsupported,
            .source = .generated(),
            .message = "unsupported quantified binder",
        },
    };
    const set: ObligationSet = .{ .diagnostics = &diagnostics };
    try std.testing.expect(set.hasBlockingDiagnostic());
}

test "term references validate canonical formulas" {
    const args = [_]TermId{1};
    const terms = [_]Term{
        .{ .variable = .{ .name = "amount", .ty = .{ .spelling = "u256" } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .refinement_predicate = .{ .name = "MinValue", .value = 0, .args = &args } },
    };
    const obligation: Obligation = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "deposit" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "refinement_guard", .ordinal = 0 } },
        .kind = .{ .logical = .{
            .role = .refinement,
            .formula = .{ .term = 2 },
        } },
    };
    const set: ObligationSet = .{ .obligations = &.{obligation}, .terms = &terms };
    try set.validateTermReferences();
}

test "term reference validation rejects dangling ids" {
    const terms = [_]Term{
        .{ .unary = .{ .op = .not, .operand = 1 } },
    };
    const set: ObligationSet = .{ .terms = &terms };
    try std.testing.expectError(error.InvalidTermReference, set.validateTermReferences());
}

test "diagnostic-only entries do not block artifacts" {
    const diagnostics = [_]ObligationDiagnostic{
        .{
            .kind = .unsupported,
            .source = .generated(),
            .message = "informational",
            .blocks_artifacts = false,
        },
    };
    const set: ObligationSet = .{ .diagnostics = &diagnostics };
    try std.testing.expect(!set.hasBlockingDiagnostic());
}

test "artifact policy allows blocking obligation only after successful proof query" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unsat },
        },
    };
    const set: ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &.{.{ .bool_lit = true }},
    };

    try std.testing.expect(set.artifactDecision().isAllowed());
}

test "artifact policy blocks unknown and degraded query results" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const unknown_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unknown },
        },
    };
    const degraded_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unsat, .degraded = true },
        },
    };

    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .queries = &unknown_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.degraded_query, (ObligationSet{
        .obligations = &obligations,
        .queries = &degraded_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "artifact policy blocks invalid dependencies and missing proofs" {
    const invalid_dependencies = [_]Id{42};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const invalid_obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .dependencies = &invalid_dependencies,
        },
    };

    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &invalid_obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.missing_proof, (ObligationSet{
        .obligations = &obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "artifact policy treats Lean-required proof gaps as hard failures" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const failed_lean_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .failed },
        },
    };
    const proved_lean_queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .proved },
        },
    };

    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .obligations = &obligations,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.lean_required_failure, (ObligationSet{
        .obligations = &obligations,
        .queries = &failed_lean_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .queries = &proved_lean_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision().isAllowed());
}

test "userland Lean proof artifact attaches to a required obligation" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{ .file = "proofs/ERC20/Transfer.lean", .line = 1 },
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .path = "proofs/ERC20/Transfer.lean",
            .content_hash = 0x1234,
            .obligation_ids = &obligation_ids,
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
    };
    try std.testing.expect((ObligationSet{
        .obligations = &obligations,
        .queries = &queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision().isAllowed());
}

test "userland Lean proof attachment fails closed when missing or mismatched" {
    const obligation_ids = [_]Id{1};
    const other_obligation_ids = [_]Id{2};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "mint" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 1 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "mint" } },
            .source = .generated(),
            .module_name = "ERC20.Mint",
            .theorem_name = "mint_supply",
            .obligation_ids = &other_obligation_ids,
        },
    };
    const missing_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 99,
            .result = .{ .status = .proved },
        },
    };
    const mismatched_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
    };
    const z3_with_artifact_queries = [_]VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .unsat },
        },
    };

    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &missing_artifact_queries,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &mismatched_artifact_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
    try expectArtifactBlocked(.invalid_dependency, (ObligationSet{
        .obligations = &obligations,
        .queries = &z3_with_artifact_queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "userland Lean proof artifact does not bypass Z3 unknown or degradation" {
    const obligation_ids = [_]Id{1};
    const obligations = [_]Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]ProofArtifact{
        .{
            .id = 10,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .obligation_ids = &obligation_ids,
        },
    };
    const queries = [_]VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .proof_artifact_id = 10,
            .result = .{ .status = .proved },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .result = .{ .status = .unknown },
        },
    };

    try expectArtifactBlocked(.unknown_query, (ObligationSet{
        .obligations = &obligations,
        .queries = &queries,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    }).artifactDecision());
}

test "obligation can point at MLIR origin without proof term" {
    const obligation: Obligation = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .{ .file = "erc20.ora", .line = 10, .column = 5 },
        .phase = .ora_mlir,
        .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .origin_value = .{
                .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer" } },
            } },
        } },
    };

    try std.testing.expect(obligation.blocksArtifacts());
}

test "derived obligations keep provenance instead of rewriting originals" {
    const derived = [_]Id{ 1, 2 };
    const obligation: Obligation = .{
        .id = 3,
        .owner = .{ .backend = .{ .component = .dispatcher, .name = "erc20" } },
        .source = .generated(),
        .phase = .sinora,
        .origin = .{ .backend_fact = .{ .component = .dispatcher, .fact = "selector_table_complete" } },
        .kind = .{ .backend_fact = .{
            .component = .dispatcher,
            .property = .complete,
        } },
        .derived_from = &derived,
    };

    try std.testing.expectEqualSlices(Id, &derived, obligation.derived_from);
}
