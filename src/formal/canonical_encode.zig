//! Canonical obligation manifest to Z3 encoding adapter.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const z3 = z3_verification.z3_c;
const Context = z3_verification.Z3Context;
const Solver = z3_verification.Z3Solver;
const obligation = @import("obligation.zig");
const fixture = @import("formal_test_fixture.zig");
const ct = @import("canonical_types.zig");
const support = @import("canonical_support.zig");

const CanonicalPlaceSymbolKind = ct.CanonicalPlaceSymbolKind;
const CanonicalPromotionShape = ct.CanonicalPromotionShape;
const CanonicalQueryState = ct.CanonicalQueryState;
const CanonicalUnsupportedReason = ct.CanonicalUnsupportedReason;
const CheckStatus = ct.CheckStatus;
const EncodeError = ct.EncodeError;
const LoopQueryCheck = ct.LoopQueryCheck;
const QueryHash = ct.QueryHash;
const TypeInfo = ct.TypeInfo;
const canonical_promotion_table = ct.canonical_promotion_table;
const expectArgCount = ct.expectArgCount;
const refinement_builtin_map = ct.refinement_builtin_map;
const quantifierBinderTypeInfo = ct.quantifierBinderTypeInfo;
const typeInfo = ct.typeInfo;
const typeInfoIsU256 = ct.typeInfoIsU256;
const u256TypeInfo = ct.u256TypeInfo;
const expectedInfoForBinaryOperands = support.expectedInfoForBinaryOperands;
const requireSupportedScalarPlace = support.requireSupportedScalarPlace;
const staticTermTypeInfo = support.staticTermTypeInfo;
const termContainsResult = support.termContainsResult;
const typeInfoEql = support.typeInfoEql;
const typeInfoSameZ3Sort = support.typeInfoSameZ3Sort;
const walkTerm = support.walkTerm;
const queryCanonicalCrosscheckRequiredByPolicy = support.queryCanonicalCrosscheckRequiredByPolicy;
const queryCanonicalPromotionShape = support.queryCanonicalPromotionShape;
const queryCanonicalRequiredModePromoted = support.queryCanonicalRequiredModePromoted;
const queryCanonicalSupport = support.queryCanonicalSupport;

const LoopStore = struct {
    info: TypeInfo,
    index_info: TypeInfo,
    domain: []const u8,
    current: z3.Z3_ast,
    entry: z3.Z3_ast,
};

const LoopRead = struct {
    id: obligation.FreeVarId,
    info: TypeInfo,
    value: z3.Z3_ast,
};

const LoopRoot = struct {
    name: []const u8,
    key: z3.Z3_ast,
};

const LoopEncodeState = struct {
    summary: obligation.LoopSummaryRow,
    stores: std.ArrayList(LoopStore) = .empty,
    reads: std.ArrayList(LoopRead) = .empty,
    roots: std.ArrayList(LoopRoot) = .empty,

    fn deinit(self: *LoopEncodeState, allocator: std.mem.Allocator) void {
        for (self.stores.items) |store| allocator.free(store.domain);
        self.stores.deinit(allocator);
        self.reads.deinit(allocator);
        self.roots.deinit(allocator);
    }
};

pub const Adapter = struct {
    context: *Context,
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    query_state: ?*CanonicalQueryState,
    loop_state: ?*LoopEncodeState = null,
    use_free_variable_identity_names: bool = false,

    pub const EncodeTermState = struct {
        expected: ?TypeInfo = null,
        enforce_expected_type: bool = false,
    };
    pub const State = EncodeTermState;
    pub const Result = EncodeError!z3.Z3_ast;

    pub fn init(
        context: *Context,
        allocator: std.mem.Allocator,
        set: obligation.ObligationSet,
    ) Adapter {
        return .{
            .context = context,
            .allocator = allocator,
            .set = set,
            .query_state = null,
        };
    }

    pub fn checkObligation(self: *Adapter, id: obligation.Id) EncodeError!CheckStatus {
        const query = (try self.findUniqueQueryForObligation(id)) orelse return error.UnknownQuery;
        return self.checkQueryRow(query);
    }

    pub fn checkQuery(self: *Adapter, id: obligation.Id) EncodeError!CheckStatus {
        const query = obligation.findById(self.set.queries, id) orelse return error.UnknownQuery;
        return self.checkQueryRow(query);
    }

    pub fn queryHash(self: *Adapter, id: obligation.Id) EncodeError!QueryHash {
        const query = obligation.findById(self.set.queries, id) orelse return error.UnknownQuery;
        return self.queryHashForRow(query);
    }

    pub fn queryHashForRow(self: *Adapter, query: obligation.VerificationQuery) EncodeError!QueryHash {
        try self.set.validateTermReferences();

        var constraints: std.ArrayList(z3.Z3_ast) = .empty;
        defer constraints.deinit(self.allocator);
        try self.appendQueryConstraints(&constraints, query);

        const smtlib_hash = try self.hashConstraints(constraints.items, query.solver_logic);
        return .{
            .constraint_count = std.math.cast(u32, constraints.items.len) orelse return error.Overflow,
            .smtlib_hash = smtlib_hash,
        };
    }

    /// Checks one of the canonical LoopVC obligations. These constraints are
    /// built from `LoopSummaryRow`; they do not reconstruct loop semantics from
    /// verifier annotations or source syntax.
    pub fn loopQueryHashForRow(
        self: *Adapter,
        query: obligation.VerificationQuery,
    ) EncodeError!QueryHash {
        try self.set.validateTermReferences();
        try self.set.validateIdReferences();

        var constraints: std.ArrayList(z3.Z3_ast) = .empty;
        defer constraints.deinit(self.allocator);
        try self.appendLoopQueryConstraints(&constraints, query);
        return self.loopQueryHash(constraints.items, query.solver_logic);
    }

    pub fn checkLoopQueryForRow(
        self: *Adapter,
        query: obligation.VerificationQuery,
    ) EncodeError!LoopQueryCheck {
        try self.set.validateTermReferences();
        try self.set.validateIdReferences();

        var constraints: std.ArrayList(z3.Z3_ast) = .empty;
        defer constraints.deinit(self.allocator);
        try self.appendLoopQueryConstraints(&constraints, query);

        const query_hash = try self.loopQueryHash(constraints.items, query.solver_logic);
        var solver = try Solver.init(self.context, self.allocator);
        defer solver.deinit();
        for (constraints.items) |constraint| try solver.assertChecked(constraint);
        const status: CheckStatus = switch (try solver.checkChecked()) {
            z3.Z3_L_FALSE => .proved,
            z3.Z3_L_TRUE => .disproved,
            else => .unknown,
        };
        return .{ .status = status, .query_hash = query_hash };
    }

    fn loopQueryHash(
        self: *Adapter,
        constraints: []const z3.Z3_ast,
        solver_logic: obligation.VerificationSolverLogic,
    ) EncodeError!QueryHash {
        return .{
            .constraint_count = std.math.cast(u32, constraints.len) orelse return error.Overflow,
            .smtlib_hash = try self.hashConstraints(constraints, solver_logic),
        };
    }

    fn checkQueryRow(self: *Adapter, query: obligation.VerificationQuery) EncodeError!CheckStatus {
        try self.set.validateTermReferences();

        var solver = try Solver.init(self.context, self.allocator);
        defer solver.deinit();

        var constraints: std.ArrayList(z3.Z3_ast) = .empty;
        defer constraints.deinit(self.allocator);
        try self.appendQueryConstraints(&constraints, query);

        for (constraints.items) |constraint| try solver.assertChecked(constraint);

        return switch (try solver.checkChecked()) {
            z3.Z3_L_FALSE => .proved,
            z3.Z3_L_TRUE => .disproved,
            else => .unknown,
        };
    }

    fn appendQueryConstraints(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        query: obligation.VerificationQuery,
    ) EncodeError!void {
        var query_state: CanonicalQueryState = .{};
        defer query_state.deinit(self.allocator);
        self.query_state = &query_state;
        defer self.query_state = null;

        if (query.obligation_ids.len != 1) return error.UnsupportedObligationKind;
        const target = obligation.findById(self.set.obligations, query.obligation_ids[0]) orelse return error.UnknownObligation;

        // The canonical byte-parity contract is order-sensitive over formal
        // ids, independent of the query builder's stored slice order.
        var sorted_assumption_ids: std.ArrayList(obligation.Id) = .empty;
        defer sorted_assumption_ids.deinit(self.allocator);
        const assumption_ids = if (query.assumption_ids.len <= 1) query.assumption_ids else blk: {
            try sorted_assumption_ids.appendSlice(self.allocator, query.assumption_ids);
            std.mem.sort(obligation.Id, sorted_assumption_ids.items, {}, std.sort.asc(obligation.Id));
            break :blk sorted_assumption_ids.items;
        };

        for (assumption_ids) |assumption_id| {
            const assumption = obligation.findById(self.set.assumptions, assumption_id) orelse return error.UnknownAssumption;
            if (assumption.formula) |formula| {
                const assumption_ast = try self.encodeFormula(formula);
                try self.appendSideConstraints(constraints);
                try constraints.append(self.allocator, assumption_ast);
            }
        }

        const goal = try self.goalCounterexampleFormula(target.kind);
        try self.appendSideConstraints(constraints);
        const negated_goal = z3.Z3_mk_not(self.context.ctx, goal);
        try self.context.checkNoError();
        try constraints.append(self.allocator, negated_goal);
    }

    fn appendLoopQueryConstraints(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        query: obligation.VerificationQuery,
    ) EncodeError!void {
        const summary_id = query.loop_summary_id orelse return error.InvalidLoopQuery;
        const summary = obligation.findById(self.set.loop_summaries, summary_id) orelse
            return error.UnknownLoopSummary;
        if (!summary.projectionSupported()) return error.InvalidLoopQuery;

        var query_state: CanonicalQueryState = .{};
        defer query_state.deinit(self.allocator);
        self.query_state = &query_state;
        defer self.query_state = null;
        var loop_state: LoopEncodeState = .{ .summary = summary };
        defer loop_state.deinit(self.allocator);
        self.loop_state = &loop_state;
        defer self.loop_state = null;
        self.use_free_variable_identity_names = true;
        defer self.use_free_variable_identity_names = false;

        try self.appendLoopAssumptions(constraints, query.assumption_ids);

        switch (query.kind) {
            .loop_base => {
                if (query.loop_edge_id != null) return error.InvalidLoopQuery;
                const goal = try self.encodeFormulaConjunction(constraints, summary.invariant_formulas);
                const replacements = try self.encodeFormulaValues(constraints, summary.init_formulas);
                defer self.allocator.free(replacements);
                const initialized_goal = try self.substituteLoopVariables(goal, summary.variables, replacements);
                try self.appendNegatedGoal(constraints, initialized_goal);
            },
            .loop_body_safety, .loop_invariant_step, .loop_invariant_post => {
                const edge = findLoopEdge(summary, query.loop_edge_id orelse return error.InvalidLoopQuery) orelse
                    return error.UnknownLoopEdge;
                try self.appendFormulaPremises(constraints, summary.invariant_formulas);
                const safety_goal = try self.appendLoopTimeline(
                    constraints,
                    edge,
                    query.kind == .loop_body_safety,
                );
                if (query.kind == .loop_body_safety) {
                    const goal = safety_goal orelse return error.InvalidLoopQuery;
                    try self.appendNegatedGoal(constraints, goal);
                    return;
                }
                const target = edge.target orelse return error.InvalidLoopQuery;
                const goal_formulas = if (query.kind == .loop_invariant_step) blk: {
                    if (target != .backedge) return error.InvalidLoopQuery;
                    break :blk summary.invariant_formulas;
                } else switch (target) {
                    .successor => summary.post_formulas,
                    .function_return, .error_return => edge.target_formulas,
                    .backedge, .explicit_revert => return error.InvalidLoopQuery,
                };
                const goal = try self.encodeFormulaConjunction(constraints, goal_formulas);
                const replacements = try self.encodeBindingValues(constraints, summary.variables, edge.next_state);
                defer self.allocator.free(replacements);
                const following_goal = try self.substituteLoopVariables(goal, summary.variables, replacements);
                try self.appendNegatedGoal(constraints, following_goal);
            },
            else => return error.InvalidLoopQuery,
        }
    }

    fn appendLoopTimeline(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        edge: obligation.LoopEdgeRow,
        collect_safety_goal: bool,
    ) EncodeError!?z3.Z3_ast {
        var prefix: std.ArrayList(z3.Z3_ast) = .empty;
        defer prefix.deinit(self.allocator);
        const premises = if (collect_safety_goal) &prefix else constraints;
        var safety = z3.Z3_mk_true(self.context.ctx);
        try self.context.checkNoError();
        for (edge.timeline) |event| switch (event.payload) {
            .path_assumption => |path| for (path.formulas) |expression| {
                try self.appendFormulaPremises(premises, &.{expression.formula});
            },
            .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |expression| {
                if (collect_safety_goal) {
                    const formula = try self.encodeFormulaWithSideConstraints(expression.formula);
                    const staged = try self.implyFromPremises(prefix.items, formula);
                    safety = z3.Z3_mk_and(
                        self.context.ctx,
                        2,
                        &[_]z3.Z3_ast{ safety, staged },
                    );
                    try self.context.checkNoError();
                    // Reaching a later checkpoint requires this checkpoint to
                    // have executed successfully, so it becomes part of that
                    // later checkpoint's prefix and never a global premise.
                    try prefix.append(self.allocator, formula);
                } else {
                    try self.appendFormulaPremises(constraints, &.{expression.formula});
                }
            },
            .storage_read => |read| {
                try self.applyLoopRead(read);
                try self.appendSideConstraints(premises);
            },
            .storage_write => |write| {
                try self.applyLoopWrite(write);
                try self.appendSideConstraints(premises);
            },
            .summary_dependency => |dependency| try self.appendLoopSummaryDependencyPremise(
                premises,
                edge,
                dependency,
            ),
        };
        return if (collect_safety_goal) safety else null;
    }

    fn implyFromPremises(
        self: *Adapter,
        premises: []const z3.Z3_ast,
        goal: z3.Z3_ast,
    ) EncodeError!z3.Z3_ast {
        if (premises.len == 0) return goal;
        var antecedent = premises[0];
        for (premises[1..]) |premise| {
            antecedent = z3.Z3_mk_and(
                self.context.ctx,
                2,
                &[_]z3.Z3_ast{ antecedent, premise },
            );
            try self.context.checkNoError();
        }
        const implication = z3.Z3_mk_implies(self.context.ctx, antecedent, goal);
        try self.context.checkNoError();
        return implication;
    }

    fn encodeFormulaWithSideConstraints(
        self: *Adapter,
        formula: obligation.FormulaRef,
    ) EncodeError!z3.Z3_ast {
        var result = try self.encodeFormula(formula);
        const state = try self.canonicalQueryState();
        for (state.side_constraints.items) |side| {
            result = z3.Z3_mk_and(
                self.context.ctx,
                2,
                &[_]z3.Z3_ast{ result, side },
            );
            try self.context.checkNoError();
        }
        state.side_constraints.clearRetainingCapacity();
        return result;
    }

    fn applyLoopRead(self: *Adapter, read: obligation.LoopStorageReadEvent) EncodeError!void {
        const state = self.loop_state orelse return error.InvalidLoopQuery;
        for (state.reads.items) |binding| {
            if (obligation.freeVarIdEql(binding.id, read.result)) return error.InvalidTimelineReadReference;
        }
        const info = try typeInfo(read.location.value_ty);
        const value = try self.selectLoopLocation(read.location, info, false);
        try state.reads.append(self.allocator, .{ .id = read.result, .info = info, .value = value });
    }

    fn applyLoopWrite(self: *Adapter, write: obligation.LoopStorageWriteEvent) EncodeError!void {
        const info = try typeInfo(write.location.value_ty);
        const value = try self.encodeValueFormula(write.value.formula);
        const resolved = try self.loopLocationStoreAndKey(write.location, info);
        resolved.store.current = z3.Z3_mk_store(
            self.context.ctx,
            resolved.store.current,
            resolved.key,
            value,
        );
        try self.context.checkNoError();
    }

    fn appendLoopSummaryDependencyPremise(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        parent_edge: obligation.LoopEdgeRow,
        dependency: obligation.LoopSummaryDependency,
    ) EncodeError!void {
        const inner = obligation.findById(self.set.loop_summaries, dependency.summary_id) orelse
            return error.UnknownLoopSummary;
        if (!inner.projectionSupported()) return error.InvalidLoopQuery;
        const exit_edge = uniqueLoopEdgeForTarget(inner, dependency.target) orelse
            return error.InvalidLoopQuery;

        // `initial` is existential in CertifiedControlRelation. Evaluating
        // the exact init values retains their typing and checked-operation
        // side conditions without inventing a relation to the exit state.
        const initial_values = try self.encodeFormulaValues(constraints, inner.init_formulas);
        self.allocator.free(initial_values);

        const current_values = try self.allocator.alloc(z3.Z3_ast, inner.variables.len);
        defer self.allocator.free(current_values);
        for (inner.variables, current_values, 0..) |variable, *value, variable_index| {
            const name = try std.fmt.allocPrint(
                self.allocator,
                "$ora.loop.dep.s{d}.d{d}.e{d}.v{d}.current",
                .{ dependency.summary_id, dependency.id, parent_edge.id, variable_index },
            );
            defer self.allocator.free(name);
            value.* = try self.namedConst(name, try self.sortForType(try typeInfo(variable.ty)));
        }

        try self.appendSubstitutedLoopPremises(
            constraints,
            inner.variables,
            current_values,
            inner.invariant_formulas,
        );
        for (exit_edge.timeline) |event| switch (event.payload) {
            .path_assumption, .safety_checkpoint => |formulas| for (formulas.formulas) |formula| {
                try self.appendSubstitutedLoopPremises(
                    constraints,
                    inner.variables,
                    current_values,
                    &.{formula.formula},
                );
            },
            .summary_dependency => |nested| try self.appendLoopSummaryDependencyPremise(
                constraints,
                exit_edge,
                nested,
            ),
            .storage_read, .storage_write => return error.InvalidLoopQuery,
        };

        const raw_following = try self.encodeBindingValues(
            constraints,
            inner.variables,
            exit_edge.next_state,
        );
        defer self.allocator.free(raw_following);
        for (inner.variables, raw_following) |variable, raw_value| {
            const following_value = try self.encodeFreeVariableIdentity(
                variable.id orelse return error.InvalidLoopQuery,
                variable.ty,
            );
            const substituted = try self.substituteLoopVariables(raw_value, inner.variables, current_values);
            const equality = z3.Z3_mk_eq(self.context.ctx, following_value, substituted);
            try self.context.checkNoError();
            try constraints.append(self.allocator, equality);
        }
    }

    fn appendSubstitutedLoopPremises(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        variables: []const obligation.LoopVariable,
        replacements: []const z3.Z3_ast,
        formulas: []const obligation.FormulaRef,
    ) EncodeError!void {
        for (formulas) |formula| {
            const raw = try self.encodeFormula(formula);
            try self.appendSideConstraints(constraints);
            const substituted = try self.substituteLoopVariables(raw, variables, replacements);
            try constraints.append(self.allocator, substituted);
        }
    }

    fn appendLoopAssumptions(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        assumption_ids: []const obligation.Id,
    ) EncodeError!void {
        for (assumption_ids) |assumption_id| {
            const assumption = obligation.findById(self.set.assumptions, assumption_id) orelse
                return error.UnknownAssumption;
            const formula = assumption.formula orelse return error.MissingFormula;
            const ast = try self.encodeFormula(formula);
            try self.appendSideConstraints(constraints);
            try constraints.append(self.allocator, ast);
        }
    }

    fn appendFormulaPremises(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        formulas: []const obligation.FormulaRef,
    ) EncodeError!void {
        for (formulas) |formula| {
            const ast = try self.encodeFormula(formula);
            try self.appendSideConstraints(constraints);
            try constraints.append(self.allocator, ast);
        }
    }

    fn encodeFormulaConjunction(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        formulas: []const obligation.FormulaRef,
    ) EncodeError!z3.Z3_ast {
        if (formulas.len == 0) return z3.Z3_mk_true(self.context.ctx);
        var result = try self.encodeFormula(formulas[0]);
        try self.appendSideConstraints(constraints);
        for (formulas[1..]) |formula| {
            const next = try self.encodeFormula(formula);
            try self.appendSideConstraints(constraints);
            result = z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ result, next });
            try self.context.checkNoError();
        }
        return result;
    }

    fn encodeFormulaValues(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        formulas: []const obligation.FormulaRef,
    ) EncodeError![]const z3.Z3_ast {
        const values = try self.allocator.alloc(z3.Z3_ast, formulas.len);
        errdefer self.allocator.free(values);
        for (formulas, values) |formula, *value| {
            value.* = try self.encodeValueFormula(formula);
            try self.appendSideConstraints(constraints);
        }
        return values;
    }

    fn encodeBindingValues(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        variables: []const obligation.LoopVariable,
        bindings: []const obligation.LoopStepAssignment,
    ) EncodeError![]const z3.Z3_ast {
        if (bindings.len != variables.len) return error.InvalidLoopQuery;
        const values = try self.allocator.alloc(z3.Z3_ast, bindings.len);
        errdefer self.allocator.free(values);
        for (bindings, variables, values, 0..) |binding, variable, *value, index| {
            if (binding.variable_index != index or binding.target == null or variable.id == null or
                !obligation.freeVarIdEql(binding.target.?, variable.id.?)) return error.InvalidLoopQuery;
            value.* = try self.encodeValueFormula(binding.value);
            try self.appendSideConstraints(constraints);
        }
        return values;
    }

    fn encodeValueFormula(self: *Adapter, formula: obligation.FormulaRef) EncodeError!z3.Z3_ast {
        return switch (formula) {
            .term => |term_id| self.encodeTermId(term_id),
            .origin_value => error.UnsupportedOriginValue,
        };
    }

    fn substituteLoopVariables(
        self: *Adapter,
        formula: z3.Z3_ast,
        variables: []const obligation.LoopVariable,
        replacements: []const z3.Z3_ast,
    ) EncodeError!z3.Z3_ast {
        if (variables.len != replacements.len) return error.InvalidLoopQuery;
        if (variables.len == 0) return formula;
        const current = try self.allocator.alloc(z3.Z3_ast, variables.len);
        defer self.allocator.free(current);
        for (variables, current) |variable, *ast| {
            const id = variable.id orelse return error.InvalidLoopQuery;
            ast.* = try self.encodeFreeVariableIdentity(id, variable.ty);
        }
        const count = std.math.cast(c_uint, variables.len) orelse return error.Overflow;
        const result = z3.Z3_substitute(self.context.ctx, formula, count, current.ptr, replacements.ptr);
        try self.context.checkNoError();
        return result;
    }

    fn appendNegatedGoal(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
        goal: z3.Z3_ast,
    ) EncodeError!void {
        const negated = z3.Z3_mk_not(self.context.ctx, goal);
        try self.context.checkNoError();
        try constraints.append(self.allocator, negated);
    }

    fn hashConstraints(
        self: *Adapter,
        constraints: []const z3.Z3_ast,
        logic: obligation.VerificationSolverLogic,
    ) EncodeError!u64 {
        var temp_solver = switch (logic) {
            .all => try Solver.init(self.context, self.allocator),
            .qf_aufbv => try Solver.initForLogic(self.context, self.allocator, "QF_AUFBV"),
        };
        defer temp_solver.deinit();

        for (constraints) |constraint| try temp_solver.assertChecked(constraint);

        const raw = z3.Z3_solver_to_string(self.context.ctx, temp_solver.solver);
        const text = if (raw == null) "" else std.mem.span(raw);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, text);
        if (!std.mem.endsWith(u8, text, "\n")) {
            try out.appendSlice(self.allocator, "\n");
        }
        try out.appendSlice(self.allocator, "(check-sat)\n");
        return std.hash.Wyhash.hash(0, out.items);
    }

    fn findUniqueQueryForObligation(
        self: *const Adapter,
        id: obligation.Id,
    ) EncodeError!?obligation.VerificationQuery {
        var found: ?obligation.VerificationQuery = null;
        for (self.set.queries) |query| {
            if (query.obligation_ids.len != 1 or query.obligation_ids[0] != id) continue;
            if (found != null) return error.AmbiguousQuery;
            found = query;
        }
        return found;
    }

    fn formulaForObligation(self: *Adapter, kind: obligation.Kind) EncodeError!z3.Z3_ast {
        const formula = obligation.kindFormula(kind) orelse return error.UnsupportedObligationKind;
        return self.encodeFormula(formula);
    }

    fn goalCounterexampleFormula(self: *Adapter, kind: obligation.Kind) EncodeError!z3.Z3_ast {
        const formula = obligation.kindFormula(kind) orelse return error.UnsupportedObligationKind;
        return self.encodeGoalFormula(formula);
    }

    pub fn encodeFormula(self: *Adapter, formula: obligation.FormulaRef) EncodeError!z3.Z3_ast {
        const ast = switch (formula) {
            .term => |term_id| try self.encodeTermId(term_id),
            .origin_value => return error.UnsupportedOriginValue,
        };
        try self.requireBool(ast);
        return ast;
    }

    fn encodeGoalFormula(self: *Adapter, formula: obligation.FormulaRef) EncodeError!z3.Z3_ast {
        const ast = switch (formula) {
            .term => |term_id| try self.encodeLeadingGoalForalls(term_id, 0, self.set.terms.len + 1),
            .origin_value => return error.UnsupportedOriginValue,
        };
        try self.requireBool(ast);
        return ast;
    }

    fn encodeLeadingGoalForalls(self: *Adapter, id: obligation.TermId, depth: u32, fuel: usize) EncodeError!z3.Z3_ast {
        if (fuel == 0 or id >= self.set.terms.len) return error.InvalidTermReference;
        const term = self.set.terms[id];
        if (term != .quantified or term.quantified.quantifier != .forall) {
            return self.encodeTermIdWithState(id, fuel, .{});
        }
        if (term.quantified.binder.origin == .function_param) {
            return self.encodeFunctionParamForallWrapper(term.quantified, depth, fuel);
        }
        return self.encodeLeadingGoalForall(term.quantified, depth, fuel);
    }

    fn encodeTermId(self: *Adapter, id: obligation.TermId) EncodeError!z3.Z3_ast {
        return self.encodeTermIdWithState(id, self.set.terms.len + 1, .{});
    }

    fn encodeTermIdWithState(
        self: *Adapter,
        id: obligation.TermId,
        fuel: usize,
        state: EncodeTermState,
    ) EncodeError!z3.Z3_ast {
        if (state.enforce_expected_type) {
            const expected_info = state.expected orelse return error.UnsupportedResultTerm;
            if (id >= self.set.terms.len) return error.InvalidTermReference;
            if (self.set.terms[id] != .result) {
                const actual = try staticTermTypeInfo(self.set, id);
                if (!typeInfoEql(expected_info, actual)) return error.TypeMismatch;
            }
        }
        return walkTerm(Adapter, self, self.set, id, fuel, state);
    }

    fn encodeTermIdWithExpected(
        self: *Adapter,
        id: obligation.TermId,
        expected: ?TypeInfo,
        enforce_expected_type: bool,
    ) EncodeError!z3.Z3_ast {
        return self.encodeTermIdWithExpectedFuel(id, self.set.terms.len + 1, expected, enforce_expected_type);
    }

    fn encodeTermIdWithExpectedFuel(
        self: *Adapter,
        id: obligation.TermId,
        fuel: usize,
        expected: ?TypeInfo,
        enforce_expected_type: bool,
    ) EncodeError!z3.Z3_ast {
        return self.encodeTermIdWithState(id, fuel, .{
            .expected = expected,
            .enforce_expected_type = enforce_expected_type,
        });
    }

    pub fn visitInvalidTerm(self: *Adapter) EncodeError!z3.Z3_ast {
        _ = self;
        return error.InvalidTermReference;
    }

    pub fn visitDefault(self: *Adapter, set: obligation.ObligationSet, term: obligation.Term, fuel: usize, state: EncodeTermState) EncodeError!z3.Z3_ast {
        _ = set;
        _ = fuel;
        return switch (term) {
            .bool_lit => |value| if (value)
                z3.Z3_mk_true(self.context.ctx)
            else
                z3.Z3_mk_false(self.context.ctx),
            .int_lit => |literal| self.encodeIntegerLiteral(literal),
            .variable => |variable| self.encodeVariable(variable),
            .old => |operand| self.encodeOld(operand),
            .result => blk: {
                const info = state.expected orelse return error.UnsupportedResultTerm;
                if (info.kind != .bitvector) return error.UnsupportedResultTerm;
                break :blk self.encodeResult(info);
            },
            .place_read => |place| self.encodePlaceRead(place),
            else => error.UnsupportedType,
        };
    }

    pub fn visitUnary(
        self: *Adapter,
        set: obligation.ObligationSet,
        unary: obligation.UnaryTerm,
        fuel: usize,
        state: EncodeTermState,
    ) EncodeError!z3.Z3_ast {
        _ = set;
        _ = state;
        return self.encodeUnary(unary, fuel);
    }

    pub fn visitBinary(
        self: *Adapter,
        set: obligation.ObligationSet,
        binary: obligation.BinaryTerm,
        fuel: usize,
        state: EncodeTermState,
    ) EncodeError!z3.Z3_ast {
        _ = set;
        _ = state;
        return self.encodeBinary(binary, fuel);
    }

    pub fn visitRefinementPredicate(
        self: *Adapter,
        set: obligation.ObligationSet,
        predicate: obligation.RefinementPredicateTerm,
        fuel: usize,
        state: EncodeTermState,
    ) EncodeError!z3.Z3_ast {
        _ = set;
        _ = state;
        return self.encodeRefinementPredicate(predicate, fuel);
    }

    pub fn visitQuantified(
        self: *Adapter,
        set: obligation.ObligationSet,
        quantified: obligation.QuantifiedTerm,
        fuel: usize,
        state: EncodeTermState,
    ) EncodeError!z3.Z3_ast {
        _ = set;
        _ = state;
        return self.encodeQuantified(quantified, fuel);
    }

    fn encodeResult(self: *Adapter, info: TypeInfo) EncodeError!z3.Z3_ast {
        const sort = try self.sortForType(info);
        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, "$ora.result");
        const ast = z3.Z3_mk_const(self.context.ctx, symbol, sort);
        try self.context.checkNoError();
        return ast;
    }

    fn encodePlaceRead(self: *Adapter, place: obligation.PlaceRef) EncodeError!z3.Z3_ast {
        try requireSupportedScalarPlace(place);
        if (self.loop_state != null) {
            const ty = try self.loopRootTypeRef(place.root);
            return self.selectLoopLocation(.{ .root = place.root, .value_ty = ty }, try typeInfo(ty), false);
        }
        const state = try self.canonicalQueryState();
        const root_state = try state.getOrPutPlaceRoot(self.allocator, place.root);
        const current = root_state.current orelse blk: {
            const created = root_state.entry orelse .global;
            root_state.current = created;
            if (root_state.entry == null) root_state.entry = created;
            break :blk created;
        };
        return self.encodePlaceSymbol(place.root, current);
    }

    fn encodeOld(self: *Adapter, operand: obligation.TermId) EncodeError!z3.Z3_ast {
        if (operand >= self.set.terms.len) return error.InvalidTermReference;
        const place = switch (self.set.terms[operand]) {
            .place_read => |place| place,
            else => return error.UnsupportedOldTerm,
        };
        requireSupportedScalarPlace(place) catch return error.UnsupportedOldTerm;

        if (self.loop_state != null) {
            const ty = try self.loopRootTypeRef(place.root);
            return self.selectLoopLocation(.{ .root = place.root, .value_ty = ty }, try typeInfo(ty), true);
        }

        const state = try self.canonicalQueryState();
        const root_state = try state.getOrPutPlaceRoot(self.allocator, place.root);
        const entry = root_state.entry orelse blk: {
            const created = root_state.current orelse .entry;
            root_state.entry = created;
            break :blk created;
        };
        const old_ast = try self.encodeOldPlaceSymbol(place.root);
        const entry_ast = try self.encodePlaceSymbol(place.root, entry);
        const linkage = z3.Z3_mk_eq(self.context.ctx, old_ast, entry_ast);
        try self.context.checkNoError();
        try state.side_constraints.append(self.allocator, linkage);
        return old_ast;
    }

    fn loopRootTypeRef(self: *Adapter, root: []const u8) EncodeError!obligation.TypeRef {
        const state = self.loop_state orelse return error.InvalidLoopQuery;
        for (state.summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
            .storage_read => |read| if (std.mem.eql(u8, read.location.root, root)) return read.location.value_ty,
            .storage_write => |write| if (std.mem.eql(u8, write.location.root, root)) return write.location.value_ty,
            else => {},
        };
        return error.LoopStorageRootUntouched;
    }

    fn loopStore(
        self: *Adapter,
        info: TypeInfo,
        index_info: TypeInfo,
        domain: []const u8,
    ) EncodeError!*LoopStore {
        const state = self.loop_state orelse return error.InvalidLoopQuery;
        for (state.stores.items) |*store| if (typeInfoEql(store.info, info) and
            typeInfoEql(store.index_info, index_info) and
            std.mem.eql(u8, store.domain, domain)) return store;
        const array_sort = z3.Z3_mk_array_sort(
            self.context.ctx,
            try self.sortForType(index_info),
            try self.sortForType(info),
        );
        try self.context.checkNoError();
        const shape = try std.fmt.allocPrint(self.allocator, "{s}{d}{s}", .{
            if (info.kind == .bool) "bool" else "bv",
            info.width,
            if (info.signed) "s" else "u",
        });
        defer self.allocator.free(shape);
        const direct = std.mem.eql(u8, domain, "direct");
        const current_name = if (direct)
            try std.fmt.allocPrint(self.allocator, "$ora.loop.store.{s}.current", .{shape})
        else
            try std.fmt.allocPrint(
                self.allocator,
                "$ora.loop.store.{s}.{s}.current",
                .{ domain, shape },
            );
        defer self.allocator.free(current_name);
        const entry_name = if (direct)
            try std.fmt.allocPrint(self.allocator, "$ora.loop.store.{s}.entry", .{shape})
        else
            try std.fmt.allocPrint(
                self.allocator,
                "$ora.loop.store.{s}.{s}.entry",
                .{ domain, shape },
            );
        defer self.allocator.free(entry_name);
        const current = try self.namedConst(current_name, array_sort);
        const entry = try self.namedConst(entry_name, array_sort);
        const owned_domain = try self.allocator.dupe(u8, domain);
        errdefer self.allocator.free(owned_domain);
        try state.stores.append(self.allocator, .{
            .info = info,
            .index_info = index_info,
            .domain = owned_domain,
            .current = current,
            .entry = entry,
        });
        return &state.stores.items[state.stores.items.len - 1];
    }

    fn loopRootKey(
        self: *Adapter,
        location: obligation.LoopStorageLocationExpr,
    ) EncodeError!z3.Z3_ast {
        if (location.root.len == 0) return error.InvalidLoopQuery;
        const state = self.loop_state orelse return error.InvalidLoopQuery;
        for (state.roots.items) |root| if (std.mem.eql(u8, root.name, location.root)) return root.key;
        const name = try std.fmt.allocPrint(self.allocator, "$ora.loop.root.{s}", .{location.root});
        defer self.allocator.free(name);
        const key = try self.namedConst(name, try self.sortForType(u256TypeInfo()));
        const query_state = try self.canonicalQueryState();
        for (state.roots.items) |root| {
            const distinct = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, key, root.key));
            try self.context.checkNoError();
            try query_state.side_constraints.append(self.allocator, distinct);
        }
        try state.roots.append(self.allocator, .{ .name = location.root, .key = key });
        return key;
    }

    const LoopLocationStoreAndKey = struct {
        store: *LoopStore,
        key: z3.Z3_ast,
    };

    fn loopLocationStoreAndKey(
        self: *Adapter,
        location: obligation.LoopStorageLocationExpr,
        info: TypeInfo,
    ) EncodeError!LoopLocationStoreAndKey {
        if (location.root.len == 0) return error.InvalidLoopQuery;
        if (location.projections.len == 0) return .{
            .store = try self.loopStore(info, u256TypeInfo(), "direct"),
            .key = try self.loopRootKey(location),
        };
        if (location.projections.len != 1) return error.InvalidLoopQuery;

        return switch (location.projections[0]) {
            .field => |field| blk: {
                if (field.len == 0) return error.InvalidLoopQuery;
                const domain = try std.fmt.allocPrint(
                    self.allocator,
                    "field.{s}.{s}",
                    .{ location.root, field },
                );
                defer self.allocator.free(domain);
                break :blk .{
                    .store = try self.loopStore(info, u256TypeInfo(), domain),
                    .key = try self.integerLiteralForType("0", u256TypeInfo()),
                };
            },
            .fixed_index => |index| try self.loopProjectedStoreAndKey(
                location.root,
                "fixed",
                index,
                info,
                true,
            ),
            .mapping_key => |key| try self.loopProjectedStoreAndKey(
                location.root,
                "mapping",
                key,
                info,
                false,
            ),
        };
    }

    fn loopProjectedStoreAndKey(
        self: *Adapter,
        root: []const u8,
        tag: []const u8,
        expression: obligation.LoopTypedFormula,
        value_info: TypeInfo,
        require_unsigned: bool,
    ) EncodeError!LoopLocationStoreAndKey {
        const key_info = try typeInfo(expression.ty);
        if (require_unsigned and (key_info.kind != .bitvector or key_info.signed)) {
            return error.InvalidLoopQuery;
        }
        const formula_info = switch (expression.value.formula) {
            .term => |term| try self.termTypeInfo(term),
            .origin_value => return error.UnsupportedOriginValue,
        };
        if (!typeInfoEql(key_info, formula_info)) return error.TypeMismatch;
        const key = try self.encodeValueFormula(expression.value.formula);
        const key_shape = try std.fmt.allocPrint(self.allocator, "{s}{d}{s}", .{
            if (key_info.kind == .bool) "bool" else "bv",
            key_info.width,
            if (key_info.signed) "s" else "u",
        });
        defer self.allocator.free(key_shape);
        const domain = try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}.{s}",
            .{ tag, root, key_shape },
        );
        defer self.allocator.free(domain);
        return .{
            .store = try self.loopStore(value_info, key_info, domain),
            .key = key,
        };
    }

    fn selectLoopLocation(
        self: *Adapter,
        location: obligation.LoopStorageLocationExpr,
        info: TypeInfo,
        entry: bool,
    ) EncodeError!z3.Z3_ast {
        if (!typeInfoEql(info, try typeInfo(location.value_ty))) return error.TypeMismatch;
        const resolved = try self.loopLocationStoreAndKey(location, info);
        const value = z3.Z3_mk_select(
            self.context.ctx,
            if (entry) resolved.store.entry else resolved.store.current,
            resolved.key,
        );
        try self.context.checkNoError();
        return value;
    }

    fn appendSideConstraints(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
    ) EncodeError!void {
        const state = try self.canonicalQueryState();
        try constraints.appendSlice(self.allocator, state.side_constraints.items);
        state.side_constraints.clearRetainingCapacity();
    }

    fn canonicalQueryState(self: *Adapter) EncodeError!*CanonicalQueryState {
        return self.query_state orelse error.MissingCanonicalQueryState;
    }

    fn encodePlaceSymbol(
        self: *Adapter,
        root: []const u8,
        kind: CanonicalPlaceSymbolKind,
    ) EncodeError!z3.Z3_ast {
        const name_text = switch (kind) {
            .global => try z3_verification.state_symbols.currentStorageName(self.allocator, root),
            .entry => try z3_verification.state_symbols.entryStorageName(self.allocator, root),
        };
        return self.encodeStorageSymbol(name_text);
    }

    fn encodeOldPlaceSymbol(self: *Adapter, root: []const u8) EncodeError!z3.Z3_ast {
        const name_text = try z3_verification.state_symbols.oldStorageName(self.allocator, root);
        return self.encodeStorageSymbol(name_text);
    }

    fn encodeStorageSymbol(self: *Adapter, name_text: []const u8) EncodeError!z3.Z3_ast {
        defer self.allocator.free(name_text);
        const sort = try self.sortForType(u256TypeInfo());
        return self.namedConst(name_text, sort);
    }

    fn encodeIntegerLiteral(self: *Adapter, literal: obligation.IntegerLiteralTerm) EncodeError!z3.Z3_ast {
        const ty = literal.ty orelse return error.MissingType;
        const info = try typeInfo(ty);
        if (info.kind != .bitvector) return error.ExpectedBitVector;
        return self.integerLiteralForType(literal.value, info);
    }

    fn encodeVariable(self: *Adapter, variable: obligation.VarRef) EncodeError!z3.Z3_ast {
        const free = switch (variable) {
            .free => |value| value,
            .bound => |bound| return self.encodeBoundVariable(bound),
        };
        const ty = free.ty orelse return error.MissingType;
        if (self.loop_state) |state| for (state.reads.items) |read| {
            if (!obligation.freeVarIdEql(read.id, free.id)) continue;
            if (!typeInfoEql(read.info, try typeInfo(ty))) return error.TypeMismatch;
            return read.value;
        };
        if (self.use_free_variable_identity_names) {
            return self.encodeFreeVariableIdentity(free.id, ty);
        }
        const sort = try self.sortForType(try typeInfo(ty));
        // Canonical SMT is byte-parity with the live encoder, which names free
        // variables from the source surface. Semantic identity remains FreeVarId
        // in the manifest and Lean denotation.
        return self.namedConst(free.name, sort);
    }

    fn encodeFreeVariableIdentity(
        self: *Adapter,
        id: obligation.FreeVarId,
        ty: obligation.TypeRef,
    ) EncodeError!z3.Z3_ast {
        const name = try std.fmt.allocPrint(
            self.allocator,
            "$ora.loop.f{d}.p{d}",
            .{ id.file_id, id.pattern_id },
        );
        defer self.allocator.free(name);
        return self.namedConst(name, try self.sortForType(try typeInfo(ty)));
    }

    fn encodeBoundVariable(self: *Adapter, bound: obligation.BoundVarRef) EncodeError!z3.Z3_ast {
        const state = try self.canonicalQueryState();
        const binding = state.lookupBound(bound.index) orelse return error.BoundVariableOutOfScope;
        const ty = bound.ty orelse return error.MissingType;
        const info = try typeInfo(ty);
        if (binding.origin == .function_param) {
            if (!typeInfoSameZ3Sort(info, binding.info)) return error.TypeMismatch;
            return binding.ast;
        }
        if (!typeInfoEql(info, binding.info)) return error.TypeMismatch;
        if (!typeInfoIsU256(info)) return error.UnsupportedBoundVariableType;
        return binding.ast;
    }

    fn encodeQuantified(self: *Adapter, quantified: obligation.QuantifiedTerm, fuel: usize) EncodeError!z3.Z3_ast {
        const info = try quantifierBinderTypeInfo(quantified.binder.ty);
        const sort = try self.sortForType(info);
        const bound_ast = try self.namedConst(quantified.binder.name, sort);

        const state = try self.canonicalQueryState();
        try state.pushBound(self.allocator, .{
            .name = quantified.binder.name,
            .info = info,
            .ast = bound_ast,
            .origin = quantified.binder.origin,
        });
        defer state.popBound();

        var quantified_body = try self.encodeTermIdWithState(quantified.body, fuel - 1, .{});
        try self.requireBool(quantified_body);

        if (quantified.condition) |condition_id| {
            const condition = try self.encodeTermIdWithState(condition_id, fuel - 1, .{});
            try self.requireBool(condition);
            quantified_body = switch (quantified.quantifier) {
                .exists => z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ condition, quantified_body }),
                .forall => z3.Z3_mk_implies(self.context.ctx, condition, quantified_body),
            };
            try self.context.checkNoError();
        }

        var bounds = [_]z3.Z3_app{z3.Z3_to_app(self.context.ctx, bound_ast)};
        const ast = switch (quantified.quantifier) {
            .exists => z3.Z3_mk_exists_const(
                self.context.ctx,
                0,
                @intCast(bounds.len),
                bounds[0..].ptr,
                0,
                null,
                quantified_body,
            ),
            .forall => z3.Z3_mk_forall_const(
                self.context.ctx,
                0,
                @intCast(bounds.len),
                bounds[0..].ptr,
                0,
                null,
                quantified_body,
            ),
        };
        try self.context.checkNoError();
        return ast;
    }

    fn encodeLeadingGoalForall(self: *Adapter, quantified: obligation.QuantifiedTerm, depth: u32, fuel: usize) EncodeError!z3.Z3_ast {
        try self.predeclareLeadingGoalForallSurface(quantified, fuel);

        const info = try quantifierBinderTypeInfo(quantified.binder.ty);
        const sort = try self.sortForType(info);
        var name_buf: [z3_verification.goal_skolem.name_buffer_len]u8 = undefined;
        var binder_buf: [z3_verification.goal_skolem.binder_buffer_len]u8 = undefined;
        const witness_name = z3_verification.goal_skolem.nameZ(&name_buf, &binder_buf, quantified.binder.name, depth);
        const witness = try self.namedConst(witness_name, sort);

        const state = try self.canonicalQueryState();
        try state.pushBound(self.allocator, .{
            .name = quantified.binder.name,
            .info = info,
            .ast = witness,
            .origin = .user,
        });
        defer state.popBound();

        if (quantified.condition) |condition_id| {
            const condition = try self.encodeTermIdWithState(condition_id, fuel - 1, .{});
            try self.requireBool(condition);
            const body = try self.encodeTermIdWithState(quantified.body, fuel - 1, .{});
            try self.requireBool(body);
            const ast = z3.Z3_mk_implies(self.context.ctx, condition, body);
            try self.context.checkNoError();
            return ast;
        }

        return self.encodeLeadingGoalForalls(quantified.body, depth + 1, fuel - 1);
    }

    fn encodeFunctionParamForallWrapper(self: *Adapter, quantified: obligation.QuantifiedTerm, depth: u32, fuel: usize) EncodeError!z3.Z3_ast {
        if (quantified.condition != null) return error.UnsupportedFunctionParamWrapperCondition;

        const info = try typeInfo(quantified.binder.ty orelse return error.MissingType);
        const sort = try self.sortForType(info);
        const param_ast = try self.namedConst(quantified.binder.name, sort);

        const state = try self.canonicalQueryState();
        try state.pushBound(self.allocator, .{
            .name = quantified.binder.name,
            .info = info,
            .ast = param_ast,
            .origin = .function_param,
        });
        defer state.popBound();

        // Function-param wrappers are a Lean/formal-manifest device. The live
        // SMT row treats params as free constants and asserts requires/assumes
        // as separate query constraints, so canonical byte parity unwraps the
        // binder and leaves user quantifiers to the goal-skolem path.
        return self.encodeLeadingGoalForalls(quantified.body, depth, fuel - 1);
    }

    fn predeclareLeadingGoalForallSurface(self: *Adapter, quantified: obligation.QuantifiedTerm, fuel: usize) EncodeError!void {
        const state = try self.canonicalQueryState();
        const snapshot_value = state.snapshot();
        defer state.restore(snapshot_value);

        const info = try quantifierBinderTypeInfo(quantified.binder.ty);
        const sort = try self.sortForType(info);
        const bound_ast = try self.namedConst(quantified.binder.name, sort);

        try state.pushBound(self.allocator, .{
            .name = quantified.binder.name,
            .info = info,
            .ast = bound_ast,
            .origin = quantified.binder.origin,
        });

        // Live quantifier encoding visits the body before the `where` condition,
        // then goal-position skolemization replaces the binder afterward. This
        // prepass preserves that symbol-interning order while discarding the raw
        // quantified term and any canonical side constraints it created.
        const body = try self.encodeTermIdWithState(quantified.body, fuel - 1, .{});
        try self.requireBool(body);

        if (quantified.condition) |condition_id| {
            const condition = try self.encodeTermIdWithState(condition_id, fuel - 1, .{});
            try self.requireBool(condition);
        }
    }

    fn namedConst(self: *Adapter, raw_name: []const u8, sort: z3.Z3_sort) EncodeError!z3.Z3_ast {
        const name = try self.allocator.dupeZ(u8, raw_name);
        defer self.allocator.free(name);
        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name.ptr);
        const ast = z3.Z3_mk_const(self.context.ctx, symbol, sort);
        try self.context.checkNoError();
        return ast;
    }

    fn encodeUnary(self: *Adapter, unary: obligation.UnaryTerm, fuel: usize) EncodeError!z3.Z3_ast {
        const operand = try self.encodeTermIdWithState(unary.operand, fuel - 1, .{});
        const ast = switch (unary.op) {
            .not => blk: {
                try self.requireBool(operand);
                break :blk z3.Z3_mk_not(self.context.ctx, operand);
            },
            .neg => blk: {
                try self.requireBitVector(operand);
                const sort = z3.Z3_get_sort(self.context.ctx, operand);
                const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
                break :blk z3.Z3_mk_bv_sub(self.context.ctx, zero, operand);
            },
        };
        try self.context.checkNoError();
        return ast;
    }

    fn encodeBinary(self: *Adapter, binary: obligation.BinaryTerm, fuel: usize) EncodeError!z3.Z3_ast {
        const expected = try expectedInfoForBinaryOperands(self.set, binary);
        const enforce_expected_type = expected != null and
            (termContainsResult(self.set, binary.lhs, self.set.terms.len + 1) or
                termContainsResult(self.set, binary.rhs, self.set.terms.len + 1));
        const lhs = try self.encodeTermIdWithExpectedFuel(binary.lhs, fuel - 1, expected, enforce_expected_type);
        const rhs = try self.encodeTermIdWithExpectedFuel(binary.rhs, fuel - 1, expected, enforce_expected_type);

        const ast = switch (binary.op) {
            .eq => z3.Z3_mk_eq(self.context.ctx, lhs, rhs),
            .ne => z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, lhs, rhs)),
            .and_ => blk: {
                try self.requireBool(lhs);
                try self.requireBool(rhs);
                break :blk z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs });
            },
            .or_ => blk: {
                try self.requireBool(lhs);
                try self.requireBool(rhs);
                break :blk z3.Z3_mk_or(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs, rhs });
            },
            .implies => blk: {
                try self.requireBool(lhs);
                try self.requireBool(rhs);
                break :blk z3.Z3_mk_implies(self.context.ctx, lhs, rhs);
            },
            .add => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bv_add(self.context.ctx, lhs, rhs);
            },
            .sub => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bv_sub(self.context.ctx, lhs, rhs);
            },
            .mul => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bv_mul(self.context.ctx, lhs, rhs);
            },
            .bit_and => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bvand(self.context.ctx, lhs, rhs);
            },
            .bit_xor => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bvxor(self.context.ctx, lhs, rhs);
            },
            .pow => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk self.encodePowerTotal(lhs, rhs);
            },
            .shl => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk z3.Z3_mk_bvshl(self.context.ctx, lhs, rhs);
            },
            .shr => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk if (try self.binaryOperandsSigned(binary, expected))
                    z3.Z3_mk_bvashr(self.context.ctx, lhs, rhs)
                else
                    z3.Z3_mk_bvlshr(self.context.ctx, lhs, rhs);
            },
            .div => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk if (try self.binaryOperandsSigned(binary, expected))
                    self.encodeSignedDivTotal(lhs, rhs)
                else
                    self.encodeUnsignedDivTotal(lhs, rhs);
            },
            .mod => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk if (try self.binaryOperandsSigned(binary, expected))
                    self.encodeSignedRemTotal(lhs, rhs)
                else
                    self.encodeUnsignedRemTotal(lhs, rhs);
            },
            .lt => try self.encodeComparison(lhs, rhs, .lt, false),
            .le => try self.encodeComparison(lhs, rhs, .le, false),
            .gt => try self.encodeComparison(lhs, rhs, .gt, false),
            .ge => try self.encodeComparison(lhs, rhs, .ge, false),
            .slt => try self.encodeComparison(lhs, rhs, .lt, true),
            .sle => try self.encodeComparison(lhs, rhs, .le, true),
            .sgt => try self.encodeComparison(lhs, rhs, .gt, true),
            .sge => try self.encodeComparison(lhs, rhs, .ge, true),
        };
        try self.context.checkNoError();
        return ast;
    }

    fn encodePowerTotal(self: *Adapter, base: z3.Z3_ast, exponent: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, base);
        const width = z3.Z3_get_bv_sort_size(self.context.ctx, sort);
        const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, sort);

        if (self.tryParseBitVectorNumeral(exponent)) |constant_exponent| {
            var result = one;
            var factor = base;
            var remaining = constant_exponent;
            while (remaining != 0) : (remaining >>= 1) {
                if ((remaining & 1) != 0) {
                    result = z3.Z3_mk_bv_mul(self.context.ctx, result, factor);
                }
                if (remaining != 1) {
                    factor = z3.Z3_mk_bv_mul(self.context.ctx, factor, factor);
                }
            }
            return result;
        }

        const bit_sort = z3.Z3_mk_bv_sort(self.context.ctx, 1);
        const bit_one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, bit_sort);
        var result = one;
        var factor = base;
        var bit_index: u32 = 0;
        while (bit_index < width) : (bit_index += 1) {
            const bit = z3.Z3_mk_extract(self.context.ctx, bit_index, bit_index, exponent);
            const bit_is_set = z3.Z3_mk_eq(self.context.ctx, bit, bit_one);
            const multiplied = z3.Z3_mk_bv_mul(self.context.ctx, result, factor);
            result = z3.Z3_mk_ite(self.context.ctx, bit_is_set, multiplied, result);
            factor = z3.Z3_mk_bv_mul(self.context.ctx, factor, factor);
        }
        return result;
    }

    fn tryParseBitVectorNumeral(self: *Adapter, ast: z3.Z3_ast) ?u256 {
        const sort = z3.Z3_get_sort(self.context.ctx, ast);
        if (z3.Z3_get_sort_kind(self.context.ctx, sort) != z3.Z3_BV_SORT) return null;
        const numeral_z = z3.Z3_get_numeral_string(self.context.ctx, ast) orelse return null;
        return std.fmt.parseInt(u256, std.mem.span(numeral_z), 10) catch null;
    }

    fn signedMinIntAst(self: *Adapter, sort: z3.Z3_sort) z3.Z3_ast {
        const width = z3.Z3_get_bv_sort_size(self.context.ctx, sort);
        const one = z3.Z3_mk_unsigned_int64(self.context.ctx, 1, sort);
        const shift_amount = z3.Z3_mk_unsigned_int64(self.context.ctx, width - 1, sort);
        return z3.Z3_mk_bvshl(self.context.ctx, one, shift_amount);
    }

    fn signedMinDivNegOneCondition(self: *Adapter, lhs: z3.Z3_ast, rhs: z3.Z3_ast, sort: z3.Z3_sort) z3.Z3_ast {
        const min_int = self.signedMinIntAst(sort);
        const neg_one = z3.Z3_mk_numeral(self.context.ctx, "-1", sort);
        const lhs_is_min = z3.Z3_mk_eq(self.context.ctx, lhs, min_int);
        const rhs_is_neg_one = z3.Z3_mk_eq(self.context.ctx, rhs, neg_one);
        return z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lhs_is_min, rhs_is_neg_one });
    }

    fn encodeUnsignedDivTotal(self: *Adapter, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const divisor_zero = z3.Z3_mk_eq(self.context.ctx, rhs, zero);
        const raw = z3.Z3_mk_bv_udiv(self.context.ctx, lhs, rhs);
        return z3.Z3_mk_ite(self.context.ctx, divisor_zero, zero, raw);
    }

    fn encodeUnsignedRemTotal(self: *Adapter, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const divisor_zero = z3.Z3_mk_eq(self.context.ctx, rhs, zero);
        const raw = z3.Z3_mk_bv_urem(self.context.ctx, lhs, rhs);
        return z3.Z3_mk_ite(self.context.ctx, divisor_zero, zero, raw);
    }

    fn encodeSignedDivTotal(self: *Adapter, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const divisor_zero = z3.Z3_mk_eq(self.context.ctx, rhs, zero);
        const min_int = self.signedMinIntAst(sort);
        const special = self.signedMinDivNegOneCondition(lhs, rhs, sort);
        const raw = z3.Z3_mk_bvsdiv(self.context.ctx, lhs, rhs);
        const nonzero_result = z3.Z3_mk_ite(self.context.ctx, special, min_int, raw);
        return z3.Z3_mk_ite(self.context.ctx, divisor_zero, zero, nonzero_result);
    }

    fn encodeSignedRemTotal(self: *Adapter, lhs: z3.Z3_ast, rhs: z3.Z3_ast) z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const zero = z3.Z3_mk_unsigned_int64(self.context.ctx, 0, sort);
        const divisor_zero = z3.Z3_mk_eq(self.context.ctx, rhs, zero);
        const special = self.signedMinDivNegOneCondition(lhs, rhs, sort);
        const raw = z3.Z3_mk_bvsrem(self.context.ctx, lhs, rhs);
        const nonzero_result = z3.Z3_mk_ite(self.context.ctx, special, zero, raw);
        return z3.Z3_mk_ite(self.context.ctx, divisor_zero, zero, nonzero_result);
    }

    const Comparison = enum(u8) {
        lt,
        le,
        gt,
        ge,
    };

    fn encodeComparison(
        self: *Adapter,
        lhs: z3.Z3_ast,
        rhs: z3.Z3_ast,
        comparison: Comparison,
        signed: bool,
    ) EncodeError!z3.Z3_ast {
        return self.compareWithSignedness(comparison, lhs, rhs, signed);
    }

    fn encodeRefinementPredicate(
        self: *Adapter,
        predicate: obligation.RefinementPredicateTerm,
        fuel: usize,
    ) EncodeError!z3.Z3_ast {
        const value = try self.encodeTermIdWithState(predicate.value, fuel - 1, .{});
        try self.requireBitVector(value);
        const value_info = try self.termTypeInfo(predicate.value);

        return switch (refinement_builtin_map.get(predicate.name) orelse return error.UnsupportedRefinement) {
            .non_zero,
            .non_zero_address,
            => blk: {
                try expectArgCount(predicate, 0);
                break :blk self.notEqualZero(value, value_info);
            },
            .min_value => blk: {
                try expectArgCount(predicate, 1);
                const bound = try self.encodeTermIdWithState(predicate.args[0], fuel - 1, .{});
                break :blk self.compareWithSignedness(.ge, value, bound, value_info.signed);
            },
            .max_value => blk: {
                try expectArgCount(predicate, 1);
                const bound = try self.encodeTermIdWithState(predicate.args[0], fuel - 1, .{});
                break :blk self.compareWithSignedness(.le, value, bound, value_info.signed);
            },
            .in_range => blk: {
                try expectArgCount(predicate, 2);
                const lower = try self.encodeTermIdWithState(predicate.args[0], fuel - 1, .{});
                const upper = try self.encodeTermIdWithState(predicate.args[1], fuel - 1, .{});
                const lower_ok = try self.compareWithSignedness(.ge, value, lower, value_info.signed);
                const upper_ok = try self.compareWithSignedness(.le, value, upper, value_info.signed);
                const result = z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lower_ok, upper_ok });
                try self.context.checkNoError();
                break :blk result;
            },
            .basis_points => blk: {
                try expectArgCount(predicate, 0);
                const zero = try self.integerLiteralForType("0", value_info);
                const ten_thousand = try self.integerLiteralForType("10000", value_info);
                const lower_ok = try self.compareWithSignedness(.ge, value, zero, value_info.signed);
                const upper_ok = try self.compareWithSignedness(.le, value, ten_thousand, value_info.signed);
                const result = z3.Z3_mk_and(self.context.ctx, 2, &[_]z3.Z3_ast{ lower_ok, upper_ok });
                try self.context.checkNoError();
                break :blk result;
            },
        };
    }

    fn compareWithSignedness(
        self: *Adapter,
        comparison: Comparison,
        lhs: z3.Z3_ast,
        rhs: z3.Z3_ast,
        signed: bool,
    ) EncodeError!z3.Z3_ast {
        try self.requireBitVector(lhs);
        try self.requireBitVector(rhs);
        const result = switch (comparison) {
            .lt => if (signed) z3.Z3_mk_bvslt(self.context.ctx, lhs, rhs) else z3.Z3_mk_bvult(self.context.ctx, lhs, rhs),
            .le => if (signed) z3.Z3_mk_bvsle(self.context.ctx, lhs, rhs) else z3.Z3_mk_bvule(self.context.ctx, lhs, rhs),
            .gt => if (signed) z3.Z3_mk_bvsgt(self.context.ctx, lhs, rhs) else z3.Z3_mk_bvugt(self.context.ctx, lhs, rhs),
            .ge => if (signed) z3.Z3_mk_bvsge(self.context.ctx, lhs, rhs) else z3.Z3_mk_bvuge(self.context.ctx, lhs, rhs),
        };
        try self.context.checkNoError();
        return result;
    }

    fn notEqualZero(self: *Adapter, value: z3.Z3_ast, info: TypeInfo) EncodeError!z3.Z3_ast {
        const zero = try self.integerLiteralForType("0", info);
        const result = z3.Z3_mk_not(self.context.ctx, z3.Z3_mk_eq(self.context.ctx, value, zero));
        try self.context.checkNoError();
        return result;
    }

    fn binaryOperandsSigned(self: *Adapter, binary: obligation.BinaryTerm, expected: ?TypeInfo) EncodeError!bool {
        if (expected) |info| {
            if (info.kind != .bitvector) return error.ExpectedBitVector;
            return info.signed;
        }
        const lhs = try self.termTypeInfo(binary.lhs);
        const rhs = try self.termTypeInfo(binary.rhs);
        if (lhs.kind != .bitvector or rhs.kind != .bitvector) return error.ExpectedBitVector;
        if (lhs.width != rhs.width) return error.TypeMismatch;
        if (lhs.signed != rhs.signed) return error.TypeMismatch;
        return lhs.signed;
    }

    fn termTypeInfo(self: *Adapter, id: obligation.TermId) EncodeError!TypeInfo {
        return staticTermTypeInfo(self.set, id);
    }

    fn integerLiteralForType(self: *Adapter, value: []const u8, info: TypeInfo) EncodeError!z3.Z3_ast {
        if (info.kind != .bitvector) return error.ExpectedBitVector;
        const sort = try self.sortForType(info);
        const value_z = try self.allocator.dupeZ(u8, value);
        defer self.allocator.free(value_z);

        const ast = z3.Z3_mk_numeral(self.context.ctx, value_z.ptr, sort);
        try self.context.checkNoError();
        return ast;
    }

    fn sortForType(self: *Adapter, info: TypeInfo) EncodeError!z3.Z3_sort {
        const sort = switch (info.kind) {
            .bool => z3.Z3_mk_bool_sort(self.context.ctx),
            .bitvector => z3.Z3_mk_bv_sort(self.context.ctx, info.width),
        };
        try self.context.checkNoError();
        return sort;
    }

    fn requireBool(self: *Adapter, ast: z3.Z3_ast) EncodeError!void {
        const sort = z3.Z3_get_sort(self.context.ctx, ast);
        if (z3.Z3_get_sort_kind(self.context.ctx, sort) != z3.Z3_BOOL_SORT) return error.ExpectedBool;
    }

    fn requireBitVector(self: *Adapter, ast: z3.Z3_ast) EncodeError!void {
        const sort = z3.Z3_get_sort(self.context.ctx, ast);
        if (z3.Z3_get_sort_kind(self.context.ctx, sort) != z3.Z3_BV_SORT) return error.ExpectedBitVector;
    }
};

fn findLoopEdge(
    summary: obligation.LoopSummaryRow,
    edge_id: u32,
) ?obligation.LoopEdgeRow {
    for (summary.edges) |edge| if (edge.id == edge_id) return edge;
    return null;
}

fn uniqueLoopEdgeForTarget(
    summary: obligation.LoopSummaryRow,
    target: obligation.LoopControlTarget,
) ?obligation.LoopEdgeRow {
    var matched: ?obligation.LoopEdgeRow = null;
    for (summary.edges) |edge| {
        if (edge.target != target) continue;
        if (matched != null) return null;
        matched = edge;
    }
    return matched;
}

fn expectCanonicalSupported(set: obligation.ObligationSet, query: obligation.VerificationQuery) !void {
    switch (queryCanonicalSupport(set, query)) {
        .supported => {},
        .unsupported => |reason| {
            std.debug.print("unexpected canonical Z3 unsupported reason: {s}\n", .{@tagName(reason)});
            return error.UnexpectedUnsupportedCanonicalQuery;
        },
    }
}

fn expectCanonicalUnsupported(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    expected: CanonicalUnsupportedReason,
) !void {
    switch (queryCanonicalSupport(set, query)) {
        .supported => return error.UnexpectedSupportedCanonicalQuery,
        .unsupported => |reason| try std.testing.expectEqual(expected, reason),
    }
}

fn expectCanonicalHashSucceeds(
    context: *Context,
    terms: []const obligation.Term,
    target_term: obligation.TermId,
) !void {
    const obligations = [_]obligation.Obligation{fixture.logicalEnsuresObligation(1, "matrix", target_term)};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{fixture.obligationQuery(2, "matrix", &obligation_ids, &.{})};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    _ = queryCanonicalPromotionShape(set, queries[0]) orelse return error.MissingCanonicalPromotionShape;
    var adapter = Adapter.init(context, std.testing.allocator, set);
    const hash = try adapter.queryHash(2);
    try std.testing.expect(hash.constraint_count > 0);
}

fn expectCanonicalHashSucceedsWithShape(
    context: *Context,
    terms: []const obligation.Term,
    target_term: obligation.TermId,
    expected_shape: CanonicalPromotionShape,
) !void {
    const obligations = [_]obligation.Obligation{fixture.logicalEnsuresObligation(1, "matrix", target_term)};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{fixture.obligationQuery(2, "matrix", &obligation_ids, &.{})};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    try std.testing.expectEqual(expected_shape, queryCanonicalPromotionShape(set, queries[0]).?);
    var adapter = Adapter.init(context, std.testing.allocator, set);
    const hash = try adapter.queryHash(2);
    try std.testing.expect(hash.constraint_count > 0);
}

fn expectCanonicalHashSucceedsWithAssumption(
    context: *Context,
    terms: []const obligation.Term,
    assumption_term: obligation.TermId,
    target_term: obligation.TermId,
) !void {
    const assumptions = [_]obligation.Assumption{fixture.assumeAssumption(1, "matrix", assumption_term)};
    const obligations = [_]obligation.Obligation{fixture.logicalEnsuresObligation(2, "matrix", target_term)};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{fixture.obligationQuery(3, "matrix", &obligation_ids, &assumption_ids)};
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    _ = queryCanonicalPromotionShape(set, queries[0]) orelse return error.MissingCanonicalPromotionShape;
    var adapter = Adapter.init(context, std.testing.allocator, set);
    const hash = try adapter.queryHash(3);
    try std.testing.expect(hash.constraint_count > 0);
}

fn expectCanonicalHashUnsupported(
    terms: []const obligation.Term,
    target_term: obligation.TermId,
    expected: CanonicalUnsupportedReason,
) !void {
    const obligations = [_]obligation.Obligation{fixture.logicalEnsuresObligation(1, "matrix", target_term)};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{fixture.obligationQuery(2, "matrix", &obligation_ids, &.{})};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalUnsupported(set, queries[0], expected);
}

fn expectCanonicalHashUnsupportedWithAssumption(
    terms: []const obligation.Term,
    assumption_term: obligation.TermId,
    target_term: obligation.TermId,
    expected: CanonicalUnsupportedReason,
) !void {
    const assumptions = [_]obligation.Assumption{fixture.assumeAssumption(1, "matrix", assumption_term)};
    const obligations = [_]obligation.Obligation{fixture.logicalEnsuresObligation(2, "matrix", target_term)};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{fixture.obligationQuery(3, "matrix", &obligation_ids, &assumption_ids)};
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalUnsupported(set, queries[0], expected);
}

test "canonical LoopVC safety excludes premises after each checkpoint" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const u256_ty: obligation.TypeRef = .{ .spelling = "u256" };
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "0", .ty = u256_ty } },
        .{ .variable = .{ .free = .{ .id = loop_id, .name = "i", .ty = u256_ty } } },
        .{ .bool_lit = true },
        .{ .bool_lit = false },
    };
    const variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = u256_ty,
    }};
    const bindings = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop:i",
        .value = .{ .term = 1 },
    }};
    const true_formulas = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 2 } }};
    const false_formulas = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 3 } }};
    const backedge_timeline = [_]obligation.LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "path:before",
            .payload = .{ .path_assumption = .{ .formulas = &true_formulas } },
        },
        .{
            .id = 1,
            .structural_key = "check:early",
            .payload = .{ .safety_checkpoint = .{ .formulas = &false_formulas } },
        },
        .{
            .id = 2,
            .structural_key = "path:after",
            .payload = .{ .path_assumption = .{ .formulas = &false_formulas } },
        },
    };
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "path:exit",
        .payload = .{ .path_assumption = .{ .formulas = &true_formulas } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "edge:backedge",
            .timeline = &backedge_timeline,
            .next_state = &bindings,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "edge:exit",
            .timeline = &exit_timeline,
            .next_state = &bindings,
            .target = .successor,
        },
    };
    const summaries = [_]obligation.LoopSummaryRow{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "prefixSafety" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .variables = &variables,
        .init_formulas = &.{.{ .term = 0 }},
        .invariant_formulas = &.{.{ .term = 2 }},
        .edges = &edges,
        .projection_completeness = .complete,
        .post_formulas = &.{.{ .term = 2 }},
    }};
    const query: obligation.VerificationQuery = .{
        .id = 10,
        .owner = .{ .function = .{ .name = "prefixSafety" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .artifact_policy = .blocks_verified_artifacts,
        .proof_requirement = .backend_result,
        .backend = .z3,
        .kind = .loop_body_safety,
        .loop_summary_id = 1,
        .loop_edge_id = 0,
    };
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .queries = &.{query},
        .loop_summaries = &summaries,
    };
    try std.testing.expect(summaries[0].projectionSupported());

    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    var constraints: std.ArrayList(z3.Z3_ast) = .empty;
    defer constraints.deinit(std.testing.allocator);
    try adapter.appendLoopQueryConstraints(&constraints, query);
    // The invariant remains global and the complete prefix-staged safety
    // statement is the only goal. The later false path is not a premise.
    try std.testing.expectEqual(@as(usize, 2), constraints.items.len);
    var solver = try Solver.init(&z3_ctx, std.testing.allocator);
    defer solver.deinit();
    for (constraints.items) |constraint| try solver.assertChecked(constraint);
    try std.testing.expectEqual(z3.Z3_L_TRUE, try solver.checkChecked());
}

test "canonical LoopVC untouched storage root has a named failure" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const u256_ty: obligation.TypeRef = .{ .spelling = "u256" };
    const timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "write:counter",
        .payload = .{ .storage_write = .{
            .location = .{ .root = "counter", .value_ty = u256_ty },
            .value = .{ .formula = .{ .term = 0 } },
        } },
    }};
    const edges = [_]obligation.LoopEdgeRow{.{ .id = 0, .timeline = &timeline }};
    const summary: obligation.LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "untouchedRoot" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .loop_source_op_id = 1,
        .loop_kind = .scf_while,
        .edges = &edges,
    };
    const set: obligation.ObligationSet = .{};
    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    var loop_state: LoopEncodeState = .{ .summary = summary };
    defer loop_state.deinit(std.testing.allocator);
    adapter.loop_state = &loop_state;
    defer adapter.loop_state = null;

    try std.testing.expectError(
        error.LoopStorageRootUntouched,
        adapter.loopRootTypeRef("untouched"),
    );
    try std.testing.expectEqualStrings(
        "loop_storage_root_untouched",
        ct.encodeErrorReason(error.LoopStorageRootUntouched),
    );
}

test "canonical Z3 classifier matrix hashes every supported core formula shape" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const bool_terms = [_]obligation.Term{
        .{ .bool_lit = true },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &bool_terms, 0);

    const not_terms = [_]obligation.Term{
        .{ .bool_lit = true },
        .{ .unary = .{ .op = .not, .operand = 0 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &not_terms, 1);

    const connective_terms = [_]obligation.Term{
        .{ .bool_lit = true },
        .{ .bool_lit = false },
        .{ .binary = .{ .op = .and_, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .or_, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .implies, .lhs = 2, .rhs = 3 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &connective_terms, 4);

    const unsigned_comparison_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "10", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &unsigned_comparison_terms, 2);

    const signed_comparison_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 2 }, .name = "delta", .ty = .{ .spelling = "i256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "i256" } } },
        .{ .binary = .{ .op = .sge, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &signed_comparison_terms, 2);

    const result_comparison_terms = [_]obligation.Term{
        .result,
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &result_comparison_terms, 2);

    const scalar_place_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &scalar_place_terms, 2);

    const old_place_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 1, .rhs = 2 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &old_place_terms, 3);

    const arithmetic_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 3 }, .name = "lhs", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 4 }, .name = "rhs", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .add, .lhs = 0, .rhs = 2 } },
        .{ .binary = .{ .op = .sub, .lhs = 3, .rhs = 2 } },
        .{ .binary = .{ .op = .mul, .lhs = 4, .rhs = 2 } },
        .{ .binary = .{ .op = .div, .lhs = 5, .rhs = 2 } },
        .{ .binary = .{ .op = .mod, .lhs = 6, .rhs = 2 } },
        .{ .binary = .{ .op = .eq, .lhs = 7, .rhs = 1 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &arithmetic_terms, 8);

    const refinement_args = [_]obligation.TermId{ 1, 2 };
    const refinement_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 5 }, .name = "basis", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .int_lit = .{ .value = "10000", .ty = .{ .spelling = "u256" } } },
        .{ .refinement_predicate = .{ .name = "InRange", .value = 0, .args = &refinement_args } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &refinement_terms, 3);

    const formula_combination_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .result,
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .add, .lhs = 1, .rhs = 3, .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .and_, .lhs = 5, .rhs = 6 } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &formula_combination_terms, 7);

    const forall_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 2,
        } },
        .{ .bool_lit = true },
    };
    try expectCanonicalHashSucceedsWithAssumption(&z3_ctx, &forall_terms, 3, 4);

    const goal_forall_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 1,
        } },
    };
    try expectCanonicalHashSucceeds(&z3_ctx, &goal_forall_terms, 2);

    const signed_function_param_wrapper_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "x", .ty = .{ .spelling = "i256" } } } },
        .{ .binary = .{ .op = .sle, .lhs = 0, .rhs = 0, .ty = .{ .spelling = "i256" } } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "x", .ty = .{ .spelling = "i256" }, .origin = .function_param },
            .body = 1,
        } },
    };
    try expectCanonicalHashSucceedsWithShape(&z3_ctx, &signed_function_param_wrapper_terms, 2, .core_formula);

    const conditioned_function_param_wrapper_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "x", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 0 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "x", .ty = .{ .spelling = "u256" }, .origin = .function_param },
            .condition = 2,
            .body = 3,
        } },
    };
    try expectCanonicalHashUnsupported(
        &conditioned_function_param_wrapper_terms,
        4,
        .unsupported_function_param_wrapper_condition,
    );
}

test "canonical Z3 classifier matrix names unsupported core formula shapes" {
    const old_terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .old = 0 },
        .{ .binary = .{ .op = .eq, .lhs = 1, .rhs = 0 } },
    };
    try expectCanonicalHashUnsupported(&old_terms, 2, .unsupported_old_term);

    const bare_result_terms = [_]obligation.Term{
        .result,
    };
    try expectCanonicalHashUnsupported(&bare_result_terms, 0, .unsupported_result_term);

    const untyped_result_equality_terms = [_]obligation.Term{
        .result,
        .result,
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashUnsupported(&untyped_result_equality_terms, 2, .unsupported_result_term);

    const result_mismatched_width_terms = [_]obligation.Term{
        .result,
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u32" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashUnsupported(&result_mismatched_width_terms, 2, .unsupported_type);

    const place_keys = [_]obligation.PlaceKey{.{ .constant = "1" }};
    const keyed_place_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .place_read = .{ .root = "balance", .region = .storage, .keys = &place_keys } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 1 } },
    };
    try expectCanonicalHashUnsupported(&keyed_place_terms, 3, .unsupported_place_read_term);

    const transient_place_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "scratch", .region = .transient } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
    };
    try expectCanonicalHashUnsupported(&transient_place_terms, 2, .unsupported_place_read_term);

    const out_of_scope_bound_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
    };
    try expectCanonicalHashUnsupported(&out_of_scope_bound_terms, 0, .bound_variable_out_of_scope);

    const non_u256_binder_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "i256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "i256" } } },
        .{ .binary = .{ .op = .sge, .lhs = 0, .rhs = 1 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "i256" } },
            .body = 2,
        } },
        .{ .bool_lit = true },
    };
    try expectCanonicalHashUnsupportedWithAssumption(&non_u256_binder_terms, 3, 4, .unsupported_quantifier_binder_type);

    const mismatched_bound_type_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "i256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "i256" } } },
        .{ .binary = .{ .op = .sge, .lhs = 0, .rhs = 1 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 2,
        } },
        .{ .bool_lit = true },
    };
    try expectCanonicalHashUnsupportedWithAssumption(&mismatched_bound_type_terms, 3, 4, .unsupported_bound_variable_type);
}

test "canonical Z3 adapter gives repeated scalar place reads one storage symbol" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "place" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "place" } },
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

    try expectCanonicalSupported(set, queries[0]);
    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    try std.testing.expectEqual(CheckStatus.proved, try adapter.checkObligation(1));
}

test "canonical Z3 adapter links old scalar place reads to entry storage" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const old_first_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .{ .binary = .{ .op = .eq, .lhs = 1, .rhs = 0 } },
    };
    const current_first_terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .old = 0 },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
    };

    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "place" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "place" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};

    const old_first_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &old_first_terms,
    };
    try expectCanonicalSupported(old_first_set, queries[0]);
    var old_first_adapter = Adapter.init(&z3_ctx, std.testing.allocator, old_first_set);
    const old_first_hash = try old_first_adapter.queryHash(2);
    try std.testing.expectEqual(@as(u32, 2), old_first_hash.constraint_count);
    try std.testing.expectEqual(CheckStatus.proved, try old_first_adapter.checkObligation(1));

    const current_first_set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &current_first_terms,
    };
    try expectCanonicalSupported(current_first_set, queries[0]);
    var current_first_adapter = Adapter.init(&z3_ctx, std.testing.allocator, current_first_set);
    const current_first_hash = try current_first_adapter.queryHash(2);
    try std.testing.expectEqual(@as(u32, 2), current_first_hash.constraint_count);
    try std.testing.expectEqual(CheckStatus.proved, try current_first_adapter.checkObligation(1));
}

test "canonical Z3 query state absence fails closed" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .place_read = .{ .root = "balance", .region = .storage } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
    };
    const set: obligation.ObligationSet = .{
        .terms = &terms,
    };

    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    try std.testing.expectError(error.MissingCanonicalQueryState, adapter.encodeFormula(.{ .term = 2 }));
}

test "canonical Z3 quantifier conditions use live connective shapes" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const forall_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .lt, .lhs = 0, .rhs = 1 } },
        .{ .bool_lit = false },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .condition = 2,
            .body = 3,
        } },
        .{ .bool_lit = false },
    };
    const exists_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .lt, .lhs = 0, .rhs = 1 } },
        .{ .bool_lit = true },
        .{ .quantified = .{
            .quantifier = .exists,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .condition = 2,
            .body = 3,
        } },
        .{ .bool_lit = false },
    };
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = 4 },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 5 } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = &assumption_ids,
        .obligation_ids = &obligation_ids,
        .solver_logic = .all,
    }};

    const forall_set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &forall_terms,
    };
    try expectCanonicalSupported(forall_set, queries[0]);
    var forall_adapter = Adapter.init(&z3_ctx, std.testing.allocator, forall_set);
    try std.testing.expectEqual(CheckStatus.disproved, try forall_adapter.checkObligation(2));

    const exists_set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &exists_terms,
    };
    try expectCanonicalSupported(exists_set, queries[0]);
    var exists_adapter = Adapter.init(&z3_ctx, std.testing.allocator, exists_set);
    try std.testing.expectEqual(CheckStatus.proved, try exists_adapter.checkObligation(2));
}

test "canonical Z3 quantifier alpha rename preserves support surface only" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const original_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 1,
        } },
        .{ .bool_lit = true },
    };
    const renamed_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "z", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "z", .ty = .{ .spelling = "u256" } },
            .body = 1,
        } },
        .{ .bool_lit = true },
    };

    // This pins semantic support and promotion shape only. Byte hashes remain
    // name-sensitive because live Z3 prints user binder names.
    try expectCanonicalHashSucceedsWithAssumption(&z3_ctx, &original_terms, 2, 3);
    try expectCanonicalHashSucceedsWithAssumption(&z3_ctx, &renamed_terms, 2, 3);
}

test "canonical Z3 quantifier stack handles nesting and shadowing" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const nested_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 1, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .bound = .{ .index = 0, .name = "j", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 1 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "j", .ty = .{ .spelling = "u256" } },
            .body = 2,
        } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 3,
        } },
        .{ .bool_lit = false },
    };
    const shadow_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 1 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 2,
        } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 3,
        } },
        .{ .bool_lit = false },
    };
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "nested" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = 4 },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "nested" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 5 } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "nested" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = &assumption_ids,
        .obligation_ids = &obligation_ids,
        .solver_logic = .all,
    }};

    const nested_set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &nested_terms,
    };
    try expectCanonicalSupported(nested_set, queries[0]);
    try std.testing.expectEqual(CanonicalPromotionShape.quantified_formula, queryCanonicalPromotionShape(nested_set, queries[0]).?);
    var nested_adapter = Adapter.init(&z3_ctx, std.testing.allocator, nested_set);
    const nested_hash = try nested_adapter.queryHash(3);
    try std.testing.expect(nested_hash.constraint_count > 0);
    try std.testing.expectEqual(CheckStatus.proved, try nested_adapter.checkObligation(2));

    const shadow_set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &shadow_terms,
    };
    try expectCanonicalSupported(shadow_set, queries[0]);
    var shadow_adapter = Adapter.init(&z3_ctx, std.testing.allocator, shadow_set);
    try std.testing.expectEqual(CheckStatus.disproved, try shadow_adapter.checkObligation(2));
}

test "canonical Z3 quantifier shape remains diagnostic only when flagged" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 1,
        } },
        .{ .bool_lit = false },
    };
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = 2 },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 3 } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "quantified" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = &assumption_ids,
        .obligation_ids = &obligation_ids,
        .canonical_smt_crosscheck_required = true,
        .canonical_smt_annotation_pure = true,
        .solver_logic = .all,
    }};
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    try std.testing.expectEqual(CanonicalPromotionShape.quantified_formula, queryCanonicalPromotionShape(set, queries[0]).?);
    try std.testing.expect(!queryCanonicalCrosscheckRequiredByPolicy(set, queries[0]));
    try std.testing.expect(!queryCanonicalRequiredModePromoted(set, queries[0]));
}

test "canonical Z3 classifier agrees with hash adapter on supported and unsupported rows" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 1, .pattern_id = 10 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
        .{ .old = 0 },
        .{ .binary = .{ .op = .eq, .lhs = 3, .rhs = 0 } },
        .result,
        .result,
        .{ .binary = .{ .op = .eq, .lhs = 5, .rhs = 6 } },
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
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 2 } } },
        },
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 1 } },
            .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 4 } } },
        },
        .{
            .id = 6,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 2 } },
            .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 7 } } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const supported_obligation_ids = [_]obligation.Id{2};
    const unsupported_obligation_ids = [_]obligation.Id{3};
    const unsupported_result_obligation_ids = [_]obligation.Id{6};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &supported_obligation_ids,
            .assumption_ids = &assumption_ids,
        },
        .{
            .id = 5,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &unsupported_obligation_ids,
            .assumption_ids = &assumption_ids,
        },
        .{
            .id = 7,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &unsupported_result_obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    const hash = try adapter.queryHash(4);
    try std.testing.expect(hash.constraint_count > 0);

    try expectCanonicalUnsupported(set, queries[1], .unsupported_old_term);
    try std.testing.expectError(error.UnsupportedOldTerm, adapter.queryHash(5));
    try expectCanonicalUnsupported(set, queries[2], .unsupported_result_term);
    try std.testing.expectError(error.UnsupportedResultTerm, adapter.queryHash(7));
}

test "canonical Z3 hash contract sorts stored assumptions by formal id" {
    var z3_ctx = try z3_verification.Z3Context.init(std.testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 2, .pattern_id = 20 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .int_lit = .{ .value = "2", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 2 } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 0 } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 3 },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 1 } },
            .kind = .requires,
            .formula = .{ .term = 4 },
        },
    };
    const obligations = [_]obligation.Obligation{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "checked" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = 5 } } },
    }};
    const obligation_ids = [_]obligation.Id{3};
    const assumptions_forward = [_]obligation.Id{ 1, 2 };
    const assumptions_reversed = [_]obligation.Id{ 2, 1 };
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumptions_forward,
        },
        .{
            .id = 5,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumptions_reversed,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectCanonicalSupported(set, queries[0]);
    try expectCanonicalSupported(set, queries[1]);

    var adapter = Adapter.init(&z3_ctx, std.testing.allocator, set);
    const forward = try adapter.queryHash(4);
    const reversed = try adapter.queryHash(5);
    try std.testing.expectEqual(forward.constraint_count, reversed.constraint_count);
    try std.testing.expectEqual(forward.smtlib_hash, reversed.smtlib_hash);
}
