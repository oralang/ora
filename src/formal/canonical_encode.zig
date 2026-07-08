//! Canonical obligation manifest to Z3 encoding adapter.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const z3 = z3_verification.z3_c;
const Context = z3_verification.Z3Context;
const Solver = z3_verification.Z3Solver;
const obligation = @import("obligation.zig");
const ct = @import("canonical_types.zig");
const support = @import("canonical_support.zig");

const CanonicalPlaceSymbolKind = ct.CanonicalPlaceSymbolKind;
const CanonicalPromotionShape = ct.CanonicalPromotionShape;
const CanonicalQueryState = ct.CanonicalQueryState;
const CanonicalUnsupportedReason = ct.CanonicalUnsupportedReason;
const CheckStatus = ct.CheckStatus;
const EncodeError = ct.EncodeError;
const QueryHash = ct.QueryHash;
const TypeInfo = ct.TypeInfo;
const canonical_promotion_table = ct.canonical_promotion_table;
const expectArgCount = ct.expectArgCount;
const refinement_builtin_map = ct.refinement_builtin_map;
const quantifierBinderTypeInfo = ct.quantifierBinderTypeInfo;
const typeInfo = ct.typeInfo;
const typeInfoIsU256 = ct.typeInfoIsU256;
const u256TypeInfo = ct.u256TypeInfo;
const variableTypeRef = ct.variableTypeRef;
const expectedInfoForBinaryOperands = support.expectedInfoForBinaryOperands;
const placeReadCanonicalSupport = support.placeReadCanonicalSupport;
const staticTermTypeInfo = support.staticTermTypeInfo;
const termContainsResult = support.termContainsResult;
const typeInfoEql = support.typeInfoEql;
const typeInfoSameZ3Sort = support.typeInfoSameZ3Sort;
const walkTerm = support.walkTerm;
const queryCanonicalCrosscheckRequiredByPolicy = support.queryCanonicalCrosscheckRequiredByPolicy;
const queryCanonicalPromotionShape = support.queryCanonicalPromotionShape;
const queryCanonicalRequiredModePromoted = support.queryCanonicalRequiredModePromoted;
const queryCanonicalSupport = support.queryCanonicalSupport;

pub const Adapter = struct {
    context: *Context,
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    query_state: ?*CanonicalQueryState,

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
        const query = self.findQuery(id) orelse return error.UnknownQuery;
        return self.checkQueryRow(query);
    }

    pub fn queryHash(self: *Adapter, id: obligation.Id) EncodeError!QueryHash {
        const query = self.findQuery(id) orelse return error.UnknownQuery;
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
        const target = self.findObligation(query.obligation_ids[0]) orelse return error.UnknownObligation;

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
            const assumption = self.findAssumption(assumption_id) orelse return error.UnknownAssumption;
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

    fn findQuery(self: *const Adapter, id: obligation.Id) ?obligation.VerificationQuery {
        for (self.set.queries) |item| {
            if (item.id == id) return item;
        }
        return null;
    }

    fn findObligation(self: *const Adapter, id: obligation.Id) ?obligation.Obligation {
        for (self.set.obligations) |item| {
            if (item.id == id) return item;
        }
        return null;
    }

    fn findAssumption(self: *const Adapter, id: obligation.Id) ?obligation.Assumption {
        for (self.set.assumptions) |item| {
            if (item.id == id) return item;
        }
        return null;
    }

    fn formulaForObligation(self: *Adapter, kind: obligation.Kind) EncodeError!z3.Z3_ast {
        return switch (kind) {
            .logical => |logical| self.encodeFormula(logical.formula),
            .runtime_guard => |guard| self.encodeFormula(guard.formula),
            .resource,
            .quantifier,
            .type_wf,
            .type_relation,
            .region_relation,
            .effect_frame,
            .filtered_input,
            .backend_fact,
            => error.UnsupportedObligationKind,
        };
    }

    fn goalCounterexampleFormula(self: *Adapter, kind: obligation.Kind) EncodeError!z3.Z3_ast {
        return switch (kind) {
            .logical => |logical| self.encodeGoalFormula(logical.formula),
            .runtime_guard => |guard| self.encodeGoalFormula(guard.formula),
            .resource,
            .quantifier,
            .type_wf,
            .type_relation,
            .region_relation,
            .effect_frame,
            .filtered_input,
            .backend_fact,
            => error.UnsupportedObligationKind,
        };
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
        switch (placeReadCanonicalSupport(place)) {
            .supported => {},
            .unsupported => return error.UnsupportedPlaceReadTerm,
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
        switch (placeReadCanonicalSupport(place)) {
            .supported => {},
            .unsupported => return error.UnsupportedOldTerm,
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
        const name = try self.allocator.dupeZ(u8, name_text);
        defer self.allocator.free(name);

        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name.ptr);
        const ast = z3.Z3_mk_const(self.context.ctx, symbol, sort);
        try self.context.checkNoError();
        return ast;
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
        const sort = try self.sortForType(try typeInfo(ty));
        // Canonical SMT is byte-parity with the live encoder, which names free
        // variables from the source surface. Semantic identity remains FreeVarId
        // in the manifest and Lean denotation.
        const name = try self.allocator.dupeZ(u8, free.name);
        defer self.allocator.free(name);

        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name.ptr);
        const ast = z3.Z3_mk_const(self.context.ctx, symbol, sort);
        try self.context.checkNoError();
        return ast;
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
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
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
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
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
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = assumption_term },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = &assumption_ids,
        .obligation_ids = &obligation_ids,
        .solver_logic = .all,
    }};
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
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    }};
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
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
    const assumptions = [_]obligation.Assumption{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = assumption_term },
    }};
    const obligations = [_]obligation.Obligation{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    }};
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 3,
        .owner = .{ .function = .{ .name = "matrix" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = &assumption_ids,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    try expectCanonicalUnsupported(set, queries[0], expected);
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
