//! Canonical Z3 support and promotion classification.

const std = @import("std");
const obligation = @import("obligation.zig");
const ct = @import("canonical_types.zig");

const CanonicalPromotionShape = ct.CanonicalPromotionShape;
const CanonicalSupport = ct.CanonicalSupport;
const CanonicalSupportScope = ct.CanonicalSupportScope;
const EncodeError = ct.EncodeError;
const TypeInfo = ct.TypeInfo;
const canonical_promotion_table = ct.canonical_promotion_table;
const canonicalPromotionPolicy = ct.canonicalPromotionPolicy;
const canonicalPromotionShapeRequired = ct.canonicalPromotionShapeRequired;
const expectArgCount = ct.expectArgCount;
const refinement_builtin_map = ct.refinement_builtin_map;
const quantifierBinderTypeInfo = ct.quantifierBinderTypeInfo;
const typeInfo = ct.typeInfo;
const typeInfoIsU256 = ct.typeInfoIsU256;
const u256TypeInfo = ct.u256TypeInfo;
const variableTypeRef = ct.variableTypeRef;

pub fn queryCanonicalSupport(set: obligation.ObligationSet, query: obligation.VerificationQuery) CanonicalSupport {
    if (query.kind != .obligation) return .{ .unsupported = .unsupported_query_kind };
    if (query.obligation_ids.len != 1) return .{ .unsupported = .query_not_single_obligation };

    for (query.assumption_ids) |assumption_id| {
        const assumption = obligation.findById(set.assumptions, assumption_id) orelse return .{ .unsupported = .unknown_assumption };
        const formula = assumption.formula orelse return .{ .unsupported = .null_assumption_formula };
        switch (formulaCanonicalSupport(set, formula, set.terms.len + 1, .{})) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }

    const target = obligation.findById(set.obligations, query.obligation_ids[0]) orelse return .{ .unsupported = .unknown_obligation };
    return kindCanonicalSupport(set, target.kind, set.terms.len + 1);
}

pub fn queryCanonicalPromotionShape(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) ?CanonicalPromotionShape {
    if (query.kind != .obligation or query.obligation_ids.len != 1) return null;

    var features: PromotionFeatures = .{};
    for (query.assumption_ids) |assumption_id| {
        const assumption = obligation.findById(set.assumptions, assumption_id) orelse return null;
        const formula = assumption.formula orelse return null;
        if (!collectFormulaPromotionFeatures(set, formula, set.terms.len + 1, &features)) return null;
    }

    const target = obligation.findById(set.obligations, query.obligation_ids[0]) orelse return null;
    if (!collectKindPromotionFeatures(set, target.kind, set.terms.len + 1, &features)) return null;
    return features.shape();
}

pub fn queryCanonicalRequiredModePromoted(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) bool {
    if (!query.canonical_smt_crosscheck_required) return false;
    if (!query.canonical_smt_annotation_pure) return false;
    const shape = queryCanonicalPromotionShape(set, query) orelse return false;
    return canonicalPromotionShapeRequired(shape);
}

pub fn queryCanonicalCrosscheckRequiredByPolicy(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) bool {
    const shape = queryCanonicalPromotionShape(set, query) orelse return false;
    const policy = canonicalPromotionPolicy(shape) orelse return false;
    return policy.required_mode and policy.rollout_enabled;
}

const PromotionFeatures = struct {
    formula_count: u32 = 0,
    result_terms: u32 = 0,
    scalar_place_reads: u32 = 0,
    old_scalar_place_reads: u32 = 0,
    arithmetic_ops: u32 = 0,
    connective_ops: u32 = 0,
    refinement_ops: u32 = 0,
    quantifier_terms: u32 = 0,

    fn markFormula(self: *PromotionFeatures) void {
        self.formula_count +|= 1;
    }

    fn shape(self: PromotionFeatures) ?CanonicalPromotionShape {
        if (self.formula_count == 0) return null;
        if (self.quantifier_terms > 0) return .quantified_formula;

        var atom_classes: u32 = 0;
        if (self.result_terms > 0) atom_classes += 1;
        if (self.scalar_place_reads > 0) atom_classes += 1;
        if (self.old_scalar_place_reads > 0) atom_classes += 1;

        if (self.formula_count > 1 or
            self.arithmetic_ops > 0 or
            self.connective_ops > 0 or
            self.refinement_ops > 0 or
            atom_classes > 1)
        {
            return .formula_combination;
        }
        if (self.old_scalar_place_reads > 0) return .old_scalar_place_read;
        if (self.scalar_place_reads > 0) return .scalar_place_read;
        if (self.result_terms > 0) return .result_term;
        return .core_formula;
    }
};

const PromotionValueContext = enum(u8) {
    none,
    value,
};

pub fn walkTerm(
    comptime Visitor: type,
    visitor: *Visitor,
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    state: Visitor.State,
) Visitor.Result {
    if (fuel == 0 or id >= set.terms.len) return visitor.visitInvalidTerm();
    return switch (set.terms[id]) {
        .bool_lit => |value| if (@hasDecl(Visitor, "visitBoolLit"))
            visitor.visitBoolLit(set, value, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .int_lit => |literal| if (@hasDecl(Visitor, "visitIntLit"))
            visitor.visitIntLit(set, literal, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .variable => |variable| if (@hasDecl(Visitor, "visitVariable"))
            visitor.visitVariable(set, variable, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .old => |operand| if (@hasDecl(Visitor, "visitOld"))
            visitor.visitOld(set, operand, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .result => if (@hasDecl(Visitor, "visitResult"))
            visitor.visitResult(set, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .place_read => |place| if (@hasDecl(Visitor, "visitPlaceRead"))
            visitor.visitPlaceRead(set, place, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .unary => |unary| if (@hasDecl(Visitor, "visitUnary"))
            visitor.visitUnary(set, unary, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .binary => |binary| if (@hasDecl(Visitor, "visitBinary"))
            visitor.visitBinary(set, binary, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .refinement_predicate => |predicate| if (@hasDecl(Visitor, "visitRefinementPredicate"))
            visitor.visitRefinementPredicate(set, predicate, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
        .quantified => |quantified| if (@hasDecl(Visitor, "visitQuantified"))
            visitor.visitQuantified(set, quantified, fuel, state)
        else
            visitor.visitDefault(set, set.terms[id], fuel, state),
    };
}

const PromotionWalkState = struct {
    value_context: PromotionValueContext = .none,
    scope: CanonicalSupportScope = .{},
};

const PromotionVisitor = struct {
    features: *PromotionFeatures,

    const State = PromotionWalkState;
    const Result = bool;

    fn visitInvalidTerm(self: *PromotionVisitor) bool {
        _ = self;
        return false;
    }

    fn visitDefault(self: *PromotionVisitor, set: obligation.ObligationSet, term: obligation.Term, fuel: usize, state: State) bool {
        _ = fuel;
        return switch (term) {
            .bool_lit,
            .int_lit,
            => true,
            .variable => |variable| switch (variable) {
                .free => true,
                .bound => |bound| state.scope.containsBound(bound.index),
            },
            .old => |operand| {
                if (operand >= set.terms.len) return false;
                const place = switch (set.terms[operand]) {
                    .place_read => |place| place,
                    else => return false,
                };
                switch (placeReadCanonicalSupport(place)) {
                    .supported => {
                        self.features.old_scalar_place_reads +|= 1;
                        return true;
                    },
                    .unsupported => return false,
                }
            },
            .result => {
                if (state.value_context != .value) return false;
                self.features.result_terms +|= 1;
                return true;
            },
            .place_read => |place| switch (placeReadCanonicalSupport(place)) {
                .supported => {
                    self.features.scalar_place_reads +|= 1;
                    return true;
                },
                .unsupported => return false,
            },
            else => false,
        };
    }

    fn visitUnary(self: *PromotionVisitor, set: obligation.ObligationSet, unary: obligation.UnaryTerm, fuel: usize, state: State) bool {
        const child_context: PromotionValueContext = switch (unary.op) {
            .not => blk: {
                self.features.connective_ops +|= 1;
                break :blk .none;
            },
            .neg => .value,
        };
        return walkTerm(PromotionVisitor, self, set, unary.operand, fuel - 1, .{
            .value_context = child_context,
            .scope = state.scope,
        });
    }

    fn visitBinary(self: *PromotionVisitor, set: obligation.ObligationSet, binary: obligation.BinaryTerm, fuel: usize, state: State) bool {
        const child_context: PromotionValueContext = switch (binary.op) {
            .and_,
            .or_,
            .implies,
            => .none,
            else => .value,
        };
        switch (binary.op) {
            .add,
            .sub,
            .mul,
            .pow,
            .shl,
            .shr,
            .div,
            .mod,
            .bit_and,
            .bit_xor,
            => self.features.arithmetic_ops +|= 1,
            .and_,
            .or_,
            .implies,
            => self.features.connective_ops +|= 1,
            .eq,
            .ne,
            .lt,
            .le,
            .gt,
            .ge,
            .slt,
            .sle,
            .sgt,
            .sge,
            => {},
        }
        const child_state: State = .{
            .value_context = child_context,
            .scope = state.scope,
        };
        return walkTerm(PromotionVisitor, self, set, binary.lhs, fuel - 1, child_state) and
            walkTerm(PromotionVisitor, self, set, binary.rhs, fuel - 1, child_state);
    }

    fn visitRefinementPredicate(self: *PromotionVisitor, set: obligation.ObligationSet, predicate: obligation.RefinementPredicateTerm, fuel: usize, state: State) bool {
        if (refinement_builtin_map.get(predicate.name) == null) return false;
        self.features.refinement_ops +|= 1;
        const child_state: State = .{
            .value_context = .value,
            .scope = state.scope,
        };
        if (!walkTerm(PromotionVisitor, self, set, predicate.value, fuel - 1, child_state)) return false;
        for (predicate.args) |arg| {
            if (!walkTerm(PromotionVisitor, self, set, arg, fuel - 1, child_state)) return false;
        }
        return true;
    }

    fn visitQuantified(self: *PromotionVisitor, set: obligation.ObligationSet, quantified: obligation.QuantifiedTerm, fuel: usize, state: State) bool {
        if (quantified.binder.origin == .function_param and quantified.condition != null) return false;
        if (quantified.binder.origin != .function_param) {
            self.features.quantifier_terms +|= 1;
        }
        const child_scope = state.scope.push(quantified.binder.origin);
        const child_state: State = .{
            .value_context = .none,
            .scope = child_scope,
        };
        if (quantified.binder.origin != .function_param) {
            if (quantified.condition) |condition| {
                if (!walkTerm(PromotionVisitor, self, set, condition, fuel - 1, child_state)) return false;
            }
        }
        return walkTerm(PromotionVisitor, self, set, quantified.body, fuel - 1, child_state);
    }
};

fn collectKindPromotionFeatures(
    set: obligation.ObligationSet,
    kind: obligation.Kind,
    fuel: usize,
    features: *PromotionFeatures,
) bool {
    const formula = obligation.kindFormula(kind) orelse return false;
    return collectFormulaPromotionFeatures(set, formula, fuel, features);
}

fn collectFormulaPromotionFeatures(
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    fuel: usize,
    features: *PromotionFeatures,
) bool {
    features.markFormula();
    return switch (formula) {
        .term => |term_id| collectTermPromotionFeatures(set, term_id, fuel, .none, .{}, features),
        .origin_value => false,
    };
}

fn collectTermPromotionFeatures(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    value_context: PromotionValueContext,
    scope: CanonicalSupportScope,
    features: *PromotionFeatures,
) bool {
    var visitor: PromotionVisitor = .{ .features = features };
    return walkTerm(PromotionVisitor, &visitor, set, id, fuel, .{
        .value_context = value_context,
        .scope = scope,
    });
}

fn kindCanonicalSupport(set: obligation.ObligationSet, kind: obligation.Kind, fuel: usize) CanonicalSupport {
    const formula = obligation.kindFormula(kind) orelse return .{ .unsupported = .unsupported_obligation_kind };
    return formulaCanonicalSupport(set, formula, fuel, .{});
}

fn formulaCanonicalSupport(
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    fuel: usize,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    return switch (formula) {
        .term => |term_id| termCanonicalSupport(set, term_id, fuel, scope),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

const SupportWalkState = struct {
    scope: CanonicalSupportScope = .{},
    expected: ?TypeInfo = null,
};

const SupportVisitor = struct {
    const State = SupportWalkState;
    const Result = CanonicalSupport;

    fn visitInvalidTerm(self: *SupportVisitor) CanonicalSupport {
        _ = self;
        return .{ .unsupported = .unsupported_obligation_kind };
    }

    fn visitDefault(self: *SupportVisitor, set: obligation.ObligationSet, term: obligation.Term, fuel: usize, state: State) CanonicalSupport {
        _ = self;
        return switch (term) {
            .bool_lit => .supported,
            .int_lit => |literal| typeRefCanonicalSupport(literal.ty),
            .variable => |variable| varRefCanonicalSupport(variable, state.scope),
            .old => |operand| oldCanonicalSupport(set, operand, fuel - 1),
            .result => blk: {
                const info = state.expected orelse break :blk .{ .unsupported = .unsupported_result_term };
                if (info.kind != .bitvector) break :blk .{ .unsupported = .unsupported_result_term };
                break :blk .supported;
            },
            .place_read => |place| placeReadCanonicalSupport(place),
            else => .{ .unsupported = .unsupported_obligation_kind },
        };
    }

    fn visitUnary(
        self: *SupportVisitor,
        set: obligation.ObligationSet,
        unary: obligation.UnaryTerm,
        fuel: usize,
        state: State,
    ) CanonicalSupport {
        _ = self;
        return termCanonicalSupport(set, unary.operand, fuel - 1, state.scope);
    }

    fn visitBinary(
        self: *SupportVisitor,
        set: obligation.ObligationSet,
        binary: obligation.BinaryTerm,
        fuel: usize,
        state: State,
    ) CanonicalSupport {
        _ = self;
        return binaryCanonicalSupport(set, binary, fuel - 1, state.scope);
    }

    fn visitRefinementPredicate(
        self: *SupportVisitor,
        set: obligation.ObligationSet,
        predicate: obligation.RefinementPredicateTerm,
        fuel: usize,
        state: State,
    ) CanonicalSupport {
        _ = self;
        return refinementPredicateCanonicalSupport(set, predicate, fuel - 1, state.scope);
    }

    fn visitQuantified(
        self: *SupportVisitor,
        set: obligation.ObligationSet,
        quantified: obligation.QuantifiedTerm,
        fuel: usize,
        state: State,
    ) CanonicalSupport {
        _ = self;
        return quantifiedCanonicalSupport(set, quantified, fuel - 1, state.scope);
    }
};

fn termCanonicalSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    var visitor: SupportVisitor = .{};
    return walkTerm(SupportVisitor, &visitor, set, id, fuel, .{ .scope = scope });
}

pub fn placeReadCanonicalSupport(place: obligation.PlaceRef) CanonicalSupport {
    if (place.region != .storage) return .{ .unsupported = .unsupported_place_read_term };
    if (place.fields.len != 0 or place.keys.len != 0) return .{ .unsupported = .unsupported_place_read_term };
    return .supported;
}

pub fn requireSupportedScalarPlace(place: obligation.PlaceRef) EncodeError!void {
    switch (placeReadCanonicalSupport(place)) {
        .supported => {},
        .unsupported => return error.UnsupportedPlaceReadTerm,
    }
}

fn oldCanonicalSupport(set: obligation.ObligationSet, operand: obligation.TermId, fuel: usize) CanonicalSupport {
    if (fuel == 0 or operand >= set.terms.len) return .{ .unsupported = .unsupported_old_term };
    return switch (set.terms[operand]) {
        .place_read => |place| switch (placeReadCanonicalSupport(place)) {
            .supported => .supported,
            .unsupported => .{ .unsupported = .unsupported_old_term },
        },
        else => .{ .unsupported = .unsupported_old_term },
    };
}

fn binaryCanonicalSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    const expected = expectedInfoForBinaryOperands(set, binary) catch |err| return canonicalReasonForTypeError(err);
    const support = firstUnsupported(.{
        termCanonicalSupportWithExpected(set, binary.lhs, fuel, expected, scope),
        termCanonicalSupportWithExpected(set, binary.rhs, fuel, expected, scope),
    });
    switch (support) {
        .supported => {},
        .unsupported => return support,
    }
    return binaryResultOperandTypesMatchExpected(set, binary, expected, fuel);
}

fn termCanonicalSupportWithExpected(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    expected: ?TypeInfo,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    var visitor: SupportVisitor = .{};
    // `result` carries no TypeRef of its own; only a typed parent may pick its sort.
    return walkTerm(SupportVisitor, &visitor, set, id, fuel, .{
        .scope = scope,
        .expected = expected,
    });
}

fn binaryResultOperandTypesMatchExpected(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    expected: ?TypeInfo,
    fuel: usize,
) CanonicalSupport {
    const expected_info = expected orelse return .supported;
    if (!termContainsResult(set, binary.lhs, fuel) and !termContainsResult(set, binary.rhs, fuel)) {
        return .supported;
    }
    return firstUnsupported(.{
        termTypeMatchesExpected(set, binary.lhs, expected_info),
        termTypeMatchesExpected(set, binary.rhs, expected_info),
    });
}

fn termTypeMatchesExpected(set: obligation.ObligationSet, id: obligation.TermId, expected: TypeInfo) CanonicalSupport {
    if (id >= set.terms.len) return .{ .unsupported = .unsupported_obligation_kind };
    if (set.terms[id] == .result) return .supported;
    const actual = staticTermTypeInfo(set, id) catch |err| return canonicalReasonForTypeError(err);
    if (!typeInfoEql(expected, actual)) return .{ .unsupported = .unsupported_type };
    return .supported;
}

const ContainsResultVisitor = struct {
    const State = void;
    const Result = bool;

    fn visitInvalidTerm(self: *ContainsResultVisitor) bool {
        _ = self;
        return false;
    }

    fn visitDefault(self: *ContainsResultVisitor, set: obligation.ObligationSet, term: obligation.Term, fuel: usize, state: State) bool {
        _ = self;
        _ = set;
        _ = fuel;
        _ = state;
        return term == .result;
    }

    fn visitOld(self: *ContainsResultVisitor, set: obligation.ObligationSet, operand: obligation.TermId, fuel: usize, state: State) bool {
        return walkTerm(ContainsResultVisitor, self, set, operand, fuel - 1, state);
    }

    fn visitUnary(self: *ContainsResultVisitor, set: obligation.ObligationSet, unary: obligation.UnaryTerm, fuel: usize, state: State) bool {
        return walkTerm(ContainsResultVisitor, self, set, unary.operand, fuel - 1, state);
    }

    fn visitBinary(self: *ContainsResultVisitor, set: obligation.ObligationSet, binary: obligation.BinaryTerm, fuel: usize, state: State) bool {
        return walkTerm(ContainsResultVisitor, self, set, binary.lhs, fuel - 1, state) or
            walkTerm(ContainsResultVisitor, self, set, binary.rhs, fuel - 1, state);
    }

    fn visitRefinementPredicate(self: *ContainsResultVisitor, set: obligation.ObligationSet, predicate: obligation.RefinementPredicateTerm, fuel: usize, state: State) bool {
        if (walkTerm(ContainsResultVisitor, self, set, predicate.value, fuel - 1, state)) return true;
        for (predicate.args) |arg| {
            if (walkTerm(ContainsResultVisitor, self, set, arg, fuel - 1, state)) return true;
        }
        return false;
    }

    fn visitQuantified(self: *ContainsResultVisitor, set: obligation.ObligationSet, quantified: obligation.QuantifiedTerm, fuel: usize, state: State) bool {
        if (quantified.condition) |condition| {
            if (walkTerm(ContainsResultVisitor, self, set, condition, fuel - 1, state)) return true;
        }
        return walkTerm(ContainsResultVisitor, self, set, quantified.body, fuel - 1, state);
    }
};

pub fn termContainsResult(set: obligation.ObligationSet, id: obligation.TermId, fuel: usize) bool {
    var visitor: ContainsResultVisitor = .{};
    return walkTerm(ContainsResultVisitor, &visitor, set, id, fuel, {});
}

fn expectedInfoFromBinaryType(ty: ?obligation.TypeRef) EncodeError!?TypeInfo {
    const value = ty orelse return null;
    const info = try typeInfo(value);
    if (info.kind != .bitvector) return error.ExpectedBitVector;
    return info;
}

fn expectedInfoForEquality(set: obligation.ObligationSet, binary: obligation.BinaryTerm) EncodeError!?TypeInfo {
    if (try expectedInfoFromBinaryType(binary.ty)) |info| return info;

    const lhs_result = termIsResult(set, binary.lhs);
    const rhs_result = termIsResult(set, binary.rhs);
    if (lhs_result and !rhs_result) return try staticTermTypeInfo(set, binary.rhs);
    if (rhs_result and !lhs_result) return try staticTermTypeInfo(set, binary.lhs);
    return null;
}

pub fn expectedInfoForBinaryOperands(set: obligation.ObligationSet, binary: obligation.BinaryTerm) EncodeError!?TypeInfo {
    return switch (binary.op) {
        .lt,
        .le,
        .gt,
        .ge,
        => .{ .kind = .bitvector, .width = 256, .signed = false },
        .slt,
        .sle,
        .sgt,
        .sge,
        .add,
        .sub,
        .mul,
        .pow,
        .shl,
        .shr,
        .div,
        .mod,
        .bit_and,
        .bit_xor,
        => if (try expectedInfoFromBinaryType(binary.ty)) |info| info else null,
        .eq,
        .ne,
        => try expectedInfoForEquality(set, binary),
        .and_,
        .or_,
        .implies,
        => null,
    };
}

fn termIsResult(set: obligation.ObligationSet, id: obligation.TermId) bool {
    return id < set.terms.len and set.terms[id] == .result;
}

const StaticTypeVisitor = struct {
    const State = void;
    const Result = EncodeError!TypeInfo;

    fn visitInvalidTerm(self: *StaticTypeVisitor) EncodeError!TypeInfo {
        _ = self;
        return error.InvalidTermReference;
    }

    fn visitDefault(self: *StaticTypeVisitor, set: obligation.ObligationSet, term: obligation.Term, fuel: usize, state: State) EncodeError!TypeInfo {
        _ = self;
        _ = set;
        _ = fuel;
        _ = state;
        return switch (term) {
            .bool_lit,
            .refinement_predicate,
            .quantified,
            => .{ .kind = .bool },
            .int_lit => |literal| typeInfo(literal.ty orelse return error.MissingType),
            .variable => |variable| typeInfo(variableTypeRef(variable) orelse return error.MissingType),
            .result => error.MissingType,
            .place_read => |place| blk: {
                try requireSupportedScalarPlace(place);
                break :blk u256TypeInfo();
            },
            else => error.InvalidTermReference,
        };
    }

    fn visitOld(self: *StaticTypeVisitor, set: obligation.ObligationSet, operand: obligation.TermId, fuel: usize, state: State) EncodeError!TypeInfo {
        return walkTerm(StaticTypeVisitor, self, set, operand, fuel - 1, state);
    }

    fn visitUnary(self: *StaticTypeVisitor, set: obligation.ObligationSet, unary: obligation.UnaryTerm, fuel: usize, state: State) EncodeError!TypeInfo {
        return switch (unary.op) {
            .not => .{ .kind = .bool },
            .neg => walkTerm(StaticTypeVisitor, self, set, unary.operand, fuel - 1, state),
        };
    }

    fn visitBinary(self: *StaticTypeVisitor, set: obligation.ObligationSet, nested: obligation.BinaryTerm, fuel: usize, state: State) EncodeError!TypeInfo {
        return switch (nested.op) {
            .eq,
            .ne,
            .lt,
            .le,
            .gt,
            .ge,
            .slt,
            .sle,
            .sgt,
            .sge,
            .and_,
            .or_,
            .implies,
            => .{ .kind = .bool },
            .add,
            .sub,
            .mul,
            .pow,
            .shl,
            .shr,
            .div,
            .mod,
            .bit_and,
            .bit_xor,
            => if (try expectedInfoFromBinaryType(nested.ty)) |info| info else try walkTerm(StaticTypeVisitor, self, set, nested.lhs, fuel - 1, state),
        };
    }
};

pub fn staticTermTypeInfo(set: obligation.ObligationSet, id: obligation.TermId) EncodeError!TypeInfo {
    var visitor: StaticTypeVisitor = .{};
    return walkTerm(StaticTypeVisitor, &visitor, set, id, set.terms.len + 1, {});
}

fn canonicalReasonForTypeError(err: EncodeError) CanonicalSupport {
    return .{ .unsupported = switch (err) {
        error.MissingType => .missing_type,
        error.UnsupportedCompilerTypeId => .unsupported_compiler_type_id,
        error.UnsupportedType => .unsupported_type,
        else => .unsupported_type,
    } };
}

pub fn typeInfoEql(lhs: TypeInfo, rhs: TypeInfo) bool {
    return lhs.kind == rhs.kind and lhs.width == rhs.width and lhs.signed == rhs.signed;
}

pub fn typeInfoSameZ3Sort(lhs: TypeInfo, rhs: TypeInfo) bool {
    return lhs.kind == rhs.kind and lhs.width == rhs.width;
}

fn varRefCanonicalSupport(variable: obligation.VarRef, scope: CanonicalSupportScope) CanonicalSupport {
    return switch (variable) {
        .free => |free| typeRefCanonicalSupport(free.ty),
        .bound => |bound| boundVarCanonicalSupport(bound, scope),
    };
}

fn refinementPredicateCanonicalSupport(
    set: obligation.ObligationSet,
    predicate: obligation.RefinementPredicateTerm,
    fuel: usize,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    if (refinement_builtin_map.get(predicate.name) == null) return .{ .unsupported = .unsupported_refinement };
    switch (termCanonicalSupport(set, predicate.value, fuel, scope)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    for (predicate.args) |arg| {
        switch (termCanonicalSupport(set, arg, fuel, scope)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return switch (refinement_builtin_map.get(predicate.name).?) {
        .non_zero,
        .non_zero_address,
        .basis_points,
        => if (predicate.args.len == 0) .supported else .{ .unsupported = .invalid_refinement_arity },
        .min_value,
        .max_value,
        => if (predicate.args.len == 1) .supported else .{ .unsupported = .invalid_refinement_arity },
        .in_range => if (predicate.args.len == 2) .supported else .{ .unsupported = .invalid_refinement_arity },
    };
}

fn quantifiedCanonicalSupport(
    set: obligation.ObligationSet,
    quantified: obligation.QuantifiedTerm,
    fuel: usize,
    scope: CanonicalSupportScope,
) CanonicalSupport {
    if (quantified.binder.origin == .function_param) {
        if (quantified.condition != null) return .{ .unsupported = .unsupported_function_param_wrapper_condition };
        switch (typeRefCanonicalSupport(quantified.binder.ty)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
        return termCanonicalSupport(set, quantified.body, fuel, scope.push(.function_param));
    }

    switch (quantifierBinderCanonicalSupport(quantified.binder.ty)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    const child_scope = scope.push(quantified.binder.origin);
    if (quantified.condition) |condition| {
        switch (termCanonicalSupport(set, condition, fuel, child_scope)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return termCanonicalSupport(set, quantified.body, fuel, child_scope);
}

fn quantifierBinderCanonicalSupport(maybe_ty: ?obligation.TypeRef) CanonicalSupport {
    _ = quantifierBinderTypeInfo(maybe_ty) catch |err| {
        return .{ .unsupported = switch (err) {
            error.MissingType => .missing_type,
            error.UnsupportedCompilerTypeId => .unsupported_compiler_type_id,
            error.UnsupportedQuantifierBinderType => .unsupported_quantifier_binder_type,
            else => .unsupported_quantifier_binder_type,
        } };
    };
    return .supported;
}

fn boundVarCanonicalSupport(bound: obligation.BoundVarRef, scope: CanonicalSupportScope) CanonicalSupport {
    const origin = scope.boundOrigin(bound.index) orelse return .{ .unsupported = .bound_variable_out_of_scope };
    if (origin == .function_param) return typeRefCanonicalSupport(bound.ty);
    const info = typeInfo(bound.ty orelse return .{ .unsupported = .missing_type }) catch |err| {
        return .{ .unsupported = switch (err) {
            error.UnsupportedCompilerTypeId => .unsupported_compiler_type_id,
            else => .unsupported_bound_variable_type,
        } };
    };
    if (!typeInfoIsU256(info)) return .{ .unsupported = .unsupported_bound_variable_type };
    return .supported;
}

fn typeRefCanonicalSupport(maybe_ty: ?obligation.TypeRef) CanonicalSupport {
    const ty = maybe_ty orelse return .{ .unsupported = .missing_type };
    _ = typeInfo(ty) catch |err| return .{ .unsupported = switch (err) {
        error.UnsupportedCompilerTypeId => .unsupported_compiler_type_id,
        error.UnsupportedType => .unsupported_type,
        else => .unsupported_type,
    } };
    return .supported;
}

fn firstUnsupported(results: anytype) CanonicalSupport {
    inline for (results) |result| {
        switch (result) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

test "canonical Z3 required promotion table covers every shape exactly once" {
    const shape_count = std.meta.fields(CanonicalPromotionShape).len;
    var seen: [shape_count]bool = .{false} ** shape_count;
    var rollout_count: usize = 0;

    for (canonical_promotion_table) |row| {
        const index: usize = @intFromEnum(row.shape);
        try std.testing.expect(index < shape_count);
        try std.testing.expect(!seen[index]);
        seen[index] = true;
        if (row.rollout_enabled) {
            rollout_count += 1;
            try std.testing.expectEqual(CanonicalPromotionShape.core_formula, row.shape);
        }
    }
    for (seen) |item| try std.testing.expect(item);
    try std.testing.expectEqual(@as(usize, 1), rollout_count);
}
