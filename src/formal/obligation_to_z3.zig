//! Canonical obligation manifest to Z3 adapter.
//!
//! This is the first solver-facing adapter for the backend-neutral obligation
//! model. It intentionally encodes only formulas that have already been
//! projected into `obligation.Term`; MLIR-owned `origin_value` formulas still
//! belong to the existing verifier path and fail closed here.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const z3 = z3_verification.z3_c;
const Context = z3_verification.Z3Context;
const Solver = z3_verification.Z3Solver;
const obligation = @import("obligation.zig");
const type_builtin = @import("ora_types").builtin;

const RefinementBuiltin = enum(u8) {
    min_value,
    max_value,
    in_range,
    non_zero_address,
    non_zero,
    basis_points,
};

const refinement_builtin_map = std.StaticStringMap(RefinementBuiltin).initComptime(.{
    .{ "MinValue", .min_value },
    .{ "MaxValue", .max_value },
    .{ "InRange", .in_range },
    .{ "NonZeroAddress", .non_zero_address },
    .{ "NonZero", .non_zero },
    .{ "BasisPoints", .basis_points },
});

pub const CheckStatus = enum(u8) {
    proved,
    disproved,
    unknown,
};

pub const QueryHash = struct {
    constraint_count: u32,
    smtlib_hash: u64,
};

pub const CanonicalSupport = union(enum) {
    supported,
    unsupported: CanonicalUnsupportedReason,
};

pub const CanonicalUnsupportedReason = enum(u8) {
    unsupported_query_kind,
    query_not_single_obligation,
    unknown_assumption,
    null_assumption_formula,
    unknown_obligation,
    unsupported_obligation_kind,
    unsupported_origin_value,
    unsupported_bound_variable_term,
    unsupported_old_term,
    unsupported_result_term,
    unsupported_place_read_term,
    unsupported_quantified_term,
    unsupported_refinement,
    invalid_refinement_arity,
    missing_type,
    unsupported_type,
    unsupported_compiler_type_id,
};

pub const CanonicalPromotionShape = enum(u8) {
    core_formula,
    result_term,
    scalar_place_read,
    old_scalar_place_read,
    formula_combination,
};

pub const CanonicalPromotionPolicy = struct {
    shape: CanonicalPromotionShape,
    required_mode: bool,
};

// Required mode is currently two-key: the query row must explicitly request
// canonical SMT crosscheck and its syntax must be present in this table. The
// table is the audit surface for which shapes may block; the row flag is the
// rollout switch for when production starts arming those shapes.
pub const canonical_promotion_table = [_]CanonicalPromotionPolicy{
    .{ .shape = .core_formula, .required_mode = true },
    .{ .shape = .result_term, .required_mode = true },
    .{ .shape = .scalar_place_read, .required_mode = true },
    .{ .shape = .old_scalar_place_read, .required_mode = true },
    // Full-corpus measurement showed formula combinations often lack live rows;
    // promoted mismatches still need diagnosis before this shape can block.
    .{ .shape = .formula_combination, .required_mode = false },
};

pub const EncodeError = std.mem.Allocator.Error || error{
    AmbiguousQuery,
    ExpectedBitVector,
    ExpectedBool,
    InvalidCharacter,
    InvalidRefinementArity,
    InvalidTermReference,
    MissingFormula,
    MissingType,
    Overflow,
    SolverInitFailed,
    TypeMismatch,
    UnknownAssumption,
    UnknownObligation,
    UnknownQuery,
    UnsupportedCompilerTypeId,
    UnsupportedObligationKind,
    UnsupportedBoundVariableTerm,
    UnsupportedOldTerm,
    UnsupportedOriginValue,
    UnsupportedPlaceReadTerm,
    UnsupportedQuantifiedTerm,
    UnsupportedRefinement,
    UnsupportedResultTerm,
    UnsupportedType,
    Z3ApiError,
};

const CanonicalPlaceSymbolKind = enum(u8) {
    global,
    entry,
};

const CanonicalPlaceRootState = struct {
    root: []const u8,
    current: ?CanonicalPlaceSymbolKind = null,
    entry: ?CanonicalPlaceSymbolKind = null,
};

const CanonicalQueryState = struct {
    place_roots: std.ArrayList(CanonicalPlaceRootState) = .empty,
    side_constraints: std.ArrayList(z3.Z3_ast) = .empty,

    fn deinit(self: *CanonicalQueryState, allocator: std.mem.Allocator) void {
        self.place_roots.deinit(allocator);
        self.side_constraints.deinit(allocator);
    }

    fn getOrPutPlaceRoot(self: *CanonicalQueryState, allocator: std.mem.Allocator, root: []const u8) !*CanonicalPlaceRootState {
        for (self.place_roots.items) |*item| {
            if (std.mem.eql(u8, item.root, root)) return item;
        }
        try self.place_roots.append(allocator, .{ .root = root });
        return &self.place_roots.items[self.place_roots.items.len - 1];
    }
};

pub fn queryCanonicalSupport(set: obligation.ObligationSet, query: obligation.VerificationQuery) CanonicalSupport {
    if (query.kind != .obligation) return .{ .unsupported = .unsupported_query_kind };
    if (query.obligation_ids.len != 1) return .{ .unsupported = .query_not_single_obligation };

    for (query.assumption_ids) |assumption_id| {
        const assumption = findAssumption(set, assumption_id) orelse return .{ .unsupported = .unknown_assumption };
        const formula = assumption.formula orelse return .{ .unsupported = .null_assumption_formula };
        switch (formulaCanonicalSupport(set, formula, set.terms.len + 1)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }

    const target = findObligation(set, query.obligation_ids[0]) orelse return .{ .unsupported = .unknown_obligation };
    return kindCanonicalSupport(set, target.kind, set.terms.len + 1);
}

pub fn queryCanonicalPromotionShape(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) ?CanonicalPromotionShape {
    if (query.kind != .obligation or query.obligation_ids.len != 1) return null;

    var features: PromotionFeatures = .{};
    for (query.assumption_ids) |assumption_id| {
        const assumption = findAssumption(set, assumption_id) orelse return null;
        const formula = assumption.formula orelse return null;
        if (!collectFormulaPromotionFeatures(set, formula, set.terms.len + 1, &features)) return null;
    }

    const target = findObligation(set, query.obligation_ids[0]) orelse return null;
    if (!collectKindPromotionFeatures(set, target.kind, set.terms.len + 1, &features)) return null;
    return features.shape();
}

pub fn queryCanonicalRequiredModePromoted(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) bool {
    if (!query.canonical_smt_crosscheck_required) return false;
    const shape = queryCanonicalPromotionShape(set, query) orelse return false;
    return canonicalPromotionShapeRequired(shape);
}

fn canonicalPromotionShapeRequired(shape: CanonicalPromotionShape) bool {
    for (canonical_promotion_table) |row| {
        if (row.shape == shape) return row.required_mode;
    }
    return false;
}

const PromotionFeatures = struct {
    formula_count: u32 = 0,
    result_terms: u32 = 0,
    scalar_place_reads: u32 = 0,
    old_scalar_place_reads: u32 = 0,
    arithmetic_ops: u32 = 0,
    connective_ops: u32 = 0,
    refinement_ops: u32 = 0,

    fn markFormula(self: *PromotionFeatures) void {
        self.formula_count +|= 1;
    }

    fn shape(self: PromotionFeatures) ?CanonicalPromotionShape {
        if (self.formula_count == 0) return null;

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

fn collectKindPromotionFeatures(
    set: obligation.ObligationSet,
    kind: obligation.Kind,
    fuel: usize,
    features: *PromotionFeatures,
) bool {
    return switch (kind) {
        .logical => |logical| collectFormulaPromotionFeatures(set, logical.formula, fuel, features),
        .runtime_guard => |guard| collectFormulaPromotionFeatures(set, guard.formula, fuel, features),
        .resource,
        .quantifier,
        .type_wf,
        .type_relation,
        .region_relation,
        .effect_frame,
        .filtered_input,
        .backend_fact,
        => false,
    };
}

fn collectFormulaPromotionFeatures(
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    fuel: usize,
    features: *PromotionFeatures,
) bool {
    features.markFormula();
    return switch (formula) {
        .term => |term_id| collectTermPromotionFeatures(set, term_id, fuel, .none, features),
        .origin_value => false,
    };
}

fn collectTermPromotionFeatures(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    value_context: PromotionValueContext,
    features: *PromotionFeatures,
) bool {
    if (fuel == 0 or id >= set.terms.len) return false;
    switch (set.terms[id]) {
        .bool_lit => return true,
        .int_lit => return true,
        .variable => |variable| return switch (variable) {
            .free => true,
            .bound => false,
        },
        .old => |operand| {
            if (operand >= set.terms.len) return false;
            const place = switch (set.terms[operand]) {
                .place_read => |place| place,
                else => return false,
            };
            switch (placeReadCanonicalSupport(place)) {
                .supported => {
                    features.old_scalar_place_reads +|= 1;
                    return true;
                },
                .unsupported => return false,
            }
        },
        .result => {
            if (value_context != .value) return false;
            features.result_terms +|= 1;
            return true;
        },
        .place_read => |place| switch (placeReadCanonicalSupport(place)) {
            .supported => {
                features.scalar_place_reads +|= 1;
                return true;
            },
            .unsupported => return false,
        },
        .unary => |unary| {
            const child_context: PromotionValueContext = switch (unary.op) {
                .not => blk: {
                    features.connective_ops +|= 1;
                    break :blk .none;
                },
                .neg => .value,
            };
            return collectTermPromotionFeatures(set, unary.operand, fuel - 1, child_context, features);
        },
        .binary => |binary| {
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
                .div,
                .mod,
                => features.arithmetic_ops +|= 1,
                .and_,
                .or_,
                .implies,
                => features.connective_ops +|= 1,
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
            return collectTermPromotionFeatures(set, binary.lhs, fuel - 1, child_context, features) and
                collectTermPromotionFeatures(set, binary.rhs, fuel - 1, child_context, features);
        },
        .refinement_predicate => |predicate| {
            if (refinement_builtin_map.get(predicate.name) == null) return false;
            features.refinement_ops +|= 1;
            if (!collectTermPromotionFeatures(set, predicate.value, fuel - 1, .value, features)) return false;
            for (predicate.args) |arg| {
                if (!collectTermPromotionFeatures(set, arg, fuel - 1, .value, features)) return false;
            }
            return true;
        },
        .quantified => return false,
    }
}

fn findAssumption(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Assumption {
    for (set.assumptions) |item| {
        if (item.id == id) return item;
    }
    return null;
}

fn findObligation(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Obligation {
    for (set.obligations) |item| {
        if (item.id == id) return item;
    }
    return null;
}

fn kindCanonicalSupport(set: obligation.ObligationSet, kind: obligation.Kind, fuel: usize) CanonicalSupport {
    return switch (kind) {
        .logical => |logical| formulaCanonicalSupport(set, logical.formula, fuel),
        .runtime_guard => |guard| formulaCanonicalSupport(set, guard.formula, fuel),
        .resource,
        .quantifier,
        .type_wf,
        .type_relation,
        .region_relation,
        .effect_frame,
        .filtered_input,
        .backend_fact,
        => .{ .unsupported = .unsupported_obligation_kind },
    };
}

fn formulaCanonicalSupport(set: obligation.ObligationSet, formula: obligation.FormulaRef, fuel: usize) CanonicalSupport {
    return switch (formula) {
        .term => |term_id| termCanonicalSupport(set, term_id, fuel),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

fn termCanonicalSupport(set: obligation.ObligationSet, id: obligation.TermId, fuel: usize) CanonicalSupport {
    if (fuel == 0 or id >= set.terms.len) return .{ .unsupported = .unsupported_obligation_kind };
    return switch (set.terms[id]) {
        .bool_lit => .supported,
        .int_lit => |literal| typeRefCanonicalSupport(literal.ty),
        .variable => |variable| varRefCanonicalSupport(variable),
        .old => |operand| oldCanonicalSupport(set, operand, fuel - 1),
        .result => .{ .unsupported = .unsupported_result_term },
        .place_read => |place| placeReadCanonicalSupport(place),
        .unary => |unary| termCanonicalSupport(set, unary.operand, fuel - 1),
        .binary => |binary| binaryCanonicalSupport(set, binary, fuel - 1),
        .refinement_predicate => |predicate| refinementPredicateCanonicalSupport(set, predicate, fuel - 1),
        .quantified => .{ .unsupported = .unsupported_quantified_term },
    };
}

fn placeReadCanonicalSupport(place: obligation.PlaceRef) CanonicalSupport {
    if (place.region != .storage) return .{ .unsupported = .unsupported_place_read_term };
    if (place.fields.len != 0 or place.keys.len != 0) return .{ .unsupported = .unsupported_place_read_term };
    return .supported;
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
) CanonicalSupport {
    const expected = expectedInfoForBinaryOperands(set, binary) catch |err| return canonicalReasonForTypeError(err);
    const support = firstUnsupported(.{
        termCanonicalSupportWithExpected(set, binary.lhs, fuel, expected),
        termCanonicalSupportWithExpected(set, binary.rhs, fuel, expected),
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
) CanonicalSupport {
    if (fuel == 0 or id >= set.terms.len) return .{ .unsupported = .unsupported_obligation_kind };
    // `result` carries no TypeRef of its own; only a typed parent may pick its sort.
    if (set.terms[id] == .result) {
        const info = expected orelse return .{ .unsupported = .unsupported_result_term };
        if (info.kind != .bitvector) return .{ .unsupported = .unsupported_result_term };
        return .supported;
    }
    return termCanonicalSupport(set, id, fuel);
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

fn termContainsResult(set: obligation.ObligationSet, id: obligation.TermId, fuel: usize) bool {
    if (fuel == 0 or id >= set.terms.len) return false;
    return switch (set.terms[id]) {
        .result => true,
        .old => |operand| termContainsResult(set, operand, fuel - 1),
        .unary => |unary| termContainsResult(set, unary.operand, fuel - 1),
        .binary => |binary| termContainsResult(set, binary.lhs, fuel - 1) or
            termContainsResult(set, binary.rhs, fuel - 1),
        .refinement_predicate => |predicate| blk: {
            if (termContainsResult(set, predicate.value, fuel - 1)) break :blk true;
            for (predicate.args) |arg| {
                if (termContainsResult(set, arg, fuel - 1)) break :blk true;
            }
            break :blk false;
        },
        .quantified => |quantified| blk: {
            if (quantified.condition) |condition| {
                if (termContainsResult(set, condition, fuel - 1)) break :blk true;
            }
            break :blk termContainsResult(set, quantified.body, fuel - 1);
        },
        .bool_lit,
        .int_lit,
        .variable,
        .place_read,
        => false,
    };
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

fn expectedInfoForBinaryOperands(set: obligation.ObligationSet, binary: obligation.BinaryTerm) EncodeError!?TypeInfo {
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
        .div,
        .mod,
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

fn staticTermTypeInfo(set: obligation.ObligationSet, id: obligation.TermId) EncodeError!TypeInfo {
    if (id >= set.terms.len) return error.InvalidTermReference;
    return switch (set.terms[id]) {
        .bool_lit => .{ .kind = .bool },
        .int_lit => |literal| typeInfo(literal.ty orelse return error.MissingType),
        .variable => |variable| typeInfo(variableTypeRef(variable) orelse return error.MissingType),
        .old => |operand| staticTermTypeInfo(set, operand),
        .unary => |unary| switch (unary.op) {
            .not => .{ .kind = .bool },
            .neg => staticTermTypeInfo(set, unary.operand),
        },
        .binary => |nested| switch (nested.op) {
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
            .div,
            .mod,
            => if (try expectedInfoFromBinaryType(nested.ty)) |info| info else try staticTermTypeInfo(set, nested.lhs),
        },
        .refinement_predicate,
        .quantified,
        => .{ .kind = .bool },
        .result => error.MissingType,
        .place_read => |place| switch (placeReadCanonicalSupport(place)) {
            .supported => u256TypeInfo(),
            .unsupported => error.UnsupportedPlaceReadTerm,
        },
    };
}

fn canonicalReasonForTypeError(err: EncodeError) CanonicalSupport {
    return .{ .unsupported = switch (err) {
        error.MissingType => .missing_type,
        error.UnsupportedCompilerTypeId => .unsupported_compiler_type_id,
        error.UnsupportedType => .unsupported_type,
        else => .unsupported_type,
    } };
}

fn typeInfoEql(lhs: TypeInfo, rhs: TypeInfo) bool {
    return lhs.kind == rhs.kind and lhs.width == rhs.width and lhs.signed == rhs.signed;
}

fn varRefCanonicalSupport(variable: obligation.VarRef) CanonicalSupport {
    return switch (variable) {
        .free => |free| typeRefCanonicalSupport(free.ty),
        .bound => .{ .unsupported = .unsupported_bound_variable_term },
    };
}

fn refinementPredicateCanonicalSupport(
    set: obligation.ObligationSet,
    predicate: obligation.RefinementPredicateTerm,
    fuel: usize,
) CanonicalSupport {
    if (refinement_builtin_map.get(predicate.name) == null) return .{ .unsupported = .unsupported_refinement };
    switch (termCanonicalSupport(set, predicate.value, fuel)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    for (predicate.args) |arg| {
        switch (termCanonicalSupport(set, arg, fuel)) {
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

pub const Adapter = struct {
    context: *Context,
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,
    query_state: ?*CanonicalQueryState,

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

        const goal = try self.formulaForObligation(target.kind);
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

    pub fn encodeFormula(self: *Adapter, formula: obligation.FormulaRef) EncodeError!z3.Z3_ast {
        const ast = switch (formula) {
            .term => |term_id| try self.encodeTermId(term_id),
            .origin_value => return error.UnsupportedOriginValue,
        };
        try self.requireBool(ast);
        return ast;
    }

    fn encodeTermId(self: *Adapter, id: obligation.TermId) EncodeError!z3.Z3_ast {
        if (id >= self.set.terms.len) return error.InvalidTermReference;
        return self.encodeTerm(self.set.terms[id]);
    }

    fn encodeTermIdWithExpected(
        self: *Adapter,
        id: obligation.TermId,
        expected: ?TypeInfo,
        enforce_expected_type: bool,
    ) EncodeError!z3.Z3_ast {
        if (id >= self.set.terms.len) return error.InvalidTermReference;
        if (self.set.terms[id] == .result) {
            const info = expected orelse return error.UnsupportedResultTerm;
            if (info.kind != .bitvector) return error.UnsupportedResultTerm;
            return self.encodeResult(info);
        }
        if (enforce_expected_type) {
            const expected_info = expected orelse return error.UnsupportedResultTerm;
            const actual = try staticTermTypeInfo(self.set, id);
            if (!typeInfoEql(expected_info, actual)) return error.TypeMismatch;
        }
        return self.encodeTerm(self.set.terms[id]);
    }

    fn encodeTerm(self: *Adapter, term: obligation.Term) EncodeError!z3.Z3_ast {
        return switch (term) {
            .bool_lit => |value| if (value)
                z3.Z3_mk_true(self.context.ctx)
            else
                z3.Z3_mk_false(self.context.ctx),
            .int_lit => |literal| self.encodeIntegerLiteral(literal),
            .variable => |variable| self.encodeVariable(variable),
            .old => |operand| self.encodeOld(operand),
            .result => error.UnsupportedResultTerm,
            .place_read => |place| self.encodePlaceRead(place),
            .unary => |unary| self.encodeUnary(unary),
            .binary => |binary| self.encodeBinary(binary),
            .refinement_predicate => |predicate| self.encodeRefinementPredicate(predicate),
            .quantified => error.UnsupportedQuantifiedTerm,
        };
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
        const root_state = try self.canonicalQueryState().getOrPutPlaceRoot(self.allocator, place.root);
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

        const root_state = try self.canonicalQueryState().getOrPutPlaceRoot(self.allocator, place.root);
        const entry = root_state.entry orelse blk: {
            const created = root_state.current orelse .entry;
            root_state.entry = created;
            break :blk created;
        };
        const old_ast = try self.encodeOldPlaceSymbol(place.root);
        const entry_ast = try self.encodePlaceSymbol(place.root, entry);
        const linkage = z3.Z3_mk_eq(self.context.ctx, old_ast, entry_ast);
        try self.context.checkNoError();
        try self.canonicalQueryState().side_constraints.append(self.allocator, linkage);
        return old_ast;
    }

    fn appendSideConstraints(
        self: *Adapter,
        constraints: *std.ArrayList(z3.Z3_ast),
    ) EncodeError!void {
        const state = self.canonicalQueryState();
        try constraints.appendSlice(self.allocator, state.side_constraints.items);
        state.side_constraints.clearRetainingCapacity();
    }

    fn canonicalQueryState(self: *Adapter) *CanonicalQueryState {
        return self.query_state orelse unreachable;
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
            .bound => return error.UnsupportedBoundVariableTerm,
        };
        const ty = free.ty orelse return error.MissingType;
        const sort = try self.sortForType(try typeInfo(ty));
        const name_text = try std.fmt.allocPrint(self.allocator, "file{d}::{s}#pattern{d}", .{
            free.id.file_id,
            free.name,
            free.id.pattern_id,
        });
        defer self.allocator.free(name_text);
        const name = try self.allocator.dupeZ(u8, name_text);
        defer self.allocator.free(name);

        const symbol = z3.Z3_mk_string_symbol(self.context.ctx, name.ptr);
        const ast = z3.Z3_mk_const(self.context.ctx, symbol, sort);
        try self.context.checkNoError();
        return ast;
    }

    fn encodeUnary(self: *Adapter, unary: obligation.UnaryTerm) EncodeError!z3.Z3_ast {
        const operand = try self.encodeTermId(unary.operand);
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

    fn encodeBinary(self: *Adapter, binary: obligation.BinaryTerm) EncodeError!z3.Z3_ast {
        const expected = try expectedInfoForBinaryOperands(self.set, binary);
        const enforce_expected_type = expected != null and
            (termContainsResult(self.set, binary.lhs, self.set.terms.len + 1) or
                termContainsResult(self.set, binary.rhs, self.set.terms.len + 1));
        const lhs = try self.encodeTermIdWithExpected(binary.lhs, expected, enforce_expected_type);
        const rhs = try self.encodeTermIdWithExpected(binary.rhs, expected, enforce_expected_type);

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
    ) EncodeError!z3.Z3_ast {
        const value = try self.encodeTermId(predicate.value);
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
                const bound = try self.encodeTermId(predicate.args[0]);
                break :blk self.compareWithSignedness(.ge, value, bound, value_info.signed);
            },
            .max_value => blk: {
                try expectArgCount(predicate, 1);
                const bound = try self.encodeTermId(predicate.args[0]);
                break :blk self.compareWithSignedness(.le, value, bound, value_info.signed);
            },
            .in_range => blk: {
                try expectArgCount(predicate, 2);
                const lower = try self.encodeTermId(predicate.args[0]);
                const upper = try self.encodeTermId(predicate.args[1]);
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

fn variableTypeRef(variable: obligation.VarRef) ?obligation.TypeRef {
    return switch (variable) {
        .free => |free| free.ty,
        .bound => |bound| bound.ty,
    };
}

fn expectArgCount(predicate: obligation.RefinementPredicateTerm, expected: usize) EncodeError!void {
    if (predicate.args.len != expected) return error.InvalidRefinementArity;
}

const TypeKind = enum(u8) {
    bool,
    bitvector,
};

const TypeInfo = struct {
    kind: TypeKind,
    width: u32 = 0,
    signed: bool = false,
};

fn u256TypeInfo() TypeInfo {
    return .{ .kind = .bitvector, .width = 256, .signed = false };
}

fn typeInfo(ty: obligation.TypeRef) EncodeError!TypeInfo {
    return switch (ty) {
        .spelling => |spelling| typeInfoFromSpelling(spelling),
        .compiler_type_id => |id| typeInfoFromCompilerTypeId(id),
    };
}

fn typeInfoFromCompilerTypeId(id: u32) EncodeError!TypeInfo {
    const spec = type_builtin.lookupBuiltinByComptimeTypeId(id) orelse return error.UnsupportedCompilerTypeId;
    return switch (spec.category) {
        .Bool => .{ .kind = .bool },
        .Address => .{ .kind = .bitvector, .width = 160, .signed = false },
        .Integer => blk: {
            const info = type_builtin.integerInfoByComptimeTypeId(id) orelse return error.UnsupportedCompilerTypeId;
            break :blk .{
                .kind = .bitvector,
                .width = info.width,
                .signed = info.signed,
            };
        },
        else => error.UnsupportedCompilerTypeId,
    };
}

fn typeInfoFromSpelling(raw: []const u8) EncodeError!TypeInfo {
    const spelling = std.mem.trim(u8, raw, " \t\r\n");
    if (std.mem.eql(u8, spelling, "bool") or std.mem.eql(u8, spelling, "i1")) return .{ .kind = .bool };
    if (std.mem.eql(u8, spelling, "address") or std.mem.eql(u8, spelling, "!ora.address")) {
        return .{ .kind = .bitvector, .width = 160, .signed = false };
    }

    if (try parseSurfaceIntegerSpelling(spelling)) |info| return info;
    if (try parseOraIntegerSpelling(spelling)) |info| return info;
    return error.UnsupportedType;
}

fn parseSurfaceIntegerSpelling(spelling: []const u8) EncodeError!?TypeInfo {
    if (spelling.len < 2) return null;
    const signed = switch (spelling[0]) {
        'i' => true,
        'u' => false,
        else => return null,
    };
    for (spelling[1..]) |byte| {
        if (!std.ascii.isDigit(byte)) return null;
    }
    const width = try std.fmt.parseInt(u32, spelling[1..], 10);
    if (width == 0) return error.UnsupportedType;
    if (width == 1) return .{ .kind = .bool };
    return .{ .kind = .bitvector, .width = width, .signed = signed };
}

fn parseOraIntegerSpelling(spelling: []const u8) EncodeError!?TypeInfo {
    const prefix = "!ora.int<";
    if (!std.mem.startsWith(u8, spelling, prefix) or !std.mem.endsWith(u8, spelling, ">")) return null;

    const body = spelling[prefix.len .. spelling.len - 1];
    const comma = std.mem.indexOfScalar(u8, body, ',') orelse return error.UnsupportedType;
    const width_text = std.mem.trim(u8, body[0..comma], " \t\r\n");
    const signed_text = std.mem.trim(u8, body[comma + 1 ..], " \t\r\n");
    const width = try std.fmt.parseInt(u32, width_text, 10);
    if (width == 0) return error.UnsupportedType;
    if (width == 1) return .{ .kind = .bool };

    const signed = if (std.mem.eql(u8, signed_text, "true"))
        true
    else if (std.mem.eql(u8, signed_text, "false"))
        false
    else
        return error.UnsupportedType;

    return .{ .kind = .bitvector, .width = width, .signed = signed };
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

test "canonical Z3 required promotion table covers every shape exactly once" {
    const shape_count = std.meta.fields(CanonicalPromotionShape).len;
    var seen: [shape_count]bool = .{false} ** shape_count;

    for (canonical_promotion_table) |row| {
        const index: usize = @intFromEnum(row.shape);
        try std.testing.expect(index < shape_count);
        try std.testing.expect(!seen[index]);
        seen[index] = true;
    }
    for (seen) |item| try std.testing.expect(item);
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

    const quantified_terms = [_]obligation.Term{
        .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = .{ .spelling = "u256" } } } },
        .{ .bool_lit = true },
        .{ .quantified = .{
            .quantifier = .forall,
            .binder = .{ .name = "i", .ty = .{ .spelling = "u256" } },
            .body = 1,
        } },
    };
    try expectCanonicalHashUnsupported(&quantified_terms, 2, .unsupported_quantified_term);
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
