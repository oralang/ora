//! Shared canonical Z3 adapter types and type-resolution helpers.

const std = @import("std");
const z3_verification = @import("ora_z3_verification");
const z3 = z3_verification.z3_c;
const obligation = @import("obligation.zig");
const type_builtin = @import("ora_types").builtin;

pub const RefinementBuiltin = enum(u8) {
    min_value,
    max_value,
    in_range,
    non_zero_address,
    non_zero,
    basis_points,
};

pub const refinement_builtin_map = std.StaticStringMap(RefinementBuiltin).initComptime(.{
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
    unsupported_quantifier_binder_type,
    unsupported_function_param_wrapper_condition,
    unsupported_refinement,
    invalid_refinement_arity,
    missing_type,
    unsupported_type,
    unsupported_compiler_type_id,
    bound_variable_out_of_scope,
    unsupported_bound_variable_type,
};

pub const CanonicalPromotionShape = enum(u8) {
    core_formula,
    result_term,
    scalar_place_read,
    old_scalar_place_read,
    formula_combination,
    quantified_formula,
};

pub const CanonicalPromotionPolicy = struct {
    shape: CanonicalPromotionShape,
    required_mode: bool,
    rollout_enabled: bool,
};

// Required mode is currently three-key: the query row must explicitly request
// canonical SMT crosscheck, the live prepared row must attest that it is
// annotation-pure, and its syntax must be present in this table. The table is
// the audit surface for which shapes may block; rollout_enabled is the
// compiler-side policy for which shapes set the row flag today.
pub const canonical_promotion_table = [_]CanonicalPromotionPolicy{
    .{ .shape = .core_formula, .required_mode = true, .rollout_enabled = true },
    .{ .shape = .result_term, .required_mode = true, .rollout_enabled = false },
    .{ .shape = .scalar_place_read, .required_mode = true, .rollout_enabled = false },
    .{ .shape = .old_scalar_place_read, .required_mode = true, .rollout_enabled = false },
    // Full-corpus measurement showed formula combinations often lack live rows;
    // promoted mismatches still need diagnosis before this shape can block.
    .{ .shape = .formula_combination, .required_mode = false, .rollout_enabled = false },
    // Quantifiers are solver-option and trigger sensitive. They stay
    // diagnostic-only until corpus measurement justifies arming them.
    .{ .shape = .quantified_formula, .required_mode = false, .rollout_enabled = false },
};

pub fn canonicalPromotionShapeRequired(shape: CanonicalPromotionShape) bool {
    const policy = canonicalPromotionPolicy(shape) orelse return false;
    return policy.required_mode;
}

pub fn canonicalPromotionPolicy(shape: CanonicalPromotionShape) ?CanonicalPromotionPolicy {
    for (canonical_promotion_table) |row| {
        if (row.shape == shape) return row;
    }
    return null;
}

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
    UnsupportedQuantifierBinderType,
    UnsupportedFunctionParamWrapperCondition,
    UnsupportedRefinement,
    UnsupportedResultTerm,
    UnsupportedType,
    BoundVariableOutOfScope,
    MissingCanonicalQueryState,
    UnsupportedBoundVariableType,
    Z3ApiError,
};

pub const CanonicalPlaceSymbolKind = enum(u8) {
    global,
    entry,
};

pub const CanonicalPlaceRootState = struct {
    root: []const u8,
    current: ?CanonicalPlaceSymbolKind = null,
    entry: ?CanonicalPlaceSymbolKind = null,
};

pub const CanonicalBoundBinding = struct {
    name: []const u8,
    info: TypeInfo,
    ast: z3.Z3_ast,
    origin: obligation.BinderOrigin,
};

pub const CanonicalQuerySnapshot = struct {
    place_roots_len: usize,
    side_constraints_len: usize,
    bound_stack_len: usize,
};

pub const CanonicalQueryState = struct {
    place_roots: std.ArrayList(CanonicalPlaceRootState) = .empty,
    side_constraints: std.ArrayList(z3.Z3_ast) = .empty,
    bound_stack: std.ArrayList(CanonicalBoundBinding) = .empty,

    pub fn deinit(self: *CanonicalQueryState, allocator: std.mem.Allocator) void {
        self.place_roots.deinit(allocator);
        self.side_constraints.deinit(allocator);
        self.bound_stack.deinit(allocator);
    }

    pub fn getOrPutPlaceRoot(self: *CanonicalQueryState, allocator: std.mem.Allocator, root: []const u8) !*CanonicalPlaceRootState {
        for (self.place_roots.items) |*item| {
            if (std.mem.eql(u8, item.root, root)) return item;
        }
        try self.place_roots.append(allocator, .{ .root = root });
        return &self.place_roots.items[self.place_roots.items.len - 1];
    }

    pub fn pushBound(self: *CanonicalQueryState, allocator: std.mem.Allocator, binding: CanonicalBoundBinding) !void {
        try self.bound_stack.append(allocator, binding);
    }

    pub fn popBound(self: *CanonicalQueryState) void {
        _ = self.bound_stack.pop();
    }

    pub fn lookupBound(self: *const CanonicalQueryState, index: usize) ?CanonicalBoundBinding {
        if (index >= self.bound_stack.items.len) return null;
        return self.bound_stack.items[self.bound_stack.items.len - 1 - index];
    }

    pub fn snapshot(self: *const CanonicalQueryState) CanonicalQuerySnapshot {
        return .{
            .place_roots_len = self.place_roots.items.len,
            .side_constraints_len = self.side_constraints.items.len,
            .bound_stack_len = self.bound_stack.items.len,
        };
    }

    pub fn restore(self: *CanonicalQueryState, snapshot_value: CanonicalQuerySnapshot) void {
        self.place_roots.shrinkRetainingCapacity(snapshot_value.place_roots_len);
        self.side_constraints.shrinkRetainingCapacity(snapshot_value.side_constraints_len);
        self.bound_stack.shrinkRetainingCapacity(snapshot_value.bound_stack_len);
    }
};

pub const CanonicalSupportScope = struct {
    bound_depth: usize = 0,
    function_param_mask: u64 = 0,

    pub fn push(self: CanonicalSupportScope, origin: obligation.BinderOrigin) CanonicalSupportScope {
        var next = self;
        if (origin == .function_param and self.bound_depth < 64) {
            next.function_param_mask |= (@as(u64, 1) << @intCast(self.bound_depth));
        }
        next.bound_depth += 1;
        return next;
    }

    pub fn containsBound(self: CanonicalSupportScope, index: usize) bool {
        return index < self.bound_depth;
    }

    pub fn boundOrigin(self: CanonicalSupportScope, index: usize) ?obligation.BinderOrigin {
        if (!self.containsBound(index)) return null;
        const absolute_index = self.bound_depth - 1 - index;
        if (absolute_index >= 64) return .user;
        return if ((self.function_param_mask & (@as(u64, 1) << @intCast(absolute_index))) != 0)
            .function_param
        else
            .user;
    }
};

pub fn variableTypeRef(variable: obligation.VarRef) ?obligation.TypeRef {
    return switch (variable) {
        .free => |free| free.ty,
        .bound => |bound| bound.ty,
    };
}

pub fn expectArgCount(predicate: obligation.RefinementPredicateTerm, expected: usize) EncodeError!void {
    if (predicate.args.len != expected) return error.InvalidRefinementArity;
}

pub const TypeKind = enum(u8) {
    bool,
    bitvector,
};

pub const TypeInfo = struct {
    kind: TypeKind,
    width: u32 = 0,
    signed: bool = false,
};

pub fn u256TypeInfo() TypeInfo {
    return .{ .kind = .bitvector, .width = 256, .signed = false };
}

pub fn typeInfoIsU256(info: TypeInfo) bool {
    return info.kind == .bitvector and info.width == 256 and !info.signed;
}

pub fn quantifierBinderTypeInfo(maybe_ty: ?obligation.TypeRef) EncodeError!TypeInfo {
    const ty = maybe_ty orelse return error.MissingType;
    const info = try typeInfo(ty);
    if (!typeInfoIsU256(info)) return error.UnsupportedQuantifierBinderType;
    return info;
}

pub fn typeInfo(ty: obligation.TypeRef) EncodeError!TypeInfo {
    return switch (ty) {
        .spelling => |spelling| typeInfoFromSpelling(spelling),
        .compiler_type_id => |id| typeInfoFromCompilerTypeId(id),
    };
}

pub fn typeInfoFromCompilerTypeId(id: u32) EncodeError!TypeInfo {
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

pub fn typeInfoFromSpelling(raw: []const u8) EncodeError!TypeInfo {
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
