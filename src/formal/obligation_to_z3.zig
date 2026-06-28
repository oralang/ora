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

pub const EncodeError = std.mem.Allocator.Error || error{
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
    UnknownObligation,
    UnsupportedCompilerTypeId,
    UnsupportedObligationKind,
    UnsupportedOldTerm,
    UnsupportedOriginValue,
    UnsupportedPlaceReadTerm,
    UnsupportedQuantifiedTerm,
    UnsupportedRefinement,
    UnsupportedResultTerm,
    UnsupportedType,
    Z3ApiError,
};

pub const Adapter = struct {
    context: *Context,
    allocator: std.mem.Allocator,
    set: obligation.ObligationSet,

    pub fn init(
        context: *Context,
        allocator: std.mem.Allocator,
        set: obligation.ObligationSet,
    ) Adapter {
        return .{
            .context = context,
            .allocator = allocator,
            .set = set,
        };
    }

    pub fn checkObligation(self: *Adapter, id: obligation.Id) EncodeError!CheckStatus {
        try self.set.validateTermReferences();

        const target = self.findObligation(id) orelse return error.UnknownObligation;
        const goal = try self.formulaForObligation(target.kind);

        var solver = try Solver.init(self.context, self.allocator);
        defer solver.deinit();

        for (self.set.assumptions) |assumption| {
            if (assumption.formula) |formula| {
                const ast = try self.encodeFormula(formula);
                try solver.assertChecked(ast);
            }
        }

        const negated_goal = z3.Z3_mk_not(self.context.ctx, goal);
        try self.context.checkNoError();
        try solver.assertChecked(negated_goal);

        return switch (try solver.checkChecked()) {
            z3.Z3_L_FALSE => .proved,
            z3.Z3_L_TRUE => .disproved,
            else => .unknown,
        };
    }

    fn findObligation(self: *const Adapter, id: obligation.Id) ?obligation.Obligation {
        for (self.set.obligations) |item| {
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

    fn encodeTerm(self: *Adapter, term: obligation.Term) EncodeError!z3.Z3_ast {
        return switch (term) {
            .bool_lit => |value| if (value)
                z3.Z3_mk_true(self.context.ctx)
            else
                z3.Z3_mk_false(self.context.ctx),
            .int_lit => |literal| self.encodeIntegerLiteral(literal),
            .variable => |variable| self.encodeVariable(variable),
            .old => error.UnsupportedOldTerm,
            .result => error.UnsupportedResultTerm,
            .place_read => error.UnsupportedPlaceReadTerm,
            .unary => |unary| self.encodeUnary(unary),
            .binary => |binary| self.encodeBinary(binary),
            .refinement_predicate => |predicate| self.encodeRefinementPredicate(predicate),
            .quantified => error.UnsupportedQuantifiedTerm,
        };
    }

    fn encodeIntegerLiteral(self: *Adapter, literal: obligation.IntegerLiteralTerm) EncodeError!z3.Z3_ast {
        const ty = literal.ty orelse return error.MissingType;
        const info = try typeInfo(ty);
        if (info.kind != .bitvector) return error.ExpectedBitVector;
        return self.integerLiteralForType(literal.value, info);
    }

    fn encodeVariable(self: *Adapter, variable: obligation.VarRef) EncodeError!z3.Z3_ast {
        const ty = variable.ty orelse return error.MissingType;
        const sort = try self.sortForType(try typeInfo(ty));
        const name = try self.allocator.dupeZ(u8, variable.name);
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
        const lhs = try self.encodeTermId(binary.lhs);
        const rhs = try self.encodeTermId(binary.rhs);

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
                break :blk if (try self.binaryOperandsSigned(binary))
                    z3.Z3_mk_bvsdiv(self.context.ctx, lhs, rhs)
                else
                    z3.Z3_mk_bv_udiv(self.context.ctx, lhs, rhs);
            },
            .mod => blk: {
                try self.requireBitVector(lhs);
                try self.requireBitVector(rhs);
                break :blk if (try self.binaryOperandsSigned(binary))
                    z3.Z3_mk_bvsrem(self.context.ctx, lhs, rhs)
                else
                    z3.Z3_mk_bv_urem(self.context.ctx, lhs, rhs);
            },
            .lt => try self.encodeComparison(binary, lhs, rhs, .lt),
            .le => try self.encodeComparison(binary, lhs, rhs, .le),
            .gt => try self.encodeComparison(binary, lhs, rhs, .gt),
            .ge => try self.encodeComparison(binary, lhs, rhs, .ge),
        };
        try self.context.checkNoError();
        return ast;
    }

    const Comparison = enum(u8) {
        lt,
        le,
        gt,
        ge,
    };

    fn encodeComparison(
        self: *Adapter,
        binary: obligation.BinaryTerm,
        lhs: z3.Z3_ast,
        rhs: z3.Z3_ast,
        comparison: Comparison,
    ) EncodeError!z3.Z3_ast {
        const signed = try self.binaryOperandsSigned(binary);
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

    fn binaryOperandsSigned(self: *Adapter, binary: obligation.BinaryTerm) EncodeError!bool {
        const lhs = try self.termTypeInfo(binary.lhs);
        const rhs = try self.termTypeInfo(binary.rhs);
        if (lhs.kind != .bitvector or rhs.kind != .bitvector) return error.ExpectedBitVector;
        if (lhs.width != rhs.width) return error.TypeMismatch;
        if (lhs.signed != rhs.signed) return error.TypeMismatch;
        return lhs.signed;
    }

    fn termTypeInfo(self: *Adapter, id: obligation.TermId) EncodeError!TypeInfo {
        if (id >= self.set.terms.len) return error.InvalidTermReference;
        return switch (self.set.terms[id]) {
            .bool_lit => .{ .kind = .bool },
            .int_lit => |literal| typeInfo(literal.ty orelse return error.MissingType),
            .variable => |variable| typeInfo(variable.ty orelse return error.MissingType),
            .old => |operand| self.termTypeInfo(operand),
            .unary => |unary| switch (unary.op) {
                .not => .{ .kind = .bool },
                .neg => self.termTypeInfo(unary.operand),
            },
            .binary => |binary| switch (binary.op) {
                .eq,
                .ne,
                .lt,
                .le,
                .gt,
                .ge,
                .and_,
                .or_,
                .implies,
                => .{ .kind = .bool },
                .add,
                .sub,
                .mul,
                .div,
                .mod,
                => self.termTypeInfo(binary.lhs),
            },
            .refinement_predicate => .{ .kind = .bool },
            .quantified => .{ .kind = .bool },
            .result,
            .place_read,
            => error.MissingType,
        };
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

fn typeInfo(ty: obligation.TypeRef) EncodeError!TypeInfo {
    return switch (ty) {
        .spelling => |spelling| typeInfoFromSpelling(spelling),
        .compiler_type_id => error.UnsupportedCompilerTypeId,
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
