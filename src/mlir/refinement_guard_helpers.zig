// ============================================================================
// Refinement Guard Helpers
// ============================================================================
//
// Shared guard emission and deduplication logic used by statement, expression,
// and declaration lowerers.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const h = @import("helpers.zig");
const lib = @import("ora_lib");
const OraDialect = @import("dialect.zig").OraDialect;
const LocationTracker = @import("locations.zig").LocationTracker;
const ParamMap = @import("symbols.zig").ParamMap;

pub fn emitRefinementGuard(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    guard_cache: ?*std.AutoHashMap(u128, void),
    span: lib.ast.SourceSpan,
    condition: c.MlirValue,
    message: []const u8,
    refinement_kind: []const u8,
    var_name: ?[]const u8,
    allocator: std.mem.Allocator,
) void {
    const name_slice = var_name orelse "";
    var hasher_hi = std.hash.Wyhash.init(0);
    var hasher_lo = std.hash.Wyhash.init(1);
    const span_is_unknown = span.line == 0 or span.column == 0 or span.length == 0;
    hasher_hi.update(locations.filename);
    hasher_hi.update(std.mem.asBytes(&span.line));
    hasher_hi.update(std.mem.asBytes(&span.column));
    hasher_hi.update(std.mem.asBytes(&span.length));
    hasher_hi.update(refinement_kind);
    hasher_hi.update(name_slice);
    hasher_lo.update(locations.filename);
    hasher_lo.update(std.mem.asBytes(&span.line));
    hasher_lo.update(std.mem.asBytes(&span.column));
    hasher_lo.update(std.mem.asBytes(&span.length));
    hasher_lo.update(refinement_kind);
    hasher_lo.update(name_slice);
    if (span_is_unknown) {
        const cond_key: usize = @intFromPtr(condition.ptr);
        hasher_hi.update(std.mem.asBytes(&cond_key));
        hasher_lo.update(std.mem.asBytes(&cond_key));
    }
    const key: u128 = (@as(u128, hasher_hi.final()) << 64) | hasher_lo.final();
    if (guard_cache) |cache| {
        if (cache.contains(key)) {
            return;
        }
        cache.put(key, {}) catch {};
    }

    const loc = c.oraLocationFileLineColGet(
        ctx,
        h.strRef(locations.filename),
        span.line,
        span.column,
    );
    const guard_op = ora_dialect.createRefinementGuard(condition, loc, message);
    h.appendOp(block, guard_op);

    const guard_id = if (var_name) |name|
        std.fmt.allocPrint(
            allocator,
            "guard:{s}:{d}:{d}:{d}:{s}:{s}",
            .{ locations.filename, span.line, span.column, span.length, refinement_kind, name },
        ) catch return
    else
        std.fmt.allocPrint(
            allocator,
            "guard:{s}:{d}:{d}:{d}:{s}",
            .{ locations.filename, span.line, span.column, span.length, refinement_kind },
        ) catch return;
    defer allocator.free(guard_id);

    c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(ctx, guard_id));
    c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(ctx, refinement_kind));
}

// ============================================================================
// Contract-Refinement Deduplication
// ============================================================================
//
// Detects when a requires/ensures clause is already covered by a refinement
// type guard. When covered, the clause is emitted as ora.assume (SMT-only)
// instead of cf.assert (runtime), avoiding duplicate runtime checks.

const ExprNode = lib.ast.Expressions.ExprNode;
const BinaryOp = lib.ast.Expressions.BinaryOp;
const OraType = lib.ast.Types.OraType;
const CastType = lib.ast.Expressions.CastType;

pub fn extractLiteralValue(expr: *const ExprNode) ?u256 {
    switch (expr.*) {
        .Literal => |lit| switch (lit) {
            .Integer => |int_lit| return std.fmt.parseInt(u256, int_lit.value, 10) catch null,
            .Hex => |hex_lit| {
                const raw = hex_lit.value;
                const trimmed = if (std.mem.startsWith(u8, raw, "0x") or std.mem.startsWith(u8, raw, "0X"))
                    raw[2..]
                else
                    raw;
                return std.fmt.parseInt(u256, trimmed, 16) catch null;
            },
            else => return null,
        },
        else => return null,
    }
}

fn strictIdentifier(expr: *const ExprNode) ?[]const u8 {
    return switch (expr.*) {
        .Identifier => |id| id.name,
        else => null,
    };
}

fn isSupportedComparisonOp(op: BinaryOp) bool {
    return switch (op) {
        .Greater, .GreaterEqual, .Less, .LessEqual, .BangEqual => true,
        else => false,
    };
}

fn isIdentitySelfCall(expr: *const ExprNode, current_function_name: []const u8, param_map: *const ParamMap) bool {
    switch (expr.*) {
        .Call => |call| {
            switch (call.callee.*) {
                .Identifier => |callee_id| {
                    if (!std.mem.eql(u8, callee_id.name, current_function_name)) return false;
                },
                else => return false,
            }
            if (call.arguments.len != param_map.names.count()) return false;
            for (call.arguments, 0..) |arg, i| {
                switch (arg.*) {
                    .Identifier => |id| {
                        const param_index = param_map.getParamIndex(id.name) orelse return false;
                        if (param_index != i) return false;
                    },
                    else => return false,
                }
            }
            return true;
        },
        else => return false,
    }
}

pub fn flipOp(op: BinaryOp) BinaryOp {
    return switch (op) {
        .Greater => .Less,
        .Less => .Greater,
        .GreaterEqual => .LessEqual,
        .LessEqual => .GreaterEqual,
        else => op,
    };
}

/// Check whether a refinement type logically implies `value <op> literal_value`.
pub fn refinementImplies(ref_type: OraType, op: BinaryOp, lit_val: u256) bool {
    switch (ref_type) {
        .min_value => |mv| {
            return switch (op) {
                .GreaterEqual => lit_val <= mv.min,
                .Greater => lit_val < mv.min,
                .BangEqual => lit_val < mv.min,
                else => false,
            };
        },
        .max_value => |mv| {
            return switch (op) {
                .LessEqual => lit_val >= mv.max,
                .Less => lit_val > mv.max,
                else => false,
            };
        },
        .in_range => |ir| {
            return switch (op) {
                .GreaterEqual => lit_val <= ir.min,
                .Greater => lit_val < ir.min,
                .LessEqual => lit_val >= ir.max,
                .Less => lit_val > ir.max,
                .BangEqual => lit_val < ir.min or lit_val > ir.max,
                else => false,
            };
        },
        .non_zero_address => {
            return switch (op) {
                .BangEqual => lit_val == 0,
                .Greater => lit_val == 0,
                .GreaterEqual => lit_val <= 1,
                else => false,
            };
        },
        else => return false,
    }
}

fn isZeroAddressHexString(value: []const u8) bool {
    var hex = value;
    if (std.mem.startsWith(u8, hex, "0x") or std.mem.startsWith(u8, hex, "0X")) {
        hex = hex[2..];
    }
    if (hex.len != 40) return false;
    for (hex) |ch| {
        if (ch != '0') return false;
    }
    return true;
}

fn isStdConstantsZeroAddress(expr: *const ExprNode) bool {
    switch (expr.*) {
        .FieldAccess => |fa| {
            if (!std.mem.eql(u8, fa.field, "ZERO_ADDRESS")) return false;
            switch (fa.target.*) {
                .FieldAccess => |inner| {
                    if (!std.mem.eql(u8, inner.field, "constants")) return false;
                    switch (inner.target.*) {
                        .Identifier => |id| return std.mem.eql(u8, id.name, "std"),
                        else => return false,
                    }
                },
                else => return false,
            }
        },
        else => return false,
    }
}

pub fn isZeroAddressExpr(expr: *const ExprNode) bool {
    if (isStdConstantsZeroAddress(expr)) return true;
    switch (expr.*) {
        .Literal => |lit| switch (lit) {
            .Address => |addr| return isZeroAddressHexString(addr.value),
            .Hex => |hex| return isZeroAddressHexString(hex.value),
            .Integer => |int_lit| return std.mem.eql(u8, int_lit.value, "0"),
            else => return false,
        },
        else => return false,
    }
}

/// Check if a requires clause is logically covered by a parameter's refinement type.
pub fn isRequiresCoveredByRefinement(
    clause: *const ExprNode,
    param_refinements: *const std.StringHashMap(OraType),
) bool {
    switch (clause.*) {
        .Binary => |bin| {
            if (!isSupportedComparisonOp(bin.operator)) return false;
            return matchBinaryAgainstRefinements(bin.lhs, bin.operator, bin.rhs, param_refinements, strictIdentifier);
        },
        else => return false,
    }
}

/// Check if an ensures clause is logically covered by the return type refinement.
/// In ensures context, dedup is allowed only for strict identity self-calls.
pub fn isEnsuresCoveredByReturnRefinement(
    clause: *const ExprNode,
    return_refinement: OraType,
    current_function_name: ?[]const u8,
    param_map: ?*const ParamMap,
) bool {
    const fn_name = current_function_name orelse return false;
    const pm = param_map orelse return false;
    switch (clause.*) {
        .Binary => |bin| {
            if (!isSupportedComparisonOp(bin.operator)) return false;
            if (isIdentitySelfCall(bin.lhs, fn_name, pm)) {
                if (extractLiteralValue(bin.rhs)) |lit_val| {
                    return refinementImplies(return_refinement, bin.operator, lit_val);
                }
                if (bin.operator == .BangEqual and isZeroAddressExpr(bin.rhs))
                    return return_refinement == .non_zero_address;
            }
            if (isIdentitySelfCall(bin.rhs, fn_name, pm)) {
                if (extractLiteralValue(bin.lhs)) |lit_val| {
                    return refinementImplies(return_refinement, flipOp(bin.operator), lit_val);
                }
                if (bin.operator == .BangEqual and isZeroAddressExpr(bin.lhs))
                    return return_refinement == .non_zero_address;
            }
            return false;
        },
        else => return false,
    }
}

fn matchBinaryAgainstRefinements(
    lhs: *const ExprNode,
    op: BinaryOp,
    rhs: *const ExprNode,
    param_refinements: *const std.StringHashMap(OraType),
    nameExtractor: fn (*const ExprNode) ?[]const u8,
) bool {
    if (nameExtractor(lhs)) |name| {
        if (param_refinements.get(name)) |ref_type| {
            if (extractLiteralValue(rhs)) |lit_val| {
                return refinementImplies(ref_type, op, lit_val);
            }
            if (op == .BangEqual and isZeroAddressExpr(rhs))
                return ref_type == .non_zero_address;
        }
    }
    if (nameExtractor(rhs)) |name| {
        if (param_refinements.get(name)) |ref_type| {
            if (extractLiteralValue(lhs)) |lit_val| {
                return refinementImplies(ref_type, flipOp(op), lit_val);
            }
            if (op == .BangEqual and isZeroAddressExpr(lhs))
                return ref_type == .non_zero_address;
        }
    }
    return false;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const TypeInfo = lib.ast.Types.TypeInfo;
const SourceSpan = lib.ast.SourceSpan;

const test_span = SourceSpan{ .line = 1, .column = 1, .length = 1 };
const test_type_info = TypeInfo{ .category = .Integer, .ora_type = null, .source = .unknown, .span = null };

fn makeIntLiteral(value: []const u8) ExprNode {
    return ExprNode{ .Literal = .{ .Integer = .{ .value = value, .type_info = test_type_info, .span = test_span } } };
}

fn makeIdentifier(name: []const u8) ExprNode {
    return ExprNode{ .Identifier = .{ .name = name, .type_info = test_type_info, .span = test_span } };
}

fn makeBinary(lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode) ExprNode {
    return ExprNode{ .Binary = .{ .lhs = lhs, .operator = op, .rhs = rhs, .type_info = test_type_info, .span = test_span } };
}

fn makeCall(callee: *ExprNode, args: []*ExprNode) ExprNode {
    return ExprNode{
        .Call = .{
            .callee = callee,
            .arguments = args,
            .type_info = test_type_info,
            .span = test_span,
        },
    };
}

fn makeUnsafeCast(operand: *ExprNode) ExprNode {
    return ExprNode{
        .Cast = .{
            .operand = operand,
            .target_type = test_type_info,
            .cast_type = CastType.Unsafe,
            .span = test_span,
        },
    };
}

test "refinementImplies: MinValue covers GreaterEqual" {
    const base = OraType{ .u256 = {} };
    const mv = OraType{ .min_value = .{ .base = &base, .min = 10 } };
    try testing.expect(refinementImplies(mv, .GreaterEqual, 10));
    try testing.expect(refinementImplies(mv, .GreaterEqual, 5));
    try testing.expect(!refinementImplies(mv, .GreaterEqual, 11));
}

test "refinementImplies: MinValue covers Greater" {
    const base = OraType{ .u256 = {} };
    const mv = OraType{ .min_value = .{ .base = &base, .min = 10 } };
    try testing.expect(refinementImplies(mv, .Greater, 9));
    try testing.expect(refinementImplies(mv, .Greater, 0));
    try testing.expect(!refinementImplies(mv, .Greater, 10));
}

test "refinementImplies: MaxValue covers LessEqual" {
    const base = OraType{ .u256 = {} };
    const mv = OraType{ .max_value = .{ .base = &base, .max = 100 } };
    try testing.expect(refinementImplies(mv, .LessEqual, 100));
    try testing.expect(refinementImplies(mv, .LessEqual, 200));
    try testing.expect(!refinementImplies(mv, .LessEqual, 99));
}

test "refinementImplies: InRange covers both bounds" {
    const base = OraType{ .u256 = {} };
    const ir = OraType{ .in_range = .{ .base = &base, .min = 5, .max = 50 } };
    try testing.expect(refinementImplies(ir, .GreaterEqual, 5));
    try testing.expect(refinementImplies(ir, .GreaterEqual, 3));
    try testing.expect(refinementImplies(ir, .LessEqual, 50));
    try testing.expect(refinementImplies(ir, .LessEqual, 100));
    try testing.expect(!refinementImplies(ir, .GreaterEqual, 6));
    try testing.expect(!refinementImplies(ir, .LessEqual, 49));
}

test "refinementImplies: NonZeroAddress covers BangEqual 0" {
    const nza = OraType{ .non_zero_address = {} };
    try testing.expect(refinementImplies(nza, .BangEqual, 0));
    try testing.expect(!refinementImplies(nza, .BangEqual, 1));
    try testing.expect(refinementImplies(nza, .Greater, 0));
}

test "flipOp: symmetry" {
    try testing.expect(flipOp(.Greater) == .Less);
    try testing.expect(flipOp(.Less) == .Greater);
    try testing.expect(flipOp(.GreaterEqual) == .LessEqual);
    try testing.expect(flipOp(.LessEqual) == .GreaterEqual);
    try testing.expect(flipOp(.BangEqual) == .BangEqual);
}

test "isRequiresCoveredByRefinement: param >= N with MinValue" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("amount", OraType{ .min_value = .{ .base = &base, .min = 1 } });

    var id_node = makeIdentifier("amount");
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&id_node, .Greater, &lit_node);
    try testing.expect(isRequiresCoveredByRefinement(&clause, &map));
}

test "isRequiresCoveredByRefinement: uncovered param" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("amount", OraType{ .min_value = .{ .base = &base, .min = 1 } });

    // requires(other > 0) -- "other" not in refinement map
    var id_node = makeIdentifier("other");
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&id_node, .Greater, &lit_node);
    try testing.expect(!isRequiresCoveredByRefinement(&clause, &map));
}

test "isRequiresCoveredByRefinement: flipped operands" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("x", OraType{ .max_value = .{ .base = &base, .max = 100 } });

    // requires(100 >= x) is equivalent to requires(x <= 100)
    var lit_node = makeIntLiteral("100");
    var id_node = makeIdentifier("x");
    var clause = makeBinary(&lit_node, .GreaterEqual, &id_node);
    try testing.expect(isRequiresCoveredByRefinement(&clause, &map));
}

test "isRequiresCoveredByRefinement: not covered when bound is tighter" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("amount", OraType{ .min_value = .{ .base = &base, .min = 5 } });

    // requires(amount >= 10) -- refinement only guarantees >= 5, not >= 10
    var id_node = makeIdentifier("amount");
    var lit_node = makeIntLiteral("10");
    var clause = makeBinary(&id_node, .GreaterEqual, &lit_node);
    try testing.expect(!isRequiresCoveredByRefinement(&clause, &map));
}

test "isRequiresCoveredByRefinement: call expression is not deduplicated" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("amount", OraType{ .min_value = .{ .base = &base, .min = 1 } });

    var arg = makeIdentifier("amount");
    var callee = makeIdentifier("f");
    var args = [_]*ExprNode{&arg};
    var call_node = makeCall(&callee, args[0..]);
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&call_node, .Greater, &lit_node);
    try testing.expect(!isRequiresCoveredByRefinement(&clause, &map));
}

test "isRequiresCoveredByRefinement: cast expression is not deduplicated" {
    const base = OraType{ .u256 = {} };
    var map = std.StringHashMap(OraType).init(testing.allocator);
    defer map.deinit();
    try map.put("amount", OraType{ .min_value = .{ .base = &base, .min = 1 } });

    var id_node = makeIdentifier("amount");
    var cast_node = makeUnsafeCast(&id_node);
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&cast_node, .Greater, &lit_node);
    try testing.expect(!isRequiresCoveredByRefinement(&clause, &map));
}

test "isEnsuresCoveredByReturnRefinement: identity self-call is deduplicated" {
    const base = OraType{ .u256 = {} };
    const ret_ref = OraType{ .min_value = .{ .base = &base, .min = 1 } };

    var param_map = ParamMap.init(testing.allocator);
    defer param_map.deinit();
    try param_map.addParam("amount", 0);

    var arg = makeIdentifier("amount");
    var callee = makeIdentifier("deposit");
    var args = [_]*ExprNode{&arg};
    var self_call = makeCall(&callee, args[0..]);
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&self_call, .Greater, &lit_node);
    try testing.expect(isEnsuresCoveredByReturnRefinement(&clause, ret_ref, "deposit", &param_map));
}

test "isEnsuresCoveredByReturnRefinement: non-identity call is not deduplicated" {
    const base = OraType{ .u256 = {} };
    const ret_ref = OraType{ .min_value = .{ .base = &base, .min = 1 } };

    var param_map = ParamMap.init(testing.allocator);
    defer param_map.deinit();
    try param_map.addParam("amount", 0);

    var arg = makeIdentifier("amount");
    var callee = makeIdentifier("otherFn");
    var args = [_]*ExprNode{&arg};
    var other_call = makeCall(&callee, args[0..]);
    var lit_node = makeIntLiteral("0");
    var clause = makeBinary(&other_call, .Greater, &lit_node);
    try testing.expect(!isEnsuresCoveredByReturnRefinement(&clause, ret_ref, "deposit", &param_map));
}
