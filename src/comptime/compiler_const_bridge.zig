const std = @import("std");
const ast = @import("../ast/mod.zig");
const comptime_eval = @import("mod.zig");
const error_mod = @import("error.zig");
const model = @import("../sema/model.zig");

const BigInt = std.math.big.int.Managed;
pub const ConstValue = model.ConstValue;
pub const CtEnv = comptime_eval.CtEnv;
pub const CtHeap = comptime_eval.CtHeap;
pub const CtValue = comptime_eval.CtValue;
const Evaluator = comptime_eval.Evaluator;
const EvalResult = comptime_eval.EvalResult;
const EvalMode = comptime_eval.EvalMode;
const TryEvalPolicy = comptime_eval.TryEvalPolicy;
const EvalBinaryOp = comptime_eval.BinaryOp;
const EvalUnaryOp = comptime_eval.UnaryOp;
const SourceSpan = error_mod.SourceSpan;

pub fn parseIntegerLiteral(allocator: std.mem.Allocator, text: []const u8) !?ConstValue {
    const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
    const digits = if (base == 10) text else text[2..];
    var value = BigInt.init(allocator) catch return null;
    value.setString(base, digits) catch return null;
    return .{ .integer = value };
}

pub fn evalUnary(allocator: std.mem.Allocator, op: ast.UnaryOp, value: ?ConstValue) !?ConstValue {
    if (value) |v| {
        if (try tryEvalUnaryWithSharedEngine(allocator, op, v)) |shared| return shared;
    }
    if (value) |v| {
        return switch (op) {
            .neg => switch (v) {
                .integer => |integer| .{ .integer = try negateInteger(allocator, integer) },
                else => null,
            },
            .not_ => switch (v) {
                .boolean => |boolean| .{ .boolean = !boolean },
                else => null,
            },
            .bit_not => switch (v) {
                .integer => |integer| .{ .integer = try bitwiseNotInteger(allocator, integer) },
                else => null,
            },
            .try_ => value,
        };
    }
    return null;
}

pub fn evalBinary(allocator: std.mem.Allocator, op: ast.BinaryOp, lhs: ?ConstValue, rhs: ?ConstValue) !?ConstValue {
    if (lhs == null or rhs == null) return null;
    const left = lhs.?;
    const right = rhs.?;
    if (try tryEvalBinaryWithSharedEngine(allocator, op, left, right)) |shared| return shared;
    return switch (op) {
        .add => switch (left) {
            .string => |a| switch (right) {
                .string => |b| .{ .string = try std.fmt.allocPrint(allocator, "{s}{s}", .{ a, b }) },
                else => null,
            },
            else => try evalIntInt(allocator, left, right, BigInt.add),
        },
        .wrapping_add => try evalIntInt(allocator, left, right, BigInt.add),
        .sub, .wrapping_sub => try evalIntInt(allocator, left, right, BigInt.sub),
        .mul, .wrapping_mul => try evalIntInt(allocator, left, right, BigInt.mul),
        .div => try evalIntDiv(allocator, left, right),
        .mod => try evalIntMod(allocator, left, right),
        .pow, .wrapping_pow => try evalIntPow(allocator, left, right),
        .eq => .{ .boolean = constEquals(left, right) },
        .ne => .{ .boolean = !constEquals(left, right) },
        .lt => evalCompare(left, right, .lt),
        .le => evalCompare(left, right, .lte),
        .gt => evalCompare(left, right, .gt),
        .ge => evalCompare(left, right, .gte),
        .and_and => evalBoolBool(left, right, struct {
            fn apply(a: bool, b: bool) bool {
                return a and b;
            }
        }.apply),
        .or_or => evalBoolBool(left, right, struct {
            fn apply(a: bool, b: bool) bool {
                return a or b;
            }
        }.apply),
        .bit_and => try evalIntInt(allocator, left, right, BigInt.bitAnd),
        .bit_or => try evalIntInt(allocator, left, right, BigInt.bitOr),
        .bit_xor => try evalIntInt(allocator, left, right, BigInt.bitXor),
        .shl, .wrapping_shl => try evalShift(allocator, left, right, true),
        .shr, .wrapping_shr => try evalShift(allocator, left, right, false),
    };
}

pub fn wrapIntegerConstToType(allocator: std.mem.Allocator, value: ConstValue, integer: model.IntegerType) !?ConstValue {
    return switch (value) {
        .integer => |integer_value| .{ .integer = try wrapIntegerToType(allocator, integer_value, integer) },
        else => null,
    };
}

pub fn wrapIntegerToType(allocator: std.mem.Allocator, value: BigInt, integer: model.IntegerType) !BigInt {
    const bits = integer.bits orelse return cloneInteger(allocator, value);
    const signed = integer.signed orelse return cloneInteger(allocator, value);
    if (bits == 0) return BigInt.initSet(allocator, 0);

    var modulus = try BigInt.initSet(allocator, 1);
    try BigInt.shiftLeft(&modulus, &modulus, bits);

    var quotient = try BigInt.init(allocator);
    var remainder = try BigInt.init(allocator);
    try BigInt.divTrunc(&quotient, &remainder, &value, &modulus);

    const zero = try BigInt.initSet(allocator, 0);
    if (remainder.order(zero).compare(.lt)) {
        var adjusted = try BigInt.init(allocator);
        try BigInt.add(&adjusted, &remainder, &modulus);
        remainder = adjusted;
    }

    if (!signed) return remainder;

    var sign_threshold = try BigInt.initSet(allocator, 1);
    try BigInt.shiftLeft(&sign_threshold, &sign_threshold, bits - 1);
    if (remainder.order(sign_threshold).compare(.gte)) {
        var adjusted = try BigInt.init(allocator);
        try BigInt.sub(&adjusted, &remainder, &modulus);
        return adjusted;
    }
    return remainder;
}

pub fn constEquals(lhs: ConstValue, rhs: ConstValue) bool {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| a.eql(b),
            else => false,
        },
        .boolean => |a| switch (rhs) {
            .boolean => |b| a == b,
            else => false,
        },
        .address => |a| switch (rhs) {
            .address => |b| a == b,
            else => false,
        },
        .string => |a| switch (rhs) {
            .string => |b| std.mem.eql(u8, a, b),
            else => false,
        },
        .tuple => |a| switch (rhs) {
            .tuple => |b| blk: {
                if (a.len != b.len) break :blk false;
                for (a, b) |lhs_elem, rhs_elem| {
                    if (!constEquals(lhs_elem, rhs_elem)) break :blk false;
                }
                break :blk true;
            },
            else => false,
        },
    };
}

fn negateInteger(allocator: std.mem.Allocator, value: BigInt) !BigInt {
    var zero = try BigInt.initSet(allocator, 0);
    var result = try BigInt.init(allocator);
    try BigInt.sub(&result, &zero, &value);
    return result;
}

fn cloneInteger(allocator: std.mem.Allocator, value: BigInt) !BigInt {
    const zero = try BigInt.initSet(allocator, 0);
    var result = try BigInt.init(allocator);
    try BigInt.add(&result, &value, &zero);
    return result;
}

fn bitwiseNotInteger(allocator: std.mem.Allocator, value: BigInt) !BigInt {
    var one = try BigInt.initSet(allocator, 1);
    defer one.deinit();
    var plus_one = try BigInt.init(allocator);
    defer plus_one.deinit();
    try BigInt.add(&plus_one, &value, &one);
    return negateInteger(allocator, plus_one);
}

fn evalIntInt(
    allocator: std.mem.Allocator,
    lhs: ConstValue,
    rhs: ConstValue,
    comptime op: fn (*BigInt, *const BigInt, *const BigInt) anyerror!void,
) !?ConstValue {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| blk: {
                var result = try BigInt.init(allocator);
                try op(&result, &a, &b);
                break :blk .{ .integer = result };
            },
            else => null,
        },
        else => null,
    };
}

fn evalIntDiv(allocator: std.mem.Allocator, lhs: ConstValue, rhs: ConstValue) !?ConstValue {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| blk: {
                if (b.eqlZero()) break :blk null;
                var quotient = try BigInt.init(allocator);
                var remainder = try BigInt.init(allocator);
                try BigInt.divTrunc(&quotient, &remainder, &a, &b);
                break :blk .{ .integer = quotient };
            },
            else => null,
        },
        else => null,
    };
}

fn evalIntMod(allocator: std.mem.Allocator, lhs: ConstValue, rhs: ConstValue) !?ConstValue {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| blk: {
                if (b.eqlZero()) break :blk null;
                var quotient = try BigInt.init(allocator);
                var remainder = try BigInt.init(allocator);
                try BigInt.divTrunc(&quotient, &remainder, &a, &b);
                break :blk .{ .integer = remainder };
            },
            else => null,
        },
        else => null,
    };
}

fn evalShift(allocator: std.mem.Allocator, lhs: ConstValue, rhs: ConstValue, comptime left_shift: bool) !?ConstValue {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| blk: {
                const amount = positiveShiftAmount(b) orelse break :blk null;
                var result = try BigInt.init(allocator);
                if (left_shift) {
                    try BigInt.shiftLeft(&result, &a, amount);
                } else {
                    try BigInt.shiftRight(&result, &a, amount);
                }
                break :blk .{ .integer = result };
            },
            else => null,
        },
        else => null,
    };
}

fn evalIntPow(allocator: std.mem.Allocator, lhs: ConstValue, rhs: ConstValue) !?ConstValue {
    return switch (lhs) {
        .integer => |base| switch (rhs) {
            .integer => |exp| blk: {
                const amount = positiveShiftAmount(exp) orelse break :blk null;
                var result = try BigInt.initSet(allocator, 1);
                var i: usize = 0;
                while (i < amount) : (i += 1) {
                    var next = try BigInt.init(allocator);
                    try BigInt.mul(&next, &result, &base);
                    result = next;
                }
                break :blk .{ .integer = result };
            },
            else => null,
        },
        else => null,
    };
}

fn evalCompare(lhs: ConstValue, rhs: ConstValue, op: std.math.CompareOperator) ?ConstValue {
    return switch (lhs) {
        .integer => |a| switch (rhs) {
            .integer => |b| .{ .boolean = a.order(b).compare(op) },
            else => null,
        },
        else => null,
    };
}

fn evalBoolBool(lhs: ConstValue, rhs: ConstValue, comptime op: fn (bool, bool) bool) ?ConstValue {
    return switch (lhs) {
        .boolean => |a| switch (rhs) {
            .boolean => |b| .{ .boolean = op(a, b) },
            else => null,
        },
        else => null,
    };
}

pub fn positiveShiftAmount(value: BigInt) ?usize {
    if (!value.isPositive() and !value.eqlZero()) return null;
    return value.toInt(usize) catch null;
}

fn tryEvalUnaryWithSharedEngine(allocator: std.mem.Allocator, op: ast.UnaryOp, value: ConstValue) !?ConstValue {
    switch (op) {
        .neg, .bit_not => switch (value) {
            .integer => return null,
            else => {},
        },
        else => {},
    }

    const eval_op = switch (op) {
        .neg => EvalUnaryOp.neg,
        .not_ => EvalUnaryOp.not,
        .bit_not => EvalUnaryOp.bnot,
        .try_ => return value,
    };

    const operand = try constToCtValue(value) orelse return null;
    var env = CtEnv.init(allocator, .{});
    defer env.deinit();
    var evaluator = Evaluator.init(&env, EvalMode.must_eval, TryEvalPolicy.strict);
    const result = evaluator.evalUnaryOp(eval_op, operand, zeroSpan());
    return try evalResultToConstValue(allocator, result);
}

fn tryEvalBinaryWithSharedEngine(allocator: std.mem.Allocator, op: ast.BinaryOp, lhs: ConstValue, rhs: ConstValue) !?ConstValue {
    const eval_op = switch (op) {
        .add => EvalBinaryOp.add,
        .sub => EvalBinaryOp.sub,
        .mul => EvalBinaryOp.mul,
        .pow => EvalBinaryOp.pow,
        .div => EvalBinaryOp.div,
        .mod => EvalBinaryOp.mod,
        .wrapping_add => EvalBinaryOp.wadd,
        .wrapping_sub => EvalBinaryOp.wsub,
        .wrapping_mul => EvalBinaryOp.wmul,
        .wrapping_pow => EvalBinaryOp.wpow,
        .eq => EvalBinaryOp.eq,
        .ne => EvalBinaryOp.neq,
        .lt => EvalBinaryOp.lt,
        .le => EvalBinaryOp.lte,
        .gt => EvalBinaryOp.gt,
        .ge => EvalBinaryOp.gte,
        .bit_and => EvalBinaryOp.band,
        .bit_or => EvalBinaryOp.bor,
        .bit_xor => EvalBinaryOp.bxor,
        .shl => EvalBinaryOp.shl,
        .shr => EvalBinaryOp.shr,
        .wrapping_shl => EvalBinaryOp.wshl,
        .wrapping_shr => EvalBinaryOp.wshr,
        .and_and => EvalBinaryOp.land,
        .or_or => EvalBinaryOp.lor,
    };

    const left = try constToCtValue(lhs) orelse return null;
    const right = try constToCtValue(rhs) orelse return null;

    var env = CtEnv.init(allocator, .{});
    defer env.deinit();
    var evaluator = Evaluator.init(&env, EvalMode.must_eval, TryEvalPolicy.strict);
    const result = evaluator.evalBinaryOp(eval_op, left, right, zeroSpan());
    return try evalResultToConstValue(allocator, result);
}

pub fn constToCtValue(value: ConstValue) !?CtValue {
    return switch (value) {
        .integer => |integer| blk: {
            if (!integer.isPositive() and !integer.eqlZero()) break :blk null;
            const as_u256 = integer.toInt(u256) catch break :blk null;
            break :blk CtValue{ .integer = as_u256 };
        },
        .boolean => |boolean| CtValue{ .boolean = boolean },
        .address => |address| CtValue{ .address = address },
        .string => null,
        .tuple => null,
    };
}

fn evalResultToConstValue(allocator: std.mem.Allocator, result: EvalResult) !?ConstValue {
    return switch (result) {
        .value => |value| switch (value) {
            .integer => |integer| .{ .integer = try BigInt.initSet(allocator, integer) },
            .boolean => |boolean| .{ .boolean = boolean },
            .address => |address| .{ .address = address },
            else => null,
        },
        .runtime, .control => null,
        .err => null,
    };
}

fn zeroSpan() SourceSpan {
    return .{ .line = 0, .column = 0, .length = 0 };
}

pub fn ctValueToConstValue(allocator: std.mem.Allocator, heap: ?*const CtHeap, value: CtValue) !?ConstValue {
    return switch (value) {
        .integer => |integer| .{ .integer = try BigInt.initSet(allocator, integer) },
        .boolean => |boolean| .{ .boolean = boolean },
        .address => |address| .{ .address = address },
        .string_ref => |heap_id| blk: {
            const actual_heap = heap orelse break :blk null;
            break :blk .{ .string = try allocator.dupe(u8, actual_heap.getString(heap_id)) };
        },
        .tuple_ref => |heap_id| blk: {
            const actual_heap = heap orelse break :blk null;
            const tuple = actual_heap.getTuple(heap_id);
            const elems = try allocator.alloc(ConstValue, tuple.elems.len);
            for (tuple.elems, 0..) |elem, index| {
                elems[index] = (try ctValueToConstValue(allocator, actual_heap, elem)) orelse break :blk null;
            }
            break :blk .{ .tuple = elems };
        },
        else => null,
    };
}
