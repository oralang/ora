const std = @import("std");
const ast = @import("../compiler/ast/mod.zig");
const model = @import("../compiler/sema/model.zig");

const BigInt = std.math.big.int.Managed;
pub const ConstValue = model.ConstValue;

pub fn parseIntegerLiteral(allocator: std.mem.Allocator, text: []const u8) !?ConstValue {
    const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
    const digits = if (base == 10) text else text[2..];
    var value = BigInt.init(allocator) catch return null;
    value.setString(base, digits) catch return null;
    return .{ .integer = value };
}

pub fn evalUnary(allocator: std.mem.Allocator, op: ast.UnaryOp, value: ?ConstValue) !?ConstValue {
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
    return switch (op) {
        .add, .wrapping_add => try evalIntInt(allocator, left, right, BigInt.add),
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
        .string => |a| switch (rhs) {
            .string => |b| std.mem.eql(u8, a, b),
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

fn positiveShiftAmount(value: BigInt) ?usize {
    if (!value.isPositive() and !value.eqlZero()) return null;
    return value.toInt(usize) catch null;
}
