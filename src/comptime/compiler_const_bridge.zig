const std = @import("std");
const ast = @import("../ast/mod.zig");
const comptime_eval = @import("mod.zig");
const ora_types = @import("ora_types");
const units = @import("../units.zig");

const BigInt = std.math.big.int.Managed;
const ConstValue = ora_types.ConstValue;
pub const CtEnv = comptime_eval.CtEnv;
pub const CtHeap = comptime_eval.CtHeap;
pub const CtValue = comptime_eval.CtValue;

pub fn parseIntegerLiteral(allocator: std.mem.Allocator, text: []const u8) !?ConstValue {
    const parsed = (try parseIntegerLiteralParts(allocator, text)) orelse return null;
    defer allocator.free(parsed.digits);

    const base: u8 = if (std.mem.startsWith(u8, parsed.digits, "0x")) 16 else if (std.mem.startsWith(u8, parsed.digits, "0b")) 2 else 10;
    if (parsed.unit_factor != 1 and base != 10) return null;
    const digits = if (base == 10) parsed.digits else parsed.digits[2..];
    var value = BigInt.init(allocator) catch return null;
    value.setString(base, digits) catch return null;
    if (parsed.unit_factor != 1) {
        var factor = try BigInt.initSet(allocator, parsed.unit_factor);
        var scaled = try BigInt.init(allocator);
        try BigInt.mul(&scaled, &value, &factor);
        value = scaled;
    }
    return .{ .integer = value };
}

const IntegerLiteralParts = struct {
    digits: []const u8,
    unit_factor: u64,
};

fn parseIntegerLiteralParts(allocator: std.mem.Allocator, text: []const u8) !?IntegerLiteralParts {
    var parts = std.mem.tokenizeAny(u8, std.mem.trim(u8, text, " \t\n\r"), " \t\n\r");
    const digits_text = parts.next() orelse "";
    const unit_text = parts.next();
    if (parts.next() != null) return null;

    const normalized = try removeUnderscores(allocator, digits_text);
    errdefer allocator.free(normalized);
    return .{
        .digits = normalized,
        .unit_factor = if (unit_text) |unit| units.etherUnitFactor(unit) orelse return null else 1,
    };
}

fn removeUnderscores(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    for (text) |c| {
        if (c == '_') continue;
        try out.append(allocator, c);
    }
    return out.toOwnedSlice(allocator);
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
        .bit_xor => switch (left) {
            .fixed_bytes => try evalFixedBytesFixedBytes(allocator, left, right, xorByte),
            else => try evalIntInt(allocator, left, right, BigInt.bitXor),
        },
        .shl, .wrapping_shl => try evalShift(allocator, left, right, true),
        .shr, .wrapping_shr => try evalShift(allocator, left, right, false),
    };
}

fn xorByte(a: u8, b: u8) u8 {
    return a ^ b;
}

fn evalFixedBytesFixedBytes(
    allocator: std.mem.Allocator,
    lhs: ConstValue,
    rhs: ConstValue,
    comptime op: fn (u8, u8) u8,
) !?ConstValue {
    const left = switch (lhs) {
        .fixed_bytes => |bytes| bytes,
        else => return null,
    };
    const right = switch (rhs) {
        .fixed_bytes => |bytes| bytes,
        else => return null,
    };
    if (left.len != right.len) return null;
    const out = try allocator.alloc(u8, left.len);
    for (left, right, 0..) |a, b, index| {
        out[index] = op(a, b);
    }
    return .{ .fixed_bytes = out };
}

pub fn wrapIntegerConstToType(allocator: std.mem.Allocator, value: ConstValue, integer: ora_types.IntegerType) !?ConstValue {
    return switch (value) {
        .integer => |integer_value| .{ .integer = try wrapIntegerToType(allocator, integer_value, integer) },
        else => null,
    };
}

pub fn wrapIntegerToType(allocator: std.mem.Allocator, value: BigInt, integer: ora_types.IntegerType) !BigInt {
    const bits = integer.bits;
    const signed = integer.signed;
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
        .fixed_bytes => |a| switch (rhs) {
            .fixed_bytes => |b| std.mem.eql(u8, a, b),
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

pub fn constToCtValue(value: ConstValue) !?CtValue {
    return switch (value) {
        .integer => |integer| blk: {
            if (!integer.isPositive() and !integer.eqlZero()) break :blk null;
            const as_u256 = integer.toInt(u256) catch break :blk null;
            break :blk CtValue{ .integer = as_u256 };
        },
        .boolean => |boolean| CtValue{ .boolean = boolean },
        .address => |address| CtValue{ .address = address },
        .fixed_bytes => null,
        .string => null,
        .tuple => null,
    };
}

pub fn ctValueToConstValue(allocator: std.mem.Allocator, heap: ?*const CtHeap, value: CtValue) !?ConstValue {
    return switch (value) {
        .integer => |integer| .{ .integer = try BigInt.initSet(allocator, integer) },
        .boolean => |boolean| .{ .boolean = boolean },
        .address => |address| .{ .address = address },
        .bytes_ref => |heap_id| blk: {
            const actual_heap = heap orelse break :blk null;
            break :blk .{ .fixed_bytes = try allocator.dupe(u8, actual_heap.getBytes(heap_id)) };
        },
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
