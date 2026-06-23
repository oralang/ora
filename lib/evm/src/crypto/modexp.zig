const std = @import("std");

pub const Error = error{
    DivisionByZero,
    InvalidInput,
    AllocationFailed,
    NotImplemented,
    OutOfMemory,
    InvalidBase,
    InvalidCharacter,
    NoSpaceLeft,
    InvalidLength,
} || std.mem.Allocator.Error;

const GAS_QUADRATIC_THRESHOLD: usize = 64;
const GAS_LINEAR_THRESHOLD: usize = 1024;

fn isZero(bytes: []const u8) bool {
    var acc: u8 = 0;
    for (bytes) |byte| acc |= byte;
    return acc == 0;
}

fn isOne(bytes: []const u8) bool {
    if (bytes.len == 0) return false;
    for (bytes[0 .. bytes.len - 1]) |byte| {
        if (byte != 0) return false;
    }
    return bytes[bytes.len - 1] == 1;
}

fn bytesToU64(bytes: []const u8) u64 {
    var result: u64 = 0;
    for (bytes) |byte| {
        result = (result << 8) | byte;
    }
    return result;
}

fn calculateMultiplicationComplexity(x: usize) u64 {
    const x64: u64 = @intCast(x);
    if (x <= GAS_QUADRATIC_THRESHOLD) {
        return x64 * x64;
    }
    if (x <= GAS_LINEAR_THRESHOLD) {
        return (x64 * x64) / 4 + 96 * x64 - 3072;
    }
    return (x64 * x64) / 16 + 480 * x64 - 199680;
}

fn calculateAdjustedExponentLength(exp_len: usize, exp_bytes: []const u8) u64 {
    if (exp_len == 0) return 0;

    var leading_zeros: usize = 0;
    for (exp_bytes) |byte| {
        if (byte != 0) break;
        leading_zeros += 1;
    }

    if (leading_zeros == exp_bytes.len) return 0;

    const first_non_zero = exp_bytes[leading_zeros];
    const bit_length = 8 - @clz(first_non_zero);
    return @intCast((exp_len - leading_zeros - 1) * 8 + bit_length);
}

pub fn calculateGas(base_len: usize, exp_len: usize, mod_len: usize, exp_bytes: []const u8) u64 {
    const mult_complexity = calculateMultiplicationComplexity(@max(base_len, mod_len));
    const adjusted_exp_len = calculateAdjustedExponentLength(exp_len, exp_bytes);
    const iteration_count = @max(adjusted_exp_len, 1);
    return @max(200, (mult_complexity * iteration_count) / 3);
}

pub fn modexp(allocator: std.mem.Allocator, base: []const u8, exponent: []const u8, modulus: []const u8) Error![]u8 {
    if (modulus.len == 0 or isZero(modulus)) return error.DivisionByZero;

    const output = try allocator.alloc(u8, modulus.len);
    errdefer allocator.free(output);
    @memset(output, 0);

    if (exponent.len == 0 or isZero(exponent)) {
        if (isOne(modulus)) return output;
        output[output.len - 1] = 1;
        return output;
    }

    if (base.len == 0 or isZero(base)) {
        return output;
    }

    if (base.len <= 8 and exponent.len <= 8 and modulus.len <= 8) {
        const result = modexpSmall(bytesToU64(base), bytesToU64(exponent), bytesToU64(modulus));
        const write_len = @min(output.len, 8);
        var i: usize = 0;
        while (i < write_len) : (i += 1) {
            const shift: u6 = @intCast((write_len - 1 - i) * 8);
            output[output.len - write_len + i] = @intCast((result >> shift) & 0xff);
        }
        return output;
    }

    try modexpBig(allocator, base, exponent, modulus, output);
    return output;
}

fn modexpSmall(base: u64, exponent: u64, modulus: u64) u64 {
    const mod128: u128 = modulus;
    var result: u128 = 1;
    var base_mod: u128 = base % mod128;
    var exp_remaining = exponent;

    while (exp_remaining > 0) {
        if (exp_remaining & 1 == 1) {
            result = (result * base_mod) % mod128;
        }
        base_mod = (base_mod * base_mod) % mod128;
        exp_remaining >>= 1;
    }

    return @intCast(result);
}

fn readBigEndian(big: *std.math.big.int.Managed, bytes: []const u8) !void {
    if (bytes.len == 0) {
        try big.set(0);
        return;
    }

    try big.ensureCapacity(std.math.big.int.calcTwosCompLimbCount(bytes.len * 8));
    var mutable = big.toMutable();
    mutable.readTwosComplement(bytes, bytes.len * 8, .big, .unsigned);
    big.setMetadata(mutable.positive, mutable.len);
}

fn writeBigEndian(big: *const std.math.big.int.Managed, output: []u8) void {
    @memset(output, 0);
    big.toConst().writeTwosComplement(output, .big);
}

fn modexpBig(
    allocator: std.mem.Allocator,
    base_bytes: []const u8,
    exp_bytes: []const u8,
    mod_bytes: []const u8,
    output: []u8,
) Error!void {
    const Managed = std.math.big.int.Managed;

    var base = try Managed.init(allocator);
    defer base.deinit();
    var exp = try Managed.init(allocator);
    defer exp.deinit();
    var modulus = try Managed.init(allocator);
    defer modulus.deinit();

    try readBigEndian(&base, base_bytes);
    try readBigEndian(&exp, exp_bytes);
    try readBigEndian(&modulus, mod_bytes);
    if (modulus.eqlZero()) return error.DivisionByZero;

    var zero = try Managed.init(allocator);
    defer zero.deinit();
    try zero.set(0);

    var result = try Managed.init(allocator);
    defer result.deinit();
    try result.set(1);

    var base_mod = try Managed.init(allocator);
    defer base_mod.deinit();

    var quotient = try Managed.init(allocator);
    defer quotient.deinit();
    try quotient.divFloor(&base_mod, &base, &modulus);

    var temp = try Managed.init(allocator);
    defer temp.deinit();
    var exp_copy = try exp.clone();
    defer exp_copy.deinit();

    while (!exp_copy.eql(zero)) {
        if (exp_copy.isOdd()) {
            try temp.mul(&result, &base_mod);
            try quotient.divFloor(&result, &temp, &modulus);
        }

        try temp.sqr(&base_mod);
        try quotient.divFloor(&base_mod, &temp, &modulus);

        try exp_copy.shiftRight(&exp_copy, 1);
    }

    writeBigEndian(&result, output);
}

test "modexp gas uses EIP-2565 minimum" {
    const exp = [_]u8{3};
    try std.testing.expectEqual(@as(u64, 200), calculateGas(1, 1, 1, &exp));
}

test "modexp computes small exponentiation" {
    const out = try modexp(std.testing.allocator, &[_]u8{2}, &[_]u8{3}, &[_]u8{5});
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(u8, &[_]u8{3}, out);
}

test "modexp exponent zero is reduced by modulus" {
    const out = try modexp(std.testing.allocator, &[_]u8{2}, &[_]u8{0}, &[_]u8{1});
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(u8, &[_]u8{0}, out);
}

test "modexp small fast path does not overflow u64 multiplication" {
    const base = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    const exp = [_]u8{2};
    const modulus = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc5 };

    const out = try modexp(std.testing.allocator, &base, &exp, &modulus);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, modulus.len), out.len);
}

test "modexp computes large exponentiation" {
    const base = [_]u8{2};
    const exp = [_]u8{0x7f};
    const modulus = [_]u8{1} ++ ([_]u8{0} ** 16);

    const out = try modexp(std.testing.allocator, &base, &exp, &modulus);
    defer std.testing.allocator.free(out);

    var expected = [_]u8{0} ** 17;
    expected[1] = 0x80;
    try std.testing.expectEqualSlices(u8, &expected, out);
}
