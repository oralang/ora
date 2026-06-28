//! Ora-owned EVM precompile execution facade.
//!
//! Ora owns the precompile execution boundary and hardfork address range here.

const std = @import("std");
const bls12_381 = @import("crypto/bls12_381.zig");
const bn254 = @import("crypto/bn254.zig");
const blake2f = @import("crypto/blake2f.zig");
const modexp = @import("crypto/modexp.zig");
const point_evaluation = @import("crypto/point_evaluation.zig");
const primitives = @import("primitives.zig");
const ripemd160 = @import("crypto/ripemd160.zig");
const secp256k1 = @import("crypto/secp256k1.zig");

pub const PrecompileResult = struct {
    output: []u8,
    gas_used: u64,

    pub fn deinit(self: PrecompileResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

pub const PrecompileError = error{
    InvalidInput,
    InvalidSignature,
    InvalidPoint,
    InvalidPairing,
    OutOfGas,
    NotImplemented,
    ExecutionError,
    StateError,
    MemoryError,
    InvalidOutputSize,
    Unknown,
} || std.mem.Allocator.Error;

pub const ECRECOVER_ADDRESS = primitives.Address.fromU256(0x01);
pub const SHA256_ADDRESS = primitives.Address.fromU256(0x02);
pub const RIPEMD160_ADDRESS = primitives.Address.fromU256(0x03);
pub const IDENTITY_ADDRESS = primitives.Address.fromU256(0x04);
pub const MODEXP_ADDRESS = primitives.Address.fromU256(0x05);
pub const ECADD_ADDRESS = primitives.Address.fromU256(0x06);
pub const ECMUL_ADDRESS = primitives.Address.fromU256(0x07);
pub const ECPAIRING_ADDRESS = primitives.Address.fromU256(0x08);
pub const BLAKE2F_ADDRESS = primitives.Address.fromU256(0x09);
pub const POINT_EVALUATION_ADDRESS = primitives.Address.fromU256(0x0A);
pub const BLS12_G1ADD_ADDRESS = primitives.Address.fromU256(0x0B);
pub const BLS12_G1MUL_ADDRESS = primitives.Address.fromU256(0x0C);
pub const BLS12_G1MSM_ADDRESS = primitives.Address.fromU256(0x0D);
pub const BLS12_G2ADD_ADDRESS = primitives.Address.fromU256(0x0E);
pub const BLS12_G2MUL_ADDRESS = primitives.Address.fromU256(0x0F);
pub const BLS12_G2MSM_ADDRESS = primitives.Address.fromU256(0x10);
pub const BLS12_PAIRING_ADDRESS = primitives.Address.fromU256(0x11);
pub const BLS12_MAP_FP_TO_G1_ADDRESS = primitives.Address.fromU256(0x12);
pub const BLS12_MAP_FP2_TO_G2_ADDRESS = primitives.Address.fromU256(0x13);

pub const MAX_PRECOMPILE_ADDRESS: u8 = 0x13;

pub fn maxAddressForHardfork(hardfork: primitives.Hardfork) u8 {
    if (hardfork.isAtLeast(.PRAGUE)) return 0x13;
    if (hardfork.isAtLeast(.CANCUN)) return 0x0A;
    if (hardfork.isAtLeast(.ISTANBUL)) return 0x09;
    if (hardfork.isAtLeast(.BYZANTIUM)) return 0x08;
    if (hardfork.isAtLeast(.FRONTIER)) return 0x04;
    return 0;
}

pub fn isPrecompile(address: primitives.Address, hardfork: primitives.Hardfork) bool {
    const addr = address.toU256();
    return addr >= 1 and addr <= maxAddressForHardfork(hardfork);
}

pub fn execute(
    allocator: std.mem.Allocator,
    address: primitives.Address,
    input: []const u8,
    gas_limit: u64,
    hardfork: primitives.Hardfork,
) PrecompileError!PrecompileResult {
    if (!isPrecompile(address, hardfork)) return error.NotImplemented;

    return switch (address.toU256()) {
        0x01 => executeEcrecover(allocator, input, gas_limit),
        0x02 => executeSha256(allocator, input, gas_limit),
        0x03 => executeRipemd160(allocator, input, gas_limit),
        0x04 => executeIdentity(allocator, input, gas_limit),
        0x05 => executeModExp(allocator, input, gas_limit),
        0x06 => executeEcAdd(allocator, input, gas_limit),
        0x07 => executeEcMul(allocator, input, gas_limit),
        0x08 => executeEcPairing(allocator, input, gas_limit),
        0x09 => executeBlake2f(allocator, input, gas_limit),
        0x0A => executePointEvaluation(allocator, input, gas_limit),
        0x0B => executeBlsG1Add(allocator, input, gas_limit),
        0x0C => executeBlsG1Mul(allocator, input, gas_limit),
        0x0D => executeBlsG1Msm(allocator, input, gas_limit),
        0x0E => executeBlsG2Add(allocator, input, gas_limit),
        0x0F => executeBlsG2Mul(allocator, input, gas_limit),
        0x10 => executeBlsG2Msm(allocator, input, gas_limit),
        0x11 => executeBlsPairing(allocator, input, gas_limit),
        0x12 => executeBlsMapFpToG1(allocator, input, gas_limit),
        0x13 => executeBlsMapFp2ToG2(allocator, input, gas_limit),
        else => error.NotImplemented,
    };
}

const IDENTITY_BASE_GAS: u64 = 15;
const IDENTITY_PER_WORD_GAS: u64 = 3;
const ECRECOVER_GAS: u64 = 3000;
const SHA256_BASE_GAS: u64 = 60;
const SHA256_PER_WORD_GAS: u64 = 12;
const RIPEMD160_BASE_GAS: u64 = 600;
const RIPEMD160_PER_WORD_GAS: u64 = 120;
const ECADD_GAS: u64 = 150;
const ECMUL_GAS: u64 = 6000;
const ECPAIRING_BASE_GAS: u64 = 45_000;
const ECPAIRING_PER_PAIR_GAS: u64 = 34_000;
const BLAKE2F_PER_ROUND_GAS: u64 = 1;
const BLS_G1ADD_GAS: u64 = 500;
const BLS_G1MUL_GAS: u64 = 12_000;
const BLS_G1MSM_BASE_GAS: u64 = 12_000;
const BLS_G2ADD_GAS: u64 = 800;
const BLS_G2MUL_GAS: u64 = 45_000;
const BLS_G2MSM_BASE_GAS: u64 = 45_000;
const BLS_PAIRING_BASE_GAS: u64 = 65_000;
const BLS_PAIRING_PER_PAIR_GAS: u64 = 43_000;
const BLS_MAP_FP_TO_G1_GAS: u64 = 5_500;
const BLS_MAP_FP2_TO_G2_GAS: u64 = 75_000;

fn wordCount(byte_len: usize) u64 {
    return (byte_len + 31) / 32;
}

fn zeroAddressOutput(allocator: std.mem.Allocator, gas_used: u64) PrecompileError!PrecompileResult {
    const output = try allocator.alloc(u8, 32);
    @memset(output, 0);
    return .{
        .output = output,
        .gas_used = gas_used,
    };
}

fn executeEcrecover(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (gas_limit < ECRECOVER_GAS) return error.OutOfGas;

    var padded_input = [_]u8{0} ** 128;
    const copy_len = @min(input.len, padded_input.len);
    @memcpy(padded_input[0..copy_len], input[0..copy_len]);

    const hash = padded_input[0..32];
    const v = padded_input[63];
    const r = padded_input[64..96];
    const s = padded_input[96..128];

    const public_key = secp256k1.recoverPubkey(hash, r, s, v) catch {
        return zeroAddressOutput(allocator, ECRECOVER_GAS);
    };

    var public_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(&public_key, &public_hash, .{});

    const output = try allocator.alloc(u8, 32);
    @memset(output[0..12], 0);
    @memcpy(output[12..32], public_hash[12..32]);
    return .{
        .output = output,
        .gas_used = ECRECOVER_GAS,
    };
}

fn executeIdentity(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    const gas_cost = IDENTITY_BASE_GAS + IDENTITY_PER_WORD_GAS * wordCount(input.len);
    if (gas_limit < gas_cost) return error.OutOfGas;

    return .{
        .output = try allocator.dupe(u8, input),
        .gas_used = gas_cost,
    };
}

fn executeSha256(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    const gas_cost = SHA256_BASE_GAS + SHA256_PER_WORD_GAS * wordCount(input.len);
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 32);
    std.crypto.hash.sha2.Sha256.hash(input, output[0..32], .{});

    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeRipemd160(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    const gas_cost = RIPEMD160_BASE_GAS + RIPEMD160_PER_WORD_GAS * wordCount(input.len);
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 32);
    @memset(output[0..12], 0);

    var hash_output: [20]u8 = undefined;
    ripemd160.Ripemd160.hash(input, &hash_output);
    @memcpy(output[12..32], &hash_output);

    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeModExp(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len < 96) return error.InvalidInput;

    const base_len = std.mem.readInt(u256, input[0..][0..32], .big);
    const exp_len = std.mem.readInt(u256, input[32..][0..32], .big);
    const mod_len = std.mem.readInt(u256, input[64..][0..32], .big);

    if (base_len > std.math.maxInt(usize) or
        exp_len > std.math.maxInt(usize) or
        mod_len > std.math.maxInt(usize))
    {
        return error.InvalidInput;
    }

    const base_len_usize: usize = @intCast(base_len);
    const exp_len_usize: usize = @intCast(exp_len);
    const mod_len_usize: usize = @intCast(mod_len);

    const base_start: usize = 96;
    const exp_start = base_start + base_len_usize;
    const mod_start = exp_start + exp_len_usize;

    const exponent_for_gas = if (exp_start + exp_len_usize <= input.len)
        input[exp_start .. exp_start + exp_len_usize]
    else
        &[_]u8{};

    const gas_cost = modexp.calculateGas(base_len_usize, exp_len_usize, mod_len_usize, exponent_for_gas);
    if (gas_limit < gas_cost) return error.OutOfGas;

    const base = if (base_start + base_len_usize <= input.len)
        input[base_start .. base_start + base_len_usize]
    else
        &[_]u8{};

    const exponent = if (exp_start + exp_len_usize <= input.len)
        input[exp_start .. exp_start + exp_len_usize]
    else
        &[_]u8{};

    const modulus = if (mod_start + mod_len_usize <= input.len)
        input[mod_start .. mod_start + mod_len_usize]
    else
        &[_]u8{};

    const result = modexp.modexp(allocator, base, exponent, modulus) catch |err| switch (err) {
        error.DivisionByZero,
        error.InvalidInput,
        error.InvalidBase,
        error.InvalidCharacter,
        error.InvalidLength,
        => return error.InvalidInput,
        error.AllocationFailed,
        error.OutOfMemory,
        error.NoSpaceLeft,
        => return error.OutOfMemory,
        error.NotImplemented => return error.NotImplemented,
    };
    defer allocator.free(result);

    const output = try allocator.alloc(u8, mod_len_usize);
    @memset(output, 0);
    if (result.len <= mod_len_usize) {
        const offset = mod_len_usize - result.len;
        @memcpy(output[offset..], result);
    } else {
        @memcpy(output, result[result.len - mod_len_usize ..]);
    }

    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeEcAdd(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (gas_limit < ECADD_GAS) return error.OutOfGas;

    var padded_input = [_]u8{0} ** 128;
    const copy_len = @min(input.len, padded_input.len);
    @memcpy(padded_input[0..copy_len], input[0..copy_len]);

    const output = try allocator.alloc(u8, 64);
    errdefer allocator.free(output);
    bn254.add(&padded_input, output) catch return error.InvalidPoint;
    return .{
        .output = output,
        .gas_used = ECADD_GAS,
    };
}

fn executeEcMul(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (gas_limit < ECMUL_GAS) return error.OutOfGas;

    var padded_input = [_]u8{0} ** 96;
    const copy_len = @min(input.len, padded_input.len);
    @memcpy(padded_input[0..copy_len], input[0..copy_len]);

    const output = try allocator.alloc(u8, 64);
    errdefer allocator.free(output);
    bn254.mul(&padded_input, output) catch return error.InvalidPoint;
    return .{
        .output = output,
        .gas_used = ECMUL_GAS,
    };
}

fn executeEcPairing(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len % 192 != 0) return error.InvalidInput;

    const pair_count: u64 = @intCast(input.len / 192);
    const gas_cost = ECPAIRING_BASE_GAS + ECPAIRING_PER_PAIR_GAS * pair_count;
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 32);
    errdefer allocator.free(output);
    @memset(output, 0);

    const success = bn254.pairing(input) catch return error.InvalidPairing;
    if (success) output[31] = 1;
    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executePointEvaluation(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (gas_limit < point_evaluation.GAS) return error.OutOfGas;
    if (input.len != point_evaluation.INPUT_SIZE) return error.InvalidInput;

    const output = try allocator.alloc(u8, point_evaluation.OUTPUT_SIZE);
    errdefer allocator.free(output);

    point_evaluation.execute(input, output) catch |err| switch (err) {
        error.InvalidInput => return error.InvalidInput,
        else => return error.ExecutionError,
    };

    return .{
        .output = output,
        .gas_used = point_evaluation.GAS,
    };
}

fn msmDiscount(k: usize) u64 {
    return if (k >= 128)
        174
    else if (k >= 64)
        200
    else if (k >= 32)
        250
    else if (k >= 16)
        320
    else if (k >= 8)
        430
    else if (k >= 4)
        580
    else if (k >= 2)
        820
    else
        1000;
}

fn mapBlsError(err: bls12_381.Error) PrecompileError {
    return switch (err) {
        error.InvalidInput, error.InvalidScalar => error.InvalidInput,
        error.InvalidPoint => error.InvalidPoint,
        error.ComputationFailed => error.ExecutionError,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn executeBlsFixed(
    comptime op: fn ([]const u8, []u8) bls12_381.Error!void,
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
    gas_cost: u64,
    expected_input_len: usize,
    output_len: usize,
) PrecompileError!PrecompileResult {
    if (gas_limit < gas_cost) return error.OutOfGas;
    if (input.len != expected_input_len) return error.InvalidInput;

    const output = try allocator.alloc(u8, output_len);
    errdefer allocator.free(output);
    op(input, output) catch |err| return mapBlsError(err);
    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeBlsG1Add(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.g1Add, allocator, input, gas_limit, BLS_G1ADD_GAS, 256, 128);
}

fn executeBlsG1Mul(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.g1Mul, allocator, input, gas_limit, BLS_G1MUL_GAS, 160, 128);
}

fn executeBlsG1Msm(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len == 0 or input.len % 160 != 0) return error.InvalidInput;

    const k = input.len / 160;
    const gas_cost = (BLS_G1MSM_BASE_GAS * @as(u64, @intCast(k)) * msmDiscount(k)) / 1000;
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 128);
    errdefer allocator.free(output);
    bls12_381.g1Msm(allocator, input, output) catch |err| return mapBlsError(err);
    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeBlsG2Add(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.g2Add, allocator, input, gas_limit, BLS_G2ADD_GAS, 512, 256);
}

fn executeBlsG2Mul(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.g2Mul, allocator, input, gas_limit, BLS_G2MUL_GAS, 288, 256);
}

fn executeBlsG2Msm(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len == 0 or input.len % 288 != 0) return error.InvalidInput;

    const k = input.len / 288;
    const gas_cost = (BLS_G2MSM_BASE_GAS * @as(u64, @intCast(k)) * msmDiscount(k)) / 1000;
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 256);
    errdefer allocator.free(output);
    bls12_381.g2Msm(allocator, input, output) catch |err| return mapBlsError(err);
    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeBlsPairing(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len % 384 != 0) return error.InvalidInput;

    const k = input.len / 384;
    const gas_cost = BLS_PAIRING_BASE_GAS + BLS_PAIRING_PER_PAIR_GAS * @as(u64, @intCast(k));
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 32);
    errdefer allocator.free(output);
    bls12_381.pairing(allocator, input, output) catch |err| return mapBlsError(err);
    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

fn executeBlsMapFpToG1(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.mapFpToG1, allocator, input, gas_limit, BLS_MAP_FP_TO_G1_GAS, 64, 128);
}

fn executeBlsMapFp2ToG2(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    return executeBlsFixed(bls12_381.mapFp2ToG2, allocator, input, gas_limit, BLS_MAP_FP2_TO_G2_GAS, 128, 256);
}

fn executeBlake2f(
    allocator: std.mem.Allocator,
    input: []const u8,
    gas_limit: u64,
) PrecompileError!PrecompileResult {
    if (input.len != 213) return error.InvalidInput;

    const rounds = std.mem.readInt(u32, input[0..][0..4], .big);
    const gas_cost = BLAKE2F_PER_ROUND_GAS * rounds;
    if (gas_limit < gas_cost) return error.OutOfGas;

    const output = try allocator.alloc(u8, 64);
    errdefer allocator.free(output);
    blake2f.compress(input, output) catch return error.InvalidInput;

    return .{
        .output = output,
        .gas_used = gas_cost,
    };
}

test "precompile address ranges cover Prague 0x01 through 0x13" {
    const testing = std.testing;

    try testing.expectEqual(@as(u8, 0x04), maxAddressForHardfork(.FRONTIER));
    try testing.expectEqual(@as(u8, 0x08), maxAddressForHardfork(.BYZANTIUM));
    try testing.expectEqual(@as(u8, 0x09), maxAddressForHardfork(.ISTANBUL));
    try testing.expectEqual(@as(u8, 0x0A), maxAddressForHardfork(.CANCUN));
    try testing.expectEqual(@as(u8, 0x13), maxAddressForHardfork(.PRAGUE));
    try testing.expectEqual(@as(u8, 0x13), maxAddressForHardfork(.OSAKA));

    try testing.expect(isPrecompile(primitives.Address.fromU256(0x13), .PRAGUE));
    try testing.expect(isPrecompile(primitives.Address.fromU256(0x13), .OSAKA));
    try testing.expect(!isPrecompile(primitives.Address.fromU256(0x13), .CANCUN));
    try testing.expect(!isPrecompile(primitives.Address.fromU256(0x14), .OSAKA));
}

test "identity precompile executes through Ora facade" {
    const testing = std.testing;
    const input = "ora";

    const result = try execute(
        testing.allocator,
        IDENTITY_ADDRESS,
        input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u8, input, result.output);
    try testing.expect(result.gas_used > 0);
}

test "ecrecover precompile executes through Ora facade with local body" {
    const testing = std.testing;

    const input = ecrecoverPrivateKeyOneFixture();
    const result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    const expected = [_]u8{
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x7e, 0x5f, 0x45, 0x52, 0x09, 0x1a,
        0x69, 0x12, 0x5d, 0x5d, 0xfc, 0xb7,
        0xb8, 0xc2, 0x65, 0x90, 0x29, 0x39,
        0x5b, 0xdf,
    };
    try testing.expectEqualSlices(u8, &expected, result.output);
    try testing.expectEqual(@as(u64, ECRECOVER_GAS), result.gas_used);
}

test "ecrecover precompile returns zero address for invalid signatures" {
    const testing = std.testing;

    var input = [_]u8{0} ** 128;
    const result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(u64, ECRECOVER_GAS), result.gas_used);
    for (result.output) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
}

test "ecrecover precompile accepts raw recovery id and rejects invalid v" {
    const testing = std.testing;

    var raw_v_input = ecrecoverPrivateKeyOneFixture();
    raw_v_input[63] = 1;
    const raw_v_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &raw_v_input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer raw_v_result.deinit(testing.allocator);

    const word_v_input = ecrecoverPrivateKeyOneFixture();
    const word_v_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &word_v_input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer word_v_result.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, word_v_result.output, raw_v_result.output);

    var invalid_v_input = ecrecoverPrivateKeyOneFixture();
    invalid_v_input[63] = 29;
    const invalid_v_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &invalid_v_input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer invalid_v_result.deinit(testing.allocator);
    for (invalid_v_result.output) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
}

test "ecrecover precompile rejects high-s signatures with zero address" {
    const testing = std.testing;

    var input = ecrecoverPrivateKeyOneFixture();
    input[96] = 0x80;
    @memset(input[97..128], 0);

    const result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    for (result.output) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
}

test "ecrecover precompile pads short input and truncates long input" {
    const testing = std.testing;

    var short_input = ecrecoverPrivateKeyOneFixture();
    const short_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        short_input[0..96],
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer short_result.deinit(testing.allocator);
    for (short_result.output) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }

    var long_input = [_]u8{0} ** 160;
    const valid_input = ecrecoverPrivateKeyOneFixture();
    @memcpy(long_input[0..128], &valid_input);
    @memset(long_input[128..], 0xff);

    const valid_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &valid_input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer valid_result.deinit(testing.allocator);

    const long_result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &long_input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer long_result.deinit(testing.allocator);

    try testing.expectEqualSlices(u8, valid_result.output, long_result.output);
}

test "ecrecover precompile enforces fixed gas cost" {
    const testing = std.testing;

    const input = ecrecoverPrivateKeyOneFixture();
    try testing.expectError(
        error.OutOfGas,
        execute(testing.allocator, ECRECOVER_ADDRESS, &input, ECRECOVER_GAS - 1, .OSAKA),
    );

    const result = try execute(
        testing.allocator,
        ECRECOVER_ADDRESS,
        &input,
        ECRECOVER_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(u64, ECRECOVER_GAS), result.gas_used);
}

test "sha256 precompile executes through Ora facade with local body" {
    const testing = std.testing;

    const result = try execute(
        testing.allocator,
        SHA256_ADDRESS,
        "",
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    const expected = [_]u8{
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
    };
    try testing.expectEqualSlices(u8, &expected, result.output);
    try testing.expectEqual(@as(u64, SHA256_BASE_GAS), result.gas_used);
}

test "ripemd160 precompile executes through Ora facade with local body" {
    const testing = std.testing;

    const result = try execute(
        testing.allocator,
        RIPEMD160_ADDRESS,
        "",
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    const expected = [_]u8{
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9,
        0xfc, 0x54, 0x61, 0x28, 0x08, 0x97,
        0x7e, 0xe8, 0xf5, 0x48, 0xb2, 0x25,
        0x8d, 0x31,
    };
    try testing.expectEqualSlices(u8, &expected, result.output);
    try testing.expectEqual(@as(u64, RIPEMD160_BASE_GAS), result.gas_used);
}

test "modexp precompile executes through Ora facade with local body" {
    const testing = std.testing;

    var input = [_]u8{0} ** 99;
    input[31] = 1;
    input[63] = 1;
    input[95] = 1;
    input[96] = 2;
    input[97] = 3;
    input[98] = 5;

    const result = try execute(
        testing.allocator,
        MODEXP_ADDRESS,
        &input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u8, &[_]u8{3}, result.output);
    try testing.expectEqual(@as(u64, 200), result.gas_used);
}

test "ecadd precompile executes through Ora facade with local body" {
    const testing = std.testing;
    const input = hexFixture(
        128,
        "0000000000000000000000000000000000000000000000000000000000000001" ++
            "0000000000000000000000000000000000000000000000000000000000000002" ++
            "0000000000000000000000000000000000000000000000000000000000000001" ++
            "0000000000000000000000000000000000000000000000000000000000000002",
    );

    const result = try execute(
        testing.allocator,
        ECADD_ADDRESS,
        &input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    const expected = hexFixture(
        64,
        "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3" ++
            "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4",
    );
    try testing.expectEqualSlices(u8, &expected, result.output);
    try testing.expectEqual(@as(u64, ECADD_GAS), result.gas_used);
}

test "ecadd precompile pads short input as point at infinity" {
    const testing = std.testing;
    const input = hexFixture(
        64,
        "0000000000000000000000000000000000000000000000000000000000000001" ++
            "0000000000000000000000000000000000000000000000000000000000000002",
    );

    const result = try execute(
        testing.allocator,
        ECADD_ADDRESS,
        &input,
        ECADD_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(u8, &input, result.output);
}

test "ecadd precompile rejects invalid points and enforces gas" {
    const testing = std.testing;
    const input = hexFixture(
        128,
        "0000000000000000000000000000000000000000000000000000000000000001" ++
            "0000000000000000000000000000000000000000000000000000000000000001" ++
            "0000000000000000000000000000000000000000000000000000000000000000" ++
            "0000000000000000000000000000000000000000000000000000000000000000",
    );

    try testing.expectError(
        error.OutOfGas,
        execute(testing.allocator, ECADD_ADDRESS, &input, ECADD_GAS - 1, .OSAKA),
    );
    try testing.expectError(
        error.InvalidPoint,
        execute(testing.allocator, ECADD_ADDRESS, &input, ECADD_GAS, .OSAKA),
    );
}

test "ecmul precompile executes through Ora facade with local body" {
    const testing = std.testing;
    const input = hexFixture(
        96,
        "039730ea8dff1254c0fee9c0ea777d29a9c710b7e616683f194f18c43b43b869" ++
            "073a5ffcc6fc7a28c30723d6e58ce577356982d65b833a5a5c15bf9024b43d98" ++
            "0000000000000000000000000000000000000000000000000000000000000002",
    );

    const result = try execute(
        testing.allocator,
        ECMUL_ADDRESS,
        &input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    const expected = hexFixture(
        64,
        "2dbc7ba68f840c758c76373cd37b2cd78d6b02bee047cf401e8db90d73ce56f7" ++
            "062800987ee0dae9f9f36e1f050eb2621cbb4aa7c50b1c168ecc319370889de2",
    );
    try testing.expectEqualSlices(u8, &expected, result.output);
    try testing.expectEqual(@as(u64, ECMUL_GAS), result.gas_used);
}

test "ecmul precompile returns infinity for zero scalar and enforces gas" {
    const testing = std.testing;
    const input = hexFixture(
        96,
        "039730ea8dff1254c0fee9c0ea777d29a9c710b7e616683f194f18c43b43b869" ++
            "073a5ffcc6fc7a28c30723d6e58ce577356982d65b833a5a5c15bf9024b43d98" ++
            "0000000000000000000000000000000000000000000000000000000000000000",
    );

    try testing.expectError(
        error.OutOfGas,
        execute(testing.allocator, ECMUL_ADDRESS, &input, ECMUL_GAS - 1, .OSAKA),
    );

    const result = try execute(
        testing.allocator,
        ECMUL_ADDRESS,
        &input,
        ECMUL_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    for (result.output) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
}

test "ecpairing precompile executes through Ora facade with local body" {
    const testing = std.testing;

    const result = try execute(
        testing.allocator,
        ECPAIRING_ADDRESS,
        "",
        ECPAIRING_BASE_GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 32), result.output.len);
    try testing.expectEqual(@as(u8, 1), result.output[31]);
    try testing.expectEqual(@as(u64, ECPAIRING_BASE_GAS), result.gas_used);
}

test "ecpairing precompile treats all-infinity pair as success" {
    const testing = std.testing;
    const input = [_]u8{0} ** 192;
    const gas_cost = ECPAIRING_BASE_GAS + ECPAIRING_PER_PAIR_GAS;

    const result = try execute(
        testing.allocator,
        ECPAIRING_ADDRESS,
        &input,
        gas_cost,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 32), result.output.len);
    try testing.expectEqual(@as(u8, 1), result.output[31]);
    try testing.expectEqual(gas_cost, result.gas_used);
}

test "ecpairing precompile rejects bad length and enforces gas" {
    const testing = std.testing;

    const bad_length = [_]u8{0} ** 191;
    try testing.expectError(
        error.InvalidInput,
        execute(testing.allocator, ECPAIRING_ADDRESS, &bad_length, 1_000_000, .OSAKA),
    );

    const input = [_]u8{0} ** 192;
    const gas_cost = ECPAIRING_BASE_GAS + ECPAIRING_PER_PAIR_GAS;
    try testing.expectError(
        error.OutOfGas,
        execute(testing.allocator, ECPAIRING_ADDRESS, &input, gas_cost - 1, .OSAKA),
    );
}

test "point evaluation precompile executes through Ora facade with local body" {
    const testing = std.testing;
    var input = [_]u8{0} ** point_evaluation.INPUT_SIZE;
    input[96] = 0xc0;
    input[144] = 0xc0;
    std.crypto.hash.sha2.Sha256.hash(input[96..144], input[0..32], .{});
    input[0] = 0x01;

    const result = try execute(
        testing.allocator,
        POINT_EVALUATION_ADDRESS,
        &input,
        point_evaluation.GAS,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, point_evaluation.OUTPUT_SIZE), result.output.len);
    try testing.expectEqual(@as(u8, 0x10), result.output[30]);
    try testing.expectEqual(@as(u8, 0x00), result.output[31]);
    try testing.expectEqual(@as(u8, 0x73), result.output[32]);
    try testing.expectEqual(@as(u64, point_evaluation.GAS), result.gas_used);
}

test "point evaluation precompile rejects bad length and enforces gas" {
    const testing = std.testing;
    const input = [_]u8{0} ** point_evaluation.INPUT_SIZE;

    try testing.expectError(
        error.OutOfGas,
        execute(testing.allocator, POINT_EVALUATION_ADDRESS, &input, point_evaluation.GAS - 1, .OSAKA),
    );

    const bad_length = [_]u8{0} ** (point_evaluation.INPUT_SIZE - 1);
    try testing.expectError(
        error.InvalidInput,
        execute(testing.allocator, POINT_EVALUATION_ADDRESS, &bad_length, point_evaluation.GAS, .OSAKA),
    );
}

test "BLS precompiles execute through Ora facade with local body" {
    const testing = std.testing;

    const g1_add_input = [_]u8{0} ** 256;
    const g1_add = try execute(testing.allocator, BLS12_G1ADD_ADDRESS, &g1_add_input, BLS_G1ADD_GAS, .OSAKA);
    defer g1_add.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 128), g1_add.output.len);
    try testing.expectEqual(@as(u64, BLS_G1ADD_GAS), g1_add.gas_used);
    try testing.expectEqualSlices(u8, &([_]u8{0} ** 128), g1_add.output);

    const g2_add_input = [_]u8{0} ** 512;
    const g2_add = try execute(testing.allocator, BLS12_G2ADD_ADDRESS, &g2_add_input, BLS_G2ADD_GAS, .OSAKA);
    defer g2_add.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 256), g2_add.output.len);
    try testing.expectEqual(@as(u64, BLS_G2ADD_GAS), g2_add.gas_used);
    try testing.expectEqualSlices(u8, &([_]u8{0} ** 256), g2_add.output);

    const pairing = try execute(testing.allocator, BLS12_PAIRING_ADDRESS, "", BLS_PAIRING_BASE_GAS, .OSAKA);
    defer pairing.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 32), pairing.output.len);
    try testing.expectEqual(@as(u8, 1), pairing.output[31]);
}

test "BLS MSM and map precompiles validate gas and shape locally" {
    const testing = std.testing;

    const g1_msm_input = [_]u8{0} ** 160;
    const g1_msm_gas = (BLS_G1MSM_BASE_GAS * msmDiscount(1)) / 1000;
    const g1_msm = try execute(testing.allocator, BLS12_G1MSM_ADDRESS, &g1_msm_input, g1_msm_gas, .OSAKA);
    defer g1_msm.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 128), g1_msm.output.len);
    try testing.expectEqual(@as(u64, g1_msm_gas), g1_msm.gas_used);
    try testing.expectEqualSlices(u8, &([_]u8{0} ** 128), g1_msm.output);

    const g2_msm_input = [_]u8{0} ** 288;
    const g2_msm_gas = (BLS_G2MSM_BASE_GAS * msmDiscount(1)) / 1000;
    const g2_msm = try execute(testing.allocator, BLS12_G2MSM_ADDRESS, &g2_msm_input, g2_msm_gas, .OSAKA);
    defer g2_msm.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 256), g2_msm.output.len);
    try testing.expectEqual(@as(u64, g2_msm_gas), g2_msm.gas_used);
    try testing.expectEqualSlices(u8, &([_]u8{0} ** 256), g2_msm.output);

    try testing.expectError(
        error.InvalidInput,
        execute(testing.allocator, BLS12_G1MSM_ADDRESS, "", BLS_G1MSM_BASE_GAS, .OSAKA),
    );

    const fp_input = [_]u8{0} ** 64;
    const g1_map = try execute(testing.allocator, BLS12_MAP_FP_TO_G1_ADDRESS, &fp_input, BLS_MAP_FP_TO_G1_GAS, .OSAKA);
    defer g1_map.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 128), g1_map.output.len);

    const fp2_input = [_]u8{0} ** 128;
    const g2_map = try execute(testing.allocator, BLS12_MAP_FP2_TO_G2_ADDRESS, &fp2_input, BLS_MAP_FP2_TO_G2_GAS, .OSAKA);
    defer g2_map.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 256), g2_map.output.len);
}

test "BLS precompiles reject noncanonical EIP-2537 field words" {
    const testing = std.testing;
    var input = [_]u8{0} ** 256;
    input[0] = 1;

    try testing.expectError(
        error.InvalidInput,
        execute(testing.allocator, BLS12_G1ADD_ADDRESS, &input, BLS_G1ADD_GAS, .OSAKA),
    );
}

test "blake2f precompile executes through Ora facade with local body" {
    const testing = std.testing;

    var input = [_]u8{0} ** 213;
    std.mem.writeInt(u32, input[0..][0..4], 12, .big);
    input[212] = 1;

    const result = try execute(
        testing.allocator,
        BLAKE2F_ADDRESS,
        &input,
        1_000_000,
        .OSAKA,
    );
    defer result.deinit(testing.allocator);

    var direct: [64]u8 = undefined;
    try blake2f.compress(&input, &direct);

    try testing.expectEqualSlices(u8, &direct, result.output);
    try testing.expectEqual(@as(u64, 12), result.gas_used);
}

fn hexFixture(comptime byte_len: usize, comptime hex: []const u8) [byte_len]u8 {
    comptime std.debug.assert(hex.len == byte_len * 2);

    var output: [byte_len]u8 = undefined;
    _ = std.fmt.hexToBytes(&output, hex) catch unreachable;
    return output;
}

fn ecrecoverPrivateKeyOneFixture() [128]u8 {
    var input = [_]u8{0} ** 128;
    const hash = [_]u8{
        0x47, 0x17, 0x32, 0x85, 0xa8, 0xd7, 0x34, 0x1e,
        0x5e, 0x97, 0x2f, 0xc6, 0x77, 0x28, 0x63, 0x84,
        0xf8, 0x02, 0xf8, 0xef, 0x42, 0xa5, 0xec, 0x5f,
        0x03, 0xbb, 0xfa, 0x25, 0x4c, 0xb0, 0x1f, 0xad,
    };
    const r = [_]u8{
        0xe5, 0xce, 0x97, 0x87, 0x6a, 0x5f, 0x8b, 0xd0,
        0x70, 0xd7, 0xa7, 0xac, 0x38, 0xd0, 0x94, 0x8c,
        0x2c, 0x83, 0x01, 0x2d, 0xe5, 0xaf, 0xcb, 0x6a,
        0xa0, 0xa3, 0xaa, 0x07, 0xf2, 0xd0, 0xc3, 0xcd,
    };
    const s = [_]u8{
        0x10, 0xaf, 0xb2, 0xd6, 0xc6, 0xef, 0x98, 0x3e,
        0x13, 0x82, 0xd3, 0xe2, 0xad, 0x9c, 0x55, 0xa8,
        0x76, 0x5a, 0xa5, 0x02, 0x8b, 0xdd, 0x0b, 0x82,
        0x5c, 0x7c, 0x9c, 0xba, 0xe5, 0xf7, 0xe8, 0xc2,
    };

    @memcpy(input[0..32], &hash);
    input[63] = 28;
    @memcpy(input[64..96], &r);
    @memcpy(input[96..128], &s);
    return input;
}
