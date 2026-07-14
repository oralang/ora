const std = @import("std");

const Secp256k1 = std.crypto.ecc.Secp256k1;

pub const SECP256K1_P: u256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;
pub const SECP256K1_N: u256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141;
pub const SECP256K1_B: u256 = 7;
pub const SECP256K1_GX: u256 = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798;
pub const SECP256K1_GY: u256 = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8;

const HALF_N = SECP256K1_N >> 1;

const AffinePoint = struct {
    x: u256,
    y: u256,
    infinity: bool,

    fn zero() AffinePoint {
        return .{ .x = 0, .y = 0, .infinity = true };
    }

    fn generator() AffinePoint {
        return .{ .x = SECP256K1_GX, .y = SECP256K1_GY, .infinity = false };
    }

    fn negate(self: AffinePoint) AffinePoint {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = SECP256K1_P - self.y, .infinity = false };
    }

    fn add(self: AffinePoint, other: AffinePoint) AffinePoint {
        if (self.infinity) return other;
        if (other.infinity) return self;

        const sec1_a = self.toUncompressedSec1() catch return AffinePoint.zero();
        const sec1_b = other.toUncompressedSec1() catch return AffinePoint.zero();
        const pt_a = Secp256k1.fromSec1(&sec1_a) catch return AffinePoint.zero();
        const pt_b = Secp256k1.fromSec1(&sec1_b) catch return AffinePoint.zero();

        return fromStdPoint(pt_a.add(pt_b)) catch AffinePoint.zero();
    }

    fn scalarMul(self: AffinePoint, scalar: u256) AffinePoint {
        if (scalar == 0 or self.infinity) return AffinePoint.zero();

        var scalar_bytes: [32]u8 = undefined;
        std.mem.writeInt(u256, &scalar_bytes, scalar, .big);

        const sec1 = self.toUncompressedSec1() catch return AffinePoint.zero();
        const pt = Secp256k1.fromSec1(&sec1) catch return AffinePoint.zero();
        const result = pt.mul(scalar_bytes, .big) catch return AffinePoint.zero();

        return fromStdPoint(result) catch AffinePoint.zero();
    }

    fn toUncompressedSec1(self: AffinePoint) ![65]u8 {
        if (self.infinity) return error.PointAtInfinity;

        var result: [65]u8 = undefined;
        result[0] = 0x04;
        std.mem.writeInt(u256, result[1..][0..32], self.x, .big);
        std.mem.writeInt(u256, result[33..][0..32], self.y, .big);
        return result;
    }

    fn fromStdPoint(pt: Secp256k1) !AffinePoint {
        const sec1 = pt.toUncompressedSec1();
        if (sec1[0] == 0) {
            for (sec1[1..]) |byte| {
                if (byte != 0) return error.InvalidPoint;
            }
            return AffinePoint.zero();
        }

        return .{
            .x = std.mem.readInt(u256, sec1[1..][0..32], .big),
            .y = std.mem.readInt(u256, sec1[33..][0..32], .big),
            .infinity = false,
        };
    }
};

pub fn recoverPubkey(hash: []const u8, r: []const u8, s: []const u8, v: u8) ![64]u8 {
    if (hash.len != 32) return error.InvalidHashLength;
    if (r.len != 32) return error.InvalidRLength;
    if (s.len != 32) return error.InvalidSLength;

    const r_value = std.mem.readInt(u256, r[0..][0..32], .big);
    const s_value = std.mem.readInt(u256, s[0..][0..32], .big);

    const recovery_id: u8 = if (v >= 27 and v <= 28)
        v - 27
    else if (v <= 1)
        v
    else
        return error.InvalidRecoveryId;

    if (!isValidEthereumSignature(r_value, s_value)) return error.InvalidSignature;

    var hash_array: [32]u8 = undefined;
    @memcpy(&hash_array, hash);

    const public_key = try recoverPublicKeyPoint(hash_array, r_value, s_value, recovery_id);

    var result: [64]u8 = undefined;
    std.mem.writeInt(u256, result[0..][0..32], public_key.x, .big);
    std.mem.writeInt(u256, result[32..][0..32], public_key.y, .big);
    return result;
}

fn isValidEthereumSignature(r: u256, s: u256) bool {
    return r != 0 and r < SECP256K1_N and s != 0 and s <= HALF_N;
}

fn recoverPublicKeyPoint(hash: [32]u8, r: u256, s: u256, recovery_id: u8) !AffinePoint {
    if (r >= SECP256K1_P) return error.InvalidSignature;

    const r_point = try computePointFromX(r, recovery_id);
    const e = std.mem.readInt(u256, &hash, .big);
    const r_inverse = modInversePrime(r, SECP256K1_N) orelse return error.InvalidSignature;

    const s_r = r_point.scalarMul(s);
    if (s_r.infinity) return error.InvalidSignature;

    const e_g = AffinePoint.generator().scalarMul(e);
    const diff = s_r.add(e_g.negate());
    if (diff.infinity) return error.InvalidSignature;

    const public_key = diff.scalarMul(r_inverse);
    if (public_key.infinity) return error.InvalidSignature;

    if (!verifySignature(hash, r, s, public_key)) return error.InvalidSignature;
    return public_key;
}

fn computePointFromX(x: u256, recovery_id: u8) !AffinePoint {
    const x3 = mulmod(mulmod(x, x, SECP256K1_P), x, SECP256K1_P);
    const y2 = addmod(x3, SECP256K1_B, SECP256K1_P);
    const y = powmod(y2, (SECP256K1_P + 1) >> 2, SECP256K1_P);

    if (mulmod(y, y, SECP256K1_P) != y2) return error.InvalidSignature;

    const use_odd_y = (recovery_id & 1) == 1;
    const y_is_odd = (y & 1) == 1;
    const final_y = if (y_is_odd == use_odd_y) y else SECP256K1_P - y;

    return .{ .x = x, .y = final_y, .infinity = false };
}

fn verifySignature(hash: [32]u8, r: u256, s: u256, public_key: AffinePoint) bool {
    if (public_key.infinity) return false;

    const e = std.mem.readInt(u256, &hash, .big);
    const s_inverse = modInversePrime(s, SECP256K1_N) orelse return false;

    const scalar_from_hash = mulmod(e, s_inverse, SECP256K1_N);
    const scalar_from_r = mulmod(r, s_inverse, SECP256K1_N);

    const hash_point = AffinePoint.generator().scalarMul(scalar_from_hash);
    const key_point = public_key.scalarMul(scalar_from_r);
    const r_point = hash_point.add(key_point);

    if (r_point.infinity) return false;
    return r_point.x % SECP256K1_N == r;
}

fn addmod(a: u256, b: u256, modulus: u256) u256 {
    if (modulus == 0) return 0;

    const a_mod = a % modulus;
    const b_mod = b % modulus;
    if (a_mod > modulus - b_mod) return a_mod - (modulus - b_mod);
    return a_mod + b_mod;
}

fn mulmod(a: u256, b: u256, modulus: u256) u256 {
    if (modulus == 0 or a == 0 or b == 0) return 0;

    var result: u256 = 0;
    var multiplicand = a % modulus;
    var multiplier = b % modulus;

    while (multiplier > 0) {
        if ((multiplier & 1) == 1) {
            result = addmod(result, multiplicand, modulus);
        }
        multiplicand = addmod(multiplicand, multiplicand, modulus);
        multiplier >>= 1;
    }

    return result;
}

fn powmod(base: u256, exponent: u256, modulus: u256) u256 {
    if (modulus == 1) return 0;

    var result: u256 = 1;
    var base_mod = base % modulus;
    var exponent_remaining = exponent;

    while (exponent_remaining > 0) {
        if ((exponent_remaining & 1) == 1) {
            result = mulmod(result, base_mod, modulus);
        }
        base_mod = mulmod(base_mod, base_mod, modulus);
        exponent_remaining >>= 1;
    }

    return result;
}

fn modInversePrime(value: u256, prime_modulus: u256) ?u256 {
    if (value == 0 or prime_modulus <= 2) return null;
    return powmod(value, prime_modulus - 2, prime_modulus);
}

test "secp256k1 signature validation rejects zero and high-s values" {
    try std.testing.expect(!isValidEthereumSignature(0, 1));
    try std.testing.expect(!isValidEthereumSignature(1, 0));
    try std.testing.expect(!isValidEthereumSignature(1, HALF_N + 1));
    try std.testing.expect(isValidEthereumSignature(1, HALF_N));
}

test "secp256k1 recovers public key for fixed raw hash signature" {
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

    const public_key = try recoverPubkey(&hash, &r, &s, 28);

    var public_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(&public_key, &public_hash, .{});
    const expected_address = [_]u8{
        0x7e, 0x5f, 0x45, 0x52, 0x09, 0x1a, 0x69, 0x12,
        0x5d, 0x5d, 0xfc, 0xb7, 0xb8, 0xc2, 0x65, 0x90,
        0x29, 0x39, 0x5b, 0xdf,
    };
    try std.testing.expectEqualSlices(u8, &expected_address, public_hash[12..]);
}
