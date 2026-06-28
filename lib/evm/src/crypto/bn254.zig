const std = @import("std");

const Fp12Mont = @import("bn254/Fp12Mont.zig");
const Fp2Mont = @import("bn254/Fp2Mont.zig");
const FpMont = @import("bn254/FpMont.zig");
const G1 = @import("bn254/G1.zig");
const G2 = @import("bn254/G2.zig");
const curve_parameters = @import("bn254/curve_parameters.zig");
const pairing_impl = @import("bn254/pairing.zig");

pub fn add(input: *const [128]u8, output: []u8) !void {
    if (output.len < 64) return error.InvalidOutput;

    const p1 = try parseG1(input[0..64]);
    const p2 = try parseG1(input[64..128]);
    writeG1(output[0..64], p1.add(&p2));
}

pub fn mul(input: *const [96]u8, output: []u8) !void {
    if (output.len < 64) return error.InvalidOutput;

    const point = try parseG1(input[0..64]);
    const scalar = std.mem.readInt(u256, input[64..96], .big);
    writeG1(output[0..64], point.mulByInt(scalar, curve_parameters.G1_SCALAR.window_size));
}

pub fn pairing(input: []const u8) !bool {
    if (input.len % 192 != 0) return error.InvalidInput;
    if (input.len == 0) return true;

    var miller_result = Fp12Mont.ONE;
    const pair_count = input.len / 192;
    var index: usize = 0;
    while (index < pair_count) : (index += 1) {
        const offset = index * 192;
        const g1 = try parseG1(input[offset .. offset + 64]);
        const g2 = try parseG2(input[offset + 64 .. offset + 192]);
        const pair_result = try pairing_impl.millerLoop(&g1, &g2);
        miller_result = miller_result.mul(&pair_result);
    }

    const final = try pairing_impl.finalExponentiation(&miller_result);
    return final.equal(&Fp12Mont.ONE);
}

fn parseG1(input: []const u8) !G1 {
    std.debug.assert(input.len == 64);

    const x_value = std.mem.readInt(u256, input[0..32], .big);
    const y_value = std.mem.readInt(u256, input[32..64], .big);
    if (x_value == 0 and y_value == 0) return G1.INFINITY;

    if (x_value >= curve_parameters.FP_MOD or y_value >= curve_parameters.FP_MOD) {
        return error.InvalidPoint;
    }

    const x = FpMont.init(x_value);
    const y = FpMont.init(y_value);
    const z = FpMont.ONE;
    return G1.init(&x, &y, &z);
}

fn parseG2(input: []const u8) !G2 {
    std.debug.assert(input.len == 128);

    const x_c0 = std.mem.readInt(u256, input[0..32], .big);
    const x_c1 = std.mem.readInt(u256, input[32..64], .big);
    const y_c0 = std.mem.readInt(u256, input[64..96], .big);
    const y_c1 = std.mem.readInt(u256, input[96..128], .big);
    if (x_c0 == 0 and x_c1 == 0 and y_c0 == 0 and y_c1 == 0) return G2.INFINITY;

    if (x_c0 >= curve_parameters.FP_MOD or
        x_c1 >= curve_parameters.FP_MOD or
        y_c0 >= curve_parameters.FP_MOD or
        y_c1 >= curve_parameters.FP_MOD)
    {
        return error.InvalidPoint;
    }

    const x = Fp2Mont.initFromInt(x_c0, x_c1);
    const y = Fp2Mont.initFromInt(y_c0, y_c1);
    const z = Fp2Mont.ONE;
    return G2.init(&x, &y, &z);
}

fn writeG1(output: []u8, point: G1) void {
    std.debug.assert(output.len >= 64);

    const affine = point.toAffine();
    if (affine.isInfinity()) {
        @memset(output[0..64], 0);
        return;
    }

    std.mem.writeInt(u256, output[0..32], affine.x.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[32..64], affine.y.toStandardRepresentation(), .big);
}

test "bn254 add generator plus generator" {
    const input = [_]u8{
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
    };
    var output: [64]u8 = undefined;
    try add(&input, &output);

    const expected = [_]u8{
        0x03, 0x06, 0x44, 0xe7, 0x2e, 0x13, 0x1a, 0x02,
        0x9b, 0x85, 0x04, 0x5b, 0x68, 0x18, 0x15, 0x85,
        0xd9, 0x78, 0x16, 0xa9, 0x16, 0x87, 0x1c, 0xa8,
        0xd3, 0xc2, 0x08, 0xc1, 0x6d, 0x87, 0xcf, 0xd3,
        0x15, 0xed, 0x73, 0x8c, 0x0e, 0x0a, 0x7c, 0x92,
        0xe7, 0x84, 0x5f, 0x96, 0xb2, 0xae, 0x9c, 0x0a,
        0x68, 0xa6, 0xa4, 0x49, 0xe3, 0x53, 0x8f, 0xc7,
        0xff, 0x3e, 0xbf, 0x7a, 0x5a, 0x18, 0xa2, 0xc4,
    };
    try std.testing.expectEqualSlices(u8, &expected, &output);
}

test "bn254 rejects coordinates outside the field" {
    var input = [_]u8{0} ** 128;
    std.mem.writeInt(u256, input[0..32], curve_parameters.FP_MOD, .big);
    input[63] = 2;
    try std.testing.expectError(error.InvalidPoint, add(&input, input[64..128]));
}

test "bn254 mul matches Anvil for non-generator point by two" {
    const input = [_]u8{
        0x03, 0x97, 0x30, 0xea, 0x8d, 0xff, 0x12, 0x54,
        0xc0, 0xfe, 0xe9, 0xc0, 0xea, 0x77, 0x7d, 0x29,
        0xa9, 0xc7, 0x10, 0xb7, 0xe6, 0x16, 0x68, 0x3f,
        0x19, 0x4f, 0x18, 0xc4, 0x3b, 0x43, 0xb8, 0x69,
        0x07, 0x3a, 0x5f, 0xfc, 0xc6, 0xfc, 0x7a, 0x28,
        0xc3, 0x07, 0x23, 0xd6, 0xe5, 0x8c, 0xe5, 0x77,
        0x35, 0x69, 0x82, 0xd6, 0x5b, 0x83, 0x3a, 0x5a,
        0x5c, 0x15, 0xbf, 0x90, 0x24, 0xb4, 0x3d, 0x98,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
    };
    var output: [64]u8 = undefined;
    try mul(&input, &output);

    const expected = [_]u8{
        0x2d, 0xbc, 0x7b, 0xa6, 0x8f, 0x84, 0x0c, 0x75,
        0x8c, 0x76, 0x37, 0x3c, 0xd3, 0x7b, 0x2c, 0xd7,
        0x8d, 0x6b, 0x02, 0xbe, 0xe0, 0x47, 0xcf, 0x40,
        0x1e, 0x8d, 0xb9, 0x0d, 0x73, 0xce, 0x56, 0xf7,
        0x06, 0x28, 0x00, 0x98, 0x7e, 0xe0, 0xda, 0xe9,
        0xf9, 0xf3, 0x6e, 0x1f, 0x05, 0x0e, 0xb2, 0x62,
        0x1c, 0xbb, 0x4a, 0xa7, 0xc5, 0x0b, 0x1c, 0x16,
        0x8e, 0xcc, 0x31, 0x93, 0x70, 0x88, 0x9d, 0xe2,
    };
    try std.testing.expectEqualSlices(u8, &expected, &output);
}

test "bn254 pairing empty input succeeds" {
    try std.testing.expect(try pairing(&[_]u8{}));
}

test "bn254 pairing single generator pair is non-identity" {
    var input: [192]u8 = undefined;
    @memset(&input, 0);
    writePair(input[0..], G1.GENERATOR.toAffine(), G2.GENERATOR.toAffine());

    try std.testing.expect(!try pairing(&input));
}

test "bn254 pairing bilinearity identity" {
    const p1 = G1.GENERATOR.mulByInt(2, curve_parameters.G1_SCALAR.window_size).toAffine();
    const q1 = G2.GENERATOR.mulByInt(3, curve_parameters.G2_SCALAR.window_size).toAffine();
    const p2 = G1.GENERATOR.mulByInt(6, curve_parameters.G1_SCALAR.window_size).neg().toAffine();
    const q2 = G2.GENERATOR.toAffine();

    var input: [384]u8 = undefined;
    @memset(&input, 0);
    writePair(input[0..192], p1, q1);
    writePair(input[192..384], p2, q2);

    try std.testing.expect(try pairing(&input));
}

test "bn254 pairing rejects invalid length and out-of-field coordinates" {
    const short_input = [_]u8{0} ** 191;
    try std.testing.expectError(error.InvalidInput, pairing(&short_input));

    var bad_input = [_]u8{0} ** 192;
    std.mem.writeInt(u256, bad_input[64..96], curve_parameters.FP_MOD, .big);
    bad_input[159] = 1;
    try std.testing.expectError(error.InvalidPoint, pairing(&bad_input));
}

fn writePair(output: []u8, g1: G1, g2: G2) void {
    std.debug.assert(output.len == 192);

    std.mem.writeInt(u256, output[0..32], g1.x.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[32..64], g1.y.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[64..96], g2.x.u0.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[96..128], g2.x.u1.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[128..160], g2.y.u0.toStandardRepresentation(), .big);
    std.mem.writeInt(u256, output[160..192], g2.y.u1.toStandardRepresentation(), .big);
}
