const std = @import("std");
const c_kzg = @import("c_kzg");
const kzg_setup = @import("kzg_setup.zig");

pub const GAS: u64 = 50_000;
pub const INPUT_SIZE: usize = 192;
pub const OUTPUT_SIZE: usize = 64;

const FIELD_ELEMENTS_PER_BLOB_WORD = [_]u8{0} ** 30 ++ [_]u8{ 0x10, 0x00 };
const BLS_MODULUS = [_]u8{
    0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
    0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
    0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
};

pub fn execute(input: []const u8, output: []u8) !void {
    if (input.len != INPUT_SIZE) return error.InvalidInput;
    if (output.len < OUTPUT_SIZE) return error.InvalidOutput;

    const versioned_hash = input[0..32];
    const z_bytes = input[32..64];
    const y_bytes = input[64..96];
    const commitment_bytes = input[96..144];
    const proof_bytes = input[144..192];

    var computed_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(commitment_bytes, &computed_hash, .{});
    computed_hash[0] = 0x01;
    if (!std.mem.eql(u8, versioned_hash, &computed_hash)) return error.InvalidInput;

    try kzg_setup.init();

    var commitment: c_kzg.KZGCommitment = undefined;
    var z: c_kzg.Bytes32 = undefined;
    var y: c_kzg.Bytes32 = undefined;
    var proof: c_kzg.KZGProof = undefined;
    @memcpy(&commitment, commitment_bytes);
    @memcpy(&z, z_bytes);
    @memcpy(&y, y_bytes);
    @memcpy(&proof, proof_bytes);

    const valid = kzg_setup.verifyProof(&commitment, &z, &y, &proof) catch return error.InvalidInput;

    @memset(output[0..OUTPUT_SIZE], 0);
    if (valid) {
        @memcpy(output[0..32], &FIELD_ELEMENTS_PER_BLOB_WORD);
        @memcpy(output[32..64], &BLS_MODULUS);
    }
}

test "point evaluation rejects invalid input length" {
    var output: [OUTPUT_SIZE]u8 = undefined;
    const short = [_]u8{0} ** (INPUT_SIZE - 1);
    try std.testing.expectError(error.InvalidInput, execute(&short, &output));
}

test "point evaluation rejects versioned hash mismatch" {
    var input = [_]u8{0} ** INPUT_SIZE;
    input[0] = 0xff;
    input[96] = 0xc0;
    var output: [OUTPUT_SIZE]u8 = undefined;
    try std.testing.expectError(error.InvalidInput, execute(&input, &output));
}

test "point evaluation verifies point-at-infinity proof case" {
    var input = [_]u8{0} ** INPUT_SIZE;
    input[96] = 0xc0;
    input[144] = 0xc0;

    std.crypto.hash.sha2.Sha256.hash(input[96..144], input[0..32], .{});
    input[0] = 0x01;

    var output: [OUTPUT_SIZE]u8 = undefined;
    try execute(&input, &output);

    try std.testing.expectEqualSlices(u8, &FIELD_ELEMENTS_PER_BLOB_WORD, output[0..32]);
    try std.testing.expectEqualSlices(u8, &BLS_MODULUS, output[32..64]);
}
