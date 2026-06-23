const std = @import("std");

pub const Error = error{
    InvalidInput,
    InvalidOutputSize,
};

const IV = [8]u64{
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
};

const SIGMA = [12][16]u8{
    .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    .{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    .{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    .{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    .{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    .{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    .{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    .{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    .{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    .{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    .{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

fn g(v: *[16]u64, a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) void {
    v[a] = v[a] +% v[b] +% x;
    v[d] = std.math.rotr(u64, v[d] ^ v[a], 32);
    v[c] = v[c] +% v[d];
    v[b] = std.math.rotr(u64, v[b] ^ v[c], 24);
    v[a] = v[a] +% v[b] +% y;
    v[d] = std.math.rotr(u64, v[d] ^ v[a], 16);
    v[c] = v[c] +% v[d];
    v[b] = std.math.rotr(u64, v[b] ^ v[c], 63);
}

fn round(v: *[16]u64, message: *const [16]u64, round_index: u32) void {
    const s = &SIGMA[round_index % 12];

    g(v, 0, 4, 8, 12, message[s[0]], message[s[1]]);
    g(v, 1, 5, 9, 13, message[s[2]], message[s[3]]);
    g(v, 2, 6, 10, 14, message[s[4]], message[s[5]]);
    g(v, 3, 7, 11, 15, message[s[6]], message[s[7]]);

    g(v, 0, 5, 10, 15, message[s[8]], message[s[9]]);
    g(v, 1, 6, 11, 12, message[s[10]], message[s[11]]);
    g(v, 2, 7, 8, 13, message[s[12]], message[s[13]]);
    g(v, 3, 4, 9, 14, message[s[14]], message[s[15]]);
}

fn compressState(h: *[8]u64, message: *const [16]u64, offset: [2]u64, final_block: bool, rounds: u32) void {
    var v: [16]u64 = undefined;
    for (0..8) |i| {
        v[i] = h[i];
        v[i + 8] = IV[i];
    }

    v[12] ^= offset[0];
    v[13] ^= offset[1];
    if (final_block) v[14] = ~v[14];

    for (0..rounds) |round_index| {
        round(&v, message, @intCast(round_index));
    }

    for (0..8) |i| {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// EIP-152 BLAKE2F compression wrapper.
/// Input: rounds(4)||h(64)||m(128)||t(16)||f(1), output: h(64).
pub fn compress(input: []const u8, output: []u8) Error!void {
    if (input.len != 213) return error.InvalidInput;
    if (output.len != 64) return error.InvalidOutputSize;

    const rounds = std.mem.readInt(u32, input[0..][0..4], .big);

    var h: [8]u64 = undefined;
    for (0..8) |i| {
        const offset = 4 + i * 8;
        h[i] = std.mem.readInt(u64, input[offset..][0..8], .little);
    }

    var message: [16]u64 = undefined;
    for (0..16) |i| {
        const offset = 68 + i * 8;
        message[i] = std.mem.readInt(u64, input[offset..][0..8], .little);
    }

    const counter = [2]u64{
        std.mem.readInt(u64, input[196..][0..8], .little),
        std.mem.readInt(u64, input[204..][0..8], .little),
    };

    // Match the existing accepted behavior: any non-zero flag means final block.
    const final_block = input[212] != 0;
    compressState(&h, &message, counter, final_block, rounds);

    for (0..8) |i| {
        const offset = i * 8;
        std.mem.writeInt(u64, output[offset..][0..8], h[i], .little);
    }
}

test "blake2f compress rejects malformed sizes" {
    var out: [64]u8 = undefined;
    try std.testing.expectError(error.InvalidInput, compress(&[_]u8{}, &out));

    var input = [_]u8{0} ** 213;
    var short_out: [63]u8 = undefined;
    try std.testing.expectError(error.InvalidOutputSize, compress(&input, &short_out));
}

test "blake2f compress is deterministic for exact EIP-152 frame" {
    var input = [_]u8{0} ** 213;
    std.mem.writeInt(u32, input[0..][0..4], 12, .big);
    for (0..8) |i| {
        std.mem.writeInt(u64, input[4 + i * 8 ..][0..8], IV[i], .little);
    }
    input[212] = 1;

    var first: [64]u8 = undefined;
    var second: [64]u8 = undefined;
    try compress(&input, &first);
    try compress(&input, &second);

    try std.testing.expectEqualSlices(u8, &first, &second);
    try std.testing.expect(!std.mem.allEqual(u8, &first, 0));
}
