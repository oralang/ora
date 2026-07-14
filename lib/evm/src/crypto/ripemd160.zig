const std = @import("std");

pub const Ripemd160 = struct {
    pub fn hash(input: []const u8, output: *[20]u8) void {
        var hasher = RIPEMD160.init();
        hasher.update(input);
        const result = hasher.final();
        @memcpy(output, &result);
    }
};

const RIPEMD160 = struct {
    s: [5]u32,
    buf: [64]u8,
    bytes: u64,

    fn init() RIPEMD160 {
        return .{
            .s = .{
                0x67452301,
                0xefcdab89,
                0x98badcfe,
                0x10325476,
                0xc3d2e1f0,
            },
            .buf = undefined,
            .bytes = 0,
        };
    }

    fn update(self: *RIPEMD160, data: []const u8) void {
        var input = data;
        const buf_used: usize = @intCast(self.bytes % 64);

        if (buf_used > 0) {
            const to_copy = @min(64 - buf_used, input.len);
            @memcpy(self.buf[buf_used .. buf_used + to_copy], input[0..to_copy]);
            self.bytes += to_copy;
            input = input[to_copy..];
            if (self.bytes % 64 == 0) transform(&self.s, &self.buf);
        }

        while (input.len >= 64) {
            var block: [64]u8 = undefined;
            @memcpy(&block, input[0..64]);
            transform(&self.s, &block);
            self.bytes += 64;
            input = input[64..];
        }

        if (input.len > 0) {
            const buf_start: usize = @intCast(self.bytes % 64);
            @memcpy(self.buf[buf_start .. buf_start + input.len], input);
            self.bytes += input.len;
        }
    }

    fn final(self: *RIPEMD160) [20]u8 {
        const msg_len = self.bytes;
        const buf_used: usize = @intCast(msg_len % 64);

        self.buf[buf_used] = 0x80;
        if (buf_used < 56) {
            @memset(self.buf[buf_used + 1 .. 56], 0);
        } else {
            @memset(self.buf[buf_used + 1 .. 64], 0);
            transform(&self.s, &self.buf);
            @memset(self.buf[0..56], 0);
        }

        const bits = msg_len * 8;
        std.mem.writeInt(u64, self.buf[56..][0..8], bits, .little);
        transform(&self.s, &self.buf);

        var result: [20]u8 = undefined;
        for (self.s, 0..) |word, i| {
            std.mem.writeInt(u32, result[i * 4 ..][0..4], word, .little);
        }
        return result;
    }
};

fn f(round_num: u32, x: u32, y: u32, z: u32) u32 {
    return switch (round_num) {
        0 => x ^ y ^ z,
        1 => (x & y) | (~x & z),
        2 => (x | ~y) ^ z,
        3 => (x & z) | (y & ~z),
        4 => x ^ (y | ~z),
        else => unreachable,
    };
}

fn rol(x: u32, n: u5) u32 {
    return std.math.rotl(u32, x, n);
}

const LEFT_X_INDICES = [80]u8{
    0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    7, 4,  13, 1,  10, 6,  15, 3,  12, 0, 9,  5,  2,  14, 11, 8,
    3, 10, 14, 4,  9,  15, 8,  1,  2,  7, 0,  6,  13, 11, 5,  12,
    1, 9,  11, 10, 0,  8,  12, 4,  13, 3, 7,  15, 14, 5,  6,  2,
    4, 0,  5,  9,  7,  12, 2,  10, 14, 1, 3,  8,  11, 6,  15, 13,
};

const RIGHT_X_INDICES = [80]u8{
    5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12,
    6,  11, 3,  7, 0, 13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,
    15, 5,  1,  3, 7, 14, 6,  9,  11, 8,  12, 2,  10, 0,  4,  13,
    8,  6,  4,  1, 3, 11, 15, 0,  5,  12, 2,  13, 9,  7,  10, 14,
    12, 15, 10, 4, 1, 5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11,
};

const LEFT_ROTATIONS = [80]u5{
    11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,
    7,  6,  8,  13, 11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12,
    11, 13, 6,  7,  14, 9,  13, 15, 14, 8,  13, 6,  5,  12, 7,  5,
    11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,  8,  6,  5,  12,
    9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6,
};

const RIGHT_ROTATIONS = [80]u5{
    8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,
    9,  13, 15, 7,  12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11,
    9,  7,  15, 11, 8,  6,  6,  14, 12, 13, 5,  14, 13, 13, 7,  5,
    15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,  12, 5,  15, 8,
    8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11,
};

const ROUND_CONSTANTS = [5]u32{
    0x00000000,
    0x5a827999,
    0x6ed9eba1,
    0x8f1bbcdc,
    0xa953fd4e,
};

const RIGHT_ROUND_CONSTANTS = [5]u32{
    0x50a28be6,
    0x5c4dd124,
    0x6d703ef3,
    0x7a6d76e9,
    0x00000000,
};

fn round(a: *u32, b: *u32, c: *u32, d: *u32, e: *u32, x: u32, s: u5) void {
    a.* = a.* +% x;
    a.* = rol(a.*, s) +% e.*;
    c.* = rol(c.*, 10);
    _ = b;
    _ = d;
}

fn transform(s: *[5]u32, chunk: *const [64]u8) void {
    var x: [16]u32 = undefined;
    for (&x, 0..) |*word, i| {
        word.* = std.mem.readInt(u32, chunk[i * 4 ..][0..4], .little);
    }

    var al = s[0];
    var bl = s[1];
    var cl = s[2];
    var dl = s[3];
    var el = s[4];
    var ar = al;
    var br = bl;
    var cr = cl;
    var dr = dl;
    var er = el;

    var i: usize = 0;
    while (i < 80) : (i += 1) {
        const left_f_idx = i / 16;
        round(
            &al,
            &bl,
            &cl,
            &dl,
            &el,
            f(@intCast(left_f_idx), bl, cl, dl) +% x[LEFT_X_INDICES[i]] +% ROUND_CONSTANTS[left_f_idx],
            LEFT_ROTATIONS[i],
        );
        const temp_l = al;
        al = el;
        el = dl;
        dl = cl;
        cl = bl;
        bl = temp_l;

        const right_f_idx = 4 - (i / 16);
        round(
            &ar,
            &br,
            &cr,
            &dr,
            &er,
            f(@intCast(right_f_idx), br, cr, dr) +% x[RIGHT_X_INDICES[i]] +% RIGHT_ROUND_CONSTANTS[i / 16],
            RIGHT_ROTATIONS[i],
        );
        const temp_r = ar;
        ar = er;
        er = dr;
        dr = cr;
        cr = br;
        br = temp_r;
    }

    const t = s[1] +% cl +% dr;
    s[1] = s[2] +% dl +% er;
    s[2] = s[3] +% el +% ar;
    s[3] = s[4] +% al +% br;
    s[4] = s[0] +% bl +% cr;
    s[0] = t;
}

test "ripemd160 official empty vector" {
    var output: [20]u8 = undefined;
    Ripemd160.hash("", &output);
    const expected = [_]u8{
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54, 0x61, 0x28,
        0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48, 0xb2, 0x25, 0x8d, 0x31,
    };
    try std.testing.expectEqualSlices(u8, &expected, &output);
}

test "ripemd160 official abc vector" {
    var output: [20]u8 = undefined;
    Ripemd160.hash("abc", &output);
    const expected = [_]u8{
        0x8e, 0xb2, 0x08, 0xf7, 0xe0, 0x5d, 0x98, 0x7a, 0x9b, 0x04,
        0x4a, 0x8e, 0x98, 0xc6, 0xb0, 0x87, 0xf1, 0x5a, 0x0b, 0xfc,
    };
    try std.testing.expectEqualSlices(u8, &expected, &output);
}
