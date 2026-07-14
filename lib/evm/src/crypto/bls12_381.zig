const std = @import("std");
const c_kzg = @import("c_kzg");

const blst = c_kzg.blst;

extern fn ora_bls12_381_g1_add(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_g1_mul(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_g1_msm(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_g2_add(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_g2_mul(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_g2_msm(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;
extern fn ora_bls12_381_pairing(input: [*]const u8, input_len: u32, output: [*]u8, output_len: u32) c_int;

pub const Error = error{
    InvalidInput,
    InvalidPoint,
    InvalidScalar,
    ComputationFailed,
} || std.mem.Allocator.Error;

const FP_MODULUS = [_]u8{
    0x1a, 0x01, 0x11, 0xea, 0x39, 0x7f, 0xe6, 0x9a,
    0x4b, 0x1b, 0xa7, 0xb6, 0x43, 0x4b, 0xac, 0xd7,
    0x64, 0x77, 0x4b, 0x84, 0xf3, 0x85, 0x12, 0xbf,
    0x67, 0x30, 0xd2, 0xa0, 0xf6, 0xb0, 0xf6, 0x24,
    0x1e, 0xab, 0xff, 0xfe, 0xb1, 0x53, 0xff, 0xff,
    0xb9, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xaa, 0xab,
};

const SCALAR_MODULUS = [_]u8{
    0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
    0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
    0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
};

fn resultToError(result: c_int) Error!void {
    return switch (result) {
        0 => {},
        1 => error.InvalidInput,
        2 => error.InvalidPoint,
        3 => error.InvalidScalar,
        else => error.ComputationFailed,
    };
}

fn allZero(bytes: []const u8) bool {
    for (bytes) |byte| {
        if (byte != 0) return false;
    }
    return true;
}

fn compactFieldFromWord(word: []const u8, out: []u8) Error!void {
    if (word.len != 64 or out.len != 48) return error.InvalidInput;
    if (!allZero(word[0..16])) return error.InvalidInput;
    const field = word[16..64];
    if (std.mem.order(u8, field, &FP_MODULUS) != .lt) return error.InvalidInput;
    @memcpy(out, field);
}

fn compactScalarFromWord(word: []const u8, out: []u8) Error!void {
    if (word.len != 32 or out.len != 32) return error.InvalidInput;
    if (std.mem.order(u8, word, &SCALAR_MODULUS) != .lt) return error.InvalidScalar;
    @memcpy(out, word);
}

fn expandFieldToWord(field: []const u8, out: []u8) Error!void {
    if (field.len != 48 or out.len != 64) return error.InvalidInput;
    @memset(out[0..16], 0);
    @memcpy(out[16..64], field);
}

fn compactG1FromWords(words: []const u8, out: []u8) Error!void {
    if (words.len != 128 or out.len != 96) return error.InvalidInput;
    try compactFieldFromWord(words[0..64], out[0..48]);
    try compactFieldFromWord(words[64..128], out[48..96]);
    try validateG1Compact(out);
}

fn compactG2FromWords(words: []const u8, out: []u8) Error!void {
    if (words.len != 256 or out.len != 192) return error.InvalidInput;
    try compactFieldFromWord(words[0..64], out[0..48]);
    try compactFieldFromWord(words[64..128], out[48..96]);
    try compactFieldFromWord(words[128..192], out[96..144]);
    try compactFieldFromWord(words[192..256], out[144..192]);
    try validateG2Compact(out);
}

fn expandG1ToWords(compact: []const u8, out: []u8) Error!void {
    if (compact.len != 96 or out.len != 128) return error.InvalidInput;
    try expandFieldToWord(compact[0..48], out[0..64]);
    try expandFieldToWord(compact[48..96], out[64..128]);
}

fn expandG2ToWords(compact: []const u8, out: []u8) Error!void {
    if (compact.len != 192 or out.len != 256) return error.InvalidInput;
    try expandFieldToWord(compact[0..48], out[0..64]);
    try expandFieldToWord(compact[48..96], out[64..128]);
    try expandFieldToWord(compact[96..144], out[128..192]);
    try expandFieldToWord(compact[144..192], out[192..256]);
}

fn validateG1Compact(compact: []const u8) Error!void {
    if (compact.len != 96) return error.InvalidInput;
    if (allZero(compact)) return;

    var point: blst.blst_p1_affine = undefined;
    blst.blst_fp_from_bendian(&point.x, compact[0..48].ptr);
    blst.blst_fp_from_bendian(&point.y, compact[48..96].ptr);
    if (!blst.blst_p1_affine_on_curve(&point)) return error.InvalidPoint;
    if (!blst.blst_p1_affine_in_g1(&point)) return error.InvalidPoint;
}

fn validateG2Compact(compact: []const u8) Error!void {
    if (compact.len != 192) return error.InvalidInput;
    if (allZero(compact)) return;

    var point: blst.blst_p2_affine = undefined;
    blst.blst_fp_from_bendian(&point.x.fp[0], compact[0..48].ptr);
    blst.blst_fp_from_bendian(&point.x.fp[1], compact[48..96].ptr);
    blst.blst_fp_from_bendian(&point.y.fp[0], compact[96..144].ptr);
    blst.blst_fp_from_bendian(&point.y.fp[1], compact[144..192].ptr);
    if (!blst.blst_p2_affine_on_curve(&point)) return error.InvalidPoint;
    if (!blst.blst_p2_affine_in_g2(&point)) return error.InvalidPoint;
}

pub fn g1Add(input: []const u8, output: []u8) Error!void {
    if (input.len != 256 or output.len != 128) return error.InvalidInput;
    var compact_input: [192]u8 = undefined;
    var compact_output: [96]u8 = undefined;
    try compactG1FromWords(input[0..128], compact_input[0..96]);
    try compactG1FromWords(input[128..256], compact_input[96..192]);
    try resultToError(ora_bls12_381_g1_add(&compact_input, compact_input.len, &compact_output, compact_output.len));
    try expandG1ToWords(&compact_output, output);
}

pub fn g1Mul(input: []const u8, output: []u8) Error!void {
    if (input.len != 160 or output.len != 128) return error.InvalidInput;
    var compact_input: [128]u8 = undefined;
    var compact_output: [96]u8 = undefined;
    try compactG1FromWords(input[0..128], compact_input[0..96]);
    try compactScalarFromWord(input[128..160], compact_input[96..128]);
    try resultToError(ora_bls12_381_g1_mul(&compact_input, compact_input.len, &compact_output, compact_output.len));
    try expandG1ToWords(&compact_output, output);
}

pub fn g1Msm(allocator: std.mem.Allocator, input: []const u8, output: []u8) Error!void {
    if (input.len == 0 or input.len % 160 != 0 or output.len != 128) return error.InvalidInput;
    const count = input.len / 160;
    const compact_input = try allocator.alloc(u8, count * 128);
    defer allocator.free(compact_input);
    var compact_output: [96]u8 = undefined;

    for (0..count) |i| {
        const src = i * 160;
        const dst = i * 128;
        try compactG1FromWords(input[src .. src + 128], compact_input[dst .. dst + 96]);
        try compactScalarFromWord(input[src + 128 .. src + 160], compact_input[dst + 96 .. dst + 128]);
    }

    try resultToError(ora_bls12_381_g1_msm(compact_input.ptr, @intCast(compact_input.len), &compact_output, compact_output.len));
    try expandG1ToWords(&compact_output, output);
}

pub fn g2Add(input: []const u8, output: []u8) Error!void {
    if (input.len != 512 or output.len != 256) return error.InvalidInput;
    var compact_input: [384]u8 = undefined;
    var compact_output: [192]u8 = undefined;
    try compactG2FromWords(input[0..256], compact_input[0..192]);
    try compactG2FromWords(input[256..512], compact_input[192..384]);
    try resultToError(ora_bls12_381_g2_add(&compact_input, compact_input.len, &compact_output, compact_output.len));
    try expandG2ToWords(&compact_output, output);
}

pub fn g2Mul(input: []const u8, output: []u8) Error!void {
    if (input.len != 288 or output.len != 256) return error.InvalidInput;
    var compact_input: [224]u8 = undefined;
    var compact_output: [192]u8 = undefined;
    try compactG2FromWords(input[0..256], compact_input[0..192]);
    try compactScalarFromWord(input[256..288], compact_input[192..224]);
    try resultToError(ora_bls12_381_g2_mul(&compact_input, compact_input.len, &compact_output, compact_output.len));
    try expandG2ToWords(&compact_output, output);
}

pub fn g2Msm(allocator: std.mem.Allocator, input: []const u8, output: []u8) Error!void {
    if (input.len == 0 or input.len % 288 != 0 or output.len != 256) return error.InvalidInput;
    const count = input.len / 288;
    const compact_input = try allocator.alloc(u8, count * 224);
    defer allocator.free(compact_input);
    var compact_output: [192]u8 = undefined;

    for (0..count) |i| {
        const src = i * 288;
        const dst = i * 224;
        try compactG2FromWords(input[src .. src + 256], compact_input[dst .. dst + 192]);
        try compactScalarFromWord(input[src + 256 .. src + 288], compact_input[dst + 192 .. dst + 224]);
    }

    try resultToError(ora_bls12_381_g2_msm(compact_input.ptr, @intCast(compact_input.len), &compact_output, compact_output.len));
    try expandG2ToWords(&compact_output, output);
}

pub fn pairing(allocator: std.mem.Allocator, input: []const u8, output: []u8) Error!void {
    if (input.len % 384 != 0 or output.len != 32) return error.InvalidInput;
    const count = input.len / 384;
    const compact_input = try allocator.alloc(u8, count * 288);
    defer allocator.free(compact_input);

    for (0..count) |i| {
        const src = i * 384;
        const dst = i * 288;
        try compactG1FromWords(input[src .. src + 128], compact_input[dst .. dst + 96]);
        try compactG2FromWords(input[src + 128 .. src + 384], compact_input[dst + 96 .. dst + 288]);
    }

    try resultToError(ora_bls12_381_pairing(compact_input.ptr, @intCast(compact_input.len), output.ptr, @intCast(output.len)));
}

pub fn mapFpToG1(input: []const u8, output: []u8) Error!void {
    if (input.len != 64 or output.len != 128) return error.InvalidInput;
    var field: [48]u8 = undefined;
    var compact_output: [96]u8 = undefined;
    try compactFieldFromWord(input, &field);

    var fp: blst.blst_fp = undefined;
    blst.blst_fp_from_bendian(&fp, &field);

    var p1: blst.blst_p1 = undefined;
    blst.blst_map_to_g1(&p1, &fp, null);

    var affine: blst.blst_p1_affine = undefined;
    blst.blst_p1_to_affine(&affine, &p1);
    blst.blst_bendian_from_fp(compact_output[0..48].ptr, &affine.x);
    blst.blst_bendian_from_fp(compact_output[48..96].ptr, &affine.y);
    try expandG1ToWords(&compact_output, output);
}

pub fn mapFp2ToG2(input: []const u8, output: []u8) Error!void {
    if (input.len != 128 or output.len != 256) return error.InvalidInput;
    var c0: [48]u8 = undefined;
    var c1: [48]u8 = undefined;
    var compact_output: [192]u8 = undefined;
    try compactFieldFromWord(input[0..64], &c0);
    try compactFieldFromWord(input[64..128], &c1);

    var fp2: blst.blst_fp2 = undefined;
    blst.blst_fp_from_bendian(&fp2.fp[0], &c0);
    blst.blst_fp_from_bendian(&fp2.fp[1], &c1);

    var p2: blst.blst_p2 = undefined;
    blst.blst_map_to_g2(&p2, &fp2, null);

    var affine: blst.blst_p2_affine = undefined;
    blst.blst_p2_to_affine(&affine, &p2);
    blst.blst_bendian_from_fp(compact_output[0..48].ptr, &affine.x.fp[0]);
    blst.blst_bendian_from_fp(compact_output[48..96].ptr, &affine.x.fp[1]);
    blst.blst_bendian_from_fp(compact_output[96..144].ptr, &affine.y.fp[0]);
    blst.blst_bendian_from_fp(compact_output[144..192].ptr, &affine.y.fp[1]);
    try expandG2ToWords(&compact_output, output);
}

test "BLS field words reject noncanonical high bytes" {
    var word = [_]u8{0} ** 64;
    var out: [48]u8 = undefined;
    word[0] = 1;
    try std.testing.expectError(error.InvalidInput, compactFieldFromWord(&word, &out));
}

test "BLS scalar words reject values outside Fr" {
    var out: [32]u8 = undefined;
    try std.testing.expectError(error.InvalidScalar, compactScalarFromWord(&SCALAR_MODULUS, &out));
}
