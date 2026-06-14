//! Structured ABI fuzzer for dynamic ABI dispatcher inputs.
//!
//! Property: for dynamic-argument public functions, structurally-broken calldata
//! (truncated below the head, out-of-bounds dynamic offset, oversized length)
//! must REVERT — never accept-and-misbehave by returning a value. Valid baselines
//! must SUCCEED (non-vacuity: otherwise "everything reverts" would pass
//! trivially). The gate covers bytes/string, dynamic-array, and dynamic-tuple
//! dispatcher branches.
//!
//! Deterministic: a FIXED seed drives the mutation choices so the gate is
//! reproducible. Mutations are constructed to be definitely-invalid (not random
//! byte flips, which could stay valid), so "must revert" is unambiguous.

const std = @import("std");
const testing = std.testing;
const runner = @import("runner.zig");
const types = @import("types.zig");

const FUZZ_SEED: u64 = 0xA11CE; // fixed for gate reproducibility
const FUZZ_ITERATIONS: usize = 128;

const fuzz_source =
    \\contract FuzzTarget {
    \\    pub fn echo_len(s: string) -> u256 {
    \\        return s.len;
    \\    }
    \\
    \\    pub fn sum(xs: slice[u256]) -> u256 {
    \\        return xs[0] + xs[1] + xs[2];
    \\    }
    \\
    \\    pub fn value(t: (u256, string)) -> u256 {
    \\        return t.0 + t.1.len;
    \\    }
    \\
    \\    pub fn count_flags(values: slice[bool]) -> u256 {
    \\        var total: u256 = 0;
    \\        for (values) |value| {
    \\            if (value) {
    \\                total = total + 1;
    \\            }
    \\        }
    \\        return total;
    \\    }
    \\
    \\    pub fn first_address(values: slice[address]) -> address {
    \\        return values[0];
    \\    }
    \\
    \\    pub fn count_tags(values: slice[bytes4]) -> u256 {
    \\        return values.len;
    \\    }
    \\
    \\    pub fn signed(value: i16) -> i16 {
    \\        return value;
    \\    }
    \\
    \\    pub fn tuple_flag(t: (bool, string)) -> u256 {
    \\        if (t.0) {
    \\            return t.1.len + 1;
    \\        }
    \\        return t.1.len;
    \\    }
    \\
    \\    pub fn tuple_who(t: (address, string)) -> address {
    \\        return t.0;
    \\    }
    \\
    \\    pub fn tuple_tag(t: (bytes4, string)) -> u256 {
    \\        return t.1.len + 2;
    \\    }
    \\
    \\    pub fn tuple_sum(t: (slice[u256], string)) -> u256 {
    \\        return t.0[0] + t.0[1] + t.0[2] + t.1.len;
    \\    }
    \\
    \\    pub fn tuple_count_flags(t: (slice[bool], string)) -> u256 {
    \\        var total: u256 = 0;
    \\        for (t.0) |value| {
    \\            if (value) {
    \\                total = total + 1;
    \\            }
    \\        }
    \\        return total + t.1.len;
    \\    }
    \\}
;

// Selector for echo_len(string) = keccak("echo_len(string)")[0..4].
const ECHO_LEN_SELECTOR = [4]u8{ 0x7b, 0x67, 0xe7, 0x9d };
// Selector for sum(uint256[]) = keccak("sum(uint256[])")[0..4].
const SUM_SELECTOR = [4]u8{ 0x01, 0x94, 0xdb, 0x8e };
// Selector for value((uint256,string)) = keccak("value((uint256,string))")[0..4].
const VALUE_SELECTOR = [4]u8{ 0x1b, 0x4c, 0x5f, 0x62 };
// Selector for count_flags(bool[]) = keccak("count_flags(bool[])")[0..4].
const COUNT_FLAGS_SELECTOR = [4]u8{ 0x37, 0xb1, 0xa2, 0x32 };
// Selector for first_address(address[]) = keccak("first_address(address[])")[0..4].
const FIRST_ADDRESS_SELECTOR = [4]u8{ 0x3d, 0xd1, 0xf7, 0x69 };
// Selector for count_tags(bytes4[]) = keccak("count_tags(bytes4[])")[0..4].
const COUNT_TAGS_SELECTOR = [4]u8{ 0x45, 0xd8, 0xa1, 0xfb };
// Selector for signed(int16) = keccak("signed(int16)")[0..4].
const SIGNED_SELECTOR = [4]u8{ 0x3e, 0x41, 0x85, 0x14 };
// Selector for tuple_flag((bool,string)) = keccak("tuple_flag((bool,string))")[0..4].
const TUPLE_FLAG_SELECTOR = [4]u8{ 0x37, 0x50, 0xf6, 0xe9 };
// Selector for tuple_who((address,string)) = keccak("tuple_who((address,string))")[0..4].
const TUPLE_WHO_SELECTOR = [4]u8{ 0xbd, 0xd0, 0x4e, 0x6a };
// Selector for tuple_tag((bytes4,string)) = keccak("tuple_tag((bytes4,string))")[0..4].
const TUPLE_TAG_SELECTOR = [4]u8{ 0xbb, 0xd2, 0x63, 0xcb };
// Selector for tuple_sum((uint256[],string)) = keccak("tuple_sum((uint256[],string))")[0..4].
const TUPLE_SUM_SELECTOR = [4]u8{ 0x20, 0xf9, 0x60, 0x22 };
// Selector for tuple_count_flags((bool[],string)) = keccak("tuple_count_flags((bool[],string))")[0..4].
const TUPLE_COUNT_FLAGS_SELECTOR = [4]u8{ 0x5b, 0xf2, 0xfb, 0x72 };

const DynamicShape = struct {
    label: []const u8,
    offset_word: usize,
    length_word: usize,
};

fn putWord(buf: []u8, value: u256) void {
    std.mem.writeInt(u256, buf[0..32], value, .big);
}

fn putSignedWord(buf: []u8, value: i256) void {
    const encoded: u256 = @bitCast(value);
    std.mem.writeInt(u256, buf[0..32], encoded, .big);
}

/// Build a well-formed echo_len(string) calldata for a string of `len` bytes.
fn validStringCalldata(allocator: std.mem.Allocator, len: usize) ![]u8 {
    const padded = (len + 31) / 32 * 32;
    const total = 4 + 32 + 32 + padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &ECHO_LEN_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the string
    putWord(buf[36..68], len); // string length
    var i: usize = 0;
    while (i < len) : (i += 1) buf[68 + i] = @intCast('a' + (i % 26));
    return buf;
}

/// Build a well-formed sum(uint256[]) calldata.
fn validU256ArrayCalldata(allocator: std.mem.Allocator, values: []const u256) ![]u8 {
    const total = 4 + 32 + 32 + values.len * 32;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &SUM_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the array
    putWord(buf[36..68], values.len); // array length
    for (values, 0..) |value, i| {
        putWord(buf[68 + i * 32 .. 100 + i * 32], value);
    }
    return buf;
}

/// Build a well-formed value((uint256,string)) calldata.
fn validDynamicTupleCalldata(allocator: std.mem.Allocator, number: u256, text: []const u8) ![]u8 {
    const padded = (text.len + 31) / 32 * 32;
    const total = 4 + 32 + 32 + 32 + 32 + padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &VALUE_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    putWord(buf[36..68], number); // tuple field 0
    putWord(buf[68..100], 0x40); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], text.len); // string length
    @memcpy(buf[132 .. 132 + text.len], text);
    return buf;
}

/// Build a well-formed tuple_flag((bool,string)) calldata.
fn validBoolStringTupleCalldata(allocator: std.mem.Allocator, value: bool, text: []const u8) ![]u8 {
    const padded = (text.len + 31) / 32 * 32;
    const total = 4 + 32 + 32 + 32 + 32 + padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &TUPLE_FLAG_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    putWord(buf[36..68], if (value) 1 else 0); // tuple field 0
    putWord(buf[68..100], 0x40); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], text.len); // string length
    @memcpy(buf[132 .. 132 + text.len], text);
    return buf;
}

/// Build a well-formed tuple_who((address,string)) calldata.
fn validAddressStringTupleCalldata(allocator: std.mem.Allocator, address: [20]u8, text: []const u8) ![]u8 {
    const padded = (text.len + 31) / 32 * 32;
    const total = 4 + 32 + 32 + 32 + 32 + padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &TUPLE_WHO_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    @memcpy(buf[48..68], address[0..]); // tuple field 0
    putWord(buf[68..100], 0x40); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], text.len); // string length
    @memcpy(buf[132 .. 132 + text.len], text);
    return buf;
}

/// Build a well-formed tuple_tag((bytes4,string)) calldata.
fn validBytes4StringTupleCalldata(allocator: std.mem.Allocator, value: [4]u8, text: []const u8) ![]u8 {
    const padded = (text.len + 31) / 32 * 32;
    const total = 4 + 32 + 32 + 32 + 32 + padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &TUPLE_TAG_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    @memcpy(buf[36..40], value[0..]); // tuple field 0, left-aligned
    putWord(buf[68..100], 0x40); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], text.len); // string length
    @memcpy(buf[132 .. 132 + text.len], text);
    return buf;
}

/// Build a well-formed tuple_sum((uint256[],string)) calldata.
fn validU256ArrayStringTupleCalldata(allocator: std.mem.Allocator, values: []const u256, text: []const u8) ![]u8 {
    const text_padded = (text.len + 31) / 32 * 32;
    const array_tail_bytes = 32 + values.len * 32;
    const string_tail_offset = 0x40 + array_tail_bytes;
    const total = 4 + 32 + 32 + 32 + array_tail_bytes + 32 + text_padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &TUPLE_SUM_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    putWord(buf[36..68], 0x40); // tuple field 0 offset, relative to tuple
    putWord(buf[68..100], string_tail_offset); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], values.len); // array length
    for (values, 0..) |value, i| {
        putWord(buf[132 + i * 32 .. 164 + i * 32], value);
    }
    const string_tail = 4 + 32 + string_tail_offset;
    putWord(buf[string_tail .. string_tail + 32], text.len);
    @memcpy(buf[string_tail + 32 .. string_tail + 32 + text.len], text);
    return buf;
}

/// Build a well-formed tuple_count_flags((bool[],string)) calldata.
fn validBoolArrayStringTupleCalldata(allocator: std.mem.Allocator, values: []const bool, text: []const u8) ![]u8 {
    const text_padded = (text.len + 31) / 32 * 32;
    const array_tail_bytes = 32 + values.len * 32;
    const string_tail_offset = 0x40 + array_tail_bytes;
    const total = 4 + 32 + 32 + 32 + array_tail_bytes + 32 + text_padded;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &TUPLE_COUNT_FLAGS_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the tuple
    putWord(buf[36..68], 0x40); // tuple field 0 offset, relative to tuple
    putWord(buf[68..100], string_tail_offset); // tuple field 1 offset, relative to tuple
    putWord(buf[100..132], values.len); // array length
    for (values, 0..) |value, i| {
        putWord(buf[132 + i * 32 .. 164 + i * 32], if (value) 1 else 0);
    }
    const string_tail = 4 + 32 + string_tail_offset;
    putWord(buf[string_tail .. string_tail + 32], text.len);
    @memcpy(buf[string_tail + 32 .. string_tail + 32 + text.len], text);
    return buf;
}

/// Build a well-formed count_flags(bool[]) calldata.
fn validBoolArrayCalldata(allocator: std.mem.Allocator, values: []const bool) ![]u8 {
    const total = 4 + 32 + 32 + values.len * 32;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &COUNT_FLAGS_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the array
    putWord(buf[36..68], values.len); // array length
    for (values, 0..) |value, i| {
        putWord(buf[68 + i * 32 .. 100 + i * 32], if (value) 1 else 0);
    }
    return buf;
}

/// Build a well-formed first_address(address[]) calldata.
fn validAddressArrayCalldata(allocator: std.mem.Allocator, values: []const [20]u8) ![]u8 {
    const total = 4 + 32 + 32 + values.len * 32;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &FIRST_ADDRESS_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the array
    putWord(buf[36..68], values.len); // array length
    for (values, 0..) |value, i| {
        @memcpy(buf[68 + i * 32 + 12 .. 100 + i * 32], value[0..]);
    }
    return buf;
}

/// Build a well-formed count_tags(bytes4[]) calldata.
fn validBytes4ArrayCalldata(allocator: std.mem.Allocator, values: []const [4]u8) ![]u8 {
    const total = 4 + 32 + 32 + values.len * 32;
    var buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], &COUNT_TAGS_SELECTOR);
    putWord(buf[4..36], 0x20); // offset to the array
    putWord(buf[36..68], values.len); // array length
    for (values, 0..) |value, i| {
        @memcpy(buf[68 + i * 32 .. 68 + i * 32 + 4], value[0..]);
    }
    return buf;
}

/// Build a well-formed signed(int16) calldata.
fn validSignedI16Calldata(allocator: std.mem.Allocator, value: i16) ![]u8 {
    var buf = try allocator.alloc(u8, 4 + 32);
    @memset(buf, 0);
    @memcpy(buf[0..4], &SIGNED_SELECTOR);
    putSignedWord(buf[4..36], @as(i256, value));
    return buf;
}

const Mutation = enum { truncate_head, offset_oob, length_oob, length_overflow_word };

const StaticElementMutation = enum { bool_noncanonical, address_high_padding, bytes4_right_padding, int16_bad_sign_extension };

fn applyInvalidMutation(allocator: std.mem.Allocator, base: []const u8, shape: DynamicShape, m: Mutation, rng: *std.Random.DefaultPrng) ![]u8 {
    const r = rng.random();
    switch (m) {
        // Cut somewhere inside the head (selector + offset + length region): the
        // dispatcher must reject calldata too short to hold the declared head.
        .truncate_head => {
            const end = @min(base.len - 1, shape.length_word + 31);
            const cut = r.intRangeAtMost(usize, 4, end);
            return allocator.dupe(u8, base[0..@min(cut, base.len)]);
        },
        // Point the selected dynamic offset far past the end of calldata. For
        // tuples this is the nested dynamic field offset, not the top-level arg
        // offset, so the fuzzer reaches the tuple-tail decoder too.
        .offset_oob => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[shape.offset_word .. shape.offset_word + 32], 0x10000 + r.intRangeAtMost(u256, 0, 0xffff));
            return buf;
        },
        // Claim a length larger than the bytes actually present.
        .length_oob => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[shape.length_word .. shape.length_word + 32], @as(u256, base.len) + 0x1000 + r.intRangeAtMost(u256, 0, 0xffff));
            return buf;
        },
        // Enormous length that would overflow offset+length arithmetic.
        .length_overflow_word => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[shape.length_word .. shape.length_word + 32], std.math.maxInt(u256) - r.intRangeAtMost(u256, 0, 0xff));
            return buf;
        },
    }
}

fn expectU256Return(result: types.Evm.CallResult, expected: u256) !void {
    if (!result.success) return error.FuzzBaselineRejected;
    if (result.output.len != 32) return error.FuzzBaselineBadReturn;
    if (std.mem.readInt(u256, result.output[0..32], .big) != expected) return error.FuzzBaselineWrongValue;
}

fn rejectInvalidMutations(runtime: *runner.PropertyRuntime, base: []const u8, shape: DynamicShape, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const muts = [_]Mutation{ .truncate_head, .offset_oob, .length_oob, .length_overflow_word };
    var i: usize = 0;
    while (i < FUZZ_ITERATIONS) : (i += 1) {
        const m = muts[i % muts.len];
        const mutated = try applyInvalidMutation(runtime.allocator, base, shape, m, &prng);
        const result = try runtime.callRaw(mutated);
        if (result.success) {
            std.debug.print("FUZZ: malformed {s} calldata ACCEPTED (mutation {s}, {d} bytes)\n", .{ shape.label, @tagName(m), mutated.len });
            return error.FuzzAcceptedMalformedCalldata;
        }
    }
}

fn applyInvalidStaticElementMutation(
    allocator: std.mem.Allocator,
    base: []const u8,
    element_word_base: usize,
    element_count: usize,
    kind: StaticElementMutation,
    rng: *std.Random.DefaultPrng,
) ![]u8 {
    const r = rng.random();
    const element_index = r.intRangeLessThan(usize, 0, element_count);
    const element_word = element_word_base + element_index * 32;
    return try applyInvalidStaticWordMutation(allocator, base, element_word, kind, rng);
}

fn applyInvalidStaticWordMutation(
    allocator: std.mem.Allocator,
    base: []const u8,
    word_offset: usize,
    kind: StaticElementMutation,
    rng: *std.Random.DefaultPrng,
) ![]u8 {
    const r = rng.random();
    var buf = try allocator.dupe(u8, base);
    switch (kind) {
        .bool_noncanonical => putWord(buf[word_offset .. word_offset + 32], 2 + r.intRangeAtMost(u256, 0, 0xff)),
        .address_high_padding => buf[word_offset + r.intRangeLessThan(usize, 0, 12)] = 1,
        .bytes4_right_padding => buf[word_offset + r.intRangeLessThan(usize, 4, 32)] = 1,
        .int16_bad_sign_extension => {
            @memset(buf[word_offset .. word_offset + 32], 0);
            buf[word_offset + 30] = 0xff;
            buf[word_offset + 31] = 0xf9;
        },
    }
    return buf;
}

fn rejectInvalidStaticElementMutations(
    runtime: *runner.PropertyRuntime,
    base: []const u8,
    label: []const u8,
    element_word_base: usize,
    element_count: usize,
    kind: StaticElementMutation,
    seed: u64,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    var i: usize = 0;
    while (i < FUZZ_ITERATIONS) : (i += 1) {
        const mutated = try applyInvalidStaticElementMutation(runtime.allocator, base, element_word_base, element_count, kind, &prng);
        const result = try runtime.callRaw(mutated);
        if (result.success) {
            std.debug.print("FUZZ: non-canonical {s} element calldata ACCEPTED ({d} bytes)\n", .{ label, mutated.len });
            return error.FuzzAcceptedMalformedCalldata;
        }
    }
}

fn rejectInvalidStaticWordMutations(
    runtime: *runner.PropertyRuntime,
    base: []const u8,
    label: []const u8,
    word_offset: usize,
    kind: StaticElementMutation,
    seed: u64,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    var i: usize = 0;
    while (i < FUZZ_ITERATIONS) : (i += 1) {
        const mutated = try applyInvalidStaticWordMutation(runtime.allocator, base, word_offset, kind, &prng);
        const result = try runtime.callRaw(mutated);
        if (result.success) {
            std.debug.print("FUZZ: non-canonical {s} word calldata ACCEPTED ({d} bytes)\n", .{ label, mutated.len });
            return error.FuzzAcceptedMalformedCalldata;
        }
    }
}

fn addressToU256(address: [20]u8) u256 {
    var word: [32]u8 = .{0} ** 32;
    @memcpy(word[12..32], address[0..]);
    return std.mem.readInt(u256, &word, .big);
}

fn runFuzz(runtime: *runner.PropertyRuntime) !void {
    const arena = runtime.allocator;

    // Non-vacuity self-tests: valid baselines MUST succeed and return the
    // expected values before malformed variants are meaningful.
    const string_base = try validStringCalldata(arena, 5);
    try expectU256Return(try runtime.callRaw(string_base), 5);
    try rejectInvalidMutations(runtime, string_base, .{ .label = "string", .offset_word = 4, .length_word = 36 }, FUZZ_SEED);

    const array_base = try validU256ArrayCalldata(arena, &.{ 5, 8, 13 });
    try expectU256Return(try runtime.callRaw(array_base), 26);
    try rejectInvalidMutations(runtime, array_base, .{ .label = "uint256[]", .offset_word = 4, .length_word = 36 }, FUZZ_SEED ^ 0x5a5a5a5a);

    const tuple_base = try validDynamicTupleCalldata(arena, 7, "ora");
    try expectU256Return(try runtime.callRaw(tuple_base), 10);
    try rejectInvalidMutations(runtime, tuple_base, .{ .label = "(uint256,string)", .offset_word = 68, .length_word = 100 }, FUZZ_SEED ^ 0x7d7d7d7d);

    const bool_array_base = try validBoolArrayCalldata(arena, &.{ true, false, true });
    try expectU256Return(try runtime.callRaw(bool_array_base), 2);
    try rejectInvalidStaticElementMutations(runtime, bool_array_base, "bool[]", 68, 3, .bool_noncanonical, FUZZ_SEED ^ 0xb001);

    const addr0 = [20]u8{ 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    const addr1 = [20]u8{ 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 };
    const address_array_base = try validAddressArrayCalldata(arena, &.{ addr0, addr1 });
    try expectU256Return(try runtime.callRaw(address_array_base), addressToU256(addr0));
    try rejectInvalidStaticElementMutations(runtime, address_array_base, "address[]", 68, 2, .address_high_padding, FUZZ_SEED ^ 0xadad);

    const bytes4_array_base = try validBytes4ArrayCalldata(arena, &.{ .{ 0x11, 0x22, 0x33, 0x44 }, .{ 0xaa, 0xbb, 0xcc, 0xdd } });
    try expectU256Return(try runtime.callRaw(bytes4_array_base), 2);
    try rejectInvalidStaticElementMutations(runtime, bytes4_array_base, "bytes4[]", 68, 2, .bytes4_right_padding, FUZZ_SEED ^ 0xb4b4);

    const signed_base = try validSignedI16Calldata(arena, -7);
    try expectU256Return(try runtime.callRaw(signed_base), @bitCast(@as(i256, -7)));
    try rejectInvalidStaticWordMutations(runtime, signed_base, "int16", 4, .int16_bad_sign_extension, FUZZ_SEED ^ 0x516e);

    const bool_tuple_base = try validBoolStringTupleCalldata(arena, true, "ora");
    try expectU256Return(try runtime.callRaw(bool_tuple_base), 4);
    try rejectInvalidStaticWordMutations(runtime, bool_tuple_base, "(bool,string).0", 36, .bool_noncanonical, FUZZ_SEED ^ 0x7175);

    const address_tuple_base = try validAddressStringTupleCalldata(arena, addr0, "ora");
    try expectU256Return(try runtime.callRaw(address_tuple_base), addressToU256(addr0));
    try rejectInvalidStaticWordMutations(runtime, address_tuple_base, "(address,string).0", 36, .address_high_padding, FUZZ_SEED ^ 0xadd5);

    const bytes4_tuple_base = try validBytes4StringTupleCalldata(arena, .{ 0x11, 0x22, 0x33, 0x44 }, "ora");
    try expectU256Return(try runtime.callRaw(bytes4_tuple_base), 5);
    try rejectInvalidStaticWordMutations(runtime, bytes4_tuple_base, "(bytes4,string).0", 36, .bytes4_right_padding, FUZZ_SEED ^ 0xb475);

    const array_string_tuple_base = try validU256ArrayStringTupleCalldata(arena, &.{ 5, 8, 13 }, "ora");
    try expectU256Return(try runtime.callRaw(array_string_tuple_base), 29);
    try rejectInvalidMutations(runtime, array_string_tuple_base, .{ .label = "(uint256[],string).0", .offset_word = 36, .length_word = 100 }, FUZZ_SEED ^ 0xa775);
    try rejectInvalidMutations(runtime, array_string_tuple_base, .{ .label = "(uint256[],string).1", .offset_word = 68, .length_word = 228 }, FUZZ_SEED ^ 0x5712);

    const bool_array_string_tuple_base = try validBoolArrayStringTupleCalldata(arena, &.{ true, false, true }, "ora");
    try expectU256Return(try runtime.callRaw(bool_array_string_tuple_base), 5);
    try rejectInvalidStaticElementMutations(runtime, bool_array_string_tuple_base, "(bool[],string).0", 132, 3, .bool_noncanonical, FUZZ_SEED ^ 0xb005);
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_fuzz", fuzz_source, runFuzz);
}

// Teeth self-test (D5): feed a VALID calldata through the must-revert check and
// confirm the acceptance-detection path fires. If this did NOT error, the
// fuzzer's "malformed must revert" property would be unable to detect a real
// accept-and-misbehave bug. The caller asserts FuzzAcceptedMalformedCalldata.
fn runFuzzTeeth(runtime: *runner.PropertyRuntime) !void {
    const base = try validStringCalldata(runtime.allocator, 5);
    const result = try runtime.callRaw(base);
    // A valid call succeeds; the malformed-check must treat that as a failure.
    if (result.success) return error.FuzzAcceptedMalformedCalldata;
    return error.FuzzTeethHarnessFailed;
}

pub fn runTeeth(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_fuzz_teeth", fuzz_source, runFuzzTeeth);
}
