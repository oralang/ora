//! Structured ABI fuzzer (test-quality program T2.3).
//!
//! Property: for a dynamic-argument public function, structurally-broken calldata
//! (truncated below the head, out-of-bounds dynamic offset, oversized length)
//! must REVERT — never accept-and-misbehave by returning a value. The valid
//! baseline must SUCCEED (non-vacuity: otherwise "everything reverts" would pass
//! trivially).
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
    \\}
;

// Selector for echo_len(string) = keccak("echo_len(string)")[0..4].
const ECHO_LEN_SELECTOR = [4]u8{ 0x7b, 0x67, 0xe7, 0x9d };

fn putWord(buf: []u8, value: u256) void {
    std.mem.writeInt(u256, buf[0..32], value, .big);
}

/// Build a well-formed echo_len(string) calldata for a string of `len` bytes.
fn validCalldata(allocator: std.mem.Allocator, len: usize) ![]u8 {
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

const Mutation = enum { truncate_head, offset_oob, length_oob, length_overflow_word };

fn applyInvalidMutation(allocator: std.mem.Allocator, base: []const u8, m: Mutation, rng: *std.Random.DefaultPrng) ![]u8 {
    const r = rng.random();
    switch (m) {
        // Cut somewhere inside the head (selector + offset + length region): the
        // dispatcher must reject calldata too short to hold the declared head.
        .truncate_head => {
            const cut = 4 + r.intRangeAtMost(usize, 0, 63); // 4..67 bytes (< full head of 68)
            return allocator.dupe(u8, base[0..@min(cut, base.len)]);
        },
        // Point the dynamic offset far past the end of calldata.
        .offset_oob => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[4..36], 0x10000 + r.intRangeAtMost(u256, 0, 0xffff));
            return buf;
        },
        // Claim a length larger than the bytes actually present.
        .length_oob => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[36..68], @as(u256, base.len) + 0x1000 + r.intRangeAtMost(u256, 0, 0xffff));
            return buf;
        },
        // Enormous length that would overflow offset+length arithmetic.
        .length_overflow_word => {
            var buf = try allocator.dupe(u8, base);
            putWord(buf[36..68], std.math.maxInt(u256) - r.intRangeAtMost(u256, 0, 0xff));
            return buf;
        },
    }
}

fn runFuzz(runtime: *runner.PropertyRuntime) !void {
    const arena = runtime.allocator;

    // Non-vacuity self-test: the valid baseline MUST succeed and return len.
    const base = try validCalldata(arena, 5);
    const base_result = try runtime.callRaw(base);
    if (!base_result.success) return error.FuzzBaselineRejected;
    if (base_result.output.len != 32) return error.FuzzBaselineBadReturn;
    if (std.mem.readInt(u256, base_result.output[0..32], .big) != 5) return error.FuzzBaselineWrongLen;

    // Every structurally-broken mutation MUST revert.
    var prng = std.Random.DefaultPrng.init(FUZZ_SEED);
    const muts = [_]Mutation{ .truncate_head, .offset_oob, .length_oob, .length_overflow_word };
    var i: usize = 0;
    while (i < FUZZ_ITERATIONS) : (i += 1) {
        const m = muts[i % muts.len];
        const mutated = try applyInvalidMutation(arena, base, m, &prng);
        const result = try runtime.callRaw(mutated);
        if (result.success) {
            std.debug.print("FUZZ: malformed calldata ACCEPTED (mutation {s}, {d} bytes)\n", .{ @tagName(m), mutated.len });
            return error.FuzzAcceptedMalformedCalldata;
        }
    }
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_fuzz", fuzz_source, runFuzz);
}

// Teeth self-test (D5): feed a VALID calldata through the must-revert check and
// confirm the acceptance-detection path fires. If this did NOT error, the
// fuzzer's "malformed must revert" property would be unable to detect a real
// accept-and-misbehave bug. The caller asserts FuzzAcceptedMalformedCalldata.
fn runFuzzTeeth(runtime: *runner.PropertyRuntime) !void {
    const base = try validCalldata(runtime.allocator, 5);
    const result = try runtime.callRaw(base);
    // A valid call succeeds; the malformed-check must treat that as a failure.
    if (result.success) return error.FuzzAcceptedMalformedCalldata;
    return error.FuzzTeethHarnessFailed;
}

pub fn runTeeth(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_fuzz_teeth", fuzz_source, runFuzzTeeth);
}
