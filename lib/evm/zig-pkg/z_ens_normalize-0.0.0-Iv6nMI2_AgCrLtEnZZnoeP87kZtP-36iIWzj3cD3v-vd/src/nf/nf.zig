const std = @import("std");
const Allocator = std.mem.Allocator;
const Decoder = @import("../util/decoder.zig").Decoder;
const RuneSet = @import("../util/runeset.zig").RuneSet;

/// Embed the compressed normalization data
const compressed = @embedFile("nf.bin");

// Packing constants for combining class and codepoint into single value
// Format: [CC: 8 bits][CP: 24 bits]
const SHIFT: u5 = 24;
const MASK: u32 = (1 << SHIFT) - 1;
const NONE: i32 = -1;

// Hangul syllable constants for algorithmic decomposition/composition
const S0: u21 = 0xAC00; // First Hangul syllable
const L0: u21 = 0x1100; // First leading consonant (Choseong)
const V0: u21 = 0x1161; // First vowel (Jungseong)
const T0: u21 = 0x11A7; // First trailing consonant (Jongseong)
const L_COUNT: u21 = 19; // Number of leading consonants
const V_COUNT: u21 = 21; // Number of vowels
const T_COUNT: u21 = 28; // Number of trailing consonants
const N_COUNT: u21 = V_COUNT * T_COUNT; // 588
const S_COUNT: u21 = L_COUNT * N_COUNT; // 11172 - total Hangul syllables
const S1: u21 = S0 + S_COUNT; // One past last Hangul syllable
const L1: u21 = L0 + L_COUNT;
const V1: u21 = V0 + V_COUNT;
const T1: u21 = T0 + T_COUNT;

// Helper function: Check if codepoint is a Hangul syllable
fn isHangul(cp: u21) bool {
    return cp >= S0 and cp < S1;
}

// Helper function: Extract combining class from packed value
fn unpackCC(packed_val: i32) u8 {
    const upacked: u32 = @bitCast(packed_val);
    return @intCast((upacked >> SHIFT) & 0xFF);
}

// Helper function: Extract codepoint from packed value
fn unpackCP(packed_val: i32) u21 {
    const upacked: u32 = @bitCast(packed_val);
    return @intCast(upacked & MASK);
}

// NF struct: Holds all Unicode normalization data
pub const NF = struct {
    unicodeVersion: []const u8,
    exclusions: RuneSet,
    quickCheck: RuneSet,
    decomps: std.AutoHashMap(u21, []u21),
    recomps: std.AutoHashMap(u21, std.AutoHashMap(u21, u21)),
    ranks: std.AutoHashMap(u21, u8),

    /// Initialize NF by decoding the embedded nf.bin data
    /// This loads all normalization tables needed for NFC/NFD operations
    pub fn init(allocator: Allocator) !NF {
        // Initialize decoder from embedded binary data
        var decoder = try Decoder.init(compressed, allocator);

        // Read unicode version string
        const unicodeVersion = try decoder.ReadString(allocator);
        errdefer allocator.free(unicodeVersion);

        // Read exclusions set
        const exclusions_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(exclusions_ints);
        const exclusions = try RuneSet.fromInts(allocator, exclusions_ints);
        errdefer exclusions.deinit(allocator);

        // Read quickCheck set
        const quickcheck_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(quickcheck_ints);
        const quickCheck = try RuneSet.fromInts(allocator, quickcheck_ints);
        errdefer quickCheck.deinit(allocator);

        // Initialize maps
        var decomps = std.AutoHashMap(u21, []u21).init(allocator);
        errdefer {
            var decomps_iter = decomps.iterator();
            while (decomps_iter.next()) |entry| {
                allocator.free(entry.value_ptr.*);
            }
            decomps.deinit();
        }

        var recomps = std.AutoHashMap(u21, std.AutoHashMap(u21, u21)).init(allocator);
        errdefer {
            var recomps_iter = recomps.iterator();
            while (recomps_iter.next()) |entry| {
                entry.value_ptr.deinit();
            }
            recomps.deinit();
        }

        var ranks = std.AutoHashMap(u21, u8).init(allocator);
        errdefer ranks.deinit();

        // Phase 1 - Read 1-character decompositions
        const decomp1 = try decoder.ReadSortedUnique(allocator);
        defer allocator.free(decomp1);
        const decomp1A = try decoder.ReadUnsortedDeltas(@intCast(decomp1.len), allocator);
        defer allocator.free(decomp1A);

        for (decomp1, decomp1A) |cp, target| {
            const decomp = try allocator.alloc(u21, 1);
            decomp[0] = @intCast(target);
            try decomps.put(@intCast(cp), decomp);
        }

        // Phase 2 - Read 2-character decompositions
        const decomp2 = try decoder.ReadSortedUnique(allocator);
        defer allocator.free(decomp2);
        const decomp2A = try decoder.ReadUnsortedDeltas(@intCast(decomp2.len), allocator);
        defer allocator.free(decomp2A);
        const decomp2B = try decoder.ReadUnsortedDeltas(@intCast(decomp2.len), allocator);
        defer allocator.free(decomp2B);

        for (decomp2, decomp2A, decomp2B) |cp, targetA, targetB| {
            const cp_u21: u21 = @intCast(cp);
            const cpA: u21 = @intCast(targetA);
            const cpB: u21 = @intCast(targetB);

            // Build decomps map: cp -> [cpB, cpA] (Note: B comes first!)
            const decomp = try allocator.alloc(u21, 2);
            decomp[0] = cpB;
            decomp[1] = cpA;
            try decomps.put(cp_u21, decomp);

            // Build recomps map (only if not excluded)
            if (!exclusions.contains(cp_u21)) {
                // Get or create inner map for cpA
                var entry = try recomps.getOrPut(cpA);
                if (!entry.found_existing) {
                    entry.value_ptr.* = std.AutoHashMap(u21, u21).init(allocator);
                }
                // Map cpB -> cp in the inner map
                try entry.value_ptr.put(cpB, cp_u21);
            }
        }

        // Read ranks data (infinite loop until empty array)
        var rank_value: u8 = 1;
        while (true) : (rank_value += 1) {
            const cps = try decoder.ReadUnique(allocator);
            defer allocator.free(cps);

            if (cps.len == 0) break;

            for (cps) |cp| {
                try ranks.put(@intCast(cp), rank_value);
            }
        }

        // Assert we've consumed all data
        decoder.assertEOF();

        // Clean up decoder's allocated memory
        decoder.deinit(allocator);

        return NF{
            .unicodeVersion = unicodeVersion,
            .exclusions = exclusions,
            .quickCheck = quickCheck,
            .decomps = decomps,
            .recomps = recomps,
            .ranks = ranks,
        };
    }

    // Packer struct: Accumulates decomposed codepoints with combining classes
    const Packer = struct {
        nf: *const NF,
        buf: std.ArrayListUnmanaged(i32),
        check: bool,

        // Add codepoint to buffer, packing with combining class if present
        fn add(self: *Packer, allocator: Allocator, cp: u21) !void {
            var packed_val: i32 = @bitCast(@as(u32, cp));

            if (self.nf.ranks.get(cp)) |cc| {
                self.check = true;
                packed_val = @bitCast(@as(u32, cp) | (@as(u32, cc) << SHIFT));
            }

            try self.buf.append(allocator, packed_val);
        }

        // Reorder codepoints by combining class (canonical ordering)
        fn fixOrder(self: *Packer) void {
            if (!self.check) return;

            const v = self.buf.items;
            if (v.len == 0) return;

            var prev = unpackCC(v[0]);
            var i: usize = 1;

            while (i < v.len) : (i += 1) {
                const cc = unpackCC(v[i]);
                if (cc == 0 or prev <= cc) {
                    prev = cc;
                    continue;
                }

                var j = i - 1;
                while (true) {
                    // Swap v[j+1] and v[j]
                    const temp = v[j + 1];
                    v[j + 1] = v[j];
                    v[j] = temp;

                    if (j == 0) break;
                    j -= 1;

                    prev = unpackCC(v[j]);
                    if (prev <= cc) break;
                }
                prev = unpackCC(v[i]);
            }
        }
    };

    // Attempt to compose two codepoints into a single codepoint
    // Handles Hangul algorithmic composition and table-based composition
    fn composePair(self: *const NF, a: u21, b: u21) i32 {
        // Hangul LV composition: L + V -> LV syllable
        if (a >= L0 and a < L1 and b >= V0 and b < V1) {
            return @intCast(S0 + (a - L0) * N_COUNT + (b - V0) * T_COUNT);
        }

        // Hangul LVT composition: LV + T -> LVT syllable
        if (isHangul(a) and b > T0 and b < T1 and (a - S0) % T_COUNT == 0) {
            return @intCast(a + (b - T0));
        }

        // Table-based composition
        if (self.recomps.get(a)) |recomp_map| {
            if (recomp_map.get(b)) |cp| {
                return @intCast(cp);
            }
        }

        return NONE;
    }

    // Recursively decompose codepoints with Hangul special handling
    // Returns packed values (CP + CC in single i32)
    fn decomposed(self: *const NF, allocator: Allocator, cps: []const u21) ![]i32 {
        var p = Packer{
            .nf = self,
            .buf = .{},
            .check = false,
        };
        errdefer p.buf.deinit(allocator);

        var work_buf: std.ArrayListUnmanaged(u21) = .{};
        defer work_buf.deinit(allocator);

        for (cps) |cp0| {
            var cp = cp0;

            while (true) {
                // ASCII fast path
                if (cp < 0x80) {
                    try p.buf.append(allocator, @bitCast(@as(u32, cp)));
                } else if (isHangul(cp)) {
                    // Hangul algorithmic decomposition
                    const sIndex = cp - S0;
                    const lIndex = sIndex / N_COUNT;
                    const vIndex = (sIndex % N_COUNT) / T_COUNT;
                    const tIndex = sIndex % T_COUNT;

                    try p.add(allocator, L0 + lIndex);
                    try p.add(allocator, V0 + vIndex);
                    if (tIndex > 0) {
                        try p.add(allocator, T0 + tIndex);
                    }
                } else {
                    // Table lookup for decomposition
                    if (self.decomps.get(cp)) |decomp| {
                        try work_buf.appendSlice(allocator, decomp);
                    } else {
                        try p.add(allocator, cp);
                    }
                }

                // Continue with next item from work buffer
                if (work_buf.items.len == 0) break;
                cp = work_buf.pop() orelse break;
            }
        }

        p.fixOrder();
        return p.buf.toOwnedSlice(allocator);
    }

    // Recompose decomposed+packed codepoints while respecting blocking rules
    fn composedFromPacked(self: *const NF, allocator: Allocator, packed_cps: []const i32) ![]u21 {
        var cps: std.ArrayListUnmanaged(u21) = .{};
        errdefer cps.deinit(allocator);

        var stack: std.ArrayListUnmanaged(u21) = .{};
        defer stack.deinit(allocator);

        var prevCp: i32 = NONE;
        var prevCc: u8 = 0;

        for (packed_cps) |p| {
            const cc = unpackCC(p);
            const cp = unpackCP(p);

            if (prevCp == NONE) {
                if (cc == 0) {
                    prevCp = @intCast(cp);
                } else {
                    try cps.append(allocator, cp);
                }
            } else if (prevCc > 0 and prevCc >= cc) {
                if (cc == 0) {
                    try cps.append(allocator, @intCast(prevCp));
                    try cps.appendSlice(allocator, stack.items);
                    stack.clearRetainingCapacity();
                    prevCp = @intCast(cp);
                } else {
                    try stack.append(allocator, cp);
                }
                prevCc = cc;
            } else {
                const composed = self.composePair(@intCast(prevCp), cp);
                if (composed != NONE) {
                    prevCp = composed;
                } else if (prevCc == 0 and cc == 0) {
                    try cps.append(allocator, @intCast(prevCp));
                    prevCp = @intCast(cp);
                } else {
                    try stack.append(allocator, cp);
                    prevCc = cc;
                }
            }
        }

        if (prevCp != NONE) {
            try cps.append(allocator, @intCast(prevCp));
            try cps.appendSlice(allocator, stack.items);
        }

        return cps.toOwnedSlice(allocator);
    }

    // Public method: NFD (Canonical Decomposition)
    // Decomposes all characters to their canonical decomposed form
    pub fn nfd(self: *const NF, allocator: Allocator, cps: []const u21) ![]u21 {
        const packed_vals = try self.decomposed(allocator, cps);
        defer allocator.free(packed_vals);

        const result = try allocator.alloc(u21, packed_vals.len);
        for (packed_vals, 0..) |p, i| {
            result[i] = unpackCP(p);
        }

        return result;
    }

    // Public method: NFC (Canonical Composition)
    // Decomposes then recomposes where possible
    pub fn nfc(self: *const NF, allocator: Allocator, cps: []const u21) ![]u21 {
        const packed_vals = try self.decomposed(allocator, cps);
        defer allocator.free(packed_vals);

        return self.composedFromPacked(allocator, packed_vals);
    }

    // Cleanup method: Free all allocated memory
    pub fn deinit(self: *NF, allocator: Allocator) void {
        // Free unicode version string
        allocator.free(self.unicodeVersion);

        // Free RuneSets
        self.exclusions.deinit(allocator);
        self.quickCheck.deinit(allocator);

        // Free decomps map and all allocated slices
        var decomps_iter = self.decomps.iterator();
        while (decomps_iter.next()) |entry| {
            allocator.free(entry.value_ptr.*);
        }
        self.decomps.deinit();

        // Free recomps map and all nested maps
        var recomps_iter = self.recomps.iterator();
        while (recomps_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.recomps.deinit();

        // Free ranks map
        self.ranks.deinit();
    }
};
