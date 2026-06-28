//! ENSIP15 Initialization
//!
//! This module implements the init() function that loads the ENSIP15 specification
//! from the embedded spec.bin file and constructs all necessary data structures.
//!
//! Agent 5: Implements all data loading functions including:
//! - decodeNamedCodepoints: Loads fenced character mappings
//! - decodeMapped: Loads character mapping rules
//! - decodeGroups: Loads script groups (Latin, Greek, Han, etc.)
//! - decodeEmojis: Recursively loads emoji sequences
//! - makeEmojiTree: Builds trie for efficient emoji matching
//! - decodeWholes: Loads confusable detection data (partial implementation)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Decoder = @import("../util/decoder.zig").Decoder;

// Import types
const types = @import("types.zig");
const Group = types.Group;
const EmojiSequence = types.EmojiSequence;
const EmojiNode = types.EmojiNode;
const Whole = types.Whole;

// Embed the spec.bin file at compile time
const spec_data = @embedFile("spec.bin");

// ============================================================
// Decode Helper Functions
// ============================================================

/// Decode named codepoints (fenced characters)
///
/// Binary format:
/// 1. Read count (unsigned)
/// 2. Read sorted ascending codepoints
/// 3. For each codepoint, read its string name
///
/// Go reference: ensip15.go lines 38-44
pub fn decodeNamedCodepoints(decoder: *Decoder, allocator: Allocator) !std.AutoHashMap(u21, []const u8) {
    var map = std.AutoHashMap(u21, []const u8).init(allocator);
    errdefer {
        var iter = map.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        map.deinit();
    }

    const count = decoder.ReadUnsigned();
    const codepoints = try decoder.ReadSortedAscending(count, allocator);
    defer allocator.free(codepoints);

    for (codepoints) |cp| {
        const name = try decoder.ReadString(allocator);
        try map.put(@intCast(cp), name);
    }

    return map;
}

/// Decode mapped characters (complex mapping structure)
///
/// Binary format:
/// Loop until width == 0:
///   1. Read width w (unsigned)
///   2. If w == 0, break
///   3. Read sorted unique keys
///   4. Build n×w matrix (n keys, w runes each)
///   5. For each position j in width:
///      - Read unsorted deltas for all n keys
///   6. Store each key → rune sequence in map
///
/// Go reference: ensip15.go lines 46-70
pub fn decodeMapped(decoder: *Decoder, allocator: Allocator) !std.AutoHashMap(u21, []const u21) {
    var map = std.AutoHashMap(u21, []const u21).init(allocator);
    errdefer {
        var iter = map.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        map.deinit();
    }

    while (true) {
        const w = decoder.ReadUnsigned();
        if (w == 0) break;

        const keys = try decoder.ReadSortedUnique(allocator);
        defer allocator.free(keys);

        const n: usize = @intCast(keys.len);
        const w_usize: usize = @intCast(w);

        // Allocate matrix m[n][w]
        const matrix = try allocator.alloc([]u21, n);
        errdefer {
            for (matrix) |row| {
                allocator.free(row);
            }
            allocator.free(matrix);
        }

        for (matrix) |*row| {
            row.* = try allocator.alloc(u21, w_usize);
        }

        // Read transposed: for each column, read all rows
        for (0..w_usize) |j| {
            const column = try decoder.ReadUnsortedDeltas(@intCast(n), allocator);
            defer allocator.free(column);

            for (0..n) |i| {
                matrix[i][j] = @intCast(column[i]);
            }
        }

        // Store in map
        for (0..n) |i| {
            try map.put(@intCast(keys[i]), matrix[i]);
        }

        allocator.free(matrix);
    }

    return map;
}

/// Decode script groups
///
/// Binary format:
/// Loop until name is empty:
///   1. Read group name (string)
///   2. If name is empty, break
///   3. Read bits (unsigned): bit 0 = restricted, bit 1 = cmWhitelisted
///   4. Read primary codepoints (unique)
///   5. Read secondary codepoints (unique)
///
/// Go reference: groups.go lines 43-60
pub fn decodeGroups(decoder: *Decoder, allocator: Allocator) ![]Group {
    const RuneSet = @import("../util/runeset.zig").RuneSet;

    var groups: std.ArrayListUnmanaged(Group) = .{};
    errdefer {
        for (groups.items) |group| {
            allocator.free(group.name);
            group.primary.deinit(allocator);
            group.secondary.deinit(allocator);
        }
        groups.deinit(allocator);
    }

    while (true) {
        const name = try decoder.ReadString(allocator);
        if (name.len == 0) {
            allocator.free(name);
            break;
        }

        const bits = decoder.ReadUnsigned();
        const restricted = (bits & 1) != 0;
        const cm_whitelisted = (bits & 2) != 0;

        const primary_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(primary_ints);
        const primary = try RuneSet.fromInts(allocator, primary_ints);

        const secondary_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(secondary_ints);
        const secondary = try RuneSet.fromInts(allocator, secondary_ints);

        try groups.append(allocator, Group{
            .index = @intCast(groups.items.len),
            .name = name,
            .restricted = restricted,
            .cm_whitelisted = cm_whitelisted,
            .primary = primary,
            .secondary = secondary,
        });
    }

    return try groups.toOwnedSlice(allocator);
}

/// Decode emoji sequences
///
/// Recursive decoder that builds emoji sequences with optional FE0F variation selectors.
/// Each emoji has both normalized (FE0F stripped) and beautified (FE0F preserved) forms.
///
/// Binary format (recursive):
///   1. Read count of leaf emojis
///   2. Read sorted ascending codepoints for leaves
///   3. For each leaf, create emoji sequence
///   4. Read count of branches
///   5. Read sorted ascending codepoints for branches
///   6. Recursively decode each branch
///
/// Go reference: emojis.go lines 38-61
pub fn decodeEmojis(decoder: *Decoder, allocator: Allocator) ![]EmojiSequence {
    return decodeEmojisRecursive(decoder, allocator, &.{});
}

/// Recursive helper for decodeEmojis
fn decodeEmojisRecursive(decoder: *Decoder, allocator: Allocator, prev: []const u21) ![]EmojiSequence {
    const FE0F: u21 = 0xFE0F;

    var result: std.ArrayListUnmanaged(EmojiSequence) = .{};
    errdefer {
        for (result.items) |emoji| {
            allocator.free(emoji.normalized);
            allocator.free(emoji.beautified);
        }
        result.deinit(allocator);
    }

    // First loop: leaf emojis (terminal nodes)
    const count1 = decoder.ReadUnsigned();
    const cps1 = try decoder.ReadSortedAscending(count1, allocator);
    defer allocator.free(cps1);

    for (cps1) |cp| {
        // Build beautified: prev + cp
        const beautified = try allocator.alloc(u21, prev.len + 1);
        if (prev.len > 0) {
            @memcpy(beautified[0..prev.len], prev);
        }
        beautified[prev.len] = @intCast(cp);

        // Build normalized: beautified with FE0F stripped
        var normalized_list: std.ArrayListUnmanaged(u21) = .{};
        defer normalized_list.deinit(allocator);

        for (beautified) |x| {
            if (x != FE0F) {
                try normalized_list.append(allocator, x);
            }
        }

        // If no FE0F was stripped, reuse beautified
        const normalized = if (normalized_list.items.len == beautified.len)
            try allocator.dupe(u21, beautified)
        else
            try normalized_list.toOwnedSlice(allocator);

        try result.append(allocator, EmojiSequence{
            .normalized = normalized,
            .beautified = beautified,
        });
    }

    // Second loop: branch emojis (recursive)
    const count2 = decoder.ReadUnsigned();
    const cps2 = try decoder.ReadSortedAscending(count2, allocator);
    defer allocator.free(cps2);

    for (cps2) |cp| {
        // Build new prev: prev + cp
        const new_prev = try allocator.alloc(u21, prev.len + 1);
        defer allocator.free(new_prev);

        if (prev.len > 0) {
            @memcpy(new_prev[0..prev.len], prev);
        }
        new_prev[prev.len] = @intCast(cp);

        // Recursive call
        const sub_emojis = try decodeEmojisRecursive(decoder, allocator, new_prev);
        defer {
            for (sub_emojis) |emoji| {
                allocator.free(emoji.normalized);
                allocator.free(emoji.beautified);
            }
            allocator.free(sub_emojis);
        }

        // Append all sub-emojis to result
        for (sub_emojis) |emoji| {
            try result.append(allocator, EmojiSequence{
                .normalized = try allocator.dupe(u21, emoji.normalized),
                .beautified = try allocator.dupe(u21, emoji.beautified),
            });
        }
    }

    return try result.toOwnedSlice(allocator);
}

/// Decode whole-script confusables
///
/// Binary format:
/// Loop until confused set is empty:
///   1. Read confused codepoints (unique)
///   2. If empty, break
///   3. Read valid codepoints (unique)
///   4. Build complement map (which groups each codepoint belongs to)
///
/// Returns both wholes array and confusables map.
///
/// Go reference: wholes.go lines 17-84
pub fn decodeWholes(
    decoder: *Decoder,
    groups: []Group,
    allocator: Allocator,
) !struct { wholes: []Whole, confusables: std.AutoHashMap(u21, Whole) } {
    const RuneSet = @import("../util/runeset.zig").RuneSet;

    // Helper struct for tracking group extents
    const Extent = struct {
        gs: std.AutoHashMap(*const Group, void),
        cps: std.AutoHashMap(u21, void),

        fn deinit(self: *@This()) void {
            self.gs.deinit();
            self.cps.deinit();
        }
    };

    var wholes: std.ArrayListUnmanaged(Whole) = .{};
    errdefer {
        for (wholes.items) |*whole| {
            whole.valid.deinit(allocator);
            whole.confused.deinit(allocator);
            var iter = whole.complements.valueIterator();
            while (iter.next()) |value| {
                allocator.free(value.*);
            }
            whole.complements.deinit();
        }
        wholes.deinit(allocator);
    }

    var confusables = std.AutoHashMap(u21, Whole).init(allocator);
    errdefer confusables.deinit();

    while (true) {
        const confused_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(confused_ints);

        if (confused_ints.len == 0) break;

        const confused = try RuneSet.fromInts(allocator, confused_ints);
        errdefer confused.deinit(allocator);

        const valid_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(valid_ints);

        const valid = try RuneSet.fromInts(allocator, valid_ints);
        errdefer valid.deinit(allocator);

        var complements_map = std.AutoHashMap(u21, []i32).init(allocator);
        errdefer {
            var iter = complements_map.valueIterator();
            while (iter.next()) |value| {
                allocator.free(value.*);
            }
            complements_map.deinit();
        }

        const whole = Whole{
            .valid = valid,
            .confused = confused,
            .complements = complements_map,
        };

        try wholes.append(allocator, whole);

        // Build complements map using extent algorithm
        var cover = std.AutoHashMap(*const Group, void).init(allocator);
        defer cover.deinit();

        var extents: std.ArrayListUnmanaged(*Extent) = .{};
        defer {
            for (extents.items) |ext| {
                ext.deinit();
                allocator.destroy(ext);
            }
            extents.deinit(allocator);
        }

        // Combine valid and confused codepoints
        var all_cps: std.ArrayListUnmanaged(u21) = .{};
        defer all_cps.deinit(allocator);
        try all_cps.appendSlice(allocator, valid.runes);
        try all_cps.appendSlice(allocator, confused.runes);

        // For each codepoint, find which groups contain it and assign to extent
        for (all_cps.items) |cp| {
            // Find all groups containing this codepoint
            var gs = std.AutoHashMap(*const Group, void).init(allocator);
            defer gs.deinit();

            for (groups) |*g| {
                if (g.contains(cp)) {
                    try gs.put(g, {});
                }
            }

            // Find extent that shares a group with this codepoint
            var ext: ?*Extent = null;
            outer: for (extents.items) |x| {
                var gs_iter = gs.keyIterator();
                while (gs_iter.next()) |g| {
                    if (x.gs.contains(g.*)) {
                        ext = x;
                        break :outer;
                    }
                }
            }

            // Create new extent if none found
            if (ext == null) {
                const new_ext = try allocator.create(Extent);
                new_ext.* = Extent{
                    .gs = std.AutoHashMap(*const Group, void).init(allocator),
                    .cps = std.AutoHashMap(u21, void).init(allocator),
                };
                try extents.append(allocator, new_ext);
                ext = new_ext;
            }

            // Add groups and codepoint to extent
            var gs_iter = gs.keyIterator();
            while (gs_iter.next()) |g| {
                try ext.?.gs.put(g.*, {});
                try cover.put(g.*, {});
            }
            try ext.?.cps.put(cp, {});
        }

        // For each extent, compute complement (groups NOT in extent)
        for (extents.items) |x| {
            var comps: std.ArrayListUnmanaged(i32) = .{};
            defer comps.deinit(allocator);

            var cover_iter = cover.keyIterator();
            while (cover_iter.next()) |g| {
                if (!x.gs.contains(g.*)) {
                    try comps.append(allocator, g.*.index);
                }
            }

            // Sort complements
            std.mem.sort(i32, comps.items, {}, comptime std.sort.asc(i32));

            // Store complement for each codepoint in this extent
            const comps_copy = try comps.toOwnedSlice(allocator);
            var cps_iter = x.cps.keyIterator();
            while (cps_iter.next()) |cp| {
                // Store reference to same array for all codepoints in extent
                try wholes.items[wholes.items.len - 1].complements.put(cp.*, try allocator.dupe(i32, comps_copy));
            }
            allocator.free(comps_copy);
        }

        // Add to confusables map AFTER complements are populated
        const whole_ref = &wholes.items[wholes.items.len - 1];
        for (whole_ref.confused.runes) |cp| {
            try confusables.put(cp, whole_ref.*);
        }
    }

    return .{
        .wholes = try wholes.toOwnedSlice(allocator),
        .confusables = confusables,
    };
}

// ============================================================
// Emoji Tree Construction
// ============================================================

/// Build emoji trie tree for efficient sequence matching
///
/// The tree allows looking up emoji sequences by walking the trie.
/// FE0F (variation selector) is handled specially - it creates optional branches.
///
/// Algorithm:
/// 1. Create root node
/// 2. For each emoji sequence:
///    - Walk through beautified codepoints
///    - For FE0F: create parallel paths (with and without)
///    - For other codepoints: advance current path
///    - At end: mark node with emoji
///
/// Go reference: emojis.go lines 80-100
pub fn makeEmojiTree(emojis: []EmojiSequence, allocator: Allocator) !*EmojiNode {
    const FE0F: u21 = 0xFE0F;

    const root = try allocator.create(EmojiNode);
    root.* = EmojiNode{
        .children = null,
        .emoji = null,
    };

    for (emojis) |*emoji| {
        // Start with just the root node
        var nodes: std.ArrayListUnmanaged(*EmojiNode) = .{};
        defer nodes.deinit(allocator);
        try nodes.append(allocator, root);

        // Process each codepoint in the beautified sequence
        for (emoji.beautified) |cp| {
            if (cp == FE0F) {
                // FE0F creates duplicate branches
                const current_len = nodes.items.len;
                for (nodes.items[0..current_len]) |node| {
                    const new_node = try node.child(allocator, cp);
                    try nodes.append(allocator, new_node);
                }
            } else {
                // Non-FE0F updates existing branches in place
                for (nodes.items, 0..) |node, i| {
                    nodes.items[i] = try node.child(allocator, cp);
                }
            }
        }

        // Mark all end nodes with this emoji
        for (nodes.items) |node| {
            node.emoji = emoji;
        }
    }

    return root;
}

// ============================================================
// Sorting and Comparison
// ============================================================

/// Compare emoji sequences lexicographically by normalized form
///
/// Used for sorting emojis before building the tree.
///
/// Go reference: ensip15.go lines 89-91 (uses compareRunes)
fn emojiLessThan(context: void, a: EmojiSequence, b: EmojiSequence) bool {
    _ = context;
    return compareRunes(a.normalized, b.normalized) < 0;
}

/// Compare two rune slices lexicographically
///
/// Returns:
///   - negative if a < b
///   - zero if a == b
///   - positive if a > b
fn compareRunes(a: []const u21, b: []const u21) i32 {
    const min_len = @min(a.len, b.len);
    for (a[0..min_len], b[0..min_len]) |ca, cb| {
        if (ca != cb) {
            return @as(i32, @intCast(ca)) - @as(i32, @intCast(cb));
        }
    }
    return @as(i32, @intCast(a.len)) - @as(i32, @intCast(b.len));
}

// ============================================================
// Group Lookup
// ============================================================

/// Find a group by name
///
/// Used to create direct references to LATIN and GREEK groups.
///
/// Parameters:
///   - groups: Slice of all groups
///   - name: Name of the group to find (e.g., "Latin", "Greek")
///
/// Returns: Pointer to the group, or null if not found
///
/// Go reference: groups.go lines 36-41
pub fn findGroup(groups: []Group, name: []const u8) ?*Group {
    for (groups) |*group| {
        if (std.mem.eql(u8, group.name, name)) {
            return group;
        }
    }
    return null;
}

// ============================================================
// Filter Predicates
// ============================================================

/// Check if codepoint is ASCII (< 0x80)
///
/// Used to filter possiblyValid for the ASCII synthetic group.
fn isAscii(cp: u21) bool {
    return cp < 0x80;
}

// ============================================================
// Tree Cleanup
// ============================================================

/// Recursively free emoji tree structure
///
/// Walks the tree and frees all nodes and their children maps.
pub fn freeEmojiTree(root: *EmojiNode, allocator: Allocator) void {
    if (root.children) |*children| {
        var iter = children.valueIterator();
        while (iter.next()) |child_ptr| {
            freeEmojiTree(child_ptr.*, allocator);
        }
        children.deinit();
    }
    allocator.destroy(root);
}

// ============================================================
// Public Init Function (STUB)
// ============================================================

/// Placeholder Ensip15 struct for init function
/// This mirrors the real struct definition in ensip15.zig
pub const Ensip15Stub = struct {
    allocator: Allocator,
    // All other fields would go here - currently stubbed

    /// Initialize ENSIP15 from embedded spec.bin
    ///
    /// This function:
    /// 1. Decodes the binary spec file using Decoder
    /// 2. Loads all rune sets, mappings, groups, and emojis
    /// 3. Builds the emoji tree structure
    /// 4. Constructs possiblyValid and uniqueNonConfusables sets
    /// 5. Creates direct references to commonly-used groups
    ///
    /// The init process follows the exact sequence from Go's ENSIP15.New():
    /// - Read escape/ignored/combining mark sets
    /// - Read fenced and mapped character tables
    /// - Decode groups and emojis
    /// - Decode whole-confusables
    /// - Sort emojis and build tree
    /// - Construct derived sets
    ///
    /// All decode functions are currently stubbed and will be implemented in later tasks.
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for all data structures
    ///
    /// Returns: Initialized Ensip15 instance
    ///
    /// Errors: Returns error if decoding fails or memory allocation fails
    pub fn init(allocator: Allocator) !Ensip15Stub {
        // Initialize decoder from embedded spec.bin
        var decoder = try Decoder.init(spec_data, allocator);
        defer decoder.deinit(allocator);
        _ = &decoder; // Will be used in full implementation

        // For now, just return a minimal struct since all decode functions panic
        // In full implementation, this would:
        // 1. Decode all RuneSets (should_escape, ignored, combining_marks, etc.)
        // 2. Decode fenced and mapped hashmaps
        // 3. Decode groups and emojis
        // 4. Decode wholes and confusables
        // 5. Sort emojis and build tree
        // 6. Construct possiblyValid and uniqueNonConfusables
        // 7. Create group references (_LATIN, _GREEK, _ASCII, _EMOJI)

        // All of these would panic if actually called due to stubbed helper functions:
        // result.max_non_spacing_marks = decoder.ReadUnsigned();
        // result.fenced = try decodeNamedCodepoints(&decoder, allocator);
        // result.mapped = try decodeMapped(&decoder, allocator);
        // result.groups = try decodeGroups(&decoder, allocator);
        // result.emojis = try decodeEmojis(&decoder, allocator);
        // const wholes_result = try decodeWholes(&decoder, result.groups, allocator);
        // decoder.assertEOF();
        // std.mem.sort(EmojiSequence, result.emojis, {}, emojiLessThan);
        // result.emoji_root = try makeEmojiTree(result.emojis, allocator);
        // [possiblyValid and uniqueNonConfusables construction]
        // [Group references initialization]

        // Return minimal stub
        return Ensip15Stub{
            .allocator = allocator,
        };
    }
};

// ============================================================
// Tests
// ============================================================

test "spec.bin is embedded" {
    // Verify that spec.bin was successfully embedded
    try std.testing.expect(spec_data.len > 0);
    // The spec.bin file should be around 30KB
    try std.testing.expect(spec_data.len > 10000);
}

test "init compiles" {
    const allocator = std.testing.allocator;
    const result = try Ensip15Stub.init(allocator);
    _ = result;
    // Just verify it compiles and returns
}

test "decodeNamedCodepoints loads fenced characters" {
    const allocator = std.testing.allocator;

    var decoder = try Decoder.init(spec_data, allocator);
    defer decoder.deinit(allocator);

    // Skip initial RuneSets to get to named codepoints
    const should_escape = try decoder.ReadUnique(allocator);
    defer allocator.free(should_escape);
    const ignored = try decoder.ReadUnique(allocator);
    defer allocator.free(ignored);
    const combining_marks = try decoder.ReadUnique(allocator);
    defer allocator.free(combining_marks);
    _ = decoder.ReadUnsigned(); // maxNonSpacingMarks
    const non_spacing = try decoder.ReadUnique(allocator);
    defer allocator.free(non_spacing);
    const nfc_check = try decoder.ReadUnique(allocator);
    defer allocator.free(nfc_check);

    // Now decode named codepoints
    var fenced = try decodeNamedCodepoints(&decoder, allocator);
    defer {
        var iter = fenced.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        fenced.deinit();
    }

    // Should have some fenced characters
    try std.testing.expect(fenced.count() > 0);
}

test "decodeMapped loads character mappings" {
    const allocator = std.testing.allocator;

    var decoder = try Decoder.init(spec_data, allocator);
    defer decoder.deinit(allocator);

    // Skip to mapped section
    {
        const should_escape = try decoder.ReadUnique(allocator);
        defer allocator.free(should_escape);
        const ignored = try decoder.ReadUnique(allocator);
        defer allocator.free(ignored);
        const combining_marks = try decoder.ReadUnique(allocator);
        defer allocator.free(combining_marks);
        _ = decoder.ReadUnsigned();
        const non_spacing = try decoder.ReadUnique(allocator);
        defer allocator.free(non_spacing);
        const nfc_check = try decoder.ReadUnique(allocator);
        defer allocator.free(nfc_check);

        // Skip fenced
        const count = decoder.ReadUnsigned();
        const cps = try decoder.ReadSortedAscending(count, allocator);
        defer allocator.free(cps);
        for (cps) |_| {
            const name = try decoder.ReadString(allocator);
            allocator.free(name);
        }
    }

    // Now decode mapped
    var mapped = try decodeMapped(&decoder, allocator);
    defer {
        var iter = mapped.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        mapped.deinit();
    }

    // Should have mapped characters
    try std.testing.expect(mapped.count() > 0);
}

test "decodeGroups loads script groups" {
    const allocator = std.testing.allocator;

    var decoder = try Decoder.init(spec_data, allocator);
    defer decoder.deinit(allocator);

    // Skip to groups section
    {
        const should_escape = try decoder.ReadUnique(allocator);
        defer allocator.free(should_escape);
        const ignored = try decoder.ReadUnique(allocator);
        defer allocator.free(ignored);
        const combining_marks = try decoder.ReadUnique(allocator);
        defer allocator.free(combining_marks);
        _ = decoder.ReadUnsigned();
        const non_spacing = try decoder.ReadUnique(allocator);
        defer allocator.free(non_spacing);
        const nfc_check = try decoder.ReadUnique(allocator);
        defer allocator.free(nfc_check);

        // Skip fenced
        const fenced_count = decoder.ReadUnsigned();
        const fenced_cps = try decoder.ReadSortedAscending(fenced_count, allocator);
        defer allocator.free(fenced_cps);
        for (fenced_cps) |_| {
            const name = try decoder.ReadString(allocator);
            allocator.free(name);
        }

        // Skip mapped
        while (true) {
            const w = decoder.ReadUnsigned();
            if (w == 0) break;
            const keys = try decoder.ReadSortedUnique(allocator);
            defer allocator.free(keys);
            const n: usize = @intCast(keys.len);
            for (0..@intCast(w)) |_| {
                const col = try decoder.ReadUnsortedDeltas(@intCast(n), allocator);
                allocator.free(col);
            }
        }
    }

    // Now decode groups
    const groups = try decodeGroups(&decoder, allocator);
    defer {
        for (groups) |group| {
            allocator.free(group.name);
            group.primary.deinit(allocator);
            group.secondary.deinit(allocator);
        }
        allocator.free(groups);
    }

    // Should have multiple groups
    try std.testing.expect(groups.len > 10);

    // Check for common groups
    var found_latin = false;
    var found_greek = false;
    for (groups) |group| {
        if (std.mem.eql(u8, group.name, "Latin")) found_latin = true;
        if (std.mem.eql(u8, group.name, "Greek")) found_greek = true;
    }

    try std.testing.expect(found_latin);
    try std.testing.expect(found_greek);
}

test "decodeEmojis loads emoji sequences" {
    const allocator = std.testing.allocator;

    var decoder = try Decoder.init(spec_data, allocator);
    defer decoder.deinit(allocator);

    // Skip to emoji section - same process as before
    {
        const should_escape = try decoder.ReadUnique(allocator);
        defer allocator.free(should_escape);
        const ignored = try decoder.ReadUnique(allocator);
        defer allocator.free(ignored);
        const combining_marks = try decoder.ReadUnique(allocator);
        defer allocator.free(combining_marks);
        _ = decoder.ReadUnsigned();
        const non_spacing = try decoder.ReadUnique(allocator);
        defer allocator.free(non_spacing);
        const nfc_check = try decoder.ReadUnique(allocator);
        defer allocator.free(nfc_check);

        const fenced_count = decoder.ReadUnsigned();
        const fenced_cps = try decoder.ReadSortedAscending(fenced_count, allocator);
        defer allocator.free(fenced_cps);
        for (fenced_cps) |_| {
            const name = try decoder.ReadString(allocator);
            allocator.free(name);
        }

        while (true) {
            const w = decoder.ReadUnsigned();
            if (w == 0) break;
            const keys = try decoder.ReadSortedUnique(allocator);
            defer allocator.free(keys);
            const n: usize = @intCast(keys.len);
            for (0..@intCast(w)) |_| {
                const col = try decoder.ReadUnsortedDeltas(@intCast(n), allocator);
                allocator.free(col);
            }
        }

        while (true) {
            const name = try decoder.ReadString(allocator);
            if (name.len == 0) {
                allocator.free(name);
                break;
            }
            allocator.free(name);
            _ = decoder.ReadUnsigned();
            const primary = try decoder.ReadUnique(allocator);
            allocator.free(primary);
            const secondary = try decoder.ReadUnique(allocator);
            allocator.free(secondary);
        }
    }

    // Now decode emojis
    const emojis = try decodeEmojis(&decoder, allocator);
    defer {
        for (emojis) |emoji| {
            allocator.free(emoji.normalized);
            allocator.free(emoji.beautified);
        }
        allocator.free(emojis);
    }

    // Should have many emoji sequences
    try std.testing.expect(emojis.len > 100);

    // Check that normalized and beautified differ for some emojis (FE0F handling)
    var has_fe0f = false;
    for (emojis) |emoji| {
        if (emoji.normalized.len < emoji.beautified.len) {
            has_fe0f = true;
            break;
        }
    }
    try std.testing.expect(has_fe0f);
}
