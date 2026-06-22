//! ENSIP15 Normalization Methods
//!
//! This file contains the main normalization pipeline and validation logic
//! for ENS name normalization according to ENSIP15 specification.
//!
//! The normalization process:
//! 1. Split name by dots into labels
//! 2. For each label: tokenize → normalize → validate
//! 3. Join labels back with dots
//!
//! Note: This is a STUB implementation. All methods use @panic("TODO: implement")
//! and will be implemented in future tasks.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import types from other modules
const types = @import("types.zig");
const OutputToken = types.OutputToken;
const Group = types.Group;
const EmojiSequence = types.EmojiSequence;
const EmojiNode = types.EmojiNode;
const Whole = types.Whole;

const errors = @import("errors.zig");
const Error = errors.Error;

const RuneSet = @import("../util/runeset.zig").RuneSet;
const utils = @import("utils.zig");
const init_mod = @import("init.zig");
const NF = @import("../nf/nf.zig").NF;
const Decoder = @import("../util/decoder.zig").Decoder;

// Embed the spec.bin file at compile time
const compressed = @embedFile("spec.bin");

// ============================================================
// Main ENSIP15 Structure
// ============================================================

/// Main ENSIP15 normalization context
/// Contains all data structures needed for ENS name normalization
pub const Ensip15 = struct {
    allocator: Allocator,

    // Placeholder fields - these would be populated by Task 11 (ensip15-init)
    // For now, we just define the structure with fields needed for validation

    // Normalization context
    nf: ?*const @import("../nf/nf.zig").NF = null,

    // Character sets
    should_escape: RuneSet = undefined,
    ignored: RuneSet = undefined,
    combining_marks: RuneSet = undefined,
    non_spacing_marks: RuneSet = undefined,
    max_non_spacing_marks: usize = 4, // Default max NSM count
    nfc_check: RuneSet = undefined,

    // Character mappings
    fenced: std.AutoHashMap(u21, []const u8) = undefined,
    mapped: std.AutoHashMap(u21, []const u21) = undefined,

    // Script groups
    groups: []Group = &[_]Group{},

    // Emoji sequences
    emojis: []EmojiSequence = &[_]EmojiSequence{},
    emoji_root: ?*EmojiNode = null,

    // Confusable detection
    possibly_valid: RuneSet = undefined,
    wholes: []Whole = &[_]Whole{},
    confusables: std.AutoHashMap(u21, Whole) = undefined,
    unique_non_confusables: RuneSet = undefined,

    // Common group references
    _ASCII: ?*const Group = null,
    _EMOJI: ?*const Group = null,
    _LATIN: ?*const Group = null,
    _GREEK: ?*const Group = null,

    /// Initialize ENSIP15 normalization context
    pub fn init(allocator: Allocator) !Ensip15 {
        var decoder = try Decoder.init(compressed, allocator);
        defer decoder.deinit(allocator);

        // Initialize NF
        var nf_ptr = try allocator.create(NF);
        nf_ptr.* = try NF.init(allocator);
        errdefer {
            nf_ptr.deinit(allocator);
            allocator.destroy(nf_ptr);
        }

        // Read all RuneSets
        const should_escape_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(should_escape_ints);
        const should_escape = try RuneSet.fromInts(allocator, should_escape_ints);
        errdefer should_escape.deinit(allocator);

        const ignored_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(ignored_ints);
        const ignored = try RuneSet.fromInts(allocator, ignored_ints);
        errdefer ignored.deinit(allocator);

        const combining_marks_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(combining_marks_ints);
        const combining_marks = try RuneSet.fromInts(allocator, combining_marks_ints);
        errdefer combining_marks.deinit(allocator);

        const max_non_spacing_marks = decoder.ReadUnsigned();

        const non_spacing_marks_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(non_spacing_marks_ints);
        const non_spacing_marks = try RuneSet.fromInts(allocator, non_spacing_marks_ints);
        errdefer non_spacing_marks.deinit(allocator);

        const nfc_check_ints = try decoder.ReadUnique(allocator);
        defer allocator.free(nfc_check_ints);
        const nfc_check = try RuneSet.fromInts(allocator, nfc_check_ints);
        errdefer nfc_check.deinit(allocator);

        // Decode maps
        var fenced = try init_mod.decodeNamedCodepoints(&decoder, allocator);
        errdefer {
            var iter = fenced.valueIterator();
            while (iter.next()) |value| {
                allocator.free(value.*);
            }
            fenced.deinit();
        }

        var mapped = try init_mod.decodeMapped(&decoder, allocator);
        errdefer {
            var iter = mapped.valueIterator();
            while (iter.next()) |value| {
                allocator.free(value.*);
            }
            mapped.deinit();
        }

        // Decode groups and emojis
        const groups = try init_mod.decodeGroups(&decoder, allocator);
        errdefer {
            for (groups) |group| {
                allocator.free(group.name);
                group.primary.deinit(allocator);
                group.secondary.deinit(allocator);
            }
            allocator.free(groups);
        }

        const emojis = try init_mod.decodeEmojis(&decoder, allocator);
        errdefer {
            for (emojis) |emoji| {
                allocator.free(emoji.normalized);
                allocator.free(emoji.beautified);
            }
            allocator.free(emojis);
        }

        // Decode wholes
        var wholes_result = try init_mod.decodeWholes(&decoder, groups, allocator);
        errdefer {
            for (wholes_result.wholes) |*whole| {
                whole.valid.deinit(allocator);
                whole.confused.deinit(allocator);
                var iter = whole.complements.valueIterator();
                while (iter.next()) |value| {
                    allocator.free(value.*);
                }
                whole.complements.deinit();
            }
            allocator.free(wholes_result.wholes);
            wholes_result.confusables.deinit();
        }

        decoder.assertEOF();

        // Sort emojis
        std.mem.sort(EmojiSequence, emojis, {}, struct {
            fn lessThan(_: void, a: EmojiSequence, b: EmojiSequence) bool {
                return utils.compareRunes(a.normalized, b.normalized) < 0;
            }
        }.lessThan);

        // Build emoji tree
        const emoji_root = try init_mod.makeEmojiTree(emojis, allocator);
        errdefer init_mod.freeEmojiTree(emoji_root, allocator);

        // Build possibly_valid set
        var union_map = std.AutoHashMap(u21, void).init(allocator);
        defer union_map.deinit();
        var multi_map = std.AutoHashMap(u21, void).init(allocator);
        defer multi_map.deinit();

        for (groups) |*g| {
            const primary = try g.primary.toArray(allocator);
            defer allocator.free(primary);
            const secondary = try g.secondary.toArray(allocator);
            defer allocator.free(secondary);

            for (primary) |cp| {
                if (union_map.contains(cp)) {
                    try multi_map.put(cp, {});
                } else {
                    try union_map.put(cp, {});
                }
            }
            for (secondary) |cp| {
                if (union_map.contains(cp)) {
                    try multi_map.put(cp, {});
                } else {
                    try union_map.put(cp, {});
                }
            }
        }

        var possibly_valid_map = std.AutoHashMap(u21, void).init(allocator);
        defer possibly_valid_map.deinit();

        var union_iter = union_map.keyIterator();
        while (union_iter.next()) |cp_ptr| {
            const cp = cp_ptr.*;
            try possibly_valid_map.put(cp, {});

            const nfd_result = try nf_ptr.nfd(allocator, &[_]u21{cp});
            defer allocator.free(nfd_result);
            for (nfd_result) |nfd_cp| {
                try possibly_valid_map.put(nfd_cp, {});
            }
        }

        const possibly_valid = blk: {
            var list: std.ArrayListUnmanaged(u21) = .{};
            var pv_iter = possibly_valid_map.keyIterator();
            while (pv_iter.next()) |cp_ptr| {
                try list.append(allocator, cp_ptr.*);
            }
            std.mem.sort(u21, list.items, {}, comptime std.sort.asc(u21));
            break :blk RuneSet{ .runes = try list.toOwnedSlice(allocator) };
        };
        errdefer possibly_valid.deinit(allocator);

        // Build unique_non_confusables
        var multi_iter = multi_map.keyIterator();
        while (multi_iter.next()) |cp_ptr| {
            _ = union_map.remove(cp_ptr.*);
        }
        var conf_iter = wholes_result.confusables.keyIterator();
        while (conf_iter.next()) |cp_ptr| {
            _ = union_map.remove(cp_ptr.*);
        }

        const unique_non_confusables = blk: {
            var list: std.ArrayListUnmanaged(u21) = .{};
            var u_iter = union_map.keyIterator();
            while (u_iter.next()) |cp_ptr| {
                try list.append(allocator, cp_ptr.*);
            }
            std.mem.sort(u21, list.items, {}, comptime std.sort.asc(u21));
            break :blk RuneSet{ .runes = try list.toOwnedSlice(allocator) };
        };
        errdefer unique_non_confusables.deinit(allocator);

        // Find groups
        const latin_group = init_mod.findGroup(groups, "Latin").?;
        const greek_group = init_mod.findGroup(groups, "Greek").?;

        // Create ASCII group
        const ascii_group = try allocator.create(Group);
        errdefer allocator.destroy(ascii_group);

        const ascii_filter = struct {
            fn isAscii(cp: u21) bool {
                return cp < 0x80;
            }
        }.isAscii;
        const ascii_primary = try possibly_valid.filter(allocator, &ascii_filter);
        errdefer ascii_primary.deinit(allocator);

        ascii_group.* = Group{
            .index = -1,
            .name = "ASCII",
            .restricted = false,
            .cm_whitelisted = false,
            .primary = ascii_primary,
            .secondary = RuneSet.fromSlice(&[_]u21{}),
        };

        // Create EMOJI group
        const emoji_group = try allocator.create(Group);
        errdefer allocator.destroy(emoji_group);

        emoji_group.* = Group{
            .index = -1,
            .name = "EMOJI",
            .restricted = false,
            .cm_whitelisted = false,
            .primary = RuneSet.fromSlice(&[_]u21{}),
            .secondary = RuneSet.fromSlice(&[_]u21{}),
        };

        return Ensip15{
            .allocator = allocator,
            .nf = nf_ptr,
            .should_escape = should_escape,
            .ignored = ignored,
            .combining_marks = combining_marks,
            .non_spacing_marks = non_spacing_marks,
            .max_non_spacing_marks = @intCast(max_non_spacing_marks),
            .nfc_check = nfc_check,
            .fenced = fenced,
            .mapped = mapped,
            .groups = groups,
            .emojis = emojis,
            .emoji_root = emoji_root,
            .possibly_valid = possibly_valid,
            .wholes = wholes_result.wholes,
            .confusables = wholes_result.confusables,
            .unique_non_confusables = unique_non_confusables,
            ._ASCII = ascii_group,
            ._EMOJI = emoji_group,
            ._LATIN = latin_group,
            ._GREEK = greek_group,
        };
    }

    /// Cleanup resources
    pub fn deinit(self: *Ensip15) void {
        // Free NF
        if (self.nf) |nf_ptr| {
            var nf_mut = @constCast(nf_ptr);
            nf_mut.deinit(self.allocator);
            self.allocator.destroy(nf_mut);
        }

        // Free RuneSets
        self.should_escape.deinit(self.allocator);
        self.ignored.deinit(self.allocator);
        self.combining_marks.deinit(self.allocator);
        self.non_spacing_marks.deinit(self.allocator);
        self.nfc_check.deinit(self.allocator);
        self.possibly_valid.deinit(self.allocator);
        self.unique_non_confusables.deinit(self.allocator);

        // Free maps
        var fenced_iter = self.fenced.valueIterator();
        while (fenced_iter.next()) |value| {
            self.allocator.free(value.*);
        }
        self.fenced.deinit();

        var mapped_iter = self.mapped.valueIterator();
        while (mapped_iter.next()) |value| {
            self.allocator.free(value.*);
        }
        self.mapped.deinit();

        var confusables_iter = self.confusables.valueIterator();
        while (confusables_iter.next()) |_| {
            // Values in confusables map are references to wholes, not owned
        }
        self.confusables.deinit();

        // Free groups array
        for (self.groups) |group| {
            self.allocator.free(group.name);
            group.primary.deinit(self.allocator);
            group.secondary.deinit(self.allocator);
        }
        self.allocator.free(self.groups);

        // Free emojis array
        for (self.emojis) |emoji| {
            self.allocator.free(emoji.normalized);
            self.allocator.free(emoji.beautified);
        }
        self.allocator.free(self.emojis);

        // Free emoji tree
        if (self.emoji_root) |root| {
            init_mod.freeEmojiTree(root, self.allocator);
        }

        // Free wholes array
        for (self.wholes) |*whole| {
            whole.valid.deinit(self.allocator);
            whole.confused.deinit(self.allocator);
            var comp_iter = whole.complements.valueIterator();
            while (comp_iter.next()) |value| {
                self.allocator.free(value.*);
            }
            whole.complements.deinit();
        }
        self.allocator.free(self.wholes);

        // Free special groups
        if (self._ASCII) |ascii| {
            var ascii_mut = @constCast(ascii);
            ascii_mut.primary.deinit(self.allocator);
            self.allocator.destroy(ascii_mut);
        }
        if (self._EMOJI) |emoji| {
            self.allocator.destroy(@constCast(emoji));
        }
    }

    // ============================================================
    // Public API - Normalization Methods
    // ============================================================

    /// Normalize a name according to ENSIP15 specification
    ///
    /// Takes an input name and returns its normalized form.
    /// The normalized form is suitable for on-chain storage and comparison.
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for result
    ///   - name: Input name as UTF-8 bytes
    ///
    /// Returns: Normalized name as UTF-8 bytes
    ///
    /// Errors: See Error enum for all possible validation failures
    ///
    /// Example:
    ///   const result = try ensip15.normalize(allocator, "Nick.ETH");
    ///   defer allocator.free(result);
    ///   // result should be "nick.eth"
    ///
    /// Note: Currently stubbed with @panic
    pub fn normalize(self: *const Ensip15, allocator: Allocator, name: []const u8) ![]u8 {
        return self.transform(
            allocator,
            name,
            nfcWrapper,
            emojiNormalized,
            normalizerNormalize,
        );
    }

    /// Beautify a name according to ENSIP15 specification
    ///
    /// Similar to normalize() but produces a more visually appealing result.
    /// Uses beautified emoji forms (preserves FE0F variation selectors).
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for result
    ///   - name: Input name as UTF-8 bytes
    ///
    /// Returns: Beautified name as UTF-8 bytes
    ///
    /// Errors: See Error enum for all possible validation failures
    ///
    /// Example:
    ///   const result = try ensip15.beautify(allocator, "nick.eth");
    ///   defer allocator.free(result);
    ///
    /// Note: Currently stubbed with @panic
    pub fn beautify(self: *const Ensip15, allocator: Allocator, name: []const u8) ![]u8 {
        return self.transform(
            allocator,
            name,
            nfcWrapper,
            emojiBeautified,
            normalizerBeautify,
        );
    }

    // ============================================================
    // Internal Transformation Pipeline
    // ============================================================

    /// Internal transformation pipeline
    ///
    /// Orchestrates the normalization pipeline:
    /// 1. Split name by dots into labels
    /// 2. For each label:
    ///    a. Convert to codepoints
    ///    b. Tokenize into Text/Emoji tokens
    ///    c. Apply normalization function to text tokens
    ///    d. Apply emoji function to emoji tokens
    ///    e. Run normalizer function (validation + flattening)
    ///    f. Replace label with normalized form
    /// 3. Join labels back with dots
    ///
    /// This method is used by:
    /// - normalize() - uses NFC + normalized emoji
    /// - beautify() - uses NFC + beautified emoji
    /// - normalizeFragment() - uses NFC/NFD + normalized emoji
    ///
    /// Note: Currently stubbed with @panic
    fn transform(
        self: *const Ensip15,
        allocator: Allocator,
        name: []const u8,
        nf_fn: *const fn (*const NF, Allocator, []const u21) anyerror![]u21,
        ef_fn: *const fn (*const EmojiSequence) []const u21,
        normalizer_fn: *const fn (*const Ensip15, Allocator, []const OutputToken) anyerror![]u21,
    ) ![]u8 {
        // Split name by dots into labels
        const labels = try utils.split(allocator, name);
        defer allocator.free(labels);

        // Process each label
        var normalized_labels: std.ArrayListUnmanaged([]u8) = .{};
        defer {
            for (normalized_labels.items) |label| allocator.free(label);
            normalized_labels.deinit(allocator);
        }

        for (labels) |label| {
            // Convert UTF-8 to UTF-32
            const cps = try utf8ToUtf32(allocator, label);
            defer allocator.free(cps);

            // Tokenize
            const tokens = try self.outputTokenize(allocator, cps, nf_fn, ef_fn);
            defer {
                for (tokens) |token| allocator.free(token.codepoints);
                allocator.free(tokens);
            }

            // Normalize and validate
            const normalized_cps = try normalizer_fn(self, allocator, tokens);
            defer allocator.free(normalized_cps);

            // Convert back to UTF-8
            const normalized_label = try utf32ToUtf8(allocator, normalized_cps);
            try normalized_labels.append(allocator, normalized_label);
        }

        // Join labels with dots
        return try utils.join(allocator, normalized_labels.items);
    }

    /// Tokenize codepoints into OutputToken stream
    ///
    /// Takes a slice of codepoints and produces a sequence of OutputTokens.
    /// Each token is either:
    /// - A text token (codepoints that are not emoji)
    /// - An emoji token (recognized emoji sequence)
    ///
    /// The function applies:
    /// - NFC normalization to text tokens (via nf parameter)
    /// - Emoji normalization to emoji tokens (via ef parameter)
    /// - Filtering of ignored codepoints
    /// - Mapping of mapped codepoints
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for output
    ///   - cps: Input codepoints
    ///   - nf: Normalization function (NFC or NFD)
    ///   - ef: Emoji extraction function (normalized or beautified)
    ///
    /// Returns: Slice of OutputTokens
    ///
    /// Note: Currently stubbed with @panic
    fn outputTokenize(
        self: *const Ensip15,
        allocator: Allocator,
        cps: []const u21,
        nf_fn: *const fn (*const NF, Allocator, []const u21) anyerror![]u21,
        ef_fn: *const fn (*const EmojiSequence) []const u21,
    ) ![]OutputToken {
        var tokens: std.ArrayListUnmanaged(OutputToken) = .{};
        errdefer {
            for (tokens.items) |token| {
                allocator.free(token.codepoints);
            }
            tokens.deinit(allocator);
        }

        var buf: std.ArrayListUnmanaged(u21) = .{};
        defer buf.deinit(allocator);

        var i: usize = 0;
        while (i < cps.len) {
            if (self.parseEmojiAt(cps, i)) |result| {
                // Flush text buffer
                if (buf.items.len > 0) {
                    const normalized = try nf_fn(self.nf.?, allocator, buf.items);
                    try tokens.append(allocator, OutputToken{
                        .codepoints = normalized,
                        .emoji = null,
                    });
                    buf.clearRetainingCapacity();
                }

                // Add emoji token
                const emoji_cps = ef_fn(result.emoji);
                const owned_cps = try allocator.dupe(u21, emoji_cps);
                try tokens.append(allocator, OutputToken{
                    .codepoints = owned_cps,
                    .emoji = result.emoji,
                });

                i = result.end;
            } else {
                const cp = cps[i];
                if (self.possibly_valid.contains(cp)) {
                    try buf.append(allocator, cp);
                } else if (self.mapped.get(cp)) |mapped| {
                    try buf.appendSlice(allocator, mapped);
                } else if (!self.ignored.contains(cp)) {
                    return Error.DisallowedCharacter;
                }
                i += 1;
            }
        }

        // Flush remaining buffer
        if (buf.items.len > 0) {
            const normalized = try nf_fn(self.nf.?, allocator, buf.items);
            try tokens.append(allocator, OutputToken{
                .codepoints = normalized,
                .emoji = null,
            });
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// Parse emoji sequence starting at position in codepoint array
    ///
    /// Walks the emoji trie to find the longest matching emoji sequence
    /// starting at the given position.
    ///
    /// Parameters:
    ///   - cps: Codepoint array to search
    ///   - pos: Starting position in array
    ///
    /// Returns: Struct with emoji sequence and end position, or null if no match
    fn parseEmojiAt(
        self: *const Ensip15,
        cps: []const u21,
        pos: usize,
    ) ?struct { emoji: *const EmojiSequence, end: usize } {
        var node = self.emoji_root orelse return null;
        var current_pos = pos;
        var result: ?*const EmojiSequence = null;
        var result_end: usize = 0;

        while (current_pos < cps.len) {
            if (node.children) |children| {
                if (children.get(cps[current_pos])) |child| {
                    node = child;
                    current_pos += 1;
                    if (node.emoji) |e| {
                        result = e;
                        result_end = current_pos;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (result) |e| {
            return .{ .emoji = e, .end = result_end };
        }
        return null;
    }

    // ============================================================
    // Validation Functions - Standalone
    // ============================================================

    /// Check that underscores only appear at start of label
    ///
    /// Rule: Underscores must match regex /^_*[^_]*$/
    /// This means: zero or more underscores, followed by zero or more non-underscores
    ///
    /// Valid examples:
    ///   - "_test"    (leading underscore)
    ///   - "__abc"    (multiple leading underscores)
    ///   - "hello"    (no underscores)
    ///   - "___"      (only underscores)
    ///
    /// Invalid examples:
    ///   - "ab_c"     (underscore in middle)
    ///   - "test_"    (trailing underscore)
    ///   - "_a_b"     (underscore after non-underscore)
    ///
    /// Algorithm:
    /// 1. Start with allowed = true
    /// 2. For each codepoint:
    ///    - If allowed and cp != underscore: allowed = false
    ///    - If not allowed and cp == underscore: ERROR
    fn checkLeadingUnderscore(cps: []const u21) !void {
        const UNDERSCORE: u21 = 0x5F;
        var allowed = true;
        for (cps) |cp| {
            if (allowed) {
                if (cp != UNDERSCORE) {
                    allowed = false;
                }
            } else {
                if (cp == UNDERSCORE) {
                    return Error.LeadingUnderscore;
                }
            }
        }
    }

    /// Check label extension format
    ///
    /// Rule: The 3rd and 4th characters (indices 2-3) cannot both be hyphens
    /// This prevents confusion with ACE prefix format (xn--).
    ///
    /// Valid examples:
    ///   - "ab-cd"    (single hyphen)
    ///   - "abc--d"   (hyphens not at positions 2-3)
    ///   - "-abc-"    (hyphens at other positions)
    ///   - "abc"      (no hyphens)
    ///
    /// Invalid examples:
    ///   - "xn--test" (positions 2-3 are both hyphens)
    ///   - "ab--cd"   (positions 2-3 are both hyphens)
    ///
    /// Algorithm:
    /// 1. If length < 4: valid (skip check)
    /// 2. If cps[2] == hyphen AND cps[3] == hyphen: ERROR
    fn checkLabelExtension(cps: []const u21) !void {
        const HYPHEN: u21 = 0x2D;
        if (cps.len >= 4 and cps[2] == HYPHEN and cps[3] == HYPHEN) {
            return Error.InvalidLabelExtension;
        }
    }

    // ============================================================
    // Validation Functions - Methods
    // ============================================================

    /// Check combining mark placement rules
    ///
    /// Combining marks cannot:
    /// 1. Appear at the start of a label
    /// 2. Appear immediately after an emoji
    ///
    /// This validation operates on the token stream (not raw codepoints)
    /// because we need to know which tokens are emoji.
    ///
    /// Algorithm:
    /// 1. For each token in tokens:
    ///    a. If token is text (not emoji):
    ///       - If first codepoint is combining mark:
    ///         - If token is first (i == 0): ERROR (CM at start)
    ///         - Else if previous token is emoji: ERROR (CM after emoji)
    fn checkCombiningMarks(self: *const Ensip15, tokens: []const OutputToken) !void {
        for (tokens, 0..) |token, i| {
            if (token.emoji == null and token.codepoints.len > 0) {
                const cp = token.codepoints[0];
                if (self.combining_marks.contains(cp)) {
                    if (i == 0) {
                        return Error.CMLeading;
                    } else if (tokens[i - 1].emoji != null) {
                        return Error.CMAfterEmoji;
                    }
                }
            }
        }
    }

    /// Check fenced character placement (ZWJ/ZWNJ)
    ///
    /// Fenced characters (Zero Width Joiner and Zero Width Non-Joiner)
    /// have special placement rules. They cannot:
    /// 1. Appear at the start of a label
    /// 2. Appear at the end of a label
    /// 3. Appear adjacent to each other
    ///
    /// The self.fenced map contains the fenced codepoints and their names.
    ///
    /// Algorithm:
    /// 1. If first codepoint is fenced: ERROR (fenced leading)
    /// 2. Track lastPos = -1 and lastName
    /// 3. For each codepoint (starting at index 1):
    ///    a. If codepoint is fenced:
    ///       - If lastPos == current index: ERROR (fenced adjacent)
    ///       - Update lastPos = current index + 1
    ///       - Update lastName
    /// 4. If lastPos == length: ERROR (fenced trailing)
    fn checkFenced(self: *const Ensip15, cps: []const u21) !void {
        if (cps.len == 0) return;

        // Check first character
        if (self.fenced.get(cps[0])) |_| {
            return Error.FencedLeading;
        }

        var last_pos: i32 = -1;
        for (cps[1..], 1..) |cp, i| {
            if (self.fenced.get(cp)) |_| {
                if (last_pos == @as(i32, @intCast(i))) {
                    return Error.FencedAdjacent;
                }
                last_pos = @intCast(i + 1);
            }
        }

        if (last_pos == @as(i32, @intCast(cps.len))) {
            return Error.FencedTrailing;
        }
    }

    /// Orchestrate all label validation checks
    ///
    /// This is the main validation function that coordinates all checks:
    /// 1. Check for empty label
    /// 2. Check underscore rules
    /// 3. Determine label type (ASCII, Emoji, or Script)
    /// 4. For ASCII: check label extension
    /// 5. For Emoji-only: return EMOJI group
    /// 6. For Script/Mixed: check combining marks, fenced chars, groups, confusables
    ///
    /// Returns: The Group this label belongs to (ASCII, EMOJI, or script group)
    ///
    /// Algorithm:
    /// 1. If cps is empty: ERROR (empty label)
    /// 2. Check leading underscore rule
    /// 3. Determine if hasEmoji (tokens > 1 or first token is emoji)
    /// 4. If no emoji and all ASCII:
    ///    a. Check label extension
    ///    b. Return ASCII group
    /// 5. Extract chars (non-emoji codepoints)
    /// 6. If has emoji and no chars: Return EMOJI group
    /// 7. Check combining marks on tokens
    /// 8. Check fenced characters on cps
    /// 9. Get unique chars
    /// 10. Determine script group from unique chars
    /// 11. Check group-specific rules
    /// 12. Check whole confusables
    /// 13. Return group
    fn checkValidLabel(
        self: *const Ensip15,
        allocator: Allocator,
        cps: []const u21,
        tokens: []const OutputToken,
    ) !?*const Group {
        if (cps.len == 0) return Error.EmptyLabel;

        try checkLeadingUnderscore(cps);

        const has_emoji = tokens.len > 1 or tokens[0].emoji != null;
        if (!has_emoji and utils.isAscii(cps)) {
            try checkLabelExtension(cps);
            return self._ASCII;
        }

        // Extract non-emoji chars
        var chars: std.ArrayListUnmanaged(u21) = .{};
        defer chars.deinit(allocator);

        for (tokens) |token| {
            if (token.emoji == null) {
                try chars.appendSlice(allocator, token.codepoints);
            }
        }

        if (has_emoji and chars.items.len == 0) {
            return self._EMOJI;
        }

        try self.checkCombiningMarks(tokens);
        try self.checkFenced(cps);

        const unique = try utils.uniqueRunes(allocator, chars.items);
        defer allocator.free(unique);

        const group = try self.determineGroup(unique, allocator);
        try self.checkGroup(group, chars.items, allocator);
        try self.checkWhole(group, unique, allocator);

        return group;
    }

    // ============================================================
    // Helper Methods (stubs for future implementation)
    // ============================================================

    /// Determine which script group a set of codepoints belongs to
    fn determineGroup(self: *const Ensip15, unique: []const u21, allocator: Allocator) !*const Group {
        // Clone groups array
        var gs = try allocator.alloc(*const Group, self.groups.len);
        defer allocator.free(gs);

        for (self.groups, 0..) |*g, i| {
            gs[i] = g;
        }

        var prev = gs.len;
        for (unique) |cp| {
            var next: usize = 0;
            for (0..prev) |i| {
                if (gs[i].contains(cp)) {
                    gs[next] = gs[i];
                    next += 1;
                }
            }

            if (next == 0) {
                return Error.DisallowedCharacter;
            }

            prev = next;
            if (prev == 1) break;
        }

        return gs[0];
    }

    /// Check group-specific validation rules
    fn checkGroup(self: *const Ensip15, group: *const Group, chars: []const u21, allocator: Allocator) !void {
        // Verify all chars in group
        for (chars) |cp| {
            if (!group.contains(cp)) {
                return Error.IllegalMixture;
            }
        }

        // Check NSM if not CM whitelisted
        if (!group.cm_whitelisted) {
            if (self.nf) |nf| {
                const decomposed = try nf.nfd(allocator, chars);
                defer allocator.free(decomposed);

                var i: usize = 1;
                while (i < decomposed.len) {
                    if (self.non_spacing_marks.contains(decomposed[i])) {
                        var j = i + 1;
                        while (j < decomposed.len) : (j += 1) {
                            const cp = decomposed[j];
                            if (!self.non_spacing_marks.contains(cp)) break;

                            // Check for duplicates
                            for (decomposed[i..j]) |prev_cp| {
                                if (prev_cp == cp) {
                                    return Error.NSMDuplicate;
                                }
                            }
                        }

                        const n = j - i;
                        if (n > self.max_non_spacing_marks) {
                            return Error.NSMExcessive;
                        }

                        i = j;
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    /// Check for whole confusable sequences
    ///
    /// Algorithm:
    /// 1. For each unique codepoint:
    ///    - If confusable: intersect its complement list with universe
    ///    - If unique non-confusable: return early (no confusable)
    ///    - Otherwise: add to shared list
    /// 2. Check if any group in universe contains ALL shared codepoints
    /// 3. If yes: return WholeConfusable error
    ///
    /// Go reference: wholes.go lines 86-130
    fn checkWhole(self: *const Ensip15, group: *const Group, unique: []const u21, allocator: Allocator) !void {
        _ = group; // Used for error reporting in Go, but Zig error enum doesn't carry context
        var shared: std.ArrayListUnmanaged(u21) = .{};
        defer shared.deinit(allocator);

        var universe: std.ArrayListUnmanaged(i32) = .{};
        defer universe.deinit(allocator);

        var prev: usize = 0;

        // Process each unique codepoint
        for (unique) |cp| {
            if (self.confusables.get(cp)) |whole| {
                // This is a confusable codepoint
                const comp = whole.complements.get(cp) orelse &[_]i32{};

                if (prev == 0) {
                    // First confusable: initialize universe
                    try universe.appendSlice(allocator, comp);
                    prev = comp.len;
                } else {
                    // Subsequent confusables: intersect with universe
                    var next: usize = 0;
                    for (0..prev) |i| {
                        // Check if universe[i] exists in comp (binary search)
                        if (std.mem.indexOfScalar(i32, comp, universe.items[i])) |_| {
                            universe.items[next] = universe.items[i];
                            next += 1;
                        }
                    }
                    prev = next;
                }

                // If universe is empty, no possible confusables
                if (prev == 0) {
                    return;
                }
            } else if (self.unique_non_confusables.contains(cp)) {
                // Unique non-confusable breaks confusability
                return;
            } else {
                // Shared codepoint (exists in multiple groups)
                try shared.append(allocator, cp);
            }
        }

        // Check if any group in universe contains all shared codepoints
        if (prev > 0) {
            next: for (0..prev) |i| {
                const other = &self.groups[@intCast(universe.items[i])];

                // Check if other group contains ALL shared codepoints
                for (shared.items) |cp| {
                    if (!other.contains(cp)) {
                        continue :next;
                    }
                }

                // Found a confusable group!
                return Error.WholeConfusable;
            }
        }
    }

    // ============================================================
    // Helper Functions - UTF Conversion
    // ============================================================

    /// Convert UTF-8 string to UTF-32 codepoint array
    fn utf8ToUtf32(allocator: Allocator, utf8: []const u8) ![]u21 {
        var result: std.ArrayListUnmanaged(u21) = .{};
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < utf8.len) {
            const len = try std.unicode.utf8ByteSequenceLength(utf8[i]);
            const codepoint = try std.unicode.utf8Decode(utf8[i..][0..len]);
            try result.append(allocator, codepoint);
            i += len;
        }
        return result.toOwnedSlice(allocator);
    }

    /// Convert UTF-32 codepoint array to UTF-8 string
    fn utf32ToUtf8(allocator: Allocator, utf32: []const u21) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (utf32) |cp| {
            var buf: [4]u8 = undefined;
            const len = try std.unicode.utf8Encode(cp, &buf);
            try result.appendSlice(allocator, buf[0..len]);
        }
        return result.toOwnedSlice(allocator);
    }

    /// Convert codepoint to safe display string
    /// Note: Stubbed for future implementation
    fn safeCodepoint(self: *const Ensip15, cp: u21) []const u8 {
        _ = self;
        _ = cp;
        @panic("TODO: implement safeCodepoint()");
    }

    /// Convert codepoints to safe display string
    /// Note: Stubbed for future implementation
    fn safeImplode(self: *const Ensip15, cps: []const u21) []const u8 {
        _ = self;
        _ = cps;
        @panic("TODO: implement safeImplode()");
    }
};

// ============================================================
// Callback Wrapper Functions
// ============================================================

/// Wrapper for NF.nfc() to match expected function signature
fn nfcWrapper(nf: *const NF, allocator: Allocator, cps: []const u21) ![]u21 {
    return nf.nfc(allocator, cps);
}

/// Extract normalized form from emoji sequence
fn emojiNormalized(emoji: *const EmojiSequence) []const u21 {
    return emoji.normalized;
}

/// Extract beautified form from emoji sequence
fn emojiBeautified(emoji: *const EmojiSequence) []const u21 {
    return emoji.beautified;
}

/// Normalizer function for normalize() - validates and returns codepoints as-is
fn normalizerNormalize(
    self: *const Ensip15,
    allocator: Allocator,
    tokens: []const OutputToken,
) ![]u21 {
    const cps = try utils.flattenTokens(allocator, tokens);
    errdefer allocator.free(cps);

    _ = try self.checkValidLabel(allocator, cps, tokens);

    return cps;
}

/// Normalizer function for beautify() - validates and applies Greek XI beautification
fn normalizerBeautify(
    self: *const Ensip15,
    allocator: Allocator,
    tokens: []const OutputToken,
) ![]u21 {
    const cps_const = try utils.flattenTokens(allocator, tokens);
    errdefer allocator.free(cps_const);

    const group = try self.checkValidLabel(allocator, cps_const, tokens);

    // Apply Greek XI beautification (U+03BE -> U+039E) if not Greek group
    if (group != self._GREEK) {
        // Need to make a mutable copy
        const cps = try allocator.dupe(u21, cps_const);
        allocator.free(cps_const);
        errdefer allocator.free(cps);

        for (cps) |*cp| {
            if (cp.* == 0x3BE) { // Greek lowercase xi
                cp.* = 0x39E; // Greek uppercase XI
            }
        }
        return cps;
    }

    return cps_const;
}

// ============================================================
// Utility Functions (imported from utils.zig)
// ============================================================
// The following utility functions are provided by utils module:
// - utils.split() - Split name by dots into labels
// - utils.join() - Join labels with dots
// - utils.flattenTokens() - Flatten tokens into codepoint slice
// - utils.isAscii() - Check if all codepoints are ASCII
// - utils.uniqueRunes() - Get unique codepoints preserving order

// ============================================================
// Tests
// ============================================================

test "Ensip15 init and deinit" {
    const allocator = std.testing.allocator;
    var ensip15 = try Ensip15.init(allocator);
    defer ensip15.deinit();
    // Just verify it compiles
}

test "checkLeadingUnderscore - valid cases" {
    // Valid: leading underscores only
    try Ensip15.checkLeadingUnderscore(&[_]u21{ 0x5F, 0x5F, 0x61 }); // "__a"
    try Ensip15.checkLeadingUnderscore(&[_]u21{ 0x61, 0x62 }); // "ab"
    try Ensip15.checkLeadingUnderscore(&[_]u21{0x5F}); // "_"
    try Ensip15.checkLeadingUnderscore(&[_]u21{}); // empty
}

test "checkLeadingUnderscore - invalid cases" {
    // Invalid: underscore after non-underscore
    const result = Ensip15.checkLeadingUnderscore(&[_]u21{ 0x61, 0x5F }); // "a_"
    try std.testing.expectError(Error.LeadingUnderscore, result);

    const result2 = Ensip15.checkLeadingUnderscore(&[_]u21{ 0x5F, 0x61, 0x5F }); // "_a_"
    try std.testing.expectError(Error.LeadingUnderscore, result2);
}

test "checkLabelExtension - valid cases" {
    // Valid: not xn-- pattern
    try Ensip15.checkLabelExtension(&[_]u21{ 0x61, 0x62, 0x63 }); // "abc"
    try Ensip15.checkLabelExtension(&[_]u21{ 0x61, 0x62, 0x2D, 0x63 }); // "ab-c"
    try Ensip15.checkLabelExtension(&[_]u21{ 0x61, 0x62 }); // "ab" (too short)
    try Ensip15.checkLabelExtension(&[_]u21{ 0x2D, 0x2D, 0x2D, 0x61 }); // "---a"
}

test "checkLabelExtension - invalid cases" {
    // Invalid: xn-- pattern
    const result = Ensip15.checkLabelExtension(&[_]u21{ 0x78, 0x6E, 0x2D, 0x2D }); // "xn--"
    try std.testing.expectError(Error.InvalidLabelExtension, result);

    const result2 = Ensip15.checkLabelExtension(&[_]u21{ 0x61, 0x62, 0x2D, 0x2D, 0x63 }); // "ab--c"
    try std.testing.expectError(Error.InvalidLabelExtension, result2);
}
