//! ENSIP15 Type Definitions
//!
//! This file contains the core types used in ENSIP15 normalization.
//! These types represent tokens, groups, emoji sequences, and other structures
//! needed for the normalization pipeline.

const std = @import("std");
const RuneSet = @import("../util/runeset.zig").RuneSet;

/// Represents a single token in the output stream
/// Can be either a text token (codepoints) or an emoji token
pub const OutputToken = struct {
    /// Unicode codepoints for this token (if text)
    codepoints: []const u21,

    /// Pointer to emoji sequence (if emoji, null otherwise)
    emoji: ?*const EmojiSequence,

    pub fn init(codepoints: []const u21) OutputToken {
        return .{
            .codepoints = codepoints,
            .emoji = null,
        };
    }

    pub fn initEmoji(emoji: *const EmojiSequence) OutputToken {
        return .{
            .codepoints = &.{},
            .emoji = emoji,
        };
    }
};

/// Represents an emoji sequence with normalized and beautified forms
pub const EmojiSequence = struct {
    /// Normalized form (with FE0F stripped)
    normalized: []const u21,

    /// Beautified form (with FE0F preserved where appropriate)
    beautified: []const u21,
};

/// Represents a script group (e.g., Latin, Greek, Han)
pub const Group = struct {
    /// Index in the groups array (-1 for special groups like ASCII, EMOJI)
    index: i32,

    /// Whether this group is restricted
    restricted: bool,

    /// Name of the group (e.g., "Latin", "Greek")
    name: []const u8,

    /// Whether combining marks are whitelisted for this group
    cm_whitelisted: bool,

    /// Primary codepoints for this group
    primary: RuneSet,

    /// Secondary codepoints for this group
    secondary: RuneSet,

    /// Check if this group contains a codepoint (in primary or secondary)
    pub fn contains(self: *const Group, cp: u21) bool {
        return self.primary.contains(cp) or self.secondary.contains(cp);
    }
};

/// Represents a whole confusable sequence
pub const Whole = struct {
    /// Valid codepoints for this whole
    valid: RuneSet,

    /// Confused codepoints
    confused: RuneSet,

    /// Map of codepoint to complement group indices
    complements: std.AutoHashMap(u21, []i32),
};

/// Node in the emoji trie tree
pub const EmojiNode = struct {
    /// Children nodes mapped by codepoint
    children: ?std.AutoHashMap(u21, *EmojiNode),

    /// The emoji sequence at this node (if this is a leaf)
    emoji: ?*const EmojiSequence,

    /// Get or create a child node for the given codepoint
    pub fn child(self: *EmojiNode, allocator: std.mem.Allocator, cp: u21) !*EmojiNode {
        if (self.children == null) {
            self.children = std.AutoHashMap(u21, *EmojiNode).init(allocator);
        }

        const result = try self.children.?.getOrPut(cp);
        if (!result.found_existing) {
            const new_node = try allocator.create(EmojiNode);
            new_node.* = EmojiNode{
                .children = null,
                .emoji = null,
            };
            result.value_ptr.* = new_node;
        }

        return result.value_ptr.*;
    }
};
