//! ENSIP15 Error Types
//!
//! This file defines all possible errors that can occur during
//! ENS name normalization according to ENSIP15 specification.

const std = @import("std");

/// All possible normalization errors
pub const Error = error{
    /// Empty label (consecutive dots or leading/trailing dot)
    EmptyLabel,

    /// Underscore appears after non-underscore character
    LeadingUnderscore,

    /// Characters 3-4 are both hyphens (like xn--)
    InvalidLabelExtension,

    /// Combining mark at start of label
    CMLeading,

    /// Combining mark after emoji
    CMAfterEmoji,

    /// Fenced character (ZWJ/ZWNJ) at start of label
    FencedLeading,

    /// Fenced character at end of label
    FencedTrailing,

    /// Fenced characters adjacent to each other
    FencedAdjacent,

    /// Codepoint not allowed in any group
    DisallowedCharacter,

    /// Characters from multiple conflicting groups
    MixedGroups,

    /// Whole confusable sequence detected
    WholeConfusable,

    /// Duplicate non-spacing marks in sequence
    NSMDuplicate,

    /// Excessive non-spacing marks (exceeds maximum)
    NSMExcessive,

    /// Illegal mixture of characters from incompatible groups
    IllegalMixture,

    /// Other validation errors
    InvalidCodepoint,
    InvalidEmoji,
    InvalidNormalization,
    InvalidGroup,
};
