//! Canonical obligation manifest to Z3 adapter public surface.
//!
//! The implementation is split into canonical_types, canonical_support, and
//! canonical_encode. Keep this file as the stable import point for callers.

const canonical_types = @import("canonical_types.zig");
const canonical_support = @import("canonical_support.zig");
const canonical_encode = @import("canonical_encode.zig");

pub const CheckStatus = canonical_types.CheckStatus;
pub const QueryHash = canonical_types.QueryHash;
pub const CanonicalSupport = canonical_types.CanonicalSupport;
pub const CanonicalUnsupportedReason = canonical_types.CanonicalUnsupportedReason;
pub const CanonicalPromotionShape = canonical_types.CanonicalPromotionShape;
pub const CanonicalPromotionPolicy = canonical_types.CanonicalPromotionPolicy;
pub const canonical_promotion_table = canonical_types.canonical_promotion_table;
pub const EncodeError = canonical_types.EncodeError;

pub const Adapter = canonical_encode.Adapter;

pub const queryCanonicalSupport = canonical_support.queryCanonicalSupport;
pub const queryCanonicalPromotionShape = canonical_support.queryCanonicalPromotionShape;
pub const queryCanonicalRequiredModePromoted = canonical_support.queryCanonicalRequiredModePromoted;
pub const queryCanonicalCrosscheckRequiredByPolicy = canonical_support.queryCanonicalCrosscheckRequiredByPolicy;

comptime {
    _ = canonical_types;
    _ = canonical_support;
    _ = canonical_encode;
}
