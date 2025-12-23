// ============================================================================
// Refinement Subtyping
// ============================================================================
// Phase 1: Extract subtyping rules
// ============================================================================

const OraType = @import("../../type_info.zig").OraType;

/// Check refinement subtyping rules
/// Returns true if source is a subtype of target (source <: target)
pub fn checkRefinementSubtyping(
    source: OraType,
    target: OraType,
    comptime isBaseTypeCompatible: fn (OraType, OraType) bool,
) bool {
    return switch (source) {
        .min_value => |smv| switch (target) {
            .min_value => |tmv| OraType.equals(smv.base.*, tmv.base.*) and smv.min >= tmv.min,
            else => OraType.equals(source, target) or isBaseTypeCompatible(smv.base.*, target),
        },
        .max_value => |smv| switch (target) {
            .max_value => |tmv| OraType.equals(smv.base.*, tmv.base.*) and smv.max <= tmv.max,
            else => OraType.equals(source, target) or isBaseTypeCompatible(smv.base.*, target),
        },
        .in_range => |sir| switch (target) {
            .in_range => |tir| OraType.equals(sir.base.*, tir.base.*) and sir.min >= tir.min and sir.max <= tir.max,
            .min_value => |tmv| OraType.equals(sir.base.*, tmv.base.*) and sir.min >= tmv.min,
            .max_value => |tmv| OraType.equals(sir.base.*, tmv.base.*) and sir.max <= tmv.max,
            else => OraType.equals(source, target) or isBaseTypeCompatible(sir.base.*, target),
        },
        .scaled => |ss| switch (target) {
            .scaled => |ts| OraType.equals(ss.base.*, ts.base.*) and ss.decimals == ts.decimals,
            else => OraType.equals(source, target) or isBaseTypeCompatible(ss.base.*, target),
        },
        .exact => |se| switch (target) {
            .exact => |te| isBaseTypeCompatible(se.*, te.*),
            else => OraType.equals(source, target) or isBaseTypeCompatible(se.*, target),
        },
        .non_zero_address => switch (target) {
            .non_zero_address => true,
            .address => true, // NonZeroAddress <: address
            else => false,
        },
        else => false,
    };
}
