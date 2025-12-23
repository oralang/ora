// ============================================================================
// Type Extraction Helpers
// ============================================================================
// Phase 1: Extract base type logic
// ============================================================================

const OraType = @import("../../type_info.zig").OraType;

/// Extract base type from a refined type
/// Single source of truth for type extraction
pub fn extractBaseType(ora_type: OraType) ?OraType {
    return switch (ora_type) {
        .min_value => |mv| mv.base.*,
        .max_value => |mv| mv.base.*,
        .in_range => |ir| ir.base.*,
        .scaled => |s| s.base.*,
        .exact => |e| e.*,
        .non_zero_address => .address, // Base type is address
        else => ora_type, // Not a refinement, return as-is
    };
}
