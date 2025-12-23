// ============================================================================
// Refinement Validation
// ============================================================================
// Phase 1: Extract validation logic using registry
// ============================================================================

const std = @import("std");
const OraType = @import("../../type_info.zig").OraType;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const RefinementConfig = @import("registry.zig").RefinementConfig;

/// Validate a refinement type using registry handlers
pub fn validateRefinementType(
    cfg: *const RefinementConfig,
    registry: *const @import("registry.zig").RefinementRegistry,
    ora_type: OraType,
) TypeResolutionError!void {
    switch (ora_type) {
        .min_value => |mv| {
            // Base type must be an integer type
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // Recursively validate base type
            try validateRefinementType(cfg, registry, mv.base.*);
        },
        .max_value => |mv| {
            // Base type must be an integer type
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // Recursively validate base type
            try validateRefinementType(cfg, registry, mv.base.*);
        },
        .in_range => |ir| {
            // Base type must be an integer type
            if (!ir.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // MIN <= MAX already validated in parser, but double-check
            if (ir.min > ir.max) {
                return TypeResolutionError.TypeMismatch;
            }
            // Recursively validate base type
            try validateRefinementType(cfg, registry, ir.base.*);
        },
        .scaled => |s| {
            // Base type must be an integer type
            if (!s.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // Note: Removed hardcoded decimals > 77 constraint
            // If needed, add to RefinementConfig
            // Recursively validate base type
            try validateRefinementType(cfg, registry, s.base.*);
        },
        .exact => |e| {
            // Base type must be an integer type
            if (!e.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // Recursively validate base type
            try validateRefinementType(cfg, registry, e.*);
        },
        .non_zero_address => {
            // NonZeroAddress is always valid - it's a refinement of address type
            // No base type to validate
        },
        else => {
            // Not a refinement type, no validation needed
        },
    }
}
