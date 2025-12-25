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
            // base type must be an integer type
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // recursively validate base type
            try validateRefinementType(cfg, registry, mv.base.*);
        },
        .max_value => |mv| {
            // base type must be an integer type
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // recursively validate base type
            try validateRefinementType(cfg, registry, mv.base.*);
        },
        .in_range => |ir| {
            // base type must be an integer type
            if (!ir.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // min <= MAX already validated in parser, but double-check
            if (ir.min > ir.max) {
                return TypeResolutionError.TypeMismatch;
            }
            // recursively validate base type
            try validateRefinementType(cfg, registry, ir.base.*);
        },
        .scaled => |s| {
            // base type must be an integer type
            if (!s.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // note: Removed hardcoded decimals > 77 constraint
            // if needed, add to RefinementConfig
            // recursively validate base type
            try validateRefinementType(cfg, registry, s.base.*);
        },
        .exact => |e| {
            // base type must be an integer type
            if (!e.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // recursively validate base type
            try validateRefinementType(cfg, registry, e.*);
        },
        .non_zero_address => {
            // nonZeroAddress is always valid - it's a refinement of address type
            // no base type to validate
        },
        else => {
            // not a refinement type, no validation needed
        },
    }
}
