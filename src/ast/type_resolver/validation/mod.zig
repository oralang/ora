// ============================================================================
// Validation System
// ============================================================================
// Phase 2: Implement validation system
// ============================================================================

const std = @import("std");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");

const compat = @import("compatibility.zig");
const assignment = @import("assignment.zig");

pub const ValidationSystem = struct {
    refinements: *refinements.RefinementSystem,
    utils: *utils.Utils,

    pub fn init(
        refinement_sys: *refinements.RefinementSystem,
        utils_sys: *utils.Utils,
    ) ValidationSystem {
        return ValidationSystem{
            .refinements = refinement_sys,
            .utils = utils_sys,
        };
    }

    pub fn deinit(self: *ValidationSystem) void {
        _ = self;
        // phase 2: No cleanup needed
    }

    /// Check if source type is assignable to destination type
    pub fn isAssignable(self: *ValidationSystem, src: TypeInfo, dst: TypeInfo) bool {
        return assignment.isAssignable(dst, src, self.refinements, self.utils);
    }

    /// Check if two types are compatible
    pub fn areCompatible(self: *ValidationSystem, a: TypeInfo, b: TypeInfo) bool {
        return compat.areTypesCompatible(a, b, self.refinements);
    }

    /// Check if source is a subtype of destination
    pub fn isSubtype(self: *ValidationSystem, src: TypeInfo, dst: TypeInfo) bool {
        if (src.ora_type != null and dst.ora_type != null) {
            return self.refinements.checkSubtype(
                src.ora_type.?,
                dst.ora_type.?,
                compat.isBaseTypeCompatible,
            );
        }
        return false;
    }
};
