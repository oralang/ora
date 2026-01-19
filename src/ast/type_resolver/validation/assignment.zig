// ============================================================================
// Assignment Validation
// ============================================================================
// Phase 2: Extract assignment legality rules
// ============================================================================

const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");
const compat = @import("compatibility.zig");
const log = @import("log");

/// Check if a type can be assigned to a target type
/// Assignment allows widening conversions only (value_type <: target_type)
/// Narrowing conversions are rejected to prevent data loss
pub fn isAssignable(
    target_type: TypeInfo,
    value_type: TypeInfo,
    refinement_system: *refinements.RefinementSystem,
    utils_sys: *utils.Utils,
) bool {
    // exact match
    if (target_type.ora_type != null and value_type.ora_type != null) {
        if (OraType.equals(target_type.ora_type.?, value_type.ora_type.?)) {
            return true;
        }
    }

    // check if value_type is a subtype of target_type (widening allowed)
    // this is unidirectional: value_type <: target_type
    if (target_type.ora_type != null and value_type.ora_type != null) {
        // check refinement subtyping (value_type <: target_type)
        if (refinement_system.checkSubtype(
            value_type.ora_type.?,
            target_type.ora_type.?,
            compat.isBaseTypeCompatible,
        )) {
            return true;
        }
    }

    // allow base-to-refinement assignment with a runtime guard
    // (e.g., address -> NonZeroAddress, u256 -> MinValue<u256,1>)
    // guard insertion is handled later
    if (target_type.ora_type) |target_ora| {
        if (value_type.ora_type) |value_ora| {
            const target_base = utils_sys.extractBaseType(target_ora) orelse target_ora;
            const value_base = utils_sys.extractBaseType(value_ora) orelse value_ora;
            // Only allow if:
            // 1. Target is a refinement type (target_ora != target_base)
            // 2. Base types match
            // 3. Value is NOT already a refinement (refinement-to-refinement must be subtype-checked)
            const target_is_refinement = !OraType.equals(target_ora, target_base);
            const value_is_refinement = !OraType.equals(value_ora, value_base);
            if (target_is_refinement and !value_is_refinement and OraType.equals(value_base, target_base)) {
                return true;
            }

            // allow refinement-to-refinement assignment with a runtime guard
            // when refinement kinds are compatible (e.g., MinValue -> InRange)
            if (target_is_refinement and value_is_refinement and OraType.equals(value_base, target_base)) {
                if (isExactRefinement(target_ora) or isExactRefinement(value_ora)) {
                    // Exact is an orthogonal numeric refinement; allow on matching base.
                    return value_base.isInteger();
                }

                if (sameRefinementKind(target_ora, value_ora) or
                    (isNumericRefinement(target_ora) and isNumericRefinement(value_ora)))
                {
                    return true;
                }
            }
        }
    }

    // for numeric types, check base type compatibility (widening only)
    const target_is_numeric = isNumericType(target_type);
    const value_is_numeric = isNumericType(value_type);
    if (target_is_numeric and value_is_numeric) {
        // if target has ora_type but value doesn't, allow if categories match
        // this handles integer literals (category=Integer, no ora_type) assigned to typed variables
        if (target_type.ora_type != null and value_type.ora_type == null) {
            // integer literal can be assigned to any integer type
            return true;
        }

        if (target_type.ora_type != null and value_type.ora_type != null) {
            const target_ora = target_type.ora_type.?;
            const value_ora = value_type.ora_type.?;
            const target_base = utils_sys.extractBaseType(target_ora);
            const value_base = utils_sys.extractBaseType(value_ora);
            if (target_base != null and value_base != null) {
                // If BOTH are refinement types, don't fall through to base compatibility
                // (subtyping check at the top of the function should have handled compatible refinements)
                const target_is_refinement = !OraType.equals(target_ora, target_base.?);
                const value_is_refinement = !OraType.equals(value_ora, value_base.?);
                if (target_is_refinement and value_is_refinement) {
                    // Both are refinements - subtyping check already failed, reject
                    return false;
                }
                // only allow widening: value_base <: target_base
                // this means value_base must be narrower or equal to target_base
                if (compat.isBaseTypeCompatible(value_base.?, target_base.?)) {
                    return true;
                }
            }
        }
        // if target has no ora_type but value does, also allow (reverse case)
        if (target_type.ora_type == null and value_type.ora_type != null) {
            return true;
        }
        // if we can't extract base types, reject to be safe
        return false;
    }

    // handle error unions and other special cases
    // success type is compatible with ErrorUnion (e.g., bool is compatible with !bool | Error1)
    if (target_type.category == .ErrorUnion) {
        if (target_type.ora_type) |target_ora| {
            if (target_ora == .error_union) {
                const success_ora_type = target_ora.error_union.*;
                const success_category = success_ora_type.getCategory();
                if (value_type.ora_type) |value_ora| {
                    if (value_type.category == success_category and
                        (OraType.equals(value_ora, success_ora_type) or
                            refinement_system.checkSubtype(
                                value_ora,
                                success_ora_type,
                                compat.isBaseTypeCompatible,
                            )))
                    {
                        return true;
                    }
                } else if (value_type.category == success_category) {
                    return true;
                }
            } else if (target_ora == ._union) {
                // error union with explicit errors: !T | Error1 | Error2
                if (target_ora._union.len > 0) {
                    const first_type = target_ora._union[0];
                    if (first_type == .error_union) {
                        const success_ora_type = first_type.error_union.*;
                        const success_category = success_ora_type.getCategory();
                        if (value_type.ora_type) |value_ora| {
                            if (value_type.category == success_category and
                                (OraType.equals(value_ora, success_ora_type) or
                                    refinement_system.checkSubtype(
                                        value_ora,
                                        success_ora_type,
                                        compat.isBaseTypeCompatible,
                                    )))
                            {
                                return true;
                            }
                        } else if (value_type.category == success_category) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // error category is compatible with ErrorUnion category
    if (value_type.category == .Error and target_type.category == .ErrorUnion) {
        return true;
    }

    // enum category is compatible with ErrorUnion category (for error.X expressions)
    if (value_type.category == .Enum and target_type.category == .ErrorUnion) {
        return true;
    }

    // same category without ora_type - only allow if categories match exactly
    if (target_type.category == value_type.category) {
        // if both have no ora_type, allow if categories match
        if (target_type.ora_type == null and value_type.ora_type == null) {
            return true;
        }
    }

    if (target_type.category == .ErrorUnion) {
        log.err(
            "[type_resolver] isAssignable: ErrorUnion target did not match value. target={any} value={any}\n",
            .{ target_type, value_type },
        );
    }
    return false;
}

/// Check if a type is numeric
fn isNumericType(type_info: TypeInfo) bool {
    // check category first
    if (type_info.category != .Integer) {
        return false;
    }

    // if we have ora_type, check if it's an integer type
    if (type_info.ora_type) |ora_type| {
        return ora_type.isInteger();
    }

    // category is Integer but no ora_type - assume numeric
    return true;
}

fn sameRefinementKind(target: OraType, value: OraType) bool {
    return switch (target) {
        .min_value => value == .min_value,
        .max_value => value == .max_value,
        .in_range => value == .in_range,
        .non_zero_address => value == .non_zero_address,
        else => false,
    };
}

fn isNumericRefinement(ora_type: OraType) bool {
    return switch (ora_type) {
        .min_value, .max_value, .in_range => true,
        else => false,
    };
}

fn isExactRefinement(ora_type: OraType) bool {
    return switch (ora_type) {
        .exact => true,
        else => false,
    };
}
