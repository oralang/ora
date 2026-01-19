// ============================================================================
// Type Compatibility
// ============================================================================
// Phase 2: Extract compatibility checks
// ============================================================================

const std = @import("std");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const refinements = @import("../refinements/mod.zig");

/// Check if two types are compatible
pub fn areTypesCompatible(
    type1: TypeInfo,
    type2: TypeInfo,
    refinement_system: *refinements.RefinementSystem,
) bool {
    // for primitive types with ora_type, check exact match or subtyping
    // (Refinement types can have different categories but same base, so check ora_type first)
    if (type1.ora_type != null and type2.ora_type != null) {
        const t1_ora = type1.ora_type.?;
        const t2_ora = type2.ora_type.?;

        // enums vs their named type: treat OraType.enum_type "Status" as
        // compatible with custom type references that currently use
        // oraType.struct_type "Status". This lets us type-check expressions
        // like `Status.Active` against variables of type `Status` even though
        // the parser initially represents `Status` as a struct_type.
        switch (t1_ora) {
            .enum_type => |ename1| switch (t2_ora) {
                .struct_type => |sname2| {
                    if (std.mem.eql(u8, ename1, sname2)) return true;
                },
                else => {},
            },
            .struct_type => |sname1| switch (t2_ora) {
                .enum_type => |ename2| {
                    if (std.mem.eql(u8, sname1, ename2)) return true;
                },
                else => {},
            },
            else => {},
        }

        // check refinement subtyping (handles both directions)
        if (refinement_system.checkSubtype(t1_ora, t2_ora, isBaseTypeCompatible) or
            refinement_system.checkSubtype(t2_ora, t1_ora, isBaseTypeCompatible))
        {
            return true;
        }
        // check exact match
        if (OraType.equals(t1_ora, t2_ora)) {
            return true;
        }

        // if both are refinement types, check if base types are compatible
        // (for binary operations, same base type is sufficient)
        if (isRefinementType(t1_ora) and isRefinementType(t2_ora)) {
            const t1_base = extractRefinementBase(t1_ora);
            const t2_base = extractRefinementBase(t2_ora);
            if (t1_base != null and t2_base != null) {
                return isBaseTypeCompatible(t1_base.?, t2_base.?);
            }
            return false;
        }
    }

    // error category is compatible with ErrorUnion category
    // this allows error returns (category .Error) to be compatible with ErrorUnion return types
    if (type1.category == .Error and type2.category == .ErrorUnion) {
        return true;
    }
    if (type1.category == .ErrorUnion and type2.category == .Error) {
        return true;
    }

    // enum category is compatible with ErrorUnion category (for error.X expressions)
    // this allows error enum literals to be compatible with ErrorUnion return types
    if (type1.category == .Enum and type2.category == .ErrorUnion) {
        return true;
    }
    if (type1.category == .ErrorUnion and type2.category == .Enum) {
        return true;
    }

    // success type is compatible with ErrorUnion (e.g., bool is compatible with !bool | Error1)
    // check if type2 (value) matches the success type of type1's (target) ErrorUnion
    if (type1.category == .ErrorUnion) {
        if (type1.ora_type) |type1_ora| {
            if (type1_ora == .error_union) {
                const success_ora_type = type1_ora.error_union.*;
                const success_category = success_ora_type.getCategory();
                if (type2.ora_type) |type2_ora| {
                    if (type2.category == success_category and
                        (OraType.equals(type2_ora, success_ora_type) or
                            refinement_system.checkSubtype(type2_ora, success_ora_type, isBaseTypeCompatible) or
                            refinement_system.checkSubtype(success_ora_type, type2_ora, isBaseTypeCompatible)))
                    {
                        return true;
                    }
                } else if (type2.category == success_category) {
                    return true;
                }
            } else if (type1_ora == ._union) {
                // error union with explicit errors: !T | Error1 | Error2
                if (type1_ora._union.len > 0) {
                    const first_type = type1_ora._union[0];
                    if (first_type == .error_union) {
                        const success_ora_type = first_type.error_union.*;
                        const success_category = success_ora_type.getCategory();
                        if (type2.ora_type) |type2_ora| {
                            if (type2.category == success_category and
                                (OraType.equals(type2_ora, success_ora_type) or
                                    refinement_system.checkSubtype(type2_ora, success_ora_type, isBaseTypeCompatible) or
                                    refinement_system.checkSubtype(success_ora_type, type2_ora, isBaseTypeCompatible)))
                            {
                                return true;
                            }
                        } else if (type2.category == success_category) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // same category is usually compatible
    if (type1.category != type2.category) {
        return false;
    }

    // for custom types (structs, enums, etc.), check by category match
    // custom type matching is handled by category comparison above

    // categories match and no specific type info - consider compatible
    return true;
}

/// Check if base types are compatible (handles width subtyping)
pub fn isBaseTypeCompatible(source: OraType, target: OraType) bool {
    // direct match
    if (OraType.equals(source, target)) return true;

    // width subtyping: u8 <: u16 <: u32 <: u64 <: u128 <: u256
    const width_order = [_]OraType{ .u8, .u16, .u32, .u64, .u128, .u256 };
    const signed_width_order = [_]OraType{ .i8, .i16, .i32, .i64, .i128, .i256 };

    const source_idx = getTypeIndex(source, &width_order) orelse
        getTypeIndex(source, &signed_width_order);
    const target_idx = getTypeIndex(target, &width_order) orelse
        getTypeIndex(target, &signed_width_order);

    if (source_idx) |s_idx| {
        if (target_idx) |t_idx| {
            // same sign category, check if source is narrower than target
            return s_idx <= t_idx;
        }
    }

    return false;
}

/// Get index of type in hierarchy (for width subtyping)
fn getTypeIndex(ora_type: OraType, hierarchy: []const OraType) ?usize {
    for (hierarchy, 0..) |t, i| {
        if (OraType.equals(ora_type, t)) return i;
    }
    return null;
}

/// Check if a type is a refinement type (min_value, max_value, in_range, etc.)
fn isRefinementType(ora_type: OraType) bool {
    return switch (ora_type) {
        .min_value, .max_value, .in_range, .scaled, .exact, .non_zero_address => true,
        else => false,
    };
}

/// Extract the base type from a refinement type
fn extractRefinementBase(ora_type: OraType) ?OraType {
    return switch (ora_type) {
        .min_value => |mv| mv.base.*,
        .max_value => |mv| mv.base.*,
        .in_range => |ir| ir.base.*,
        .scaled => |s| s.base.*,
        .exact => |e| e.*,
        .non_zero_address => .address,
        else => null,
    };
}
