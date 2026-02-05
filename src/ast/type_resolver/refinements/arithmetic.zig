// ============================================================================
// Refinement Arithmetic Inference
// ============================================================================
// Phase 1: Extract arithmetic inference logic
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const TypeCategory = @import("../../type_info.zig").TypeCategory;
const OraType = @import("../../type_info.zig").OraType;
const SourceSpan = @import("../../source_span.zig").SourceSpan;
const BinaryOp = @import("../../expressions.zig").BinaryOp;
const extract = @import("../utils/extraction.zig");
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const log = @import("log");

/// Validate arithmetic operation between refinement types
/// Returns error if operation is invalid (e.g., scale mismatch)
pub fn validateArithmeticOperation(
    operator: BinaryOp,
    lhs_type: ?TypeInfo,
    rhs_type: ?TypeInfo,
) TypeResolutionError!void {
    if (lhs_type == null or rhs_type == null) return;

    const lhs = lhs_type.?;
    const rhs = rhs_type.?;

    const lhs_ora = lhs.ora_type orelse return;
    const rhs_ora = rhs.ora_type orelse return;

    // Check for scale mismatch in Scaled types
    switch (lhs_ora) {
        .scaled => |lhs_s| {
            switch (rhs_ora) {
                .scaled => |rhs_s| {
                    // For addition/subtraction, scales must match
                    if (operator == .Plus or operator == .Minus) {
                        if (lhs_s.decimals != rhs_s.decimals) {
                            return TypeResolutionError.TypeMismatch;
                        }
                    }
                    // For multiplication, scales can differ (they add)
                    // For division, scales can differ (they subtract)
                },
                else => {
                    // Scaled + non-Scaled base type: only allowed if rhs is the base type
                    // This is handled in inference, but we allow it here
                },
            }
        },
        else => {
            // Check if RHS is Scaled and LHS is not
            switch (rhs_ora) {
                .scaled => |_| {
                    // non-Scaled + Scaled: same rules apply
                    switch (lhs_ora) {
                        .scaled => {}, // Already handled above
                        else => {
                            // base type + Scaled is allowed (commutative)
                        },
                    }
                },
                else => {},
            }
        },
    }
}

/// Infer arithmetic result type for refinement types
pub fn inferArithmeticResultType(
    operator: BinaryOp,
    lhs_type: ?TypeInfo,
    rhs_type: ?TypeInfo,
) ?TypeInfo {
    // only handle arithmetic operators
    if (lhs_type == null or rhs_type == null) return null;

    const lhs = lhs_type.?;
    const rhs = rhs_type.?;

    // both must have ora_type
    const lhs_ora = lhs.ora_type orelse return null;
    const rhs_ora = rhs.ora_type orelse return null;

    // extract base types
    const lhs_base = extract.extractBaseType(lhs_ora) orelse return null;
    const rhs_base = extract.extractBaseType(rhs_ora) orelse return null;

    // base types must be compatible (same type)
    if (!OraType.equals(lhs_base, rhs_base)) {
        return null;
    }

    // check if either operand is a refinement type
    const lhs_is_refined = isRefinementType(lhs_ora);
    const rhs_is_refined = isRefinementType(rhs_ora);

    // infer result type based on operator and refinement types
    return switch (operator) {
        .Plus => blk: {
            // try same-refinement inference first
            if (inferAdditionResultType(lhs_ora, rhs_ora, lhs.span)) |result| {
                break :blk result;
            }
            // handle mixed refinement + base type
            if (lhs_is_refined and !rhs_is_refined) {
                break :blk inferMixedAddition(lhs_ora, rhs_ora, lhs.span);
            }
            if (rhs_is_refined and !lhs_is_refined) {
                break :blk inferMixedAddition(rhs_ora, lhs_ora, lhs.span);
            }
            break :blk null;
        },
        .Minus => blk: {
            // try same-refinement inference first
            if (inferSubtractionResultType(lhs_ora, rhs_ora, lhs.span)) |result| {
                break :blk result;
            }
            // handle refinement - base type (only LHS refinement is meaningful)
            if (lhs_is_refined and !rhs_is_refined) {
                break :blk inferMixedSubtraction(lhs_ora, rhs_ora, lhs.span);
            }
            break :blk null;
        },
        .Star => blk: {
            // try same-refinement inference first
            if (inferMultiplicationResultType(lhs_ora, rhs_ora, lhs.span)) |result| {
                break :blk result;
            }
            // handle mixed refinement * base type
            if (lhs_is_refined and !rhs_is_refined) {
                break :blk inferMixedMultiplication(lhs_ora, rhs_ora, lhs.span);
            }
            if (rhs_is_refined and !lhs_is_refined) {
                break :blk inferMixedMultiplication(rhs_ora, lhs_ora, lhs.span);
            }
            break :blk null;
        },
        .Slash, .Percent => null, // Division/modulo lose refinement information
        else => null, // Other operators don't preserve refinements
    };
}

/// Check if an OraType is a refinement type
fn isRefinementType(ora_type: OraType) bool {
    return switch (ora_type) {
        .min_value, .max_value, .in_range, .scaled, .exact, .non_zero_address => true,
        else => false,
    };
}

/// Infer result type for refinement + base type addition
/// Conservative: preserve refinement structure but widen bounds
fn inferMixedAddition(
    refined_ora: OraType,
    _: OraType, // base type (unused, just for compatibility check)
    span: ?SourceSpan,
) ?TypeInfo {
    // For addition with unknown value, we lose the lower bound but keep structure
    // MinValue<u256, 10> + u256 = u256 (conservative: unknown addition could be 0)
    // MaxValue<u256, 100> + u256 = u256 (max is now unbounded)
    // InRange<u256, 10, 100> + u256 = u256 (bounds are lost)
    // Scaled preserves scale since it's a unit, not a bound
    return switch (refined_ora) {
        .scaled => |s| {
            // Scaled<T, D> + T = Scaled<T, D> (scale is preserved)
            const scaled_type = OraType{
                .scaled = .{
                    .base = s.base,
                    .decimals = s.decimals,
                },
            };
            return TypeInfo.inferred(TypeCategory.Integer, scaled_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
        },
        .min_value => |mv| {
            // MinValue<T, N> + T = T (adding unknown could be 0, min is lost)
            // But we can preserve MinValue<T, 0> as a conservative bound
            const min_value_type = OraType{
                .min_value = .{
                    .base = mv.base,
                    .min = mv.min, // Preserve: result is at least min if added value >= 0
                },
            };
            return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
        },
        else => null, // Other refinements lose precision
    };
}

/// Infer result type for refinement - base type subtraction
fn inferMixedSubtraction(
    refined_ora: OraType,
    _: OraType, // base type
    span: ?SourceSpan,
) ?TypeInfo {
    // For subtraction with unknown value, we typically lose bounds
    // MinValue<u256, 100> - u256 = u256 (could underflow to 0 or revert)
    // Scaled preserves scale
    return switch (refined_ora) {
        .scaled => |s| {
            const scaled_type = OraType{
                .scaled = .{
                    .base = s.base,
                    .decimals = s.decimals,
                },
            };
            return TypeInfo.inferred(TypeCategory.Integer, scaled_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
        },
        .max_value => |mv| {
            // MaxValue<T, N> - T = MaxValue<T, N> (subtraction doesn't increase max)
            const max_value_type = OraType{
                .max_value = .{
                    .base = mv.base,
                    .max = mv.max,
                },
            };
            return TypeInfo.inferred(TypeCategory.Integer, max_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
        },
        else => null,
    };
}

/// Infer result type for refinement * base type multiplication
fn inferMixedMultiplication(
    refined_ora: OraType,
    _: OraType, // base type
    span: ?SourceSpan,
) ?TypeInfo {
    // Scaled * T = T (scale is lost unless multiplier is 1/scale)
    // MinValue<T, N> * T = T (multiplier could be 0)
    // For multiplication, most bounds are lost unless multiplier is known
    return switch (refined_ora) {
        .min_value => |mv| {
            // MinValue<T, N> * T where T >= 0 = MinValue<T, 0> (conservative)
            // If the base type is unsigned, result min is 0
            const min_value_type = OraType{
                .min_value = .{
                    .base = mv.base,
                    .min = 0, // Conservative: multiplier could be 0
                },
            };
            return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
        },
        else => null,
    };
}

/// Infer result type for addition
pub fn inferAdditionResultType(
    lhs_ora: OraType,
    rhs_ora: OraType,
    span: ?SourceSpan,
) ?TypeInfo {
    return switch (lhs_ora) {
        .scaled => |lhs_s| switch (rhs_ora) {
            .scaled => |rhs_s| {
                // scaled<T, D> + Scaled<T, D> = Scaled<T, D> (preserve scale)
                if (!OraType.equals(lhs_s.base.*, rhs_s.base.*) or lhs_s.decimals != rhs_s.decimals) {
                    return null; // Different scales cannot be added directly
                }
                const scaled_type = OraType{
                    .scaled = .{
                        .base = lhs_s.base,
                        .decimals = lhs_s.decimals,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, scaled_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .min_value => |lhs_mv| switch (rhs_ora) {
            .min_value => |rhs_mv| {
                // minValue<u256, 100> + MinValue<u256, 50> = MinValue<u256, 150>
                const result_min = lhs_mv.min + rhs_mv.min;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .in_range => |rhs_ir| {
                // MinValue<u256, 100> + InRange<u256, 0, 10000> = InRange<u256, 100, 10100>
                if (!OraType.equals(lhs_mv.base.*, rhs_ir.base.*)) {
                    return null;
                }
                const result_min = lhs_mv.min + rhs_ir.min;
                const result_max = lhs_mv.min + rhs_ir.max;
                const in_range_type = OraType{
                    .in_range = .{
                        .base = rhs_ir.base,
                        .min = result_min,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, in_range_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .max_value => |lhs_mv| switch (rhs_ora) {
            .max_value => |rhs_mv| {
                // maxValue<u256, 1000> + MaxValue<u256, 500> = MaxValue<u256, 1500>
                const result_max = lhs_mv.max + rhs_mv.max;
                const max_value_type = OraType{
                    .max_value = .{
                        .base = lhs_mv.base,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, max_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .in_range => |lhs_ir| switch (rhs_ora) {
            .in_range => |rhs_ir| {
                // inRange<u256, 10, 100> + InRange<u256, 20, 200> = InRange<u256, 30, 300>
                if (!OraType.equals(lhs_ir.base.*, rhs_ir.base.*)) {
                    return null;
                }
                const result_min = lhs_ir.min + rhs_ir.min;
                const result_max = lhs_ir.max + rhs_ir.max;
                const in_range_type = OraType{
                    .in_range = .{
                        .base = lhs_ir.base,
                        .min = result_min,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, in_range_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .min_value => |rhs_mv| {
                // InRange<u256, 0, 10000> + MinValue<u256, 100> = InRange<u256, 100, 10100>
                if (!OraType.equals(lhs_ir.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = lhs_ir.min + rhs_mv.min;
                const result_max = lhs_ir.max + rhs_mv.min;
                const in_range_type = OraType{
                    .in_range = .{
                        .base = lhs_ir.base,
                        .min = result_min,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, in_range_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        else => null,
    };
}

/// Infer result type for subtraction
pub fn inferSubtractionResultType(
    lhs_ora: OraType,
    rhs_ora: OraType,
    span: ?SourceSpan,
) ?TypeInfo {
    return switch (lhs_ora) {
        .scaled => |lhs_s| switch (rhs_ora) {
            .scaled => |rhs_s| {
                // scaled<T, D> - Scaled<T, D> = Scaled<T, D> (preserve scale)
                if (!OraType.equals(lhs_s.base.*, rhs_s.base.*) or lhs_s.decimals != rhs_s.decimals) {
                    return null;
                }
                const scaled_type = OraType{
                    .scaled = .{
                        .base = lhs_s.base,
                        .decimals = lhs_s.decimals,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, scaled_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .min_value => |lhs_mv| switch (rhs_ora) {
            .min_value => |rhs_mv| {
                // minValue<u256, 100> - MinValue<u256, 50> = MinValue<u256, 50> (conservative)
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = if (lhs_mv.min >= rhs_mv.min) lhs_mv.min - rhs_mv.min else 0;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .max_value => |rhs_mv| {
                // minValue<u256, 100> - MaxValue<u256, 50> = MinValue<u256, 50> (conservative)
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = if (lhs_mv.min >= rhs_mv.max) lhs_mv.min - rhs_mv.max else 0;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .max_value => |lhs_mv| switch (rhs_ora) {
            .min_value => |rhs_mv| {
                // maxValue<u256, 1000> - MinValue<u256, 100> = MaxValue<u256, 900>
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_max = if (lhs_mv.max >= rhs_mv.min) lhs_mv.max - rhs_mv.min else 0;
                const max_value_type = OraType{
                    .max_value = .{
                        .base = lhs_mv.base,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, max_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .max_value => |rhs_mv| {
                // maxValue<u256, 1000> - MaxValue<u256, 500> = MaxValue<u256, 500> (conservative)
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_max = lhs_mv.max; // Conservative: keep lhs max
                const max_value_type = OraType{
                    .max_value = .{
                        .base = lhs_mv.base,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, max_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .in_range => |lhs_ir| switch (rhs_ora) {
            .in_range => |rhs_ir| {
                // inRange<u256, 10, 100> - InRange<u256, 20, 200> = InRange<u256, 0, 80>
                if (!OraType.equals(lhs_ir.base.*, rhs_ir.base.*)) {
                    return null;
                }
                const result_min = if (lhs_ir.min >= rhs_ir.max) lhs_ir.min - rhs_ir.max else 0;
                const result_max = if (lhs_ir.max >= rhs_ir.min) lhs_ir.max - rhs_ir.min else 0;
                const in_range_type = OraType{
                    .in_range = .{
                        .base = lhs_ir.base,
                        .min = result_min,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, in_range_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        else => null,
    };
}

/// Infer result type for multiplication
pub fn inferMultiplicationResultType(
    lhs_ora: OraType,
    rhs_ora: OraType,
    span: ?SourceSpan,
) ?TypeInfo {
    log.debug(
        "[inferMultiplicationResultType] lhs={s} rhs={s} at {any}\n",
        .{ @tagName(lhs_ora), @tagName(rhs_ora), span },
    );
    return switch (lhs_ora) {
        .scaled => |lhs_s| switch (rhs_ora) {
            .scaled => |rhs_s| {
                // scaled<T, D1> * Scaled<T, D2> = Scaled<T, D1 + D2> (scale doubles)
                if (!OraType.equals(lhs_s.base.*, rhs_s.base.*)) {
                    return null;
                }
                const scaled_type = OraType{
                    .scaled = .{
                        .base = lhs_s.base,
                        .decimals = lhs_s.decimals + rhs_s.decimals,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, scaled_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .min_value => |lhs_mv| switch (rhs_ora) {
            .min_value => |rhs_mv| {
                // minValue<u256, 100> * MinValue<u256, 50> = MinValue<u256, 5000>
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = lhs_mv.min * rhs_mv.min;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .max_value => |rhs_mv| {
                // MinValue<u256, N> * MaxValue<u256, M> = MinValue<u256, N*M>
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = lhs_mv.min * rhs_mv.max;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .max_value => |lhs_mv| switch (rhs_ora) {
            .max_value => |rhs_mv| {
                // maxValue<u256, 1000> * MaxValue<u256, 500> = MaxValue<u256, 500000>
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_max = lhs_mv.max * rhs_mv.max;
                const max_value_type = OraType{
                    .max_value = .{
                        .base = lhs_mv.base,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, max_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            .min_value => |rhs_mv| {
                // maxValue<u256, M> * MinValue<u256, N> = MinValue<u256, M*N>
                if (!OraType.equals(lhs_mv.base.*, rhs_mv.base.*)) {
                    return null;
                }
                const result_min = lhs_mv.max * rhs_mv.min;
                const min_value_type = OraType{
                    .min_value = .{
                        .base = lhs_mv.base,
                        .min = result_min,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, min_value_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        .in_range => |lhs_ir| switch (rhs_ora) {
            .in_range => |rhs_ir| {
                // inRange<u256, 10, 100> * InRange<u256, 20, 200> = InRange<u256, 200, 20000>
                if (!OraType.equals(lhs_ir.base.*, rhs_ir.base.*)) {
                    return null;
                }
                const result_min = lhs_ir.min * rhs_ir.min;
                const result_max = lhs_ir.max * rhs_ir.max;
                const in_range_type = OraType{
                    .in_range = .{
                        .base = lhs_ir.base,
                        .min = result_min,
                        .max = result_max,
                    },
                };
                return TypeInfo.inferred(TypeCategory.Integer, in_range_type, span orelse SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
            else => null,
        },
        else => null,
    };
}
