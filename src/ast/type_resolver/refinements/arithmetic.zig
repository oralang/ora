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

/// Infer arithmetic result type for refinement types
pub fn inferArithmeticResultType(
    operator: BinaryOp,
    lhs_type: ?TypeInfo,
    rhs_type: ?TypeInfo,
) ?TypeInfo {
    // Only handle arithmetic operators
    if (lhs_type == null or rhs_type == null) return null;

    const lhs = lhs_type.?;
    const rhs = rhs_type.?;

    // Both must have ora_type
    const lhs_ora = lhs.ora_type orelse return null;
    const rhs_ora = rhs.ora_type orelse return null;

    // Both must have compatible base types
    const lhs_base = extract.extractBaseType(lhs_ora) orelse return null;
    const rhs_base = extract.extractBaseType(rhs_ora) orelse return null;

    // Base types must be compatible (same type)
    if (!OraType.equals(lhs_base, rhs_base)) {
        return null;
    }

    // Infer result type based on operator and refinement types
    return switch (operator) {
        .Plus => inferAdditionResultType(lhs_ora, rhs_ora, lhs.span),
        .Minus => inferSubtractionResultType(lhs_ora, rhs_ora, lhs.span),
        .Star => inferMultiplicationResultType(lhs_ora, rhs_ora, lhs.span),
        .Slash, .Percent => null, // Division/modulo lose refinement information
        else => null, // Other operators don't preserve refinements
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
                // Scaled<T, D> + Scaled<T, D> = Scaled<T, D> (preserve scale)
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
                // MinValue<u256, 100> + MinValue<u256, 50> = MinValue<u256, 150>
                const result_min = lhs_mv.min + rhs_mv.min;
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
                // MaxValue<u256, 1000> + MaxValue<u256, 500> = MaxValue<u256, 1500>
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
                // InRange<u256, 10, 100> + InRange<u256, 20, 200> = InRange<u256, 30, 300>
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
                // Scaled<T, D> - Scaled<T, D> = Scaled<T, D> (preserve scale)
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
                // MinValue<u256, 100> - MinValue<u256, 50> = MinValue<u256, 50> (conservative)
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
                // MinValue<u256, 100> - MaxValue<u256, 50> = MinValue<u256, 50> (conservative)
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
                // MaxValue<u256, 1000> - MinValue<u256, 100> = MaxValue<u256, 900>
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
                // MaxValue<u256, 1000> - MaxValue<u256, 500> = MaxValue<u256, 500> (conservative)
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
                // InRange<u256, 10, 100> - InRange<u256, 20, 200> = InRange<u256, 0, 80>
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
    return switch (lhs_ora) {
        .scaled => |lhs_s| switch (rhs_ora) {
            .scaled => |rhs_s| {
                // Scaled<T, D1> * Scaled<T, D2> = Scaled<T, D1 + D2> (scale doubles)
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
                // MinValue<u256, 100> * MinValue<u256, 50> = MinValue<u256, 5000>
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
            else => null,
        },
        .max_value => |lhs_mv| switch (rhs_ora) {
            .max_value => |rhs_mv| {
                // MaxValue<u256, 1000> * MaxValue<u256, 500> = MaxValue<u256, 500000>
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
            else => null,
        },
        .in_range => |lhs_ir| switch (rhs_ora) {
            .in_range => |rhs_ir| {
                // InRange<u256, 10, 100> * InRange<u256, 20, 200> = InRange<u256, 200, 20000>
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
