// ============================================================================
// Type Resolver Utilities
// ============================================================================
// Phase 1: Implement extraction, placeholder for constants
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const OraType = @import("../../type_info.zig").OraType;

pub const extract = @import("extraction.zig");
pub const constants = @import("constants.zig");

pub const Utils = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Utils {
        return Utils{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Utils) void {
        _ = self;
        // Phase 1: No cleanup needed
    }

    /// Extract base type from a refined type
    pub fn extractBaseType(self: *Utils, ora_type: OraType) ?OraType {
        _ = self;
        return extract.extractBaseType(ora_type);
    }

    /// Evaluate constant expression
    /// Returns the integer value if the expression is a compile-time constant, null otherwise
    pub fn evaluateConstantExpression(
        self: *Utils,
        expr: *ast.Expressions.ExprNode,
    ) !?u256 {
        const result = try constants.evaluateConstantExpression(self.allocator, expr);
        return switch (result) {
            .Integer => |v| v,
            .NotConstant, .Error => null,
        };
    }

    /// Extract integer value from a literal expression
    /// Returns null if not an integer literal or if parsing fails
    pub fn extractIntegerValue(
        self: *Utils,
        literal: *const ast.Expressions.LiteralExpr,
    ) ?u256 {
        _ = self;
        return constants.extractIntegerValue(literal);
    }
};
