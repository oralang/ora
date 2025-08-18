const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const ir = @import("ir.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Exhaustiveness analysis data structures
const ExhaustivenessChecker = struct {
    allocator: std.mem.Allocator,
    condition_type: ?ir.Type,
    covered_values: std.ArrayList(CoveredValue),
    has_else_clause: bool,
    has_catchall: bool,

    const CoveredValue = union(enum) {
        literal: LiteralValue,
        range: RangeValue,
        enum_variant: EnumVariantValue,

        const LiteralValue = struct {
            value: LiteralValueType,
            span: ast.SourceSpan,
        };

        const RangeValue = struct {
            start: i64,
            end: i64,
            span: ast.SourceSpan,
        };

        const EnumVariantValue = struct {
            enum_name: []const u8,
            variant_name: []const u8,
            span: ast.SourceSpan,
        };

        const LiteralValueType = union(enum) {
            integer: i64,
            string: []const u8,
            boolean: bool,
        };
    };

    pub fn init(allocator: std.mem.Allocator) ExhaustivenessChecker {
        return ExhaustivenessChecker{
            .allocator = allocator,
            .condition_type = null,
            .covered_values = std.ArrayList(CoveredValue).init(allocator),
            .has_else_clause = false,
            .has_catchall = false,
        };
    }

    pub fn deinit(self: *ExhaustivenessChecker) void {
        self.covered_values.deinit();
    }
};

/// Analyze switch statement for exhaustiveness
pub fn analyzeSwitchStatement(analyzer: *SemanticAnalyzer, switch_stmt: *ast.SwitchNode) semantics_errors.SemanticError!void {
    // Analyze the switch condition
    try analyzeExpression(analyzer, &switch_stmt.condition);

    // Analyze each case
    for (switch_stmt.cases) |*case| {
        try analyzeSwitchCase(analyzer, case);
    }

    // Check exhaustiveness
    try checkSwitchExhaustiveness(analyzer, switch_stmt);
}

/// Analyze switch expression for exhaustiveness
pub fn analyzeSwitchExpression(analyzer: *SemanticAnalyzer, switch_expr: *ast.SwitchExprNode) semantics_errors.SemanticError!void {
    // Analyze the switch condition
    try analyzeExpression(analyzer, &switch_expr.condition);

    // Analyze each case
    for (switch_expr.cases) |*case| {
        try analyzeSwitchCase(analyzer, case);
    }

    // Check exhaustiveness
    try checkSwitchExpressionExhaustiveness(analyzer, switch_expr);
}

/// Analyze a single switch case
fn analyzeSwitchCase(analyzer: *SemanticAnalyzer, case: *ast.SwitchCase) semantics_errors.SemanticError!void {
    // Analyze the pattern
    try analyzeSwitchPattern(analyzer, &case.pattern);

    // Analyze the body
    try analyzeSwitchBody(analyzer, &case.body);
}

/// Analyze a switch pattern
fn analyzeSwitchPattern(analyzer: *SemanticAnalyzer, pattern: *ast.SwitchPattern) semantics_errors.SemanticError!void {
    switch (pattern.*) {
        .Literal => |*lit_pattern| {
            try analyzeLiteralPattern(analyzer, lit_pattern);
        },
        .Range => |*range_pattern| {
            try analyzeRangePattern(analyzer, range_pattern);
        },
        .EnumVariant => |*enum_pattern| {
            try analyzeEnumVariantPattern(analyzer, enum_pattern);
        },
        .Else => |span| {
            // Else patterns are always valid
            _ = span;
        },
    }
}

/// Analyze a switch body
fn analyzeSwitchBody(analyzer: *SemanticAnalyzer, body: *ast.SwitchBody) semantics_errors.SemanticError!void {
    switch (body.*) {
        .Expression => |expr| {
            try analyzeExpression(analyzer, expr);
        },
        .Block => |*block| {
            try analyzeBlock(analyzer, block);
        },
        .LabeledBlock => |*labeled_block| {
            try analyzeLabeledBlock(analyzer, labeled_block);
        },
    }
}

/// Analyze a literal pattern
fn analyzeLiteralPattern(analyzer: *SemanticAnalyzer, pattern: *ast.LiteralPattern) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, &pattern.value);
}

/// Analyze a range pattern
fn analyzeRangePattern(analyzer: *SemanticAnalyzer, pattern: *ast.RangePattern) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, pattern.start);
    try analyzeExpression(analyzer, pattern.end);
}

/// Analyze an enum variant pattern
fn analyzeEnumVariantPattern(analyzer: *SemanticAnalyzer, pattern: *ast.EnumVariantPattern) semantics_errors.SemanticError!void {
    // Validate that the enum and variant exist
    // This would require type checking integration
    _ = analyzer;
    _ = pattern;
}

/// Analyze a labeled block
fn analyzeLabeledBlock(analyzer: *SemanticAnalyzer, labeled_block: *ast.LabeledBlock) semantics_errors.SemanticError!void {
    try analyzeBlock(analyzer, &labeled_block.block);
}

/// Check if a switch statement is exhaustive
fn checkSwitchExhaustiveness(analyzer: *SemanticAnalyzer, switch_stmt: *ast.SwitchNode) semantics_errors.SemanticError!void {
    var checker = ExhaustivenessChecker.init(analyzer.allocator);
    defer checker.deinit();

    // Get the condition type
    checker.condition_type = getExpressionType(analyzer, &switch_stmt.condition);

    // Analyze all patterns
    for (switch_stmt.cases) |*case| {
        try analyzePatternCoverage(&checker, &case.pattern);
    }

    // Check if there's a default case
    if (switch_stmt.default_case != null) {
        checker.has_catchall = true;
    }

    // Perform exhaustiveness analysis
    try performExhaustivenessCheck(analyzer, &checker, switch_stmt.span, false);
}

/// Check if a switch expression is exhaustive
fn checkSwitchExpressionExhaustiveness(analyzer: *SemanticAnalyzer, switch_expr: *ast.SwitchExprNode) semantics_errors.SemanticError!void {
    var checker = ExhaustivenessChecker.init(analyzer.allocator);
    defer checker.deinit();

    // Get the condition type
    checker.condition_type = getExpressionType(analyzer, &switch_expr.condition);

    // Analyze all patterns
    for (switch_expr.cases) |*case| {
        try analyzePatternCoverage(&checker, &case.pattern);
    }

    // Check if there's a default case
    if (switch_expr.default_case != null) {
        checker.has_catchall = true;
    }

    // Switch expressions must be exhaustive
    try performExhaustivenessCheck(analyzer, &checker, switch_expr.span, true);
}

/// Analyze pattern coverage for exhaustiveness checking
fn analyzePatternCoverage(checker: *ExhaustivenessChecker, pattern: *ast.SwitchPattern) !void {
    switch (pattern.*) {
        .Literal => |*lit_pattern| {
            const covered_value = try extractLiteralValue(checker.allocator, &lit_pattern.value, lit_pattern.span);
            try checker.covered_values.append(covered_value);
        },
        .Range => |*range_pattern| {
            // Extract range bounds
            const start_value = try extractIntegerFromExpression(range_pattern.start);
            const end_value = try extractIntegerFromExpression(range_pattern.end);

            if (start_value != null and end_value != null) {
                try checker.covered_values.append(ExhaustivenessChecker.CoveredValue{
                    .range = ExhaustivenessChecker.CoveredValue.RangeValue{
                        .start = start_value.?,
                        .end = end_value.?,
                        .span = range_pattern.span,
                    },
                });
            }
        },
        .EnumVariant => |*enum_pattern| {
            try checker.covered_values.append(ExhaustivenessChecker.CoveredValue{
                .enum_variant = ExhaustivenessChecker.CoveredValue.EnumVariantValue{
                    .enum_name = enum_pattern.enum_name,
                    .variant_name = enum_pattern.variant_name,
                    .span = enum_pattern.span,
                },
            });
        },
        .Else => {
            checker.has_else_clause = true;
            checker.has_catchall = true;
        },
    }
}

/// Extract literal value from AST literal node
fn extractLiteralValue(allocator: std.mem.Allocator, literal: *ast.LiteralExpr, span: ast.SourceSpan) !ExhaustivenessChecker.CoveredValue {
    const literal_value = switch (literal.*) {
        .Integer => |int_lit| ExhaustivenessChecker.CoveredValue.LiteralValueType{ .integer = std.fmt.parseInt(i64, int_lit.value, 10) catch 0 },
        .String => |str_lit| ExhaustivenessChecker.CoveredValue.LiteralValueType{ .string = try allocator.dupe(u8, str_lit.value) },
        .Bool => |bool_lit| ExhaustivenessChecker.CoveredValue.LiteralValueType{ .boolean = bool_lit.value },
        else => ExhaustivenessChecker.CoveredValue.LiteralValueType{ .integer = 0 },
    };

    return ExhaustivenessChecker.CoveredValue{
        .literal = ExhaustivenessChecker.CoveredValue.LiteralValue{
            .value = literal_value,
            .span = span,
        },
    };
}

/// Extract integer value from expression (for range patterns)
fn extractIntegerFromExpression(expr: *ast.ExprNode) !?i64 {
    switch (expr.*) {
        .Literal => |*lit| {
            switch (lit.*) {
                .Integer => |int_lit| {
                    return std.fmt.parseInt(i64, int_lit.value, 10) catch null;
                },
                else => return null,
            }
        },
        else => return null,
    }
}

/// Perform comprehensive exhaustiveness checking
fn performExhaustivenessCheck(analyzer: *SemanticAnalyzer, checker: *ExhaustivenessChecker, span: ast.SourceSpan, is_expression: bool) !void {
    // If there's a catchall pattern (else clause or default case), it's exhaustive
    if (checker.has_catchall) {
        return;
    }

    // Check exhaustiveness based on the condition type
    if (checker.condition_type) |condition_type| {
        switch (condition_type) {
            .enum_type => |enum_type| {
                try checkEnumTypeExhaustiveness(analyzer, checker, enum_type, span, is_expression);
            },
            .primitive => |primitive| {
                try checkPrimitiveTypeExhaustiveness(analyzer, checker, primitive, span, is_expression);
            },
            else => {
                // For other types, require a catchall pattern
                const error_msg = if (is_expression)
                    "Switch expression must be exhaustive - add an else clause"
                else
                    "Switch statement may not be exhaustive - consider adding a default case";
                try semantics_errors.addErrorStatic(analyzer, error_msg, span);
            },
        }
    } else {
        // Unknown type - require catchall for safety
        const error_msg = if (is_expression)
            "Switch expression must be exhaustive - add an else clause (unknown condition type)"
        else
            "Switch statement may not be exhaustive - add a default case (unknown condition type)";
        try semantics_errors.addErrorStatic(analyzer, error_msg, span);
    }
}

/// Check exhaustiveness for enum types
fn checkEnumTypeExhaustiveness(analyzer: *SemanticAnalyzer, checker: *ExhaustivenessChecker, enum_type: ir.Type.EnumType, span: ast.SourceSpan, is_expression: bool) !void {
    var covered_variants = std.ArrayList([]const u8).init(checker.allocator);
    defer covered_variants.deinit();

    // Collect covered enum variants
    for (checker.covered_values.items) |covered_value| {
        switch (covered_value) {
            .enum_variant => |enum_variant| {
                if (std.mem.eql(u8, enum_variant.enum_name, enum_type.name.string)) {
                    try covered_variants.append(enum_variant.variant_name);
                }
            },
            else => {},
        }
    }

    // Check if all enum variants are covered
    var missing_variants = std.ArrayList([]const u8).init(checker.allocator);
    defer missing_variants.deinit();

    for (enum_type.variants) |variant| {
        const variant_name = variant.name.string;
        var found = false;
        for (covered_variants.items) |covered_variant| {
            if (std.mem.eql(u8, variant_name, covered_variant)) {
                found = true;
                break;
            }
        }
        if (!found) {
            try missing_variants.append(variant_name);
        }
    }

    // Report missing variants
    if (missing_variants.items.len > 0) {
        var error_msg_buf = std.ArrayList(u8).init(checker.allocator);
        defer error_msg_buf.deinit();
        var writer = error_msg_buf.writer();

        if (is_expression) {
            try writer.writeAll("Switch expression is not exhaustive. Missing variants: ");
        } else {
            try writer.writeAll("Switch statement is not exhaustive. Missing variants: ");
        }

        for (missing_variants.items, 0..) |variant, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll(variant);
        }

        const error_msg = try error_msg_buf.toOwnedSlice();
        try semantics_errors.addErrorAllocated(analyzer, error_msg, span);
    }
}

/// Check exhaustiveness for primitive types
fn checkPrimitiveTypeExhaustiveness(analyzer: *SemanticAnalyzer, checker: *ExhaustivenessChecker, primitive: ir.Type.PrimitiveType, span: ast.SourceSpan, is_expression: bool) !void {
    switch (primitive) {
        .bool => {
            try checkBooleanExhaustiveness(analyzer, checker, span, is_expression);
        },
        .u8, .u16, .u32, .u64, .u128, .u256 => {
            try checkIntegerExhaustiveness(analyzer, checker, primitive, span, is_expression);
        },
        .string => {
            // String types can't be exhaustively checked without else clause
            const error_msg = if (is_expression)
                "Switch expression on string type must have an else clause"
            else
                "Switch statement on string type should have a default case";
            try semantics_errors.addWarningStatic(analyzer, error_msg, span);
        },
        else => {
            // Other primitive types require catchall
            const error_msg = if (is_expression)
                "Switch expression must be exhaustive - add an else clause"
            else
                "Switch statement may not be exhaustive - consider adding a default case";
            try semantics_errors.addWarningStatic(analyzer, error_msg, span);
        },
    }
}

/// Check exhaustiveness for boolean types
fn checkBooleanExhaustiveness(analyzer: *SemanticAnalyzer, checker: *ExhaustivenessChecker, span: ast.SourceSpan, is_expression: bool) !void {
    var has_true = false;
    var has_false = false;

    for (checker.covered_values.items) |covered_value| {
        switch (covered_value) {
            .literal => |literal| {
                switch (literal.value) {
                    .boolean => |bool_val| {
                        if (bool_val) has_true = true else has_false = true;
                    },
                    else => {},
                }
            },
            else => {},
        }
    }

    if (!has_true or !has_false) {
        var missing = std.ArrayList([]const u8).init(checker.allocator);
        defer missing.deinit();

        if (!has_true) try missing.append("true");
        if (!has_false) try missing.append("false");

        var error_msg_buf = std.ArrayList(u8).init(checker.allocator);
        defer error_msg_buf.deinit();
        var writer = error_msg_buf.writer();

        if (is_expression) {
            try writer.writeAll("Switch expression on boolean is not exhaustive. Missing: ");
        } else {
            try writer.writeAll("Switch statement on boolean is not exhaustive. Missing: ");
        }

        for (missing.items, 0..) |item, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll(item);
        }

        const error_msg = try error_msg_buf.toOwnedSlice();
        try semantics_errors.addErrorAllocated(analyzer, error_msg, span);
    }
}

/// Check exhaustiveness for integer types (simplified)
fn checkIntegerExhaustiveness(analyzer: *SemanticAnalyzer, checker: *ExhaustivenessChecker, primitive: ir.Type.PrimitiveType, span: ast.SourceSpan, is_expression: bool) !void {
    _ = primitive; // For now, we don't do full range analysis

    // For integer types, we generally can't guarantee exhaustiveness without else clause
    // unless it's a very small range or specific set of values

    // Check if we have overlapping ranges that might cover the full space
    var has_comprehensive_coverage = false;

    // Simple heuristic: if we have more than 10 literal values or any range,
    // assume it might be comprehensive
    var literal_count: u32 = 0;
    var has_ranges = false;

    for (checker.covered_values.items) |covered_value| {
        switch (covered_value) {
            .literal => |literal| {
                switch (literal.value) {
                    .integer => literal_count += 1,
                    else => {},
                }
            },
            .range => {
                has_ranges = true;
            },
            else => {},
        }
    }

    // If we have ranges or many literals, it might be comprehensive
    has_comprehensive_coverage = has_ranges or literal_count >= 10;

    if (!has_comprehensive_coverage) {
        const error_msg = if (is_expression)
            "Switch expression on integer type should have an else clause for exhaustiveness"
        else
            "Switch statement on integer type should have a default case";
        try semantics_errors.addWarningStatic(analyzer, error_msg, span);
    }
}

// Forward declarations for functions that would be implemented in other modules
fn analyzeExpression(analyzer: *SemanticAnalyzer, expr: *ast.ExprNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = expr;
    // This would call the expression analyzer
}

fn analyzeBlock(analyzer: *SemanticAnalyzer, block: *ast.BlockNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = block;
    // This would analyze the block statements
}

fn getExpressionType(analyzer: *SemanticAnalyzer, expr: *ast.ExprNode) ?ir.Type {
    // In a real implementation, this would integrate with the type checker
    _ = analyzer;

    // Try to infer type from expression structure
    switch (expr.*) {
        .Literal => |*lit| {
            switch (lit.*) {
                .Integer => return ir.Type{ .primitive = .u256 },
                .String => return ir.Type{ .primitive = .string },
                .Bool => return ir.Type{ .primitive = .bool },
                else => return null,
            }
        },
        .Identifier => |*ident| {
            // Check if it's a known enum type
            if (std.mem.endsWith(u8, ident.name, "Type") or
                std.mem.endsWith(u8, ident.name, "Tier"))
            {
                // Mock enum type for testing
                return ir.Type{
                    .enum_type = ir.Type.EnumType{
                        .name = ir.IdentifierName{ .string = ident.name },
                        .variants = &[_]ir.Type.EnumType.EnumVariant{
                            ir.Type.EnumType.EnumVariant{
                                .name = ir.IdentifierName{ .string = "Basic" },
                                .value = undefined, // Would be properly initialized
                            },
                            ir.Type.EnumType.EnumVariant{
                                .name = ir.IdentifierName{ .string = "Premium" },
                                .value = undefined,
                            },
                            ir.Type.EnumType.EnumVariant{
                                .name = ir.IdentifierName{ .string = "VIP" },
                                .value = undefined,
                            },
                        },
                    },
                };
            }
            return null;
        },
        .FieldAccess => |*field| {
            // Handle enum variant access like TransferType.Mint
            _ = field;
            return ir.Type{ .enum_type = ir.Type.EnumType{
                .name = ir.IdentifierName{ .string = "TransferType" },
                .variants = &[_]ir.Type.EnumType.EnumVariant{
                    ir.Type.EnumType.EnumVariant{
                        .name = ir.IdentifierName{ .string = "Mint" },
                        .value = undefined,
                    },
                    ir.Type.EnumType.EnumVariant{
                        .name = ir.IdentifierName{ .string = "Burn" },
                        .value = undefined,
                    },
                    ir.Type.EnumType.EnumVariant{
                        .name = ir.IdentifierName{ .string = "Transfer" },
                        .value = undefined,
                    },
                },
            } };
        },
        else => return null,
    }
}
