const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Validate that an AST node pointer is safe to dereference
pub fn isValidNodePointer(analyzer: *SemanticAnalyzer, node: ?*ast.AstNode) bool {
    _ = analyzer;
    if (node == null) return false;

    // Basic pointer validation - check if it's in a reasonable memory range
    const ptr_value = @intFromPtr(node.?);

    // Check for null and obviously invalid pointers
    if (ptr_value == 0) return false;

    // Platform-independent validation using standard page size
    const page_size: usize = 4096; // Use standard page size (4KB)
    if (ptr_value < page_size) return false; // Likely null pointer dereference

    // Check alignment for typical AST node structures (should be word-aligned)
    if (ptr_value % @alignOf(*ast.AstNode) != 0) return false;

    return true;
}

/// Validate that a string is safe to use and follows our simplified string model for smart contracts
/// Only allows ASCII characters and enforces a reasonable max length
pub fn isValidString(analyzer: *SemanticAnalyzer, str: []const u8) bool {
    _ = analyzer;
    // Check for reasonable string length (prevent reading garbage memory)
    // For smart contracts, we use a much more conservative limit
    if (str.len > 1024) return false; // 1KB max string length for smart contracts

    // Check if the pointer looks valid
    const ptr_value = @intFromPtr(str.ptr);
    if (ptr_value == 0) return false;
    if (ptr_value < 0x1000) return false;

    // Ensure all characters are valid ASCII (0-127)
    for (str) |c| {
        if (c > 127) return false; // Restrict to ASCII characters only
    }

    return true;
}

/// Validate that a SourceSpan is safe to use
pub fn validateSpan(analyzer: *SemanticAnalyzer, span: ast.SourceSpan) ast.SourceSpan {
    _ = analyzer;
    // Ensure span values are reasonable
    const safe_span = ast.SourceSpan{
        .line = if (span.line > 1000000) 0 else span.line,
        .column = if (span.column > 10000) 0 else span.column,
        .length = if (span.length > 10000) 0 else span.length,
    };
    return safe_span;
}

/// Get a default safe span for error reporting
pub fn getDefaultSpan(analyzer: *SemanticAnalyzer) ast.SourceSpan {
    _ = analyzer;
    return ast.SourceSpan{ .line = 0, .column = 0, .length = 0 };
}

/// Extract span from any expression type
pub fn getExpressionSpan(analyzer: *SemanticAnalyzer, expr: *ast.ExprNode) ast.SourceSpan {
    _ = analyzer;
    return switch (expr.*) {
        .Identifier => |*ident| ident.span,
        .Literal => |*lit| switch (lit.*) {
            .Integer => |*int| int.span,
            .String => |*str| str.span,
            .Bool => |*b| b.span,
            .Address => |*addr| addr.span,
            .Hex => |*hex| hex.span,
            .Binary => |*bin| bin.span,
            .Char => |*char| char.span,
        },
        .Binary => |*bin| bin.span,
        .Unary => |*un| un.span,
        .Assignment => |*assign| assign.span,
        .CompoundAssignment => |*comp| comp.span,
        .Call => |*call| call.span,
        .Index => |*index| index.span,
        .FieldAccess => |*field| field.span,
        .Cast => |*cast| cast.span,
        .Comptime => |*comp| comp.span,
        .Old => |*old| old.span,
        .Tuple => |*tuple| tuple.span,
        .Try => |*try_expr| try_expr.span,
        .ErrorReturn => |*error_ret| error_ret.span,
        .ErrorCast => |*error_cast| error_cast.span,
        .Shift => |*shift| shift.span,
        .StructInstantiation => |*struct_inst| struct_inst.span,
        .EnumLiteral => |*enum_lit| enum_lit.span,
    };
}

/// Safely analyze a node with error recovery
pub fn safeAnalyzeNode(analyzer: *SemanticAnalyzer, node: *ast.AstNode) semantics_errors.SemanticError!void {
    // Validate node pointer before proceeding
    if (!isValidNodePointer(analyzer, node)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid node pointer detected", getDefaultSpan(analyzer));
        analyzer.validation_coverage.validation_stats.recovery_attempts += 1;
        return semantics_errors.SemanticError.PointerValidationFailed;
    }

    // Set error recovery mode
    const prev_recovery = analyzer.error_recovery_mode;
    analyzer.error_recovery_mode = true;
    defer analyzer.error_recovery_mode = prev_recovery;

    // Update analysis state
    analyzer.analysis_state.current_node_type = @as(std.meta.Tag(ast.AstNode), node.*);
    analyzer.validation_coverage.visited_node_types.insert(@as(std.meta.Tag(ast.AstNode), node.*));
    analyzer.validation_coverage.validation_stats.nodes_analyzed += 1;

    // Attempt analysis with safety checks
    analyzer.analyzeNode(node) catch |err| {
        analyzer.validation_coverage.validation_stats.recovery_attempts += 1;
        try handleAnalysisError(analyzer, err, node);
    };
}

/// Handle analysis errors gracefully
fn handleAnalysisError(analyzer: *SemanticAnalyzer, err: semantics_errors.SemanticError, node: *ast.AstNode) semantics_errors.SemanticError!void {
    const node_type_name = @tagName(@as(std.meta.Tag(ast.AstNode), node.*));

    switch (err) {
        semantics_errors.SemanticError.PointerValidationFailed => {
            if (std.fmt.allocPrint(analyzer.allocator, "Pointer validation failed for {s} node", .{node_type_name})) |message| {
                try semantics_errors.addError(analyzer, message, getDefaultSpan(analyzer)); // Takes ownership
            } else |_| {
                try semantics_errors.addErrorStatic(analyzer, "Pointer validation failed", getDefaultSpan(analyzer)); // Static fallback
            }
        },
        semantics_errors.SemanticError.OutOfMemory => {
            try semantics_errors.addErrorStatic(analyzer, "Out of memory during analysis", getDefaultSpan(analyzer));
            return err; // Don't recover from OOM
        },
        else => {
            if (std.fmt.allocPrint(analyzer.allocator, "Analysis error in {s} node: {s}", .{ node_type_name, @errorName(err) })) |message| {
                try semantics_errors.addWarning(analyzer, message, getDefaultSpan(analyzer)); // Takes ownership
            } else |_| {
                try semantics_errors.addWarningStatic(analyzer, "Analysis error occurred", getDefaultSpan(analyzer)); // Static fallback
            }
        },
    }

    analyzer.validation_coverage.validation_stats.errors_found += 1;
}
