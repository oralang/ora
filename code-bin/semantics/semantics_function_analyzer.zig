const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Analyze function declaration
pub fn analyzeFunction(analyzer: *SemanticAnalyzer, function: *ast.FunctionNode) semantics_errors.SemanticError!void {
    const prev_function = analyzer.current_function;
    analyzer.current_function = function.name;
    defer analyzer.current_function = prev_function;

    // Check if function returns error union
    const prev_returns_error_union = analyzer.current_function_returns_error_union;
    analyzer.current_function_returns_error_union = functionReturnsErrorUnion(function);
    defer analyzer.current_function_returns_error_union = prev_returns_error_union;

    // Note: The type checker has already validated the function body in Phase 1
    // The semantic analyzer should not create new scopes or interfere with type checker scopes

    // Validate init function requirements
    if (std.mem.eql(u8, function.name, "init")) {
        try validateInitFunction(analyzer, function);
    }

    // Analyze requires clauses
    for (function.requires_clauses) |clause| {
        try analyzeRequiresClause(analyzer, clause, function.span);
    }

    // Analyze ensures clauses
    for (function.ensures_clauses) |clause| {
        try analyzeEnsuresClause(analyzer, clause, function.span);
    }

    // Perform static verification on requires/ensures clauses
    try performStaticVerification(analyzer, function);

    // Analyze function body with proper scope context
    try analyzeBlock(analyzer, &function.body);

    // Validate return statements if function has return type
    if (function.return_type_info != null) {
        try validateReturnStatements(analyzer, &function.body, function.span);
    }
}

/// Check if function returns error union
fn functionReturnsErrorUnion(function: *ast.FunctionNode) bool {
    if (function.return_type_info) |return_type_info| {
        return switch (return_type_info.ora_type orelse return false) {
            .error_union => true,
            else => false,
        };
    }
    return false;
}

/// Validate init function requirements
fn validateInitFunction(analyzer: *SemanticAnalyzer, function: *ast.FunctionNode) semantics_errors.SemanticError!void {
    // Init functions should be public
    if (function.visibility == .Private) {
        try semantics_errors.addWarningStatic(analyzer, "Init function should be public", function.span);
    }

    // Init functions should not have return type
    if (function.return_type_info != null) {
        try semantics_errors.addErrorStatic(analyzer, "Init function cannot have return type", function.span);
    }

    // Init functions should not have requires/ensures (for now)
    if (function.requires_clauses.len > 0) {
        try semantics_errors.addWarningStatic(analyzer, "Init function with requires clauses - verify carefully", function.span);
    }
}

// Placeholder functions that will be implemented in other modules
fn analyzeRequiresClause(analyzer: *SemanticAnalyzer, clause: *ast.ExprNode, function_span: ast.SourceSpan) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = clause;
    _ = function_span;
    // Will be implemented in formal verification module
}

fn analyzeEnsuresClause(analyzer: *SemanticAnalyzer, clause: *ast.ExprNode, function_span: ast.SourceSpan) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = clause;
    _ = function_span;
    // Will be implemented in formal verification module
}

fn performStaticVerification(analyzer: *SemanticAnalyzer, function: *ast.FunctionNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = function;
    // Will be implemented in static verification module
}

fn analyzeBlock(analyzer: *SemanticAnalyzer, block: *ast.BlockNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = block;
    // Will be implemented in statement analyzer module
}

fn validateReturnStatements(analyzer: *SemanticAnalyzer, block: *ast.BlockNode, function_span: ast.SourceSpan) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = block;
    _ = function_span;
    // Will be implemented in validation module
}
