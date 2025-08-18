const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Analyze expression node
pub fn analyzeExpression(analyzer: *SemanticAnalyzer, expr: *ast.ExprNode) semantics_errors.SemanticError!void {
    switch (expr.*) {
        .Identifier => |*ident| {
            try analyzeIdentifier(analyzer, ident);
        },
        .Literal => |*lit| {
            try analyzeLiteral(analyzer, lit);
        },
        .Binary => |*bin| {
            try analyzeBinaryExpression(analyzer, bin);
        },
        .Unary => |*un| {
            try analyzeUnaryExpression(analyzer, un);
        },
        .Assignment => |*assign| {
            try analyzeAssignment(analyzer, assign);
        },
        .CompoundAssignment => |*comp| {
            try analyzeCompoundAssignment(analyzer, comp);
        },
        .Call => |*call| {
            try analyzeCall(analyzer, call);
        },
        .Index => |*index| {
            try analyzeIndex(analyzer, index);
        },
        .FieldAccess => |*field| {
            try analyzeFieldAccess(analyzer, field);
        },
        .Cast => |*cast| {
            try analyzeCast(analyzer, cast);
        },
        .Comptime => |*comp| {
            try analyzeComptime(analyzer, comp);
        },
        .Old => |*old| {
            try analyzeOld(analyzer, old);
        },
        .Tuple => |*tuple| {
            try analyzeTuple(analyzer, tuple);
        },
        .Try => |*try_expr| {
            try analyzeTry(analyzer, try_expr);
        },
        .ErrorReturn => |*error_ret| {
            try analyzeErrorReturn(analyzer, error_ret);
        },
        .ErrorCast => |*error_cast| {
            try analyzeErrorCast(analyzer, error_cast);
        },
        .Shift => |*shift| {
            try analyzeShift(analyzer, shift);
        },
        .StructInstantiation => |*struct_inst| {
            try analyzeStructInstantiation(analyzer, struct_inst);
        },
        .EnumLiteral => |*enum_lit| {
            try analyzeEnumLiteral(analyzer, enum_lit);
        },
        .SwitchExpression => |*switch_expr| {
            try analyzeSwitchExpression(analyzer, switch_expr);
        },
        .Quantified => |*quantified| {
            try analyzeQuantified(analyzer, quantified);
        },
        .AnonymousStruct => |*anon_struct| {
            try analyzeAnonymousStruct(analyzer, anon_struct);
        },
        .Range => |*range| {
            try analyzeRange(analyzer, range);
        },
        .LabeledBlock => |*labeled| {
            try analyzeLabeledBlock(analyzer, labeled);
        },
        .Destructuring => |*destructuring| {
            try analyzeDestructuring(analyzer, destructuring);
        },
        .ArrayLiteral => |*array_lit| {
            try analyzeArrayLiteral(analyzer, array_lit);
        },
    }
}

/// Analyze identifier expression
fn analyzeIdentifier(analyzer: *SemanticAnalyzer, ident: *ast.IdentifierExpr) semantics_errors.SemanticError!void {
    // Only check for valid string memory (not spelling rules)
    if (!semantics_memory_safety.isValidString(analyzer, ident.name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid identifier name (invalid string)", ident.span);
        return;
    }
    // Declaration/type checks are handled by the type checker
}

/// Analyze literal expression
fn analyzeLiteral(analyzer: *SemanticAnalyzer, lit: *ast.LiteralExpr) semantics_errors.SemanticError!void {
    switch (lit.*) {
        .Integer => |*int| {
            // Validate integer literal
            if (!semantics_memory_safety.isValidString(analyzer, int.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid integer literal", int.span);
            }
        },
        .String => |*str| {
            // Validate string literal
            if (!semantics_memory_safety.isValidString(analyzer, str.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid string literal", str.span);
            }
        },
        .Bool => |*b| {
            // Boolean literals are always valid
            _ = b;
        },
        .Address => |*addr| {
            // Validate address literal format
            if (!semantics_memory_safety.isValidString(analyzer, addr.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid address literal", addr.span);
            }
        },
        .Hex => |*hex| {
            // Validate hex literal format
            if (!semantics_memory_safety.isValidString(analyzer, hex.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid hex literal", hex.span);
            }
        },
        .Binary => |*bin| {
            // Validate binary literal format
            if (!semantics_memory_safety.isValidString(analyzer, bin.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid binary literal", bin.span);
            }
        },
    }
}

/// Analyze binary expression
fn analyzeBinaryExpression(analyzer: *SemanticAnalyzer, bin: *ast.BinaryExpr) semantics_errors.SemanticError!void {
    // Analyze left and right operands
    try analyzeExpression(analyzer, bin.lhs);
    try analyzeExpression(analyzer, bin.rhs);

    // Additional semantic checks for binary operations
    switch (bin.operator) {
        .Slash, .Percent => {
            // Check for potential division by zero (basic static analysis)
            if (bin.rhs.* == .Literal and bin.rhs.Literal == .Integer) {
                if (std.mem.eql(u8, bin.rhs.Literal.Integer.value, "0")) {
                    try semantics_errors.addWarningStatic(analyzer, "Potential division by zero", bin.span);
                }
            }
        },
        else => {},
    }
}

/// Analyze unary expression
fn analyzeUnaryExpression(analyzer: *SemanticAnalyzer, un: *ast.UnaryExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, un.operand);

    // Additional semantic checks for unary operations
    // Type validation is handled by the type checker
    switch (un.operator) {
        .Bang => {
            // Logical NOT should be applied to boolean expressions
            // This will be validated by the type checker
        },
        .Minus => {
            // Negation should be applied to numeric expressions
            // This will be validated by the type checker
        },
        else => {},
    }
}

/// Analyze assignment expression
fn analyzeAssignment(analyzer: *SemanticAnalyzer, assign: *ast.AssignmentExpr) semantics_errors.SemanticError!void {
    // Set assignment target flag
    const prev_in_assignment_target = analyzer.in_assignment_target;
    analyzer.in_assignment_target = true;
    defer analyzer.in_assignment_target = prev_in_assignment_target;

    // Analyze target (left side)
    try analyzeExpression(analyzer, assign.target);

    // Reset flag for value analysis
    analyzer.in_assignment_target = false;

    // Analyze value (right side)
    try analyzeExpression(analyzer, assign.value);

    // Additional semantic checks for assignments
    // Check for immutable variable violations, etc.
}

/// Analyze compound assignment expression
fn analyzeCompoundAssignment(analyzer: *SemanticAnalyzer, comp: *ast.CompoundAssignmentExpr) semantics_errors.SemanticError!void {
    // Similar to regular assignment but with operation
    const prev_in_assignment_target = analyzer.in_assignment_target;
    analyzer.in_assignment_target = true;
    defer analyzer.in_assignment_target = prev_in_assignment_target;

    try analyzeExpression(analyzer, comp.target);
    analyzer.in_assignment_target = false;
    try analyzeExpression(analyzer, comp.value);
}

// Placeholder implementations for other expression types
fn analyzeCall(analyzer: *SemanticAnalyzer, call: *ast.CallExpr) semantics_errors.SemanticError!void {
    // Analyze function expression
    try analyzeExpression(analyzer, call.callee);

    // Analyze arguments
    for (call.arguments) |arg| {
        try analyzeExpression(analyzer, arg);
    }
}

fn analyzeIndex(analyzer: *SemanticAnalyzer, index: *ast.IndexExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, index.target);
    try analyzeExpression(analyzer, index.index);
}

fn analyzeFieldAccess(analyzer: *SemanticAnalyzer, field: *ast.FieldAccessExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, field.target);
    // Field name validation
    if (!semantics_memory_safety.isValidString(analyzer, field.field)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid field name", field.span);
    }
}

fn analyzeCast(analyzer: *SemanticAnalyzer, cast: *ast.CastExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, cast.operand);
    // Type validation will be handled by type checker
}

fn analyzeComptime(analyzer: *SemanticAnalyzer, comp: *ast.ComptimeExpr) semantics_errors.SemanticError!void {
    // ComptimeExpr contains a block, not an expression - analyze it as a block
    // Note: Proper block analysis would be handled by the statement analyzer
    _ = comp; // ComptimeExpr doesn't need expression analysis
    _ = analyzer;
}

fn analyzeOld(analyzer: *SemanticAnalyzer, old: *ast.OldExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, old.expr);
    // Validate that 'old' is used in appropriate context (ensures clauses)
}

fn analyzeTuple(analyzer: *SemanticAnalyzer, tuple: *ast.TupleExpr) semantics_errors.SemanticError!void {
    for (tuple.elements) |elem| {
        try analyzeExpression(analyzer, elem);
    }
}

fn analyzeTry(analyzer: *SemanticAnalyzer, try_expr: *ast.TryExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, try_expr.expr);
    // Validate error propagation context
    if (!analyzer.in_error_propagation_context and !analyzer.current_function_returns_error_union) {
        try semantics_errors.addErrorStatic(analyzer, "Try expression in non-error context", try_expr.span);
    }
}

fn analyzeErrorReturn(analyzer: *SemanticAnalyzer, error_ret: *ast.ErrorReturnExpr) semantics_errors.SemanticError!void {
    // ErrorReturnExpr just contains an error name, validate the name
    if (!semantics_memory_safety.isValidString(analyzer, error_ret.error_name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid error name", error_ret.span);
    }
}

fn analyzeErrorCast(analyzer: *SemanticAnalyzer, error_cast: *ast.ErrorCastExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, error_cast.operand);
}

fn analyzeShift(analyzer: *SemanticAnalyzer, shift: *ast.ShiftExpr) semantics_errors.SemanticError!void {
    try analyzeExpression(analyzer, shift.mapping);
    try analyzeExpression(analyzer, shift.source);
    try analyzeExpression(analyzer, shift.dest);
    try analyzeExpression(analyzer, shift.amount);
}

fn analyzeStructInstantiation(analyzer: *SemanticAnalyzer, struct_inst: *ast.StructInstantiationExpr) semantics_errors.SemanticError!void {
    for (struct_inst.fields) |*field| {
        try analyzeExpression(analyzer, field.value);
    }
}

fn analyzeEnumLiteral(analyzer: *SemanticAnalyzer, enum_lit: *ast.EnumLiteralExpr) semantics_errors.SemanticError!void {
    if (!semantics_memory_safety.isValidString(analyzer, enum_lit.enum_name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid enum name", enum_lit.span);
    }
    if (!semantics_memory_safety.isValidString(analyzer, enum_lit.variant_name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid variant name", enum_lit.span);
    }
}

fn analyzeSwitchExpression(analyzer: *SemanticAnalyzer, switch_expr: *ast.SwitchExprNode) semantics_errors.SemanticError!void {
    // Analyze the switch condition
    try analyzeExpression(analyzer, switch_expr.condition);

    // Analyze each case
    for (switch_expr.cases) |*case| {
        // Analyze the case body
        switch (case.body) {
            .Expression => |expr| {
                try analyzeExpression(analyzer, expr);
            },
            .Block => {
                // Block analysis is handled by the statement analyzer
            },
            .LabeledBlock => {
                // LabeledBlock analysis is handled by the statement analyzer
            },
        }
    }

    // Analyze default case if present
    if (switch_expr.default_case) |*default_case| {
        // Default case is just a block - block analysis would be handled by the statement analyzer
        _ = default_case; // Suppress unused variable warning
    }
}

fn analyzeQuantified(analyzer: *SemanticAnalyzer, quantified: *ast.QuantifiedExpr) semantics_errors.SemanticError!void {
    // Analyze the domain expression
    // Analyze the condition if present
    if (quantified.condition) |condition| {
        try analyzeExpression(analyzer, condition);
    }

    // Analyze the body expression
    try analyzeExpression(analyzer, quantified.body);

    // Validate quantifier variable name
    if (!semantics_memory_safety.isValidString(analyzer, quantified.variable)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid quantifier variable name", quantified.span);
    }
}

fn analyzeAnonymousStruct(analyzer: *SemanticAnalyzer, anon_struct: *ast.AnonymousStructExpr) semantics_errors.SemanticError!void {
    // Analyze all field initializers
    for (anon_struct.fields) |*field| {
        try analyzeExpression(analyzer, field.value);

        // Validate field name
        if (!semantics_memory_safety.isValidString(analyzer, field.name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid anonymous struct field name", anon_struct.span);
        }
    }
}

fn analyzeRange(analyzer: *SemanticAnalyzer, range: *ast.RangeExpr) semantics_errors.SemanticError!void {
    // Analyze start expression
    try analyzeExpression(analyzer, range.start);

    // Analyze end expression
    try analyzeExpression(analyzer, range.end);

    // TODO: Validate that start and end are comparable types
}

fn analyzeLabeledBlock(analyzer: *SemanticAnalyzer, labeled: *ast.LabeledBlockExpr) semantics_errors.SemanticError!void {
    // Validate label name
    if (!semantics_memory_safety.isValidString(analyzer, labeled.label)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid label name", labeled.span);
    }

    // Block analysis is handled by the statement analyzer
    // The block itself will be analyzed when the statement analyzer processes it
}

fn analyzeDestructuring(analyzer: *SemanticAnalyzer, destructuring: *ast.DestructuringExpr) semantics_errors.SemanticError!void {
    // Analyze the source expression being destructured
    try analyzeExpression(analyzer, destructuring.value);

    // Validate destructuring pattern based on its type
    switch (destructuring.pattern) {
        .Array => |names| {
            // Validate array destructuring element names
            for (names) |name| {
                if (!semantics_memory_safety.isValidString(analyzer, name)) {
                    try semantics_errors.addErrorStatic(analyzer, "Invalid destructuring variable name", destructuring.span);
                }
            }
        },
        .Struct => |fields| {
            // Validate struct destructuring fields
            for (fields) |*field| {
                if (!semantics_memory_safety.isValidString(analyzer, field.name)) {
                    try semantics_errors.addErrorStatic(analyzer, "Invalid destructuring field name", destructuring.span);
                }
                if (!semantics_memory_safety.isValidString(analyzer, field.variable)) {
                    try semantics_errors.addErrorStatic(analyzer, "Invalid destructuring variable name", destructuring.span);
                }
            }
        },
        .Tuple => |names| {
            // Validate tuple destructuring element names
            for (names) |name| {
                if (!semantics_memory_safety.isValidString(analyzer, name)) {
                    try semantics_errors.addErrorStatic(analyzer, "Invalid destructuring variable name", destructuring.span);
                }
            }
        },
    }
}

fn analyzeArrayLiteral(analyzer: *SemanticAnalyzer, array_lit: *ast.ArrayLiteralExpr) semantics_errors.SemanticError!void {
    // Analyze all array elements
    for (array_lit.elements) |element| {
        try analyzeExpression(analyzer, element);
    }

    // TODO: Validate that all elements have compatible types
}
