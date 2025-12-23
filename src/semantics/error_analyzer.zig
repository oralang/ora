// ============================================================================
// Error Analyzer - Error Handling Validation
// ============================================================================
//
// Validates error declarations, error returns, and error handling.
//
// VALIDATION RULES:
//   • Error return expressions must reference declared errors
//   • Error parameters must match error declaration signatures
//   • Error union types must be valid
//   • Try-catch blocks must handle errors correctly
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");

pub const ErrorValidationResult = struct {
    diagnostics: std.ArrayList(ast.SourceSpan),
};

/// Validate error declarations and error usage
pub fn validateErrors(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    nodes: []const ast.AstNode,
) !ErrorValidationResult {
    var diags = std.ArrayList(ast.SourceSpan).init(allocator);

    // Walk AST and validate error usage
    for (nodes) |node| {
        try validateNodeErrors(allocator, table, node, &diags);
    }

    return .{ .diagnostics = diags };
}

/// Recursively validate errors in a node
fn validateNodeErrors(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    node: ast.AstNode,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    switch (node) {
        .Contract => |*contract| {
            for (contract.body) |*member| {
                try validateNodeErrors(allocator, table, member.*, diags);
            }
        },
        .Function => |*function| {
            // Validate function body for error returns
            try validateFunctionErrors(table, function, diags);
        },
        else => {},
    }
}

/// Validate error usage in a function
fn validateFunctionErrors(
    table: *state.SymbolTable,
    function: *const ast.FunctionNode,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    // Validate function return type contains valid error names
    if (function.return_type_info) |ret_type| {
        try validateErrorUnionType(table, ret_type, diags);
    }

    // Walk function body and validate error returns
    try validateBlockErrors(table, &function.body, diags);
}

/// Validate errors in a block
fn validateBlockErrors(
    table: *state.SymbolTable,
    block: *const ast.Statements.BlockNode,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    for (block.statements) |*stmt| {
        try validateStatementErrors(table, stmt, diags);
    }
}

/// Validate errors in a statement
fn validateStatementErrors(
    table: *state.SymbolTable,
    stmt: *const ast.Statements.StmtNode,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    switch (stmt.*) {
        .Return => |*ret| {
            if (ret.value) |*value| {
                try validateExpressionErrors(table, value, diags);
            }
        },
        .If => |*if_stmt| {
            try validateBlockErrors(table, &if_stmt.then_block, diags);
            if (if_stmt.else_block) |*else_block| {
                try validateBlockErrors(table, else_block, diags);
            }
        },
        .While => |*while_stmt| {
            try validateBlockErrors(table, &while_stmt.body, diags);
        },
        .ForLoop => |*for_stmt| {
            try validateBlockErrors(table, &for_stmt.body, diags);
        },
        .Switch => |*switch_stmt| {
            for (switch_stmt.cases) |*case| {
                try validateBlockErrors(table, &case.block, diags);
            }
        },
        .TryBlock => |*try_block| {
            try validateBlockErrors(table, &try_block.try_block, diags);
            if (try_block.catch_block) |*catch_block| {
                try validateBlockErrors(table, &catch_block.block, diags);
            }
        },
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |*value| {
                try validateExpressionErrors(table, value, diags);
            }
        },
        .Assignment => |*assignment| {
            try validateExpressionErrors(table, assignment.value, diags);
        },
        else => {},
    }
}

/// Validate errors in an expression
fn validateExpressionErrors(
    table: *state.SymbolTable,
    expr: *const ast.Expressions.ExprNode,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    switch (expr.*) {
        .ErrorReturn => |*error_return| {
            // Validate that the error is declared (error.SomeError syntax)
            if (state.SymbolTable.findUp(null, error_return.error_name)) |symbol| {
                if (symbol.kind != .Error) {
                    try diags.append(error_return.span);
                }
                // Check if error has parameters (shouldn't use error.X syntax for errors with params)
                if (table.error_signatures.get(error_return.error_name)) |params| {
                    if (params != null and params.?.len > 0) {
                        // Error has parameters but used error.X syntax (should use ErrorName(args))
                        try diags.append(error_return.span);
                    }
                }
            } else {
                // Error not found
                try diags.append(error_return.span);
            }
        },
        .Call => |*call| {
            // Check if this is an error call (ErrorName(args))
            if (call.callee.* == .Identifier) {
                const callee_name = call.callee.Identifier.name;

                // Check if callee is an error
                if (state.SymbolTable.findUp(null, callee_name)) |symbol| {
                    if (symbol.kind == .Error) {
                        // This is an error call - validate parameters
                        try validateErrorCall(table, callee_name, call.arguments, call.span, diags);
                    }
                }
            }

            // Validate call arguments (may contain error expressions)
            for (call.arguments) |*arg| {
                try validateExpressionErrors(table, arg, diags);
            }
        },
        .Binary => |*binary| {
            try validateExpressionErrors(table, binary.lhs, diags);
            try validateExpressionErrors(table, binary.rhs, diags);
        },
        .Unary => |*unary| {
            try validateExpressionErrors(table, unary.operand, diags);
        },
        .Try => |*try_expr| {
            try validateExpressionErrors(table, try_expr.expr, diags);
        },
        else => {},
    }
}

/// Validate an error call matches the error signature
fn validateErrorCall(
    table: *state.SymbolTable,
    error_name: []const u8,
    arguments: []const ast.Expressions.ExprNode,
    span: ast.SourceSpan,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    const error_params = table.error_signatures.get(error_name);

    if (error_params == null) {
        // Error not found in signatures (shouldn't happen if symbol exists, but check anyway)
        try diags.append(span);
        return;
    }

    const params = error_params.?;

    if (params == null) {
        // Error has no parameters
        if (arguments.len > 0) {
            // But arguments were provided
            try diags.append(span);
        }
        return;
    }

    // Error has parameters - validate count and types
    const param_list = params.?;

    if (arguments.len != param_list.len) {
        // Argument count mismatch
        try diags.append(span);
        return;
    }

    // Type checking for error parameters is done in ast/type_resolver/
    // This function only validates parameter count
}

/// Validate error union type contains only declared errors
fn validateErrorUnionType(
    table: *state.SymbolTable,
    type_info: ast.Types.TypeInfo,
    diags: *std.ArrayList(ast.SourceSpan),
) !void {
    if (type_info.ora_type) |ora_type| {
        switch (ora_type) {
            .error_union => |success_type| {
                // Simple error union !T - no error names to validate
                _ = success_type;
            },
            ._union => |union_types| {
                // Error union with explicit errors: !T | Error1 | Error2
                // First type should be error_union, rest should be error types
                for (union_types, 0..) |union_type, i| {
                    if (i == 0) {
                        // First should be error_union
                        switch (union_type) {
                            .error_union => {},
                            else => {
                                // Invalid: first type in union should be error_union
                                if (type_info.span) |span| {
                                    try diags.append(span);
                                }
                            },
                        }
                    } else {
                        // Subsequent types should be error type names
                        switch (union_type) {
                            .struct_type => |error_name| {
                                // Check if this is actually an error name
                                // First check in symbol table for error declarations
                                if (table.error_signatures.get(error_name)) |_| {
                                    // Found in error signatures - it's a valid error
                                } else if (state.SymbolTable.findUp(null, error_name)) |symbol| {
                                    if (symbol.kind != .Error) {
                                        // Not an error type
                                        if (type_info.span) |span| {
                                            try diags.append(span);
                                        }
                                    }
                                } else {
                                    // Error name not found
                                    if (type_info.span) |span| {
                                        try diags.append(span);
                                    }
                                }
                            },
                            else => {
                                // Invalid: should be an error type name
                                if (type_info.span) |span| {
                                    try diags.append(span);
                                }
                            },
                        }
                    }
                }
            },
            else => {
                // Not an error union type, nothing to validate
            },
        }
    }
}
