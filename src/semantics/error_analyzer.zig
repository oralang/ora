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
const ManagedArrayList = std.array_list.Managed;

pub const ErrorValidationResult = struct {
    diagnostics: ManagedArrayList(ast.SourceSpan),
};

/// Validate error declarations and error usage
pub fn validateErrors(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    nodes: []const ast.AstNode,
) std.mem.Allocator.Error!ErrorValidationResult {
    var diags = ManagedArrayList(ast.SourceSpan).init(allocator);

    // walk AST and validate error usage
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
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    switch (node) {
        .Contract => |*contract| {
            for (contract.body) |*member| {
                try validateNodeErrors(allocator, table, member.*, diags);
            }
        },
        .Function => |*function| {
            // validate function body for error returns
            try validateFunctionErrors(table, function, diags);
        },
        else => {},
    }
}

/// Validate error usage in a function
fn validateFunctionErrors(
    table: *state.SymbolTable,
    function: *const ast.FunctionNode,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    // validate function return type contains valid error names
    if (function.return_type_info) |ret_type| {
        try validateErrorUnionType(table, ret_type, diags);
    }

    // walk function body and validate error returns
    try validateBlockErrors(table, &function.body, diags);
}

/// Validate errors in a block
fn validateBlockErrors(
    table: *state.SymbolTable,
    block: *const ast.Statements.BlockNode,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    for (block.statements) |*stmt| {
        try validateStatementErrors(table, stmt, diags);
    }
}

/// Validate errors in a statement
fn validateStatementErrors(
    table: *state.SymbolTable,
    stmt: *const ast.Statements.StmtNode,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    switch (stmt.*) {
        .Expr => |*expr| {
            try validateExpressionErrors(table, expr, diags);
        },
        .Return => |*ret| {
            if (ret.value) |*value| {
                try validateExpressionErrors(table, value, diags);
            }
        },
        .If => |*if_stmt| {
            try validateExpressionErrors(table, &if_stmt.condition, diags);
            try validateBlockErrors(table, &if_stmt.then_branch, diags);
            if (if_stmt.else_branch) |*else_block| {
                try validateBlockErrors(table, else_block, diags);
            }
        },
        .While => |*while_stmt| {
            try validateExpressionErrors(table, &while_stmt.condition, diags);
            try validateBlockErrors(table, &while_stmt.body, diags);
        },
        .ForLoop => |*for_stmt| {
            try validateExpressionErrors(table, &for_stmt.iterable, diags);
            try validateBlockErrors(table, &for_stmt.body, diags);
        },
        .Switch => |*switch_stmt| {
            for (switch_stmt.cases) |*case| {
                switch (case.body) {
                    .Expression => |expr_ptr| try validateExpressionErrors(table, expr_ptr, diags),
                    .Block => |*blk| try validateBlockErrors(table, blk, diags),
                    .LabeledBlock => |*lb| try validateBlockErrors(table, &lb.block, diags),
                }
            }
            if (switch_stmt.default_case) |*default_block| {
                try validateBlockErrors(table, default_block, diags);
            }
        },
        .TryBlock => |*try_block| {
            try validateBlockErrors(table, &try_block.try_block, diags);
            if (try_block.catch_block) |*catch_block| {
                try validateBlockErrors(table, &catch_block.block, diags);
            }
        },
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |value| {
                try validateExpressionErrors(table, value, diags);
            }
        },
        .CompoundAssignment => |*assignment| {
            try validateExpressionErrors(table, assignment.target, diags);
            try validateExpressionErrors(table, assignment.value, diags);
        },
        else => {},
    }
}

/// Validate errors in an expression
fn validateExpressionErrors(
    table: *state.SymbolTable,
    expr: *const ast.Expressions.ExprNode,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    switch (expr.*) {
        .ErrorReturn => |*error_return| {
            // validate that the error is declared (error.SomeError syntax)
            if (state.SymbolTable.findUp(null, error_return.error_name)) |symbol| {
                if (symbol.kind != .Error) {
                    try diags.append(error_return.span);
                }
                // check if error has parameters (shouldn't use error.X syntax for errors with params)
                if (table.error_signatures.get(error_return.error_name)) |params| {
                    if (params != null and params.?.len > 0) {
                        // error has parameters but used error.X syntax (should use ErrorName(args))
                        try diags.append(error_return.span);
                    }
                }
            } else {
                // error not found
                try diags.append(error_return.span);
            }
        },
        .Call => |*call| {
            // check if this is an error call (ErrorName(args))
            if (call.callee.* == .Identifier) {
                const callee_name = call.callee.Identifier.name;

                // check if callee is an error
                if (state.SymbolTable.findUp(null, callee_name)) |symbol| {
                    if (symbol.kind == .Error) {
                        // this is an error call - validate parameters
                        try validateErrorCall(table, callee_name, call.arguments, call.span, diags);
                    }
                }
            }

            // validate call arguments (may contain error expressions)
            for (call.arguments) |arg| {
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
    arguments: []const *ast.Expressions.ExprNode,
    span: ast.SourceSpan,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    const error_params = table.error_signatures.get(error_name);

    if (error_params == null) {
        // error not found in signatures (shouldn't happen if symbol exists, but check anyway)
        try diags.append(span);
        return;
    }

    const params = error_params.?;

    if (params == null) {
        // error has no parameters
        if (arguments.len > 0) {
            // but arguments were provided
            try diags.append(span);
        }
        return;
    }

    // error has parameters - validate count and types
    const param_list = params.?;

    if (arguments.len != param_list.len) {
        // argument count mismatch
        try diags.append(span);
        return;
    }

    // type checking for error parameters is done in ast/type_resolver/
    // this function only validates parameter count
}

/// Validate error union type contains only declared errors
fn validateErrorUnionType(
    table: *state.SymbolTable,
    type_info: ast.Types.TypeInfo,
    diags: *ManagedArrayList(ast.SourceSpan),
) std.mem.Allocator.Error!void {
    if (type_info.ora_type) |ora_type| {
        switch (ora_type) {
            .error_union => |success_type| {
                // simple error union !T - no error names to validate
                _ = success_type;
            },
            ._union => |union_types| {
                // error union with explicit errors: !T | Error1 | Error2
                // first type should be error_union, rest should be error types
                for (union_types, 0..) |union_type, i| {
                    if (i == 0) {
                        // first should be error_union
                        switch (union_type) {
                            .error_union => {},
                            else => {
                                // invalid: first type in union should be error_union
                                if (type_info.span) |span| {
                                    try diags.append(span);
                                }
                            },
                        }
                    } else {
                        // subsequent types should be error type names
                        switch (union_type) {
                            .struct_type => |error_name| {
                                // check if this is actually an error name
                                // first check in symbol table for error declarations
                                if (table.error_signatures.get(error_name)) |_| {
                                    // found in error signatures - it's a valid error
                                } else if (state.SymbolTable.findUp(null, error_name)) |symbol| {
                                    if (symbol.kind != .Error) {
                                        // not an error type
                                        if (type_info.span) |span| {
                                            try diags.append(span);
                                        }
                                    }
                                } else {
                                    // error name not found
                                    if (type_info.span) |span| {
                                        try diags.append(span);
                                    }
                                }
                            },
                            else => {
                                // invalid: should be an error type name
                                if (type_info.span) |span| {
                                    try diags.append(span);
                                }
                            },
                        }
                    }
                }
            },
            else => {
                // not an error union type, nothing to validate
            },
        }
    }
}
