// ============================================================================
// Log Statement Analyzer
// ============================================================================
//
// Validates log statements against declared log signatures.
//
// VALIDATION RULES:
//   • Log name must exist in symbol table
//   • Argument count must match signature
//   • Argument types must match field types
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");

pub fn checkLogStatement(
    allocator: std.mem.Allocator,
    symbols: *state.SymbolTable,
    scope: *state.Scope,
    log_stmt: *const ast.Statements.LogNode,
) ![]const ast.SourceSpan {
    var issues = std.ArrayList(ast.SourceSpan).init(allocator);
    // Resolve log declaration by name
    const sig_fields_opt = symbols.log_signatures.get(log_stmt.event_name);
    if (sig_fields_opt == null) {
        // Unknown log name
        try issues.append(log_stmt.span);
        return try issues.toOwnedSlice();
    }
    const sig_fields = sig_fields_opt.?;

    // Arity check
    if (sig_fields.len != log_stmt.args.len) {
        try issues.append(log_stmt.span);
        return try issues.toOwnedSlice();
    }

    // Type check each argument
    var i: usize = 0;
    while (i < sig_fields.len) : (i += 1) {
        const field = sig_fields[i];
        const arg_expr = log_stmt.args[i];
        const arg_ti = expr.inferExprType(symbols, scope, arg_expr);
        if (!ast.Types.TypeInfo.equals(arg_ti, field.type_info)) {
            try issues.append(log_stmt.span);
            // continue to collect more issues
        }
    }

    return try issues.toOwnedSlice();
}

pub fn checkLogsInFunction(
    allocator: std.mem.Allocator,
    symbols: *state.SymbolTable,
    scope: *state.Scope,
    f: *const ast.FunctionNode,
) ![]const ast.SourceSpan {
    var issues = std.ArrayList(ast.SourceSpan).init(allocator);
    try walkBlockForLogs(&issues, symbols, scope, &f.body);
    return try issues.toOwnedSlice();
}

fn walkBlockForLogs(
    issues: *std.ArrayList(ast.SourceSpan),
    symbols: *state.SymbolTable,
    scope: *state.Scope,
    block: *const ast.Statements.BlockNode,
) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .Log => |log_stmt| {
                const spans = try checkLogStatement(issues.allocator, symbols, scope, &log_stmt);
                defer issues.allocator.free(spans);
                for (spans) |sp| try issues.append(sp);
            },
            .Expr => |e| {
                // Dive into expression statements only if they contain blocks (e.g., labeled blocks)
                switch (e) {
                    .LabeledBlock => |lb| try walkBlockForLogs(issues, symbols, scope, &lb.block),
                    .Comptime => |ct| try walkBlockForLogs(issues, symbols, scope, &ct.block),
                    else => {},
                }
            },
            .If => |iff| {
                try walkBlockForLogs(issues, symbols, scope, &iff.then_branch);
                if (iff.else_branch) |*eb| try walkBlockForLogs(issues, symbols, scope, eb);
            },
            .While => |wh| try walkBlockForLogs(issues, symbols, scope, &wh.body),
            .ForLoop => |fl| try walkBlockForLogs(issues, symbols, scope, &fl.body),
            .TryBlock => |tb| {
                try walkBlockForLogs(issues, symbols, scope, &tb.try_block);
                if (tb.catch_block) |cb| try walkBlockForLogs(issues, symbols, scope, &cb.block);
            },
            // No plain Block variant in StmtNode
            .Switch => |sw| {
                for (sw.cases) |*case| switch (case.body) {
                    .Block => |*blk| try walkBlockForLogs(issues, symbols, scope, blk),
                    .LabeledBlock => |*lbl| try walkBlockForLogs(issues, symbols, scope, &lbl.block),
                    else => {},
                };
                if (sw.default_case) |*db| try walkBlockForLogs(issues, symbols, scope, db);
            },
            else => {},
        }
    }
}
