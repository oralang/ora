const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const collect = @import("collect.zig");
const contract = @import("contract_analyzer.zig");
const func = @import("function_analyzer.zig");
const stmt = @import("statement_analyzer.zig");
const logs = @import("log_analyzer.zig");
const expr = @import("expression_analyzer.zig");
// TODO(semantics): Re-enable unknown-identifier walker when scope mapping is fully hardened.
// Current crash root cause: walking identifiers while some nested block/function scopes
// are not yet recorded leads to invalid parent chains in findUp().
// Hardening plan:
//  - Ensure all function/block scopes are registered before Phase 2
//  - Add defensive parent-chain validation in SymbolTable.findUp
//  - Consider constraining unknowns to current + function scope first
const ENABLE_UNKNOWN_WALKER: bool = false;
const locals = @import("locals_binder.zig");

pub const Diagnostic = struct {
    message: []const u8,
    span: ast.SourceSpan,
};

pub const SemanticsResult = struct {
    symbols: state.SymbolTable,
    diagnostics: []Diagnostic,
};

pub fn analyzePhase1(allocator: std.mem.Allocator, nodes: []const ast.AstNode) !SemanticsResult {
    var cr = try collect.collectSymbols(allocator, nodes);
    defer cr.diagnostics.deinit();
    // Convert simple span list into diagnostics for now
    var diags = std.ArrayList(Diagnostic).init(allocator);
    for (cr.diagnostics.items) |sp| {
        try diags.append(.{ .message = "Redeclaration in scope", .span = sp });
    }
    // Build function scopes for top-level functions and contract members
    for (nodes) |*n| switch (n.*) {
        .Contract => |*c| try contract.collectContractSymbols(&cr.table, &cr.table.root, c),
        .Function => |*f| try func.collectFunctionSymbols(&cr.table, &cr.table.root, f),
        else => {},
    };
    // Bind locals and nested block scopes for all functions (top-level and contract members)
    for (nodes) |*n| switch (n.*) {
        .Function => |*f| {
            if (cr.table.function_scopes.get(f.name)) |fs| {
                try locals.bindFunctionLocals(&cr.table, fs, f);
            }
        },
        .Contract => |*c| {
            for (c.body) |*m| switch (m.*) {
                .Function => |*f2| {
                    if (cr.table.function_scopes.get(f2.name)) |fs2| {
                        try locals.bindFunctionLocals(&cr.table, fs2, f2);
                    }
                },
                else => {},
            };
        },
        else => {},
    };
    return .{ .symbols = cr.table, .diagnostics = try diags.toOwnedSlice() };
}

// Phase 2 scaffolding: type-check and validation will be added here, orchestrating analyzers.
pub fn analyzePhase2(allocator: std.mem.Allocator, nodes: []const ast.AstNode, symbols: *state.SymbolTable) ![]Diagnostic {
    var diags = std.ArrayList(Diagnostic).init(allocator);
    // Walk functions and check returns basic rule
    for (nodes) |*n| switch (n.*) {
        .Function => |*f| {
            const scope = symbols.function_scopes.get(f.name) orelse null;
            if (scope) |s| {
                const spans = try stmt.checkFunctionBody(allocator, symbols, s, f);
                defer allocator.free(spans);
                for (spans) |sp| try diags.append(.{ .message = "Return type mismatch", .span = sp });

                // Unknown identifiers in function body
                if (ENABLE_UNKNOWN_WALKER and f.body.statements.len > 0) {
                    const unknowns = try stmt.collectUnknownIdentifierSpans(allocator, symbols, s, f);
                    defer allocator.free(unknowns);
                    for (unknowns) |usp| try diags.append(.{ .message = "Unknown identifier", .span = usp });
                }

                // Log statement checks (signature and argument typing)
                const log_spans = try logs.checkLogsInFunction(allocator, symbols, s, f);
                defer allocator.free(log_spans);
                for (log_spans) |lsp| try diags.append(.{ .message = "Invalid log call", .span = lsp });

                // Spec usage checks: quantified allowed only in requires/ensures/invariant; old only in ensures
                for (f.requires_clauses) |rq| {
                    if (try expr.validateSpecUsage(allocator, rq, .Requires)) |spx| {
                        try diags.append(.{ .message = "Invalid spec expression usage", .span = spx });
                    }
                }
                for (f.ensures_clauses) |en| {
                    if (try expr.validateSpecUsage(allocator, en, .Ensures)) |spy| {
                        try diags.append(.{ .message = "Invalid spec expression usage", .span = spy });
                    }
                }
            }
        },
        .Contract => |*c| {
            // Check each member function
            for (c.body) |*m| switch (m.*) {
                .Function => |*f2| {
                    const scope2 = symbols.function_scopes.get(f2.name) orelse null;
                    if (scope2) |s2| {
                        const spans2 = try stmt.checkFunctionBody(allocator, symbols, s2, f2);
                        defer allocator.free(spans2);
                        for (spans2) |sp2| try diags.append(.{ .message = "Return type mismatch", .span = sp2 });

                        // Unknown identifiers in member function body
                        if (ENABLE_UNKNOWN_WALKER and f2.body.statements.len > 0) {
                            const unknowns2 = try stmt.collectUnknownIdentifierSpans(allocator, symbols, s2, f2);
                            defer allocator.free(unknowns2);
                            for (unknowns2) |usp2| try diags.append(.{ .message = "Unknown identifier", .span = usp2 });
                        }

                        // Log statement checks
                        const log_spans2 = try logs.checkLogsInFunction(allocator, symbols, s2, f2);
                        defer allocator.free(log_spans2);
                        for (log_spans2) |lsp2| try diags.append(.{ .message = "Invalid log call", .span = lsp2 });

                        // Spec usage checks: quantified allowed only in requires/ensures/invariant; old only in ensures
                        for (f2.requires_clauses) |rq2| {
                            if (try expr.validateSpecUsage(allocator, rq2, .Requires)) |spx2| {
                                try diags.append(.{ .message = "Invalid spec expression usage", .span = spx2 });
                            }
                        }
                        for (f2.ensures_clauses) |en2| {
                            if (try expr.validateSpecUsage(allocator, en2, .Ensures)) |spy2| {
                                try diags.append(.{ .message = "Invalid spec expression usage", .span = spy2 });
                            }
                        }
                    }
                },
                else => {},
            };
        },
        else => {},
    };
    return try diags.toOwnedSlice();
}
