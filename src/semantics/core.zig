// ============================================================================
// Semantic Analysis Core - Orchestrator
// ============================================================================
//
// Coordinates the two-phase semantic analysis pipeline.
//
// TWO-PHASE ARCHITECTURE:
//   **Phase 1:** Symbol collection & scope building
//     Collect symbols → Build scopes → Bind locals
//   **Phase 2:** Validation & type checking
//     Check bodies → Validate returns → Check logs → Validate specs
//
// Delegates to: collect.zig, contract_analyzer.zig, function_analyzer.zig,
//               locals_binder.zig, statement_analyzer.zig, log_analyzer.zig
//
// NOTE: Unknown-identifier checking currently disabled (see ENABLE_UNKNOWN_WALKER).
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const collect = @import("collect.zig");
const contract = @import("contract_analyzer.zig");
const func = @import("function_analyzer.zig");
const stmt = @import("statement_analyzer.zig");
const logs = @import("log_analyzer.zig");
const errors = @import("error_analyzer.zig");
const expr = @import("expression_analyzer.zig");
// Unknown-identifier checking: safe by default using safeFindUpOpt() and scope guards.
const ENABLE_UNKNOWN_WALKER: bool = true;
const locals = @import("locals_binder.zig");
const ManagedArrayList = std.array_list.Managed;

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
    defer cr.diagnostics.deinit(allocator);
    // convert simple span list into diagnostics for now
    var diags = std.ArrayListUnmanaged(Diagnostic){};
    defer diags.deinit(allocator);
    for (cr.diagnostics.items) |sp| {
        try diags.append(allocator, .{ .message = "Redeclaration in scope", .span = sp });
    }
    // build function scopes for top-level functions and contract members
    for (nodes) |*n| switch (n.*) {
        .Contract => |*c| try contract.collectContractSymbols(&cr.table, &cr.table.root, c),
        .Function => |*f| try func.collectFunctionSymbols(&cr.table, &cr.table.root, f),
        else => {},
    };
    // bind locals and nested block scopes for all functions (top-level and contract members)
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
    return .{ .symbols = cr.table, .diagnostics = try diags.toOwnedSlice(allocator) };
}

// Helper function to check a single function (top-level or contract member)
fn checkFunction(
    allocator: std.mem.Allocator,
    symbols: *state.SymbolTable,
    f: *const ast.FunctionNode,
    diags: *ManagedArrayList(Diagnostic),
) !void {
    const scope = symbols.function_scopes.get(f.name) orelse return;

    // check return statements
    const spans = try stmt.checkFunctionBody(allocator, symbols, scope, f);
    defer allocator.free(spans);
    for (spans) |sp| try diags.append(.{ .message = "Return type mismatch", .span = sp });

    // unknown identifiers in function body
    if (ENABLE_UNKNOWN_WALKER and f.body.statements.len > 0) {
        const unknowns = try stmt.collectUnknownIdentifierSpans(allocator, symbols, scope, f);
        defer allocator.free(unknowns);
        for (unknowns) |usp| try diags.append(.{ .message = "Unknown identifier", .span = usp });
    }

    // log statement checks (signature and argument typing)
    const log_spans = try logs.checkLogsInFunction(allocator, symbols, scope, f);
    defer allocator.free(log_spans);
    for (log_spans) |lsp| try diags.append(.{ .message = "Invalid log call", .span = lsp });

    // error validation
    const error_result = try errors.validateErrors(allocator, symbols, &[_]ast.AstNode{.{ .Function = f.* }});
    defer error_result.diagnostics.deinit();
    for (error_result.diagnostics.items) |esp| try diags.append(.{ .message = "Invalid error usage", .span = esp });

    // spec usage checks: quantified allowed only in requires/ensures/invariant; old only in ensures
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

// Phase 2 scaffolding: type-check and validation will be added here, orchestrating analyzers.
pub fn analyzePhase2(allocator: std.mem.Allocator, nodes: []const ast.AstNode, symbols: *state.SymbolTable) ![]Diagnostic {
    var diags = ManagedArrayList(Diagnostic).init(allocator);

    // check all functions (top-level and contract members)
    for (nodes) |*n| switch (n.*) {
        .Function => |*f| {
            try checkFunction(allocator, symbols, f, &diags);
        },
        .Contract => |*c| {
            for (c.body) |*m| switch (m.*) {
                .Function => |*f| {
                    try checkFunction(allocator, symbols, f, &diags);
                },
                else => {},
            };
        },
        else => {},
    };

    // global error validation (validate errors across entire AST)
    const global_error_result = try errors.validateErrors(allocator, symbols, nodes);
    defer global_error_result.diagnostics.deinit();
    for (global_error_result.diagnostics.items) |esp| {
        try diags.append(.{ .message = "Invalid error usage", .span = esp });
    }

    return try diags.toOwnedSlice();
}
