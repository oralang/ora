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
//     Check bodies → Validate returns → Validate specs
//
// Delegates to: collect.zig, contract_analyzer.zig, function_analyzer.zig,
//               locals_binder.zig, statement_analyzer.zig
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
        .Contract => |*c| try contract.collectContractSymbols(&cr.table, cr.table.root, c),
        .Function => |*f| try func.collectFunctionSymbols(&cr.table, cr.table.root, f),
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

    // unknown identifiers in function body
    if (ENABLE_UNKNOWN_WALKER and f.body.statements.len > 0) {
        const unknowns = try stmt.collectUnknownIdentifierSpans(allocator, symbols, scope, f);
        defer allocator.free(unknowns);
        for (unknowns) |usp| try diags.append(.{ .message = "Unknown identifier", .span = usp });
    }

    // spec usage checks are handled in the type resolver
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


    return try diags.toOwnedSlice();
}
