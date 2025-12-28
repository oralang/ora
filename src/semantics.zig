// ============================================================================
// Semantics - Unified Entry Point
// ============================================================================
//
// Provides a clean facade for semantic analysis, orchestrating the two-phase
// analysis pipeline (symbol collection → validation).
//
// PUBLIC API:
//   • analyze() - Full two-phase semantic analysis
//   • validateLValues() - Assignment target validation
//
// Delegates to: semantics/core.zig, semantics/state.zig
//
// ============================================================================

const std = @import("std");
const ast = @import("ast.zig");
const core = @import("semantics/core.zig");
const ManagedArrayList = std.array_list.Managed;

// Export builtins module for use by MLIR lowering
pub const builtins = @import("semantics/builtins.zig");
// Export state module for type resolution
pub const state = @import("semantics/state.zig");

pub const Diagnostic = struct {
    message: []const u8,
    span: ast.SourceSpan,
};

pub const AnalysisResult = struct {
    diagnostics: []Diagnostic,
};

fn isLValue(expr: ast.Expressions.ExprNode) bool {
    return switch (expr) {
        .Identifier => true,
        .FieldAccess => true,
        .IndexAccess => true,
        // optionally allow pointer deref or storage paths if modeled
        else => false,
    };
}

pub fn validateLValues(allocator: std.mem.Allocator, nodes: []const ast.AstNode) !AnalysisResult {
    var diags = ManagedArrayList(Diagnostic).init(allocator);
    defer diags.deinit();

    for (nodes) |node| {
        switch (node) {
            .Function => |f| {
                for (f.body.statements) |stmt| {
                    switch (stmt) {
                        .Assignment => |a| {
                            if (!isLValue(a.target)) {
                                try diags.append(.{ .message = "Left-hand side of assignment is not assignable (lvalue)", .span = a.span });
                            }
                        },
                        .CompoundAssignment => |c| {
                            if (!isLValue(c.target)) {
                                try diags.append(.{ .message = "Left-hand side of compound assignment is not assignable (lvalue)", .span = c.span });
                            }
                        },
                        else => {},
                    }
                }
            },
            else => {},
        }
    }

    return AnalysisResult{ .diagnostics = try diags.toOwnedSlice() };
}

// Unified entry point for semantics
// Runs collection (phase 1) then checks (phase 2) and returns combined diagnostics with symbols
pub fn analyze(allocator: std.mem.Allocator, nodes: []const ast.AstNode) !core.SemanticsResult {
    var p1 = try core.analyzePhase1(allocator, nodes);
    // phase2 uses the same symbol table by reference
    const p2_diags = try core.analyzePhase2(allocator, nodes, &p1.symbols);

    // combine diagnostics deterministically in encounter order (phase1 then phase2)
    var combined = ManagedArrayList(core.Diagnostic).init(allocator);
    defer combined.deinit();

    for (p1.diagnostics) |d| try combined.append(d);
    allocator.free(p1.diagnostics);
    for (p2_diags) |d2| try combined.append(d2);
    allocator.free(p2_diags);

    return .{ .symbols = p1.symbols, .diagnostics = try combined.toOwnedSlice() };
}
