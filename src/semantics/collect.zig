// ============================================================================
// Symbol Collector - Phase 1 of Semantic Analysis
// ============================================================================
//
// Collects top-level declarations and populates the root scope.
//
// RESPONSIBILITIES:
//   • Collect top-level declarations (contracts, functions, structs, enums, etc.)
//   • Detect duplicate declarations at the global scope
//   • Record enum variants for exhaustiveness checking
//
// NOTE: Contract and function members are collected by contract_analyzer.zig
//       and function_analyzer.zig in later phases.
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");

pub const CollectResult = struct {
    table: state.SymbolTable,
    diagnostics: std.ArrayList(ast.SourceSpan), // spans of redeclarations (message can be added later)
};

pub fn collectSymbols(allocator: std.mem.Allocator, nodes: []const ast.AstNode) !CollectResult {
    var table = state.SymbolTable.init(allocator);
    var diags = std.ArrayList(ast.SourceSpan){};

    // Top-level declarations
    for (nodes) |node| switch (node) {
        .Contract => |c| {
            const sym = state.Symbol{ .name = c.name, .kind = .Contract, .span = c.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, c.span);
            // Member scopes and symbols are handled in contract_analyzer
        },
        .Function => |f| {
            const sym = state.Symbol{ .name = f.name, .kind = .Function, .span = f.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, f.span);
        },
        .VariableDecl => |v| {
            const sym = state.Symbol{
                .name = v.name,
                .kind = .Var,
                .typ = v.type_info,
                .span = v.span,
                .mutable = (v.kind == .Var),
                .region = v.region,
            };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, v.span);
        },
        .StructDecl => |s| {
            const sym = state.Symbol{ .name = s.name, .kind = .Struct, .span = s.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, s.span);
            // Store struct fields for type resolution
            try table.struct_fields.put(s.name, s.fields);
        },
        .EnumDecl => |e| {
            const sym = state.Symbol{ .name = e.name, .kind = .Enum, .span = e.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, e.span);
            // Record enum variants for coverage checks (direct slice copy)
            const slice = try allocator.alloc([]const u8, e.variants.len);
            for (e.variants, 0..) |v, i| slice[i] = v.name;
            try table.enum_variants.put(e.name, slice);
        },
        .LogDecl => |l| {
            const sym = state.Symbol{ .name = l.name, .kind = .Log, .span = l.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, l.span);
        },
        .ErrorDecl => |err| {
            const sym = state.Symbol{ .name = err.name, .kind = .Error, .span = err.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, err.span);
            // Store error signature (parameters) for validation
            try table.error_signatures.put(err.name, err.parameters);
        },
        .Import => |im| {
            const name = im.alias orelse im.path; // simplistic; refine later
            const sym = state.Symbol{ .name = name, .kind = .Module, .span = im.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(allocator, im.span);
        },
        else => {},
    };

    return .{ .table = table, .diagnostics = diags };
}
