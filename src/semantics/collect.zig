const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");

pub const CollectResult = struct {
    table: state.SymbolTable,
    diagnostics: std.ArrayList(ast.SourceSpan), // spans of redeclarations (message can be added later)
};

pub fn collectSymbols(allocator: std.mem.Allocator, nodes: []const ast.AstNode) !CollectResult {
    var table = state.SymbolTable.init(allocator);
    var diags = std.ArrayList(ast.SourceSpan).init(allocator);

    // Top-level declarations
    for (nodes) |node| switch (node) {
        .Contract => |c| {
            const sym = state.Symbol{ .name = c.name, .kind = .Contract, .span = c.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(c.span);
            // Member scopes and symbols are handled in contract_analyzer
        },
        .Function => |f| {
            const sym = state.Symbol{ .name = f.name, .kind = .Function, .span = f.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(f.span);
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
            if (try table.declare(&table.root, sym)) |_| try diags.append(v.span);
        },
        .StructDecl => |s| {
            const sym = state.Symbol{ .name = s.name, .kind = .Struct, .span = s.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(s.span);
        },
        .EnumDecl => |e| {
            const sym = state.Symbol{ .name = e.name, .kind = .Enum, .span = e.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(e.span);
            // Record enum variants for coverage checks (direct slice copy)
            const slice = try allocator.alloc([]const u8, e.variants.len);
            for (e.variants, 0..) |v, i| slice[i] = v.name;
            try table.enum_variants.put(e.name, slice);
        },
        .LogDecl => |l| {
            const sym = state.Symbol{ .name = l.name, .kind = .Log, .span = l.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(l.span);
        },
        .ErrorDecl => |err| {
            const sym = state.Symbol{ .name = err.name, .kind = .Error, .span = err.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(err.span);
        },
        .Import => |im| {
            const name = im.alias orelse im.path; // simplistic; refine later
            const sym = state.Symbol{ .name = name, .kind = .Module, .span = im.span };
            if (try table.declare(&table.root, sym)) |_| try diags.append(im.span);
        },
        else => {},
    };

    return .{ .table = table, .diagnostics = diags };
}
