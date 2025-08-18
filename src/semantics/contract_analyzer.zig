const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const fun = @import("function_analyzer.zig");

pub fn collectContractSymbols(table: *state.SymbolTable, root: *state.Scope, c: *const ast.ContractNode) !void {
    const contract_scope = try table.allocator.create(state.Scope);
    contract_scope.* = state.Scope.init(table.allocator, root, c.name);
    try table.scopes.append(contract_scope);
    try table.contract_scopes.put(c.name, contract_scope);
    for (c.body) |*member| switch (member.*) {
        .Function => |*f| try fun.collectFunctionSymbols(table, contract_scope, f),
        .VariableDecl => |v| {
            const sym = state.Symbol{ .name = v.name, .kind = .Var, .typ = v.type_info, .span = v.span, .mutable = (v.kind == .Var), .region = v.region };
            _ = try table.declare(contract_scope, sym);
        },
        .StructDecl => |s| {
            const sym = state.Symbol{ .name = s.name, .kind = .Struct, .span = s.span };
            _ = try table.declare(contract_scope, sym);
        },
        .EnumDecl => |e| {
            const sym = state.Symbol{ .name = e.name, .kind = .Enum, .span = e.span };
            _ = try table.declare(contract_scope, sym);
        },
        .LogDecl => |l| {
            const sym = state.Symbol{ .name = l.name, .kind = .Log, .span = l.span };
            _ = try table.declare(contract_scope, sym);
            // Record log signature for semantic checks
            try table.log_signatures.put(l.name, l.fields);
        },
        .ErrorDecl => |err| {
            const sym = state.Symbol{ .name = err.name, .kind = .Error, .span = err.span };
            _ = try table.declare(contract_scope, sym);
        },
        else => {},
    };
}
