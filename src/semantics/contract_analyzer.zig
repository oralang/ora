// ============================================================================
// Contract Analyzer
// ============================================================================
//
// Collects symbols from contract members and creates contract scopes.
//
// RESPONSIBILITIES:
//   • Create contract scope
//   • Collect member declarations
//   • Delegate function collection to function_analyzer
//   • Record log signatures
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const fun = @import("function_analyzer.zig");

pub fn collectContractSymbols(table: *state.SymbolTable, root: *state.Scope, c: *const ast.ContractNode) !void {
    const contract_scope = try table.allocator.create(state.Scope);
    contract_scope.* = state.Scope.init(table.allocator, root, c.name);
    try table.scopes.append(table.allocator, contract_scope);
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
            // Store struct fields for type resolution
            try table.struct_fields.put(s.name, s.fields);
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
            // Store error signature (parameters) for validation
            try table.error_signatures.put(err.name, err.parameters);
        },
        .Constant => |const_decl| {
            // Constants are collected but type resolution happens later
            // Store with type if available, otherwise it will be resolved during type resolution
            const sym = state.Symbol{
                .name = const_decl.name,
                .kind = .Var, // Constants are treated as variables for symbol lookup
                .typ = if (const_decl.typ.ora_type != null) const_decl.typ else null,
                .span = const_decl.span,
                .mutable = false, // Constants are immutable
                .region = null, // Constants don't have a memory region
            };
            _ = try table.declare(contract_scope, sym);
        },
        else => {},
    };
}
