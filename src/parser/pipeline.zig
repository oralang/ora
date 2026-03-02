// ============================================================================
// Parser Pipeline — full parse + semantics + type resolution
// ============================================================================
//
// Contains the convenience wrappers that chain raw parsing with semantic
// analysis and type resolution. Separated from parser_core.zig so that
// the pure-parse path (parseRaw) has no compile-time dependency on
// semantics or the type resolver.
//
// ============================================================================

const std = @import("std");
const builtin = @import("builtin");
const ast = @import("ora_ast");
const ast_arena = @import("ora_types").ast_arena;
const lexer = @import("ora_lexer");
const log = @import("log");
const semantics_core = @import("../semantics/core.zig");

const Token = lexer.Token;
const AstNode = ast.AstNode;
const Allocator = std.mem.Allocator;

const parser_core = @import("parser_core.zig");
const Parser = parser_core.Parser;
const ParserError = parser_core.ParserError;
const ParseResult = parser_core.ParseResult;

fn analyzePhase1ForParser(allocator: Allocator, nodes: []const AstNode) ParserError!semantics_core.SemanticsResult {
    return semantics_core.analyzePhase1(allocator, nodes) catch |err| {
        if (!builtin.is_test) {
            log.err("Semantic analysis phase 1 failed: {s}\n", .{@errorName(err)});
            if (err == error.MissingParameterType) {
                log.help("all function parameters must have an explicit or resolved type before lowering\n", .{});
            }
        }
        return switch (err) {
            error.OutOfMemory => ParserError.OutOfMemory,
            else => ParserError.TypeResolutionFailed,
        };
    };
}

fn ensureLogSignatures(symbols: *@import("../semantics/state.zig").SymbolTable, nodes: []const AstNode) !void {
    for (nodes) |node| switch (node) {
        .LogDecl => |l| {
            if (symbols.log_signatures.get(l.name) == null) {
                try symbols.log_signatures.put(l.name, l.fields);
            }
        },
        .Contract => |c| {
            if (symbols.contract_log_signatures.getPtr(c.name) == null) {
                const log_map = std.StringHashMap([]const ast.LogField).init(symbols.allocator);
                try symbols.contract_log_signatures.put(c.name, log_map);
            }
            for (c.body) |member| switch (member) {
                .LogDecl => |l| {
                    if (symbols.contract_log_signatures.getPtr(c.name)) |log_map| {
                        if (log_map.get(l.name) == null) {
                            try log_map.put(l.name, l.fields);
                        }
                    }
                },
                else => {},
            };
        },
        else => {},
    };
}

fn runTypeResolution(allocator: Allocator, nodes: []AstNode, arena_alloc: std.mem.Allocator, symbols: *@import("../semantics/state.zig").SymbolTable) ParserError!void {
    const TypeResolver = @import("../ast/type_resolver/mod.zig").TypeResolver;
    const TypeResolutionError = @import("../ast/type_resolver/mod.zig").TypeResolutionError;
    var type_resolver = TypeResolver.init(allocator, arena_alloc, symbols);
    errdefer type_resolver.deinit();
    type_resolver.resolveTypes(nodes) catch |err| {
        if (!builtin.is_test) {
            const is_user_facing_type_error = err == TypeResolutionError.ErrorUnionOutsideTry or
                err == TypeResolutionError.GenericContractNotSupported or
                err == TypeResolutionError.TopLevelGenericInstantiationNotSupported or
                err == TypeResolutionError.ComptimeArithmeticError or
                err == TypeResolutionError.ComptimeEvaluationError;
            log.err("Type resolution failed: {s}\n", .{@errorName(err)});
            if (err == TypeResolutionError.ErrorUnionOutsideTry) {
                log.help("use `try` to unwrap error unions or wrap the code in a try/catch block\n", .{});
            } else if (err == TypeResolutionError.GenericContractNotSupported) {
                log.help("generic contracts are parsed but not implemented yet; remove type parameters for now\n", .{});
            } else if (err == TypeResolutionError.TopLevelGenericInstantiationNotSupported) {
                log.help("generic functions/structs currently require a contract scope; move the generic usage inside a contract\n", .{});
            } else if (err == TypeResolutionError.ComptimeArithmeticError) {
                log.help("checked arithmetic in a compile-time-known expression failed; use wrapping operators (e.g. **%) or smaller values\n", .{});
            } else if (err == TypeResolutionError.ComptimeEvaluationError) {
                log.help("explicit/known comptime evaluation failed and cannot fall back to runtime\n", .{});
            }
            if (!is_user_facing_type_error) {
                const trace = @errorReturnTrace();
                if (trace) |t| std.debug.dumpStackTrace(t.*);
            }
        }
        return ParserError.TypeResolutionFailed;
    };
    type_resolver.deinit();
}

/// Parse tokens into a fully typed AST (parse + semantics phase 1 + type resolution).
/// Returns both the AST nodes and the arena (caller must keep arena alive while using nodes).
pub fn parseWithArena(allocator: Allocator, tokens: []const Token) ParserError!ParseResult {
    var result = try parser_core.parseRaw(allocator, tokens);
    errdefer result.arena.deinit();

    var semantics_result = try analyzePhase1ForParser(allocator, result.nodes);
    defer allocator.free(semantics_result.diagnostics);
    defer semantics_result.symbols.deinit();
    ensureLogSignatures(&semantics_result.symbols, result.nodes) catch |err| {
        if (!builtin.is_test) {
            log.err("Failed to collect log signatures: {s}\n", .{@errorName(err)});
        }
        return ParserError.TypeResolutionFailed;
    };

    try runTypeResolution(allocator, result.nodes, result.arena.allocator(), &semantics_result.symbols);
    return result;
}

/// Parse tokens into a fully typed AST (parse + semantics phase 1 + type resolution).
/// WARNING: For --emit-ast, use parseWithArena instead to keep arena alive.
pub fn parse(allocator: Allocator, tokens: []const Token) ParserError![]AstNode {
    var result = try parser_core.parseRaw(allocator, tokens);
    defer result.arena.deinit();

    var semantics_result = try analyzePhase1ForParser(allocator, result.nodes);
    defer allocator.free(semantics_result.diagnostics);
    defer semantics_result.symbols.deinit();
    ensureLogSignatures(&semantics_result.symbols, result.nodes) catch |err| {
        if (!builtin.is_test) {
            log.err("Failed to collect log signatures: {s}\n", .{@errorName(err)});
        }
        return ParserError.TypeResolutionFailed;
    };

    try runTypeResolution(allocator, result.nodes, result.arena.allocator(), &semantics_result.symbols);
    return result.nodes;
}
