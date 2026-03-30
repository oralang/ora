const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const HintKind = enum {
    type_hint,
    parameter_hint,
};

pub const InlayHint = struct {
    position: frontend.Position,
    label: []u8,
    kind: HintKind,
    padding_left: bool,
    padding_right: bool,

    pub fn deinit(self: *InlayHint, allocator: Allocator) void {
        allocator.free(self.label);
    }
};

/// Find inlay hints in the given range of source.
pub fn hintsInRange(
    allocator: Allocator,
    source: []const u8,
    range: frontend.Range,
) ![]InlayHint {
    var hints = std.ArrayList(InlayHint){};
    errdefer {
        for (hints.items) |*h| h.deinit(allocator);
        hints.deinit(allocator);
    }

    var sources = compiler.source.SourceStore.init(allocator);
    defer sources.deinit();
    const file_id = try sources.addFile("<lsp>", source);

    var parse_result = compiler.syntax.parse(allocator, file_id, source) catch return try hints.toOwnedSlice(allocator);
    defer parse_result.deinit();

    var lower_result = compiler.ast.lower(allocator, &parse_result.tree) catch return try hints.toOwnedSlice(allocator);
    defer lower_result.deinit();

    // Build semantic index for function parameter lookup.
    var index = semantic_index.indexDocument(allocator, source) catch null;
    defer if (index) |*idx| idx.deinit(allocator);

    // Walk all items to find function bodies with call expressions.
    for (lower_result.file.root_items) |item_id| {
        try collectHintsFromItem(&hints, allocator, &lower_result.file, item_id, &sources, file_id, range, index);
    }

    return hints.toOwnedSlice(allocator);
}

pub fn deinitHints(allocator: Allocator, items: []InlayHint) void {
    for (items) |*h| h.deinit(allocator);
    allocator.free(items);
}

const HintError = Allocator.Error;

fn collectHintsFromItem(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: frontend.Range,
    maybe_index: ?semantic_index.SemanticIndex,
) HintError!void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |contract| {
            for (contract.members) |member_id| {
                try collectHintsFromItem(hints, allocator, file, member_id, sources, file_id, range, maybe_index);
            }
        },
        .Function => |function| {
            try collectHintsFromBody(hints, allocator, file, function.body, sources, file_id, range, maybe_index);
        },
        else => {},
    }
}

fn collectHintsFromBody(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    body_id: compiler.ast.BodyId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: frontend.Range,
    maybe_index: ?semantic_index.SemanticIndex,
) HintError!void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        try collectHintsFromStmt(hints, allocator, file, stmt_id, sources, file_id, range, maybe_index);
    }
}

fn collectHintsFromStmt(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    stmt_id: compiler.ast.StmtId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: frontend.Range,
    maybe_index: ?semantic_index.SemanticIndex,
) HintError!void {
    const stmt = file.statement(stmt_id).*;
    switch (stmt) {
        .Expr => |expr_stmt| {
            try collectHintsFromExpr(hints, allocator, file, expr_stmt.expr, sources, file_id, range, maybe_index);
        },
        .VariableDecl => |decl| {
            if (decl.value) |value| {
                try collectHintsFromExpr(hints, allocator, file, value, sources, file_id, range, maybe_index);
            }
            if (decl.type_expr == null) {
                if (decl.value) |value| {
                    if (inferExprType(file, value, maybe_index)) |type_str| {
                        const pat = file.pattern(decl.pattern).*;
                        const name_range = switch (pat) {
                            .Name => |n| n.range,
                            else => null,
                        };
                        if (name_range) |nr| {
                            const pos = textRangeToPosition(sources, file_id, nr.end);
                            if (positionInRange(pos, range)) {
                                try hints.append(allocator, .{
                                    .position = pos,
                                    .label = try std.fmt.allocPrint(allocator, ": {s}", .{type_str}),
                                    .kind = .type_hint,
                                    .padding_left = false,
                                    .padding_right = false,
                                });
                            }
                        }
                    }
                }
            }
        },
        .Assign => |assign| {
            try collectHintsFromExpr(hints, allocator, file, assign.value, sources, file_id, range, maybe_index);
        },
        .Return => |ret| {
            if (ret.value) |value| {
                try collectHintsFromExpr(hints, allocator, file, value, sources, file_id, range, maybe_index);
            }
        },
        .If => |if_stmt| {
            try collectHintsFromExpr(hints, allocator, file, if_stmt.condition, sources, file_id, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, if_stmt.then_body, sources, file_id, range, maybe_index);
            if (if_stmt.else_body) |else_body| {
                try collectHintsFromBody(hints, allocator, file, else_body, sources, file_id, range, maybe_index);
            }
        },
        .While => |while_stmt| {
            try collectHintsFromExpr(hints, allocator, file, while_stmt.condition, sources, file_id, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, while_stmt.body, sources, file_id, range, maybe_index);
        },
        .For => |for_stmt| {
            try collectHintsFromExpr(hints, allocator, file, for_stmt.iterable, sources, file_id, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, for_stmt.body, sources, file_id, range, maybe_index);
        },
        .Block => |block_stmt| {
            try collectHintsFromBody(hints, allocator, file, block_stmt.body, sources, file_id, range, maybe_index);
        },
        else => {},
    }
}

fn collectHintsFromExpr(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    expr_id: compiler.ast.ExprId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: frontend.Range,
    maybe_index: ?semantic_index.SemanticIndex,
) HintError!void {
    const expr = file.expression(expr_id).*;
    switch (expr) {
        .Call => |call| {
            // Recurse into callee and args first.
            try collectHintsFromExpr(hints, allocator, file, call.callee, sources, file_id, range, maybe_index);
            for (call.args) |arg| {
                try collectHintsFromExpr(hints, allocator, file, arg, sources, file_id, range, maybe_index);
            }

            // Now try to add parameter name hints.
            const callee_name = resolveCalleeName(file, call.callee) orelse return;
            const idx = maybe_index orelse return;
            const func_symbol = findFunctionSymbol(idx.symbols, callee_name) orelse return;

            // Find the function's parameter symbols (children of the function symbol).
            const func_idx = findSymbolIndex(idx.symbols, func_symbol);
            const param_names = collectParameterNames(allocator, idx.symbols, func_idx) catch return;
            defer allocator.free(param_names);

            for (call.args, 0..) |arg, i| {
                if (i >= param_names.len) break;
                const param_name = param_names[i] orelse continue;

                // Skip if the argument is a simple name that matches the parameter.
                const arg_expr = file.expression(arg).*;
                if (arg_expr == .Name) {
                    if (std.mem.eql(u8, arg_expr.Name.name, param_name)) continue;
                }

                const arg_range = getExprRange(file, arg);
                const arg_pos = textRangeToPosition(sources, file_id, arg_range.start);

                // Only emit if within the requested range.
                if (!positionInRange(arg_pos, range)) continue;

                try hints.append(allocator, .{
                    .position = arg_pos,
                    .label = try std.fmt.allocPrint(allocator, "{s}:", .{param_name}),
                    .kind = .parameter_hint,
                    .padding_left = false,
                    .padding_right = true,
                });
            }
        },
        .Binary => |bin| {
            try collectHintsFromExpr(hints, allocator, file, bin.lhs, sources, file_id, range, maybe_index);
            try collectHintsFromExpr(hints, allocator, file, bin.rhs, sources, file_id, range, maybe_index);
        },
        .Unary => |un| {
            try collectHintsFromExpr(hints, allocator, file, un.operand, sources, file_id, range, maybe_index);
        },
        .Field => |field| {
            try collectHintsFromExpr(hints, allocator, file, field.base, sources, file_id, range, maybe_index);
        },
        .Index => |index| {
            try collectHintsFromExpr(hints, allocator, file, index.base, sources, file_id, range, maybe_index);
            try collectHintsFromExpr(hints, allocator, file, index.index, sources, file_id, range, maybe_index);
        },
        .Group => |group| {
            try collectHintsFromExpr(hints, allocator, file, group.expr, sources, file_id, range, maybe_index);
        },
        .Old => |old| {
            try collectHintsFromExpr(hints, allocator, file, old.expr, sources, file_id, range, maybe_index);
        },
        .Tuple => |tuple| {
            for (tuple.elements) |elem| {
                try collectHintsFromExpr(hints, allocator, file, elem, sources, file_id, range, maybe_index);
            }
        },
        else => {},
    }
}

fn resolveCalleeName(file: *const compiler.ast.AstFile, callee_id: compiler.ast.ExprId) ?[]const u8 {
    const expr = file.expression(callee_id).*;
    return switch (expr) {
        .Name => |name| name.name,
        .Field => |field| field.name,
        else => null,
    };
}

fn findFunctionSymbol(symbols: []const semantic_index.Symbol, name: []const u8) ?*const semantic_index.Symbol {
    for (symbols) |*symbol| {
        if (!std.mem.eql(u8, symbol.name, name)) continue;
        if (symbol.kind == .function or symbol.kind == .method or symbol.kind == .event or symbol.kind == .error_decl) return symbol;
    }
    return null;
}

fn findSymbolIndex(symbols: []const semantic_index.Symbol, target: *const semantic_index.Symbol) usize {
    const base = @intFromPtr(symbols.ptr);
    const target_addr = @intFromPtr(target);
    return (target_addr - base) / @sizeOf(semantic_index.Symbol);
}

fn collectParameterNames(allocator: Allocator, symbols: []const semantic_index.Symbol, parent_idx: usize) ![]?[]const u8 {
    var names = std.ArrayList(?[]const u8){};
    errdefer names.deinit(allocator);

    for (symbols) |symbol| {
        if (symbol.parent != null and symbol.parent.? == parent_idx and symbol.kind == .parameter) {
            try names.append(allocator, symbol.name);
        }
    }

    return names.toOwnedSlice(allocator);
}

fn getExprRange(file: *const compiler.ast.AstFile, expr_id: compiler.ast.ExprId) compiler.TextRange {
    const expr = file.expression(expr_id).*;
    return switch (expr) {
        .IntegerLiteral => |e| e.range,
        .StringLiteral => |e| e.range,
        .BoolLiteral => |e| e.range,
        .AddressLiteral => |e| e.range,
        .BytesLiteral => |e| e.range,
        .Name => |e| e.range,
        .Result => |e| e.range,
        .Call => |e| e.range,
        .Builtin => |e| e.range,
        .Field => |e| e.range,
        .Index => |e| e.range,
        .Binary => |e| e.range,
        .Unary => |e| e.range,
        .Group => |e| e.range,
        .Old => |e| e.range,
        .Tuple => |e| e.range,
        .ArrayLiteral => |e| e.range,
        .StructLiteral => |e| e.range,
        .Switch => |e| e.range,
        .Comptime => |e| e.range,
        .TypeValue => |e| e.range,
        .Quantified => |e| e.range,
        .ExternalProxy => |e| e.range,
        .ErrorReturn => |e| e.range,
        .Error => |e| e.range,
    };
}

fn textRangeToPosition(sources: *const compiler.source.SourceStore, file_id: compiler.FileId, offset: u32) frontend.Position {
    const lc = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = offset, .end = offset },
    });
    return .{
        .line = if (lc.line > 0) lc.line - 1 else 0,
        .character = if (lc.column > 0) lc.column - 1 else 0,
    };
}

fn inferExprType(
    file: *const compiler.ast.AstFile,
    expr_id: compiler.ast.ExprId,
    maybe_index: ?semantic_index.SemanticIndex,
) ?[]const u8 {
    const expr = file.expression(expr_id).*;
    return switch (expr) {
        .IntegerLiteral => "u256",
        .BoolLiteral => "bool",
        .AddressLiteral => "address",
        .StringLiteral => "string",
        .BytesLiteral => "bytes",
        .Call => |call| {
            const callee_name = resolveCalleeName(file, call.callee) orelse return null;
            const idx = maybe_index orelse return null;
            const func_sym = findFunctionSymbol(idx.symbols, callee_name) orelse return null;
            const detail = func_sym.detail orelse return null;
            return extractReturnType(detail);
        },
        .Tuple => "(tuple)",
        else => null,
    };
}

/// Given a detail string like `(x: u256, y: u256) -> bool`, return `bool`.
fn extractReturnType(detail: []const u8) ?[]const u8 {
    const arrow = std.mem.indexOf(u8, detail, " -> ") orelse return null;
    const ret = detail[arrow + 4 ..];
    if (std.mem.indexOfScalar(u8, ret, '\n')) |nl| return ret[0..nl];
    return ret;
}

fn positionInRange(pos: frontend.Position, range: frontend.Range) bool {
    if (pos.line < range.start.line) return false;
    if (pos.line > range.end.line) return false;
    if (pos.line == range.start.line and pos.character < range.start.character) return false;
    if (pos.line == range.end.line and pos.character > range.end.character) return false;
    return true;
}
