const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const line_index_api = @import("line_index.zig");
const semantic_index = @import("semantic_index.zig");
const text_edits = @import("text_edits.zig");

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

pub fn hintsInRangeCached(
    allocator: Allocator,
    source: []const u8,
    range: frontend.Range,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    file: *const compiler.ast.AstFile,
    index: *const semantic_index.SemanticIndex,
) ![]InlayHint {
    var hints = std.ArrayList(InlayHint).empty;
    errdefer {
        for (hints.items) |*h| h.deinit(allocator);
        hints.deinit(allocator);
    }

    const position_resolver = PositionResolver{ .line_index = .{
        .source = source,
        .line_index = line_index,
        .encoding = encoding,
    } };
    for (file.root_items) |item_id| {
        try collectHintsFromItem(&hints, allocator, file, item_id, &position_resolver, range, index);
    }

    return hints.toOwnedSlice(allocator);
}

pub fn deinitHints(allocator: Allocator, items: []InlayHint) void {
    for (items) |*h| h.deinit(allocator);
    allocator.free(items);
}

const HintError = Allocator.Error;

const PositionResolver = union(enum) {
    line_index: struct {
        source: []const u8,
        line_index: *const line_index_api.LineIndex,
        encoding: text_edits.PositionEncoding,
    },

    fn offsetToPosition(self: *const PositionResolver, offset: u32) frontend.Position {
        return switch (self.*) {
            .line_index => |index| index.line_index.offsetToPosition(index.source, offset, index.encoding),
        };
    }
};

fn collectHintsFromItem(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    positions: *const PositionResolver,
    range: frontend.Range,
    maybe_index: ?*const semantic_index.SemanticIndex,
) HintError!void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |contract| {
            for (contract.members) |member_id| {
                try collectHintsFromItem(hints, allocator, file, member_id, positions, range, maybe_index);
            }
        },
        .Function => |function| {
            try collectHintsFromBody(hints, allocator, file, function.body, positions, range, maybe_index);
        },
        else => {},
    }
}

fn collectHintsFromBody(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    body_id: compiler.ast.BodyId,
    positions: *const PositionResolver,
    range: frontend.Range,
    maybe_index: ?*const semantic_index.SemanticIndex,
) HintError!void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        try collectHintsFromStmt(hints, allocator, file, stmt_id, positions, range, maybe_index);
    }
}

fn collectHintsFromStmt(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    stmt_id: compiler.ast.StmtId,
    positions: *const PositionResolver,
    range: frontend.Range,
    maybe_index: ?*const semantic_index.SemanticIndex,
) HintError!void {
    const stmt = file.statement(stmt_id).*;
    switch (stmt) {
        .Expr => |expr_stmt| {
            try collectHintsFromExpr(hints, allocator, file, expr_stmt.expr, positions, range, maybe_index);
        },
        .VariableDecl => |decl| {
            if (decl.value) |value| {
                try collectHintsFromExpr(hints, allocator, file, value, positions, range, maybe_index);
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
                            const pos = positions.offsetToPosition(nr.end);
                            if (positionInRange(pos, range)) {
                                const label = try std.fmt.allocPrint(allocator, ": {s}", .{type_str});
                                var label_owned = true;
                                errdefer if (label_owned) allocator.free(label);

                                try hints.append(allocator, .{
                                    .position = pos,
                                    .label = label,
                                    .kind = .type_hint,
                                    .padding_left = false,
                                    .padding_right = false,
                                });
                                label_owned = false;
                            }
                        }
                    }
                }
            }
        },
        .Assign => |assign| {
            try collectHintsFromExpr(hints, allocator, file, assign.value, positions, range, maybe_index);
        },
        .Return => |ret| {
            if (ret.value) |value| {
                try collectHintsFromExpr(hints, allocator, file, value, positions, range, maybe_index);
            }
        },
        .If => |if_stmt| {
            try collectHintsFromExpr(hints, allocator, file, if_stmt.condition, positions, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, if_stmt.then_body, positions, range, maybe_index);
            if (if_stmt.else_body) |else_body| {
                try collectHintsFromBody(hints, allocator, file, else_body, positions, range, maybe_index);
            }
        },
        .While => |while_stmt| {
            try collectHintsFromExpr(hints, allocator, file, while_stmt.condition, positions, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, while_stmt.body, positions, range, maybe_index);
        },
        .For => |for_stmt| {
            try collectHintsFromExpr(hints, allocator, file, for_stmt.iterable, positions, range, maybe_index);
            try collectHintsFromBody(hints, allocator, file, for_stmt.body, positions, range, maybe_index);
        },
        .Block => |block_stmt| {
            try collectHintsFromBody(hints, allocator, file, block_stmt.body, positions, range, maybe_index);
        },
        else => {},
    }
}

fn collectHintsFromExpr(
    hints: *std.ArrayList(InlayHint),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    expr_id: compiler.ast.ExprId,
    positions: *const PositionResolver,
    range: frontend.Range,
    maybe_index: ?*const semantic_index.SemanticIndex,
) HintError!void {
    const expr = file.expression(expr_id).*;
    switch (expr) {
        .Call => |call| {
            try collectHintsFromExpr(hints, allocator, file, call.callee, positions, range, maybe_index);
            for (call.args) |arg| {
                try collectHintsFromExpr(hints, allocator, file, arg, positions, range, maybe_index);
            }

            const callee_name = resolveCalleeName(file, call.callee) orelse return;
            const idx = maybe_index orelse return;
            const func_symbol = findFunctionSymbol(idx.symbols, callee_name) orelse return;
            const func_idx = findSymbolIndex(idx.symbols, func_symbol);

            for (call.args, 0..) |arg, i| {
                const param_name = parameterNameAt(idx.symbols, func_idx, i) orelse continue;

                const arg_expr = file.expression(arg).*;
                if (arg_expr == .Name) {
                    if (std.mem.eql(u8, arg_expr.Name.name, param_name)) continue;
                }

                const arg_range = getExprRange(file, arg);
                const arg_pos = positions.offsetToPosition(arg_range.start);

                if (!positionInRange(arg_pos, range)) continue;

                const label = try std.fmt.allocPrint(allocator, "{s}:", .{param_name});
                var label_owned = true;
                errdefer if (label_owned) allocator.free(label);

                try hints.append(allocator, .{
                    .position = arg_pos,
                    .label = label,
                    .kind = .parameter_hint,
                    .padding_left = false,
                    .padding_right = true,
                });
                label_owned = false;
            }
        },
        .Binary => |bin| {
            try collectHintsFromExpr(hints, allocator, file, bin.lhs, positions, range, maybe_index);
            try collectHintsFromExpr(hints, allocator, file, bin.rhs, positions, range, maybe_index);
        },
        .Unary => |un| {
            try collectHintsFromExpr(hints, allocator, file, un.operand, positions, range, maybe_index);
        },
        .Field => |field| {
            try collectHintsFromExpr(hints, allocator, file, field.base, positions, range, maybe_index);
        },
        .Index => |index| {
            try collectHintsFromExpr(hints, allocator, file, index.base, positions, range, maybe_index);
            try collectHintsFromExpr(hints, allocator, file, index.index, positions, range, maybe_index);
        },
        .Group => |group| {
            try collectHintsFromExpr(hints, allocator, file, group.expr, positions, range, maybe_index);
        },
        .Old => |old| {
            try collectHintsFromExpr(hints, allocator, file, old.expr, positions, range, maybe_index);
        },
        .Tuple => |tuple| {
            for (tuple.elements) |elem| {
                try collectHintsFromExpr(hints, allocator, file, elem, positions, range, maybe_index);
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

fn parameterNameAt(symbols: []const semantic_index.Symbol, parent_idx: usize, parameter_index: usize) ?[]const u8 {
    var seen: usize = 0;
    for (symbols) |symbol| {
        if (symbol.parent == null or symbol.parent.? != parent_idx or symbol.kind != .parameter) continue;
        if (seen == parameter_index) return symbol.name;
        seen += 1;
    }
    return null;
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

fn inferExprType(
    file: *const compiler.ast.AstFile,
    expr_id: compiler.ast.ExprId,
    maybe_index: ?*const semantic_index.SemanticIndex,
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
