const std = @import("std");
const ast = @import("../ast.zig");

pub const SlotKeyError = error{OutOfMemory};

/// Build a canonical slot-path key from an AST expression.
/// Examples:
/// - balances
/// - balances[to]
/// - balances[user.address]
/// - state.flags.enabled
///
/// Returns null when the expression is not a path-shaped expression.
pub fn buildPathSlotKey(allocator: std.mem.Allocator, expr: *const ast.Expressions.ExprNode) SlotKeyError!?[]const u8 {
    var out = std.ArrayList(u8){};
    defer out.deinit(allocator);

    if (!(try appendPathExpr(allocator, &out, expr))) return null;
    const owned = try out.toOwnedSlice(allocator);
    return @as(?[]const u8, owned);
}

/// Runtime lock-key lowering currently supports only:
/// - identifier
/// - identifier[index]
///
/// This keeps runtime slot hashing semantics precise for supported cases.
pub fn runtimeLockPathSupported(expr: *const ast.Expressions.ExprNode) bool {
    return switch (expr.*) {
        .Identifier => true,
        .Index => |ix| ix.target.* == .Identifier,
        else => false,
    };
}

fn appendPathExpr(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    expr: *const ast.Expressions.ExprNode,
) SlotKeyError!bool {
    return switch (expr.*) {
        .Identifier => |id| blk: {
            try out.appendSlice(allocator, id.name);
            break :blk true;
        },
        .FieldAccess => |fa| blk: {
            if (!(try appendPathExpr(allocator, out, fa.target))) break :blk false;
            try out.append(allocator, '.');
            try out.appendSlice(allocator, fa.field);
            break :blk true;
        },
        .Index => |ix| blk: {
            if (!(try appendPathExpr(allocator, out, ix.target))) break :blk false;
            try out.append(allocator, '[');
            if (!(try appendExprFragment(allocator, out, ix.index))) break :blk false;
            try out.append(allocator, ']');
            break :blk true;
        },
        else => false,
    };
}

fn appendExprFragment(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    expr: *const ast.Expressions.ExprNode,
) SlotKeyError!bool {
    return switch (expr.*) {
        .Identifier, .FieldAccess, .Index => try appendPathExpr(allocator, out, expr),
        .Literal => |lit| blk: {
            try appendLiteralFragment(allocator, out, lit);
            break :blk true;
        },
        .Unary => |u| blk: {
            try out.appendSlice(allocator, unaryToken(u.operator));
            try out.append(allocator, '(');
            if (!(try appendExprFragment(allocator, out, u.operand))) break :blk false;
            try out.append(allocator, ')');
            break :blk true;
        },
        .Binary => |b| blk: {
            try out.append(allocator, '(');
            if (!(try appendExprFragment(allocator, out, b.lhs))) break :blk false;
            try out.appendSlice(allocator, binaryToken(b.operator));
            if (!(try appendExprFragment(allocator, out, b.rhs))) break :blk false;
            try out.append(allocator, ')');
            break :blk true;
        },
        .Call => |c| blk: {
            if (!(try appendExprFragment(allocator, out, c.callee))) break :blk false;
            try out.append(allocator, '(');
            for (c.arguments, 0..) |arg, i| {
                if (i != 0) try out.append(allocator, ',');
                if (!(try appendExprFragment(allocator, out, arg))) break :blk false;
            }
            try out.append(allocator, ')');
            break :blk true;
        },
        .EnumLiteral => |enum_lit| blk: {
            try out.appendSlice(allocator, enum_lit.enum_name);
            try out.append(allocator, '.');
            try out.appendSlice(allocator, enum_lit.variant_name);
            break :blk true;
        },
        else => false,
    };
}

fn appendLiteralFragment(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    lit: ast.Expressions.LiteralExpr,
) SlotKeyError!void {
    switch (lit) {
        .Integer => |i| try out.appendSlice(allocator, i.value),
        .String => |s| {
            try out.append(allocator, '"');
            try appendEscaped(allocator, out, s.value);
            try out.append(allocator, '"');
        },
        .Bool => |b| try out.appendSlice(allocator, if (b.value) "true" else "false"),
        .Address => |a| try out.appendSlice(allocator, a.value),
        .Hex => |h| try out.appendSlice(allocator, h.value),
        .Binary => |b| try out.appendSlice(allocator, b.value),
        .Character => |ch| {
            var writer = out.writer(allocator);
            try writer.print("char({d})", .{ch.value});
        },
        .Bytes => |bytes| {
            try out.appendSlice(allocator, "bytes(\"");
            try appendEscaped(allocator, out, bytes.value);
            try out.appendSlice(allocator, "\")");
        },
    }
}

fn appendEscaped(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    s: []const u8,
) SlotKeyError!void {
    for (s) |ch| {
        switch (ch) {
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '"' => try out.appendSlice(allocator, "\\\""),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, ch),
        }
    }
}

fn unaryToken(op: ast.Expressions.UnaryOp) []const u8 {
    return switch (op) {
        .Minus => "-",
        .Bang => "!",
        .BitNot => "~",
    };
}

fn binaryToken(op: ast.Expressions.BinaryOp) []const u8 {
    return switch (op) {
        .Plus => "+",
        .Minus => "-",
        .Star => "*",
        .Slash => "/",
        .Percent => "%",
        .StarStar => "**",
        .WrappingAdd => "+%",
        .WrappingSub => "-%",
        .WrappingMul => "*%",
        .WrappingPow => "**%",
        .WrappingShl => "<<%",
        .WrappingShr => ">>%",
        .EqualEqual => "==",
        .BangEqual => "!=",
        .Less => "<",
        .LessEqual => "<=",
        .Greater => ">",
        .GreaterEqual => ">=",
        .And => "&&",
        .Or => "||",
        .BitwiseAnd => "&",
        .BitwiseOr => "|",
        .BitwiseXor => "^",
        .LeftShift => "<<",
        .RightShift => ">>",
        .Comma => ",",
    };
}
