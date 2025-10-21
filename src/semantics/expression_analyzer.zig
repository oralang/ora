// ============================================================================
// Expression Type Analyzer
// ============================================================================
//
// Infers types for expressions and validates spec expression usage.
//
// TYPE INFERENCE:
//   Identifiers → lookup | Literals → direct mapping | Field access → resolve
//   Index → element type | Call → return type | Try → unwrap error union
//
// SPEC VALIDATION:
//   • Quantified expressions: requires/ensures/invariant only
//   • Old expressions: ensures only
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");

pub const ExprAnalysis = struct {
    typ: ast.Types.TypeInfo,
};

// Removed: now using table.isScopeKnown() and table.safeFindUp() instead

pub fn inferExprType(table: *state.SymbolTable, scope: *state.Scope, expr: ast.Expressions.ExprNode) ast.Types.TypeInfo {
    return switch (expr) {
        .Identifier => |id| blk: {
            if (table.isScopeKnown(scope)) {
                if (table.safeFindUp(scope, id.name)) |sym| {
                    if (sym.typ) |ti| break :blk ti;
                }
            }
            break :blk ast.Types.TypeInfo.unknown();
        },
        .Literal => |lit| switch (lit) {
            .Integer => ast.Types.TypeInfo.fromOraType(.u256),
            .String => ast.Types.TypeInfo.fromOraType(.string),
            .Bool => ast.Types.TypeInfo.fromOraType(.bool),
            .Address => ast.Types.TypeInfo.fromOraType(.address),
            .Hex => ast.Types.TypeInfo.fromOraType(.bytes),
            .Binary => ast.Types.TypeInfo.fromOraType(.bytes),
            .Character => ast.Types.TypeInfo.fromOraType(.u8),
            .Bytes => ast.Types.TypeInfo.fromOraType(.bytes),
        },
        .FieldAccess => |fa| blk_fa: {
            const target_ti = inferExprType(table, scope, fa.target.*);
            if (target_ti.ora_type) |ot| switch (ot) {
                .anonymous_struct => |fields| {
                    var i: usize = 0;
                    while (i < fields.len) : (i += 1) {
                        if (std.mem.eql(u8, fields[i].name, fa.field)) {
                            break :blk_fa ast.Types.TypeInfo.fromOraType(@constCast(fields[i].typ).*);
                        }
                    }
                    break :blk_fa ast.Types.TypeInfo.unknown();
                },
                else => break :blk_fa ast.Types.TypeInfo.unknown(),
            } else break :blk_fa ast.Types.TypeInfo.unknown();
        },
        .Index => |ix| blk_idx: {
            const target_ti = inferExprType(table, scope, ix.target.*);
            if (target_ti.ora_type) |ot| switch (ot) {
                .array => |arr| break :blk_idx ast.Types.TypeInfo.fromOraType(@constCast(arr.elem).*),
                .slice => |elem| break :blk_idx ast.Types.TypeInfo.fromOraType(@constCast(elem).*),
                .map => |m| break :blk_idx ast.Types.TypeInfo.fromOraType(@constCast(m.value).*),
                else => break :blk_idx ast.Types.TypeInfo.unknown(),
            } else break :blk_idx ast.Types.TypeInfo.unknown();
        },
        .Call => |c| blk_call: {
            // Prefer direct function symbol lookup for simple identifiers
            if (c.callee.* == .Identifier and table.isScopeKnown(scope)) {
                const fname = c.callee.Identifier.name;
                if (table.safeFindUp(scope, fname)) |sym| {
                    if (sym.typ) |ti| {
                        if (ti.ora_type) |ot| switch (ot) {
                            .function => |fnty| {
                                if (fnty.return_type) |ret| break :blk_call ast.Types.TypeInfo.fromOraType(@constCast(ret).*);
                                break :blk_call ast.Types.TypeInfo.fromOraType(.void);
                            },
                            else => {},
                        };
                    }
                }
            }
            // Fallback to callee type inference
            const callee_ti = inferExprType(table, scope, c.callee.*);
            if (callee_ti.ora_type) |ot| switch (ot) {
                .function => |fnty| {
                    if (fnty.return_type) |ret| break :blk_call ast.Types.TypeInfo.fromOraType(@constCast(ret).*);
                    break :blk_call ast.Types.TypeInfo.fromOraType(.void);
                },
                else => break :blk_call ast.Types.TypeInfo.unknown(),
            } else break :blk_call ast.Types.TypeInfo.unknown();
        },
        .EnumLiteral => |el| {
            // Treat as the enum's type
            return ast.Types.TypeInfo.fromOraType(.{ .enum_type = el.enum_name });
        },
        .Cast => |c| c.target_type,
        .Try => |t| blk_try: {
            const inner = inferExprType(table, scope, t.expr.*);
            if (inner.ora_type) |ot| switch (ot) {
                .error_union => |succ_ptr| {
                    break :blk_try ast.Types.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                },
                ._union => |members| {
                    var i: usize = 0;
                    while (i < members.len) : (i += 1) {
                        const m = members[i];
                        switch (m) {
                            .error_union => |succ_ptr| {
                                break :blk_try ast.Types.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                            },
                            else => {},
                        }
                    }
                    break :blk_try ast.Types.TypeInfo.unknown();
                },
                else => break :blk_try ast.Types.TypeInfo.unknown(),
            } else break :blk_try ast.Types.TypeInfo.unknown();
        },
        .ErrorReturn => |_| ast.Types.TypeInfo{ .category = .Error, .ora_type = null, .source = .inferred, .span = null },
        else => ast.Types.TypeInfo.unknown(),
    };
}

pub const SpecContext = enum { None, Requires, Ensures, Invariant };

pub fn validateSpecUsage(_allocator: std.mem.Allocator, expr_node: *ast.Expressions.ExprNode, ctx: SpecContext) !?ast.SourceSpan {
    // Validate current node
    switch (expr_node.*) {
        .Quantified => |*q| {
            _ = q;
            if (ctx == .None) return expr_node.Quantified.span;
        },
        .Old => |*o| {
            _ = o;
            if (ctx != .Ensures) return expr_node.Old.span;
        },
        else => {},
    }

    // Recurse into common children
    switch (expr_node.*) {
        .Binary => |*b| {
            if (try validateSpecUsage(_allocator, b.lhs, ctx)) |sp| return sp;
            if (try validateSpecUsage(_allocator, b.rhs, ctx)) |sp| return sp;
        },
        .Unary => |*u| {
            if (try validateSpecUsage(_allocator, u.operand, ctx)) |sp| return sp;
        },
        .Assignment => |*a| {
            if (try validateSpecUsage(_allocator, a.value, ctx)) |sp| return sp;
        },
        .Call => |*c| {
            if (try validateSpecUsage(_allocator, c.callee, ctx)) |sp| return sp;
            for (c.arguments) |arg| if (try validateSpecUsage(_allocator, arg, ctx)) |sp| return sp;
        },
        .Tuple => |*t| {
            for (t.elements) |e| if (try validateSpecUsage(_allocator, e, ctx)) |sp| return sp;
        },
        else => {},
    }

    return null;
}
