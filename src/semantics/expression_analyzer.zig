const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");

pub const ExprAnalysis = struct {
    typ: ast.TypeInfo,
};

pub fn inferExprType(table: *state.SymbolTable, scope: *state.Scope, expr: ast.ExprNode) ast.TypeInfo {
    return switch (expr) {
        .Identifier => |id| blk: {
            if (state.SymbolTable.findUp(scope, id.name)) |sym| {
                if (sym.typ) |ti| {
                    break :blk ti;
                }
            }
            break :blk ast.TypeInfo.unknown();
        },
        .Literal => |lit| switch (lit) {
            .Integer => ast.TypeInfo.fromOraType(.u256),
            .String => ast.TypeInfo.fromOraType(.string),
            .Bool => ast.TypeInfo.fromOraType(.bool),
            .Address => ast.TypeInfo.fromOraType(.address),
            .Hex => ast.TypeInfo.fromOraType(.bytes),
            .Binary => ast.TypeInfo.fromOraType(.bytes),
        },
        .FieldAccess => |fa| blk_fa: {
            const target_ti = inferExprType(table, scope, fa.target.*);
            if (target_ti.ora_type) |ot| switch (ot) {
                .anonymous_struct => |fields| {
                    var i: usize = 0;
                    while (i < fields.len) : (i += 1) {
                        if (std.mem.eql(u8, fields[i].name, fa.field)) {
                            break :blk_fa ast.TypeInfo.fromOraType(@constCast(fields[i].typ).*);
                        }
                    }
                    break :blk_fa ast.TypeInfo.unknown();
                },
                else => break :blk_fa ast.TypeInfo.unknown(),
            } else break :blk_fa ast.TypeInfo.unknown();
        },
        .Index => |ix| blk_idx: {
            const target_ti = inferExprType(table, scope, ix.target.*);
            if (target_ti.ora_type) |ot| switch (ot) {
                .array => |arr| break :blk_idx ast.TypeInfo.fromOraType(@constCast(arr.elem).*),
                .slice => |elem| break :blk_idx ast.TypeInfo.fromOraType(@constCast(elem).*),
                .mapping => |m| break :blk_idx ast.TypeInfo.fromOraType(@constCast(m.value).*),
                else => break :blk_idx ast.TypeInfo.unknown(),
            } else break :blk_idx ast.TypeInfo.unknown();
        },
        .Call => |c| blk_call: {
            // Prefer direct function symbol lookup for simple identifiers
            if (c.callee.* == .Identifier) {
                const fname = c.callee.Identifier.name;
                if (state.SymbolTable.findUp(scope, fname)) |sym| {
                    if (sym.typ) |ti| {
                        if (ti.ora_type) |ot| switch (ot) {
                            .function => |fnty| {
                                if (fnty.return_type) |ret| break :blk_call ast.TypeInfo.fromOraType(@constCast(ret).*);
                                break :blk_call ast.TypeInfo.fromOraType(.void);
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
                    if (fnty.return_type) |ret| break :blk_call ast.TypeInfo.fromOraType(@constCast(ret).*);
                    break :blk_call ast.TypeInfo.fromOraType(.void);
                },
                else => break :blk_call ast.TypeInfo.unknown(),
            } else break :blk_call ast.TypeInfo.unknown();
        },
        .Cast => |c| c.target_type,
        .Try => |t| blk_try: {
            const inner = inferExprType(table, scope, t.expr.*);
            if (inner.ora_type) |ot| switch (ot) {
                .error_union => |succ_ptr| {
                    break :blk_try ast.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                },
                ._union => |members| {
                    var i: usize = 0;
                    while (i < members.len) : (i += 1) {
                        const m = members[i];
                        switch (m) {
                            .error_union => |succ_ptr| {
                                break :blk_try ast.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                            },
                            else => {},
                        }
                    }
                    break :blk_try ast.TypeInfo.unknown();
                },
                else => break :blk_try ast.TypeInfo.unknown(),
            } else break :blk_try ast.TypeInfo.unknown();
        },
        .ErrorReturn => |_| ast.TypeInfo{ .category = .Error, .ora_type = null, .ast_type = null, .source = .inferred, .span = null },
        else => ast.TypeInfo.unknown(),
    };
}
