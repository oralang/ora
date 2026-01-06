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
const builtins = @import("../semantics.zig").builtins;

pub const ExprAnalysis = struct {
    typ: ast.Types.TypeInfo,
};

// Removed: now using table.isScopeKnown() and table.safeFindUpOpt() instead

pub fn inferExprType(table: *state.SymbolTable, scope: *state.Scope, expr: ast.Expressions.ExprNode) ast.Types.TypeInfo {
    return switch (expr) {
        .Identifier => |id| blk: {
            // check if this is the 'std' namespace itself
            if (std.mem.eql(u8, id.name, "std")) {
                // 'std' by itself is not a value - it's a namespace
                // return unknown (this should be an error in proper usage)
                break :blk ast.Types.TypeInfo.unknown();
            }

            if (table.isScopeKnown(scope)) {
                if (table.safeFindUpOpt(scope, id.name)) |sym| {
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
            // check if this is a builtin constant (e.g., std.constants.ZERO_ADDRESS)
            // build the path and check if it's a builtin
            if (fa.target.* == .Identifier) {
                const base = fa.target.Identifier.name;
                if (std.mem.eql(u8, base, "std")) {
                    // this is std.something - build the full path
                    const path = std.fmt.allocPrint(table.allocator, "{s}.{s}", .{ base, fa.field }) catch break :blk_fa ast.Types.TypeInfo.unknown();
                    defer table.allocator.free(path);

                    if (table.builtin_registry.lookup(path)) |builtin_info| {
                        // it's a builtin (constant or partial namespace)
                        if (!builtin_info.is_call) {
                            // it's a constant - return its type
                            break :blk_fa ast.Types.TypeInfo.fromOraType(builtin_info.return_type);
                        }
                        // it's a function - this is a partial access, should be followed by ()
                        // for now, treat as unknown (will be handled in Call case)
                        break :blk_fa ast.Types.TypeInfo.unknown();
                    }
                }
            } else if (fa.target.* == .FieldAccess) {
                // multi-level access like std.block.timestamp
                // try to build the full path
                const path = builtins.getMemberAccessPath(table.allocator, &expr) catch break :blk_fa ast.Types.TypeInfo.unknown();
                defer table.allocator.free(path);

                if (table.builtin_registry.lookup(path)) |builtin_info| {
                    if (!builtin_info.is_call) {
                        // it's a constant
                        break :blk_fa ast.Types.TypeInfo.fromOraType(builtin_info.return_type);
                    }
                    // it's a function - treat as unknown until called
                    break :blk_fa ast.Types.TypeInfo.unknown();
                }
            }

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
            // check if this is a builtin call (e.g., std.block.timestamp())
            if (builtins.isMemberAccessChain(c.callee)) {
                const path = builtins.getMemberAccessPath(table.allocator, c.callee) catch break :blk_call ast.Types.TypeInfo.unknown();
                defer table.allocator.free(path);

                if (table.builtin_registry.lookup(path)) |builtin_info| {
                    // validate it's callable
                    if (!builtin_info.is_call) {
                        // error: trying to call a constant
                        // for now, return unknown (semantic validation will catch this properly later)
                        break :blk_call ast.Types.TypeInfo.unknown();
                    }

                    // validate parameter count
                    if (c.arguments.len != builtin_info.param_types.len) {
                        // error: wrong number of arguments
                        break :blk_call ast.Types.TypeInfo.unknown();
                    }

                    // return the builtin's return type
                    break :blk_call ast.Types.TypeInfo.fromOraType(builtin_info.return_type);
                }
            }

            // prefer direct function symbol lookup for simple identifiers
            if (c.callee.* == .Identifier and table.isScopeKnown(scope)) {
                const fname = c.callee.Identifier.name;
                if (table.safeFindUpOpt(scope, fname)) |sym| {
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
            // fallback to callee type inference
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
            // treat as the enum's type
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
    // validate current node
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

    // recurse into common children
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
