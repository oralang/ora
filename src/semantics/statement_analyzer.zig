const std = @import("std");
const ast = @import("../ast.zig");
const DEBUG_SEMANTICS: bool = false;
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");
const locals = @import("locals_binder.zig");
const MemoryRegion = @import("../ast.zig").Memory.Region;

pub fn checkFunctionBody(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    scope: *state.Scope,
    f: *const ast.FunctionNode,
) ![]const ast.SourceSpan {
    var issues = std.ArrayList(ast.SourceSpan).init(allocator);
    // Walk blocks recursively and check returns using the provided function scope
    try walkBlock(&issues, table, scope, &f.body, f.return_type_info);
    return try issues.toOwnedSlice();
}

pub fn collectUnknownIdentifierSpans(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    scope: *state.Scope,
    f: *const ast.FunctionNode,
) ![]const ast.SourceSpan {
    var issues = std.ArrayList(ast.SourceSpan).init(allocator);
    try walkBlockForUnknowns(&issues, table, scope, &f.body);
    return try issues.toOwnedSlice();
}

fn resolveBlockScope(table: *state.SymbolTable, default_scope: *state.Scope, block: *const ast.Statements.BlockNode) *state.Scope {
    const key: usize = @intFromPtr(block);
    if (table.block_scopes.get(key)) |sc| return sc;
    return default_scope;
}

fn walkBlockForUnknowns(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .Expr => |e| try visitExprForUnknowns(issues, table, scope, e),
            .Return => |r| if (r.value) |v| try visitExprForUnknowns(issues, table, scope, v),
            .If => |iff| {
                try visitExprForUnknowns(issues, table, scope, iff.condition);
                const then_scope = resolveBlockScope(table, scope, &iff.then_branch);
                try walkBlockForUnknowns(issues, table, then_scope, &iff.then_branch);
                if (iff.else_branch) |*eb| {
                    const else_scope = resolveBlockScope(table, scope, eb);
                    try walkBlockForUnknowns(issues, table, else_scope, eb);
                }
            },
            .While => |wh| {
                try visitExprForUnknowns(issues, table, scope, wh.condition);
                const body_scope = resolveBlockScope(table, scope, &wh.body);
                try walkBlockForUnknowns(issues, table, body_scope, &wh.body);
            },
            .ForLoop => |fl| {
                try visitExprForUnknowns(issues, table, scope, fl.iterable);
                const body_scope = resolveBlockScope(table, scope, &fl.body);
                try walkBlockForUnknowns(issues, table, body_scope, &fl.body);
            },
            .TryBlock => |tb| {
                const try_scope = resolveBlockScope(table, scope, &tb.try_block);
                try walkBlockForUnknowns(issues, table, try_scope, &tb.try_block);
                if (tb.catch_block) |cb| {
                    const catch_scope = resolveBlockScope(table, scope, &cb.block);
                    try walkBlockForUnknowns(issues, table, catch_scope, &cb.block);
                }
            },
            // No plain Block variant in StmtNode; ignore
            else => {},
        }
    }
}

fn isScopeKnown(table: *state.SymbolTable, scope: *const state.Scope) bool {
    if (scope == &table.root) return true;
    for (table.scopes.items) |sc| {
        if (sc == scope) return true;
    }
    return false;
}

fn visitExprForUnknowns(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, expr_node: ast.Expressions.ExprNode) !void {
    if (!isScopeKnown(table, scope)) return; // Defensive guard
    switch (expr_node) {
        .Identifier => |id| {
            if (state.SymbolTable.findUp(scope, id.name)) |_| {
                // ok
            } else {
                try issues.append(id.span);
            }
        },
        .Binary => |b| {
            try visitExprForUnknowns(issues, table, scope, b.lhs.*);
            try visitExprForUnknowns(issues, table, scope, b.rhs.*);
        },
        .Unary => |u| try visitExprForUnknowns(issues, table, scope, u.operand.*),
        .Assignment => |a| {
            try visitExprForUnknowns(issues, table, scope, a.target.*);
            try visitExprForUnknowns(issues, table, scope, a.value.*);
        },
        .CompoundAssignment => |ca| {
            try visitExprForUnknowns(issues, table, scope, ca.target.*);
            try visitExprForUnknowns(issues, table, scope, ca.value.*);
        },
        .Call => |c| {
            try visitExprForUnknowns(issues, table, scope, c.callee.*);
            for (c.arguments) |arg| try visitExprForUnknowns(issues, table, scope, arg.*);
        },
        .Index => |ix| {
            try visitExprForUnknowns(issues, table, scope, ix.target.*);
            try visitExprForUnknowns(issues, table, scope, ix.index.*);
        },
        .FieldAccess => |fa| try visitExprForUnknowns(issues, table, scope, fa.target.*),
        .Cast => |c| try visitExprForUnknowns(issues, table, scope, c.operand.*),
        .Comptime => |ct| {
            _ = ct; // Skip deeper traversal for comptime blocks in unknowns walker
        },
        .Tuple => |t| for (t.elements) |el| try visitExprForUnknowns(issues, table, scope, el.*),
        .SwitchExpression => {
            // Only condition expression here; case bodies handled in statement traversal when needed
            // Note: the payload provides access to condition; re-fetch from expr_node when necessary
            if (expr_node == .SwitchExpression) {
                try visitExprForUnknowns(issues, table, scope, expr_node.SwitchExpression.condition.*);
            }
        },
        .AnonymousStruct => |as| for (as.fields) |f| try visitExprForUnknowns(issues, table, scope, f.value.*),
        .ArrayLiteral => |al| for (al.elements) |el| try visitExprForUnknowns(issues, table, scope, el.*),
        .Range => |r| {
            try visitExprForUnknowns(issues, table, scope, r.start.*);
            try visitExprForUnknowns(issues, table, scope, r.end.*);
        },
        .StructInstantiation => |si| {
            try visitExprForUnknowns(issues, table, scope, si.struct_name.*);
            for (si.fields) |f| try visitExprForUnknowns(issues, table, scope, f.value.*);
        },
        .Quantified => |q| {
            if (q.condition) |cond| try visitExprForUnknowns(issues, table, scope, cond.*);
            try visitExprForUnknowns(issues, table, scope, q.body.*);
        },
        .Try => |t| try visitExprForUnknowns(issues, table, scope, t.expr.*),
        .ErrorCast => |ec| try visitExprForUnknowns(issues, table, scope, ec.operand.*),
        .Shift => |sh| {
            try visitExprForUnknowns(issues, table, scope, sh.mapping.*);
            try visitExprForUnknowns(issues, table, scope, sh.source.*);
            try visitExprForUnknowns(issues, table, scope, sh.dest.*);
            try visitExprForUnknowns(issues, table, scope, sh.amount.*);
        },
        else => {},
    }
}

fn inferExprRegion(table: *state.SymbolTable, scope: *state.Scope, e: ast.Expressions.ExprNode) MemoryRegion {
    return switch (e) {
        .Identifier => |id| blk: {
            if (state.SymbolTable.findUp(scope, id.name)) |sym| {
                break :blk sym.region orelse MemoryRegion.Stack;
            }
            break :blk MemoryRegion.Stack;
        },
        .FieldAccess => |fa| inferExprRegion(table, scope, fa.target.*),
        .Index => |ix| inferExprRegion(table, scope, ix.target.*),
        .Call => MemoryRegion.Stack,
        else => MemoryRegion.Stack,
    };
}

fn isStorageLike(r: MemoryRegion) bool {
    return r == .Storage or r == .TStore;
}

fn isElementLevelTarget(target: ast.Expressions.ExprNode) bool {
    return switch (target) {
        .FieldAccess, .Index => true,
        else => false,
    };
}

fn isRegionAssignmentAllowed(target_region: MemoryRegion, source_region: MemoryRegion, target_node: ast.Expressions.ExprNode) bool {
    if (isStorageLike(target_region)) {
        if (isStorageLike(source_region)) {
            return isElementLevelTarget(target_node);
        }
        return true;
    }
    return true;
}

fn checkExpr(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, e: ast.Expressions.ExprNode) !void {
    switch (e) {
        .Call => |c| {
            if (scope.name) |caller_fn| {
                const caller_allowed = table.function_allowed_errors.get(caller_fn);
                if (c.callee.* == .Identifier) {
                    const callee_name = c.callee.Identifier.name;
                    if (table.function_allowed_errors.get(callee_name)) |callee_allowed| {
                        if (caller_allowed) |caller_list| {
                            var i: usize = 0;
                            while (i < callee_allowed.len) : (i += 1) {
                                const tag = callee_allowed[i];
                                var ok = false;
                                for (caller_list) |allowed_tag| {
                                    if (std.mem.eql(u8, allowed_tag, tag)) {
                                        ok = true;
                                        break;
                                    }
                                }
                                if (!ok) {
                                    try issues.append(c.span);
                                    break;
                                }
                            }
                        } else {
                            if (callee_allowed.len > 0) try issues.append(c.span);
                        }
                    }
                }
            }
            try checkExpr(issues, table, scope, c.callee.*);
            for (c.arguments) |arg| try checkExpr(issues, table, scope, arg.*);
        },
        .Try => |t| {
            const inner = expr.inferExprType(table, scope, t.expr.*);
            var ok_try = false;
            if (inner.ora_type) |ot| switch (ot) {
                .error_union => ok_try = true,
                else => {},
            };
            if (!ok_try) try issues.append(t.span);
            // Recurse
            try checkExpr(issues, table, scope, t.expr.*);
        },
        .ErrorReturn => |_| {},
        .Assignment => |a| {
            try checkExpr(issues, table, scope, a.target.*);
            try checkExpr(issues, table, scope, a.value.*);
        },
        .CompoundAssignment => |ca| {
            try checkExpr(issues, table, scope, ca.target.*);
            try checkExpr(issues, table, scope, ca.value.*);
        },
        .Binary => |b| {
            try checkExpr(issues, table, scope, b.lhs.*);
            try checkExpr(issues, table, scope, b.rhs.*);
        },
        .Unary => |u| try checkExpr(issues, table, scope, u.operand.*),
        .Index => |ix| {
            try checkExpr(issues, table, scope, ix.target.*);
            try checkExpr(issues, table, scope, ix.index.*);
        },
        .FieldAccess => |fa| try checkExpr(issues, table, scope, fa.target.*),
        .Cast => |c| try checkExpr(issues, table, scope, c.operand.*),
        .Tuple => |t| {
            for (t.elements) |el| try checkExpr(issues, table, scope, el.*);
        },
        .AnonymousStruct => |as| {
            for (as.fields) |f| try checkExpr(issues, table, scope, f.value.*);
        },
        .ArrayLiteral => |al| {
            for (al.elements) |el| try checkExpr(issues, table, scope, el.*);
        },
        .StructInstantiation => |si| {
            try checkExpr(issues, table, scope, si.struct_name.*);
            for (si.fields) |f| try checkExpr(issues, table, scope, f.value.*);
        },
        .Shift => |sh| {
            try checkExpr(issues, table, scope, sh.mapping.*);
            try checkExpr(issues, table, scope, sh.source.*);
            try checkExpr(issues, table, scope, sh.dest.*);
            try checkExpr(issues, table, scope, sh.amount.*);
        },
        .SwitchExpression => {
            if (e == .SwitchExpression) try checkExpr(issues, table, scope, e.SwitchExpression.condition.*);
        },
        else => {},
    }
}

fn walkBlock(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode, ret_type: ?ast.Types.TypeInfo) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .VariableDecl => |v| {
                if (v.value) |vp| try checkExpr(issues, table, scope, vp.*);
                if (v.value) |vp2| {
                    const tr = v.region;
                    const sr = inferExprRegion(table, scope, vp2.*);
                    if (!isRegionAssignmentAllowed(tr, sr, .{ .Identifier = .{ .name = v.name, .span = v.span } })) {
                        try issues.append(v.span);
                    }
                }
            },
            .Expr => |e| {
                switch (e) {
                    .Assignment => |a| {
                        // LHS must be an lvalue: Identifier, FieldAccess, Index
                        const lhs_is_lvalue = switch (a.target.*) {
                            .Identifier, .FieldAccess, .Index => true,
                            else => false,
                        };
                        if (!lhs_is_lvalue) try issues.append(a.span);
                        // Mutability: reject writes to non-mutable bindings when target is an Identifier
                        if (a.target.* == .Identifier) {
                            const name = a.target.Identifier.name;
                            if (state.SymbolTable.findUp(scope, name)) |sym| {
                                if (!sym.mutable) try issues.append(a.span);
                            }
                        }
                        // Type compatibility (best-effort)
                        const lt = expr.inferExprType(table, scope, a.target.*);
                        const rt = expr.inferExprType(table, scope, a.value.*);
                        if (lt.ora_type != null and rt.ora_type != null and !ast.Types.TypeInfo.equals(lt, rt)) {
                            try issues.append(a.span);
                        }
                        // Recurse into RHS for error checks
                        try checkExpr(issues, table, scope, a.value.*);
                        // Region transition validation
                        const tr = inferExprRegion(table, scope, a.target.*);
                        const sr = inferExprRegion(table, scope, a.value.*);
                        if (!isRegionAssignmentAllowed(tr, sr, a.target.*)) {
                            try issues.append(a.span);
                        }
                    },
                    .CompoundAssignment => |ca| {
                        // LHS must be an lvalue
                        const lhs_is_lvalue = switch (ca.target.*) {
                            .Identifier, .FieldAccess, .Index => true,
                            else => false,
                        };
                        if (!lhs_is_lvalue) try issues.append(ca.span);
                        // Mutability: reject writes to non-mutable bindings when target is an Identifier
                        if (ca.target.* == .Identifier) {
                            const name = ca.target.Identifier.name;
                            if (state.SymbolTable.findUp(scope, name)) |sym| {
                                if (!sym.mutable) try issues.append(ca.span);
                            }
                        }
                        // Numeric-only first cut
                        const lt = expr.inferExprType(table, scope, ca.target.*);
                        const rt = expr.inferExprType(table, scope, ca.value.*);
                        if (lt.ora_type) |lot| {
                            if (!lot.isInteger()) try issues.append(ca.span);
                        }
                        if (rt.ora_type) |rot| {
                            if (!rot.isInteger()) try issues.append(ca.span);
                        }
                        try checkExpr(issues, table, scope, ca.value.*);
                        // Region transition validation for compound assignment
                        const tr2 = inferExprRegion(table, scope, ca.target.*);
                        const sr2 = inferExprRegion(table, scope, ca.value.*);
                        if (!isRegionAssignmentAllowed(tr2, sr2, ca.target.*)) {
                            try issues.append(ca.span);
                        }
                    },
                    else => {
                        // General expression: traverse for error rules
                        try checkExpr(issues, table, scope, e);
                    },
                }
            },
            .Return => |r| {
                if (ret_type) |rt| {
                    if (r.value) |v| {
                        const vt = expr.inferExprType(table, scope, v);
                        var ok = ast.Types.TypeInfo.equals(vt, rt) or ast.Types.TypeInfo.isCompatibleWith(vt, rt);
                        if (!ok) {
                            // Allow returning T when expected is !T (success case)
                            if (rt.ora_type) |rot| switch (rot) {
                                .error_union => |succ_ptr| {
                                    const succ = ast.Types.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                                    ok = ast.Types.TypeInfo.equals(vt, succ) or ast.Types.TypeInfo.isCompatibleWith(vt, succ);
                                },
                                ._union => |members| {
                                    var i: usize = 0;
                                    while (!ok and i < members.len) : (i += 1) {
                                        const m = members[i];
                                        switch (m) {
                                            .error_union => |succ_ptr| {
                                                const succ = ast.Types.TypeInfo.fromOraType(@constCast(succ_ptr).*);
                                                if (ast.Types.TypeInfo.equals(vt, succ) or ast.Types.TypeInfo.isCompatibleWith(vt, succ)) {
                                                    ok = true;
                                                }
                                            },
                                            else => {},
                                        }
                                    }
                                },
                                else => {},
                            };
                            // Allow returning error.SomeError when return type includes that error tag
                            if (!ok and v == .ErrorReturn) {
                                if (scope.name) |fn_name| {
                                    if (table.function_allowed_errors.get(fn_name)) |allowed| {
                                        const tag = v.ErrorReturn.error_name;
                                        var found = false;
                                        for (allowed) |nm| {
                                            if (std.mem.eql(u8, nm, tag)) {
                                                found = true;
                                                break;
                                            }
                                        }
                                        ok = found;
                                    }
                                }
                            }
                        }
                        if (!ok) {
                            // Debug log to understand mismatches (fixture debugging aid)
                            var vbuf: [256]u8 = undefined;
                            var rbuf: [256]u8 = undefined;
                            var vstream = std.io.fixedBufferStream(&vbuf);
                            var rstream = std.io.fixedBufferStream(&rbuf);
                            if (vt.ora_type) |vot| {
                                _ = (@constCast(&vot)).*; // silence unused warnings
                                (@constCast(&vot)).*.render(vstream.writer()) catch {};
                            }
                            if (rt.ora_type) |rot3| {
                                _ = (@constCast(&rot3)).*;
                                (@constCast(&rot3)).*.render(rstream.writer()) catch {};
                            }
                            if (DEBUG_SEMANTICS) {
                                const vstr = vstream.getWritten();
                                const rstr = rstream.getWritten();
                                const is_ident = (v == .Identifier);
                                const fname = scope.name orelse "<anon>";
                                std.debug.print("[semantics] Return mismatch in {s}: is_ident={}, vt='{s}', rt='{s}' at {d}:{d}\n", .{ fname, is_ident, vstr, rstr, r.span.line, r.span.column });
                            }
                            try issues.append(r.span);
                        }
                    } else {
                        // Void return only allowed when function is void
                        const void_ok = (rt.ora_type != null and rt.ora_type.? == .void);
                        if (!void_ok) try issues.append(r.span);
                    }
                } else {
                    // No declared return type -> treat as void; returning a value is an issue
                    if (r.value != null) try issues.append(r.span);
                }
            },
            .If => |iff| {
                const then_scope = resolveBlockScope(table, scope, &iff.then_branch);
                try walkBlock(issues, table, then_scope, &iff.then_branch, ret_type);
                if (iff.else_branch) |*eb| {
                    const else_scope = resolveBlockScope(table, scope, eb);
                    try walkBlock(issues, table, else_scope, eb, ret_type);
                }
            },
            .While => |wh| {
                const body_scope = resolveBlockScope(table, scope, &wh.body);
                try walkBlock(issues, table, body_scope, &wh.body, ret_type);
            },
            .ForLoop => |fl| {
                const body_scope = resolveBlockScope(table, scope, &fl.body);
                try walkBlock(issues, table, body_scope, &fl.body, ret_type);
            },
            .TryBlock => |tb| {
                const try_scope = resolveBlockScope(table, scope, &tb.try_block);
                try walkBlock(issues, table, try_scope, &tb.try_block, ret_type);
                if (tb.catch_block) |cb| {
                    const catch_scope = resolveBlockScope(table, scope, &cb.block);
                    try walkBlock(issues, table, catch_scope, &cb.block, ret_type);
                }
            },
            // No plain Block variant in StmtNode
            else => {},
        }
    }
}
