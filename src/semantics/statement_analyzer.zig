// ============================================================================
// Statement Analyzer - Phase 2 Validation
// ============================================================================
//
// Validates statement-level semantics including type checking, memory regions,
// and control flow.
//
// VALIDATION RULES:
//   • Return types must match declarations
//   • Storage → Storage assignments only for element-level (a[0] = b[1])
//   • Switch patterns must be exhaustive for enums
//   • Assignment targets must be mutable lvalues
//
// SECTIONS:
//   • Entry points
//   • Unknown identifier checking (optional)
//   • Memory region inference
//   • Switch pattern validation
//   • Expression validation
//   • Block walking
//
// NOTE: Unknown identifier checking is enabled in semantics/core.zig.
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const DEBUG_SEMANTICS: bool = false;
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");
const builtins = @import("../semantics.zig").builtins;
const locals = @import("locals_binder.zig");
const MemoryRegion = @import("../ast.zig").Memory.Region;

// ============================================================================
// SECTION 1: Entry Points
// ============================================================================

// Removed: now using table.safeFindUp() instead

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

// ============================================================================
// SECTION 2: Unknown Identifier Checking
// ============================================================================

fn walkBlockForUnknowns(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .Expr => |e| try visitExprForUnknowns(issues, table, scope, e),
            .VariableDecl => |v| {
                if (v.value) |val| try visitExprForUnknowns(issues, table, scope, val.*);
            },
            .DestructuringAssignment => |da| try visitExprForUnknowns(issues, table, scope, da.value.*),
            .Return => |r| if (r.value) |v| try visitExprForUnknowns(issues, table, scope, v),
            .CompoundAssignment => |ca| {
                try visitExprForUnknowns(issues, table, scope, ca.target);
                try visitExprForUnknowns(issues, table, scope, ca.value);
            },
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
            .Log => |log| {
                for (log.args) |arg| try visitExprForUnknowns(issues, table, scope, arg);
            },
            .Lock => |lock| try visitExprForUnknowns(issues, table, scope, lock.path),
            .Unlock => |unlock| try visitExprForUnknowns(issues, table, scope, unlock.path),
            .Assert => |assert_stmt| try visitExprForUnknowns(issues, table, scope, assert_stmt.condition),
            .Invariant => |inv| try visitExprForUnknowns(issues, table, scope, inv.condition),
            .Requires => |req| try visitExprForUnknowns(issues, table, scope, req.condition),
            .Ensures => |ens| try visitExprForUnknowns(issues, table, scope, ens.condition),
            .Assume => |assume| try visitExprForUnknowns(issues, table, scope, assume.condition),
            .Havoc => |havoc| try visitExprForUnknowns(issues, table, scope, havoc.condition),
            .Break => |br| if (br.value) |val| try visitExprForUnknowns(issues, table, scope, val.*),
            .Continue => |cont| if (cont.value) |val| try visitExprForUnknowns(issues, table, scope, val.*),
            .Switch => |sw| {
                try visitExprForUnknowns(issues, table, scope, sw.condition);
                for (sw.cases) |*case| {
                    switch (case.pattern) {
                        .Range => |r| {
                            try visitExprForUnknowns(issues, table, scope, r.start.*);
                            try visitExprForUnknowns(issues, table, scope, r.end.*);
                        },
                        else => {},
                    }
                    switch (case.body) {
                        .Expression => |expr_ptr| try visitExprForUnknowns(issues, table, scope, expr_ptr.*),
                        .Block => |*blk| {
                            const case_scope = resolveBlockScope(table, scope, blk);
                            try walkBlockForUnknowns(issues, table, case_scope, blk);
                        },
                        .LabeledBlock => |*lb| {
                            const case_scope = resolveBlockScope(table, scope, &lb.block);
                            try walkBlockForUnknowns(issues, table, case_scope, &lb.block);
                        },
                    }
                }
                if (sw.default_case) |*defb| {
                    const def_scope = resolveBlockScope(table, scope, defb);
                    try walkBlockForUnknowns(issues, table, def_scope, defb);
                }
            },
            .LabeledBlock => |lb| {
                const inner_scope = resolveBlockScope(table, scope, &lb.block);
                try walkBlockForUnknowns(issues, table, inner_scope, &lb.block);
            },
            // No plain Block variant in StmtNode; ignore
            else => {},
        }
    }
}

// Removed: now using table.isScopeKnown() instead

// Add: leaf value type helper for storage composite checks
fn isLeafValueType(ti: ast.Types.TypeInfo) bool {
    if (ti.ora_type) |ot| switch (ot.getCategory()) {
        .Integer, .Bool, .Address, .Bytes, .String, .Enum => return true,
        else => return false,
    };
    return false;
}

fn visitExprForUnknowns(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, expr_node: ast.Expressions.ExprNode) !void {
    if (!table.isScopeKnown(scope)) return; // Defensive guard
    switch (expr_node) {
        .Identifier => |id| {
            if (table.isScopeKnown(scope) and table.safeFindUp(scope, id.name) != null) {
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
            if (builtins.isMemberAccessChain(c.callee)) {
                const path = builtins.getMemberAccessPath(table.allocator, c.callee) catch "";
                defer if (path.len > 0) table.allocator.free(path);
                if (path.len > 0 and table.builtin_registry.lookup(path) != null) {
                    for (c.arguments) |arg| try visitExprForUnknowns(issues, table, scope, arg.*);
                    return;
                }
            }
            try visitExprForUnknowns(issues, table, scope, c.callee.*);
            for (c.arguments) |arg| try visitExprForUnknowns(issues, table, scope, arg.*);
        },
        .Index => |ix| {
            try visitExprForUnknowns(issues, table, scope, ix.target.*);
            try visitExprForUnknowns(issues, table, scope, ix.index.*);
        },
        .FieldAccess => |fa| {
            if (builtins.isMemberAccessChain(&expr_node)) {
                const path = builtins.getMemberAccessPath(table.allocator, &expr_node) catch "";
                defer if (path.len > 0) table.allocator.free(path);
                if (path.len > 0 and table.builtin_registry.lookup(path) != null) return;
            }
            try visitExprForUnknowns(issues, table, scope, fa.target.*);
        },
        .Cast => |c| try visitExprForUnknowns(issues, table, scope, c.operand.*),
        .Comptime => |ct| {
            _ = ct; // Skip deeper traversal for comptime blocks in unknowns walker
        },
        .Tuple => |t| for (t.elements) |el| try visitExprForUnknowns(issues, table, scope, el.*),
        .SwitchExpression => {
            // Condition expression
            if (expr_node == .SwitchExpression) try visitExprForUnknowns(issues, table, scope, expr_node.SwitchExpression.condition.*);
            // Semantic checks: pattern compatibility against condition type
            const cond_ti = expr.inferExprType(table, scope, expr_node.SwitchExpression.condition.*);
            try checkSwitchPatterns(issues, table, scope, cond_ti, expr_node.SwitchExpression.cases);
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
        .Destructuring => |d| try visitExprForUnknowns(issues, table, scope, d.value.*),
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

// ============================================================================
// SECTION 3: Memory Region Inference
// ============================================================================

fn inferExprRegion(table: *state.SymbolTable, scope: *state.Scope, e: ast.Expressions.ExprNode) MemoryRegion {
    if (!table.isScopeKnown(scope)) return MemoryRegion.Stack;
    return switch (e) {
        .Identifier => |id| blk: {
            var cur: ?*const state.Scope = scope;
            while (cur) |s| : (cur = s.parent) {
                if (!table.isScopeKnown(s)) break;
                if (s.findInCurrent(id.name)) |idx| {
                    const sym = s.symbols.items[idx];
                    break :blk sym.region orelse MemoryRegion.Stack;
                }
            }
            // Fallback: check root symbols for top-level variables (e.g., storage vars)
            if (table.root.findInCurrent(id.name)) |idx_root| {
                const rsym = table.root.symbols.items[idx_root];
                break :blk rsym.region orelse MemoryRegion.Stack;
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

// ============================================================================
// SECTION 4: Switch Expression Validation
// ============================================================================

fn checkSwitchPatterns(
    issues: *std.ArrayList(ast.SourceSpan),
    table: *state.SymbolTable,
    scope: *state.Scope,
    cond_type: ast.Types.TypeInfo,
    cases: []const ast.Switch.Case,
) !void {
    var seen_else = false;
    var enum_coverage: ?struct { name: []const u8, covered: std.StringHashMap(void) } = null;
    var seen_literals = std.StringHashMap(void).init(table.allocator);
    defer seen_literals.deinit();
    var seen_enum_variants = std.StringHashMap(void).init(table.allocator);
    defer seen_enum_variants.deinit();
    // Track numeric ranges precisely to detect overlaps
    var numeric_ranges = std.ArrayList(struct { start: u128, end: u128, span: ast.SourceSpan }).init(table.allocator);
    defer numeric_ranges.deinit();
    // If condition is enum, seed coverage map
    if (cond_type.ora_type) |ot| switch (ot) {
        .enum_type => |ename| {
            enum_coverage = .{ .name = ename, .covered = std.StringHashMap(void).init(table.allocator) };
        },
        else => {},
    };
    defer if (enum_coverage) |*ec| ec.covered.deinit();
    for (cases) |case| {
        switch (case.pattern) {
            .Else => {
                // else must be last
                if (!seen_else) {
                    seen_else = true;
                } else {
                    // Duplicate else
                    try issues.append(case.span);
                }
            },
            .Literal => |lit| {
                // Best-effort: if condition has an ora_type, require literal category compatibility
                if (cond_type.ora_type) |ot| switch (lit.value) {
                    .Integer => if (!ot.isInteger()) try issues.append(lit.span),
                    .Bool => if (ot.getCategory() != .Bool) try issues.append(lit.span),
                    .String => if (ot.getCategory() != .String) try issues.append(lit.span),
                    .Address => if (ot.getCategory() != .Address) try issues.append(lit.span),
                    .Hex, .Binary => if (!ot.isUnsignedInteger()) try issues.append(lit.span),
                    .Character => if (!ot.isUnsignedInteger()) try issues.append(lit.span),
                    .Bytes => if (!ot.isUnsignedInteger()) try issues.append(lit.span),
                };
                // Duplicate literal detection (key by category + value when available)
                const key: []const u8 = switch (lit.value) {
                    .Integer => lit.value.Integer.value,
                    .String => lit.value.String.value,
                    .Bool => if (lit.value.Bool.value) "true" else "false",
                    .Address => lit.value.Address.value,
                    .Hex => lit.value.Hex.value,
                    .Binary => lit.value.Binary.value,
                    .Character => blk: {
                        var buf: [4]u8 = undefined;
                        const result = std.fmt.bufPrint(&buf, "{c}", .{lit.value.Character.value}) catch "?";
                        break :blk result;
                    },
                    .Bytes => lit.value.Bytes.value,
                };
                if (seen_literals.contains(key)) {
                    try issues.append(lit.span);
                } else {
                    _ = try seen_literals.put(key, {});
                }
            },
            .EnumValue => |ev| {
                if (cond_type.ora_type) |ot| switch (ot) {
                    .enum_type => |ename| {
                        // If pattern provided a qualified enum name, it must match the condition's enum
                        if (ev.enum_name.len != 0 and !std.mem.eql(u8, ev.enum_name, ename)) {
                            try issues.append(ev.span);
                        }
                        if (enum_coverage) |*ec| {
                            if (std.mem.eql(u8, ec.name, ename)) {
                                _ = try ec.covered.put(ev.variant_name, {});
                            }
                        }
                        // Duplicate enum variant detection (by variant name)
                        if (seen_enum_variants.contains(ev.variant_name)) {
                            try issues.append(ev.span);
                        } else {
                            _ = try seen_enum_variants.put(ev.variant_name, {});
                        }
                    },
                    else => try issues.append(ev.span),
                };
            },
            .Range => |rg| {
                // Both endpoints must be orderable and compatible with condition type
                const st = expr.inferExprType(table, scope, rg.start.*);
                const et = expr.inferExprType(table, scope, rg.end.*);
                if (cond_type.ora_type != null) {
                    const ok_start = ast.Types.TypeInfo.equals(st, cond_type) or ast.Types.TypeInfo.isCompatibleWith(st, cond_type);
                    const ok_end = ast.Types.TypeInfo.equals(et, cond_type) or ast.Types.TypeInfo.isCompatibleWith(et, cond_type);
                    if (!ok_start or !ok_end) try issues.append(rg.span);
                }
                // Precise overlap detection for integer-like literal ranges
                const parseU128FromLiteral = struct {
                    fn stripPrefixDigits(s: []const u8, base: u8) []const u8 {
                        if (base == 16) {
                            if (s.len >= 2 and (s[0] == '0' and (s[1] == 'x' or s[1] == 'X'))) return s[2..];
                        } else if (base == 2) {
                            if (s.len >= 2 and (s[0] == '0' and (s[1] == 'b' or s[1] == 'B'))) return s[2..];
                        }
                        return s;
                    }
                    fn parse(l: ast.Expressions.LiteralExpr) ?u128 {
                        return switch (l) {
                            .Integer => |ival| blk: {
                                const raw = ival.value;
                                if (raw.len > 0 and raw[0] == '-') break :blk null;
                                std.debug.assert(raw.len > 0);
                                const cleaned = raw;
                                return std.fmt.parseInt(u128, cleaned, 10) catch null;
                            },
                            .Hex => |hval| blk: {
                                const raw = hval.value;
                                if (raw.len > 0 and raw[0] == '-') break :blk null;
                                const digits = stripPrefixDigits(raw, 16);
                                return std.fmt.parseInt(u128, digits, 16) catch null;
                            },
                            .Binary => |bval| blk: {
                                const raw = bval.value;
                                if (raw.len > 0 and raw[0] == '-') break :blk null;
                                const digits = stripPrefixDigits(raw, 2);
                                return std.fmt.parseInt(u128, digits, 2) catch null;
                            },
                            .Bool, .String, .Address, .Character, .Bytes => null,
                        };
                    }
                };

                var start_val_opt: ?u128 = null;
                var end_val_opt: ?u128 = null;
                if (rg.start.* == .Literal) start_val_opt = parseU128FromLiteral.parse(rg.start.Literal);
                if (rg.end.* == .Literal) end_val_opt = parseU128FromLiteral.parse(rg.end.Literal);
                if (start_val_opt) |sv| {
                    if (end_val_opt) |ev| {
                        const smin: u128 = if (sv <= ev) sv else ev;
                        const smax: u128 = if (sv <= ev) ev else sv;
                        // Overlap if max(starts) <= min(ends)
                        var i: usize = 0;
                        while (i < numeric_ranges.items.len) : (i += 1) {
                            const prev = numeric_ranges.items[i];
                            const lo = if (prev.start >= smin) prev.start else smin;
                            const hi = if (prev.end <= smax) prev.end else smax;
                            if (lo <= hi) { // intervals intersect
                                try issues.append(rg.span);
                                break;
                            }
                        }
                        try numeric_ranges.append(.{ .start = smin, .end = smax, .span = rg.span });
                    }
                }
            },
        }
        // If we've seen else, no further cases allowed
        if (seen_else and case.pattern != .Else) {
            try issues.append(case.span);
        }
    }
    // Enum exhaustiveness: if no else, ensure all variants are covered by enum values (ignore ranges for now)
    if (!seen_else) {
        if (enum_coverage) |*ec| {
            if (table.enum_variants.get(ec.name)) |all| {
                var i: usize = 0;
                while (i < all.len) : (i += 1) {
                    if (!ec.covered.contains(all[i])) {
                        // Missing variant
                        // Use first case span if available to report; otherwise nothing
                        if (cases.len > 0) try issues.append(cases[0].span);
                        break;
                    }
                }
            }
        }
    }
    // Note: else positioning enforced in parser; here we could validate if needed.
}

fn checkSwitchExpressionResultTypes(
    issues: *std.ArrayList(ast.SourceSpan),
    table: *state.SymbolTable,
    scope: *state.Scope,
    sw: *const ast.Expressions.SwitchExprNode,
) !void {
    var result_ti: ?ast.Types.TypeInfo = null;
    for (sw.cases) |case| {
        switch (case.body) {
            .Expression => |expr_ptr| {
                const ti = expr.inferExprType(table, scope, expr_ptr.*);
                if (result_ti == null) {
                    result_ti = ti;
                } else if (!ast.Types.TypeInfo.equals(result_ti.?, ti) and !ast.Types.TypeInfo.isCompatibleWith(ti, result_ti.?)) {
                    try issues.append(case.span);
                }
            },
            .Block, .LabeledBlock => {
                // Switch expressions should not contain block bodies; parser prevents this, but double-check
                try issues.append(case.span);
            },
        }
    }
    if (sw.default_case) |*defb| {
        // Default is synthesized as a block with a single expression statement; check its type
        if (defb.statements.len > 0) {
            const first = defb.statements[0];
            if (first == .Expr) {
                const dti = expr.inferExprType(table, scope, first.Expr);
                if (result_ti == null) {
                    result_ti = dti;
                } else if (!ast.Types.TypeInfo.equals(result_ti.?, dti) and !ast.Types.TypeInfo.isCompatibleWith(dti, result_ti.?)) {
                    try issues.append(defb.span);
                }
            } else {
                // Non-expression in default for switch expression is invalid
                try issues.append(defb.span);
            }
        }
    }
}

// ============================================================================
// SECTION 5: Expression Validation (Assignments, Mutability)
// ============================================================================

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
            if (e == .SwitchExpression) {
                try checkExpr(issues, table, scope, e.SwitchExpression.condition.*);
                const cond_ti = expr.inferExprType(table, scope, e.SwitchExpression.condition.*);
                try checkSwitchPatterns(issues, table, scope, cond_ti, e.SwitchExpression.cases);
                try checkSwitchExpressionResultTypes(issues, table, scope, &e.SwitchExpression);
            }
        },
        else => {},
    }
}

// ============================================================================
// SECTION 6: Block Walking & Statement Checking
// ============================================================================

fn walkBlock(issues: *std.ArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode, ret_type: ?ast.Types.TypeInfo) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .VariableDecl => |v| {
                if (v.value) |vp| try checkExpr(issues, table, scope, vp.*);
                if (v.value) |vp2| {
                    const tr = v.region;
                    const sr = inferExprRegion(table, scope, vp2.*);
                    if (!isRegionAssignmentAllowed(tr, sr, .{ .Identifier = .{ .name = v.name, .type_info = ast.Types.TypeInfo.unknown(), .span = v.span } })) {
                        try issues.append(v.span);
                    }
                    // New: forbid composite-type bulk copies into storage
                    if (isStorageLike(tr) and isStorageLike(sr)) {
                        const declared_ti = v.type_info;
                        if (!isLeafValueType(declared_ti)) try issues.append(v.span);
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
                            if (table.isScopeKnown(scope)) {
                                if (table.safeFindUp(scope, name)) |sym| {
                                    if (!sym.mutable) try issues.append(a.span);
                                }
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
                        // New: forbid storage-to-storage bulk copy of composite types only when assigning identifiers (whole value)
                        if (isStorageLike(tr) and isStorageLike(sr) and a.target.* == .Identifier) {
                            if (!isLeafValueType(lt)) try issues.append(a.span);
                        }
                        // Explicit root-scope storage-to-storage bulk copy guard (identifiers)
                        if (a.target.* == .Identifier and a.value.* == .Identifier) {
                            if (table.root.findInCurrent(a.target.Identifier.name)) |idx_t| {
                                if (table.root.findInCurrent(a.value.Identifier.name)) |idx_s| {
                                    const ts = table.root.symbols.items[idx_t];
                                    const ss = table.root.symbols.items[idx_s];
                                    if (isStorageLike(ts.region orelse MemoryRegion.Stack) and isStorageLike(ss.region orelse MemoryRegion.Stack)) {
                                        try issues.append(a.span);
                                    }
                                }
                            }
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
                            if (table.isScopeKnown(scope)) {
                                if (table.safeFindUp(scope, name)) |sym| {
                                    if (!sym.mutable) try issues.append(ca.span);
                                }
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
                        // New: compound ops must target leaf types in storage when targeting identifiers (bulk not allowed)
                        if (isStorageLike(tr2) and isStorageLike(sr2) and ca.target.* == .Identifier) {
                            if (!isLeafValueType(lt)) try issues.append(ca.span);
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
