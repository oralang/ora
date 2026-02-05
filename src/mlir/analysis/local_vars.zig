// ============================================================================
// Local Variable Representation Analysis
// ============================================================================
// Decide whether a stack local can be lowered as SSA or requires a memref slot.
// ============================================================================

const std = @import("std");
const lib = @import("ora_lib");

pub const LocalVarRepr = enum {
    SSA,
    MemRef,
};

const Usage = struct {
    def_count: usize = 0,
    has_lvalue_use: bool = false,
    kind: ?lib.ast.Statements.VariableKind = null,
    region: ?lib.ast.Statements.MemoryRegion = null,
};

pub fn analyzeLocalVarReprs(
    allocator: std.mem.Allocator,
    func: *const lib.FunctionNode,
) std.StringHashMap(LocalVarRepr) {
    var usage = std.StringHashMap(Usage).init(allocator);
    scanBlock(func.body, &usage);

    var reprs = std.StringHashMap(LocalVarRepr).init(allocator);
    var it = usage.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const info = entry.value_ptr.*;
        if (info.region != .Stack) continue;
        if (info.kind != .Var) continue;
        const needs_memref = info.def_count > 1 or info.has_lvalue_use;
        _ = reprs.put(name, if (needs_memref) .MemRef else .SSA) catch {};
    }

    usage.deinit();
    return reprs;
}

fn scanBlock(block: lib.ast.Statements.BlockNode, usage: *std.StringHashMap(Usage)) void {
    for (block.statements) |stmt| {
        scanStmt(stmt, usage);
    }
}

fn scanStmt(stmt: lib.ast.Statements.StmtNode, usage: *std.StringHashMap(Usage)) void {
    switch (stmt) {
        .VariableDecl => |decl| {
            if (decl.region != .Stack) return;
            if (decl.tuple_names) |names| {
                for (names) |name| {
                    var entry = usage.get(name) orelse Usage{};
                    entry.kind = decl.kind;
                    entry.region = decl.region;
                    entry.def_count += 1;
                    usage.put(name, entry) catch {};
                }
            } else {
                var entry = usage.get(decl.name) orelse Usage{};
                entry.kind = decl.kind;
                entry.region = decl.region;
                entry.def_count += 1;
                usage.put(decl.name, entry) catch {};
            }
        },
        .Expr => |expr| {
            scanExpr(&expr, usage);
        },
        .CompoundAssignment => |comp| {
            noteAssignmentTarget(comp.target, usage);
            scanExpr(comp.value, usage);
        },
        .DestructuringAssignment => |destruct| {
            scanExpr(destruct.value, usage);
            // treat destructuring as lvalue use on targets
            switch (destruct.pattern) {
                .Struct => |fields| {
                    for (fields) |field| {
                        markLValue(field.variable, usage);
                    }
                },
                .Tuple => |names| for (names) |name| markLValue(name, usage),
                .Array => |names| for (names) |name| markLValue(name, usage),
            }
        },
        .If => |if_stmt| {
            scanExpr(&if_stmt.condition, usage);
            scanBlock(if_stmt.then_branch, usage);
            if (if_stmt.else_branch) |else_branch| scanBlock(else_branch, usage);
        },
        .While => |while_stmt| {
            scanExpr(&while_stmt.condition, usage);
            scanBlock(while_stmt.body, usage);
        },
        .ForLoop => |for_stmt| {
            scanExpr(&for_stmt.iterable, usage);
            scanBlock(for_stmt.body, usage);
        },
        .Switch => |switch_stmt| {
            scanExpr(&switch_stmt.condition, usage);
            for (switch_stmt.cases) |case| {
                scanSwitchPattern(case.pattern, usage);
                switch (case.body) {
                    .Expression => |expr| scanExpr(expr, usage),
                    .Block => |block| scanBlock(block, usage),
                    .LabeledBlock => |labeled| scanBlock(labeled.block, usage),
                }
            }
            if (switch_stmt.default_case) |default_block| scanBlock(default_block, usage);
        },
        .TryBlock => |try_stmt| {
            scanBlock(try_stmt.try_block, usage);
            if (try_stmt.catch_block) |catch_block| scanBlock(catch_block.block, usage);
        },
        .LabeledBlock => |labeled| {
            scanBlock(labeled.block, usage);
        },
        else => {},
    }
}

fn scanExpr(expr: *const lib.ast.Expressions.ExprNode, usage: *std.StringHashMap(Usage)) void {
    switch (expr.*) {
        .Assignment => |assign| {
            noteAssignmentTarget(assign.target, usage);
            scanExpr(assign.value, usage);
        },
        .CompoundAssignment => |comp| {
            noteAssignmentTarget(comp.target, usage);
            scanExpr(comp.value, usage);
        },
        .Binary => |bin| {
            scanExpr(bin.lhs, usage);
            scanExpr(bin.rhs, usage);
        },
        .Unary => |un| scanExpr(un.operand, usage),
        .Cast => |cast| scanExpr(cast.operand, usage),
        .Comptime => |ct| scanBlock(ct.block, usage),
        .Old => |old_expr| scanExpr(old_expr.expr, usage),
        .Quantified => |quant| {
            if (quant.condition) |cond| scanExpr(cond, usage);
            scanExpr(quant.body, usage);
        },
        .Call => |call| {
            scanExpr(call.callee, usage);
            for (call.arguments) |arg| scanExpr(arg, usage);
        },
        .FieldAccess => |fa| scanExpr(fa.target, usage),
        .Index => |idx| {
            scanExpr(idx.target, usage);
            scanExpr(idx.index, usage);
        },
        .Range => |range| {
            scanExpr(range.start, usage);
            scanExpr(range.end, usage);
        },
        .StructInstantiation => |si| {
            scanExpr(si.struct_name, usage);
            for (si.fields) |field| scanExpr(field.value, usage);
        },
        .Tuple => |tuple| for (tuple.elements) |el| scanExpr(el, usage),
        .ArrayLiteral => |arr| for (arr.elements) |el| scanExpr(el, usage),
        .Try => |try_expr| scanExpr(try_expr.expr, usage),
        .ErrorReturn => |err_ret| {
            if (err_ret.parameters) |params| for (params) |p| scanExpr(p, usage);
        },
        .ErrorCast => |err_cast| scanExpr(err_cast.operand, usage),
        .Shift => |shift| {
            scanExpr(shift.mapping, usage);
            scanExpr(shift.source, usage);
            scanExpr(shift.dest, usage);
            scanExpr(shift.amount, usage);
        },
        .Destructuring => |destruct| scanExpr(destruct.value, usage),
        .SwitchExpression => |switch_expr| {
            scanExpr(switch_expr.condition, usage);
            for (switch_expr.cases) |case| {
                scanSwitchPattern(case.pattern, usage);
                switch (case.body) {
                    .Expression => |body_expr| scanExpr(body_expr, usage),
                    .Block => |block| scanBlock(block, usage),
                    .LabeledBlock => |labeled| scanBlock(labeled.block, usage),
                }
            }
            if (switch_expr.default_case) |default_block| scanBlock(default_block, usage);
        },
        else => {},
    }
}

fn scanSwitchPattern(pattern: lib.ast.Expressions.SwitchPattern, usage: *std.StringHashMap(Usage)) void {
    switch (pattern) {
        .Literal => |lit| {
            // literals don't affect local var usage
            _ = lit;
        },
        .Range => |range| {
            scanExpr(range.start, usage);
            scanExpr(range.end, usage);
        },
        .EnumValue => {},
        .Else => {},
    }
}

fn noteAssignmentTarget(target: *const lib.ast.Expressions.ExprNode, usage: *std.StringHashMap(Usage)) void {
    if (getRootIdentifierName(target)) |name| {
        var entry = usage.get(name) orelse Usage{};
        entry.def_count += 1;
        entry.has_lvalue_use = true;
        usage.put(name, entry) catch {};
    }
}

fn markLValue(name: []const u8, usage: *std.StringHashMap(Usage)) void {
    var entry = usage.get(name) orelse Usage{};
    entry.def_count += 1;
    entry.has_lvalue_use = true;
    usage.put(name, entry) catch {};
}

fn getRootIdentifierName(target: *const lib.ast.Expressions.ExprNode) ?[]const u8 {
    return switch (target.*) {
        .Identifier => |ident| ident.name,
        .FieldAccess => |fa| getRootIdentifierName(fa.target),
        .Index => |idx| getRootIdentifierName(idx.target),
        else => null,
    };
}
