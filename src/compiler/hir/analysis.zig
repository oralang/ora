const std = @import("std");
const ast = @import("../ast/mod.zig");
const hir_locals = @import("locals.zig");

const LocalEnv = hir_locals.LocalEnv;
const LocalIdList = hir_locals.LocalIdList;
const LocalIdSet = hir_locals.LocalIdSet;

pub fn bodyMayReturn(file: *const ast.AstFile, body_id: ast.BodyId) bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (stmtMayReturn(file, statement_id)) return true;
    }
    return false;
}

pub fn bodyContainsLoopControl(file: *const ast.AstFile, body_id: ast.BodyId) bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (stmtContainsLoopControl(file, statement_id)) return true;
    }
    return false;
}

pub fn bodyContainsStructuredLoopControl(file: *const ast.AstFile, body_id: ast.BodyId) bool {
    return bodyContainsLoopControlInContext(file, body_id, false);
}

pub fn collectLoopCarriedLocals(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    outer_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    var current_env = try outer_env.clone();
    return collectCarriedLocalsInEnv(allocator, file, body_id, outer_env, &current_env, carried_locals, carried_seen);
}

pub fn collectIfCarriedLocals(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    outer_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    var current_env = try outer_env.clone();
    return collectCarriedLocalsInEnv(allocator, file, body_id, outer_env, &current_env, carried_locals, carried_seen);
}

pub fn collectTryCarriedLocals(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    try_stmt: ast.TryStmt,
    outer_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    var current_env = try outer_env.clone();
    return collectTryCarriedLocalsFromEnv(allocator, file, try_stmt, outer_env, &current_env, carried_locals, carried_seen);
}

pub fn collectSwitchCarriedLocals(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    switch_stmt: ast.SwitchStmt,
    outer_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    var current_env = try outer_env.clone();
    return collectSwitchCarriedLocalsFromEnv(allocator, file, switch_stmt, outer_env, &current_env, carried_locals, carried_seen);
}

fn collectTryCarriedLocalsFromEnv(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    try_stmt: ast.TryStmt,
    outer_env: *const LocalEnv,
    current_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    var try_env = try current_env.clone();
    if (!try collectCarriedLocalsInEnv(allocator, file, try_stmt.try_body, outer_env, &try_env, carried_locals, carried_seen)) {
        return false;
    }

    if (try_stmt.catch_clause) |catch_clause| {
        var catch_env = try current_env.clone();
        if (catch_clause.error_pattern) |pattern_id| {
            try catch_env.bindPatternWithoutValue(file, pattern_id);
        }
        if (!try collectCarriedLocalsInEnv(allocator, file, catch_clause.body, outer_env, &catch_env, carried_locals, carried_seen)) {
            return false;
        }
    }

    return true;
}

fn collectSwitchCarriedLocalsFromEnv(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    switch_stmt: ast.SwitchStmt,
    outer_env: *const LocalEnv,
    current_env: *const LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    for (switch_stmt.arms) |arm| {
        if (bodyContainsSwitchBreak(file, arm.body)) return false;

        var arm_env = try current_env.clone();
        if (!try collectCarriedLocalsInEnv(allocator, file, arm.body, outer_env, &arm_env, carried_locals, carried_seen)) {
            return false;
        }
    }

    if (switch_stmt.else_body) |else_body| {
        if (bodyContainsSwitchBreak(file, else_body)) return false;

        var else_env = try current_env.clone();
        if (!try collectCarriedLocalsInEnv(allocator, file, else_body, outer_env, &else_env, carried_locals, carried_seen)) {
            return false;
        }
    }

    return true;
}

pub fn switchMayReturn(file: *const ast.AstFile, switch_stmt: ast.SwitchStmt) bool {
    for (switch_stmt.arms) |arm| {
        if (bodyMayReturn(file, arm.body)) return true;
    }
    if (switch_stmt.else_body) |else_body| {
        return bodyMayReturn(file, else_body);
    }
    return false;
}

pub fn bodyContainsSwitchBreak(file: *const ast.AstFile, body_id: ast.BodyId) bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (stmtContainsSwitchBreak(file, statement_id)) return true;
    }
    return false;
}

fn collectCarriedLocalsInEnv(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    outer_env: *const LocalEnv,
    current_env: *LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (!try collectCarriedLocalsFromStmt(allocator, file, statement_id, outer_env, current_env, carried_locals, carried_seen)) {
            return false;
        }
    }
    return true;
}

fn collectCarriedLocalsFromStmt(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    statement_id: ast.StmtId,
    outer_env: *const LocalEnv,
    current_env: *LocalEnv,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) anyerror!bool {
    return switch (file.statement(statement_id).*) {
        .VariableDecl => |decl| blk: {
            try current_env.bindPatternWithoutValue(file, decl.pattern);
            break :blk true;
        },
        .Assign => |assign| blk: {
            try recordCarriedLocal(allocator, outer_env, current_env.resolvePatternTarget(file, assign.target), carried_locals, carried_seen);
            break :blk true;
        },
        .Havoc => |havoc_stmt| blk: {
            try recordCarriedLocal(allocator, outer_env, current_env.lookupName(havoc_stmt.name), carried_locals, carried_seen);
            break :blk true;
        },
        .Block => |block_stmt| blk: {
            var block_env = try current_env.clone();
            break :blk collectCarriedLocalsInEnv(allocator, file, block_stmt.body, outer_env, &block_env, carried_locals, carried_seen);
        },
        .LabeledBlock => |block_stmt| blk: {
            var block_env = try current_env.clone();
            break :blk collectCarriedLocalsInEnv(allocator, file, block_stmt.body, outer_env, &block_env, carried_locals, carried_seen);
        },
        .If => |if_stmt| blk: {
            var then_env = try current_env.clone();
            if (!try collectCarriedLocalsInEnv(allocator, file, if_stmt.then_body, outer_env, &then_env, carried_locals, carried_seen)) {
                break :blk false;
            }
            if (if_stmt.else_body) |else_body| {
                var else_env = try current_env.clone();
                if (!try collectCarriedLocalsInEnv(allocator, file, else_body, outer_env, &else_env, carried_locals, carried_seen)) {
                    break :blk false;
                }
            }
            break :blk true;
        },
        .Switch => |switch_stmt| collectSwitchCarriedLocalsFromEnv(allocator, file, switch_stmt, outer_env, current_env, carried_locals, carried_seen),
        .Try => |try_stmt| collectTryCarriedLocalsFromEnv(allocator, file, try_stmt, outer_env, current_env, carried_locals, carried_seen),
        .While => |while_stmt| blk: {
            var loop_env = try current_env.clone();
            if (bodyMayReturn(file, while_stmt.body) or bodyContainsLoopControl(file, while_stmt.body)) {
                break :blk !(try bodyMutatesOuterLocalsInEnv(file, while_stmt.body, outer_env, &loop_env));
            }
            break :blk collectCarriedLocalsInEnv(allocator, file, while_stmt.body, outer_env, &loop_env, carried_locals, carried_seen);
        },
        .For => |for_stmt| blk: {
            var loop_env = try current_env.clone();
            try loop_env.bindPatternWithoutValue(file, for_stmt.item_pattern);
            if (for_stmt.index_pattern) |index_pattern| {
                try loop_env.bindPatternWithoutValue(file, index_pattern);
            }
            break :blk !(try bodyMutatesOuterLocalsInEnv(file, for_stmt.body, outer_env, &loop_env));
        },
        else => true,
    };
}

fn bodyMutatesOuterLocalsInEnv(
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    outer_env: *const LocalEnv,
    current_env: *LocalEnv,
) anyerror!bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (try stmtMutatesOuterLocals(file, statement_id, outer_env, current_env)) return true;
    }
    return false;
}

fn stmtMutatesOuterLocals(
    file: *const ast.AstFile,
    statement_id: ast.StmtId,
    outer_env: *const LocalEnv,
    current_env: *LocalEnv,
) anyerror!bool {
    return switch (file.statement(statement_id).*) {
        .VariableDecl => |decl| blk: {
            try current_env.bindPatternWithoutValue(file, decl.pattern);
            break :blk false;
        },
        .Assign => |assign| if (current_env.resolvePatternTarget(file, assign.target)) |local_id|
            outer_env.hasLocal(local_id)
        else
            false,
        .Havoc => |havoc_stmt| if (current_env.lookupName(havoc_stmt.name)) |local_id|
            outer_env.hasLocal(local_id)
        else
            false,
        .Block => |block_stmt| blk: {
            var block_env = try current_env.clone();
            break :blk bodyMutatesOuterLocalsInEnv(file, block_stmt.body, outer_env, &block_env);
        },
        .LabeledBlock => |block_stmt| blk: {
            var block_env = try current_env.clone();
            break :blk bodyMutatesOuterLocalsInEnv(file, block_stmt.body, outer_env, &block_env);
        },
        .If => |if_stmt| blk: {
            var then_env = try current_env.clone();
            if (try bodyMutatesOuterLocalsInEnv(file, if_stmt.then_body, outer_env, &then_env)) break :blk true;
            if (if_stmt.else_body) |else_body| {
                var else_env = try current_env.clone();
                break :blk try bodyMutatesOuterLocalsInEnv(file, else_body, outer_env, &else_env);
            }
            break :blk false;
        },
        .While => |while_stmt| blk: {
            var loop_env = try current_env.clone();
            break :blk try bodyMutatesOuterLocalsInEnv(file, while_stmt.body, outer_env, &loop_env);
        },
        .For => |for_stmt| blk: {
            var loop_env = try current_env.clone();
            try loop_env.bindPatternWithoutValue(file, for_stmt.item_pattern);
            if (for_stmt.index_pattern) |index_pattern| {
                try loop_env.bindPatternWithoutValue(file, index_pattern);
            }
            break :blk try bodyMutatesOuterLocalsInEnv(file, for_stmt.body, outer_env, &loop_env);
        },
        .Switch => |switch_stmt| blk: {
            for (switch_stmt.arms) |arm| {
                var arm_env = try current_env.clone();
                if (try bodyMutatesOuterLocalsInEnv(file, arm.body, outer_env, &arm_env)) break :blk true;
            }
            if (switch_stmt.else_body) |else_body| {
                var else_env = try current_env.clone();
                break :blk try bodyMutatesOuterLocalsInEnv(file, else_body, outer_env, &else_env);
            }
            break :blk false;
        },
        .Try => |try_stmt| blk: {
            var try_env = try current_env.clone();
            if (try bodyMutatesOuterLocalsInEnv(file, try_stmt.try_body, outer_env, &try_env)) break :blk true;
            if (try_stmt.catch_clause) |catch_clause| {
                var catch_env = try current_env.clone();
                if (catch_clause.error_pattern) |pattern_id| {
                    try catch_env.bindPatternWithoutValue(file, pattern_id);
                }
                break :blk try bodyMutatesOuterLocalsInEnv(file, catch_clause.body, outer_env, &catch_env);
            }
            break :blk false;
        },
        else => false,
    };
}

fn recordCarriedLocal(
    allocator: std.mem.Allocator,
    outer_env: *const LocalEnv,
    maybe_local_id: ?hir_locals.LocalId,
    carried_locals: *LocalIdList,
    carried_seen: *LocalIdSet,
) !void {
    const local_id = maybe_local_id orelse return;
    if (!outer_env.hasLocal(local_id)) return;
    if (carried_seen.contains(local_id)) return;

    try carried_seen.put(local_id, {});
    try carried_locals.append(allocator, local_id);
}

fn stmtMayReturn(file: *const ast.AstFile, statement_id: ast.StmtId) bool {
    return switch (file.statement(statement_id).*) {
        .Return => true,
        .If => |if_stmt| bodyMayReturn(file, if_stmt.then_body) or
            (if_stmt.else_body != null and bodyMayReturn(file, if_stmt.else_body.?)),
        .While => |while_stmt| bodyMayReturn(file, while_stmt.body),
        .For => |for_stmt| bodyMayReturn(file, for_stmt.body),
        .Switch => |switch_stmt| switchMayReturn(file, switch_stmt),
        .Try => |try_stmt| bodyMayReturn(file, try_stmt.try_body) or
            (try_stmt.catch_clause != null and bodyMayReturn(file, try_stmt.catch_clause.?.body)),
        .Block => |block_stmt| bodyMayReturn(file, block_stmt.body),
        .LabeledBlock => |block_stmt| bodyMayReturn(file, block_stmt.body),
        else => false,
    };
}

fn stmtContainsSwitchBreak(file: *const ast.AstFile, statement_id: ast.StmtId) bool {
    return stmtContainsSwitchBreakInContext(file, statement_id, false);
}

fn stmtContainsSwitchBreakInContext(file: *const ast.AstFile, statement_id: ast.StmtId, nested_break_scope: bool) bool {
    return switch (file.statement(statement_id).*) {
        .Break => !nested_break_scope,
        .If => |if_stmt| stmtBodyContainsSwitchBreakInContext(file, if_stmt.then_body, nested_break_scope) or
            (if_stmt.else_body != null and stmtBodyContainsSwitchBreakInContext(file, if_stmt.else_body.?, nested_break_scope)),
        .While => |while_stmt| stmtBodyContainsSwitchBreakInContext(file, while_stmt.body, true),
        .For => |for_stmt| stmtBodyContainsSwitchBreakInContext(file, for_stmt.body, true),
        .Switch => |switch_stmt| blk: {
            for (switch_stmt.arms) |arm| {
                if (stmtBodyContainsSwitchBreakInContext(file, arm.body, true)) break :blk true;
            }
            if (switch_stmt.else_body) |else_body| break :blk stmtBodyContainsSwitchBreakInContext(file, else_body, true);
            break :blk false;
        },
        .Try => |try_stmt| stmtBodyContainsSwitchBreakInContext(file, try_stmt.try_body, nested_break_scope) or
            (try_stmt.catch_clause != null and stmtBodyContainsSwitchBreakInContext(file, try_stmt.catch_clause.?.body, nested_break_scope)),
        .Block => |block_stmt| stmtBodyContainsSwitchBreakInContext(file, block_stmt.body, nested_break_scope),
        .LabeledBlock => |block_stmt| stmtBodyContainsSwitchBreakInContext(file, block_stmt.body, nested_break_scope),
        else => false,
    };
}

fn stmtBodyContainsSwitchBreakInContext(file: *const ast.AstFile, body_id: ast.BodyId, nested_break_scope: bool) bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (stmtContainsSwitchBreakInContext(file, statement_id, nested_break_scope)) return true;
    }
    return false;
}

fn stmtContainsLoopControl(file: *const ast.AstFile, statement_id: ast.StmtId) bool {
    return switch (file.statement(statement_id).*) {
        .Break, .Continue => true,
        .If => |if_stmt| bodyContainsLoopControl(file, if_stmt.then_body) or
            (if_stmt.else_body != null and bodyContainsLoopControl(file, if_stmt.else_body.?)),
        .While => |while_stmt| bodyContainsLoopControl(file, while_stmt.body),
        .For => |for_stmt| bodyContainsLoopControl(file, for_stmt.body),
        .Switch => |switch_stmt| blk: {
            for (switch_stmt.arms) |arm| {
                if (bodyContainsLoopControl(file, arm.body)) break :blk true;
            }
            if (switch_stmt.else_body) |else_body| break :blk bodyContainsLoopControl(file, else_body);
            break :blk false;
        },
        .Try => |try_stmt| bodyContainsLoopControl(file, try_stmt.try_body) or
            (try_stmt.catch_clause != null and bodyContainsLoopControl(file, try_stmt.catch_clause.?.body)),
        .Block => |block_stmt| bodyContainsLoopControl(file, block_stmt.body),
        .LabeledBlock => |block_stmt| bodyContainsLoopControl(file, block_stmt.body),
        else => false,
    };
}

fn bodyContainsLoopControlInContext(file: *const ast.AstFile, body_id: ast.BodyId, nested: bool) bool {
    const body = file.body(body_id).*;
    for (body.statements) |statement_id| {
        if (stmtContainsLoopControlInContext(file, statement_id, nested)) return true;
    }
    return false;
}

fn stmtContainsLoopControlInContext(file: *const ast.AstFile, statement_id: ast.StmtId, nested: bool) bool {
    return switch (file.statement(statement_id).*) {
        .Break, .Continue => nested,
        .If => |if_stmt| bodyContainsLoopControlInContext(file, if_stmt.then_body, nested) or
            (if_stmt.else_body != null and bodyContainsLoopControlInContext(file, if_stmt.else_body.?, nested)),
        .While => |while_stmt| bodyContainsLoopControlInContext(file, while_stmt.body, false),
        .For => |for_stmt| bodyContainsLoopControlInContext(file, for_stmt.body, false),
        .Switch => |switch_stmt| blk: {
            for (switch_stmt.arms) |arm| {
                if (bodyContainsLoopControlInContext(file, arm.body, nested)) break :blk true;
            }
            if (switch_stmt.else_body) |else_body| break :blk bodyContainsLoopControlInContext(file, else_body, nested);
            break :blk false;
        },
        .Try => |try_stmt| bodyContainsLoopControlInContext(file, try_stmt.try_body, nested) or
            (try_stmt.catch_clause != null and bodyContainsLoopControlInContext(file, try_stmt.catch_clause.?.body, nested)),
        .Block => |block_stmt| bodyContainsLoopControlInContext(file, block_stmt.body, nested),
        .LabeledBlock => |block_stmt| bodyContainsLoopControlInContext(file, block_stmt.body, nested),
        else => false,
    };
}
