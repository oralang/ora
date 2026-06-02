//! Generic walkers over AST bodies, statements, expressions, and patterns.
//!
//! The walker is intentionally structural: it does not resolve scopes,
//! infer types, or interpret bindings. Stage-specific passes provide a
//! small visitor type with optional `enter*` / `exit*` hooks and choose
//! traversal options for the few boundaries where existing compiler
//! passes intentionally differ.

const std = @import("std");
const source = @import("../source/mod.zig");
const file_mod = @import("file.zig");
const nodes = @import("nodes.zig");
const ids = @import("ids.zig");

const AstFile = file_mod.AstFile;
const BodyId = ids.BodyId;
const ExprId = ids.ExprId;
const PatternId = ids.PatternId;
const StmtId = ids.StmtId;

pub const WalkControl = enum {
    descend,
    skip_children,
    stop,
};

pub const WalkError = error{
    WalkStopped,
};

pub const WalkOptions = struct {
    /// Walk expressions inside switch patterns. Some legacy expression-only
    /// callers intentionally consider only switch condition/value trees.
    walk_switch_patterns: bool = true,

    /// Walk assignment targets. Pattern targets can contain index expressions.
    walk_assignment_target_patterns: bool = true,

    /// Walk pattern sub-bindings such as destructure fields and ok/err
    /// bindings. Passes that only care about value expressions can disable it.
    walk_pattern_bindings: bool = true,

    /// Enter statement-bearing comptime expression bodies.
    enter_comptime_bodies: bool = false,

    /// Enter quantified expression condition/body. The quantified pattern and
    /// type expression are not value-expression trees and are never walked here.
    enter_quantified_bodies: bool = false,

    /// Walk the address/gas expressions inside external proxies.
    walk_external_proxy_exprs: bool = true,

    /// Walk error-constructor arguments.
    walk_error_return_args: bool = true,

    /// Walk the operand inside `old(...)` verification expressions.
    walk_old_exprs: bool = true,
};

pub fn walkBody(
    comptime Visitor: type,
    visitor: *Visitor,
    ast_file: *const AstFile,
    body_id: BodyId,
    comptime options: WalkOptions,
) anyerror!void {
    switch (try enterBody(Visitor, visitor, ast_file, body_id)) {
        .descend => {},
        .skip_children => {
            try exitBody(Visitor, visitor, ast_file, body_id);
            return;
        },
        .stop => return WalkError.WalkStopped,
    }

    const body = ast_file.body(body_id).*;
    for (body.statements) |statement_id| {
        try walkStmt(Visitor, visitor, ast_file, statement_id, options);
    }

    try exitBody(Visitor, visitor, ast_file, body_id);
}

pub fn walkStmt(
    comptime Visitor: type,
    visitor: *Visitor,
    ast_file: *const AstFile,
    statement_id: StmtId,
    comptime options: WalkOptions,
) anyerror!void {
    switch (try enterStmt(Visitor, visitor, ast_file, statement_id)) {
        .descend => {},
        .skip_children => {
            try exitStmt(Visitor, visitor, ast_file, statement_id);
            return;
        },
        .stop => return WalkError.WalkStopped,
    }

    switch (ast_file.statement(statement_id).*) {
        .VariableDecl => |decl| if (decl.value) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options),
        .Return => |ret| if (ret.value) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options),
        .If => |if_stmt| {
            try walkExpr(Visitor, visitor, ast_file, if_stmt.condition, options);
            try walkBody(Visitor, visitor, ast_file, if_stmt.then_body, options);
            if (if_stmt.else_body) |else_body| try walkBody(Visitor, visitor, ast_file, else_body, options);
        },
        .While => |while_stmt| {
            try walkExpr(Visitor, visitor, ast_file, while_stmt.condition, options);
            for (while_stmt.invariants) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options);
            try walkBody(Visitor, visitor, ast_file, while_stmt.body, options);
        },
        .For => |for_stmt| {
            try walkExpr(Visitor, visitor, ast_file, for_stmt.iterable, options);
            if (for_stmt.range_end) |end_expr| try walkExpr(Visitor, visitor, ast_file, end_expr, options);
            for (for_stmt.invariants) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options);
            try walkBody(Visitor, visitor, ast_file, for_stmt.body, options);
        },
        .Switch => |switch_stmt| {
            try walkExpr(Visitor, visitor, ast_file, switch_stmt.condition, options);
            for (switch_stmt.arms) |arm| {
                if (options.walk_switch_patterns) try walkSwitchPattern(Visitor, visitor, ast_file, arm.pattern, options);
                try walkBody(Visitor, visitor, ast_file, arm.body, options);
            }
            if (switch_stmt.else_body) |else_body| try walkBody(Visitor, visitor, ast_file, else_body, options);
        },
        .Try => |try_stmt| {
            try walkBody(Visitor, visitor, ast_file, try_stmt.try_body, options);
            if (try_stmt.catch_clause) |catch_clause| try walkBody(Visitor, visitor, ast_file, catch_clause.body, options);
        },
        .Log => |log_stmt| for (log_stmt.args) |arg| try walkExpr(Visitor, visitor, ast_file, arg, options),
        .Lock => |lock_stmt| try walkExpr(Visitor, visitor, ast_file, lock_stmt.path, options),
        .Unlock => |unlock_stmt| try walkExpr(Visitor, visitor, ast_file, unlock_stmt.path, options),
        .Assert => |assert_stmt| try walkExpr(Visitor, visitor, ast_file, assert_stmt.condition, options),
        .Assume => |assume_stmt| try walkExpr(Visitor, visitor, ast_file, assume_stmt.condition, options),
        .Assign => |assign| {
            try walkExpr(Visitor, visitor, ast_file, assign.value, options);
            if (options.walk_assignment_target_patterns) try walkPattern(Visitor, visitor, ast_file, assign.target, options);
        },
        .Expr => |expr_stmt| try walkExpr(Visitor, visitor, ast_file, expr_stmt.expr, options),
        .Block => |block| try walkBody(Visitor, visitor, ast_file, block.body, options),
        .LabeledBlock => |block| try walkBody(Visitor, visitor, ast_file, block.body, options),
        .Break => |jump| if (jump.value) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options),
        .Continue => |jump| if (jump.value) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options),
        .Havoc, .Error => {},
    }

    try exitStmt(Visitor, visitor, ast_file, statement_id);
}

pub fn walkExpr(
    comptime Visitor: type,
    visitor: *Visitor,
    ast_file: *const AstFile,
    expr_id: ExprId,
    comptime options: WalkOptions,
) anyerror!void {
    switch (try enterExpr(Visitor, visitor, ast_file, expr_id)) {
        .descend => {},
        .skip_children => {
            try exitExpr(Visitor, visitor, ast_file, expr_id);
            return;
        },
        .stop => return WalkError.WalkStopped,
    }

    switch (ast_file.expression(expr_id).*) {
        .Tuple => |tuple| for (tuple.elements) |element| try walkExpr(Visitor, visitor, ast_file, element, options),
        .ArrayLiteral => |array| for (array.elements) |element| try walkExpr(Visitor, visitor, ast_file, element, options),
        .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try walkExpr(Visitor, visitor, ast_file, field.value, options),
        .Switch => |switch_expr| {
            try walkExpr(Visitor, visitor, ast_file, switch_expr.condition, options);
            for (switch_expr.arms) |arm| {
                if (options.walk_switch_patterns) try walkSwitchPattern(Visitor, visitor, ast_file, arm.pattern, options);
                try walkExpr(Visitor, visitor, ast_file, arm.value, options);
            }
            if (switch_expr.else_expr) |else_expr| try walkExpr(Visitor, visitor, ast_file, else_expr, options);
        },
        .ExternalProxy => |external_proxy| if (options.walk_external_proxy_exprs) {
            try walkExpr(Visitor, visitor, ast_file, external_proxy.address_expr, options);
            try walkExpr(Visitor, visitor, ast_file, external_proxy.gas_expr, options);
        },
        .Comptime => |comptime_expr| if (options.enter_comptime_bodies) {
            try walkBody(Visitor, visitor, ast_file, comptime_expr.body, options);
        },
        .ErrorReturn => |error_return| if (options.walk_error_return_args) {
            for (error_return.args) |arg| try walkExpr(Visitor, visitor, ast_file, arg, options);
        },
        .Unary => |unary| try walkExpr(Visitor, visitor, ast_file, unary.operand, options),
        .Binary => |binary| {
            try walkExpr(Visitor, visitor, ast_file, binary.lhs, options);
            try walkExpr(Visitor, visitor, ast_file, binary.rhs, options);
        },
        .Call => |call| {
            try walkExpr(Visitor, visitor, ast_file, call.callee, options);
            for (call.args) |arg| try walkExpr(Visitor, visitor, ast_file, arg, options);
        },
        .Builtin => |builtin| for (builtin.args) |arg| try walkExpr(Visitor, visitor, ast_file, arg, options),
        .Field => |field| try walkExpr(Visitor, visitor, ast_file, field.base, options),
        .Index => |index| {
            try walkExpr(Visitor, visitor, ast_file, index.base, options);
            try walkExpr(Visitor, visitor, ast_file, index.index, options);
        },
        .Group => |group| try walkExpr(Visitor, visitor, ast_file, group.expr, options),
        .Old => |old| if (options.walk_old_exprs) try walkExpr(Visitor, visitor, ast_file, old.expr, options),
        .Quantified => |quantified| if (options.enter_quantified_bodies) {
            if (quantified.condition) |condition| try walkExpr(Visitor, visitor, ast_file, condition, options);
            try walkExpr(Visitor, visitor, ast_file, quantified.body, options);
        },
        .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .TypeValue, .Name, .Result, .Error => {},
    }

    try exitExpr(Visitor, visitor, ast_file, expr_id);
}

pub fn walkPattern(
    comptime Visitor: type,
    visitor: *Visitor,
    ast_file: *const AstFile,
    pattern_id: PatternId,
    comptime options: WalkOptions,
) anyerror!void {
    switch (try enterPattern(Visitor, visitor, ast_file, pattern_id)) {
        .descend => {},
        .skip_children => {
            try exitPattern(Visitor, visitor, ast_file, pattern_id);
            return;
        },
        .stop => return WalkError.WalkStopped,
    }

    switch (ast_file.pattern(pattern_id).*) {
        .Field => |field| try walkPattern(Visitor, visitor, ast_file, field.base, options),
        .Index => |index| {
            try walkPattern(Visitor, visitor, ast_file, index.base, options);
            try walkExpr(Visitor, visitor, ast_file, index.index, options);
        },
        .StructDestructure => |destructure| if (options.walk_pattern_bindings) {
            for (destructure.fields) |field| try walkPattern(Visitor, visitor, ast_file, field.binding, options);
        },
        .Name, .Error => {},
    }

    try exitPattern(Visitor, visitor, ast_file, pattern_id);
}

pub fn walkSwitchPattern(
    comptime Visitor: type,
    visitor: *Visitor,
    ast_file: *const AstFile,
    pattern: nodes.SwitchPattern,
    comptime options: WalkOptions,
) anyerror!void {
    switch (try enterSwitchPattern(Visitor, visitor, ast_file, pattern)) {
        .descend => {},
        .skip_children => {
            try exitSwitchPattern(Visitor, visitor, ast_file, pattern);
            return;
        },
        .stop => return WalkError.WalkStopped,
    }

    switch (pattern) {
        .Expr => |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options),
        .Range => |range_pattern| {
            try walkExpr(Visitor, visitor, ast_file, range_pattern.start, options);
            try walkExpr(Visitor, visitor, ast_file, range_pattern.end, options);
        },
        .Or => |or_pattern| for (or_pattern.alternatives) |alternative| try walkSwitchPattern(Visitor, visitor, ast_file, alternative, options),
        .Ok => |pattern_id| if (options.walk_pattern_bindings) try walkPattern(Visitor, visitor, ast_file, pattern_id, options),
        .Err => |pattern_id| if (options.walk_pattern_bindings) try walkPattern(Visitor, visitor, ast_file, pattern_id, options),
        .NamedError => |named_error| {
            try walkExpr(Visitor, visitor, ast_file, named_error.callee, options);
            if (options.walk_pattern_bindings) {
                for (named_error.bindings) |pattern_id| try walkPattern(Visitor, visitor, ast_file, pattern_id, options);
            }
        },
        .Else => {},
    }

    try exitSwitchPattern(Visitor, visitor, ast_file, pattern);
}

fn enterBody(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, body_id: BodyId) anyerror!WalkControl {
    if (@hasDecl(Visitor, "enterBody")) return visitor.enterBody(ast_file, body_id);
    return .descend;
}

fn exitBody(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, body_id: BodyId) anyerror!void {
    if (@hasDecl(Visitor, "exitBody")) try visitor.exitBody(ast_file, body_id);
}

fn enterStmt(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, statement_id: StmtId) anyerror!WalkControl {
    if (@hasDecl(Visitor, "enterStmt")) return visitor.enterStmt(ast_file, statement_id);
    return .descend;
}

fn exitStmt(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, statement_id: StmtId) anyerror!void {
    if (@hasDecl(Visitor, "exitStmt")) try visitor.exitStmt(ast_file, statement_id);
}

fn enterExpr(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, expr_id: ExprId) anyerror!WalkControl {
    if (@hasDecl(Visitor, "enterExpr")) return visitor.enterExpr(ast_file, expr_id);
    return .descend;
}

fn exitExpr(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, expr_id: ExprId) anyerror!void {
    if (@hasDecl(Visitor, "exitExpr")) try visitor.exitExpr(ast_file, expr_id);
}

fn enterPattern(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, pattern_id: PatternId) anyerror!WalkControl {
    if (@hasDecl(Visitor, "enterPattern")) return visitor.enterPattern(ast_file, pattern_id);
    return .descend;
}

fn exitPattern(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, pattern_id: PatternId) anyerror!void {
    if (@hasDecl(Visitor, "exitPattern")) try visitor.exitPattern(ast_file, pattern_id);
}

fn enterSwitchPattern(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, pattern: nodes.SwitchPattern) anyerror!WalkControl {
    if (@hasDecl(Visitor, "enterSwitchPattern")) return visitor.enterSwitchPattern(ast_file, pattern);
    return .descend;
}

fn exitSwitchPattern(comptime Visitor: type, visitor: *Visitor, ast_file: *const AstFile, pattern: nodes.SwitchPattern) anyerror!void {
    if (@hasDecl(Visitor, "exitSwitchPattern")) try visitor.exitSwitchPattern(ast_file, pattern);
}

/// Append every `Name` reference reachable from `expr_id` to `out`.
/// Names are borrowed from the AST — caller must not free them.
pub fn collectNamesInExpr(
    allocator: std.mem.Allocator,
    ast_file: *const AstFile,
    expr_id: ExprId,
    out: *std.ArrayList([]const u8),
) !void {
    var visitor = CollectNamesVisitor{
        .allocator = allocator,
        .out = out,
    };
    try walkExpr(CollectNamesVisitor, &visitor, ast_file, expr_id, .{
        .walk_switch_patterns = false,
        .enter_comptime_bodies = false,
        .enter_quantified_bodies = false,
    });
}

const CollectNamesVisitor = struct {
    allocator: std.mem.Allocator,
    out: *std.ArrayList([]const u8),

    pub fn enterExpr(self: *@This(), ast_file: *const AstFile, expr_id: ExprId) anyerror!WalkControl {
        switch (ast_file.expression(expr_id).*) {
            .Name => |name| try self.out.append(self.allocator, name.name),
            .Quantified => return .skip_children,
            else => {},
        }
        return .descend;
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "collectNamesInExpr type check" {
    // No-op type-level assertion: ensures the file compiles and
    // the function is referenced from the test target. AST-driven
    // coverage lives in compiler tests that already build full
    // ASTs; reproducing that machinery here would be wasteful.
    const fn_ptr: *const fn (
        std.mem.Allocator,
        *const AstFile,
        ExprId,
        *std.ArrayList([]const u8),
    ) anyerror!void = collectNamesInExpr;
    _ = fn_ptr;
    try testing.expect(true);
}

test "walkExpr switch pattern option preserves legacy name collection" {
    const r = source.TextRange.empty(0);

    var arms = [_]nodes.SwitchExprArm{.{
        .range = r,
        .pattern = .{ .Expr = ExprId.fromIndex(1) },
        .value = ExprId.fromIndex(2),
    }};
    var expressions = [_]nodes.Expr{
        .{ .Name = .{ .range = r, .name = "condition" } },
        .{ .Name = .{ .range = r, .name = "pattern" } },
        .{ .Name = .{ .range = r, .name = "value" } },
        .{ .Switch = .{
            .range = r,
            .condition = ExprId.fromIndex(0),
            .arms = arms[0..],
            .else_expr = null,
        } },
    };
    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var bodies: [0]nodes.Body = .{};
    var statements: [0]nodes.Stmt = .{};
    var type_exprs: [0]nodes.TypeExpr = .{};
    var patterns: [0]nodes.Pattern = .{};
    var ast_file = AstFile{
        .arena = std.heap.ArenaAllocator.init(testing.allocator),
        .file_id = source.FileId.fromIndex(0),
        .root_items = root_items[0..],
        .items = items[0..],
        .bodies = bodies[0..],
        .statements = statements[0..],
        .expressions = expressions[0..],
        .type_exprs = type_exprs[0..],
        .patterns = patterns[0..],
    };
    defer ast_file.deinit();

    var legacy_names: std.ArrayList([]const u8) = .{};
    defer legacy_names.deinit(testing.allocator);
    try collectNamesInExpr(testing.allocator, &ast_file, ExprId.fromIndex(3), &legacy_names);
    try testing.expectEqual(@as(usize, 2), legacy_names.items.len);
    try testing.expectEqualStrings("condition", legacy_names.items[0]);
    try testing.expectEqualStrings("value", legacy_names.items[1]);

    var walked_names: std.ArrayList([]const u8) = .{};
    defer walked_names.deinit(testing.allocator);
    var visitor = CollectNamesVisitor{
        .allocator = testing.allocator,
        .out = &walked_names,
    };
    try walkExpr(CollectNamesVisitor, &visitor, &ast_file, ExprId.fromIndex(3), .{});
    try testing.expectEqual(@as(usize, 3), walked_names.items.len);
    try testing.expectEqualStrings("condition", walked_names.items[0]);
    try testing.expectEqualStrings("pattern", walked_names.items[1]);
    try testing.expectEqualStrings("value", walked_names.items[2]);
}
