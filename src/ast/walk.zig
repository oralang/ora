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
            for (switch_stmt.invariants) |expr_id| try walkExpr(Visitor, visitor, ast_file, expr_id, options);
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
        .CallHint => {},
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

const TestEventKind = enum(u8) {
    body,
    stmt,
    expr,
    pattern,
    switch_pattern,
    exit_expr,
};

const TestSwitchPatternTag = enum(u8) {
    expr,
    range,
    or_,
    ok,
    err,
    named_error,
    else_,
};

fn bid(index: usize) BodyId {
    return BodyId.fromIndex(index);
}

fn sid(index: usize) StmtId {
    return StmtId.fromIndex(index);
}

fn eid(index: usize) ExprId {
    return ExprId.fromIndex(index);
}

fn pid(index: usize) PatternId {
    return PatternId.fromIndex(index);
}

fn testEvent(kind: TestEventKind, value: usize) u32 {
    return (@as(u32, @intFromEnum(kind)) << 24) | @as(u32, @intCast(value));
}

fn switchPatternTag(pattern: nodes.SwitchPattern) TestSwitchPatternTag {
    return switch (pattern) {
        .Expr => .expr,
        .Range => .range,
        .Or => .or_,
        .Ok => .ok,
        .Err => .err,
        .NamedError => .named_error,
        .Else => .else_,
    };
}

fn spEvent(tag: TestSwitchPatternTag) u32 {
    return testEvent(.switch_pattern, @intFromEnum(tag));
}

fn initTestAstFile(
    root_items: []ids.ItemId,
    items: []nodes.Item,
    bodies: []nodes.Body,
    statements: []nodes.Stmt,
    expressions: []nodes.Expr,
    type_exprs: []nodes.TypeExpr,
    patterns: []nodes.Pattern,
) AstFile {
    return .{
        .arena = std.heap.ArenaAllocator.init(testing.allocator),
        .file_id = source.FileId.fromIndex(0),
        .root_items = root_items,
        .items = items,
        .bodies = bodies,
        .statements = statements,
        .expressions = expressions,
        .type_exprs = type_exprs,
        .patterns = patterns,
    };
}

fn fillNameExprs(expressions: []nodes.Expr, range: source.TextRange) void {
    for (expressions) |*expr| {
        expr.* = .{ .Name = .{ .range = range, .name = "leaf" } };
    }
}

const EnterOrderVisitor = struct {
    allocator: std.mem.Allocator,
    events: *std.ArrayList(u32),

    fn record(self: *@This(), kind: TestEventKind, value: usize) !void {
        try self.events.append(self.allocator, testEvent(kind, value));
    }

    pub fn enterBody(self: *@This(), file: *const AstFile, body_id: BodyId) !WalkControl {
        _ = file;
        try self.record(.body, body_id.index());
        return .descend;
    }

    pub fn enterStmt(self: *@This(), file: *const AstFile, statement_id: StmtId) !WalkControl {
        _ = file;
        try self.record(.stmt, statement_id.index());
        return .descend;
    }

    pub fn enterExpr(self: *@This(), file: *const AstFile, expr_id: ExprId) !WalkControl {
        _ = file;
        try self.record(.expr, expr_id.index());
        return .descend;
    }

    pub fn enterPattern(self: *@This(), file: *const AstFile, pattern_id: PatternId) !WalkControl {
        _ = file;
        try self.record(.pattern, pattern_id.index());
        return .descend;
    }

    pub fn enterSwitchPattern(self: *@This(), file: *const AstFile, pattern: nodes.SwitchPattern) !WalkControl {
        _ = file;
        try self.events.append(self.allocator, spEvent(switchPatternTag(pattern)));
        return .descend;
    }
};

const ExprEnterExitVisitor = struct {
    allocator: std.mem.Allocator,
    events: *std.ArrayList(u32),

    pub fn enterExpr(self: *@This(), file: *const AstFile, expr_id: ExprId) !WalkControl {
        _ = file;
        try self.events.append(self.allocator, testEvent(.expr, expr_id.index()));
        return .descend;
    }

    pub fn exitExpr(self: *@This(), file: *const AstFile, expr_id: ExprId) !void {
        _ = file;
        try self.events.append(self.allocator, testEvent(.exit_expr, expr_id.index()));
    }
};

test "collectNamesInExpr type check" {
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

    var legacy_names: std.ArrayList([]const u8) = .empty;
    defer legacy_names.deinit(testing.allocator);
    try collectNamesInExpr(testing.allocator, &ast_file, ExprId.fromIndex(3), &legacy_names);
    try testing.expectEqual(@as(usize, 2), legacy_names.items.len);
    try testing.expectEqualStrings("condition", legacy_names.items[0]);
    try testing.expectEqualStrings("value", legacy_names.items[1]);

    var walked_names: std.ArrayList([]const u8) = .empty;
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

test "walkBody visits statements and nested bodies in source order" {
    const r = source.TextRange.empty(0);

    var expressions: [32]nodes.Expr = undefined;
    fillNameExprs(expressions[0..], r);

    var patterns = [_]nodes.Pattern{
        .{ .Name = .{ .range = r, .name = "target" } },
        .{ .Index = .{ .range = r, .base = pid(0), .index = eid(31) } },
    };

    var while_invariants = [_]ExprId{eid(4)};
    var for_invariants = [_]ExprId{eid(7)};
    var switch_arms = [_]nodes.SwitchArm{.{
        .range = r,
        .pattern = .{ .Range = .{ .range = r, .start = eid(9), .end = eid(10), .inclusive = true } },
        .body = bid(5),
    }};
    var log_args = [_]ExprId{ eid(11), eid(12) };
    const catch_clause = nodes.CatchClause{
        .range = r,
        .error_pattern = pid(0),
        .body = bid(8),
    };

    var statements = [_]nodes.Stmt{
        .{ .VariableDecl = .{ .range = r, .pattern = pid(0), .binding_kind = .let_, .storage_class = .none, .type_expr = null, .value = eid(0) } },
        .{ .Return = .{ .range = r, .value = eid(1) } },
        .{ .If = .{ .range = r, .condition = eid(2), .then_body = bid(1), .else_body = bid(2) } },
        .{ .While = .{ .range = r, .condition = eid(3), .invariants = while_invariants[0..], .body = bid(3) } },
        .{ .For = .{ .range = r, .iterable = eid(5), .range_end = eid(6), .item_pattern = pid(0), .index_pattern = null, .invariants = for_invariants[0..], .body = bid(4) } },
        .{ .Switch = .{ .range = r, .label = null, .condition = eid(8), .arms = switch_arms[0..], .else_body = bid(6) } },
        .{ .Try = .{ .range = r, .try_body = bid(7), .catch_clause = catch_clause } },
        .{ .Log = .{ .range = r, .name = "Logged", .args = log_args[0..] } },
        .{ .Lock = .{ .range = r, .path = eid(13) } },
        .{ .Unlock = .{ .range = r, .path = eid(14) } },
        .{ .Assert = .{ .range = r, .condition = eid(15), .message = null } },
        .{ .Assume = .{ .range = r, .condition = eid(16) } },
        .{ .Assign = .{ .range = r, .op = .assign, .target = pid(1), .value = eid(17) } },
        .{ .Expr = .{ .range = r, .expr = eid(18) } },
        .{ .Block = .{ .range = r, .body = bid(9) } },
        .{ .LabeledBlock = .{ .range = r, .label = "label", .body = bid(10) } },
        .{ .Break = .{ .range = r, .label = null, .value = eid(19) } },
        .{ .Continue = .{ .range = r, .label = null, .value = eid(20) } },
        .{ .Havoc = .{ .range = r, .name = "state" } },
        .{ .Error = .{ .range = r } },
        .{ .Expr = .{ .range = r, .expr = eid(21) } },
        .{ .Expr = .{ .range = r, .expr = eid(22) } },
        .{ .Expr = .{ .range = r, .expr = eid(23) } },
        .{ .Expr = .{ .range = r, .expr = eid(24) } },
        .{ .Expr = .{ .range = r, .expr = eid(25) } },
        .{ .Expr = .{ .range = r, .expr = eid(26) } },
        .{ .Expr = .{ .range = r, .expr = eid(27) } },
        .{ .Expr = .{ .range = r, .expr = eid(28) } },
        .{ .Expr = .{ .range = r, .expr = eid(29) } },
        .{ .Expr = .{ .range = r, .expr = eid(30) } },
    };

    var body0_statements = [_]StmtId{
        sid(0),  sid(1),  sid(2),  sid(3),  sid(4),
        sid(5),  sid(6),  sid(7),  sid(8),  sid(9),
        sid(10), sid(11), sid(12), sid(13), sid(14),
        sid(15), sid(16), sid(17), sid(18), sid(19),
    };
    var body1_statements = [_]StmtId{sid(20)};
    var body2_statements = [_]StmtId{sid(21)};
    var body3_statements = [_]StmtId{sid(22)};
    var body4_statements = [_]StmtId{sid(23)};
    var body5_statements = [_]StmtId{sid(24)};
    var body6_statements = [_]StmtId{sid(25)};
    var body7_statements = [_]StmtId{sid(26)};
    var body8_statements = [_]StmtId{sid(27)};
    var body9_statements = [_]StmtId{sid(28)};
    var body10_statements = [_]StmtId{sid(29)};
    var bodies = [_]nodes.Body{
        .{ .range = r, .statements = body0_statements[0..] },
        .{ .range = r, .statements = body1_statements[0..] },
        .{ .range = r, .statements = body2_statements[0..] },
        .{ .range = r, .statements = body3_statements[0..] },
        .{ .range = r, .statements = body4_statements[0..] },
        .{ .range = r, .statements = body5_statements[0..] },
        .{ .range = r, .statements = body6_statements[0..] },
        .{ .range = r, .statements = body7_statements[0..] },
        .{ .range = r, .statements = body8_statements[0..] },
        .{ .range = r, .statements = body9_statements[0..] },
        .{ .range = r, .statements = body10_statements[0..] },
    };

    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var type_exprs: [0]nodes.TypeExpr = .{};
    var ast_file = initTestAstFile(root_items[0..], items[0..], bodies[0..], statements[0..], expressions[0..], type_exprs[0..], patterns[0..]);
    defer ast_file.deinit();

    var events: std.ArrayList(u32) = .empty;
    defer events.deinit(testing.allocator);
    var visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &events };
    try walkBody(EnterOrderVisitor, &visitor, &ast_file, bid(0), .{});

    const expected = [_]u32{
        testEvent(.body, 0),
        testEvent(.stmt, 0),
        testEvent(.expr, 0),
        testEvent(.stmt, 1),
        testEvent(.expr, 1),
        testEvent(.stmt, 2),
        testEvent(.expr, 2),
        testEvent(.body, 1),
        testEvent(.stmt, 20),
        testEvent(.expr, 21),
        testEvent(.body, 2),
        testEvent(.stmt, 21),
        testEvent(.expr, 22),
        testEvent(.stmt, 3),
        testEvent(.expr, 3),
        testEvent(.expr, 4),
        testEvent(.body, 3),
        testEvent(.stmt, 22),
        testEvent(.expr, 23),
        testEvent(.stmt, 4),
        testEvent(.expr, 5),
        testEvent(.expr, 6),
        testEvent(.expr, 7),
        testEvent(.body, 4),
        testEvent(.stmt, 23),
        testEvent(.expr, 24),
        testEvent(.stmt, 5),
        testEvent(.expr, 8),
        spEvent(.range),
        testEvent(.expr, 9),
        testEvent(.expr, 10),
        testEvent(.body, 5),
        testEvent(.stmt, 24),
        testEvent(.expr, 25),
        testEvent(.body, 6),
        testEvent(.stmt, 25),
        testEvent(.expr, 26),
        testEvent(.stmt, 6),
        testEvent(.body, 7),
        testEvent(.stmt, 26),
        testEvent(.expr, 27),
        testEvent(.body, 8),
        testEvent(.stmt, 27),
        testEvent(.expr, 28),
        testEvent(.stmt, 7),
        testEvent(.expr, 11),
        testEvent(.expr, 12),
        testEvent(.stmt, 8),
        testEvent(.expr, 13),
        testEvent(.stmt, 9),
        testEvent(.expr, 14),
        testEvent(.stmt, 10),
        testEvent(.expr, 15),
        testEvent(.stmt, 11),
        testEvent(.expr, 16),
        testEvent(.stmt, 12),
        testEvent(.expr, 17),
        testEvent(.pattern, 1),
        testEvent(.pattern, 0),
        testEvent(.expr, 31),
        testEvent(.stmt, 13),
        testEvent(.expr, 18),
        testEvent(.stmt, 14),
        testEvent(.body, 9),
        testEvent(.stmt, 28),
        testEvent(.expr, 29),
        testEvent(.stmt, 15),
        testEvent(.body, 10),
        testEvent(.stmt, 29),
        testEvent(.expr, 30),
        testEvent(.stmt, 16),
        testEvent(.expr, 19),
        testEvent(.stmt, 17),
        testEvent(.expr, 20),
        testEvent(.stmt, 18),
        testEvent(.stmt, 19),
    };
    try testing.expectEqualSlices(u32, expected[0..], events.items);
}

test "walkExpr visits expression children in source order" {
    const r = source.TextRange.empty(0);

    var expressions: [41]nodes.Expr = undefined;
    fillNameExprs(expressions[0..], r);

    var array_elements = [_]ExprId{ eid(2), eid(3) };
    expressions[1] = .{ .ArrayLiteral = .{ .range = r, .elements = array_elements[0..] } };

    var struct_fields = [_]nodes.StructFieldInit{
        .{ .range = r, .name = "a", .value = eid(5) },
        .{ .range = r, .name = "b", .value = eid(6) },
    };
    expressions[4] = .{ .StructLiteral = .{ .range = r, .type_name = "S", .type_expr = null, .fields = struct_fields[0..] } };

    var switch_arms = [_]nodes.SwitchExprArm{.{
        .range = r,
        .pattern = .{ .Expr = eid(9) },
        .value = eid(10),
    }};
    expressions[7] = .{ .Switch = .{ .range = r, .condition = eid(8), .arms = switch_arms[0..], .else_expr = eid(11) } };

    expressions[12] = .{ .ExternalProxy = .{ .range = r, .trait_name = "Remote", .address_expr = eid(13), .gas_expr = eid(14) } };
    var error_args = [_]ExprId{eid(16)};
    expressions[15] = .{ .ErrorReturn = .{ .range = r, .name = "E", .args = error_args[0..] } };
    expressions[17] = .{ .Unary = .{ .range = r, .op = .not_, .operand = eid(18) } };
    expressions[19] = .{ .Binary = .{ .range = r, .op = .add, .lhs = eid(20), .rhs = eid(21) } };
    var call_args = [_]ExprId{eid(24)};
    expressions[22] = .{ .Call = .{ .range = r, .callee = eid(23), .args = call_args[0..] } };
    var builtin_args = [_]ExprId{eid(26)};
    expressions[25] = .{ .Builtin = .{ .range = r, .name = "foo", .type_arg = null, .args = builtin_args[0..] } };
    expressions[27] = .{ .Field = .{ .range = r, .base = eid(28), .name = "field" } };
    expressions[29] = .{ .Index = .{ .range = r, .base = eid(30), .index = eid(31) } };
    expressions[32] = .{ .Group = .{ .range = r, .expr = eid(33) } };
    expressions[34] = .{ .Old = .{ .range = r, .expr = eid(35) } };
    expressions[36] = .{ .Quantified = .{ .range = r, .quantifier = .forall, .pattern = pid(0), .type_expr = ids.TypeExprId.fromIndex(0), .condition = eid(37), .body = eid(38) } };
    expressions[39] = .{ .TypeValue = .{ .range = r, .type_expr = ids.TypeExprId.fromIndex(0) } };

    var tuple_elements = [_]ExprId{
        eid(1),  eid(4),  eid(7),  eid(12), eid(15),
        eid(17), eid(19), eid(22), eid(25), eid(27),
        eid(29), eid(32), eid(34), eid(36), eid(39),
    };
    expressions[40] = .{ .Tuple = .{ .range = r, .elements = tuple_elements[0..] } };

    var patterns = [_]nodes.Pattern{.{ .Name = .{ .range = r, .name = "x" } }};
    var type_exprs = [_]nodes.TypeExpr{.{ .Path = .{ .range = r, .name = "u256" } }};
    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var bodies: [0]nodes.Body = .{};
    var statements: [0]nodes.Stmt = .{};
    var ast_file = initTestAstFile(root_items[0..], items[0..], bodies[0..], statements[0..], expressions[0..], type_exprs[0..], patterns[0..]);
    defer ast_file.deinit();

    var events: std.ArrayList(u32) = .empty;
    defer events.deinit(testing.allocator);
    var visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &events };
    try walkExpr(EnterOrderVisitor, &visitor, &ast_file, eid(40), .{});

    const expected = [_]u32{
        testEvent(.expr, 40),
        testEvent(.expr, 1),
        testEvent(.expr, 2),
        testEvent(.expr, 3),
        testEvent(.expr, 4),
        testEvent(.expr, 5),
        testEvent(.expr, 6),
        testEvent(.expr, 7),
        testEvent(.expr, 8),
        spEvent(.expr),
        testEvent(.expr, 9),
        testEvent(.expr, 10),
        testEvent(.expr, 11),
        testEvent(.expr, 12),
        testEvent(.expr, 13),
        testEvent(.expr, 14),
        testEvent(.expr, 15),
        testEvent(.expr, 16),
        testEvent(.expr, 17),
        testEvent(.expr, 18),
        testEvent(.expr, 19),
        testEvent(.expr, 20),
        testEvent(.expr, 21),
        testEvent(.expr, 22),
        testEvent(.expr, 23),
        testEvent(.expr, 24),
        testEvent(.expr, 25),
        testEvent(.expr, 26),
        testEvent(.expr, 27),
        testEvent(.expr, 28),
        testEvent(.expr, 29),
        testEvent(.expr, 30),
        testEvent(.expr, 31),
        testEvent(.expr, 32),
        testEvent(.expr, 33),
        testEvent(.expr, 34),
        testEvent(.expr, 35),
        testEvent(.expr, 36),
        testEvent(.expr, 39),
    };
    try testing.expectEqualSlices(u32, expected[0..], events.items);
}

test "walkPattern and walkSwitchPattern visit pattern children in order" {
    const r = source.TextRange.empty(0);

    var expressions: [5]nodes.Expr = undefined;
    fillNameExprs(expressions[0..], r);

    var destructure_fields = [_]nodes.StructDestructureField{
        .{ .range = r, .name = "left", .binding = pid(2) },
        .{ .range = r, .name = "right", .binding = pid(3) },
    };
    var patterns = [_]nodes.Pattern{
        .{ .Name = .{ .range = r, .name = "root" } },
        .{ .Field = .{ .range = r, .base = pid(0), .name = "field" } },
        .{ .Index = .{ .range = r, .base = pid(1), .index = eid(0) } },
        .{ .Name = .{ .range = r, .name = "other" } },
        .{ .StructDestructure = .{ .range = r, .fields = destructure_fields[0..], .has_rest = false } },
    };

    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var bodies: [0]nodes.Body = .{};
    var statements: [0]nodes.Stmt = .{};
    var type_exprs: [0]nodes.TypeExpr = .{};
    var ast_file = initTestAstFile(root_items[0..], items[0..], bodies[0..], statements[0..], expressions[0..], type_exprs[0..], patterns[0..]);
    defer ast_file.deinit();

    var pattern_events: std.ArrayList(u32) = .empty;
    defer pattern_events.deinit(testing.allocator);
    var pattern_visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &pattern_events };
    try walkPattern(EnterOrderVisitor, &pattern_visitor, &ast_file, pid(4), .{});
    const expected_pattern = [_]u32{
        testEvent(.pattern, 4),
        testEvent(.pattern, 2),
        testEvent(.pattern, 1),
        testEvent(.pattern, 0),
        testEvent(.expr, 0),
        testEvent(.pattern, 3),
    };
    try testing.expectEqualSlices(u32, expected_pattern[0..], pattern_events.items);

    var named_bindings = [_]PatternId{pid(4)};
    var alternatives = [_]nodes.SwitchPattern{
        .{ .Expr = eid(1) },
        .{ .Range = .{ .range = r, .start = eid(2), .end = eid(3), .inclusive = true } },
        .{ .NamedError = .{ .range = r, .callee = eid(4), .bindings = named_bindings[0..] } },
        .{ .Ok = pid(0) },
        .{ .Err = pid(2) },
        .{ .Else = r },
    };
    const root_pattern: nodes.SwitchPattern = .{ .Or = .{ .range = r, .alternatives = alternatives[0..] } };

    var switch_events: std.ArrayList(u32) = .empty;
    defer switch_events.deinit(testing.allocator);
    var switch_visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &switch_events };
    try walkSwitchPattern(EnterOrderVisitor, &switch_visitor, &ast_file, root_pattern, .{});
    const expected_switch = [_]u32{
        spEvent(.or_),
        spEvent(.expr),
        testEvent(.expr, 1),
        spEvent(.range),
        testEvent(.expr, 2),
        testEvent(.expr, 3),
        spEvent(.named_error),
        testEvent(.expr, 4),
        testEvent(.pattern, 4),
        testEvent(.pattern, 2),
        testEvent(.pattern, 1),
        testEvent(.pattern, 0),
        testEvent(.expr, 0),
        testEvent(.pattern, 3),
        spEvent(.ok),
        testEvent(.pattern, 0),
        spEvent(.err),
        testEvent(.pattern, 2),
        testEvent(.pattern, 1),
        testEvent(.pattern, 0),
        testEvent(.expr, 0),
        spEvent(.else_),
    };
    try testing.expectEqualSlices(u32, expected_switch[0..], switch_events.items);
}

test "walkExpr options control body-bearing and optional expression children" {
    const r = source.TextRange.empty(0);

    var expressions: [13]nodes.Expr = undefined;
    fillNameExprs(expressions[0..], r);

    expressions[1] = .{ .Comptime = .{ .range = r, .body = bid(0) } };
    expressions[4] = .{ .Quantified = .{ .range = r, .quantifier = .exists, .pattern = pid(0), .type_expr = ids.TypeExprId.fromIndex(0), .condition = eid(2), .body = eid(3) } };
    expressions[7] = .{ .ExternalProxy = .{ .range = r, .trait_name = "Remote", .address_expr = eid(5), .gas_expr = eid(6) } };
    var error_args = [_]ExprId{eid(8)};
    expressions[9] = .{ .ErrorReturn = .{ .range = r, .name = "E", .args = error_args[0..] } };
    expressions[11] = .{ .Old = .{ .range = r, .expr = eid(10) } };
    var tuple_elements = [_]ExprId{ eid(1), eid(4), eid(7), eid(9), eid(11) };
    expressions[12] = .{ .Tuple = .{ .range = r, .elements = tuple_elements[0..] } };

    var body_statements = [_]StmtId{sid(0)};
    var bodies = [_]nodes.Body{.{ .range = r, .statements = body_statements[0..] }};
    var statements = [_]nodes.Stmt{.{ .Expr = .{ .range = r, .expr = eid(0) } }};
    var patterns = [_]nodes.Pattern{.{ .Name = .{ .range = r, .name = "x" } }};
    var type_exprs = [_]nodes.TypeExpr{.{ .Path = .{ .range = r, .name = "u256" } }};
    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var ast_file = initTestAstFile(root_items[0..], items[0..], bodies[0..], statements[0..], expressions[0..], type_exprs[0..], patterns[0..]);
    defer ast_file.deinit();

    var default_events: std.ArrayList(u32) = .empty;
    defer default_events.deinit(testing.allocator);
    var default_visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &default_events };
    try walkExpr(EnterOrderVisitor, &default_visitor, &ast_file, eid(12), .{});
    const expected_default = [_]u32{
        testEvent(.expr, 12),
        testEvent(.expr, 1),
        testEvent(.expr, 4),
        testEvent(.expr, 7),
        testEvent(.expr, 5),
        testEvent(.expr, 6),
        testEvent(.expr, 9),
        testEvent(.expr, 8),
        testEvent(.expr, 11),
        testEvent(.expr, 10),
    };
    try testing.expectEqualSlices(u32, expected_default[0..], default_events.items);

    var option_events: std.ArrayList(u32) = .empty;
    defer option_events.deinit(testing.allocator);
    var option_visitor = EnterOrderVisitor{ .allocator = testing.allocator, .events = &option_events };
    try walkExpr(EnterOrderVisitor, &option_visitor, &ast_file, eid(12), .{
        .enter_comptime_bodies = true,
        .enter_quantified_bodies = true,
        .walk_external_proxy_exprs = false,
        .walk_error_return_args = false,
        .walk_old_exprs = false,
    });
    const expected_options = [_]u32{
        testEvent(.expr, 12),
        testEvent(.expr, 1),
        testEvent(.body, 0),
        testEvent(.stmt, 0),
        testEvent(.expr, 0),
        testEvent(.expr, 4),
        testEvent(.expr, 2),
        testEvent(.expr, 3),
        testEvent(.expr, 7),
        testEvent(.expr, 9),
        testEvent(.expr, 11),
    };
    try testing.expectEqualSlices(u32, expected_options[0..], option_events.items);
}

test "walkExpr enter and exit hooks use pre-order and post-order" {
    const r = source.TextRange.empty(0);

    var call_args = [_]ExprId{eid(2)};
    var expressions = [_]nodes.Expr{
        .{ .Call = .{ .range = r, .callee = eid(1), .args = call_args[0..] } },
        .{ .Name = .{ .range = r, .name = "callee" } },
        .{ .Name = .{ .range = r, .name = "arg" } },
    };
    var root_items: [0]ids.ItemId = .{};
    var items: [0]nodes.Item = .{};
    var bodies: [0]nodes.Body = .{};
    var statements: [0]nodes.Stmt = .{};
    var type_exprs: [0]nodes.TypeExpr = .{};
    var patterns: [0]nodes.Pattern = .{};
    var ast_file = initTestAstFile(root_items[0..], items[0..], bodies[0..], statements[0..], expressions[0..], type_exprs[0..], patterns[0..]);
    defer ast_file.deinit();

    var events: std.ArrayList(u32) = .empty;
    defer events.deinit(testing.allocator);
    var visitor = ExprEnterExitVisitor{ .allocator = testing.allocator, .events = &events };
    try walkExpr(ExprEnterExitVisitor, &visitor, &ast_file, eid(0), .{});

    const expected = [_]u32{
        testEvent(.expr, 0),
        testEvent(.expr, 1),
        testEvent(.exit_expr, 1),
        testEvent(.expr, 2),
        testEvent(.exit_expr, 2),
        testEvent(.exit_expr, 0),
    };
    try testing.expectEqualSlices(u32, expected[0..], events.items);
}
