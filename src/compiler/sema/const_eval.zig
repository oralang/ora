const std = @import("std");
const ast = @import("../ast/mod.zig");
const const_values = @import("const_values.zig");
const model = @import("model.zig");

const ConstEvalResult = model.ConstEvalResult;
const ConstValue = model.ConstValue;
const constEquals = const_values.constEquals;
const evalBinary = const_values.evalBinary;
const evalUnary = const_values.evalUnary;
const parseIntegerLiteral = const_values.parseIntegerLiteral;

pub fn constEval(allocator: std.mem.Allocator, file: *const ast.AstFile) !ConstEvalResult {
    var result = ConstEvalResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .values = &[_]?ConstValue{},
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    const values = try arena.alloc(?ConstValue, file.expressions.len);
    @memset(values, null);

    var evaluator = ConstEvaluator{
        .allocator = arena,
        .file = file,
        .values = values,
    };
    for (file.root_items) |item_id| {
        evaluator.visitItem(item_id);
    }

    result.values = values;
    return result;
}

const ConstEvaluator = struct {
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    values: []?ConstValue,

    fn visitItem(self: *ConstEvaluator, item_id: ast.ItemId) void {
        switch (self.file.item(item_id).*) {
            .Contract => |contract| {
                for (contract.invariants) |expr_id| _ = self.evalExpr(expr_id) catch null;
                for (contract.members) |member_id| self.visitItem(member_id);
            },
            .Function => |function| {
                for (function.clauses) |clause| _ = self.evalExpr(clause.expr) catch null;
                self.visitBody(function.body);
            },
            .Field => |field| {
                if (field.value) |expr_id| _ = self.evalExpr(expr_id) catch null;
            },
            .Constant => |constant| _ = self.evalExpr(constant.value) catch null,
            .GhostBlock => |ghost_block| self.visitBody(ghost_block.body),
            else => {},
        }
    }

    fn visitBody(self: *ConstEvaluator, body_id: ast.BodyId) void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    if (decl.value) |expr_id| _ = self.evalExpr(expr_id) catch null;
                },
                .Return => |ret| {
                    if (ret.value) |expr_id| _ = self.evalExpr(expr_id) catch null;
                },
                .If => |if_stmt| {
                    _ = self.evalExpr(if_stmt.condition) catch null;
                    self.visitBody(if_stmt.then_body);
                    if (if_stmt.else_body) |else_body| self.visitBody(else_body);
                },
                .While => |while_stmt| {
                    _ = self.evalExpr(while_stmt.condition) catch null;
                    for (while_stmt.invariants) |expr_id| _ = self.evalExpr(expr_id) catch null;
                    self.visitBody(while_stmt.body);
                },
                .For => |for_stmt| {
                    _ = self.evalExpr(for_stmt.iterable) catch null;
                    for (for_stmt.invariants) |expr_id| _ = self.evalExpr(expr_id) catch null;
                    self.visitBody(for_stmt.body);
                },
                .Switch => |switch_stmt| {
                    _ = self.evalExpr(switch_stmt.condition) catch null;
                    for (switch_stmt.arms) |arm| {
                        switch (arm.pattern) {
                            .Expr => |expr_id| _ = self.evalExpr(expr_id) catch null,
                            .Range => |range_pattern| {
                                _ = self.evalExpr(range_pattern.start) catch null;
                                _ = self.evalExpr(range_pattern.end) catch null;
                            },
                            .Else => {},
                        }
                        self.visitBody(arm.body);
                    }
                    if (switch_stmt.else_body) |else_body| self.visitBody(else_body);
                },
                .Try => |try_stmt| {
                    self.visitBody(try_stmt.try_body);
                    if (try_stmt.catch_clause) |catch_clause| self.visitBody(catch_clause.body);
                },
                .Log => |log_stmt| {
                    for (log_stmt.args) |arg| _ = self.evalExpr(arg) catch null;
                },
                .Lock => |lock_stmt| _ = self.evalExpr(lock_stmt.path) catch null,
                .Unlock => |unlock_stmt| _ = self.evalExpr(unlock_stmt.path) catch null,
                .Assert => |assert_stmt| _ = self.evalExpr(assert_stmt.condition) catch null,
                .Assume => |assume_stmt| _ = self.evalExpr(assume_stmt.condition) catch null,
                .Havoc => {},
                .Assign => |assign| _ = self.evalExpr(assign.value) catch null,
                .Expr => |expr_stmt| _ = self.evalExpr(expr_stmt.expr) catch null,
                .Block => |block| self.visitBody(block.body),
                .LabeledBlock => |block| self.visitBody(block.body),
                else => {},
            }
        }
    }

    fn evalExpr(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?ConstValue {
        if (self.values[expr_id.index()]) |value| return value;
        const value = switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| try parseIntegerLiteral(self.allocator, literal.text),
            .StringLiteral => |literal| ConstValue{ .string = literal.text },
            .BoolLiteral => |literal| ConstValue{ .boolean = literal.value },
            .Tuple => |tuple| blk: {
                for (tuple.elements) |element| _ = try self.evalExpr(element);
                break :blk null;
            },
            .ArrayLiteral => |array| blk: {
                for (array.elements) |element| _ = try self.evalExpr(element);
                break :blk null;
            },
            .StructLiteral => |struct_literal| blk: {
                for (struct_literal.fields) |field| _ = try self.evalExpr(field.value);
                break :blk null;
            },
            .Switch => |switch_expr| blk: {
                const condition = (try self.evalExpr(switch_expr.condition)) orelse {
                    for (switch_expr.arms) |arm| {
                        switch (arm.pattern) {
                            .Expr => |pattern_expr| _ = try self.evalExpr(pattern_expr),
                            .Range => |range_pattern| {
                                _ = try self.evalExpr(range_pattern.start);
                                _ = try self.evalExpr(range_pattern.end);
                            },
                            .Else => {},
                        }
                        _ = try self.evalExpr(arm.value);
                    }
                    if (switch_expr.else_expr) |else_expr| _ = try self.evalExpr(else_expr);
                    break :blk null;
                };

                // Even when the switch condition is constant, cache every arm pattern
                // constant so later HIR lowering can build full switch metadata rather
                // than falling back after the first matched arm.
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| _ = try self.evalExpr(pattern_expr),
                        .Range => |range_pattern| {
                            _ = try self.evalExpr(range_pattern.start);
                            _ = try self.evalExpr(range_pattern.end);
                        },
                        .Else => {},
                    }
                }

                for (switch_expr.arms) |arm| {
                    if (self.patternMatches(condition, arm.pattern)) {
                        break :blk try self.evalExpr(arm.value);
                    }
                }
                if (switch_expr.else_expr) |else_expr| break :blk try self.evalExpr(else_expr);
                break :blk null;
            },
            .Comptime => |comptime_expr| blk: {
                const body = self.file.body(comptime_expr.body).*;
                if (body.statements.len == 0) break :blk null;
                const last_stmt = self.file.statement(body.statements[body.statements.len - 1]).*;
                break :blk switch (last_stmt) {
                    .Expr => |expr_stmt| try self.evalExpr(expr_stmt.expr),
                    .Return => |ret| if (ret.value) |value| try self.evalExpr(value) else null,
                    else => null,
                };
            },
            .ErrorReturn => |error_return| blk: {
                for (error_return.args) |arg| _ = try self.evalExpr(arg);
                break :blk null;
            },
            .Call => |call| blk: {
                _ = try self.evalExpr(call.callee);
                for (call.args) |arg| _ = try self.evalExpr(arg);
                break :blk null;
            },
            .Field => |field| blk: {
                _ = try self.evalExpr(field.base);
                break :blk null;
            },
            .Index => |index| blk: {
                _ = try self.evalExpr(index.base);
                _ = try self.evalExpr(index.index);
                break :blk null;
            },
            .Unary => |unary| try evalUnary(self.allocator, unary.op, try self.evalExpr(unary.operand)),
            .Binary => |binary| try evalBinary(self.allocator, binary.op, try self.evalExpr(binary.lhs), try self.evalExpr(binary.rhs)),
            .Group => |group| try self.evalExpr(group.expr),
            .Old => |old| try self.evalExpr(old.expr),
            .Builtin => |builtin| try self.evalBuiltin(builtin),
            else => null,
        };
        self.values[expr_id.index()] = value;
        return value;
    }

    fn patternMatches(self: *ConstEvaluator, condition: ConstValue, pattern: ast.SwitchPattern) bool {
        return switch (pattern) {
            .Expr => |expr_id| if (self.evalExpr(expr_id) catch null) |value| constEquals(condition, value) else false,
            .Range => |range_pattern| blk: {
                const start = (self.evalExpr(range_pattern.start) catch null) orelse break :blk false;
                const end = (self.evalExpr(range_pattern.end) catch null) orelse break :blk false;
                break :blk switch (condition) {
                    .integer => |current| switch (start) {
                        .integer => |start_value| switch (end) {
                            .integer => |end_value| if (range_pattern.inclusive)
                                current.order(start_value).compare(.gte) and current.order(end_value).compare(.lte)
                            else
                                current.order(start_value).compare(.gte) and current.order(end_value).compare(.lt),
                            else => false,
                        },
                        else => false,
                    },
                    else => false,
                };
            },
            .Else => true,
        };
    }

    fn evalBuiltin(self: *ConstEvaluator, builtin: ast.BuiltinExpr) anyerror!?ConstValue {
        if (builtin.args.len == 0) return null;

        if (std.mem.eql(u8, builtin.name, "cast")) {
            return try self.evalExpr(builtin.args[0]);
        }

        if (builtin.args.len >= 2 and (std.mem.eql(u8, builtin.name, "divTrunc") or
            std.mem.eql(u8, builtin.name, "divFloor") or
            std.mem.eql(u8, builtin.name, "divCeil") or
            std.mem.eql(u8, builtin.name, "divExact")))
        {
            const lhs = try self.evalExpr(builtin.args[0]);
            const rhs = try self.evalExpr(builtin.args[1]);
            if (lhs == null or rhs == null) return null;
            return switch (lhs.?) {
                .integer => |a| switch (rhs.?) {
                    .integer => |b| blk: {
                        if (b.eqlZero()) break :blk null;
                        var quotient = try std.math.big.int.Managed.init(self.allocator);
                        var remainder = try std.math.big.int.Managed.init(self.allocator);
                        try std.math.big.int.Managed.divTrunc(&quotient, &remainder, &a, &b);
                        break :blk .{ .integer = quotient };
                    },
                    else => null,
                },
                else => null,
            };
        }

        if (std.mem.eql(u8, builtin.name, "truncate")) {
            return try self.evalExpr(builtin.args[0]);
        }

        return null;
    }
};
