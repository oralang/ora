const std = @import("std");
const ast = @import("../compiler/ast/mod.zig");
const bridge = @import("compiler_const_bridge.zig");
const model = @import("../compiler/sema/model.zig");

const ConstEvalResult = model.ConstEvalResult;
const ConstValue = model.ConstValue;
const CtEnv = bridge.CtEnv;
const constEquals = bridge.constEquals;
const ctValueToConstValue = bridge.ctValueToConstValue;
const constToCtValue = bridge.constToCtValue;
const evalBinary = bridge.evalBinary;
const evalUnary = bridge.evalUnary;
const parseIntegerLiteral = bridge.parseIntegerLiteral;

/// Compiler-AST constant evaluator.
///
/// This is the migration boundary for moving the refactored compiler onto the
/// shared comptime subsystem. It preserves the current `ConstEvalResult` shape
/// used by sema/HIR while relocating the AST walker into `src/comptime/`.
///
/// The immediate goal is architectural:
/// - compiler DB should query comptime through `src/comptime/`
/// - legacy AST walker remains isolated in `ast_eval.zig`
/// - the lightweight compiler walker can later be upgraded to use the full
///   shared environment/value engine without moving the compiler call sites again
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
        .env = CtEnv.init(arena, .{}),
    };
    defer evaluator.env.deinit();
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
    env: CtEnv,

    const BodyControl = union(enum) {
        value: ?ConstValue,
        break_loop,
        continue_loop,
    };

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
                if (field.value) |expr_id| {
                    const value = self.evalExpr(expr_id) catch null;
                    self.bindName(field.name, value) catch {};
                    self.values[expr_id.index()] = value;
                }
            },
            .Constant => |constant| {
                const value = self.evalExpr(constant.value) catch null;
                self.bindName(constant.name, value) catch {};
                self.values[constant.value.index()] = value;
            },
            .GhostBlock => |ghost_block| self.visitBody(ghost_block.body),
            else => {},
        }
    }

    fn visitBody(self: *ConstEvaluator, body_id: ast.BodyId) void {
        self.env.pushScope(false) catch return;
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    const value = if (decl.value) |expr_id| self.evalExpr(expr_id) catch null else null;
                    self.bindPattern(decl.pattern, value) catch {};
                    if (decl.value) |expr_id| self.values[expr_id.index()] = value;
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
                .Expr => |expr_stmt| _ = self.evalExpr(expr_stmt.expr) catch null,
                .Assign => |assign| _ = self.evalExpr(assign.value) catch null,
                .Log => |log_stmt| {
                    for (log_stmt.args) |arg| _ = self.evalExpr(arg) catch null;
                },
                .Lock, .Unlock, .Break, .Continue, .Havoc => {},
                .Assert => |assert_stmt| _ = self.evalExpr(assert_stmt.condition) catch null,
                .Assume => |assume_stmt| _ = self.evalExpr(assume_stmt.condition) catch null,
                .Block => |block_stmt| self.visitBody(block_stmt.body),
                .LabeledBlock => |labeled| self.visitBody(labeled.body),
                .Error => {},
            }
        }
    }

    fn evalExpr(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?ConstValue {
        return self.evalExprImpl(expr_id, true);
    }

    fn evalExprUncached(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?ConstValue {
        return self.evalExprImpl(expr_id, false);
    }

    fn evalExprImpl(self: *ConstEvaluator, expr_id: ast.ExprId, comptime use_cache: bool) anyerror!?ConstValue {
        if (use_cache) {
            if (self.values[expr_id.index()]) |cached| return cached;
        }

        const value: ?ConstValue = switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| try parseIntegerLiteral(self.allocator, literal.text),
            .StringLiteral => |literal| ConstValue{ .string = literal.text },
            .BoolLiteral => |literal| ConstValue{ .boolean = literal.value },
            .AddressLiteral, .BytesLiteral => null,
            .Tuple => |tuple| blk: {
                for (tuple.elements) |element| _ = try self.evalExprImpl(element, use_cache);
                break :blk null;
            },
            .ArrayLiteral => |array| blk: {
                for (array.elements) |element| _ = try self.evalExprImpl(element, use_cache);
                break :blk null;
            },
            .StructLiteral => |struct_literal| blk: {
                for (struct_literal.fields) |field| _ = try self.evalExprImpl(field.value, use_cache);
                break :blk null;
            },
            .Switch => |switch_expr| blk: {
                const condition = (try self.evalExprImpl(switch_expr.condition, use_cache)) orelse {
                    for (switch_expr.arms) |arm| {
                        switch (arm.pattern) {
                            .Expr => |pattern_expr| _ = try self.evalExprImpl(pattern_expr, use_cache),
                            .Range => |range_pattern| {
                                _ = try self.evalExprImpl(range_pattern.start, use_cache);
                                _ = try self.evalExprImpl(range_pattern.end, use_cache);
                            },
                            .Else => {},
                        }
                        _ = try self.evalExprImpl(arm.value, use_cache);
                    }
                    if (switch_expr.else_expr) |else_expr| _ = try self.evalExprImpl(else_expr, use_cache);
                    break :blk null;
                };

                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| _ = try self.evalExprImpl(pattern_expr, use_cache),
                        .Range => |range_pattern| {
                            _ = try self.evalExprImpl(range_pattern.start, use_cache);
                            _ = try self.evalExprImpl(range_pattern.end, use_cache);
                        },
                        .Else => {},
                    }
                }

                for (switch_expr.arms) |arm| {
                    if (self.patternMatches(condition, arm.pattern)) {
                        break :blk try self.evalExprImpl(arm.value, use_cache);
                    }
                }
                if (switch_expr.else_expr) |else_expr| break :blk try self.evalExprImpl(else_expr, use_cache);
                break :blk null;
            },
            .Comptime => |comptime_expr| blk: {
                break :blk try self.evalComptimeBody(comptime_expr.body);
            },
            .ErrorReturn => |error_return| blk: {
                for (error_return.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
                break :blk null;
            },
            .Name => |name| blk: {
                const value = self.env.lookupValue(name.name) orelse break :blk null;
                break :blk try ctValueToConstValue(self.allocator, value);
            },
            .Result => null,
            .Unary => |unary| try evalUnary(self.allocator, unary.op, try self.evalExprImpl(unary.operand, use_cache)),
            .Binary => |binary| try evalBinary(self.allocator, binary.op, try self.evalExprImpl(binary.lhs, use_cache), try self.evalExprImpl(binary.rhs, use_cache)),
            .Call => |call| blk: {
                _ = try self.evalExprImpl(call.callee, use_cache);
                for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
                break :blk null;
            },
            .Builtin => |builtin| try self.evalBuiltin(builtin),
            .Field => |field| blk: {
                _ = try self.evalExprImpl(field.base, use_cache);
                break :blk null;
            },
            .Index => |index| blk: {
                _ = try self.evalExprImpl(index.base, use_cache);
                _ = try self.evalExprImpl(index.index, use_cache);
                break :blk null;
            },
            .Group => |group| try self.evalExprImpl(group.expr, use_cache),
            .Old => |old| try self.evalExprImpl(old.expr, use_cache),
            .Quantified => |quantified| blk: {
                if (quantified.condition) |condition| _ = try self.evalExprImpl(condition, use_cache);
                _ = try self.evalExprImpl(quantified.body, use_cache);
                break :blk null;
            },
            .Error => null,
        };
        if (use_cache) self.values[expr_id.index()] = value;
        return value;
    }

    fn patternMatches(self: *ConstEvaluator, condition: ConstValue, pattern: ast.SwitchPattern) bool {
        return switch (pattern) {
            .Expr => |expr_id| if (self.evalExpr(expr_id) catch null) |value| constEquals(condition, value) else false,
            .Range => |range_pattern| blk: {
                const start = (self.evalExpr(range_pattern.start) catch null) orelse break :blk false;
                const finish = (self.evalExpr(range_pattern.end) catch null) orelse break :blk false;
                break :blk switch (condition) {
                    .integer => |current| switch (start) {
                        .integer => |start_integer| switch (finish) {
                            .integer => |finish_integer| if (range_pattern.inclusive)
                                current.order(start_integer).compare(.gte) and current.order(finish_integer).compare(.lte)
                            else
                                current.order(start_integer).compare(.gte) and current.order(finish_integer).compare(.lt),
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

        for (builtin.args) |arg| _ = try self.evalExpr(arg);
        return null;
    }

    fn bindName(self: *ConstEvaluator, name: []const u8, value: ?ConstValue) !void {
        const const_value = value orelse return;
        const ct_value = (try constToCtValue(const_value)) orelse return;
        try self.env.set(name, ct_value);
    }

    fn bindPattern(self: *ConstEvaluator, pattern_id: ast.PatternId, value: ?ConstValue) !void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => |name| try self.bindName(name.name, value),
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.bindPattern(field.binding, null);
                }
            },
            .Field, .Index, .Error => {},
        }
    }

    fn evalComptimeBody(self: *ConstEvaluator, body_id: ast.BodyId) anyerror!?ConstValue {
        return switch (try self.evalComptimeBodyControl(body_id)) {
            .value => |value| value,
            .break_loop, .continue_loop => null,
        };
    }

    fn evalComptimeBodyControl(self: *ConstEvaluator, body_id: ast.BodyId) anyerror!BodyControl {
        self.env.pushScope(false) catch return .{ .value = null };
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        var last_value: ?ConstValue = null;
        for (body.statements) |statement_id| {
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    const value = if (decl.value) |expr_id| try self.evalExprUncached(expr_id) else null;
                    try self.bindPattern(decl.pattern, value);
                    if (decl.value) |expr_id| self.values[expr_id.index()] = value;
                    last_value = null;
                },
                .Expr => |expr_stmt| {
                    last_value = try self.evalExprUncached(expr_stmt.expr);
                },
                .Return => |ret| {
                    return .{ .value = if (ret.value) |ret_value| try self.evalExprUncached(ret_value) else null };
                },
                .Block => |block_stmt| {
                    switch (try self.evalComptimeBodyControl(block_stmt.body)) {
                        .value => |value| last_value = value,
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .LabeledBlock => |labeled| {
                    switch (try self.evalComptimeBodyControl(labeled.body)) {
                        .value => |value| last_value = value,
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .If => |if_stmt| {
                    switch (try self.evalComptimeIf(if_stmt)) {
                        .value => |value| last_value = value,
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .While => |while_stmt| {
                    last_value = try self.evalComptimeWhile(while_stmt);
                },
                .For => |for_stmt| {
                    last_value = try self.evalComptimeFor(for_stmt);
                },
                .Switch => |switch_stmt| {
                    switch (try self.evalComptimeSwitchStmt(switch_stmt)) {
                        .value => |value| last_value = value,
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .Assign => |assign| {
                    last_value = try self.evalComptimeAssign(assign);
                },
                .Break => return .break_loop,
                .Continue => return .continue_loop,
                else => {
                    self.visitBodyStatementForComptime(statement_id);
                    last_value = null;
                },
            }
        }
        return .{ .value = last_value };
    }

    fn evalComptimeIf(self: *ConstEvaluator, if_stmt: ast.IfStmt) anyerror!BodyControl {
        const condition = (try self.evalExprUncached(if_stmt.condition)) orelse return .{ .value = null };
        const take_then = self.constConditionTruthy(condition) orelse return .{ .value = null };
        if (take_then) return try self.evalComptimeBodyControl(if_stmt.then_body);
        if (if_stmt.else_body) |else_body| return try self.evalComptimeBodyControl(else_body);
        return .{ .value = null };
    }

    fn evalComptimeSwitchStmt(self: *ConstEvaluator, switch_stmt: ast.SwitchStmt) anyerror!BodyControl {
        const condition = (try self.evalExprUncached(switch_stmt.condition)) orelse return .{ .value = null };
        for (switch_stmt.arms) |arm| {
            if (self.patternMatches(condition, arm.pattern)) {
                return try self.evalComptimeBodyControl(arm.body);
            }
        }
        if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControl(else_body);
        return .{ .value = null };
    }

    fn evalComptimeWhile(self: *ConstEvaluator, while_stmt: ast.WhileStmt) anyerror!?ConstValue {
        var iterations: u64 = 0;
        var last_value: ?ConstValue = null;
        while (true) {
            iterations += 1;
            if (iterations > self.env.config.max_loop_iterations) return null;

            const condition = (try self.evalExprUncached(while_stmt.condition)) orelse return null;
            const should_continue = self.constConditionTruthy(condition) orelse return null;
            if (!should_continue) break;

            switch (try self.evalComptimeBodyControl(while_stmt.body)) {
                .value => |value| last_value = value,
                .break_loop => break,
                .continue_loop => continue,
            }
        }
        return last_value;
    }

    fn evalComptimeFor(self: *ConstEvaluator, for_stmt: ast.ForStmt) anyerror!?ConstValue {
        const iterable = (try self.evalExprUncached(for_stmt.iterable)) orelse return null;
        const trip_count = switch (iterable) {
            .integer => |integer| bridge.positiveShiftAmount(integer) orelse return null,
            else => return null,
        };

        var iteration: usize = 0;
        var last_value: ?ConstValue = null;
        while (iteration < trip_count) : (iteration += 1) {
            if (iteration >= self.env.config.max_loop_iterations) return null;

            const item_value = ConstValue{ .integer = try std.math.big.int.Managed.initSet(self.allocator, iteration) };
            try self.bindPattern(for_stmt.item_pattern, item_value);

            if (for_stmt.index_pattern) |index_pattern| {
                const index_value = ConstValue{ .integer = try std.math.big.int.Managed.initSet(self.allocator, iteration) };
                try self.bindPattern(index_pattern, index_value);
            }

            switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                .value => |value| last_value = value,
                .break_loop => break,
                .continue_loop => continue,
            }
        }
        return last_value;
    }

    fn evalComptimeAssign(self: *ConstEvaluator, assign: ast.AssignStmt) anyerror!?ConstValue {
        const rhs = (try self.evalExprUncached(assign.value)) orelse return null;
        switch (self.file.pattern(assign.target).*) {
            .Name => |name| {
                const value = switch (assign.op) {
                    .assign => rhs,
                    .add_assign => (try evalBinary(self.allocator, .add, try self.readBoundName(name.name), rhs)) orelse return null,
                    .sub_assign => (try evalBinary(self.allocator, .sub, try self.readBoundName(name.name), rhs)) orelse return null,
                    .mul_assign => (try evalBinary(self.allocator, .mul, try self.readBoundName(name.name), rhs)) orelse return null,
                    .div_assign => (try evalBinary(self.allocator, .div, try self.readBoundName(name.name), rhs)) orelse return null,
                    .mod_assign => (try evalBinary(self.allocator, .mod, try self.readBoundName(name.name), rhs)) orelse return null,
                    .bit_and_assign => (try evalBinary(self.allocator, .bit_and, try self.readBoundName(name.name), rhs)) orelse return null,
                    .bit_or_assign => (try evalBinary(self.allocator, .bit_or, try self.readBoundName(name.name), rhs)) orelse return null,
                    .bit_xor_assign => (try evalBinary(self.allocator, .bit_xor, try self.readBoundName(name.name), rhs)) orelse return null,
                    .shl_assign => (try evalBinary(self.allocator, .shl, try self.readBoundName(name.name), rhs)) orelse return null,
                    .shr_assign => (try evalBinary(self.allocator, .shr, try self.readBoundName(name.name), rhs)) orelse return null,
                    .pow_assign => (try evalBinary(self.allocator, .pow, try self.readBoundName(name.name), rhs)) orelse return null,
                    .wrapping_add_assign => (try evalBinary(self.allocator, .wrapping_add, try self.readBoundName(name.name), rhs)) orelse return null,
                    .wrapping_sub_assign => (try evalBinary(self.allocator, .wrapping_sub, try self.readBoundName(name.name), rhs)) orelse return null,
                    .wrapping_mul_assign => (try evalBinary(self.allocator, .wrapping_mul, try self.readBoundName(name.name), rhs)) orelse return null,
                };
                try self.bindName(name.name, value);
                return value;
            },
            else => return null,
        }
    }

    fn readBoundName(self: *ConstEvaluator, name: []const u8) anyerror!?ConstValue {
        const value = self.env.lookupValue(name) orelse return null;
        return try ctValueToConstValue(self.allocator, value);
    }

    fn constConditionTruthy(self: *ConstEvaluator, value: ConstValue) ?bool {
        _ = self;
        return switch (value) {
            .boolean => |boolean| boolean,
            .integer => |integer| !integer.eqlZero(),
            .string => null,
        };
    }

    fn visitBodyStatementForComptime(self: *ConstEvaluator, statement_id: ast.StmtId) void {
        switch (self.file.statement(statement_id).*) {
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
            .Assign => |assign| _ = self.evalExpr(assign.value) catch null,
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| _ = self.evalExpr(arg) catch null;
            },
            .Assert => |assert_stmt| _ = self.evalExpr(assert_stmt.condition) catch null,
            .Assume => |assume_stmt| _ = self.evalExpr(assume_stmt.condition) catch null,
            .Lock, .Unlock, .Break, .Continue, .Havoc, .Error => {},
            .VariableDecl, .Return, .Expr, .Block, .LabeledBlock => unreachable,
        }
    }

};
